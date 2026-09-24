//! Fragment transport: how RPCs reach remote holders.
//!
//! [`FragmentTransport`] abstracts delivery of [`FragmentRpc`] requests to
//! remote fragment holders. The redundancy plane never assumes a particular
//! network: production transports live above (control/consensus networking),
//! while [`LoopbackTransport`] routes to separate in-process stores for
//! deterministic tests. Real-network coverage lives in the `kivi-consensus`
//! and `kivi-lab` suites — loopback proves the plane logic, not the wire.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use kivi_codec::integrity::blake3_256;
use kivi_types::NodeId;

use crate::{
    AssetId, RedundancyError,
    proto::{FragmentReply, FragmentRpc, RefuseReason},
    store::FragmentStore,
};

/// Where one RPC goes: the node plus the incarnation the caller believes in.
///
/// `incarnation == 0` is the bootstrap probe (always servable); any other
/// value must equal the holder's current incarnation or the call is refused
/// with [`RefuseReason::StaleIncarnation`] carrying the current epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PeerTarget {
    /// Node that should serve the RPC.
    pub node: NodeId,
    /// Incarnation the caller believes the node serves.
    pub incarnation: u64,
}

/// Why one RPC attempt failed (callers try alternates, never serve lies).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportFailure {
    /// The holder could not be reached (or its bytes failed verification).
    Unreachable {
        /// Human-readable cause.
        detail: String,
    },
    /// The holder refused the request (fencing, never silent loss).
    Refused {
        /// Why the holder refused.
        reason: RefuseReason,
    },
    /// The attempt exceeded its deadline (no partial bytes were served).
    Timeout,
    /// A payload exceeded the wire caps.
    Oversize,
    /// The transport is shutting down.
    ShuttingDown,
}

impl TransportFailure {
    /// Maps the failure into the fabric error vocabulary at call sites.
    ///
    /// Precise callers match on [`TransportFailure`] directly (they know the
    /// asset and generation); this helper is the fail-closed default for
    /// simple sites, approximating unknown coordinates from `op`.
    #[must_use]
    pub fn into_error(self, op: &str) -> RedundancyError {
        match self {
            Self::Unreachable { detail } => RedundancyError::Unreachable { detail },
            Self::Refused { reason } => match reason {
                RefuseReason::StaleGeneration { current } => RedundancyError::StaleGeneration {
                    generation: current,
                    current,
                },
                RefuseReason::StaleIncarnation { current } => RedundancyError::StaleIncarnation {
                    expected: current,
                    current,
                },
                RefuseReason::Oversize => RedundancyError::Overloaded {
                    detail: format!("{op} payload exceeds wire caps"),
                },
                RefuseReason::HashMismatch => RedundancyError::CorruptFragment {
                    detail: format!("{op} bytes failed verification"),
                },
                RefuseReason::UnknownAsset => RedundancyError::MissingFragment {
                    asset: op.to_owned(),
                    generation: 0,
                    index: 0,
                },
                RefuseReason::ShuttingDown => RedundancyError::Overloaded {
                    detail: format!("{op} refused: holder shutting down"),
                },
            },
            Self::Timeout => RedundancyError::Timeout { op: op.to_owned() },
            Self::Oversize => RedundancyError::Overloaded {
                detail: format!("{op} payload exceeds wire caps"),
            },
            Self::ShuttingDown => RedundancyError::Overloaded {
                detail: format!("{op} refused: transport shutting down"),
            },
        }
    }
}

/// Delivery contract for fragment RPCs: send one request, receive one reply.
///
/// Implementations own timeouts and retries below this line; callers map
/// [`TransportFailure`] into [`RedundancyError`] (see
/// [`TransportFailure::into_error`]) and fall back to alternate holders.
pub trait FragmentTransport: Send + Sync + std::fmt::Debug {
    /// Calls one RPC on a peer, resolving within `timeout`.
    fn call<'a>(
        &'a self,
        target: PeerTarget,
        rpc: FragmentRpc,
        timeout: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<FragmentReply, TransportFailure>> + Send + 'a>>;
}

/// In-process test transport: routes RPCs to separate node-local stores.
///
/// Each node owns a **separate** [`crate::store::LocalFragmentStore`] (own
/// tempdir, no shared filesystem) behind an `Arc<Mutex<…>>`; delivery is a
/// direct call, so this proves plane logic (placement, fencing, repair)
/// while real-network coverage lives in `kivi-consensus` and `kivi-lab`.
#[derive(Debug, Default)]
pub struct LoopbackTransport {
    /// Stores by node (separate directories, in-process delivery).
    stores: Mutex<HashMap<NodeId, Arc<Mutex<crate::store::LocalFragmentStore>>>>,
}

impl LoopbackTransport {
    /// Creates an empty transport (nodes join through [`Self::add_node`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a node's store (separate tempdir per node in tests).
    pub fn add_node(&self, node: NodeId, store: Arc<Mutex<crate::store::LocalFragmentStore>>) {
        if let Ok(mut guard) = self.stores.lock() {
            guard.insert(node, store);
        }
    }

    /// Removes a node's store (simulates a dead holder: calls fail closed).
    pub fn remove_node(&self, node: NodeId) {
        if let Ok(mut guard) = self.stores.lock() {
            guard.remove(&node);
        }
    }
}

impl FragmentTransport for LoopbackTransport {
    fn call<'a>(
        &'a self,
        target: PeerTarget,
        rpc: FragmentRpc,
        timeout: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<FragmentReply, TransportFailure>> + Send + 'a>> {
        Box::pin(async move {
            // In-process delivery is immediate; the timeout is accepted for
            // interface uniformity and ignored here.
            let _ = timeout;
            let store = {
                let guard = self
                    .stores
                    .lock()
                    .map_err(|_| TransportFailure::ShuttingDown)?;
                guard
                    .get(&target.node)
                    .cloned()
                    .ok_or_else(|| TransportFailure::Unreachable {
                        detail: format!("no loopback store for node {}", target.node.as_u64()),
                    })?
            };
            let guard = store.lock().map_err(|_| TransportFailure::ShuttingDown)?;
            // Uniform incarnation gate: the store's boolean-returning probes
            // swallow mismatches, so the transport enforces refusal here for
            // every RPC (real transports get this from the remote).
            if target.incarnation != 0 && target.incarnation != guard.incarnation() {
                return Ok(FragmentReply::Refused {
                    reason: RefuseReason::StaleIncarnation {
                        current: guard.incarnation(),
                    },
                });
            }
            if rpc.is_mutation() && target.incarnation == 0 && guard.incarnation() != 0 {
                return Ok(FragmentReply::Refused {
                    reason: RefuseReason::StaleIncarnation {
                        current: guard.incarnation(),
                    },
                });
            }
            Ok(dispatch(&guard, rpc))
        })
    }
}

/// Executes one RPC against a node-local store.
#[allow(clippy::too_many_lines)]
fn dispatch(store: &crate::store::LocalFragmentStore, rpc: FragmentRpc) -> FragmentReply {
    match rpc {
        FragmentRpc::Hello { .. } => FragmentReply::Incarnation {
            current: store.incarnation(),
        },
        FragmentRpc::Put {
            key,
            role,
            params_tag,
            content,
            stored_len,
            target_incarnation,
            bytes,
        } => match store.stage_fragment(
            &key,
            role,
            params_tag,
            content,
            stored_len,
            &bytes,
            target_incarnation,
        ) {
            Ok(content) => FragmentReply::Stored { content },
            Err(error) => refuse_or_unreachable(&error),
        },
        FragmentRpc::Get {
            key,
            target_incarnation,
        } => match store.fetch_fragment(&key, target_incarnation) {
            Ok(bytes) => FragmentReply::Bytes {
                content: blake3_256(&bytes),
                bytes,
            },
            Err(error) => refuse_or_unreachable(&error),
        },
        FragmentRpc::Has {
            key,
            target_incarnation,
        } => {
            let (present, healthy) = store.has_fragment(&key, target_incarnation);
            FragmentReply::Presence { present, healthy }
        }
        FragmentRpc::Metadata {
            key,
            target_incarnation,
        } => {
            let (present, healthy) = store.metadata_fragment(&key, target_incarnation);
            FragmentReply::Presence { present, healthy }
        }
        FragmentRpc::Drop {
            key,
            target_incarnation,
        } => {
            let _ = store.remove_fragment(&key, target_incarnation);
            FragmentReply::Dropped
        }
        FragmentRpc::SweepOrphans { target_incarnation } => {
            match store.sweep_orphans_at(target_incarnation) {
                Ok(count) => FragmentReply::Swept { count },
                Err(error) => refuse_or_unreachable(&error),
            }
        }
        FragmentRpc::Inventory { target_incarnation } => {
            if target_incarnation != 0 && target_incarnation != store.incarnation() {
                FragmentReply::Refused {
                    reason: RefuseReason::StaleIncarnation {
                        current: store.incarnation(),
                    },
                }
            } else {
                FragmentReply::Inventory {
                    entries: store.inventory(),
                }
            }
        }
        FragmentRpc::InventoryPage {
            target_incarnation,
            offset,
            limit,
        } => {
            if target_incarnation != 0 && target_incarnation != store.incarnation() {
                FragmentReply::Refused {
                    reason: RefuseReason::StaleIncarnation {
                        current: store.incarnation(),
                    },
                }
            } else {
                let (entries, next_offset) = store.inventory_page(offset, limit);
                FragmentReply::InventoryPage {
                    entries,
                    next_offset,
                }
            }
        }
        FragmentRpc::SweepOrphansGraceful {
            target_incarnation,
            minimum_age_ms,
            protected,
        } => {
            if target_incarnation != 0 && target_incarnation != store.incarnation() {
                FragmentReply::Refused {
                    reason: RefuseReason::StaleIncarnation {
                        current: store.incarnation(),
                    },
                }
            } else {
                FragmentReply::Swept {
                    count: store.sweep_orphans_older_than(
                        std::time::Duration::from_millis(minimum_age_ms),
                        &protected,
                    ),
                }
            }
        }
        other => dispatch_layout(store, other),
    }
}

/// Executes one layout RPC against a node-local store.
fn dispatch_layout(store: &crate::store::LocalFragmentStore, rpc: FragmentRpc) -> FragmentReply {
    match rpc {
        FragmentRpc::PutLayout {
            asset_kind,
            asset_domain,
            asset_hash,
            generation,
            control_generation,
            layout,
            target_incarnation,
        } => {
            let asset = asset_coords(asset_kind, asset_domain, &asset_hash);
            match asset {
                Ok(asset) => match store.put_layout(
                    &asset,
                    generation,
                    control_generation,
                    &layout,
                    target_incarnation,
                ) {
                    Ok(accepted) => {
                        let _ = accepted;
                        FragmentReply::Stored {
                            content: blake3_256(&layout),
                        }
                    }
                    Err(error) => refuse_or_unreachable(&error),
                },
                Err(error) => refuse_or_unreachable(&error),
            }
        }
        FragmentRpc::AbortLayout {
            asset_kind,
            asset_domain,
            asset_hash,
            generation,
            control_generation,
            target_incarnation,
        } => {
            let asset = asset_coords(asset_kind, asset_domain, &asset_hash);
            match asset {
                Ok(asset) => match store.abort_layout(
                    &asset,
                    generation,
                    control_generation,
                    target_incarnation,
                ) {
                    Ok(count) => FragmentReply::Swept { count },
                    Err(error) => refuse_or_unreachable(&error),
                },
                Err(error) => refuse_or_unreachable(&error),
            }
        }
        FragmentRpc::GetLayout {
            asset_kind,
            asset_domain,
            asset_hash,
            generation,
            target_incarnation,
        } => match store.get_layout(
            asset_kind,
            asset_domain,
            asset_hash,
            generation,
            target_incarnation,
        ) {
            Ok(bytes) => FragmentReply::LayoutBytes { bytes },
            Err(error) => refuse_or_unreachable(&error),
        },
        FragmentRpc::ProbeAsset {
            asset_kind,
            asset_domain,
            asset_hash,
            target_incarnation,
        } => {
            let (generation, layout_hash) =
                store.probe_asset(asset_kind, asset_domain, asset_hash, target_incarnation);
            FragmentReply::AssetStatus {
                generation,
                layout_hash,
            }
        }
        // Fragment RPCs never reach the layout dispatcher.
        _ => FragmentReply::Refused {
            reason: RefuseReason::UnknownAsset,
        },
    }
}

/// Builds an asset identity from wire coordinates, rejecting unknown
/// kinds and the zero sentinel.
fn asset_coords(
    asset_kind: u8,
    asset_domain: u64,
    asset_hash: &[u8; 32],
) -> Result<AssetId, RedundancyError> {
    let kind =
        crate::AssetKind::from_u8(asset_kind).map_err(|error| RedundancyError::BadLayout {
            detail: format!("asset coordinates: {error}"),
        })?;
    AssetId::new(kind, asset_domain, *asset_hash).map_err(|error| RedundancyError::BadLayout {
        detail: format!("asset coordinates: {error}"),
    })
}

/// Maps store errors to refusals (fencing and bounds stay refusals; I/O
/// and malformed inputs refuse as `UnknownAsset` so the caller treats the
/// holder as a dead end for this attempt and tries alternates).
fn refuse_or_unreachable(error: &RedundancyError) -> FragmentReply {
    let reason = match error {
        RedundancyError::StaleGeneration { current, .. } => {
            RefuseReason::StaleGeneration { current: *current }
        }
        RedundancyError::StaleIncarnation { expected, .. } => {
            RefuseReason::StaleIncarnation { current: *expected }
        }
        RedundancyError::CorruptFragment { .. } => RefuseReason::HashMismatch,
        RedundancyError::Overloaded { .. } => RefuseReason::Oversize,
        // Absent data, I/O, and malformed inputs all mark this holder a
        // dead end for the attempt (callers try alternates, never wait).
        _ => RefuseReason::UnknownAsset,
    };
    FragmentReply::Refused { reason }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::FragmentKey;
    use crate::store::LocalFragmentStore;

    fn harness(
        incarnation: u64,
    ) -> (
        LoopbackTransport,
        Arc<Mutex<LocalFragmentStore>>,
        tempfile::TempDir,
    ) {
        let dir = tempfile::tempdir().expect("scratch");
        let (store, _) =
            LocalFragmentStore::open(&dir.path().join("store"), incarnation).expect("opens");
        let transport = LoopbackTransport::new();
        let holder = Arc::new(Mutex::new(store));
        transport.add_node(NodeId::from_u64(1), Arc::clone(&holder));
        (transport, holder, dir)
    }

    fn key() -> FragmentKey {
        FragmentKey {
            asset_kind: 0,
            asset_domain: 7,
            asset_hash: [11; 32],
            generation: 1,
            index: 0,
        }
    }

    #[test]
    fn put_get_has_drop_round_trip() {
        let (transport, _store, _guard) = harness(3);
        let target = PeerTarget {
            node: NodeId::from_u64(1),
            incarnation: 3,
        };
        let bytes = b"loopback bytes".to_vec();
        let content = blake3_256(&bytes);
        let put = FragmentRpc::Put {
            key: key(),
            role: 0,
            params_tag: 0,
            content,
            stored_len: bytes.len() as u64,
            target_incarnation: 3,
            bytes: bytes.clone(),
        };
        let reply =
            futures::executor::block_on(transport.call(target, put, Duration::from_secs(1)))
                .expect("puts");
        assert_eq!(reply, FragmentReply::Stored { content });
        let has = FragmentRpc::Has {
            key: key(),
            target_incarnation: 3,
        };
        let reply =
            futures::executor::block_on(transport.call(target, has, Duration::from_secs(1)))
                .expect("probes");
        assert_eq!(
            reply,
            FragmentReply::Presence {
                present: true,
                healthy: true,
            }
        );
        let get = FragmentRpc::Get {
            key: key(),
            target_incarnation: 3,
        };
        let reply =
            futures::executor::block_on(transport.call(target, get, Duration::from_secs(1)))
                .expect("gets");
        assert_eq!(
            reply,
            FragmentReply::Bytes {
                content,
                bytes: bytes.clone(),
            }
        );
        let drop = FragmentRpc::Drop {
            key: key(),
            target_incarnation: 3,
        };
        let reply =
            futures::executor::block_on(transport.call(target, drop, Duration::from_secs(1)))
                .expect("drops");
        assert_eq!(reply, FragmentReply::Dropped);
    }

    #[test]
    fn mutation_requires_discovered_incarnation() {
        let (transport, _store, _guard) = harness(4);
        let bytes = b"fenced".to_vec();
        let reply = futures::executor::block_on(transport.call(
            PeerTarget {
                node: NodeId::from_u64(1),
                incarnation: 0,
            },
            FragmentRpc::Put {
                key: key(),
                role: 0,
                params_tag: 0,
                content: blake3_256(&bytes),
                stored_len: bytes.len() as u64,
                target_incarnation: 0,
                bytes,
            },
            Duration::from_secs(1),
        ))
        .expect("answers");
        assert_eq!(
            reply,
            FragmentReply::Refused {
                reason: RefuseReason::StaleIncarnation { current: 4 },
            }
        );
    }

    #[test]
    fn stale_incarnation_refuses_with_current() {
        let (transport, _store, _guard) = harness(9);
        let target = PeerTarget {
            node: NodeId::from_u64(1),
            incarnation: 4,
        };
        let reply = futures::executor::block_on(transport.call(
            target,
            FragmentRpc::Has {
                key: key(),
                target_incarnation: 4,
            },
            Duration::from_secs(1),
        ))
        .expect("answers");
        assert_eq!(
            reply,
            FragmentReply::Refused {
                reason: RefuseReason::StaleIncarnation { current: 9 },
            }
        );
    }

    #[test]
    fn missing_nodes_and_fragments_fail_closed() {
        let (transport, _store, _guard) = harness(1);
        let gone = PeerTarget {
            node: NodeId::from_u64(77),
            incarnation: 1,
        };
        let error = futures::executor::block_on(transport.call(
            gone,
            FragmentRpc::Has {
                key: key(),
                target_incarnation: 1,
            },
            Duration::from_secs(1),
        ))
        .expect_err("unreachable");
        assert!(matches!(error, TransportFailure::Unreachable { .. }));
        let here = PeerTarget {
            node: NodeId::from_u64(1),
            incarnation: 1,
        };
        let reply = futures::executor::block_on(transport.call(
            here,
            FragmentRpc::Get {
                key: key(),
                target_incarnation: 1,
            },
            Duration::from_secs(1),
        ))
        .expect("answers");
        assert_eq!(
            reply,
            FragmentReply::Refused {
                reason: RefuseReason::UnknownAsset,
            }
        );
    }
}
