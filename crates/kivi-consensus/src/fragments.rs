//! Project-owned redundancy fragment transport over the peer mesh.
//!
//! Nodes move erasure/replication fragments (and published layout bytes)
//! as opaque [`kivi_redundancy::proto::FragmentRpc`] requests in
//! [`PeerRequest::Fragment`](crate::peer::PeerRequest) bodies on the bulk
//! H3 lane, group-independent like the immutable sidecars (callers pass
//! [`ConsensusGroupId::redundancy_sentinel`](crate::types::ConsensusGroupId::redundancy_sentinel);
//! serving ignores the group).
//!
//! ## Layering: transport moves bytes, store and coordinator verify
//!
//! [`FragmentClient`] never verifies: it delivers one RPC and returns the
//! decoded reply, including `Refused` answers and unverified `Bytes`
//! payloads. Hash verification lives in two places, neither here:
//!
//! * the holder's [`LocalFragmentStore`](kivi_redundancy::store::LocalFragmentStore)
//!   verifies before staging and re-verifies before serving (lies are
//!   refused, never stored, never served);
//! * the coordinator verifies what it receives against its layout records
//!   ([`FragmentId::verify_bytes`](kivi_redundancy::FragmentId::verify_bytes))
//!   before reconstructing.
//!
//! An evil peer serving bit-flipped bytes therefore passes this layer
//! untouched and fails at the coordinator — the tests prove both halves.
//!
//! ## Incarnation discipline
//!
//! The believed target incarnation comes from the control registry into
//! [`PeerTarget`](kivi_redundancy::transport::PeerTarget) and is stamped
//! onto the wire by [`FragmentClient`], overwriting whatever the caller
//! set inside the RPC. The holder serves target `0` (bootstrap probe)
//! unconditionally and refuses any other mismatch with
//! [`RefuseReason::StaleIncarnation`](kivi_redundancy::proto::RefuseReason)
//! carrying its current incarnation, so the coordinator updates its table
//! and retries exactly once with fresh belief.
//!
//! ## Bounds and metering
//!
//! Every call is bounded twice: proto-enforced sizes (the fragment caps
//! reject before allocation) plus the tighter of the per-call timeout and
//! the client ceiling. Requests encoding past
//! [`MAX_FRAGMENT_BODY_BYTES`](crate::transport::MAX_FRAGMENT_BODY_BYTES)
//! fail as [`TransportFailure::Oversize`](kivi_redundancy::transport::TransportFailure)
//! without touching the network. Bulk bytes are metered by the
//! transport's lane counters (which already count every bulk request and
//! byte); sidecar metrics are untouched.

use std::time::Duration;

use kivi_codec::integrity::blake3_256;
use kivi_redundancy::{
    AssetId, AssetKind, FragmentStore as _, LocalFragmentStore, RedundancyError,
    proto::{FragmentReply, FragmentRpc, RefuseReason},
    transport::{FragmentTransport, PeerTarget, TransportFailure},
};
use kivi_types::NodeId;

use crate::peer::{PeerRequest, PeerResponse, PeerRpcError};
use crate::transport::{PeerTransport, TransportError};
use crate::types::ConsensusGroupId;

/// Serves one fragment RPC against a node-local store: the single
/// implementation behind every replica's
/// [`PeerRequest::Fragment`](crate::peer::PeerRequest) handler, shared
/// with the wire tests so coverage proves the production path.
///
/// Protocol-level faults (undecodable version, truncation, CRC mismatch)
/// refuse the RPC itself ([`PeerRpcError`], mapped to retry-safe
/// transport errors); [`RedundancyError::Overloaded`] decodes map to an
/// encoded [`FragmentReply::Refused`] with [`RefuseReason::Oversize`] so
/// coordinators see a dead end for this attempt rather than a corrupt
/// peer. Every application verdict — stale incarnation/generation, hash
/// mismatch, unknown asset — returns as an encoded `Refused` reply (the
/// coordinator distinguishes retryable from fenced from it, never from a
/// transport error). A `None` store (fragments disabled on this node)
/// answers [`RefuseReason::ShuttingDown`].
pub(crate) fn handle_fragment_rpc(
    store: Option<&LocalFragmentStore>,
    local_incarnation: u64,
    request_bytes: &[u8],
) -> Result<PeerResponse, PeerRpcError> {
    let rpc = match FragmentRpc::decode(request_bytes) {
        Ok(rpc) => rpc,
        Err(error) => {
            if matches!(error, RedundancyError::Overloaded { .. }) {
                return Ok(refused(RefuseReason::Oversize));
            }
            return Err(PeerRpcError {
                detail: format!("fragment RPC undecodable: {error}"),
            });
        }
    };
    let target = target_of(&rpc);
    if target != 0 && target != local_incarnation {
        return Ok(refused(RefuseReason::StaleIncarnation {
            current: local_incarnation,
        }));
    }
    let Some(store) = store else {
        return Ok(refused(RefuseReason::ShuttingDown));
    };
    Ok(PeerResponse::Fragment(dispatch(store, rpc).encode()))
}

/// Encodes one application verdict as its H3 response body.
fn refused(reason: RefuseReason) -> PeerResponse {
    PeerResponse::Fragment(FragmentReply::Refused { reason }.encode())
}

/// Returns the incarnation the RPC addresses (`0` is the bootstrap probe).
fn target_of(rpc: &FragmentRpc) -> u64 {
    match rpc {
        FragmentRpc::Put {
            target_incarnation, ..
        }
        | FragmentRpc::Get {
            target_incarnation, ..
        }
        | FragmentRpc::Has {
            target_incarnation, ..
        }
        | FragmentRpc::Drop {
            target_incarnation, ..
        }
        | FragmentRpc::PutLayout {
            target_incarnation, ..
        }
        | FragmentRpc::GetLayout {
            target_incarnation, ..
        }
        | FragmentRpc::ProbeAsset {
            target_incarnation, ..
        }
        | FragmentRpc::SweepOrphans {
            target_incarnation, ..
        } => *target_incarnation,
    }
}

/// Executes one decoded fragment RPC against the store, mapping every
/// store fault to a fencing refusal (callers try alternates, never wait).
fn dispatch(store: &LocalFragmentStore, rpc: FragmentRpc) -> FragmentReply {
    match rpc {
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
            Err(error) => FragmentReply::Refused {
                reason: refuse_reason(&error),
            },
        },
        FragmentRpc::Get {
            key,
            target_incarnation,
        } => match store.fetch_fragment(&key, target_incarnation) {
            Ok(bytes) => FragmentReply::Bytes {
                content: blake3_256(&bytes),
                bytes,
            },
            Err(error) => FragmentReply::Refused {
                reason: refuse_reason(&error),
            },
        },
        FragmentRpc::Has {
            key,
            target_incarnation,
        } => {
            let (present, healthy) = store.has_fragment(&key, target_incarnation);
            FragmentReply::Presence { present, healthy }
        }
        FragmentRpc::Drop {
            key,
            target_incarnation,
        } => {
            // Idempotent: absent counts as removed.
            let _ = store.remove_fragment(&key, target_incarnation);
            FragmentReply::Dropped
        }
        FragmentRpc::SweepOrphans { target_incarnation } => {
            match store.sweep_orphans_at(target_incarnation) {
                Ok(count) => FragmentReply::Swept { count },
                Err(error) => FragmentReply::Refused {
                    reason: refuse_reason(&error),
                },
            }
        }
        other => dispatch_layout(store, other),
    }
}

/// Executes one decoded layout RPC against the store (same fencing
/// mapping as [`dispatch`]).
fn dispatch_layout(store: &LocalFragmentStore, rpc: FragmentRpc) -> FragmentReply {
    match rpc {
        FragmentRpc::PutLayout {
            asset_kind,
            asset_domain,
            asset_hash,
            generation,
            control_generation,
            layout,
            target_incarnation,
        } => match layout_asset_id(asset_kind, asset_domain, &asset_hash) {
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
                Err(error) => FragmentReply::Refused {
                    reason: refuse_reason(&error),
                },
            },
            Err(reason) => FragmentReply::Refused { reason },
        },
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
            Err(error) => FragmentReply::Refused {
                reason: refuse_reason(&error),
            },
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

/// Builds an asset identity from layout wire coordinates, rejecting
/// unknown kinds and the zero sentinel as an unknown asset (the holder is
/// a dead end for this attempt; the coordinator tries alternates).
fn layout_asset_id(
    asset_kind: u8,
    asset_domain: u64,
    asset_hash: &[u8; 32],
) -> Result<AssetId, RefuseReason> {
    let kind = AssetKind::from_u8(asset_kind).map_err(|_| RefuseReason::UnknownAsset)?;
    AssetId::new(kind, asset_domain, *asset_hash).map_err(|_| RefuseReason::UnknownAsset)
}

/// Maps store faults to fencing refusals: fencing and bounds stay
/// refusals with their kinds; absent data, I/O, and malformed inputs mark
/// this holder a dead end for the attempt.
fn refuse_reason(error: &RedundancyError) -> RefuseReason {
    match error {
        RedundancyError::StaleGeneration { current, .. } => {
            RefuseReason::StaleGeneration { current: *current }
        }
        RedundancyError::StaleIncarnation { expected, .. } => {
            RefuseReason::StaleIncarnation { current: *expected }
        }
        RedundancyError::CorruptFragment { .. } | RedundancyError::VerificationFailed { .. } => {
            RefuseReason::HashMismatch
        }
        RedundancyError::Overloaded { .. } => RefuseReason::Oversize,
        _ => RefuseReason::UnknownAsset,
    }
}

/// Stamps the believed target incarnation onto the wire RPC, overwriting
/// the inner field so the two can never disagree (the control registry's
/// belief in [`PeerTarget::incarnation`] is authoritative).
fn stamp_target(rpc: FragmentRpc, incarnation: u64) -> FragmentRpc {
    match rpc {
        FragmentRpc::Put {
            key,
            role,
            params_tag,
            content,
            stored_len,
            bytes,
            ..
        } => FragmentRpc::Put {
            key,
            role,
            params_tag,
            content,
            stored_len,
            target_incarnation: incarnation,
            bytes,
        },
        FragmentRpc::Get { key, .. } => FragmentRpc::Get {
            key,
            target_incarnation: incarnation,
        },
        FragmentRpc::Has { key, .. } => FragmentRpc::Has {
            key,
            target_incarnation: incarnation,
        },
        FragmentRpc::Drop { key, .. } => FragmentRpc::Drop {
            key,
            target_incarnation: incarnation,
        },
        FragmentRpc::PutLayout {
            asset_kind,
            asset_domain,
            asset_hash,
            generation,
            control_generation,
            layout,
            ..
        } => FragmentRpc::PutLayout {
            asset_kind,
            asset_domain,
            asset_hash,
            generation,
            control_generation,
            layout,
            target_incarnation: incarnation,
        },
        FragmentRpc::GetLayout {
            asset_kind,
            asset_domain,
            asset_hash,
            generation,
            ..
        } => FragmentRpc::GetLayout {
            asset_kind,
            asset_domain,
            asset_hash,
            generation,
            target_incarnation: incarnation,
        },
        FragmentRpc::ProbeAsset {
            asset_kind,
            asset_domain,
            asset_hash,
            ..
        } => FragmentRpc::ProbeAsset {
            asset_kind,
            asset_domain,
            asset_hash,
            target_incarnation: incarnation,
        },
        FragmentRpc::SweepOrphans { .. } => FragmentRpc::SweepOrphans {
            target_incarnation: incarnation,
        },
    }
}

/// One mesh call for the driver thread: everything reactor-local stays
/// on that thread; only the reply channel crosses back.
struct FragmentCommand {
    /// Holder to dial.
    target: NodeId,
    /// Group stamped on the H3 request.
    group: ConsensusGroupId,
    /// Opaque fragment request body.
    request: PeerRequest,
    /// Call deadline.
    timeout: Duration,
    /// Mesh answer back to the `Send` future.
    reply: futures::channel::oneshot::Sender<Result<PeerResponse, TransportError>>,
}

/// Driver handle: the `Send` half of the client bridge (see
/// [`FragmentClient`]).
#[derive(Debug, Clone)]
struct FragmentDriver {
    /// Commands for the driver thread (runtime-agnostic, `Send`).
    tx: async_channel::Sender<FragmentCommand>,
}

/// Runs mesh calls on the driver thread until the last client drops.
async fn driver_loop(transport: PeerTransport, commands: async_channel::Receiver<FragmentCommand>) {
    while let Ok(command) = commands.recv().await {
        let result = transport
            .call_bulk(
                command.target,
                command.group,
                command.request,
                command.timeout,
            )
            .await;
        let _ = command.reply.send(result);
    }
}

/// Peer-mesh delivery for fragment RPCs: sends one proto request over
/// [`PeerTransport::call_bulk`] under the redundancy sentinel group and
/// returns one decoded reply.
///
/// Store-free like [`SidecarGate`](crate::gate::SidecarGate): the holder
/// is addressed per call through [`PeerTarget`], so one client fans out
/// to every holder in its placement. Refusal of the RPC itself (peer
/// undecodable, wrong family, shutdown mid-flight) surfaces as a
/// transport error; a proto `Refused` answer returns as `Ok(Refused)` —
/// the coordinator (which owns retry policy and incarnation tables)
/// decides what each verdict means.
///
/// The mesh front's futures are reactor-local (`!Send`: Compio timers),
/// while [`FragmentTransport::call`] must return `Send`. One driver
/// thread per client bridges them: it parks the `!Send` mesh work on a
/// private Compio reactor, and the returned future only waits on a
/// runtime-agnostic channel — callable from any thread or runtime.
#[derive(Debug, Clone)]
pub struct FragmentClient {
    /// Shared QUIC/H3 mesh (bulk lane carries every call).
    transport: PeerTransport,
    /// Group stamped on the H3 request (always the redundancy sentinel;
    /// serving ignores it — the path carries no tablet).
    group: ConsensusGroupId,
    /// Ceiling for every call: the effective deadline is the tighter of
    /// this and the per-call timeout, so no call is ever unbounded.
    timeout: Duration,
    /// Bridge to the driver thread doing the `!Send` mesh work.
    driver: FragmentDriver,
}

impl FragmentClient {
    /// Creates a client over the shared mesh, spawning its driver thread.
    ///
    /// `group` must be the redundancy sentinel (serving ignores it);
    /// `timeout` ceilings every call below. Thread-spawn failure degrades
    /// to fail-closed calls ([`TransportFailure::ShuttingDown`]), never a
    /// panic.
    #[must_use]
    pub fn new(transport: PeerTransport, group: ConsensusGroupId, timeout: Duration) -> Self {
        let (tx, rx) = async_channel::bounded::<FragmentCommand>(64);
        let worker = transport.clone();
        // A failed spawn drops the receiver with the closure, so every
        // later send fails fast as `ShuttingDown`.
        let _ = std::thread::Builder::new()
            .name("kivi-fragment-client".to_owned())
            .spawn(move || {
                let Ok(runtime) = compio::runtime::Runtime::new() else {
                    return;
                };
                runtime.block_on(driver_loop(worker, rx));
            });
        Self {
            transport,
            group,
            timeout,
            driver: FragmentDriver { tx },
        }
    }

    /// Returns the shared mesh (admin diagnostics: per-peer stats count
    /// every fragment call as bulk traffic).
    #[must_use]
    pub fn transport(&self) -> &PeerTransport {
        &self.transport
    }
}

impl FragmentTransport for FragmentClient {
    fn call<'a>(
        &'a self,
        target: PeerTarget,
        rpc: FragmentRpc,
        timeout: Duration,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<FragmentReply, TransportFailure>> + Send + 'a>,
    > {
        Box::pin(async move {
            let encoded = stamp_target(rpc, target.incarnation).encode();
            if encoded.len() > crate::transport::MAX_FRAGMENT_BODY_BYTES {
                return Err(TransportFailure::Oversize);
            }
            let (tx, rx) = futures::channel::oneshot::channel();
            self.driver
                .tx
                .send(FragmentCommand {
                    target: target.node,
                    group: self.group,
                    request: PeerRequest::Fragment(encoded),
                    timeout: timeout.min(self.timeout),
                    reply: tx,
                })
                .await
                .map_err(|_| TransportFailure::ShuttingDown)?;
            let response = rx
                .await
                .map_err(|_| TransportFailure::ShuttingDown)?
                .map_err(|error| match error {
                    TransportError::UnknownPeer { node } => TransportFailure::Unreachable {
                        detail: format!("unknown fragment holder {node}"),
                    },
                    TransportError::Unreachable { detail } => {
                        TransportFailure::Unreachable { detail }
                    }
                    TransportError::Timeout => TransportFailure::Timeout,
                    TransportError::ShuttingDown => TransportFailure::ShuttingDown,
                    TransportError::RemoteRefused { detail } => TransportFailure::Unreachable {
                        detail: format!("peer refused fragment RPC: {detail}"),
                    },
                })?;
            let PeerResponse::Fragment(bytes) = response else {
                return Err(TransportFailure::Unreachable {
                    detail: "peer answered fragment with wrong family".to_owned(),
                });
            };
            match FragmentReply::decode(&bytes) {
                Ok(reply) => Ok(reply),
                Err(error) => {
                    if matches!(error, RedundancyError::Overloaded { .. }) {
                        Err(TransportFailure::Oversize)
                    } else {
                        Err(TransportFailure::Unreachable {
                            detail: format!("fragment reply undecodable: {error}"),
                        })
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::future::Future;
    use std::sync::Arc;
    use std::time::Duration;

    use kivi_codec::integrity::blake3_256;
    use kivi_redundancy::proto::FragmentKey;
    use kivi_redundancy::{FragmentStore as _, FragmentTransport as _};
    use kivi_types::{ClusterId, NodeId, NodeIncarnation};

    use super::{FragmentClient, handle_fragment_rpc};
    use crate::peer::{PeerIdentity, PeerRequest, PeerResponse, PeerRpcError};
    use crate::transport::{PeerHandler, PeerTransport, TlsMaterial, TransportConfig};
    use crate::types::ConsensusGroupId;

    const CLUSTER: u128 = 0x0F8A_6C71;
    const INCARNATION: u64 = 5;
    const GROUP_TABLET: u64 = 0;

    fn identity(node: u64) -> PeerIdentity {
        PeerIdentity {
            cluster: ClusterId::from_u128(CLUSTER),
            node: NodeId::from_u64(node),
            incarnation: NodeIncarnation::from_u64(1),
        }
    }

    /// Serves fragment RPCs from one node-local store through the
    /// production dispatch (the same function the consensus owner calls),
    /// so wire coverage proves the serving path, not a duplicate.
    struct Holder {
        store: Arc<kivi_redundancy::LocalFragmentStore>,
        incarnation: u64,
    }

    impl PeerHandler for Holder {
        fn handle(
            &self,
            _from: NodeId,
            _group: ConsensusGroupId,
            request: PeerRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<PeerResponse, PeerRpcError>> + Send + '_>,
        > {
            let store = Arc::clone(&self.store);
            let incarnation = self.incarnation;
            Box::pin(async move {
                match request {
                    PeerRequest::Fragment(bytes) => {
                        handle_fragment_rpc(Some(&store), incarnation, &bytes)
                    }
                    _ => Err(PeerRpcError {
                        detail: "holder serves fragments only".to_owned(),
                    }),
                }
            })
        }
    }

    /// Serves one crafted `Bytes` answer with bit-flipped payload under the
    /// true content hash: an evil peer the transport must not catch (the
    /// coordinator's `verify_bytes` does).
    struct Evil {
        content: [u8; 32],
        bytes: Vec<u8>,
    }

    impl PeerHandler for Evil {
        fn handle(
            &self,
            _from: NodeId,
            _group: ConsensusGroupId,
            request: PeerRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<PeerResponse, PeerRpcError>> + Send + '_>,
        > {
            let reply = kivi_redundancy::proto::FragmentReply::Bytes {
                content: self.content,
                bytes: self.bytes.clone(),
            }
            .encode();
            Box::pin(async move {
                match request {
                    PeerRequest::Fragment(_) => Ok(PeerResponse::Fragment(reply)),
                    _ => Err(PeerRpcError {
                        detail: "evil serves fragments only".to_owned(),
                    }),
                }
            })
        }
    }

    fn block_on<F, T>(future: F) -> T
    where
        F: Future<Output = T>,
    {
        compio::runtime::Runtime::new()
            .expect("compio test runtime")
            .block_on(future)
    }

    fn probe_udp() -> std::net::SocketAddr {
        std::net::UdpSocket::bind("127.0.0.1:0")
            .expect("probe binds")
            .local_addr()
            .expect("probe addr")
    }

    fn insecure_tls(dir: &std::path::Path, node: u64) -> TlsMaterial {
        TlsMaterial {
            cert: crate::tls::NodeCert::load_or_generate(dir, NodeId::from_u64(node))
                .expect("test cert generates"),
            peer_certs: HashMap::new(),
            trust: None,
            insecure_skip_verify: true,
        }
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

    fn target(node: u64) -> kivi_redundancy::transport::PeerTarget {
        kivi_redundancy::transport::PeerTarget {
            node: NodeId::from_u64(node),
            incarnation: INCARNATION,
        }
    }

    /// Fixture: two holders (nodes 1–2) with genuinely separate stores
    /// (own tempdirs, no shared filesystem) plus a store-free client
    /// (node 3) over real QUIC/H3 loopback.
    struct Fixture {
        client: FragmentClient,
        holder_a: Arc<kivi_redundancy::LocalFragmentStore>,
        _transports: Vec<PeerTransport>,
        _dir: tempfile::TempDir,
    }

    #[allow(clippy::too_many_lines)]
    async fn setup() -> Fixture {
        let group = ConsensusGroupId::redundancy_sentinel();
        assert_eq!(GROUP_TABLET, group.tablet().as_u64());
        let addrs = [probe_udp(), probe_udp(), probe_udp()];
        let peers: HashMap<NodeId, std::net::SocketAddr> = [1u64, 2, 3]
            .into_iter()
            .map(|id| {
                (
                    NodeId::from_u64(id),
                    addrs[usize::try_from(id).unwrap_or(1) - 1],
                )
            })
            .collect();
        let dir = tempfile::tempdir().expect("scratch");
        let (store_a, _) =
            kivi_redundancy::LocalFragmentStore::open(&dir.path().join("a"), INCARNATION)
                .expect("store a opens");
        let (store_b, _) =
            kivi_redundancy::LocalFragmentStore::open(&dir.path().join("b"), INCARNATION)
                .expect("store b opens");
        let holder_a = Arc::new(store_a);
        let holder_b = Arc::new(store_b);
        let (t1, _) = PeerTransport::open(
            TransportConfig::default(),
            identity(1),
            ClusterId::from_u128(CLUSTER),
            &peers,
            insecure_tls(&dir.path().join("t1"), 1),
            Arc::new(Holder {
                store: Arc::clone(&holder_a),
                incarnation: INCARNATION,
            }),
        )
        .await
        .expect("holder a opens");
        let (t2, _) = PeerTransport::open(
            TransportConfig::default(),
            identity(2),
            ClusterId::from_u128(CLUSTER),
            &peers,
            insecure_tls(&dir.path().join("t2"), 2),
            Arc::new(Holder {
                store: holder_b,
                incarnation: INCARNATION,
            }),
        )
        .await
        .expect("holder b opens");
        let (t3, _) = PeerTransport::open(
            TransportConfig::default(),
            identity(3),
            ClusterId::from_u128(CLUSTER),
            &peers,
            insecure_tls(&dir.path().join("t3"), 3),
            Arc::new(Holder {
                store: Arc::clone(&holder_a),
                incarnation: INCARNATION,
            }),
        )
        .await
        .expect("client opens");
        let client = FragmentClient::new(t3.clone(), group, Duration::from_secs(15));
        Fixture {
            client,
            holder_a,
            _transports: vec![t1, t2, t3],
            _dir: dir,
        }
    }

    fn put_rpc(bytes: &[u8]) -> kivi_redundancy::proto::FragmentRpc {
        kivi_redundancy::proto::FragmentRpc::Put {
            key: key(),
            role: 0,
            params_tag: 0,
            content: blake3_256(bytes),
            stored_len: bytes.len() as u64,
            target_incarnation: INCARNATION,
            bytes: bytes.to_vec(),
        }
    }

    #[test]
    fn put_get_round_trip_verifies_bytes() {
        block_on(async {
            let fixture = setup().await;
            let bytes = b"redundancy fragment payload".to_vec();
            let content = blake3_256(&bytes);
            let reply = fixture
                .client
                .call(target(1), put_rpc(&bytes), Duration::from_secs(15))
                .await
                .expect("puts");
            assert_eq!(
                reply,
                kivi_redundancy::proto::FragmentReply::Stored { content }
            );
            let reply = fixture
                .client
                .call(
                    target(1),
                    kivi_redundancy::proto::FragmentRpc::Get {
                        key: key(),
                        target_incarnation: INCARNATION,
                    },
                    Duration::from_secs(15),
                )
                .await
                .expect("gets");
            assert_eq!(
                reply,
                kivi_redundancy::proto::FragmentReply::Bytes { content, bytes }
            );
            // Separate stores share nothing: holder B never saw the put.
            let reply = fixture
                .client
                .call(
                    target(2),
                    kivi_redundancy::proto::FragmentRpc::Has {
                        key: key(),
                        target_incarnation: INCARNATION,
                    },
                    Duration::from_secs(15),
                )
                .await
                .expect("probes");
            assert_eq!(
                reply,
                kivi_redundancy::proto::FragmentReply::Presence {
                    present: false,
                    healthy: false,
                }
            );
        });
    }

    #[test]
    fn hash_mismatch_put_refused_and_nothing_stored() {
        block_on(async {
            let fixture = setup().await;
            let reply = fixture
                .client
                .call(
                    target(1),
                    kivi_redundancy::proto::FragmentRpc::Put {
                        key: key(),
                        role: 0,
                        params_tag: 0,
                        content: [9; 32],
                        stored_len: 3,
                        target_incarnation: INCARNATION,
                        bytes: b"abc".to_vec(),
                    },
                    Duration::from_secs(15),
                )
                .await
                .expect("answers");
            assert_eq!(
                reply,
                kivi_redundancy::proto::FragmentReply::Refused {
                    reason: kivi_redundancy::proto::RefuseReason::HashMismatch,
                }
            );
            assert_eq!(
                fixture.holder_a.has_fragment(&key(), INCARNATION),
                (false, false),
                "lies are never indexed"
            );
        });
    }

    #[test]
    fn stale_incarnation_refused_with_current_but_bootstrap_allowed() {
        block_on(async {
            let fixture = setup().await;
            let stale = kivi_redundancy::transport::PeerTarget {
                node: NodeId::from_u64(1),
                incarnation: INCARNATION - 1,
            };
            let reply = fixture
                .client
                .call(
                    stale,
                    kivi_redundancy::proto::FragmentRpc::Get {
                        key: key(),
                        target_incarnation: INCARNATION - 1,
                    },
                    Duration::from_secs(15),
                )
                .await
                .expect("answers");
            assert_eq!(
                reply,
                kivi_redundancy::proto::FragmentReply::Refused {
                    reason: kivi_redundancy::proto::RefuseReason::StaleIncarnation {
                        current: INCARNATION,
                    },
                }
            );
            // The bootstrap probe (target 0) is always servable.
            let probe = kivi_redundancy::transport::PeerTarget {
                node: NodeId::from_u64(1),
                incarnation: 0,
            };
            let reply = fixture
                .client
                .call(
                    probe,
                    kivi_redundancy::proto::FragmentRpc::Has {
                        key: key(),
                        target_incarnation: 0,
                    },
                    Duration::from_secs(15),
                )
                .await
                .expect("answers");
            assert_eq!(
                reply,
                kivi_redundancy::proto::FragmentReply::Presence {
                    present: false,
                    healthy: false,
                },
                "probe serves without fencing"
            );
        });
    }

    #[test]
    fn unknown_asset_get_refused_as_missing() {
        block_on(async {
            let fixture = setup().await;
            let reply = fixture
                .client
                .call(
                    target(1),
                    kivi_redundancy::proto::FragmentRpc::Get {
                        key: key(),
                        target_incarnation: INCARNATION,
                    },
                    Duration::from_secs(15),
                )
                .await
                .expect("answers");
            assert_eq!(
                reply,
                kivi_redundancy::proto::FragmentReply::Refused {
                    reason: kivi_redundancy::proto::RefuseReason::UnknownAsset,
                }
            );
        });
    }

    #[test]
    fn drop_is_idempotent() {
        block_on(async {
            let fixture = setup().await;
            let drop = || kivi_redundancy::proto::FragmentRpc::Drop {
                key: key(),
                target_incarnation: INCARNATION,
            };
            // Absent drops count as dropped.
            let reply = fixture
                .client
                .call(target(1), drop(), Duration::from_secs(15))
                .await
                .expect("answers");
            assert_eq!(reply, kivi_redundancy::proto::FragmentReply::Dropped);
            // Present drops remove, and re-drops stay dropped.
            let bytes = b"droppable".to_vec();
            fixture
                .client
                .call(target(1), put_rpc(&bytes), Duration::from_secs(15))
                .await
                .expect("puts");
            let reply = fixture
                .client
                .call(target(1), drop(), Duration::from_secs(15))
                .await
                .expect("answers");
            assert_eq!(reply, kivi_redundancy::proto::FragmentReply::Dropped);
            let reply = fixture
                .client
                .call(target(1), drop(), Duration::from_secs(15))
                .await
                .expect("answers");
            assert_eq!(reply, kivi_redundancy::proto::FragmentReply::Dropped);
            assert_eq!(
                fixture.holder_a.has_fragment(&key(), INCARNATION),
                (false, false)
            );
        });
    }

    /// Fixture: an evil holder (node 1) serving bit-flipped bytes under
    /// the true content hash, plus a client (node 2) over real QUIC/H3.
    /// The directory guard travels with the outputs so certs stay valid.
    async fn setup_evil(
        content: [u8; 32],
        served: Vec<u8>,
    ) -> (FragmentClient, Vec<PeerTransport>, tempfile::TempDir) {
        let addrs = [probe_udp(), probe_udp()];
        let peers: HashMap<NodeId, std::net::SocketAddr> = [1u64, 2]
            .into_iter()
            .map(|id| {
                (
                    NodeId::from_u64(id),
                    addrs[usize::try_from(id).unwrap_or(1) - 1],
                )
            })
            .collect();
        let dir = tempfile::tempdir().expect("scratch");
        let (evil_transport, _) = PeerTransport::open(
            TransportConfig::default(),
            identity(1),
            ClusterId::from_u128(CLUSTER),
            &peers,
            insecure_tls(&dir.path().join("e1"), 1),
            Arc::new(Evil {
                content,
                bytes: served,
            }),
        )
        .await
        .expect("evil opens");
        let (client_transport, _) = PeerTransport::open(
            TransportConfig::default(),
            identity(2),
            ClusterId::from_u128(CLUSTER),
            &peers,
            insecure_tls(&dir.path().join("e2"), 2),
            Arc::new(Evil {
                content,
                bytes: Vec::new(),
            }),
        )
        .await
        .expect("client opens");
        let client = FragmentClient::new(
            client_transport.clone(),
            ConsensusGroupId::redundancy_sentinel(),
            Duration::from_secs(15),
        );
        (client, vec![evil_transport, client_transport], dir)
    }

    #[test]
    fn evil_bytes_pass_transport_and_fail_coordinator_verification() {
        block_on(async {
            let bytes = b"honest fragment bytes".to_vec();
            let content = blake3_256(&bytes);
            let mut flipped = bytes.clone();
            flipped[0] ^= 0xFF;
            let (client, _transports, _dir) = setup_evil(content, flipped.clone()).await;
            // The transport moves the lie untouched: no verification here.
            let reply = client
                .call(
                    target(1),
                    kivi_redundancy::proto::FragmentRpc::Get {
                        key: key(),
                        target_incarnation: INCARNATION,
                    },
                    Duration::from_secs(15),
                )
                .await
                .expect("answers");
            assert_eq!(
                reply,
                kivi_redundancy::proto::FragmentReply::Bytes {
                    content,
                    bytes: flipped,
                }
            );
            // The coordinator catches it against its layout record.
            let asset =
                kivi_redundancy::AssetId::new(kivi_redundancy::AssetKind::Chunk, 7, [11; 32])
                    .expect("asset");
            let id = kivi_redundancy::FragmentId {
                asset,
                generation: 1,
                index: 0,
                role: kivi_redundancy::FragmentRole::Replica,
                params_tag: 0,
                content,
            };
            let kivi_redundancy::proto::FragmentReply::Bytes { bytes, .. } = reply else {
                unreachable!("evil peer must answer the Get with Bytes, got {reply:?}");
            };
            assert!(
                id.verify_bytes(&bytes).is_err(),
                "coordinator verification rejects flipped bytes"
            );
            // And the store refuses the same lie at stage time.
            let dir = tempfile::tempdir().expect("scratch");
            let (store, _) =
                kivi_redundancy::LocalFragmentStore::open(&dir.path().join("s"), INCARNATION)
                    .expect("opens");
            assert!(
                store
                    .stage_fragment(
                        &key(),
                        0,
                        0,
                        content,
                        bytes.len() as u64,
                        &bytes,
                        INCARNATION
                    )
                    .is_err(),
                "store refuses bad put"
            );
            assert_eq!(store.has_fragment(&key(), INCARNATION), (false, false));
        });
    }

    #[test]
    fn oversize_put_refused() {
        block_on(async {
            let fixture = setup().await;
            let big = vec![7u8; kivi_redundancy::MAX_FRAGMENT_WIRE_BYTES + 1];
            let reply = fixture
                .client
                .call(target(1), put_rpc(&big), Duration::from_secs(15))
                .await
                .expect("answers");
            assert_eq!(
                reply,
                kivi_redundancy::proto::FragmentReply::Refused {
                    reason: kivi_redundancy::proto::RefuseReason::Oversize,
                }
            );
            assert_eq!(
                fixture.holder_a.has_fragment(&key(), INCARNATION),
                (false, false),
                "oversize stages nothing"
            );
        });
    }
}
