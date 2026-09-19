//! Follower durability gate: sidecar acquisition before Raft persistence.
//!
//! This is the heart of replicated large values. When `OpenRaft` asks the
//! follower store to persist entries referencing immutable sidecars, this
//! gate ensures every required sidecar is locally durable and verified
//! before the Raft append may complete:
//!
//! ```text
//! Raft append -> detect sidecar dependency
//!   -> if present: verify metadata, persist Raft record, barrier, return
//!   -> if missing: fetch manifest, verify, discover missing chunks,
//!      fetch only missing, verify hashes/length/domain, durably install,
//!      persist Raft entry, barrier, return
//! ```
//!
//! The follower never fakes persistence while transfer is incomplete. A
//! corrupt sidecar fails the transfer, prevents the append from becoming
//! durable, and surfaces as an `io::Error` (which halts the node loudly,
//! never a silent ACK).
//!
//! ## `OpenRaft` 0.10 persistence contract (audited, never 0.9)
//!
//! * `append` must return with entries readable; `IOFlushed` fires once
//!   entries are persisted. Replication ACKs imply durable persistence
//!   (see `store.rs` docs): the leader commits only on quorum flushed, so
//!   gating the flush gates the commit.
//! * The gate runs before the WAL barrier submit (no lock held across the
//!   fetch), so the reactor stays responsive while bulk moves on the
//!   bulk lane.
//! * Already-durable content-addressed chunks remain cached safely if the
//!   suffix truncates mid-fetch; no physical rollback is required.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use kivi_types::{ChunkId, ManifestId, NodeId, SecurityDomainId};

use crate::peer::{PeerChunkRequest, PeerManifestRequest, PeerRequest, PeerResponse};
use crate::sidecar::{ImmutableDependencies, SidecarError, SidecarStore};
use crate::transport::{PeerTransport, TransportError};
use crate::types::ConsensusGroupId;

/// Maximum concurrent chunk fetches per gated root (bounded bulk).
const MAX_CONCURRENT_CHUNK_FETCH: usize = 4;
/// Maximum inflight gated roots (bounded acquisition table).
const MAX_INFLIGHT_ROOTS: usize = 64;

/// Durability gate for sidecar-backed Raft entries.
///
/// Minimal source abstraction: today the source is the replication peer /
/// leader passed per call; later another replica, archive, object store,
/// or coded repair. Identity stays content-based, never file offsets.
#[derive(Debug, Clone)]
pub struct SidecarGate {
    store: SidecarStore,
    transport: PeerTransport,
    peers: Arc<Vec<NodeId>>,
    group: ConsensusGroupId,
    domain: SecurityDomainId,
    bulk_timeout: Duration,
    inflight: Arc<futures::lock::Mutex<HashSet<ManifestId>>>,
}

impl SidecarGate {
    /// Creates a gate over a sidecar store and peer mesh.
    ///
    /// `peers` names candidate sidecar sources (other replicas; the local
    /// node need not be listed because the fast path already checked the
    /// local store). Static topology: fixed at node open.
    #[must_use]
    pub fn new(
        store: SidecarStore,
        transport: PeerTransport,
        peers: Vec<NodeId>,
        group: ConsensusGroupId,
        domain: SecurityDomainId,
        bulk_timeout: Duration,
    ) -> Self {
        Self {
            store,
            transport,
            peers: Arc::new(peers),
            group,
            domain,
            bulk_timeout,
            inflight: Arc::new(futures::lock::Mutex::new(HashSet::new())),
        }
    }

    /// Orders sources leader-first when a hint is available (replication
    /// peer / leader today; archive / object store later).
    fn ordered_sources(&self, hint: Option<NodeId>) -> Vec<NodeId> {
        let mut out = Vec::with_capacity(self.peers.len());
        if let Some(leader) = hint
            && self.peers.contains(&leader)
        {
            out.push(leader);
        }
        for peer in self.peers.iter() {
            if Some(*peer) != hint {
                out.push(*peer);
            }
        }
        out
    }

    /// Returns the sidecar store.
    #[must_use]
    pub fn store(&self) -> &SidecarStore {
        &self.store
    }

    /// Returns the domain.
    #[must_use]
    pub const fn domain(&self) -> SecurityDomainId {
        self.domain
    }

    /// Extracts immutable dependencies from one replicated command's bytes.
    ///
    /// Returns an empty vector for small/state-local mutations (no
    /// sidecars) and for control-plane commands (never chunked). Chunked
    /// roots — direct or carried inside transaction prepares and local
    /// commits — each contribute one entry, so followers fetch every
    /// payload a commit may reference before the entry may apply.
    #[must_use]
    pub fn dependencies_of(command: &[u8]) -> Vec<ImmutableDependencies> {
        let Some(envelope_command) = crate::command::ConsensusCommand::decode_exact(command).ok()
        else {
            return Vec::new();
        };
        let Some(mutation) = envelope_command.as_tablet() else {
            return Vec::new();
        };
        let envelope = mutation.envelope();
        // Domain binds at apply time from namespace; the gate uses
        // its configured domain (namespace-derived, single-domain
        // lanes). Cross-domain manifests fail verification later.
        let domain = SecurityDomainId::from_u64(0);
        match envelope.operation() {
            kivi_state::Mutation::ReplaceChunkedRoot {
                manifest,
                logical_len,
                ..
            }
            | kivi_state::Mutation::ReplaceChunkedRootWithExpiry {
                manifest,
                logical_len,
                ..
            } => vec![ImmutableDependencies::new(*manifest, *logical_len, domain)],
            kivi_state::Mutation::TxnPrepare {
                write:
                    kivi_state::TxnWriteKind::PutChunked {
                        manifest,
                        logical_len,
                    },
                ..
            } => vec![ImmutableDependencies::new(*manifest, *logical_len, domain)],
            kivi_state::Mutation::TxnCommitLocal { writes, .. } => writes
                .iter()
                .filter_map(|write| match &write.kind {
                    kivi_state::TxnWriteKind::PutChunked {
                        manifest,
                        logical_len,
                    } => Some(ImmutableDependencies::new(*manifest, *logical_len, domain)),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Extracts dependencies with the gate's domain filled in.
    #[must_use]
    pub fn dependencies_of_with_domain(&self, command: &[u8]) -> Vec<ImmutableDependencies> {
        Self::dependencies_of(command)
            .into_iter()
            .map(|mut deps| {
                deps.domain = self.domain;
                deps
            })
            .collect()
    }

    /// Ensures one root is locally durable, fetching missing sidecars from
    /// peers. Concurrent callers for the same manifest coalesce: only one
    /// fetch runs while others wait on the inflight entry (bounded).
    ///
    /// # Errors
    ///
    /// Returns [`SidecarError`] on verification failure, missing source,
    /// overload, or shutdown. Callers map this to `io::Error` so the Raft
    /// append never reports durable persistence for incomplete sidecars.
    pub async fn ensure_root(
        &self,
        manifest: ManifestId,
        logical_len: u64,
        hint: Option<NodeId>,
    ) -> Result<(), SidecarError> {
        if !manifest.is_valid() {
            return Err(SidecarError::Corrupt {
                detail: "zero manifest id".to_owned(),
            });
        }
        // Fast path: already durable (dedup hit, no transfer).
        if self.store.check_root(manifest, logical_len).await? {
            self.store
                .metrics()
                .cache_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        self.store
            .metrics()
            .cache_misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Coalesce: bound the inflight table, wait if another task fetches
        // the same root.
        {
            let mut inflight = self.inflight.lock().await;
            if inflight.len() >= MAX_INFLIGHT_ROOTS && !inflight.contains(&manifest) {
                return Err(SidecarError::Overloaded {
                    detail: "too many inflight sidecar acquisitions".to_owned(),
                });
            }
            if !inflight.insert(manifest) {
                // Another task fetches; drop the lock and poll until done.
                drop(inflight);
                return self.wait_for_root(manifest, logical_len).await;
            }
            self.store
                .metrics()
                .inflight
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let sources = self.ordered_sources(hint);
        let result = self.fetch_root(manifest, logical_len, &sources).await;
        {
            self.inflight.lock().await.remove(&manifest);
            self.store
                .metrics()
                .inflight
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
        result
    }

    /// Ensures all sidecar dependencies across entries (batch entry point
    /// for `append`: resolves each manifest once, reuses across entries).
    ///
    /// # Errors
    ///
    /// Returns [`SidecarError`] when any root cannot be made durable.
    pub async fn ensure_entries(
        &self,
        commands: &[Vec<u8>],
        hint: Option<NodeId>,
    ) -> Result<(), SidecarError> {
        use std::collections::HashSet;
        self.store
            .metrics()
            .pending_gated
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let result = async {
            let mut seen: HashSet<(ManifestId, u64)> = HashSet::new();
            for command in commands {
                for deps in self.dependencies_of_with_domain(command) {
                    if seen.insert((deps.manifest, deps.logical_len)) {
                        self.ensure_root(deps.manifest, deps.logical_len, hint)
                            .await?;
                    }
                }
            }
            Ok(())
        }
        .await;
        self.store
            .metrics()
            .pending_gated
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        result
    }

    async fn wait_for_root(
        &self,
        manifest: ManifestId,
        logical_len: u64,
    ) -> Result<(), SidecarError> {
        // Poll with bounded waits: the fetcher holds the inflight entry
        // until durable or failed. Timeout surfaces as overload (caller
        // retries via Raft replication backoff).
        let deadline = std::time::Instant::now() + self.bulk_timeout * 2;
        loop {
            compio::time::sleep(Duration::from_millis(50)).await;
            if self.store.check_root(manifest, logical_len).await? {
                return Ok(());
            }
            if !self.inflight.lock().await.contains(&manifest) {
                // Fetcher finished but root still missing: fetch failed.
                return Err(SidecarError::Missing {
                    detail: format!("sidecar fetch for {manifest} did not complete"),
                });
            }
            if std::time::Instant::now() >= deadline {
                return Err(SidecarError::Overloaded {
                    detail: format!("coalesced fetch for {manifest} timed out"),
                });
            }
        }
    }

    async fn fetch_root(
        &self,
        manifest: ManifestId,
        logical_len: u64,
        sources: &[NodeId],
    ) -> Result<(), SidecarError> {
        self.store
            .metrics()
            .manifest_requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // 1. Manifest (single, small): fetch if not durable.
        if !self.store.is_durable_manifest(manifest).await? {
            let canonical = self.fetch_manifest(manifest, sources).await?;
            // Verify before install: id recompute + structural validation
            // + domain + length agreement (never trust peer-supplied ids).
            let decoded = kivi_chunk::verify_manifest(manifest, &canonical).map_err(|error| {
                self.store
                    .metrics()
                    .verification_failures
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                SidecarError::from(error)
            })?;
            if decoded.domain != self.domain {
                self.store
                    .metrics()
                    .verification_failures
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(SidecarError::Corrupt {
                    detail: format!("manifest {manifest} names a foreign domain"),
                });
            }
            if decoded.total_len != logical_len {
                self.store
                    .metrics()
                    .verification_failures
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(SidecarError::Corrupt {
                    detail: format!(
                        "manifest total {} != root logical {logical_len}",
                        decoded.total_len
                    ),
                });
            }
            self.store.stage_manifest(manifest, canonical).await?;
        }
        // 2. Discover missing chunks (dedup: only these move).
        let missing = self.store.missing_chunks(manifest).await?;
        if missing.is_empty() {
            self.store.sync().await?;
            // Pin for the applying entry's lifetime is handled by the
            // caller (proposal pins); here just prove durability.
            if !self.store.check_root(manifest, logical_len).await? {
                return Err(SidecarError::Missing {
                    detail: format!("root {manifest} not durable after no-op fetch"),
                });
            }
            return Ok(());
        }
        // 3. Fetch missing chunks with bounded concurrency, verifying each.
        // Read the manifest once for per-chunk length expectations.
        let canonical = self.store.read_manifest(manifest).await?;
        let decoded = kivi_chunk::verify_manifest(manifest, &canonical)?;
        let expected: HashMap<ChunkId, u64> = decoded
            .entries
            .iter()
            .map(|entry| (entry.id, entry.len))
            .collect();
        for batch in missing.chunks(MAX_CONCURRENT_CHUNK_FETCH) {
            let mut futures_list = Vec::with_capacity(batch.len());
            for chunk in batch {
                futures_list.push(self.fetch_and_stage_chunk(*chunk, &expected, sources));
            }
            let results = futures::future::join_all(futures_list).await;
            for result in results {
                result?;
            }
        }
        // 4. One barrier for the whole batch, then prove durable.
        self.store.sync().await?;
        if !self.store.check_root(manifest, logical_len).await? {
            return Err(SidecarError::Missing {
                detail: format!("root {manifest} not durable after fetch"),
            });
        }
        Ok(())
    }

    async fn fetch_and_stage_chunk(
        &self,
        chunk: ChunkId,
        expected: &HashMap<ChunkId, u64>,
        sources: &[NodeId],
    ) -> Result<(), SidecarError> {
        // Dedup re-check inside the batch (another entry may have staged).
        if self.store.is_durable_chunk(chunk).await? {
            self.store
                .metrics()
                .cache_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        self.store
            .metrics()
            .chunk_requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let bytes = self.fetch_chunk(chunk, sources).await?;
        // Verify: id recompute + length vs manifest entry.
        let computed = kivi_codec::integrity::chunk_id(self.domain, &bytes);
        if computed != chunk {
            self.store
                .metrics()
                .verification_failures
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.store
                .metrics()
                .bulk_errors
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(SidecarError::Corrupt {
                detail: format!("chunk id mismatch: claimed {chunk}, computed {computed}"),
            });
        }
        if let Some(expected_len) = expected.get(&chunk)
            && bytes.len() as u64 != *expected_len
        {
            self.store
                .metrics()
                .verification_failures
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(SidecarError::Corrupt {
                detail: format!("chunk {chunk} length mismatch"),
            });
        }
        self.store
            .metrics()
            .bulk_bytes_received
            .fetch_add(bytes.len() as u64, std::sync::atomic::Ordering::Relaxed);
        self.store.stage_chunk(chunk, bytes).await?;
        Ok(())
    }

    async fn fetch_manifest(
        &self,
        manifest: ManifestId,
        sources: &[NodeId],
    ) -> Result<Vec<u8>, SidecarError> {
        let mut last_error = None;
        for source in sources {
            let request = PeerRequest::Manifest(PeerManifestRequest {
                manifest: *manifest.as_bytes(),
            });
            match self
                .transport
                .call_bulk(*source, self.group, request, self.bulk_timeout)
                .await
            {
                Ok(PeerResponse::Manifest(response)) => {
                    if response.manifest != *manifest.as_bytes() {
                        self.store
                            .metrics()
                            .verification_failures
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        last_error = Some(SidecarError::Corrupt {
                            detail: "manifest response id mismatch".to_owned(),
                        });
                        continue;
                    }
                    self.store.metrics().bulk_bytes_received.fetch_add(
                        response.canonical.len() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    return Ok(response.canonical);
                }
                Ok(_) => {
                    last_error = Some(SidecarError::Corrupt {
                        detail: "peer answered manifest with wrong family".to_owned(),
                    });
                }
                Err(TransportError::RemoteRefused { detail }) => {
                    last_error = Some(SidecarError::Missing {
                        detail: format!("peer refused manifest: {detail}"),
                    });
                }
                Err(error) => {
                    self.store
                        .metrics()
                        .bulk_errors
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    last_error = Some(SidecarError::Missing {
                        detail: format!("manifest fetch: {error}"),
                    });
                }
            }
        }
        Err(last_error.unwrap_or(SidecarError::Missing {
            detail: format!("no source for manifest {manifest}"),
        }))
    }

    async fn fetch_chunk(
        &self,
        chunk: ChunkId,
        sources: &[NodeId],
    ) -> Result<Vec<u8>, SidecarError> {
        let mut last_error = None;
        for source in sources {
            let request = PeerRequest::Chunk(PeerChunkRequest {
                chunk: *chunk.as_bytes(),
            });
            match self
                .transport
                .call_bulk(*source, self.group, request, self.bulk_timeout)
                .await
            {
                Ok(PeerResponse::Chunk(response)) => {
                    if response.chunk != *chunk.as_bytes() {
                        self.store
                            .metrics()
                            .verification_failures
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        last_error = Some(SidecarError::Corrupt {
                            detail: "chunk response id mismatch".to_owned(),
                        });
                        continue;
                    }
                    return Ok(response.bytes);
                }
                Ok(_) => {
                    last_error = Some(SidecarError::Corrupt {
                        detail: "peer answered chunk with wrong family".to_owned(),
                    });
                }
                Err(TransportError::RemoteRefused { detail }) => {
                    last_error = Some(SidecarError::Missing {
                        detail: format!("peer refused chunk: {detail}"),
                    });
                }
                Err(error) => {
                    self.store
                        .metrics()
                        .bulk_errors
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    last_error = Some(SidecarError::Missing {
                        detail: format!("chunk fetch: {error}"),
                    });
                }
            }
        }
        Err(last_error.unwrap_or(SidecarError::Missing {
            detail: format!("no source for chunk {chunk}"),
        }))
    }
}

/// Maps a sidecar failure to the `io::Error` the Raft store must surface.
///
/// A corrupt/missing sidecar must fail the transfer, prevent the append
/// from becoming durable, and mark a transfer error — never install bad
/// bytes, ACK the append, or continue apply.
#[must_use]
pub fn sidecar_io_error(error: &SidecarError) -> std::io::Error {
    std::io::Error::other(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::future::Future;
    use std::sync::Arc;
    use std::time::Duration;

    use kivi_types::{ClusterId, NodeId, NodeIncarnation, SecurityDomainId};

    use super::SidecarGate;
    use crate::peer::{PeerIdentity, PeerRequest, PeerResponse, PeerRpcError};
    use crate::sidecar::SidecarStore;
    use crate::transport::{PeerHandler, PeerTransport, TlsMaterial, TransportConfig};
    use crate::types::ConsensusGroupId;

    const CLUSTER: u128 = 0x0006_0A7E;
    const GROUP_TABLET: u64 = 9;

    fn identity(node: u64) -> PeerIdentity {
        PeerIdentity {
            cluster: ClusterId::from_u128(CLUSTER),
            node: NodeId::from_u64(node),
            incarnation: NodeIncarnation::from_u64(1),
        }
    }

    /// Serves one staged value's sidecars, optionally corrupting chunk
    /// bytes to prove the fetcher rejects lies.
    struct Origin {
        manifests: HashMap<[u8; 32], Vec<u8>>,
        chunks: HashMap<[u8; 32], Vec<u8>>,
        corrupt_chunks: bool,
    }

    impl PeerHandler for Origin {
        fn handle(
            &self,
            _from: NodeId,
            _group: ConsensusGroupId,
            request: PeerRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<PeerResponse, PeerRpcError>> + Send + '_>,
        > {
            let manifests = self.manifests.clone();
            let chunks = self.chunks.clone();
            let corrupt = self.corrupt_chunks;
            Box::pin(async move {
                match request {
                    PeerRequest::Manifest(request) => manifests
                        .get(&request.manifest)
                        .map(|canonical| {
                            PeerResponse::Manifest(crate::peer::PeerManifestResponse {
                                manifest: request.manifest,
                                canonical: canonical.clone(),
                            })
                        })
                        .ok_or_else(|| PeerRpcError {
                            detail: "no such manifest".to_owned(),
                        }),
                    PeerRequest::Chunk(request) => chunks
                        .get(&request.chunk)
                        .map(|bytes| {
                            let mut bytes = bytes.clone();
                            if corrupt {
                                bytes[0] ^= 0xFF;
                            }
                            PeerResponse::Chunk(crate::peer::PeerChunkResponse {
                                chunk: request.chunk,
                                bytes,
                            })
                        })
                        .ok_or_else(|| PeerRpcError {
                            detail: "no such chunk".to_owned(),
                        }),
                    _ => Err(PeerRpcError {
                        detail: "origin serves sidecars only".to_owned(),
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

    /// Fixture: an honest origin (1), an evil twin serving bit-flipped
    /// chunks (2), and a fetcher (3) with an empty sidecar store. The
    /// directory guard travels with the outputs so open files and certs
    /// stay valid for the whole test.
    struct Fixture {
        manifest: kivi_types::ManifestId,
        len: u64,
        chunks: Vec<([u8; 32], Vec<u8>)>,
        value: Vec<u8>,
        gate: SidecarGate,
        store: SidecarStore,
        _transports: Vec<PeerTransport>,
        _dir: tempfile::TempDir,
    }

    #[allow(clippy::too_many_lines)]
    async fn setup() -> Fixture {
        let domain = SecurityDomainId::from_u64(1);
        let group = ConsensusGroupId::of_tablet(kivi_types::TabletId::from_u64(GROUP_TABLET));
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
        // Origin stages a 3-chunk value (3 MiB deterministic bytes).
        let (origin_store, _worker) =
            SidecarStore::open(&dir.path().join("origin"), domain, 64 * 1024 * 1024)
                .expect("origin store opens");
        let mut value = vec![0u8; 3 * 1024 * 1024];
        for (i, byte) in value.iter_mut().enumerate() {
            *byte = u8::try_from((i * 13 + 5) % 251).unwrap_or(0);
        }
        let staged = origin_store.stage_value(&value).await.expect("stages");
        let canonical = origin_store
            .read_manifest(staged.manifest)
            .await
            .expect("manifest readable");
        let mut chunks = Vec::new();
        for id in &staged.chunks {
            chunks.push((
                *id.as_bytes(),
                origin_store.read_chunk(*id).await.expect("chunk readable"),
            ));
        }
        let manifests: HashMap<[u8; 32], Vec<u8>> = [(*staged.manifest.as_bytes(), canonical)]
            .into_iter()
            .collect();
        let chunk_map: HashMap<[u8; 32], Vec<u8>> = chunks.clone().into_iter().collect();
        let evil_map = chunk_map.clone();
        let honest = Arc::new(Origin {
            manifests: manifests.clone(),
            chunks: chunk_map,
            corrupt_chunks: false,
        });
        let evil = Arc::new(Origin {
            manifests,
            chunks: evil_map,
            corrupt_chunks: true,
        });
        let fetcher_handler: Arc<dyn PeerHandler> = Arc::new(Origin {
            manifests: HashMap::new(),
            chunks: HashMap::new(),
            corrupt_chunks: false,
        });
        let (t1, _) = PeerTransport::open(
            TransportConfig::default(),
            identity(1),
            ClusterId::from_u128(CLUSTER),
            &peers,
            insecure_tls(&dir.path().join("t1"), 1),
            honest,
        )
        .await
        .expect("origin opens");
        let (t2, _) = PeerTransport::open(
            TransportConfig::default(),
            identity(2),
            ClusterId::from_u128(CLUSTER),
            &peers,
            insecure_tls(&dir.path().join("t2"), 2),
            evil,
        )
        .await
        .expect("evil opens");
        let (t3, _) = PeerTransport::open(
            TransportConfig::default(),
            identity(3),
            ClusterId::from_u128(CLUSTER),
            &peers,
            insecure_tls(&dir.path().join("t3"), 3),
            fetcher_handler,
        )
        .await
        .expect("fetcher opens");
        // The fetcher's gate may source from either origin.
        let (fetch_store, _fetch_worker) =
            SidecarStore::open(&dir.path().join("fetch"), domain, 64 * 1024 * 1024)
                .expect("fetch store opens");
        let gate = SidecarGate::new(
            fetch_store.clone(),
            t3.clone(),
            vec![NodeId::from_u64(1), NodeId::from_u64(2)],
            group,
            domain,
            Duration::from_secs(15),
        );
        Fixture {
            manifest: staged.manifest,
            len: staged.logical_len,
            chunks,
            value,
            gate,
            store: fetch_store,
            _transports: vec![t1, t2, t3],
            _dir: dir,
        }
    }

    #[test]
    fn gate_fetches_missing_sidecars_from_honest_source() {
        block_on(async {
            let fixture = setup().await;
            fixture
                .gate
                .ensure_root(fixture.manifest, fixture.len, Some(NodeId::from_u64(1)))
                .await
                .expect("honest fetch completes");
            assert_eq!(
                fixture
                    .store
                    .read_value(fixture.manifest, fixture.len)
                    .await
                    .expect("reads"),
                fixture.value
            );
        });
    }

    #[test]
    fn gate_rejects_corrupt_sidecars_and_installs_nothing() {
        block_on(async {
            let fixture = setup().await;
            let manifest = fixture.manifest;
            let len = fixture.len;
            let before = fixture
                .store
                .metrics()
                .verification_failures
                .load(std::sync::atomic::Ordering::Relaxed);
            let error = fixture
                .gate
                .ensure_root(manifest, len, Some(NodeId::from_u64(2)))
                .await
                .expect_err("corrupt fetch must fail");
            assert!(
                matches!(error, crate::sidecar::SidecarError::Corrupt { .. }),
                "corruption fails as Corrupt, got {error:?}"
            );
            // Nothing installed: the claimed chunk is absent and the root
            // is not durable, so no Raft append could count it persisted.
            for (id, _) in &fixture.chunks {
                assert!(
                    !fixture
                        .store
                        .is_durable_chunk(kivi_types::ChunkId::from_bytes(*id))
                        .await
                        .expect("checks"),
                    "corrupt chunk never becomes durable"
                );
            }
            assert!(
                !fixture
                    .store
                    .check_root(manifest, len)
                    .await
                    .expect("checks"),
                "root never becomes durable"
            );
            let after = fixture
                .store
                .metrics()
                .verification_failures
                .load(std::sync::atomic::Ordering::Relaxed);
            assert!(after > before, "verification failure is metered");
            // The error maps to `io::Error` for the Raft store, which fails
            // the append instead of acknowledging it.
            let _ = super::sidecar_io_error(&error);
        });
    }
}
