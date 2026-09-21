//! Replicated immutable sidecars: chunk/manifest storage for Raft roots.
//!
//! A replicated large value is a tiny deterministic Raft mutation
//! ([`ReplaceChunkedRoot`](kivi_state::Mutation::ReplaceChunkedRoot))
//! plus immutable content-addressed sidecars (chunks + manifest) that make
//! the root meaningful. Raft remains authoritative for ordering; the chunk
//! fabric provides the referenced bytes.
//!
//! ```text
//! leader stages chunks + manifest (durable locally)
//!   -> proposes tiny ReplaceChunkedRoot { manifest, logical_len }
//!   -> followers acquire missing sidecars, verify, durably install
//!   -> followers may ACK the Raft append (durable persistence gate)
//!   -> quorum commit -> apply root -> reply
//! ```
//!
//! ## Core invariant (never weakened)
//!
//! A replica MUST NOT contribute a durable Raft replication acknowledgement
//! for an entry referencing [`ManifestId`] `M` until `M` and every chunk it
//! references are locally durable and verified, with all chunk-store
//! barriers complete. Only then may the Raft append complete.
//!
//! ## Identity
//!
//! [`ChunkId`], [`ManifestId`] are the single project-owned vocabulary for
//! immutable data (no `NetworkChunkId` / `RaftChunkId` parallels). The
//! network wraps them in request messages; identity stays shared. Physical
//! pack offsets never cross the network.
//!
//! ## Storage independence
//!
//! Peers request [`ChunkId`] / [`ManifestId`] and receive bytes. They never
//! learn whether the holder keeps packs, separate files, or a future tier.
//!
//! ## Pin lifecycle
//!
//! ```text
//! staged -> proposal-pinned -> committed/applied referenced
//!        -> checkpoint referenced -> reclaimable
//! ```
//!
//! A proposal pin protects the root and all sidecars while the proposal is
//! in flight, leadership changes, or the client response pends. After
//! commit/apply ownership transfers to live-state references. After
//! rejection the pin releases (orphans age out via deferred GC, never
//! immediate deletion).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::thread::JoinHandle;

use kivi_types::{ChunkId, ManifestId, SecurityDomainId};

/// Immutable dependencies of one replicated mutation.
///
/// The Raft entry carries only [`ManifestId`] + logical length + domain
/// metadata for deterministic apply; followers resolve the chunk set from
/// the manifest itself. The full chunk list never enters the Raft log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImmutableDependencies {
    /// Manifest addressing the value.
    pub manifest: ManifestId,
    /// Total logical bytes.
    pub logical_len: u64,
    /// Security domain the ids were derived under.
    pub domain: SecurityDomainId,
}

impl ImmutableDependencies {
    /// Builds the dependency record for one chunked root.
    #[must_use]
    pub const fn new(manifest: ManifestId, logical_len: u64, domain: SecurityDomainId) -> Self {
        Self {
            manifest,
            logical_len,
            domain,
        }
    }

    /// Whether this names a plausible root (non-zero manifest).
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.manifest.is_valid()
    }
}

/// Lifecycle stage of a pinned immutable root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinStage {
    /// Staged but not yet proposed (volatile until synced).
    Staged,
    /// Proposal in flight (protects across leadership change + reply).
    ProposalPinned,
    /// Committed/applied live-state reference.
    Committed,
    /// Referenced by a retained checkpoint.
    CheckpointReferenced,
    /// Eligible for deferred orphan reclamation.
    Reclaimable,
}

/// Reference counts protecting immutable roots from reclamation.
///
/// Conservative: anything pinned here plus anything reachable from live
/// roots / retained checkpoints is kept. Distributed global GC is out of
/// scope; unsafe reclamation is never attempted.
#[derive(Debug, Default)]
pub struct SidecarPins {
    manifests: HashMap<ManifestId, usize>,
    chunks: HashMap<ChunkId, usize>,
}

impl SidecarPins {
    /// Creates empty pins.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Pins one manifest for the proposal lifetime.
    pub fn pin_manifest(&mut self, id: ManifestId) {
        *self.manifests.entry(id).or_insert(0) += 1;
    }

    /// Pins one chunk for the proposal lifetime.
    pub fn pin_chunk(&mut self, id: ChunkId) {
        *self.chunks.entry(id).or_insert(0) += 1;
    }

    /// Pins a full root (manifest + chunks).
    pub fn pin_root(&mut self, manifest: ManifestId, chunks: &[ChunkId]) {
        self.pin_manifest(manifest);
        for chunk in chunks {
            self.pin_chunk(*chunk);
        }
    }

    /// Releases one manifest pin.
    pub fn unpin_manifest(&mut self, id: &ManifestId) {
        if let Some(count) = self.manifests.get_mut(id) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.manifests.remove(id);
            }
        }
    }

    /// Releases one chunk pin.
    pub fn unpin_chunk(&mut self, id: &ChunkId) {
        if let Some(count) = self.chunks.get_mut(id) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.chunks.remove(id);
            }
        }
    }

    /// Releases a full root.
    pub fn unpin_root(&mut self, manifest: &ManifestId, chunks: &[ChunkId]) {
        self.unpin_manifest(manifest);
        for chunk in chunks {
            self.unpin_chunk(chunk);
        }
    }

    /// Snapshots pinned sets for GC planning.
    #[must_use]
    pub fn snapshot(&self) -> (HashSet<ManifestId>, HashSet<ChunkId>) {
        (
            self.manifests.keys().copied().collect(),
            self.chunks.keys().copied().collect(),
        )
    }

    /// Number of pinned manifests (diagnostics).
    #[must_use]
    pub fn manifest_count(&self) -> usize {
        self.manifests.len()
    }

    /// Number of pinned chunks (diagnostics).
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Whether anything is pinned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.manifests.is_empty() && self.chunks.is_empty()
    }
}

/// RAII proposal pin: holds manifest + chunk pins until commit/abandon.
///
/// Dropping releases the pins (orphans remain content-addressed and cached
/// safely; deferred GC ages them out, never immediate deletion).
#[derive(Debug)]
pub struct ProposalPin {
    pins: Arc<futures::lock::Mutex<SidecarPins>>,
    manifest: Option<ManifestId>,
    chunks: Vec<ChunkId>,
    released: bool,
}

impl ProposalPin {
    /// Creates a pin handle (pins acquired separately via [`SidecarStore`]).
    pub(crate) fn new(
        pins: Arc<futures::lock::Mutex<SidecarPins>>,
        manifest: ManifestId,
        chunks: Vec<ChunkId>,
    ) -> Self {
        Self {
            pins,
            manifest: Some(manifest),
            chunks,
            released: false,
        }
    }

    /// Consumes the pin after commit (ownership transfers to live state).
    /// Releases the proposal counts; live roots keep data alive.
    pub async fn commit(mut self) {
        self.release().await;
        std::mem::forget(self);
    }

    /// Abandons the proposal (rejection, leadership loss).
    pub async fn abandon(mut self) {
        self.release().await;
        std::mem::forget(self);
    }

    /// Returns the pinned manifest.
    #[must_use]
    pub fn manifest(&self) -> Option<ManifestId> {
        self.manifest
    }

    /// Returns the pinned chunks.
    #[must_use]
    pub fn chunks(&self) -> &[ChunkId] {
        &self.chunks
    }

    async fn release(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        if let Some(manifest) = self.manifest.take() {
            let chunks = std::mem::take(&mut self.chunks);
            let mut pins = self.pins.lock().await;
            pins.unpin_root(&manifest, &chunks);
        }
    }
}

impl Drop for ProposalPin {
    fn drop(&mut self) {
        // Async release impossible in Drop; the counts leak conservatively
        // (safe: retention, never unsafe reclamation). The holder should
        // call commit/abandon explicitly.
        let _ = self.released;
        let _ = self.manifest.is_none();
    }
}

/// Sidecar replication metrics (per node, atomics for lock-free reads).
#[derive(Debug, Default)]
pub struct SidecarMetrics {
    /// Bulk bytes sent to peers.
    pub bulk_bytes_sent: AtomicU64,
    /// Bulk bytes received from peers.
    pub bulk_bytes_received: AtomicU64,
    /// Manifest requests issued.
    pub manifest_requests: AtomicU64,
    /// Chunk requests issued.
    pub chunk_requests: AtomicU64,
    /// Local store hits (dedup, no transfer).
    pub cache_hits: AtomicU64,
    /// Local store misses (transfer required).
    pub cache_misses: AtomicU64,
    /// Bytes avoided via dedup.
    pub deduped_bytes: AtomicU64,
    /// Bulk transfer errors.
    pub bulk_errors: AtomicU64,
    /// Content verification failures.
    pub verification_failures: AtomicU64,
    /// Currently gated appends.
    pub pending_gated: AtomicU64,
    /// Current inflight acquisitions.
    pub inflight: AtomicU64,
}

impl SidecarMetrics {
    /// Creates zeroed metrics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshots counters for admin.
    #[must_use]
    pub fn snapshot(&self) -> SidecarMetricsSnapshot {
        SidecarMetricsSnapshot {
            bulk_bytes_sent: self.bulk_bytes_sent.load(Ordering::Relaxed),
            bulk_bytes_received: self.bulk_bytes_received.load(Ordering::Relaxed),
            manifest_requests: self.manifest_requests.load(Ordering::Relaxed),
            chunk_requests: self.chunk_requests.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            deduped_bytes: self.deduped_bytes.load(Ordering::Relaxed),
            bulk_errors: self.bulk_errors.load(Ordering::Relaxed),
            verification_failures: self.verification_failures.load(Ordering::Relaxed),
            pending_gated: self.pending_gated.load(Ordering::Relaxed),
            inflight: self.inflight.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time sidecar metrics for admin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SidecarMetricsSnapshot {
    /// Bulk bytes sent.
    pub bulk_bytes_sent: u64,
    /// Bulk bytes received.
    pub bulk_bytes_received: u64,
    /// Manifest requests.
    pub manifest_requests: u64,
    /// Chunk requests.
    pub chunk_requests: u64,
    /// Cache hits.
    pub cache_hits: u64,
    /// Cache misses.
    pub cache_misses: u64,
    /// Deduped bytes.
    pub deduped_bytes: u64,
    /// Bulk errors.
    pub bulk_errors: u64,
    /// Verification failures.
    pub verification_failures: u64,
    /// Pending gated appends.
    pub pending_gated: u64,
    /// Inflight acquisitions.
    pub inflight: u64,
}

/// Why sidecar storage failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SidecarError {
    /// Underlying chunk-store I/O.
    #[error("sidecar I/O failed: {detail}")]
    Io {
        /// Human-readable cause.
        detail: String,
    },
    /// Content verification failed (claimed id != computed).
    #[error("sidecar verification failed: {detail}")]
    Corrupt {
        /// Human-readable cause.
        detail: String,
    },
    /// Missing immutable object with no available source.
    #[error("sidecar missing: {detail}")]
    Missing {
        /// Human-readable cause.
        detail: String,
    },
    /// Bounded resources exhausted.
    #[error("sidecar overloaded: {detail}")]
    Overloaded {
        /// Human-readable cause.
        detail: String,
    },
    /// Shutting down.
    #[error("sidecar store shut down")]
    ShuttingDown,
}

impl From<kivi_chunk::ChunkError> for SidecarError {
    fn from(error: kivi_chunk::ChunkError) -> Self {
        use kivi_chunk::ChunkError as E;
        match error {
            E::CorruptChunk { id, detail } => Self::Corrupt {
                detail: format!("chunk {id}: {detail}"),
            },
            E::CorruptManifest { id, detail } => Self::Corrupt {
                detail: format!("manifest {id}: {detail}"),
            },
            E::MissingChunk { id } => Self::Missing {
                detail: format!("chunk {id}"),
            },
            E::MissingManifest { id } => Self::Missing {
                detail: format!("manifest {id}"),
            },
            E::Overloaded => Self::Overloaded {
                detail: "chunk lane saturated".to_owned(),
            },
            other => Self::Io {
                detail: other.to_string(),
            },
        }
    }
}

#[derive(Debug)]
enum SidecarJob {
    StageChunk {
        id: ChunkId,
        bytes: Vec<u8>,
        reply: futures::channel::oneshot::Sender<Result<bool, SidecarError>>,
    },
    StageManifest {
        id: ManifestId,
        canonical: Vec<u8>,
        reply: futures::channel::oneshot::Sender<Result<bool, SidecarError>>,
    },
    Sync {
        reply: futures::channel::oneshot::Sender<Result<(), SidecarError>>,
    },
    HasChunk {
        id: ChunkId,
        reply: futures::channel::oneshot::Sender<bool>,
    },
    HasManifest {
        id: ManifestId,
        reply: futures::channel::oneshot::Sender<bool>,
    },
    IsDurableChunk {
        id: ChunkId,
        reply: futures::channel::oneshot::Sender<bool>,
    },
    IsDurableManifest {
        id: ManifestId,
        reply: futures::channel::oneshot::Sender<bool>,
    },
    ReadChunk {
        id: ChunkId,
        reply: futures::channel::oneshot::Sender<Result<Vec<u8>, SidecarError>>,
    },
    ReadManifest {
        id: ManifestId,
        reply: futures::channel::oneshot::Sender<Result<Vec<u8>, SidecarError>>,
    },
    MissingChunks {
        manifest: ManifestId,
        reply: futures::channel::oneshot::Sender<Result<Vec<ChunkId>, SidecarError>>,
    },
    CheckRoot {
        manifest: ManifestId,
        logical_len: u64,
        reply: futures::channel::oneshot::Sender<Result<bool, SidecarError>>,
    },
    Stats {
        reply: futures::channel::oneshot::Sender<kivi_chunk::ChunkStats>,
    },
}

/// Durable immutable sidecar store for one consensus node.
///
/// One blocking thread owns a [`kivi_chunk::ChunkStore`]; every other
/// context reaches it through one bounded channel. Verification reuses the
/// existing chunk primitives (`chunk_id`, `verify_manifest`,
/// `decode_manifest`); no second immutable subsystem exists.
#[derive(Debug, Clone)]
pub struct SidecarStore {
    sender: async_channel::Sender<SidecarJob>,
    pins: Arc<futures::lock::Mutex<SidecarPins>>,
    metrics: Arc<SidecarMetrics>,
    domain: SecurityDomainId,
}

impl SidecarStore {
    /// Queue depth for sidecar jobs.
    pub const QUEUE_DEPTH: usize = 128;

    /// Opens (or recovers) the sidecar store under `<data_dir>/sidecar`.
    ///
    /// # Errors
    ///
    /// Returns [`SidecarError::Io`] when the directory or packs cannot be
    /// opened or recovered.
    pub fn open(
        data_dir: &std::path::Path,
        domain: SecurityDomainId,
        pack_target_bytes: u64,
    ) -> Result<(Self, JoinHandle<()>), SidecarError> {
        let root = data_dir.join("sidecar");
        std::fs::create_dir_all(&root).map_err(|error| SidecarError::Io {
            detail: format!("sidecar dir {}: {error}", root.display()),
        })?;
        let (sender, receiver) = async_channel::bounded::<SidecarJob>(Self::QUEUE_DEPTH);
        let metrics = Arc::new(SidecarMetrics::new());
        let pins = Arc::new(futures::lock::Mutex::new(SidecarPins::new()));
        let worker_root = root;
        let handle = std::thread::Builder::new()
            .name("kivi-sidecar".to_owned())
            .spawn(move || {
                Self::serve(worker_root, domain, pack_target_bytes, receiver);
            })
            .map_err(|error| SidecarError::Io {
                detail: format!("sidecar thread spawn: {error}"),
            })?;
        Ok((
            Self {
                sender,
                pins,
                metrics,
                domain,
            },
            handle,
        ))
    }

    /// Opens an in-memory-ephemeral sidecar store for tests.
    ///
    /// # Panics
    ///
    /// Panics when the temp directory cannot be opened (test-only).
    #[cfg(test)]
    #[must_use]
    pub fn open_ephemeral(domain: SecurityDomainId) -> (Self, JoinHandle<()>) {
        // Process identity plus an atomic salt uniquely names the scratch
        // directory across parallel tests — no wall clock needed.
        static SALT: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "kivi-sidecar-test-{}-{}",
            std::process::id(),
            SALT.fetch_add(1, core::sync::atomic::Ordering::Relaxed)
        ));
        let _ = std::fs::create_dir_all(&dir);
        Self::open(&dir, domain, 64 * 1024 * 1024).expect("ephemeral sidecar opens")
    }

    /// Returns the security domain.
    #[must_use]
    pub const fn domain(&self) -> SecurityDomainId {
        self.domain
    }

    /// Returns shared metrics.
    #[must_use]
    pub fn metrics(&self) -> &Arc<SidecarMetrics> {
        &self.metrics
    }

    /// Returns shared pins.
    #[must_use]
    pub fn pins(&self) -> &Arc<futures::lock::Mutex<SidecarPins>> {
        &self.pins
    }

    /// Stages one verified chunk (no sync). Returns `true` when newly
    /// inserted, `false` on dedup hit. Verification rejects mismatched ids
    /// before anything is indexed.
    ///
    /// # Errors
    ///
    /// Returns [`SidecarError`] on corruption, overload, or shutdown.
    pub async fn stage_chunk(&self, id: ChunkId, bytes: Vec<u8>) -> Result<bool, SidecarError> {
        // Verify before queueing: never index a lie, never waste queue.
        let computed = kivi_codec::integrity::chunk_id(self.domain, &bytes);
        if computed != id {
            self.metrics
                .verification_failures
                .fetch_add(1, Ordering::Relaxed);
            return Err(SidecarError::Corrupt {
                detail: format!("chunk id mismatch: claimed {id}, computed {computed}"),
            });
        }
        if bytes.len() as u64 > kivi_chunk::policy::MAX_STORED_BODY {
            return Err(SidecarError::Corrupt {
                detail: format!("chunk {id} exceeds stored-body cap"),
            });
        }
        let (tx, rx) = futures::channel::oneshot::channel();
        self.sender
            .send(SidecarJob::StageChunk {
                id,
                bytes,
                reply: tx,
            })
            .await
            .map_err(|_| SidecarError::ShuttingDown)?;
        rx.await.map_err(|_| SidecarError::ShuttingDown)?
    }

    /// Stages one verified canonical manifest (no sync).
    ///
    /// # Errors
    ///
    /// Returns [`SidecarError`] on corruption, overload, or shutdown.
    pub async fn stage_manifest(
        &self,
        id: ManifestId,
        canonical: Vec<u8>,
    ) -> Result<bool, SidecarError> {
        // Structural verification happens on the worker (domain-checked
        // there); a cheap id pre-check here fails fast on lies.
        let computed = kivi_codec::integrity::manifest_id(self.domain, &canonical);
        if computed != id {
            self.metrics
                .verification_failures
                .fetch_add(1, Ordering::Relaxed);
            return Err(SidecarError::Corrupt {
                detail: format!("manifest id mismatch: claimed {id}, computed {computed}"),
            });
        }
        let (tx, rx) = futures::channel::oneshot::channel();
        self.sender
            .send(SidecarJob::StageManifest {
                id,
                canonical,
                reply: tx,
            })
            .await
            .map_err(|_| SidecarError::ShuttingDown)?;
        rx.await.map_err(|_| SidecarError::ShuttingDown)?
    }

    /// Durability barrier: syncs the pack tail once for the whole batch.
    ///
    /// # Errors
    ///
    /// Returns [`SidecarError`] on I/O failure or shutdown.
    pub async fn sync(&self) -> Result<(), SidecarError> {
        let (tx, rx) = futures::channel::oneshot::channel();
        self.sender
            .send(SidecarJob::Sync { reply: tx })
            .await
            .map_err(|_| SidecarError::ShuttingDown)?;
        rx.await.map_err(|_| SidecarError::ShuttingDown)?
    }

    /// Whether a chunk is indexed (any generation).
    ///
    /// # Errors
    ///
    /// Returns [`SidecarError::ShuttingDown`] when the worker is gone.
    pub async fn has_chunk(&self, id: ChunkId) -> Result<bool, SidecarError> {
        let (tx, rx) = futures::channel::oneshot::channel();
        self.sender
            .send(SidecarJob::HasChunk { id, reply: tx })
            .await
            .map_err(|_| SidecarError::ShuttingDown)?;
        rx.await.map_err(|_| SidecarError::ShuttingDown)
    }

    /// Whether a manifest is indexed (any generation).
    ///
    /// # Errors
    ///
    /// Returns [`SidecarError::ShuttingDown`] when the worker is gone.
    pub async fn has_manifest(&self, id: ManifestId) -> Result<bool, SidecarError> {
        let (tx, rx) = futures::channel::oneshot::channel();
        self.sender
            .send(SidecarJob::HasManifest { id, reply: tx })
            .await
            .map_err(|_| SidecarError::ShuttingDown)?;
        rx.await.map_err(|_| SidecarError::ShuttingDown)
    }

    /// Whether a chunk is durable (indexed at or below the synced gen).
    ///
    /// # Errors
    ///
    /// Returns [`SidecarError::ShuttingDown`] when the worker is gone.
    pub async fn is_durable_chunk(&self, id: ChunkId) -> Result<bool, SidecarError> {
        let (tx, rx) = futures::channel::oneshot::channel();
        self.sender
            .send(SidecarJob::IsDurableChunk { id, reply: tx })
            .await
            .map_err(|_| SidecarError::ShuttingDown)?;
        rx.await.map_err(|_| SidecarError::ShuttingDown)
    }

    /// Whether a manifest is durable.
    ///
    /// # Errors
    ///
    /// Returns [`SidecarError::ShuttingDown`] when the worker is gone.
    pub async fn is_durable_manifest(&self, id: ManifestId) -> Result<bool, SidecarError> {
        let (tx, rx) = futures::channel::oneshot::channel();
        self.sender
            .send(SidecarJob::IsDurableManifest { id, reply: tx })
            .await
            .map_err(|_| SidecarError::ShuttingDown)?;
        rx.await.map_err(|_| SidecarError::ShuttingDown)
    }

    /// Reads and re-verifies one chunk's bytes.
    ///
    /// # Errors
    ///
    /// Returns [`SidecarError`] when missing, corrupt, or shut down.
    pub async fn read_chunk(&self, id: ChunkId) -> Result<Vec<u8>, SidecarError> {
        let (tx, rx) = futures::channel::oneshot::channel();
        self.sender
            .send(SidecarJob::ReadChunk { id, reply: tx })
            .await
            .map_err(|_| SidecarError::ShuttingDown)?;
        rx.await.map_err(|_| SidecarError::ShuttingDown)?
    }

    /// Reads and re-verifies one canonical manifest.
    ///
    /// # Errors
    ///
    /// Returns [`SidecarError`] when missing, corrupt, or shut down.
    pub async fn read_manifest(&self, id: ManifestId) -> Result<Vec<u8>, SidecarError> {
        let (tx, rx) = futures::channel::oneshot::channel();
        self.sender
            .send(SidecarJob::ReadManifest { id, reply: tx })
            .await
            .map_err(|_| SidecarError::ShuttingDown)?;
        rx.await.map_err(|_| SidecarError::ShuttingDown)?
    }

    /// Lists chunks of a manifest missing locally (dedup: only these move).
    ///
    /// # Errors
    ///
    /// Returns [`SidecarError`] when the manifest is missing/corrupt or
    /// the worker is gone.
    pub async fn missing_chunks(&self, manifest: ManifestId) -> Result<Vec<ChunkId>, SidecarError> {
        let (tx, rx) = futures::channel::oneshot::channel();
        self.sender
            .send(SidecarJob::MissingChunks {
                manifest,
                reply: tx,
            })
            .await
            .map_err(|_| SidecarError::ShuttingDown)?;
        rx.await.map_err(|_| SidecarError::ShuttingDown)?
    }

    /// Verifies a root is fully durable and length-consistent.
    ///
    /// # Errors
    ///
    /// Returns [`SidecarError`] on store faults; `Ok(false)` means missing
    /// sidecars (caller must fetch), never a silent partial root.
    pub async fn check_root(
        &self,
        manifest: ManifestId,
        logical_len: u64,
    ) -> Result<bool, SidecarError> {
        let (tx, rx) = futures::channel::oneshot::channel();
        self.sender
            .send(SidecarJob::CheckRoot {
                manifest,
                logical_len,
                reply: tx,
            })
            .await
            .map_err(|_| SidecarError::ShuttingDown)?;
        rx.await.map_err(|_| SidecarError::ShuttingDown)?
    }

    /// Stages a full value deterministically: splits, hashes, stages
    /// chunks + manifest, syncs once. Returns the manifest id, canonical
    /// bytes, and chunk list. Untouched chunk ids reuse automatically via
    /// content addressing.
    ///
    /// # Errors
    ///
    /// Returns [`SidecarError`] on oversize values, I/O, or shutdown.
    pub async fn stage_value(&self, value: &[u8]) -> Result<StagedRoot, SidecarError> {
        let chunking = kivi_chunk::Chunking::DEFAULT;
        chunking.validate()?;
        let lengths = chunking
            .split_lengths(value.len() as u64)
            .map_err(SidecarError::from)?;
        if lengths.len() > kivi_chunk::MAX_CHUNKS_PER_MANIFEST {
            return Err(SidecarError::Corrupt {
                detail: format!(
                    "value needs {} chunks, cap is {}",
                    lengths.len(),
                    kivi_chunk::MAX_CHUNKS_PER_MANIFEST
                ),
            });
        }
        let mut chunks = Vec::with_capacity(lengths.len());
        let mut entries = Vec::with_capacity(lengths.len());
        let mut at = 0usize;
        for len in lengths {
            let len = usize::try_from(len).map_err(|_| SidecarError::Corrupt {
                detail: "chunk length overflow".to_owned(),
            })?;
            let piece = &value[at..at + len];
            let id = kivi_codec::integrity::chunk_id(self.domain, piece);
            self.stage_chunk(id, piece.to_vec()).await?;
            chunks.push(id);
            entries.push(kivi_chunk::ChunkEntry {
                id,
                len: len as u64,
            });
            at += len;
        }
        let (manifest_obj, computed) = kivi_chunk::build_manifest(
            self.domain,
            chunking,
            kivi_chunk::ChunkCodecId::NONE,
            value.len() as u64,
            entries,
        )?;
        let mut canonical = Vec::new();
        manifest_obj.encode_canonical(&mut canonical);
        self.stage_manifest(computed, canonical.clone()).await?;
        self.sync().await?;
        // Pin for the proposal lifetime.
        {
            let mut pins = self.pins.lock().await;
            pins.pin_root(computed, &chunks);
        }
        Ok(StagedRoot {
            manifest: computed,
            canonical,
            logical_len: value.len() as u64,
            chunks,
        })
    }

    /// Reads a full value by concatenating verified chunks.
    ///
    /// # Errors
    ///
    /// Returns [`SidecarError`] when any sidecar is missing/corrupt.
    pub async fn read_value(
        &self,
        manifest: ManifestId,
        logical_len: u64,
    ) -> Result<Vec<u8>, SidecarError> {
        if logical_len > kivi_chunk::policy::MAX_LEGACY_VALUE_BYTES {
            return Err(SidecarError::Corrupt {
                detail: format!("value {logical_len} exceeds legacy read cap"),
            });
        }
        let canonical = self.read_manifest(manifest).await?;
        let decoded = kivi_chunk::verify_manifest(manifest, &canonical).map_err(|error| {
            self.metrics
                .verification_failures
                .fetch_add(1, Ordering::Relaxed);
            SidecarError::from(error)
        })?;
        if decoded.total_len != logical_len {
            self.metrics
                .verification_failures
                .fetch_add(1, Ordering::Relaxed);
            return Err(SidecarError::Corrupt {
                detail: format!(
                    "manifest total {} != root logical {logical_len}",
                    decoded.total_len
                ),
            });
        }
        let mut out = Vec::with_capacity(usize::try_from(logical_len).unwrap_or(0));
        for entry in &decoded.entries {
            let bytes = self.read_chunk(entry.id).await?;
            if bytes.len() as u64 != entry.len {
                return Err(SidecarError::Corrupt {
                    detail: format!("chunk {} length mismatch", entry.id),
                });
            }
            out.extend_from_slice(&bytes);
        }
        if out.len() as u64 != logical_len {
            return Err(SidecarError::Corrupt {
                detail: "assembled length mismatch".to_owned(),
            });
        }
        Ok(out)
    }

    /// Acquires a proposal pin handle for an already-durable root.
    pub async fn pin_for_proposal(
        &self,
        manifest: ManifestId,
        chunks: Vec<ChunkId>,
    ) -> ProposalPin {
        {
            let mut pins = self.pins.lock().await;
            pins.pin_root(manifest, &chunks);
        }
        ProposalPin::new(Arc::clone(&self.pins), manifest, chunks)
    }

    /// Current chunk-store stats (diagnostics).
    ///
    /// # Errors
    ///
    /// Returns [`SidecarError::ShuttingDown`] when the worker is gone.
    pub async fn stats(&self) -> Result<kivi_chunk::ChunkStats, SidecarError> {
        let (tx, rx) = futures::channel::oneshot::channel();
        self.sender
            .send(SidecarJob::Stats { reply: tx })
            .await
            .map_err(|_| SidecarError::ShuttingDown)?;
        rx.await.map_err(|_| SidecarError::ShuttingDown)
    }

    #[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
    fn serve(
        root: PathBuf,
        domain: SecurityDomainId,
        pack_target_bytes: u64,
        receiver: async_channel::Receiver<SidecarJob>,
    ) {
        // Pack creation stamp only (diagnostic age ordering, never cut
        // semantics): one shared production read, failing closed to MAX.
        let wall = kivi_core::wall_now_or_max(&kivi_core::SystemClock);
        let _root_keep = root.clone();
        let mut store =
            match kivi_chunk::ChunkStore::open(&root, 0, domain, pack_target_bytes, wall) {
                Ok((store, _recovery)) => store,
                Err(error) => {
                    tracing::error!(%error, "sidecar store failed to open; draining to shutdown");
                    while receiver.recv_blocking().is_ok() {}
                    return;
                }
            };
        loop {
            let Ok(job) = receiver.recv_blocking() else {
                break;
            };
            match job {
                SidecarJob::StageChunk { id, bytes, reply } => {
                    let out = store
                        .stage_chunk(id, &bytes)
                        .map(|o| o.inserted)
                        .map_err(SidecarError::from);
                    let _ = reply.send(out);
                }
                SidecarJob::StageManifest {
                    id,
                    canonical,
                    reply,
                } => {
                    let out = store
                        .stage_manifest(id, &canonical)
                        .map(|o| o.inserted)
                        .map_err(SidecarError::from);
                    let _ = reply.send(out);
                }
                SidecarJob::Sync { reply } => {
                    let out = store.sync().map_err(SidecarError::from);
                    let _ = reply.send(out);
                }
                SidecarJob::HasChunk { id, reply } => {
                    let _ = reply.send(store.has_chunk(&id));
                }
                SidecarJob::HasManifest { id, reply } => {
                    let _ = reply.send(store.has_manifest(&id));
                }
                SidecarJob::IsDurableChunk { id, reply } => {
                    let durable = store.durable_chunk(id).is_some();
                    let _ = reply.send(durable);
                }
                SidecarJob::IsDurableManifest { id, reply } => {
                    let durable = store.durable_manifest(id).is_some();
                    let _ = reply.send(durable);
                }
                SidecarJob::ReadChunk { id, reply } => {
                    let out = store
                        .read_chunk(id)
                        .map(|b| b.to_vec())
                        .map_err(SidecarError::from);
                    let _ = reply.send(out);
                }
                SidecarJob::ReadManifest { id, reply } => {
                    let out = store
                        .read_manifest(id)
                        .map(|(_, canonical)| canonical)
                        .map_err(SidecarError::from);
                    let _ = reply.send(out);
                }
                SidecarJob::MissingChunks { manifest, reply } => {
                    // Discovery only needs the manifest indexed (staged but
                    // not yet synced is fine); durability is proven at the
                    // end via `CheckRoot` after the batch sync. Requiring
                    // durability here would fail every fresh fetch before
                    // its barrier.
                    let out = (|| -> Result<Vec<ChunkId>, SidecarError> {
                        let (decoded, _) =
                            store.read_manifest(manifest).map_err(SidecarError::from)?;
                        let mut missing = Vec::new();
                        for entry in &decoded.entries {
                            if store.durable_chunk(entry.id).is_none() {
                                missing.push(entry.id);
                            }
                        }
                        Ok(missing)
                    })();
                    let _ = reply.send(out);
                }
                SidecarJob::CheckRoot {
                    manifest,
                    logical_len,
                    reply,
                } => {
                    let out = (|| -> Result<bool, SidecarError> {
                        let Ok((decoded, _)) = store.read_manifest(manifest).map_err(|error| {
                            if matches!(error, kivi_chunk::ChunkError::MissingManifest { .. }) {
                                return SidecarError::Missing {
                                    detail: format!("manifest {manifest}"),
                                };
                            }
                            SidecarError::from(error)
                        }) else {
                            return Ok(false);
                        };
                        // Length + domain + durability agreement.
                        if decoded.total_len != logical_len || decoded.domain != domain {
                            return Ok(false);
                        }
                        if store.durable_manifest(manifest).is_none() {
                            return Ok(false);
                        }
                        for entry in &decoded.entries {
                            if store.durable_chunk(entry.id).is_none() {
                                return Ok(false);
                            }
                        }
                        Ok(true)
                    })();
                    // Normalize missing -> Ok(false) for gate convenience.
                    let out = match out {
                        Err(SidecarError::Missing { .. }) => Ok(false),
                        other => other,
                    };
                    let _ = reply.send(out);
                }
                SidecarJob::Stats { reply } => {
                    let _ = reply.send(store.stats());
                }
            }
        }
    }
}

/// One staged value proven durable, pinned for proposal.
#[derive(Debug, Clone)]
pub struct StagedRoot {
    /// Manifest addressing the value.
    pub manifest: ManifestId,
    /// Canonical manifest bytes.
    pub canonical: Vec<u8>,
    /// Total logical bytes.
    pub logical_len: u64,
    /// Chunk ids in order.
    pub chunks: Vec<ChunkId>,
}

/// Splits a value into deterministic chunk boundaries (pure, no I/O).
///
/// # Errors
///
/// Returns [`SidecarError`] when the chunking is invalid or oversize.
pub fn split_value(total_len: u64) -> Result<Vec<u64>, SidecarError> {
    kivi_chunk::Chunking::DEFAULT
        .split_lengths(total_len)
        .map_err(SidecarError::from)
}

/// Computes the chunk id for bytes under a domain (pure).
#[must_use]
pub fn chunk_id_for(domain: SecurityDomainId, bytes: &[u8]) -> ChunkId {
    kivi_codec::integrity::chunk_id(domain, bytes)
}

/// Builds a manifest for entries (pure, verified).
///
/// # Errors
///
/// Returns [`SidecarError`] when entries are invalid.
pub fn build_manifest_for(
    domain: SecurityDomainId,
    total_len: u64,
    entries: Vec<kivi_chunk::ChunkEntry>,
) -> Result<(ManifestId, Vec<u8>), SidecarError> {
    let (manifest_obj, id) = kivi_chunk::build_manifest(
        domain,
        kivi_chunk::Chunking::DEFAULT,
        kivi_chunk::ChunkCodecId::NONE,
        total_len,
        entries,
    )?;
    let mut canonical = Vec::new();
    manifest_obj.encode_canonical(&mut canonical);
    debug_assert_eq!(kivi_codec::integrity::manifest_id(domain, &canonical), id);
    Ok((id, canonical))
}

#[allow(clippy::missing_panics_doc)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dependencies_validate_manifest_nonzero() {
        let domain = SecurityDomainId::from_u64(1);
        let deps = ImmutableDependencies::new(ManifestId::from_bytes([7; 32]), 9, domain);
        assert!(deps.is_valid());
        assert!(!ImmutableDependencies::new(ManifestId::ZERO, 0, domain).is_valid());
    }

    #[test]
    fn pins_track_and_release_roots() {
        let mut pins = SidecarPins::new();
        assert!(pins.is_empty());
        let manifest = ManifestId::from_bytes([1; 32]);
        let chunks = vec![ChunkId::from_bytes([2; 32]), ChunkId::from_bytes([3; 32])];
        pins.pin_root(manifest, &chunks);
        assert_eq!(pins.manifest_count(), 1);
        assert_eq!(pins.chunk_count(), 2);
        let (m, c) = pins.snapshot();
        assert!(m.contains(&manifest));
        assert_eq!(c.len(), 2);
        pins.unpin_root(&manifest, &chunks);
        assert!(pins.is_empty());
    }

    #[test]
    fn chunk_id_is_domain_separated_and_stable() {
        let a = chunk_id_for(SecurityDomainId::from_u64(1), b"hello");
        let b = chunk_id_for(SecurityDomainId::from_u64(1), b"hello");
        let c = chunk_id_for(SecurityDomainId::from_u64(2), b"hello");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
