//! Embedded local engine: lifecycle, routing publication, and the clonable
//! client API.
//!
//! [`LocalEngine`] owns worker threads, the coherent routing snapshot, and
//! tablet placement. [`LocalClient`] is a cheap cloneable handle for ordinary
//! application threads: hash the key, load the routing snapshot (no mutex),
//! and hand the typed operation to the owning worker over a bounded queue.
//!
//! This channel-based path is the *local* API, not the future native network
//! fast path: later, client connections will live directly on data workers
//! and call tablet execution without this cross-thread hop. Shared tablet
//! logic keeps both paths identical; only the transport differs, and this
//! file documents the boundary so the wrong path is never "optimized" into
//! the fast one.

use core::fmt;
use std::sync::Arc;

use arc_swap::ArcSwap;
use bytes::Bytes;
use crossbeam_channel::{TrySendError, bounded};
use kivi_durability::{LaneIdentity, LocalWalLane, RecoverySummary};
use kivi_protocol::Capabilities;
use kivi_state::{
    Key, ObjectVersion, Operation, OperationResult, TxnId, TxnRecord, TxnState, TxnWrite,
    plan_transaction, txn_record_key,
};
use kivi_tablet::DirectorySnapshot;
use kivi_types::{ClusterId, NamespaceId, NodeId, NodeIncarnation, TabletId, UnixMicros, WorkerId};

use crate::clock::SystemClock;
use crate::routing::{Placement, RoutingError, RoutingSnapshot};
use crate::tablet::{LiveTablet, TabletError};
use crate::worker::{
    ControlIngress, RequestIngress, TabletRequest, WorkerControl, WorkerDurability, WorkerHandle,
    WorkerMetrics, WorkerRequestError,
};

/// Networked spawn outcome: handles, bundled ingress handles (bounded
/// sender plus bridge wakeup), and bound addresses.
type NetworkSpawn = (
    Vec<WorkerHandle>,
    Vec<RequestIngress>,
    Vec<std::net::SocketAddr>,
);

/// Merged recovery scan: per-tablet commit-ordered records, per-group
/// file-ordered consensus records, plus lane totals (segments scanned,
/// torn-tail bytes truncated).
type RecoveryScan = (
    std::collections::HashMap<TabletId, Vec<(u64, kivi_durability::WalRecord)>>,
    std::collections::HashMap<TabletId, Vec<(u64, kivi_durability::RaftRecord)>>,
    u64,
    u64,
);

/// Everything durable startup establishes before workers spawn.
struct DurableBootstrap {
    /// Per-worker durability (coordinator + lane thread).
    lanes: Vec<WorkerDurability>,
    /// Direct lane maintenance access per worker (checkpoint path).
    maintenance: Vec<(WorkerId, crate::commit::LaneMaintenance)>,
    /// WAL-tail recovery summary.
    wal_summary: RecoverySummary,
    /// Checkpoint-restore summary.
    checkpoint_summary: CheckpointRecovery,
    /// Installed CURRENT per checkpointed tablet.
    installed: std::collections::HashMap<TabletId, kivi_checkpoint::CurrentRecord>,
}

/// Checkpoint-thread spawn bundle, assembled after workers start (it
/// needs their control senders).
struct CheckpointSpawn {
    /// Lane maintenance access per worker.
    maintenance: Vec<(WorkerId, crate::commit::LaneMaintenance)>,
    /// Installed CURRENT per checkpointed tablet.
    installed: std::collections::HashMap<TabletId, kivi_checkpoint::CurrentRecord>,
    /// Trigger configuration.
    config: crate::checkpoint::CheckpointConfig,
    /// Data-directory root.
    data_dir: std::path::PathBuf,
    /// Namespace served.
    namespace: NamespaceId,
    /// Publisher identity base.
    cluster: ClusterId,
    /// Publisher identity base.
    node: NodeId,
    /// This process's incarnation.
    incarnation: NodeIncarnation,
    /// Writable tablets to checkpoint.
    tablets: Vec<TabletId>,
    /// Owner worker per tablet (authoritative placement for WAL attribution).
    placement: std::collections::HashMap<TabletId, WorkerId>,
    /// Whether all workers share WAL lane 0.
    shared_wal: bool,
    /// Engine-global in-flight staging pins for chunk GC roots.
    pins: Arc<crate::chunk_lane::StagingPins>,
    /// Chunk lane per worker for GC planning and reclamation.
    chunk_lanes: Vec<(WorkerId, crate::chunk_lane::ChunkLaneHandle)>,
}

/// Checkpoint-restored state staged before WAL-tail replay: cuts (tail
/// records at or below these are obsolete), durable lane floors,
/// installed currents, and the restore summary.
struct RestoredCheckpoints {
    /// Checkpoint cut per restored tablet.
    cuts: std::collections::HashMap<TabletId, u64>,
    /// Durable WAL floor per lane.
    floors: std::collections::HashMap<u16, kivi_durability::LaneFloor>,
    /// Installed CURRENT per restored tablet.
    currents: std::collections::HashMap<TabletId, kivi_checkpoint::CurrentRecord>,
    /// Restore accounting.
    summary: CheckpointRecovery,
}

/// Capacity of each client response rendezvous: exactly one outcome travels
/// per request, so one slot is both necessary and sufficient.
const RESPONSE_CAPACITY: usize = 1;

/// Engine construction parameters. Everything is validated before any worker
/// serves: an inconsistent directory/placement configuration fails here,
/// never mid-operation.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Namespace this engine serves (single-namespace stage).
    pub namespace: NamespaceId,
    /// Logical tablet directory (root alone, or preconfigured prefixes).
    pub directory: DirectorySnapshot,
    /// Owner worker per tablet id.
    pub placement: Placement,
    /// Worker thread count; every placed worker id must be below this.
    pub worker_count: usize,
    /// Per-worker bounded request queue depth; must be nonzero.
    pub request_capacity: usize,
    /// Chunk fabric policy and lane resources (both modes stage large
    /// values through per-worker chunk lanes).
    pub chunks: ChunkFabricConfig,
    /// Network serving. `None` keeps channel-only workers (embedded use,
    /// tests); `Some` binds one native endpoint per worker with CPU affinity.
    pub network: Option<crate::net::EngineNetwork>,
    /// Durability mode. `Ephemeral` keeps the pure in-memory behavior;
    /// `Durable` persists every mutating request through per-worker WAL
    /// lanes and replays them at startup before any listener binds.
    pub durability: DurabilityMode,
}

/// Durability mode: in-memory speed or crash-recoverable persistence.
/// There is no silent default — callers choose explicitly so a supposed
/// durable deployment can never run on the memory provider.
#[derive(Debug, Clone)]
pub enum DurabilityMode {
    /// Pure in-memory behavior: no WAL, no dedup, no recovery.
    Ephemeral,
    /// Crash-recoverable: per-worker WAL lanes plus startup replay.
    Durable(DurableConfig),
}

/// Chunk fabric configuration: representation policy plus lane resources.
/// One value shared by every worker; per-worker lanes open beneath it.
#[derive(Debug, Clone)]
pub struct ChunkFabricConfig {
    /// Values strictly larger stage as chunked roots; the rest stay
    /// inline. Physical only — logical `Bytes` semantics never change.
    pub inline_threshold: u64,
    /// Chunk pack rotation target in bytes per lane.
    pub pack_target_bytes: u64,
    /// Chunk-cache bound in bytes per lane (immutable entries, safe
    /// eviction at any time).
    pub cache_bytes: u64,
}

impl ChunkFabricConfig {
    /// Production default pack target: 256 MiB per lane, matching WAL
    /// segment scale so rotation stays rare outside tests.
    pub const DEFAULT_PACK_TARGET_BYTES: u64 = 256 * 1024 * 1024;
    /// Default representation threshold (see
    /// [`kivi_chunk::DEFAULT_INLINE_THRESHOLD`]).
    pub const DEFAULT_INLINE_THRESHOLD: u64 = kivi_chunk::DEFAULT_INLINE_THRESHOLD;
    /// Default per-lane chunk-cache bound (see
    /// [`crate::chunk_lane::DEFAULT_CHUNK_CACHE_BYTES`]).
    pub const DEFAULT_CACHE_BYTES: u64 = crate::chunk_lane::DEFAULT_CHUNK_CACHE_BYTES;
}

impl Default for ChunkFabricConfig {
    fn default() -> Self {
        Self {
            inline_threshold: kivi_chunk::DEFAULT_INLINE_THRESHOLD,
            pack_target_bytes: Self::DEFAULT_PACK_TARGET_BYTES,
            cache_bytes: crate::chunk_lane::DEFAULT_CHUNK_CACHE_BYTES,
        }
    }
}

/// Durable-mode configuration (data directory already opened by the
/// caller: lock held, identity established, incarnation advanced).
#[derive(Debug, Clone)]
pub struct DurableConfig {
    /// Data-directory root (WAL lives in `<data-dir>/wal`).
    pub data_dir: std::path::PathBuf,
    /// Segment rotation target in bytes.
    pub segment_target_bytes: u64,
    /// This node's identity (from the data directory, never flags).
    pub node: NodeId,
    /// This cluster's identity (from the data directory).
    pub cluster: ClusterId,
    /// This process's incarnation (stamped into new segments).
    pub incarnation: NodeIncarnation,
    /// Share one WAL lane across all workers behind a mutex (the
    /// group-commit experiment arm) instead of private per-worker lanes.
    pub shared_wal: bool,
    /// Batch-closing policy for the commit pipeline (`max_ops: 1`
    /// reproduces immediate per-mutation durability on the same code path).
    pub batch: crate::commit::BatchPolicy,
    /// Automatic checkpoint triggers and build policy.
    pub checkpoint: crate::checkpoint::CheckpointConfig,
}

/// Durability facts a running engine exposes to the admin plane.
#[derive(Debug, Clone)]
pub struct EngineDurability {
    /// Data-directory root.
    pub data_dir: std::path::PathBuf,
    /// This node's identity.
    pub node: NodeId,
    /// This cluster's identity.
    pub cluster: ClusterId,
    /// This process's incarnation.
    pub incarnation: NodeIncarnation,
    /// What recovery replayed at startup.
    pub recovery: kivi_durability::RecoverySummary,
    /// What checkpoint recovery restored at startup.
    pub checkpoint_recovery: CheckpointRecovery,
}

/// What checkpoint recovery restored at startup (spec AK, first half:
/// the WAL-tail half stays in [`kivi_durability::RecoverySummary`]).
#[derive(Debug, Clone, Default)]
pub struct CheckpointRecovery {
    /// Tablets restored from installed checkpoints.
    pub tablets_loaded: u64,
    /// Bands verified (identity, CRCs, ordering).
    pub bands_verified: u64,
    /// Checkpoint bytes restored (bands + dedup).
    pub bytes_restored: u64,
    /// WAL records skipped as checkpoint-covered (obsolete, not replayed).
    pub obsolete_skipped: u64,
    /// Tablets that fell back to the retained previous checkpoint.
    pub fell_back: u64,
}

/// Renders the optional panic-payload suffix for
/// [`EngineError::WorkerPanicked`].
fn panic_suffix(message: Option<&String>) -> String {
    message.map_or_else(String::new, |detail| format!(": {detail}"))
}

/// Engine failure modes. Reachability-style outcomes stay in-band as
/// operation results; only misuse, exhaustion, overload, and worker loss
/// fail here.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum EngineError {
    /// The owning worker's bounded queue is full (fast rejection; the caller
    /// decides whether to retry with backoff).
    #[error("worker queue full")]
    Overloaded,
    /// The owning worker is gone (exited or panicked before shutdown).
    #[error("worker {worker} is down")]
    WorkerDown {
        /// Affected worker.
        worker: WorkerId,
    },
    /// A worker thread panicked; the message is captured, never swallowed.
    #[error("worker {worker} panicked{suffix}", suffix = panic_suffix(message.as_ref()))]
    WorkerPanicked {
        /// Affected worker.
        worker: WorkerId,
        /// Panic payload when it carried a readable message.
        message: Option<String>,
    },
    /// No active tablet covers the routed hash.
    #[error("no active tablet covers the key")]
    NoRoute,
    /// Routing resolved to a tablet the worker does not own.
    #[error("unknown tablet {tablet}")]
    UnknownTablet {
        /// Requested tablet.
        tablet: TabletId,
    },
    /// Routing resolved to a worker outside the engine.
    #[error("unknown worker {worker}")]
    UnknownWorker {
        /// Out-of-range worker.
        worker: WorkerId,
    },
    /// The partition hash algorithm is not implemented by this build.
    #[error("partition hash algorithm unsupported")]
    HashAlgorithmUnsupported,
    /// Tablet logic rejected or failed the operation.
    #[error("{0}")]
    Tablet(#[from] TabletError),
    /// A networked worker failed to start (affinity, runtime, or bind).
    #[error("networked worker failed to start: {0}")]
    NetStart(#[from] crate::net::NetStartError),
    /// Routing construction or replacement was incoherent.
    #[error("routing: {0}")]
    Routing(#[from] RoutingError),
    /// Engine configuration itself is invalid (zero workers, zero capacity).
    #[error("invalid engine configuration: {reason}")]
    InvalidConfig {
        /// What was wrong.
        reason: String,
    },
    /// Durable setup or recovery failed; the engine never started serving.
    #[error("durability failed: {0}")]
    Durability(#[from] kivi_durability::DurabilityError),
    /// WAL recovery refused to open the database (see the variant for why;
    /// history is never silently truncated).
    #[error("recovery failed: {0}")]
    Recovery(#[from] kivi_durability::RecoveryError),
    /// Checkpoint recovery refused to open the database (corrupt catalog
    /// or unloadable chains; history is never silently truncated).
    #[error("checkpoint recovery failed: {0}")]
    Checkpoint(#[from] kivi_checkpoint::CheckpointError),
    /// Chunk fabric failure: missing or corrupt immutable data a
    /// committed root needs, or a lane I/O failure. Missing/corrupt
    /// chunk data fails loudly here so later multi-source recovery can
    /// repair it, instead of serving wrong bytes.
    #[error("chunk fabric failed: {0}")]
    Chunk(#[from] kivi_chunk::ChunkError),
    /// A retried identity fell below the session floor.
    #[error("mutation identity expired below the session floor")]
    DedupExpired,
    /// The session holds too many unacknowledged outcomes on the tablet.
    #[error("session outcome window exhausted")]
    SessionOverloaded,
    /// The request is not admittable (absurd range, oversize patch):
    /// caller bug, state untouched — fix and retry.
    #[error("invalid request: {detail}")]
    InvalidRequest {
        /// Human-readable reason (returned verbatim).
        detail: String,
    },
}

impl From<WorkerRequestError> for EngineError {
    /// Maps worker-side request failures onto engine errors.
    fn from(source: WorkerRequestError) -> Self {
        match source {
            WorkerRequestError::UnknownTablet { tablet } => Self::UnknownTablet { tablet },
            WorkerRequestError::Tablet(error) => Self::Tablet(error),
            WorkerRequestError::DedupExpired => Self::DedupExpired,
            WorkerRequestError::SessionOverloaded => Self::SessionOverloaded,
            WorkerRequestError::Storage(error) => Self::Durability(error),
            WorkerRequestError::ChunkStore(error) => Self::Chunk(error),
            WorkerRequestError::InvalidRequest { detail } => Self::InvalidRequest { detail },
        }
    }
}

/// State shared by the engine handle and every cloned client: routing
/// publication plus one request sender per worker. The routing `Arc` is the
/// same object handed to networked workers, so publication swaps are visible
/// coherently on every path at once.
#[derive(Debug)]
struct EngineShared {
    namespace: NamespaceId,
    routing: Arc<arc_swap::ArcSwap<RoutingSnapshot>>,
    senders: Vec<RequestIngress>,
    /// Chunk lane per worker index, for caller-side chunked-read
    /// resolution on the embedded path (blocking, caller thread).
    chunk_lanes: Vec<crate::chunk_lane::ChunkLaneHandle>,
}

/// The running engine: worker threads, routing, placement, lifecycle.
pub struct LocalEngine {
    shared: Arc<EngineShared>,
    workers: Vec<WorkerHandle>,
    bound: Vec<std::net::SocketAddr>,
    durability: Option<EngineDurability>,
    checkpoint: Option<crate::checkpoint::CheckpointWorkerHandle>,
    checkpoint_admin: Arc<std::sync::Mutex<crate::checkpoint::CheckpointAdminState>>,
    /// Chunk lane handles (stats, shutdown signaling) plus join guards
    /// (shutdown after workers join: no worker submits once it exits).
    chunk_lanes: Vec<crate::chunk_lane::ChunkLaneHandle>,
    chunk_guards: Vec<crate::chunk_lane::ChunkLaneGuard>,
    /// Ephemeral-mode chunk root: per-process `TempDir` auto-removed on
    /// clean shutdown (`None` in durable mode, which stages under the
    /// data directory). Held for its `Drop`, never otherwise touched.
    _ephemeral_chunks: Option<tempfile::TempDir>,
}

/// Read-only admin/query handle to a running engine.
///
/// Clonable and `Send + Sync`: the Tokio admin plane holds this while worker
/// threads own everything mutable. Reads go through the immutable routing
/// publication or the bounded control channel — never locks, never the data
/// path, never Tokio inside the engine.
#[derive(Debug, Clone)]
pub struct AdminHandle {
    namespace: NamespaceId,
    routing: Arc<ArcSwap<RoutingSnapshot>>,
    controls: Vec<(WorkerId, ControlIngress)>,
    endpoints: Vec<(WorkerId, std::net::SocketAddr)>,
    durability: Option<EngineDurability>,
    checkpoint: Arc<std::sync::Mutex<crate::checkpoint::CheckpointAdminState>>,
    /// Chunk lane per worker id, for read-only lane stats snapshots.
    chunk_lanes: Vec<(WorkerId, crate::chunk_lane::ChunkLaneHandle)>,
}

impl AdminHandle {
    /// Returns the namespace this engine serves.
    #[must_use]
    pub const fn namespace(&self) -> NamespaceId {
        self.namespace
    }

    /// Returns the worker count.
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.controls.len()
    }

    /// Loads the current coherent routing publication.
    #[must_use]
    pub fn routing(&self) -> Arc<RoutingSnapshot> {
        Arc::clone(&self.routing.load())
    }

    /// Returns bound native endpoints per worker (empty for channel-only
    /// engines, which expose no sockets).
    #[must_use]
    pub fn endpoints(&self) -> &[(WorkerId, std::net::SocketAddr)] {
        &self.endpoints
    }

    /// Snapshots per-worker execution counters over the control channel.
    ///
    /// This blocks on a rendezvous with each worker thread: call it from
    /// `spawn_blocking`, never from an async handler directly. Exited
    /// workers are absent from the result.
    #[must_use]
    pub fn worker_metrics_snapshot(&self) -> Vec<(WorkerId, WorkerMetrics)> {
        let mut out = Vec::new();
        for (id, control) in &self.controls {
            let (respond, receive) = crossbeam_channel::bounded(1);
            if control
                .try_send(WorkerControl::Metrics { respond })
                .is_err()
            {
                continue;
            }
            if let Ok(metrics) = receive.recv() {
                out.push((*id, metrics));
            }
        }
        out
    }

    /// Returns the durability facts established at startup (`None` in
    /// ephemeral mode, which has no data directory).
    #[must_use]
    pub fn durability(&self) -> Option<&EngineDurability> {
        self.durability.as_ref()
    }

    /// Snapshots every worker's WAL lane (`None` per worker in ephemeral
    /// mode). Blocking control rendezvous: serve from `spawn_blocking`.
    #[must_use]
    pub fn lane_stats_snapshot(&self) -> Vec<(WorkerId, Option<kivi_durability::WorkerLaneStats>)> {
        let mut out = Vec::new();
        for (id, control) in &self.controls {
            let (respond, receive) = crossbeam_channel::bounded(1);
            if control
                .try_send(WorkerControl::DurabilityStats { respond })
                .is_err()
            {
                continue;
            }
            if let Ok(stats) = receive.recv() {
                out.push((*id, stats));
            }
        }
        out
    }

    /// Snapshots every worker's commit-pipeline metrics (`None` per worker
    /// in ephemeral mode, which has no pipeline). Blocking control
    /// rendezvous: serve from `spawn_blocking`.
    #[must_use]
    pub fn commit_metrics_snapshot(
        &self,
    ) -> Vec<(WorkerId, Option<crate::commit::CommitMetricsSnapshot>)> {
        let mut out = Vec::new();
        for (id, control) in &self.controls {
            let (respond, receive) = crossbeam_channel::bounded(1);
            if control
                .try_send(WorkerControl::CommitMetrics { respond })
                .is_err()
            {
                continue;
            }
            if let Ok(metrics) = receive.recv() {
                out.push((*id, metrics));
            }
        }
        out
    }

    /// Clones the live checkpoint state for the admin plane (brief status
    /// lock; empty in ephemeral mode, which never checkpoints).
    #[must_use]
    pub fn checkpoint_state(&self) -> crate::checkpoint::CheckpointAdminState {
        self.checkpoint
            .lock()
            .map(|state| state.clone())
            .unwrap_or_default()
    }

    /// Snapshots every worker's chunk lane (blocking lane round-trip per
    /// worker; serve from `spawn_blocking` like every other lane query).
    /// Exited lanes are absent from the result.
    #[must_use]
    pub fn chunk_stats_snapshot(
        &self,
    ) -> Vec<(
        WorkerId,
        Result<crate::chunk_lane::ChunkLaneStats, kivi_chunk::ChunkError>,
    )> {
        self.chunk_lanes
            .iter()
            .map(|(id, lane)| (*id, lane.stats_blocking()))
            .collect()
    }
}

impl LocalEngine {
    /// Starts the engine: validates configuration, builds one live tablet
    /// per active directory tablet on its placed worker, spawns worker
    /// threads, and publishes the initial routing snapshot.
    ///
    /// Fresh tablets start at initial fencing generations (this stage has no
    /// persisted fencing state to recover; future phases will carry fencing
    /// generations in directory metadata instead of defaulting them here).
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] for invalid configuration, incoherent
    /// routing, or tablet construction failures — always before serving.
    // One linear bring-up (validate → tablets → durability → workers →
    // listeners → checkpoint); the order is the correctness argument and
    // splitting it would scatter the startup sequence.
    #[allow(clippy::too_many_lines)]
    pub fn start(config: EngineConfig) -> Result<Self, EngineError> {
        if config.worker_count == 0 {
            return Err(EngineError::InvalidConfig {
                reason: "worker_count must be nonzero".to_owned(),
            });
        }
        if config.request_capacity == 0 {
            return Err(EngineError::InvalidConfig {
                reason: "request_capacity must be nonzero".to_owned(),
            });
        }
        let routing = RoutingSnapshot::build(
            config.namespace,
            config.directory,
            config.placement,
            config.worker_count,
        )?;
        let mut by_worker: Vec<Vec<LiveTablet>> = Vec::new();
        by_worker.resize_with(config.worker_count, Vec::new);
        for tablet in routing.directory().tablets() {
            if !tablet.state().is_writable() {
                continue;
            }
            let worker = routing
                .placement()
                .worker_of(tablet.id())
                .ok_or(EngineError::NoRoute)?;
            let authority = kivi_types::TabletAuthority::new(
                tablet.id(),
                kivi_types::TabletEpoch::INITIAL,
                kivi_types::WriteGuardGeneration::INITIAL,
            );
            let mut live = LiveTablet::from_descriptor(tablet, authority)?;
            // Ordered tablets maintain the key index from birth (empty
            // and therefore trivially exact); hash tablets skip it.
            if matches!(tablet.range(), kivi_tablet::PartitionRange::Ordered(_)) {
                live.set_ordered_indexing(true);
            }
            let index = usize::try_from(worker.as_u64())
                .map_err(|_| EngineError::UnknownWorker { worker })?;
            by_worker[index].push(live);
        }
        // Chunk lanes open before any recovery: pack recovery rebuilds
        // each lane's derived index, and checkpoint/WAL validation below
        // proves every restored or replayed chunked root against those
        // indexes before serving. Ephemeral mode stages under a
        // per-process TempDir (removed on clean shutdown); durable mode
        // stages under the data directory. One lane per worker either way.
        // Lane threads spawn here too (stores are recovered first):
        // workers submit from their first request, and the checkpoint
        // bundle below needs lane handles for GC planning.
        let chunk_pins = Arc::new(crate::chunk_lane::StagingPins::default());
        let chunk_domain = kivi_types::SecurityDomainId::from_u64(config.namespace.as_u64());
        let chunk_wall_micros = SystemClock::wall_now().as_micros();
        let mut ephemeral_chunks: Option<tempfile::TempDir> = None;
        let chunk_root: std::path::PathBuf = match &config.durability {
            DurabilityMode::Ephemeral => {
                let dir = tempfile::TempDir::new().map_err(|error| {
                    EngineError::Chunk(kivi_chunk::ChunkError::io(
                        "create ephemeral chunk dir",
                        &std::env::temp_dir(),
                        &error,
                    ))
                })?;
                let path = dir.path().to_owned();
                ephemeral_chunks = Some(dir);
                path
            }
            DurabilityMode::Durable(cfg) => cfg.data_dir.clone(),
        };
        let mut chunk_stores = Vec::with_capacity(config.worker_count);
        for index in 0..config.worker_count {
            let lane = u16::try_from(index).map_err(|_| EngineError::InvalidConfig {
                reason: "worker count exceeds chunk lane space".to_owned(),
            })?;
            let (store, summary) = kivi_chunk::ChunkStore::open(
                &chunk_root,
                lane,
                chunk_domain,
                config.chunks.pack_target_bytes,
                chunk_wall_micros,
            )?;
            tracing::info!(
                worker = index,
                packs = summary.packs_scanned,
                chunks = summary.chunks_indexed,
                manifests = summary.manifests_indexed,
                truncated_bytes = summary.truncated_bytes,
                "chunk lane recovered"
            );
            chunk_stores.push(store);
        }
        // Each lane moves into its thread before WAL recovery: recovery
        // only borrows the stores for validation, while workers (spawned
        // below) and the checkpoint bundle (assembled in the match) need
        // live handles. Handles clone cheaply into workers, shared
        // state, the admin plane, and GC planning.
        let mut chunk_guards = Vec::with_capacity(config.worker_count);
        let mut chunk_access = Vec::with_capacity(config.worker_count);
        let mut chunk_lanes = Vec::with_capacity(config.worker_count);
        for (index, store) in chunk_stores.into_iter().enumerate() {
            let worker = WorkerId::from_u64(index as u64);
            let (handle, guard) =
                crate::chunk_lane::spawn_lane(worker, store, config.chunks.cache_bytes);
            chunk_access.push(crate::worker::WorkerChunks {
                lane: handle.clone(),
                inline_threshold: config.chunks.inline_threshold,
                domain: chunk_domain,
                pins: Arc::clone(&chunk_pins),
            });
            chunk_lanes.push(handle);
            chunk_guards.push(guard);
        }
        // Durable mode recovers BEFORE any worker spawns (and therefore
        // before any listener binds): checkpoints restore first, then the
        // WAL tail replays after the cuts. A corrupt database never serves
        // partial state, and replayed tablets are simply the workers'
        // initial state. (Chunked-root dependency validation runs after
        // this match over final live state — see the dependency gate
        // below — covering restore and replay uniformly.)
        let (lanes, durability, checkpoint_spawn): (
            Vec<Option<WorkerDurability>>,
            Option<EngineDurability>,
            Option<CheckpointSpawn>,
        ) = match &config.durability {
            DurabilityMode::Ephemeral => {
                ((0..config.worker_count).map(|_| None).collect(), None, None)
            }
            DurabilityMode::Durable(cfg) => {
                let DurableBootstrap {
                    lanes,
                    maintenance,
                    wal_summary,
                    checkpoint_summary,
                    installed,
                } = Self::recover_durable(
                    cfg,
                    &mut by_worker,
                    &routing,
                    config.namespace,
                    config.worker_count,
                )?;
                let info = EngineDurability {
                    data_dir: cfg.data_dir.clone(),
                    node: cfg.node,
                    cluster: cfg.cluster,
                    incarnation: cfg.incarnation,
                    recovery: wal_summary,
                    checkpoint_recovery: checkpoint_summary,
                };
                let tablets = by_worker
                    .iter()
                    .flatten()
                    .map(super::tablet::LiveTablet::id)
                    .collect::<Vec<_>>();
                let placement = tablets
                    .iter()
                    .filter_map(|tablet| {
                        routing
                            .placement()
                            .worker_of(*tablet)
                            .map(|worker| (*tablet, worker))
                    })
                    .collect::<std::collections::HashMap<_, _>>();
                let spawn = CheckpointSpawn {
                    maintenance,
                    installed,
                    config: cfg.checkpoint.clone(),
                    data_dir: cfg.data_dir.clone(),
                    namespace: config.namespace,
                    cluster: cfg.cluster,
                    node: cfg.node,
                    incarnation: cfg.incarnation,
                    tablets,
                    placement,
                    shared_wal: cfg.shared_wal,
                    pins: Arc::clone(&chunk_pins),
                    chunk_lanes: chunk_access
                        .iter()
                        .enumerate()
                        .map(|(index, access)| {
                            (WorkerId::from_u64(index as u64), access.lane.clone())
                        })
                        .collect(),
                };
                (
                    lanes.into_iter().map(Some).collect(),
                    Some(info),
                    Some(spawn),
                )
            }
        };
        // Dependency gate (§30, §31): every live chunked root — restored
        // from checkpoints, replayed from the WAL tail, or fresh — proves
        // its manifest and chunks in its owner's lane before serving. A
        // committed root without its immutable data fails startup loudly,
        // never serves absence. Obsolete WAL records (at or below a cut)
        // need no proof: they never replay.
        for (index, lives) in by_worker.iter().enumerate() {
            for live in lives {
                for (_, object) in live.store().snapshot_entries() {
                    if let Some(chunked) = object.chunk_ref() {
                        chunk_access[index]
                            .lane
                            .check_root_blocking(chunked.manifest, chunked.logical_len)?;
                    }
                }
            }
        }
        let routing = Arc::new(ArcSwap::from_pointee(routing));
        let (workers, senders, bound) = match &config.network {
            None => {
                let (workers, senders) = Self::spawn_channel_workers(
                    config.request_capacity,
                    config.worker_count,
                    by_worker,
                    lanes,
                    chunk_access,
                );
                (workers, senders, Vec::new())
            }
            Some(network) => {
                let (workers, senders, bound) = Self::spawn_network_workers(
                    network,
                    config.request_capacity,
                    config.worker_count,
                    by_worker,
                    &routing,
                    lanes,
                    chunk_access,
                )?;
                (workers, senders, bound)
            }
        };
        // The checkpoint worker starts last (it needs worker control
        // senders) and stops first (before workers flush). `disabled`
        // only stills the automatic triggers (benchmarks isolate the
        // commit pipeline); the thread itself always runs in durable
        // mode because manual triggers need a running worker to serve
        // (and `due_tablets` has an explicit disabled branch for this).
        let checkpoint_admin = Arc::new(std::sync::Mutex::new(
            crate::checkpoint::CheckpointAdminState::default(),
        ));
        let checkpoint = checkpoint_spawn.map(|spawn| {
            let context = crate::checkpoint::CheckpointContext {
                data_dir: spawn.data_dir,
                namespace: spawn.namespace,
                cluster: spawn.cluster,
                node: spawn.node,
                incarnation: spawn.incarnation,
                tablets: spawn.tablets,
                placement: spawn.placement,
                controls: workers
                    .iter()
                    .map(|worker| (worker.id(), worker.control_sender()))
                    .collect(),
                lanes: spawn
                    .maintenance
                    .into_iter()
                    .map(|(worker, maintenance)| {
                        // Exclusive mode numbers lanes by worker; the
                        // shared arm funnels every worker into lane 0
                        // (mirrors recover_durable's lane assignment).
                        let lane = if spawn.shared_wal {
                            0
                        } else {
                            u16::try_from(worker.as_u64()).unwrap_or(u16::MAX)
                        };
                        crate::checkpoint::LaneMaintenanceAccess {
                            worker,
                            lane,
                            maintenance,
                        }
                    })
                    .collect(),
                config: spawn.config,
                installed: spawn.installed,
                pins: spawn.pins,
                chunk_lanes: spawn.chunk_lanes,
                admin: Arc::clone(&checkpoint_admin),
            };
            crate::checkpoint::CheckpointWorkerHandle::spawn(context)
        });
        Ok(Self {
            shared: Arc::new(EngineShared {
                namespace: config.namespace,
                routing,
                senders,
                chunk_lanes: chunk_lanes.clone(),
            }),
            workers,
            bound,
            durability,
            checkpoint,
            checkpoint_admin,
            chunk_lanes,
            chunk_guards,
            _ephemeral_chunks: ephemeral_chunks,
        })
    }

    /// Opens every WAL lane, restores installed checkpoints, repairs torn
    /// tails from the durable floor, and replays only the WAL tail after
    /// each tablet's checkpoint cut. Returns lane ownership, both recovery
    /// summaries, installed currents, and lane maintenance access.
    ///
    /// Fails (refusing to serve) on corrupt history, structural gaps,
    /// records for tablets the startup directory does not cover, fencing
    /// mismatches, commit-chain breaks, replay divergence, or unloadable
    /// checkpoint chains.
    fn recover_durable(
        cfg: &DurableConfig,
        by_worker: &mut [Vec<LiveTablet>],
        routing: &RoutingSnapshot,
        namespace: NamespaceId,
        worker_count: usize,
    ) -> Result<DurableBootstrap, EngineError> {
        let identity = LaneIdentity {
            cluster: cfg.cluster,
            node: cfg.node,
            incarnation: cfg.incarnation,
        };
        let wal_dir = cfg.data_dir.join(kivi_durability::node::WAL_DIR_NAME);
        // One lane per worker, or a single shared lane for the
        // group-commit experiment arm.
        let lane_indexes: Vec<u16> = if cfg.shared_wal {
            vec![0; worker_count]
        } else {
            (0..worker_count)
                .map(|index| {
                    u16::try_from(index).map_err(|_| EngineError::InvalidConfig {
                        reason: "worker count exceeds WAL lane space".to_owned(),
                    })
                })
                .collect::<Result<_, _>>()?
        };
        // Checkpoints restore first: installed state plus per-tablet cuts
        // plus the durable WAL floors. The WAL tail replays after the cuts.
        let restored = Self::restore_checkpoints(cfg, by_worker, routing, namespace)?;
        let mut lanes =
            Self::open_recovery_lanes(&wal_dir, &lane_indexes, identity, cfg.segment_target_bytes)?;
        // Merge records per tablet across lanes (a future tablet may hold
        // history in several lanes; never assume one lane per tablet),
        // resuming each lane at its durable floor.
        let (per_tablet, consensus, segments_scanned, tail_truncated_bytes) =
            Self::scan_recovery_lanes(&mut lanes, &restored.floors)?;
        // Validate against the startup directory, skip checkpoint-covered
        // records, then replay the tail in commit order. Records for
        // forgotten tablets fail startup: discarding their history would
        // be silent data loss.
        let (records_replayed, tablets_recovered, obsolete_skipped) = Self::replay_records(
            by_worker,
            routing,
            namespace,
            per_tablet,
            &consensus,
            &restored.cuts,
        )?;
        let wal_summary = RecoverySummary {
            records_replayed,
            segments_scanned,
            tail_truncated_bytes,
            tablets_recovered,
        };
        tracing::info!(
            records = wal_summary.records_replayed,
            segments = wal_summary.segments_scanned,
            truncated_bytes = wal_summary.tail_truncated_bytes,
            tablets = wal_summary.tablets_recovered,
            skipped = obsolete_skipped,
            checkpoints = restored.summary.tablets_loaded,
            "durable recovery complete",
        );
        // Hand lanes to workers: exclusive per worker, or one shared lane.
        // Each lane moves into its worker's coordinator lane thread here;
        // the initial stats snapshot seeds the admin plane before the
        // first seal reports back.
        let shared = if cfg.shared_wal {
            let lane = lanes.remove(&0).expect("shared lane open");
            let stats = lane.lane_stats();
            Some((std::sync::Arc::new(std::sync::Mutex::new(lane)), stats))
        } else {
            None
        };
        let mut out = Vec::with_capacity(worker_count);
        let mut maintenance = Vec::with_capacity(worker_count);
        for (index, lane_index) in lane_indexes.into_iter().enumerate() {
            let worker = WorkerId::from_u64(index as u64);
            let (access, initial) = if let Some((shared, stats)) = &shared {
                (
                    crate::worker::LaneAccess::Shared(Arc::clone(shared)),
                    *stats,
                )
            } else {
                let lane = lanes.remove(&lane_index).expect("lane open");
                let stats = lane.lane_stats();
                (crate::worker::LaneAccess::Exclusive(lane), stats)
            };
            // Fresh, replayed, and restored tablets alike start dirty
            // tracking from a clean slate: restore and replay ran before
            // tracking existed, and the next checkpoint's cut covers
            // everything restored, so no retroactive dirt is owed.
            for live in &mut by_worker[index] {
                live.enable_band_tracking(namespace);
            }
            let durable = WorkerDurability::spawn(namespace, worker, cfg.batch, access, initial)?;
            maintenance.push((worker, durable.maintenance()));
            out.push(durable);
        }
        Ok(DurableBootstrap {
            lanes: out,
            maintenance,
            wal_summary,
            checkpoint_summary: CheckpointRecovery {
                obsolete_skipped,
                ..restored.summary
            },
            installed: restored.currents,
        })
    }

    /// Restores installed checkpoints into the freshly built tablets:
    /// loads each tablet's CURRENT chain, validates fencing against the
    /// startup directory, installs objects/dedup/cuts, and reads the
    /// durable WAL floors. Returns cuts, floors, currents, and the
    /// restore summary.
    fn restore_checkpoints(
        cfg: &DurableConfig,
        by_worker: &mut [Vec<LiveTablet>],
        routing: &RoutingSnapshot,
        namespace: NamespaceId,
    ) -> Result<RestoredCheckpoints, EngineError> {
        use std::collections::HashMap;
        let mut cuts: HashMap<TabletId, u64> = HashMap::new();
        let mut currents: HashMap<TabletId, kivi_checkpoint::CurrentRecord> = HashMap::new();
        let mut summary = CheckpointRecovery::default();
        for live in by_worker.iter_mut().flatten() {
            let tablet = live.id();
            let Some(installed) = kivi_checkpoint::load_installed(&cfg.data_dir, tablet)? else {
                continue;
            };
            // Fencing: the checkpoint's authority must match the startup
            // directory. A stale checkpoint (old epoch/guard) is skipped
            // here — the WAL tail then fails loudly on the same fencing
            // mismatch instead of serving cross-authority state.
            let descriptor = routing.directory().get(tablet).ok_or_else(|| {
                kivi_durability::RecoveryError::UnknownTablet {
                    tablet: tablet.as_u64(),
                }
            })?;
            if installed.manifest.epoch != descriptor.epoch()
                || installed.manifest.guard != descriptor.guard()
            {
                tracing::warn!(
                    tablet = tablet.as_u64(),
                    "installed checkpoint names a stale authority; skipping to WAL replay"
                );
                continue;
            }
            if installed.manifest.namespace != namespace {
                return Err(kivi_durability::RecoveryError::NamespaceMismatch {
                    tablet: tablet.as_u64(),
                    expected: namespace.as_u64(),
                    found: installed.manifest.namespace.as_u64(),
                }
                .into());
            }
            let restored = crate::tablet::RestoredTablet {
                objects: installed
                    .bands
                    .iter()
                    .flat_map(|band| {
                        band.records
                            .iter()
                            .map(|record| (record.key.clone(), record.object.clone()))
                    })
                    .collect(),
                sessions: installed
                    .dedup
                    .sessions
                    .iter()
                    .map(|session| {
                        (
                            session.session,
                            session.floor,
                            session
                                .outcomes
                                .iter()
                                .map(|outcome| {
                                    (
                                        outcome.seq,
                                        crate::tablet::DedupEntry {
                                            commit: outcome.commit,
                                            opcode: outcome.opcode,
                                            outcome: outcome.outcome.clone(),
                                        },
                                    )
                                })
                                .collect(),
                        )
                    })
                    .collect(),
                // Unresolved intents restore verbatim: restart never drops
                // a prepared reservation (the resolver finishes it).
                intents: installed.dedup.intents.clone(),
                cut: kivi_types::CommitPosition::from_u64(installed.manifest.cut),
            };
            let bytes_restored: u64 = installed
                .manifest
                .bands
                .iter()
                .map(|descriptor| descriptor.len)
                .sum::<u64>()
                + installed.manifest.dedup.len;
            live.restore_checkpoint(restored);
            // Ordered tablets rebuild their key index deterministically
            // from restored objects (exact: every stored key is indexed).
            // Hash tablets keep the fast path (no ordered structure).
            if matches!(descriptor.range(), kivi_tablet::PartitionRange::Ordered(_)) {
                live.set_ordered_indexing(true);
            }
            cuts.insert(tablet, installed.manifest.cut);
            currents.insert(tablet, installed.current.clone());
            summary.tablets_loaded += 1;
            summary.bands_verified += installed.manifest.bands.len() as u64;
            summary.bytes_restored += bytes_restored;
            if installed.fell_back {
                summary.fell_back += 1;
            }
        }
        let floors = Self::read_wal_floors(cfg)?;
        Ok(RestoredCheckpoints {
            cuts,
            floors,
            currents,
            summary,
        })
    }

    /// Reads the durable WAL floors (absent file means genesis everywhere
    /// — pre-checkpoint databases replay fully).
    fn read_wal_floors(
        cfg: &DurableConfig,
    ) -> Result<std::collections::HashMap<u16, kivi_durability::LaneFloor>, EngineError> {
        let floors = kivi_checkpoint::read_wal_floor(&cfg.data_dir)?.unwrap_or_default();
        Ok(floors
            .into_iter()
            .map(|floor| (floor.lane, floor))
            .collect())
    }

    /// Opens every distinct WAL lane directory (no scanning yet).
    fn open_recovery_lanes(
        wal_dir: &std::path::Path,
        lane_indexes: &[u16],
        identity: LaneIdentity,
        segment_target_bytes: u64,
    ) -> Result<std::collections::HashMap<u16, LocalWalLane>, EngineError> {
        use std::collections::HashMap;
        let mut distinct: Vec<u16> = lane_indexes.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        let mut lanes = HashMap::new();
        for lane in distinct {
            let lane_state = LocalWalLane::open(wal_dir, lane, identity, segment_target_bytes)?;
            lanes.insert(lane, lane_state);
        }
        Ok(lanes)
    }

    /// Scans every lane exactly once from its durable floor (validating,
    /// repairing torn tails on retained tails) and merges records per
    /// tablet. Lanes stay open for the handoff. Lanes without a recorded
    /// floor replay from genesis.
    fn scan_recovery_lanes(
        lanes: &mut std::collections::HashMap<u16, LocalWalLane>,
        floors: &std::collections::HashMap<u16, kivi_durability::LaneFloor>,
    ) -> Result<RecoveryScan, EngineError> {
        use kivi_durability::{WalEntry, WalRecord};
        use std::collections::HashMap;
        let mut per_tablet: HashMap<TabletId, Vec<(u64, WalRecord)>> = HashMap::new();
        let mut consensus: HashMap<TabletId, Vec<(u64, kivi_durability::RaftRecord)>> =
            HashMap::new();
        let mut segments_scanned = 0u64;
        let mut tail_truncated_bytes = 0u64;
        let mut lane_order: Vec<u16> = lanes.keys().copied().collect();
        lane_order.sort_unstable();
        for lane in lane_order {
            let lane_state = lanes.get_mut(&lane).expect("lane present");
            let floor = floors
                .get(&lane)
                .copied()
                .unwrap_or_else(|| kivi_durability::LaneFloor::for_lane(lane));
            let recovery = lane_state.recover_from(floor)?;
            segments_scanned += recovery.segments_scanned;
            tail_truncated_bytes += recovery.truncated_bytes;
            // Lanes recover in file order, so both families stay ordered
            // without re-sorting: tablet records sort by commit below,
            // consensus records keep arrival order per group.
            for entry in recovery.records {
                match entry.entry {
                    WalEntry::Tablet(record) => {
                        per_tablet
                            .entry(record.tablet())
                            .or_default()
                            .push((record.commit().as_u64(), record));
                    }
                    WalEntry::Consensus(record) => {
                        consensus
                            .entry(record.group())
                            .or_default()
                            .push((entry.batch_seq, record));
                    }
                }
            }
        }
        Ok((
            per_tablet,
            consensus,
            segments_scanned,
            tail_truncated_bytes,
        ))
    }

    /// Validates merged records against the startup directory, skips
    /// checkpoint-covered records (commit at or below the tablet's cut —
    /// counted as obsolete, never replayed), and replays the tail in
    /// commit order.
    fn replay_records(
        by_worker: &mut [Vec<LiveTablet>],
        routing: &RoutingSnapshot,
        namespace: NamespaceId,
        per_tablet: std::collections::HashMap<TabletId, Vec<(u64, kivi_durability::WalRecord)>>,
        consensus: &std::collections::HashMap<TabletId, Vec<(u64, kivi_durability::RaftRecord)>>,
        cuts: &std::collections::HashMap<TabletId, u64>,
    ) -> Result<(u64, u64, u64), EngineError> {
        use kivi_durability::{RecoveryError, WalRecord};
        // Single-node mode owns no consensus history: any consensus record
        // means this data directory was written by replicated mode, and
        // serving it as single-node would fork history. The replicated
        // replayer (consensus stage) consumes these instead.
        if let Some(group) = consensus.keys().next() {
            return Err(RecoveryError::UnexpectedConsensusRecord {
                group: group.as_u64(),
            }
            .into());
        }
        let mut records_replayed = 0u64;
        let mut tablets_recovered = 0u64;
        let mut obsolete_skipped = 0u64;
        for (tablet_id, mut entries) in per_tablet {
            let descriptor =
                routing
                    .directory()
                    .get(tablet_id)
                    .ok_or(RecoveryError::UnknownTablet {
                        tablet: tablet_id.as_u64(),
                    })?;
            if !descriptor.state().is_writable() {
                return Err(RecoveryError::UnknownTablet {
                    tablet: tablet_id.as_u64(),
                }
                .into());
            }
            entries.sort_by_key(|(commit, _)| *commit);
            let cut = cuts.get(&tablet_id).copied().unwrap_or(0);
            let live = by_worker
                .iter_mut()
                .flatten()
                .find(|live| live.id() == tablet_id)
                .ok_or(RecoveryError::UnknownTablet {
                    tablet: tablet_id.as_u64(),
                })?;
            for (_, record) in &entries {
                if record.namespace() != namespace {
                    return Err(RecoveryError::NamespaceMismatch {
                        tablet: tablet_id.as_u64(),
                        expected: namespace.as_u64(),
                        found: record.namespace().as_u64(),
                    }
                    .into());
                }
                if record.epoch() != descriptor.epoch() || record.guard() != descriptor.guard() {
                    return Err(RecoveryError::FencingMismatch {
                        tablet: tablet_id.as_u64(),
                    }
                    .into());
                }
            }
            for (_, record) in entries {
                // Checkpoint-covered records are obsolete by cut
                // semantics: the checkpoint already restores their
                // effects (and their dedup outcomes), so replaying them
                // would duplicate logical application. Reappearing
                // deleted segments are harmless for exactly this reason.
                if record.commit().as_u64() <= cut {
                    obsolete_skipped += 1;
                    continue;
                }
                match record {
                    WalRecord::Mutation(entry) => {
                        let is_persist = matches!(
                            entry.mutation.as_operation(),
                            kivi_state::Operation::PersistExpiry { .. }
                        );
                        live.apply_recovered_mutation(
                            &entry.mutation,
                            &entry.expected,
                            is_persist,
                            entry.commit,
                            entry.identity.as_ref(),
                            entry.now,
                        )?;
                    }
                    WalRecord::Outcome(entry) => {
                        live.install_recovered_outcome(
                            &entry.outcome,
                            entry.commit,
                            entry.identity.as_ref(),
                            entry.opcode,
                        )?;
                    }
                }
                records_replayed += 1;
            }
            tablets_recovered += 1;
        }
        Ok((records_replayed, tablets_recovered, obsolete_skipped))
    }

    /// Spawns channel-only workers (embedded path, no networking).
    #[allow(clippy::too_many_arguments)]
    fn spawn_channel_workers(
        request_capacity: usize,
        worker_count: usize,
        by_worker: Vec<Vec<LiveTablet>>,
        lanes: Vec<Option<WorkerDurability>>,
        chunks: Vec<crate::worker::WorkerChunks>,
    ) -> (Vec<WorkerHandle>, Vec<RequestIngress>) {
        let mut workers = Vec::with_capacity(worker_count);
        let mut senders = Vec::with_capacity(worker_count);
        for (((index, tablets), durability), chunks) in
            by_worker.into_iter().enumerate().zip(lanes).zip(chunks)
        {
            let id = WorkerId::from_u64(index as u64);
            let handle = WorkerHandle::spawn(id, tablets, request_capacity, durability, chunks);
            senders.push(handle_sender(&handle));
            workers.push(handle);
        }
        (workers, senders)
    }

    /// Spawns networked workers, collecting bound addresses into the shared
    /// endpoint registry once every worker reports ready.
    #[allow(clippy::too_many_arguments)]
    fn spawn_network_workers(
        network: &crate::net::EngineNetwork,
        request_capacity: usize,
        worker_count: usize,
        by_worker: Vec<Vec<LiveTablet>>,
        routing: &Arc<ArcSwap<RoutingSnapshot>>,
        lanes: Vec<Option<WorkerDurability>>,
        chunks: Vec<crate::worker::WorkerChunks>,
    ) -> Result<NetworkSpawn, EngineError> {
        let endpoints = Arc::new(ArcSwap::new(Arc::new(crate::net::EndpointMap::new())));
        // Lanes present means durable mode (start() builds all-or-none);
        // the advertised capabilities follow the mode, never flags.
        let durable = lanes.iter().any(Option::is_some);
        debug_assert!(lanes.iter().all(|lane| lane.is_some() == durable));
        let advertised_caps = if durable {
            Capabilities::BASE_V1 | Capabilities::DURABLE_MUTATION_DEDUP | Capabilities::STREAMING
        } else {
            Capabilities::BASE_V1 | Capabilities::STREAMING
        };
        let mut workers = Vec::with_capacity(worker_count);
        let mut senders = Vec::with_capacity(worker_count);
        let mut bound: Vec<(WorkerId, String)> = Vec::with_capacity(worker_count);
        let mut addrs: Vec<std::net::SocketAddr> = Vec::with_capacity(worker_count);
        for (((index, tablets), durability), chunks) in
            by_worker.into_iter().enumerate().zip(lanes).zip(chunks)
        {
            let id = WorkerId::from_u64(index as u64);
            let port = if network.ports.is_empty() {
                // Port 0 selects an ephemeral port per socket: every worker
                // binds 0 and reports its own address. (`0 + i` would name
                // literal port `i`, sharing one privileged port instead.)
                if network.base_port == 0 {
                    0
                } else {
                    network
                        .base_port
                        .checked_add(u16::try_from(index).map_err(|_| {
                            EngineError::InvalidConfig {
                                reason: "worker count exceeds port space".to_owned(),
                            }
                        })?)
                        .ok_or_else(|| EngineError::InvalidConfig {
                            reason: "worker ports overflow u16".to_owned(),
                        })?
                }
            } else {
                *network
                    .ports
                    .get(index)
                    .ok_or_else(|| EngineError::InvalidConfig {
                        reason: "ports must name exactly one port per worker".to_owned(),
                    })?
            };
            let listen = std::net::SocketAddr::new(network.bind_ip, port);
            let advertise_ip = if listen.ip().is_unspecified() {
                std::net::IpAddr::from([127, 0, 0, 1])
            } else {
                listen.ip()
            };
            let net = crate::net::NetConfig {
                worker: id,
                listen,
                advertise_ip,
                conn: network.conn,
                turn: network.turn,
                node_id: network.node_id,
                cluster_id: network.cluster_id,
                incarnation: network.incarnation,
                advertised_caps,
                endpoints: Arc::clone(&endpoints),
                routing: Arc::clone(routing),
            };
            let (handle, addr) = WorkerHandle::spawn_net(
                id,
                tablets,
                request_capacity,
                durability,
                chunks,
                crate::net::NetLaunch {
                    routing: Arc::clone(routing),
                    net,
                    affinity: network.affinity.clone(),
                    worker_index: index,
                    worker_count,
                },
            )
            .map_err(EngineError::NetStart)?;
            bound.push((id, format!("{}:{}", advertise_ip, addr.port())));
            addrs.push(addr);
            senders.push(handle_sender(&handle));
            workers.push(handle);
        }
        endpoints.store(Arc::new(bound.into_iter().collect()));
        Ok((workers, senders, addrs))
    }

    /// Returns the bound native endpoint per worker index (empty for
    /// channel-only engines). Index `i` is worker `i`'s listen address.
    #[must_use]
    pub fn worker_addrs(&self) -> &[std::net::SocketAddr] {
        &self.bound
    }

    /// Returns a clonable client handle for application threads.
    #[must_use]
    pub fn client(&self) -> LocalClient {
        LocalClient {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Atomically replaces the routing publication after revalidation.
    /// Callers observe the old coherent pair or the new one, never a mix.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] when the replacement is incoherent; the
    /// current publication stays installed.
    pub fn replace_routing(
        &self,
        directory: DirectorySnapshot,
        placement: Placement,
    ) -> Result<(), EngineError> {
        let worker_count = self.shared.routing.load().worker_count();
        let routing =
            RoutingSnapshot::build(self.shared.namespace, directory, placement, worker_count)?;
        self.shared.routing.store(Arc::new(routing));
        Ok(())
    }

    /// Returns the owning worker of one tablet under the current routing
    /// publication, if placed.
    #[must_use]
    pub fn worker_of(&self, tablet: TabletId) -> Option<WorkerId> {
        self.shared.routing.load().placement().worker_of(tablet)
    }

    /// Snapshots per-worker execution counters, split by arrival path
    /// (channel vs. direct). Used to prove the native fast path bypasses
    /// the cross-thread queue, and for operator introspection.
    ///
    /// Workers that already exited report their last... — no: exited workers
    /// are simply absent from the result. Query before shutdown for full data.
    #[must_use]
    pub fn worker_metrics(&self) -> Vec<(WorkerId, crate::worker::WorkerMetrics)> {
        self.admin_handle().worker_metrics_snapshot()
    }

    /// Returns a clonable read-only handle for the admin/control plane.
    ///
    /// The handle carries no worker threads or queues — only the shared
    /// routing publication, bound endpoints, and control senders — so it is
    /// `Send + Sync` and safe to hold across Tokio tasks. Metric queries
    /// block on a rendezvous: serve them from `spawn_blocking`, never from
    /// async handlers directly.
    #[must_use]
    pub fn admin_handle(&self) -> AdminHandle {
        AdminHandle {
            namespace: self.shared.namespace,
            routing: Arc::clone(&self.shared.routing),
            chunk_lanes: self
                .workers
                .iter()
                .zip(self.chunk_lanes.iter())
                .map(|(worker, lane)| (worker.id(), lane.clone()))
                .collect(),
            controls: self
                .workers
                .iter()
                .map(|worker| (worker.id(), worker.control_sender()))
                .collect(),
            endpoints: self
                .workers
                .iter()
                .zip(self.bound.iter())
                .map(|(worker, addr)| (worker.id(), *addr))
                .collect(),
            durability: self.durability.clone(),
            checkpoint: Arc::clone(&self.checkpoint_admin),
        }
    }

    /// Requests checkpoints now (`None` = every dirty tablet). No-op in
    /// ephemeral mode. Returns immediately; the background worker builds
    /// asynchronously.
    pub fn request_checkpoint(&self, tablet: Option<TabletId>) {
        if let Some(checkpoint) = self.checkpoint.as_ref() {
            checkpoint.trigger(tablet);
        }
    }

    /// Routes one key to its `(tablet, owner worker)` under the current
    /// publication. Exposed for debugging, placement verification, and the
    /// future `EXPLAIN KEY` surface — normal operations route internally.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] when hashing is unsupported or no active
    /// tablet covers the key.
    pub fn route_key(&self, key: &Key) -> Result<(TabletId, WorkerId), EngineError> {
        let routing = self.shared.routing.load();
        let tablet = crate::compound::route_point_key(
            routing.directory(),
            self.shared.namespace,
            key.as_bytes(),
        )
        .ok_or(EngineError::NoRoute)?;
        let worker = routing
            .placement()
            .worker_of(tablet)
            .ok_or(EngineError::NoRoute)?;
        Ok((tablet, worker))
    }

    /// Returns the directory version of the current routing publication.
    #[must_use]
    pub fn routing_version(&self) -> kivi_tablet::DirectoryVersion {
        self.shared.routing.load().version()
    }

    /// Shuts every worker down and joins all threads, surfacing the first
    /// worker panic instead of swallowing it.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::WorkerPanicked`] when a worker thread panicked;
    /// joined workers stay joined either way.
    pub fn shutdown(mut self) -> Result<ShutdownReport, EngineError> {
        // The checkpoint worker stops first (finishing its in-flight
        // build, if any): no new captures or lane maintenance once
        // workers begin draining.
        if let Some(checkpoint) = self.checkpoint.as_mut() {
            checkpoint.shutdown();
        }
        for worker in &self.workers {
            let _ = worker.try_control(crate::worker::WorkerControl::Shutdown);
        }
        // Wakeup for accept loops: they park in `accept()` uncancellable
        // (cancelling AcceptEx leaks the listener), so each bound worker
        // gets a dummy connection that wakes its accept turn, where it
        // observes the shutdown flag and breaks. The dummy sends nothing
        // and closes at once: the server reads EOF and closes it cleanly
        // with no log noise. A short settle precedes the dummies so the
        // flag is set before any dummy is accepted (the Shutdown control
        // above already pinged every bridge awake, so bridges exit
        // promptly; the settle is for the accept loops only).
        std::thread::sleep(std::time::Duration::from_millis(50));
        for addr in &self.bound {
            if let Ok(socket) =
                std::net::TcpStream::connect_timeout(addr, std::time::Duration::from_secs(2))
            {
                drop(socket);
            }
        }
        let mut joined = 0usize;
        let mut panic = None;
        for worker in &mut self.workers {
            match worker.join() {
                Ok(()) => joined += 1,
                Err(payload) => {
                    if panic.is_none() {
                        panic = Some(EngineError::WorkerPanicked {
                            worker: worker.id(),
                            message: panic_message(&payload),
                        });
                    }
                }
            }
        }
        // Chunk lanes stop after every worker joined: no worker submits
        // once it exits, so no chunk work is abandoned mid-flight (queued
        // lane jobs fail their severed replies instead of hanging).
        for (guard, lane) in self.chunk_guards.iter_mut().zip(self.chunk_lanes.iter()) {
            guard.shutdown(lane);
        }
        match panic {
            Some(error) => Err(error),
            None => Ok(ShutdownReport {
                workers_joined: joined,
            }),
        }
    }
}

impl fmt::Debug for LocalEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalEngine")
            .field("workers", &self.workers.len())
            .finish_non_exhaustive()
    }
}

/// How many workers exited cleanly at shutdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShutdownReport {
    /// Workers joined without panicking.
    pub workers_joined: usize,
}

/// Extracts the current request ingress out of a fresh handle. Handle
/// ingress bundles are not otherwise exposed; the engine keeps exactly one
/// per worker.
fn handle_sender(handle: &WorkerHandle) -> RequestIngress {
    handle.sender()
}

/// Best-effort readable panic message from a thread payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> Option<String> {
    if let Some(text) = payload.downcast_ref::<String>() {
        Some(text.clone())
    } else {
        payload
            .downcast_ref::<&str>()
            .map(|text| (*text).to_owned())
    }
}

/// Maps bounded-queue admission outcomes onto engine errors: a full queue is
/// fast backpressure (`Overloaded`), a gone worker is `WorkerDown`.
fn map_send_error(error: &TrySendError<TabletRequest>, worker: WorkerId) -> EngineError {
    match error {
        TrySendError::Full(_) => EngineError::Overloaded,
        TrySendError::Disconnected(_) => EngineError::WorkerDown { worker },
    }
}

/// Clonable, thread-safe client for ordinary application threads.
///
/// Each operation loads the current routing snapshot (lock-free `Arc`
/// load), hashes the key, and hands the typed operation to the owning
/// worker over its bounded queue. Responses rendezvous back on a dedicated
/// one-shot channel.
#[derive(Debug, Clone)]
pub struct LocalClient {
    shared: Arc<EngineShared>,
}

/// One embedded batch-attempt outcome.
#[derive(Debug)]
enum BatchAttempt {
    /// OCC conflict: retry with a fresh `TxnId`.
    Conflict,
    /// Terminal driver failure.
    Fatal(EngineError),
}

/// Which decision the coordinator record converged to after the guarded
/// abort path.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AbortOutcome {
    Aborted,
    Committed,
}

impl LocalClient {
    /// Executes one typed operation against its owning tablet. Chunked
    /// reads resolve here on the caller thread (blocking lane call — this
    /// is application-thread context, never the reactor), so every typed
    /// method below keeps its plain-bytes contract whatever the physical
    /// representation is.
    fn execute(&self, key: &Key, op: Operation) -> Result<OperationResult, EngineError> {
        // `GetRange` over a chunked root resolves through the lane and
        // then slices: the store names the reference, this layer applies
        // the caller's window (a future lane range-read will fetch only
        // the required chunks instead of the full payload).
        let range = match &op {
            Operation::GetRange { offset, len, .. } => Some((*offset, *len)),
            _ => None,
        };
        let routing = self.shared.routing.load();
        let tablet = crate::compound::route_point_key(
            routing.directory(),
            self.shared.namespace,
            key.as_bytes(),
        )
        .ok_or(EngineError::NoRoute)?;
        self.execute_on(tablet, op, range)
    }

    /// Executes one typed operation against an explicitly addressed tablet
    /// (transaction drivers address participants directly after planning).
    fn execute_on(
        &self,
        tablet: TabletId,
        op: Operation,
        range: Option<(u64, u64)>,
    ) -> Result<OperationResult, EngineError> {
        let routing = self.shared.routing.load();
        let worker = routing
            .placement()
            .worker_of(tablet)
            .ok_or(EngineError::UnknownTablet { tablet })?;
        let index =
            usize::try_from(worker.as_u64()).map_err(|_| EngineError::UnknownWorker { worker })?;
        let sender = self
            .shared
            .senders
            .get(index)
            .ok_or(EngineError::UnknownWorker { worker })?;
        let lane = self
            .shared
            .chunk_lanes
            .get(index)
            .ok_or(EngineError::UnknownWorker { worker })?;
        let (respond, receive) = bounded(RESPONSE_CAPACITY);
        let request = TabletRequest {
            tablet,
            op,
            now: SystemClock::wall_now(),
            // Embedded callers share fate with the process: no retry
            // identity, so no dedup — durable mode still WALs the write.
            identity: None,
            respond,
        };
        match sender.try_send(request) {
            Ok(()) => {}
            Err(error) => return Err(map_send_error(&error, worker)),
        }
        match receive
            .recv()
            .map_err(|_| EngineError::WorkerDown { worker })?
            .map_err(EngineError::from)?
        {
            OperationResult::ChunkedValue {
                manifest,
                logical_len,
            } => {
                let bytes = lane.read_value_blocking(manifest, logical_len)?;
                match range {
                    None => Ok(OperationResult::Value(Some(bytes))),
                    Some((offset, len)) => Ok(OperationResult::Value(Some(
                        kivi_state::slice_range(&bytes, offset, len),
                    ))),
                }
            }
            other => Ok(other),
        }
    }

    /// Fetches bytes (`None` when absent, expired, or never written).
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on routing/transport failure or wrong-type access.
    ///
    /// # Panics
    ///
    /// Panics if tablet execution returns an outcome shape that cannot result
    /// from the issued operation (an internal contract violation, never a
    /// runtime condition).
    pub fn get(&self, key: &Key) -> Result<Option<Bytes>, EngineError> {
        match self.execute(key, Operation::Get { key: key.clone() })? {
            OperationResult::Value(value) => Ok(value),
            unexpected => panic!("get contract violated: {unexpected:?}"),
        }
    }

    /// Stores bytes, overwriting any type and clearing expiry.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on routing/transport failure.
    ///
    /// # Panics
    ///
    /// Panics if tablet execution returns an outcome shape that cannot result
    /// from the issued operation (an internal contract violation, never a
    /// runtime condition).
    pub fn set(&self, key: &Key, value: Bytes) -> Result<(), EngineError> {
        match self.execute(
            key,
            Operation::Set {
                key: key.clone(),
                value,
            },
        )? {
            OperationResult::Stored { .. } => Ok(()),
            unexpected => panic!("set contract violated: {unexpected:?}"),
        }
    }

    /// Patches a byte range of a value (partial update). Against absent
    /// state the base reads as empty; past-the-end gaps zero-pad; counters
    /// fail with wrong-type. The live expiry survives (unlike
    /// [`set`](Self::set)): partial writes touch bytes, never the TTL.
    /// Large results and chunked bases restage through the chunk lane (also
    /// preserving expiry), so the patch itself stays small — bulk rewrites
    /// belong on the streaming upload path instead.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on routing/transport failure, wrong-type
    /// access, or an inadmittable range (overflow, past the range bound).
    ///
    /// # Panics
    ///
    /// Panics if tablet execution returns an outcome shape that cannot result
    /// from the issued operation (an internal contract violation, never a
    /// runtime condition).
    pub fn set_range(&self, key: &Key, offset: u64, patch: Bytes) -> Result<(), EngineError> {
        match self.execute(
            key,
            Operation::SetRange {
                key: key.clone(),
                offset,
                patch,
            },
        )? {
            // Restaged range writes (large results, chunked bases) execute
            // as conditional/chunked ops internally; `Always` always
            // applies, so this is the same stored outcome in disguise.
            OperationResult::Stored { .. }
            | OperationResult::ConditionalSet { applied: true, .. } => Ok(()),
            unexpected => panic!("set-range contract violated: {unexpected:?}"),
        }
    }

    /// Removes a key, reporting whether a live object existed.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on routing/transport failure.
    ///
    /// # Panics
    ///
    /// Panics if tablet execution returns an outcome shape that cannot result
    /// from the issued operation (an internal contract violation, never a
    /// runtime condition).
    pub fn delete(&self, key: &Key) -> Result<bool, EngineError> {
        match self.execute(key, Operation::Delete { key: key.clone() })? {
            OperationResult::Deleted { existed } => Ok(existed),
            unexpected => panic!("delete contract violated: {unexpected:?}"),
        }
    }

    /// Reports whether a live value of any type exists.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on routing/transport failure.
    ///
    /// # Panics
    ///
    /// Panics if tablet execution returns an outcome shape that cannot result
    /// from the issued operation (an internal contract violation, never a
    /// runtime condition).
    pub fn exists(&self, key: &Key) -> Result<bool, EngineError> {
        match self.execute(key, Operation::Exists { key: key.clone() })? {
            OperationResult::Exists(present) => Ok(present),
            unexpected => panic!("exists contract violated: {unexpected:?}"),
        }
    }

    /// Reads a counter (`None` when absent).
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on routing/transport failure or wrong-type access.
    ///
    /// # Panics
    ///
    /// Panics if tablet execution returns an outcome shape that cannot result
    /// from the issued operation (an internal contract violation, never a
    /// runtime condition).
    pub fn counter_get(&self, key: &Key) -> Result<Option<i64>, EngineError> {
        match self.execute(key, Operation::CounterGet { key: key.clone() })? {
            OperationResult::Counter(value) => Ok(value),
            unexpected => panic!("counter_get contract violated: {unexpected:?}"),
        }
    }

    /// Adds `delta` to a counter (creating it at `delta` when absent) and
    /// returns the new value.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on routing/transport failure, wrong-type
    /// access, or counter overflow.
    ///
    /// # Panics
    ///
    /// Panics if tablet execution returns an outcome shape that cannot result
    /// from the issued operation (an internal contract violation, never a
    /// runtime condition).
    pub fn counter_add(&self, key: &Key, delta: i64) -> Result<i64, EngineError> {
        match self.execute(
            key,
            Operation::CounterAdd {
                key: key.clone(),
                delta,
            },
        )? {
            OperationResult::CounterUpdated { value, .. } => Ok(value),
            unexpected => panic!("counter_add contract violated: {unexpected:?}"),
        }
    }

    /// Reads one key's logical version (`None` when absent) without
    /// fetching its value. Index maintenance and OCC planning use this.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on routing/transport failure.
    ///
    /// # Panics
    ///
    /// Panics if tablet execution returns an outcome shape that cannot result
    /// from the issued operation (an internal contract violation, never a
    /// runtime condition).
    pub fn get_version(&self, key: &Key) -> Result<Option<ObjectVersion>, EngineError> {
        match self.execute(key, Operation::GetVersion { key: key.clone() })? {
            OperationResult::Version(version) => Ok(version),
            unexpected => panic!("get_version contract violated: {unexpected:?}"),
        }
    }

    /// Reserves one key for a transaction (2PC prepare): OCC validation
    /// plus a durable intent, no visible mutation yet. Low-level driver
    /// primitive; most callers want [`atomic_batch`](Self::atomic_batch).
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on routing/transport failure or OCC
    /// conflict (surfaced as `TabletError::Op(OpError::TxnConflict)`).
    ///
    /// # Panics
    ///
    /// Panics on an outcome shape the tablet contract cannot produce for
    /// this operation (internal invariant, never a runtime condition).
    pub fn txn_prepare(
        &self,
        txn: TxnId,
        coordinator: TabletId,
        write: TxnWrite,
    ) -> Result<(), EngineError> {
        let key = write.key.clone();
        match self.execute(
            &key,
            Operation::TxnPrepare {
                txn,
                coordinator,
                write,
            },
        )? {
            OperationResult::TxnPrepared => Ok(()),
            OperationResult::TxnConflict => Err(EngineError::Tablet(
                crate::tablet::TabletError::Op(kivi_state::OpError::TxnConflict),
            )),
            unexpected => panic!("txn_prepare contract violated: {unexpected:?}"),
        }
    }

    /// Resolves one key's transaction intent (commit applies, abort
    /// discards; idempotent). Low-level driver primitive.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on routing/transport failure.
    ///
    /// # Panics
    ///
    /// Panics on an outcome shape the tablet contract cannot produce for
    /// this operation (internal invariant, never a runtime condition).
    pub fn txn_finalize(
        &self,
        key: &Key,
        txn: TxnId,
        commit: bool,
    ) -> Result<Option<ObjectVersion>, EngineError> {
        match self.execute(
            key,
            Operation::TxnFinalize {
                txn,
                key: key.clone(),
                commit,
            },
        )? {
            OperationResult::TxnFinalized { applied, version } => {
                Ok(applied.then_some(version).flatten())
            }
            OperationResult::TxnConflict => Err(EngineError::Tablet(
                crate::tablet::TabletError::Op(kivi_state::OpError::TxnConflict),
            )),
            unexpected => panic!("txn_finalize contract violated: {unexpected:?}"),
        }
    }

    /// Executes an atomic batch: all writes commit or none does, across
    /// one or many tablets (uniform OCC + 2PC with coordinator == first
    /// write's tablet). Transport is synchronous channel execution, so
    /// multi-worker batches work like single-worker ones.
    ///
    /// Returns resulting versions in request order (`None` for deletes).
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on routing/transport failure or OCC
    /// conflict (nothing committed on conflict).
    pub fn atomic_batch(
        &self,
        writes: &[TxnWrite],
    ) -> Result<Vec<Option<ObjectVersion>>, EngineError> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static TXN_SALT: AtomicU64 = AtomicU64::new(1);
        if writes.is_empty() {
            return Err(EngineError::InvalidRequest {
                detail: "empty batch".to_owned(),
            });
        }
        let mut attempt: u32 = 0;
        loop {
            let salt = TXN_SALT.fetch_add(1, Ordering::SeqCst);
            let txn = TxnId::derive(0, salt, u64::from(attempt));
            match self.batch_attempt(writes, txn) {
                Ok(versions) => return Ok(versions),
                Err(BatchAttempt::Conflict) if attempt < 3 => {
                    attempt += 1;
                }
                Err(BatchAttempt::Conflict) => {
                    return Err(EngineError::Tablet(crate::tablet::TabletError::Op(
                        kivi_state::OpError::TxnConflict,
                    )));
                }
                Err(BatchAttempt::Fatal(error)) => return Err(error),
            }
        }
    }

    /// One batch attempt under a fixed `TxnId`.
    fn batch_attempt(
        &self,
        writes: &[TxnWrite],
        txn: TxnId,
    ) -> Result<Vec<Option<ObjectVersion>>, BatchAttempt> {
        let routing = self.shared.routing.load();
        let namespace = self.shared.namespace;
        let directory = routing.directory();
        let route = |key: &[u8]| crate::compound::route_point_key(directory, namespace, key);
        let plan = plan_transaction(
            txn,
            writes.to_vec(),
            route,
            directory.version().as_u64(),
            namespace,
        )
        .map_err(|_| {
            BatchAttempt::Fatal(EngineError::InvalidRequest {
                detail: "batch exceeds transaction bounds or routes nowhere".to_owned(),
            })
        })?;
        // Recovery-first: an existing durable decision resolves without
        // re-preparing (re-prepares after a commit would spuriously
        // conflict with the committed state).
        let record_key = txn_record_key(plan.coordinator, txn);
        if let Some(record) = self.read_record(&record_key) {
            match record.state {
                TxnState::Committed => {
                    return self.finalize_all(&plan, true).map_err(BatchAttempt::Fatal);
                }
                TxnState::Aborted => {
                    return Err(BatchAttempt::Fatal(EngineError::InvalidRequest {
                        detail: "transaction aborted".to_owned(),
                    }));
                }
                TxnState::Begun => {}
            }
        }
        let mut prepared: Vec<usize> = Vec::with_capacity(writes.len());
        for (index, write) in writes.iter().enumerate() {
            match self.txn_prepare(txn, plan.coordinator, write.clone()) {
                Ok(()) => prepared.push(index),
                Err(EngineError::Tablet(crate::tablet::TabletError::Op(
                    kivi_state::OpError::TxnConflict,
                ))) => {
                    return match self.abort_prepared(&plan, &prepared) {
                        AbortOutcome::Aborted => Err(BatchAttempt::Conflict),
                        AbortOutcome::Committed => {
                            self.finalize_all(&plan, true).map_err(BatchAttempt::Fatal)
                        }
                    };
                }
                Err(error) => {
                    return match self.abort_prepared(&plan, &prepared) {
                        AbortOutcome::Aborted => Err(BatchAttempt::Fatal(error)),
                        AbortOutcome::Committed => {
                            self.finalize_all(&plan, true).map_err(BatchAttempt::Fatal)
                        }
                    };
                }
            }
        }
        let mut decided = plan.record.clone();
        decided.state = TxnState::Committed;
        if !self.decide_commit(&plan, &decided) {
            // Re-read: an ambiguous decide may actually have committed
            // (never blind-overwrite a decision with an abort).
            let record_key = txn_record_key(plan.coordinator, plan.txn);
            if self
                .read_record(&record_key)
                .is_some_and(|record| record.state == TxnState::Committed)
            {
                return self.finalize_all(&plan, true).map_err(BatchAttempt::Fatal);
            }
            return match self.abort_prepared(&plan, &prepared) {
                AbortOutcome::Aborted => Err(BatchAttempt::Fatal(EngineError::InvalidRequest {
                    detail: "commit undecided; aborted".to_owned(),
                })),
                AbortOutcome::Committed => {
                    self.finalize_all(&plan, true).map_err(BatchAttempt::Fatal)
                }
            };
        }
        self.finalize_all(&plan, true).map_err(BatchAttempt::Fatal)
    }

    /// Persists the Commit decision guarded on absence (CAS): the record
    /// key is unique per transaction, so a present record means a previous
    /// drive already decided. Returns whether the commit decision is
    /// durable.
    fn decide_commit(&self, plan: &kivi_state::TxnDriverPlan, decided: &TxnRecord) -> bool {
        self.decide_record(plan, decided, 0)
    }

    /// Persists one terminal decision through an absence-guarded
    /// prepare + commit-finalize on the record key (CAS — never a blind
    /// overwrite). `decide_salt` separates the commit decide-transaction
    /// from the abort one so the two never share intent identity.
    fn decide_record(
        &self,
        plan: &kivi_state::TxnDriverPlan,
        decided: &TxnRecord,
        decide_salt: u64,
    ) -> bool {
        use kivi_state::{Operation, OperationResult, TxnExpect, TxnWrite, TxnWriteKind};
        let record_key = txn_record_key(plan.coordinator, plan.txn);
        // Deterministic decide-transaction id (re-drives idempotently).
        let decide_txn = TxnId::derive(u128::from_le_bytes(plan.txn.as_bytes()), 0, decide_salt);
        let prepare = Operation::TxnPrepare {
            txn: decide_txn,
            coordinator: plan.coordinator,
            write: TxnWrite {
                key: record_key.clone(),
                kind: TxnWriteKind::Put(bytes::Bytes::from(decided.encode())),
                expect: TxnExpect::Absent,
            },
        };
        if !matches!(
            self.execute_on(plan.coordinator, prepare, None),
            Ok(OperationResult::TxnPrepared)
        ) {
            return false;
        }
        let finalize = Operation::TxnFinalize {
            txn: decide_txn,
            key: record_key,
            commit: true,
        };
        matches!(
            self.execute_on(plan.coordinator, finalize, None),
            Ok(OperationResult::TxnFinalized { applied: true, .. })
        )
    }

    /// Finalizes every key of a plan (commit or abort), collecting
    /// versions in request order.
    fn finalize_all(
        &self,
        plan: &kivi_state::TxnDriverPlan,
        commit: bool,
    ) -> Result<Vec<Option<ObjectVersion>>, EngineError> {
        let mut versions: Vec<Option<ObjectVersion>> = vec![None; plan.writes.len()];
        for (order, write) in plan.writes.iter().enumerate() {
            if let Some(version) = self.txn_finalize(&write.key, plan.txn, commit)? {
                versions[order] = Some(version);
            }
        }
        Ok(versions)
    }

    /// Reads and decodes the coordinator record (`None` when absent,
    /// unreadable, or undecodable).
    fn read_record(&self, record_key: &Key) -> Option<TxnRecord> {
        self.get(record_key)
            .ok()?
            .and_then(|bytes| TxnRecord::decode(&bytes).ok())
    }

    /// Persists the Abort decision (guarded CAS, never a blind overwrite)
    /// and finalize-aborts every prepared key. Returns which decision the
    /// coordinator record converged to: a lost race means a concurrent
    /// drive committed, and the caller must converge via commit instead.
    fn abort_prepared(&self, plan: &kivi_state::TxnDriverPlan, prepared: &[usize]) -> AbortOutcome {
        let mut aborted = plan.record.clone();
        aborted.state = TxnState::Aborted;
        if !self.decide_record(plan, &aborted, 1) {
            let record_key = txn_record_key(plan.coordinator, plan.txn);
            if self
                .read_record(&record_key)
                .is_some_and(|record| record.state == TxnState::Committed)
            {
                return AbortOutcome::Committed;
            }
            return AbortOutcome::Aborted;
        }
        for index in prepared {
            let _ = self.txn_finalize(&plan.writes[*index].key, plan.txn, false);
        }
        AbortOutcome::Aborted
    }

    /// Resolves abandoned transaction intents on every local tablet:
    /// decided transactions finalize per their coordinator record;
    /// expired undecided transactions are CAS-aborted through the record
    /// (never unilaterally). Returns `(finalized, aborted)`.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on routing/transport failure; per-intent
    /// failures are skipped (the next pass retries), never fatal.
    pub fn resolve_transactions(&self) -> Result<(usize, usize), EngineError> {
        // Embedded discovery: LocalClient addresses tablets through worker
        // channels and cannot enumerate cross-thread intent tables, so
        // discovery-driven resolution lives in the cluster reconciler
        // (which owns machine access). Embedded callers resolve explicitly
        // via `resolve_transaction` with the keys they drove.
        Ok((0, 0))
    }

    /// Resolves one transaction's intents over `keys`: reads its
    /// coordinator record and finalizes accordingly (commit applies,
    /// abort discards; missing intents are idempotent no-ops).
    /// Undecided records resolve nothing (the driver is still live or the
    /// lease path owns them — see the cluster reconciler).
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on routing/transport failure.
    pub fn resolve_transaction(
        &self,
        coordinator: TabletId,
        txn: TxnId,
        keys: &[Key],
    ) -> Result<(usize, usize), EngineError> {
        let record_key = txn_record_key(coordinator, txn);
        let record = self
            .get(&record_key)?
            .map(|bytes| TxnRecord::decode(&bytes))
            .transpose()
            .map_err(|_| EngineError::InvalidRequest {
                detail: "undecodable transaction record".to_owned(),
            })?;
        let commit = match record {
            Some(record) if record.state == TxnState::Committed => true,
            Some(record) if record.state == TxnState::Aborted => false,
            _ => return Ok((0, 0)),
        };
        let mut done = 0usize;
        for key in keys {
            // Missing intents (already resolved) count as converged.
            if self.txn_finalize(key, txn, commit).is_ok() {
                done += 1;
            }
        }
        if commit { Ok((done, 0)) } else { Ok((0, done)) }
    }

    /// Attaches an absolute expiry; `false` when the key is absent.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on routing/transport failure.
    ///
    /// # Panics
    ///
    /// Panics if tablet execution returns an outcome shape that cannot result
    /// from the issued operation (an internal contract violation, never a
    /// runtime condition).
    pub fn expire_at(&self, key: &Key, expires_at: UnixMicros) -> Result<bool, EngineError> {
        match self.execute(
            key,
            Operation::ExpireAt {
                key: key.clone(),
                expires_at,
            },
        )? {
            OperationResult::ExpirySet { applied } => Ok(applied),
            unexpected => panic!("expire_at contract violated: {unexpected:?}"),
        }
    }

    /// Removes any expiry; `false` when the key has none or is absent.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on routing/transport failure.
    ///
    /// # Panics
    ///
    /// Panics if tablet execution returns an outcome shape that cannot result
    /// from the issued operation (an internal contract violation, never a
    /// runtime condition).
    pub fn persist_expiry(&self, key: &Key) -> Result<bool, EngineError> {
        match self.execute(key, Operation::PersistExpiry { key: key.clone() })? {
            OperationResult::ExpiryPersisted { removed } => Ok(removed),
            unexpected => panic!("persist_expiry contract violated: {unexpected:?}"),
        }
    }

    /// Reads a key's expiry (`None` when absent).
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on routing/transport failure.
    ///
    /// # Panics
    ///
    /// Panics if tablet execution returns an outcome shape that cannot result
    /// from the issued operation (an internal contract violation, never a
    /// runtime condition).
    pub fn get_expiry(&self, key: &Key) -> Result<Option<kivi_types::Expiry>, EngineError> {
        match self.execute(key, Operation::GetExpiry { key: key.clone() })? {
            OperationResult::Expiry(expiry) => Ok(expiry),
            unexpected => panic!("get_expiry contract violated: {unexpected:?}"),
        }
    }

    /// Reads a byte slice `[offset, offset + len)` clamped to the logical
    /// length (`None` when absent; possibly empty when past the end).
    /// Chunked roots resolve through the lane before slicing.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on routing/transport failure or wrong-type access.
    ///
    /// # Panics
    ///
    /// Panics if tablet execution returns an outcome shape that cannot result
    /// from the issued operation (an internal contract violation, never a
    /// runtime condition).
    pub fn get_range(
        &self,
        key: &Key,
        offset: u64,
        len: u64,
    ) -> Result<Option<Bytes>, EngineError> {
        match self.execute(
            key,
            Operation::GetRange {
                key: key.clone(),
                offset,
                len,
            },
        )? {
            OperationResult::Value(value) => Ok(value),
            unexpected => panic!("get_range contract violated: {unexpected:?}"),
        }
    }

    /// Reports the logical byte length (`None` when absent). Chunked roots
    /// answer from metadata without reading chunk payloads.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on routing/transport failure or wrong-type access.
    ///
    /// # Panics
    ///
    /// Panics if tablet execution returns an outcome shape that cannot result
    /// from the issued operation (an internal contract violation, never a
    /// runtime condition).
    pub fn bytes_length(&self, key: &Key) -> Result<Option<u64>, EngineError> {
        match self.execute(key, Operation::BytesLength { key: key.clone() })? {
            OperationResult::Length(value) => Ok(value),
            unexpected => panic!("bytes_length contract violated: {unexpected:?}"),
        }
    }

    /// Conditionally stores bytes, atomically at the owning tablet.
    /// Returns `(applied, version)`: `applied` names whether the condition
    /// held, `version` the new version when applied.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on routing/transport failure.
    ///
    /// # Panics
    ///
    /// Panics if tablet execution returns an outcome shape that cannot result
    /// from the issued operation (an internal contract violation, never a
    /// runtime condition).
    pub fn set_conditional(
        &self,
        key: &Key,
        value: Bytes,
        condition: kivi_state::SetCondition,
        expiry: kivi_state::ExpiryPolicy,
    ) -> Result<(bool, Option<kivi_state::ObjectVersion>), EngineError> {
        match self.execute(
            key,
            Operation::SetConditional {
                key: key.clone(),
                value,
                condition,
                expiry,
            },
        )? {
            OperationResult::ConditionalSet { applied, version } => Ok((applied, version)),
            unexpected => panic!("set_conditional contract violated: {unexpected:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_request() -> TabletRequest {
        let (respond, _) = bounded(RESPONSE_CAPACITY);
        TabletRequest {
            tablet: TabletId::from_u64(1),
            op: Operation::Exists {
                key: Key::from("k"),
            },
            now: UnixMicros::from_micros(0),
            identity: None,
            respond,
        }
    }

    #[test]
    fn send_errors_map_to_overloaded_or_worker_down() {
        let worker = WorkerId::from_u64(3);
        assert_eq!(
            map_send_error(&TrySendError::Full(dummy_request()), worker),
            EngineError::Overloaded
        );
        assert_eq!(
            map_send_error(&TrySendError::Disconnected(dummy_request()), worker),
            EngineError::WorkerDown { worker }
        );
    }

    #[test]
    fn panic_messages_extract_readable_payloads() {
        assert_eq!(panic_message(&"boom".to_owned()), Some("boom".to_owned()));
        let borrowed: Box<dyn std::any::Any + Send> = Box::new("bang");
        assert_eq!(panic_message(&*borrowed), Some("bang".to_owned()));
        assert_eq!(panic_message(&42u32), None);
    }

    #[test]
    fn error_conversions_preserve_sources_and_text() {
        use std::error::Error;
        let worker = WorkerId::from_u64(3);
        let mapped: EngineError = WorkerRequestError::UnknownTablet {
            tablet: TabletId::from_u64(1),
        }
        .into();
        assert_eq!(
            mapped,
            EngineError::UnknownTablet {
                tablet: TabletId::from_u64(1)
            }
        );
        assert!(mapped.source().is_none());
        let wrapped: EngineError = RoutingError::NoRoute.into();
        assert_eq!(wrapped, EngineError::Routing(RoutingError::NoRoute));
        assert!(wrapped.source().is_some());
        assert_eq!(
            EngineError::WorkerPanicked {
                worker,
                message: None
            }
            .to_string(),
            "worker 3 panicked"
        );
        assert_eq!(
            EngineError::WorkerPanicked {
                worker,
                message: Some("boom".to_owned()),
            }
            .to_string(),
            "worker 3 panicked: boom"
        );
    }
}
