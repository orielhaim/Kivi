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
use kivi_state::{Key, Operation, OperationResult, PartitionHasher};
use kivi_tablet::DirectorySnapshot;
use kivi_types::{ClusterId, NamespaceId, NodeId, NodeIncarnation, TabletId, UnixMicros, WorkerId};

use crate::clock::SystemClock;
use crate::routing::{Placement, RoutingError, RoutingSnapshot};
use crate::tablet::{LiveTablet, TabletError};
use crate::worker::{
    TabletRequest, WorkerControl, WorkerDurability, WorkerHandle, WorkerMetrics, WorkerRequestError,
};

/// Networked spawn outcome: handles, channel senders, and bound addresses.
type NetworkSpawn = (
    Vec<WorkerHandle>,
    Vec<crossbeam_channel::Sender<TabletRequest>>,
    Vec<std::net::SocketAddr>,
);

/// Merged recovery scan: per-tablet commit-ordered records plus lane
/// totals (segments scanned, torn-tail bytes truncated).
type RecoveryScan = (
    std::collections::HashMap<TabletId, Vec<(u64, kivi_durability::WalRecord)>>,
    u64,
    u64,
);

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
    /// A retried identity fell below the session floor.
    #[error("mutation identity expired below the session floor")]
    DedupExpired,
    /// The session holds too many unacknowledged outcomes on the tablet.
    #[error("session outcome window exhausted")]
    SessionOverloaded,
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
    senders: Vec<crossbeam_channel::Sender<TabletRequest>>,
}

/// The running engine: worker threads, routing, placement, lifecycle.
pub struct LocalEngine {
    shared: Arc<EngineShared>,
    workers: Vec<WorkerHandle>,
    bound: Vec<std::net::SocketAddr>,
    durability: Option<EngineDurability>,
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
    controls: Vec<(WorkerId, crossbeam_channel::Sender<WorkerControl>)>,
    endpoints: Vec<(WorkerId, std::net::SocketAddr)>,
    durability: Option<EngineDurability>,
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
            let live = LiveTablet::from_descriptor(tablet, authority)?;
            let index = usize::try_from(worker.as_u64())
                .map_err(|_| EngineError::UnknownWorker { worker })?;
            by_worker[index].push(live);
        }
        // Durable mode recovers BEFORE any worker spawns (and therefore
        // before any listener binds): a corrupt database never serves
        // partial state, and replayed tablets are simply the workers'
        // initial state.
        let (lanes, durability) = match &config.durability {
            DurabilityMode::Ephemeral => ((0..config.worker_count).map(|_| None).collect(), None),
            DurabilityMode::Durable(cfg) => {
                let (lanes, summary) = Self::recover_durable(
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
                    recovery: summary,
                };
                (lanes, Some(info))
            }
        };
        let routing = Arc::new(ArcSwap::from_pointee(routing));
        let (workers, senders, bound) = match &config.network {
            None => {
                let (workers, senders) = Self::spawn_channel_workers(
                    config.request_capacity,
                    config.worker_count,
                    by_worker,
                    lanes,
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
                )?;
                (workers, senders, bound)
            }
        };
        Ok(Self {
            shared: Arc::new(EngineShared {
                namespace: config.namespace,
                routing,
                senders,
            }),
            workers,
            bound,
            durability,
        })
    }

    /// Opens every WAL lane, repairs torn tails, and replays committed
    /// history into the freshly built tablets. Returns per-worker lane
    /// ownership plus the recovery summary for operators.
    ///
    /// Fails (refusing to serve) on corrupt history, structural gaps,
    /// records for tablets the startup directory does not cover, fencing
    /// mismatches, commit-chain breaks, or replay divergence.
    fn recover_durable(
        cfg: &DurableConfig,
        by_worker: &mut [Vec<LiveTablet>],
        routing: &RoutingSnapshot,
        namespace: NamespaceId,
        worker_count: usize,
    ) -> Result<(Vec<Option<WorkerDurability>>, RecoverySummary), EngineError> {
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
        let mut lanes =
            Self::open_recovery_lanes(&wal_dir, &lane_indexes, identity, cfg.segment_target_bytes)?;
        // Merge records per tablet across lanes (a future tablet may hold
        // history in several lanes; never assume one lane per tablet).
        let (per_tablet, segments_scanned, tail_truncated_bytes) =
            Self::scan_recovery_lanes(&mut lanes)?;
        // Validate against the startup directory, then replay in commit
        // order. Records for forgotten tablets fail startup (spec O):
        // discarding their history would be silent data loss.
        let (records_replayed, tablets_recovered) =
            Self::replay_records(by_worker, routing, namespace, per_tablet)?;
        let summary = RecoverySummary {
            records_replayed,
            segments_scanned,
            tail_truncated_bytes,
            tablets_recovered,
        };
        tracing::info!(
            records = summary.records_replayed,
            segments = summary.segments_scanned,
            truncated_bytes = summary.tail_truncated_bytes,
            tablets = summary.tablets_recovered,
            "durable recovery complete",
        );
        // Hand lanes to workers: exclusive per worker, or one shared lane.
        let shared = if cfg.shared_wal {
            let lane = lanes.remove(&0).expect("shared lane open");
            Some(std::sync::Arc::new(std::sync::Mutex::new(lane)))
        } else {
            None
        };
        let mut out = Vec::with_capacity(worker_count);
        for (index, lane_index) in lane_indexes.into_iter().enumerate() {
            let worker = WorkerId::from_u64(index as u64);
            let access = match &shared {
                Some(shared) => crate::worker::LaneAccess::Shared(Arc::clone(shared)),
                None => crate::worker::LaneAccess::Exclusive(
                    lanes.remove(&lane_index).expect("lane open"),
                ),
            };
            out.push(Some(WorkerDurability {
                namespace,
                worker,
                lane: access,
            }));
        }
        Ok((out, summary))
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

    /// Scans every lane exactly once (validating, repairing torn tails)
    /// and merges records per tablet. Lanes stay open for the handoff.
    fn scan_recovery_lanes(
        lanes: &mut std::collections::HashMap<u16, LocalWalLane>,
    ) -> Result<RecoveryScan, EngineError> {
        use kivi_durability::WalRecord;
        use std::collections::HashMap;
        let mut per_tablet: HashMap<TabletId, Vec<(u64, WalRecord)>> = HashMap::new();
        let mut segments_scanned = 0u64;
        let mut tail_truncated_bytes = 0u64;
        let mut lane_order: Vec<u16> = lanes.keys().copied().collect();
        lane_order.sort_unstable();
        for lane in lane_order {
            let recovery = lanes.get_mut(&lane).expect("lane present").recover()?;
            segments_scanned += recovery.segments_scanned;
            tail_truncated_bytes += recovery.truncated_bytes;
            for entry in recovery.records {
                per_tablet
                    .entry(entry.record.tablet())
                    .or_default()
                    .push((entry.record.commit().as_u64(), entry.record));
            }
        }
        Ok((per_tablet, segments_scanned, tail_truncated_bytes))
    }

    /// Validates merged records against the startup directory and replays
    /// them into the freshly built tablets in commit order.
    fn replay_records(
        by_worker: &mut [Vec<LiveTablet>],
        routing: &RoutingSnapshot,
        namespace: NamespaceId,
        per_tablet: std::collections::HashMap<TabletId, Vec<(u64, kivi_durability::WalRecord)>>,
    ) -> Result<(u64, u64), EngineError> {
        use kivi_durability::{RecoveryError, WalRecord};
        let mut records_replayed = 0u64;
        let mut tablets_recovered = 0u64;
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
        Ok((records_replayed, tablets_recovered))
    }

    /// Spawns channel-only workers (embedded path, no networking).
    fn spawn_channel_workers(
        request_capacity: usize,
        worker_count: usize,
        by_worker: Vec<Vec<LiveTablet>>,
        lanes: Vec<Option<WorkerDurability>>,
    ) -> (
        Vec<WorkerHandle>,
        Vec<crossbeam_channel::Sender<TabletRequest>>,
    ) {
        let mut workers = Vec::with_capacity(worker_count);
        let mut senders = Vec::with_capacity(worker_count);
        for ((index, tablets), durability) in by_worker.into_iter().enumerate().zip(lanes) {
            let id = WorkerId::from_u64(index as u64);
            let handle = WorkerHandle::spawn(id, tablets, request_capacity, durability);
            senders.push(handle_sender(&handle));
            workers.push(handle);
        }
        (workers, senders)
    }

    /// Spawns networked workers, collecting bound addresses into the shared
    /// endpoint registry once every worker reports ready.
    fn spawn_network_workers(
        network: &crate::net::EngineNetwork,
        request_capacity: usize,
        worker_count: usize,
        by_worker: Vec<Vec<LiveTablet>>,
        routing: &Arc<ArcSwap<RoutingSnapshot>>,
        lanes: Vec<Option<WorkerDurability>>,
    ) -> Result<NetworkSpawn, EngineError> {
        let endpoints = Arc::new(ArcSwap::new(Arc::new(crate::net::EndpointMap::new())));
        // Lanes present means durable mode (start() builds all-or-none);
        // the advertised capabilities follow the mode, never flags.
        let durable = lanes.iter().any(Option::is_some);
        debug_assert!(lanes.iter().all(|lane| lane.is_some() == durable));
        let advertised_caps = if durable {
            Capabilities::BASE_V1 | Capabilities::DURABLE_MUTATION_DEDUP
        } else {
            Capabilities::BASE_V1
        };
        let mut workers = Vec::with_capacity(worker_count);
        let mut senders = Vec::with_capacity(worker_count);
        let mut bound: Vec<(WorkerId, String)> = Vec::with_capacity(worker_count);
        let mut addrs: Vec<std::net::SocketAddr> = Vec::with_capacity(worker_count);
        for ((index, tablets), durability) in by_worker.into_iter().enumerate().zip(lanes) {
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
        let hash = PartitionHasher::V1
            .hash(self.shared.namespace, key.as_bytes())
            .ok_or(EngineError::HashAlgorithmUnsupported)?;
        routing.route(hash).map_err(|_| EngineError::NoRoute)
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
        for worker in &self.workers {
            let _ = worker.try_control(crate::worker::WorkerControl::Shutdown);
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

/// Extracts the current request sender out of a fresh handle. Handle senders
/// are not otherwise exposed; the engine keeps exactly one per worker.
fn handle_sender(handle: &WorkerHandle) -> crossbeam_channel::Sender<TabletRequest> {
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

impl LocalClient {
    /// Executes one typed operation against its owning tablet.
    fn execute(&self, key: &Key, op: Operation) -> Result<OperationResult, EngineError> {
        let routing = self.shared.routing.load();
        let hash = PartitionHasher::V1
            .hash(self.shared.namespace, key.as_bytes())
            .ok_or(EngineError::HashAlgorithmUnsupported)?;
        let (tablet, worker) = routing.route(hash).map_err(|_| EngineError::NoRoute)?;
        let index =
            usize::try_from(worker.as_u64()).map_err(|_| EngineError::UnknownWorker { worker })?;
        let sender = self
            .shared
            .senders
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
        receive
            .recv()
            .map_err(|_| EngineError::WorkerDown { worker })?
            .map_err(EngineError::from)
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
