//! Baseline production execution backend: one OS thread per worker.
//!
//! Each worker owns its thread, its [`LiveTablet`]s, and their object
//! stores. Requests arrive over a bounded channel; a separate bounded
//! control channel carries shutdown and maintenance so administration can
//! never deadlock behind a full request queue. The worker loop polls control
//! first on every iteration, then blocks in `select!` on both channels.
//!
//! This is deliberately not the final I/O reactor (no Tokio/Compio/Monoio,
//! no `io_uring`, no sockets): it establishes the ownership and execution
//! model — one mutable owner per tablet, synchronous execution, bounded
//! admission — that every future reactor must preserve. Tablet count and
//! request depth stay modest; there is one thread per worker, never per
//! tablet or per request.

use core::fmt;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crossbeam_channel::{Receiver, Sender, TrySendError, bounded, select};
use kivi_durability::{DurabilityError, LaneStats, StorageHealth, WorkerLaneStats};
use kivi_state::{ExpiryPolicy, Operation, OperationResult, SetCondition};
use kivi_types::{ManifestId, MutationIdentity, NamespaceId, TabletId, WallTimestamp, WorkerId};

use crate::chunk_lane::{
    ChunkLaneHandle, LargeSetSplit, RangeBase, SetRangePlan, StagingPins, plan_set_range,
    split_large_set,
};
use crate::commit::{BatchPolicy, CommitCoordinator, PendingEntry};
use crate::fabric::TabletFabric;
use crate::net::NetStartError;

use crate::tablet::{LiveTablet, TabletError};

/// Capacity of each worker's control channel. Control traffic is rare
/// (shutdown, sweeps); 16 slots cannot realistically fill, and every send
/// site handles `Full` explicitly anyway.
pub const CONTROL_CAPACITY: usize = 16;

/// Event-driven wakeup for the networked-worker embedded bridge.
///
/// The bridge lives on the Compio reactor thread, which cannot block on
/// the crossbeam ingress channels — so it used to poll them every 100 µs,
/// taxing every embedded wake (admin, tests, the RESP edge) with a fixed
/// latency floor and idle wakeups. Every ingress send now carries a
/// coalesced ping on this side channel: the bridge parks in
/// `recv().await` instead of a timer. The ping is a hint, never a
/// count — one ping per send attempt, collapsed to a single slot — and
/// the bridge drains the real channels on every wake, so a collapsed
/// ping loses nothing. Sends that find the bridge already awake (or a
/// channel-only worker with no bridge) have nowhere to notify and that
/// is fine. `async_channel` is runtime-agnostic (waker-based parking, no
/// runtime of its own), so this adds no async runtime to the engine.
#[derive(Debug, Clone)]
pub struct BridgeWaker {
    wake: async_channel::Sender<()>,
}

impl BridgeWaker {
    /// Creates a waker pair: the sender travels with every ingress
    /// handle, the receiver parks the networked bridge. Channel-only
    /// workers drop the receiver immediately (their blocking `select!`
    /// loop needs no wakeup); pings then fail closed and cost nothing.
    #[must_use]
    pub fn channel() -> (Self, async_channel::Receiver<()>) {
        let (wake, woken) = async_channel::bounded::<()>(1);
        (Self { wake }, woken)
    }

    /// Pings the bridge. Best-effort by construction: a full slot means
    /// a wake is already pending, a closed channel means no bridge is
    /// listening — both are correct without the ping.
    pub fn ping(&self) {
        let _ = self.wake.try_send(());
    }
}

/// Bundled ingress to one worker's request queue: the bounded crossbeam
/// sender (admission, backpressure) plus the bridge wakeup (promptness on
/// networked workers). Every send pings; the ping never blocks, fails, or
/// changes admission — ingress stays bounded by the channel capacity.
#[derive(Debug, Clone)]
pub struct RequestIngress {
    sender: Sender<TabletRequest>,
    bridge: BridgeWaker,
}

impl RequestIngress {
    /// Admits one request without blocking, then pings the bridge.
    ///
    /// # Errors
    ///
    /// Returns the `TrySendError` (full queue or disconnected worker) for
    /// the caller to map onto engine errors. The bridge ping fires
    /// regardless: a full queue still wants prompt draining.
    ///
    /// # Large error type
    ///
    /// The `Err` variant carries the whole request back (it must, so the
    /// caller can observe what was refused). The single caller maps it to
    /// the small [`crate::engine::EngineError`] immediately without storing it, so the
    /// stack cost never materializes.
    #[allow(clippy::result_large_err)]
    pub fn try_send(&self, request: TabletRequest) -> Result<(), TrySendError<TabletRequest>> {
        let outcome = self.sender.try_send(request);
        self.bridge.ping();
        outcome
    }
}

/// Bundled ingress to one worker's control queue: same wakeup contract as
/// [`RequestIngress`], guarding the maintenance/shutdown path.
#[derive(Debug, Clone)]
pub struct ControlIngress {
    sender: Sender<WorkerControl>,
    bridge: BridgeWaker,
}

impl ControlIngress {
    /// Sends one control message without blocking, then pings the bridge.
    ///
    /// # Errors
    ///
    /// Returns the `TrySendError` (full control queue or disconnected
    /// worker) for the caller to handle.
    pub fn try_send(&self, control: WorkerControl) -> Result<(), TrySendError<WorkerControl>> {
        let outcome = self.sender.try_send(control);
        self.bridge.ping();
        outcome
    }
}

/// Requests admitted per worker wake beyond the first (bounded drain so
/// concurrent arrivals share batch windows without starving control).
const REQUEST_DRAIN_PER_WAKE: usize = 63;

/// Admits one request plus whatever else is already queued (bounded):
/// concurrent arrivals join the same batch window instead of sealing
/// alone one wake at a time.
fn admit_drained(
    first: TabletRequest,
    tablets: &mut HashMap<TabletId, LiveTablet>,
    metrics: &core::cell::Cell<WorkerMetrics>,
    durability: &mut Option<WorkerDurability>,
    chunks: &WorkerChunks,
    fabric: &mut TabletFabric,
    requests: &Receiver<TabletRequest>,
) {
    handle_request(first, tablets, metrics, durability.as_mut(), chunks, fabric);
    for _ in 0..REQUEST_DRAIN_PER_WAKE {
        match requests.try_recv() {
            Ok(request) => handle_request(
                request,
                tablets,
                metrics,
                durability.as_mut(),
                chunks,
                fabric,
            ),
            Err(_) => break,
        }
    }
}

/// Work delivered to a worker thread.
#[derive(Debug)]
pub struct TabletRequest {
    /// Tablet that must execute the operation.
    pub tablet: TabletId,
    /// Typed operation to execute.
    pub op: Operation,
    /// Logical wall time captured at the client boundary.
    pub now: WallTimestamp,
    /// Retry identity with acknowledgement floor. Always `None` on the
    /// embedded path (callers share fate with the process); the native
    /// path fills it from the request.
    pub identity: Option<MutationIdentity>,
    /// Where the outcome goes (bounded to one message).
    pub respond: Sender<WorkerResponse>,
}

/// Outcome delivered back to the requesting client thread.
pub type WorkerResponse = Result<OperationResult, WorkerRequestError>;

/// Why a worker could not satisfy one request.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum WorkerRequestError {
    /// The worker owns no such tablet (routing/placement skew).
    #[error("worker owns no tablet {tablet}")]
    UnknownTablet {
        /// Requested tablet.
        tablet: TabletId,
    },
    /// Tablet logic rejected or failed the operation.
    #[error("{0}")]
    Tablet(#[source] TabletError),
    /// A retried identity fell below the session floor: gone, never
    /// re-executable.
    #[error("mutation identity expired below the session floor")]
    DedupExpired,
    /// The session holds too many unacknowledged outcomes on this tablet.
    #[error("session outcome window exhausted")]
    SessionOverloaded,
    /// The durable write under this request failed; logical state is
    /// untouched and reads continue where safe.
    #[error("durable write failed: {0}")]
    Storage(#[source] DurabilityError),
    /// Chunk staging or resolution failed; logical state is untouched
    /// (staging precedes admission, so nothing was admitted) or reads
    /// continue where safe.
    #[error("chunk fabric failed: {0}")]
    ChunkStore(#[source] kivi_chunk::ChunkError),
    /// Memory Fabric staging, promotion, or resolution failed; logical
    /// state is untouched (staging precedes admission) or reads continue
    /// where safe. Saturation surfaces upstream as backpressure, never
    /// as a logical error.
    #[error("memory fabric failed: {0}")]
    Fabric(#[source] crate::fabric::FabricError),
    /// The request is not admittable (absurd range, oversize patch):
    /// caller bug, state untouched, never persisted — fix and retry.
    #[error("invalid request: {detail}")]
    InvalidRequest {
        /// Human-readable reason (returned verbatim).
        detail: String,
    },
}

/// One worker's durable state: namespace plus the commit coordinator
/// (which owns the admission queue, batch formation, lane thread, and
/// metrics). `None` everywhere means ephemeral mode (no persistence, no
/// dedup, no coordinator).
#[derive(Debug)]
pub struct WorkerDurability {
    /// Namespace served (stamped into every record).
    pub namespace: NamespaceId,
    /// Owning worker (tracing and lane selection).
    pub worker: WorkerId,
    /// Batched commit pipeline with its dedicated durability lane.
    pub commit: CommitCoordinator,
}

/// How a worker reaches its WAL lane. The lane moves into the
/// coordinator's dedicated durability thread at startup; the `DataWorker`
/// never touches it again.
#[derive(Debug)]
pub enum LaneAccess {
    /// Private lane: no synchronization, the production default.
    Exclusive(kivi_durability::LocalWalLane),
    /// One lane shared by all workers (experiment arm): appends serialize
    /// on the mutex; recovery and ordering are unchanged.
    Shared(Arc<Mutex<kivi_durability::LocalWalLane>>),
}

impl WorkerDurability {
    /// Builds the durable state and spawns the lane thread.
    ///
    /// # Errors
    ///
    /// Returns [`DurabilityError`] when the batch policy is invalid or the
    /// lane thread cannot spawn.
    pub fn spawn(
        namespace: NamespaceId,
        worker: WorkerId,
        policy: BatchPolicy,
        lane: LaneAccess,
        initial_stats: WorkerLaneStats,
        seal_store: Option<crate::commit::FabricSealStore>,
    ) -> Result<Self, DurabilityError> {
        Ok(Self {
            namespace,
            worker,
            commit: CommitCoordinator::spawn(policy, lane, initial_stats, seal_store)?,
        })
    }

    /// Current coarse lane health (cached from the last seal outcome).
    #[must_use]
    pub fn health(&self) -> StorageHealth {
        self.commit.health()
    }

    /// Cumulative lane counters (authoritative snapshots ride home on
    /// every seal; see [`WorkerDurability::lane_stats`]).
    #[must_use]
    pub fn stats(&self) -> LaneStats {
        self.commit.lane_stats().stats
    }

    /// Full lane snapshot for the admin plane.
    #[must_use]
    pub fn lane_stats(&self) -> WorkerLaneStats {
        self.commit.lane_stats()
    }

    /// Direct maintenance access to the durability lane (checkpoint worker
    /// path; seals keep flowing through the coordinator).
    #[must_use]
    pub fn maintenance(&self) -> crate::commit::LaneMaintenance {
        self.commit.maintenance()
    }
}

/// Administrative control message. Handled ahead of queued requests.
#[derive(Debug)]
pub enum WorkerControl {
    /// Exit the worker loop after draining nothing further.
    Shutdown,
    /// Reclaim expired objects across all owned tablets (bounded per
    /// tablet), reporting the total reclaimed count.
    Sweep {
        /// Logical wall time for expiry comparison.
        now: WallTimestamp,
        /// Per-tablet reclaim bound.
        limit: usize,
        /// Where the reclaimed total goes.
        respond: Sender<usize>,
    },
    /// Report worker-local execution counters. Served from either path, so
    /// the direct-execution invariant stays observable without locks.
    Metrics {
        /// Where the snapshot goes.
        respond: Sender<WorkerMetrics>,
    },
    /// Report the WAL lane snapshot (`None` in ephemeral mode, which has
    /// no lane). Served from the control channel like metrics.
    DurabilityStats {
        /// Where the snapshot goes.
        respond: Sender<Option<WorkerLaneStats>>,
    },
    /// Report the commit-pipeline metrics (`None` in ephemeral mode).
    /// Served from the control channel like metrics.
    CommitMetrics {
        /// Where the snapshot goes.
        respond: Sender<Option<crate::commit::CommitMetricsSnapshot>>,
    },
    /// Capture one tablet's checkpoint view (commit boundary, committed
    /// state only). Served from the control channel; the clone cost is
    /// the measured capture pause, never unbounded work.
    CaptureTablet {
        /// Tablet to capture.
        tablet: TabletId,
        /// Where the view goes (`None` when this worker owns no such tablet).
        respond: Sender<Option<kivi_checkpoint::TabletSnapshot>>,
    },
    /// Report per-tablet applied cuts and dirty-band counts for
    /// checkpoint trigger evaluation. Cheap metadata only.
    TabletCheckpointStatus {
        /// Where the statuses go.
        respond: Sender<Vec<TabletCheckpointStatus>>,
    },
    /// Collect owned tablets' chunked-commit journals, pruning entries at
    /// or below each tablet's bound (its older retained cut). The
    /// checkpoint worker calls this after each publish; the returned
    /// remainder feeds chunk GC roots. Journal entries are `(commit,
    /// manifest)` pairs in commit order per tablet.
    ChunkedJournal {
        /// Per-tablet prune bounds `(tablet, prune_through)`. Tablets
        /// absent from the bounds reply unpruned (the worker cannot know
        /// their retention promise).
        bounds: Vec<(TabletId, u64)>,
        /// Where `(tablet, journal)` pairs go.
        respond: Sender<ChunkJournals>,
    },
    /// List every prepared transaction intent on owned tablets as
    /// `(tablet, intent)` pairs. The embedded resolver drives decisions
    /// from these; workers never resolve on their own.
    ListIntents {
        /// Where the intent pairs go.
        respond: Sender<Vec<(TabletId, kivi_state::TxnIntent)>>,
    },
    /// Report live fabric ids on owned tablets: live roots plus prepared
    /// intent references plus open/in-flight batch payloads. The
    /// checkpoint worker compacts fabric journals against the union
    /// across workers (plus band-referenced ids it reads itself), so
    /// records for ids nothing names anymore can be reclaimed.
    FabricLiveRoots {
        /// Where the live id set goes.
        respond: Sender<std::collections::HashSet<u64>>,
    },
    /// Report the worker's fabric observability: snapshot, per-tablet
    /// footprints, and queue depths. Served from the control channel;
    /// per-tablet attribution covers roots and prepared intents (sealed
    /// but unapplied payloads count in worker totals only).
    FabricStats {
        /// Where the report goes.
        respond: Sender<FabricWorkerReport>,
    },
    /// Apply a bounded memory control policy on the worker owner.
    SetMemoryPolicy {
        /// Policy to install.
        policy: kivi_memory::MemoryControlPolicy,
        /// Whether the policy was installed.
        respond: Sender<bool>,
    },
    /// Drain bounded access telemetry for the asynchronous observation fabric.
    MemorySignals {
        /// Where the object signals go.
        respond: Sender<Vec<(u64, kivi_memory::AccessSignals)>>,
    },
}

/// Per-worker fabric observability for the admin plane.
#[derive(Debug, Clone)]
pub struct FabricWorkerReport {
    /// Fabric counters and gauges.
    pub fabric: crate::fabric::FabricStatsSnapshot,
    /// Per-tablet footprints `(tablet, bytes by residence)`.
    pub tablets: Vec<(TabletId, kivi_memory::Footprint)>,
    /// Off-core planner queue depth `(queued, submitted)`.
    pub offcore_depth: (usize, usize),
    /// Background move queue depth.
    pub move_depth: usize,
    /// Arena pressure in `[0, 1]`.
    pub arena_pressure: f64,
}

/// Chunk lane access for one worker: the lane handle, the representation
/// policy, and the engine-global staging pins. Every worker holds one in
/// both modes (the ephemeral lane is TempDir-backed); large legacy `Set`s
/// convert here, on the tablet owner, before admission.
#[derive(Debug, Clone)]
pub struct WorkerChunks {
    /// This worker's chunk lane (staging, proofs, reads, stats).
    pub lane: ChunkLaneHandle,
    /// Values strictly larger stage as chunked roots; the rest stay inline.
    pub inline_threshold: u64,
    /// Security domain staged ids bind (from the namespace).
    pub domain: kivi_types::SecurityDomainId,
    /// Engine-global in-flight staging pins (GC roots while uncommitted).
    pub pins: Arc<StagingPins>,
    /// Redundancy fabric for best-effort protection of staged immutable
    /// bytes (`None` in ephemeral mode, which has no data directory).
    /// Protection never fails staging: local durability already proved the
    /// bytes, so a fabric miss only costs redundancy.
    pub redundancy: Option<crate::redundancy::EngineRedundancy>,
}

/// Collected chunked-commit journals: `(tablet, [(commit, manifest)])` in
/// commit order per tablet. Shared by the worker control plane and the
/// checkpoint worker's GC roots so the shape never drifts between them.
pub type ChunkJournals = Vec<(TabletId, Vec<(u64, ManifestId)>)>;

/// Per-tablet checkpoint trigger metadata: applied cut plus dirt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TabletCheckpointStatus {
    /// Tablet.
    pub tablet: TabletId,
    /// Last applied commit position.
    pub applied: u64,
    /// Dirty bands since the last capture.
    pub dirty: usize,
}

/// Worker-local execution counters, split by arrival path.
///
/// `channel_ops` counts requests served from the bounded cross-thread queue
/// (embedded `LocalClient` path); `direct_ops` counts requests executed
/// inline on the owning worker (native network fast path). Owned by the
/// worker thread; snapshots cross threads through the control channel.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorkerMetrics {
    /// Requests served from the cross-thread queue.
    pub channel_ops: u64,
    /// Requests executed directly on the owner worker.
    pub direct_ops: u64,
}

/// Owns one worker thread and its queue handles. Created by [`spawn`](WorkerHandle::spawn)
/// and joined exactly once via [`join`](WorkerHandle::join).
pub struct WorkerHandle {
    id: WorkerId,
    requests: Sender<TabletRequest>,
    control: Sender<WorkerControl>,
    bridge: BridgeWaker,
    thread: Option<JoinHandle<()>>,
}

impl WorkerHandle {
    /// Spawns the worker thread owning `tablets`, with a bounded request
    /// queue of `request_capacity`. `durability` is `None` in ephemeral
    /// mode; in durable mode the worker owns its WAL lane exclusively.
    /// `chunks` is this worker's chunk lane access plus the representation
    /// policy (both modes stage large values through it). `fabric` is
    /// this worker's Memory Fabric integration (medium values stage,
    /// resolve, and retire through it on the owner thread).
    ///
    /// # Panics
    ///
    /// Panics if the OS refuses to spawn the thread (resource exhaustion at
    /// startup is fatal — there is no worker to report through).
    #[must_use]
    pub fn spawn(
        id: WorkerId,
        tablets: Vec<LiveTablet>,
        request_capacity: usize,
        durability: Option<WorkerDurability>,
        chunks: WorkerChunks,
        fabric: TabletFabric,
    ) -> Self {
        let (request_tx, request_rx) = bounded::<TabletRequest>(request_capacity.max(1));
        let (control_tx, control_rx) = bounded::<WorkerControl>(CONTROL_CAPACITY);
        // Channel-only workers block in `select!` on the real channels, so
        // no bridge ever parks on this waker: the receiver drops here and
        // every ping fails closed.
        let (bridge, _unbridged) = BridgeWaker::channel();
        let thread = thread::Builder::new()
            .name(format!("kivi-worker-{}", id.as_u64()))
            .spawn(move || {
                run(
                    id, tablets, request_rx, control_rx, durability, chunks, fabric,
                );
            })
            .expect("worker thread spawns");
        Self {
            id,
            requests: request_tx,
            control: control_tx,
            bridge,
            thread: Some(thread),
        }
    }

    /// Returns the worker identity.
    #[must_use]
    pub const fn id(&self) -> WorkerId {
        self.id
    }

    /// Clones the bundled request ingress (engine keeps one per worker and
    /// hands clones to clients; the worker exits once all senders drop).
    /// Every send through the bundle pings the networked bridge.
    #[must_use]
    pub fn sender(&self) -> RequestIngress {
        RequestIngress {
            sender: self.requests.clone(),
            bridge: self.bridge.clone(),
        }
    }

    /// Clones the bundled control ingress (metrics queries, sweeps,
    /// shutdown). Every send through the bundle pings the bridge.
    #[must_use]
    pub fn control_sender(&self) -> ControlIngress {
        ControlIngress {
            sender: self.control.clone(),
            bridge: self.bridge.clone(),
        }
    }

    /// Admits one request without blocking: `Ok` when queued, `Err` when the
    /// bounded queue is full (fast `Overloaded`-style rejection upstream) or
    /// the worker is gone.
    ///
    /// # Errors
    ///
    /// Returns the `TrySendError` (full queue or disconnected worker) for
    /// the caller to map onto engine errors.
    ///
    /// # Large error type
    ///
    /// The `Err` variant carries the whole request back (it must, so the
    /// caller can observe what was refused). The single caller maps it to
    /// the small [`crate::engine::EngineError`] immediately without storing it, so the
    /// stack cost never materializes.
    #[allow(clippy::result_large_err)]
    pub fn try_send(&self, request: TabletRequest) -> Result<(), TrySendError<TabletRequest>> {
        self.requests.try_send(request)
    }

    /// Sends one control message without blocking, then pings the bridge
    /// (covers engine shutdown and test-harness control sends).
    ///
    /// # Errors
    ///
    /// Returns the `TrySendError` (full control queue or disconnected
    /// worker) for the caller to handle.
    pub fn try_control(&self, control: WorkerControl) -> Result<(), TrySendError<WorkerControl>> {
        let outcome = self.control.try_send(control);
        self.bridge.ping();
        outcome
    }

    /// Joins the worker thread. Returns the thread's outcome: `Ok(())` on
    /// clean exit, `Err` when the worker panicked (callers must surface this,
    /// never swallow it).
    ///
    /// # Errors
    ///
    /// Returns the panic payload when the worker thread panicked.
    pub fn join(&mut self) -> thread::Result<()> {
        if let Some(thread) = self.thread.take() {
            thread.join()
        } else {
            Ok(())
        }
    }

    /// Spawns a networked `DataWorker`: the thread runs a Compio runtime
    /// serving `net.listen` plus the embedded channel bridge, sharing one
    /// tablet map with zero locks. Returns the handle and the bound address
    /// (resolving ephemeral ports) once the worker reports ready.
    ///
    /// # Errors
    ///
    /// Returns [`NetStartError`] for affinity, runtime, bind, or handshake
    /// failures — always before serving. Partially started siblings clean
    /// themselves up when their handles drop (channel disconnect exits).
    pub fn spawn_net(
        id: WorkerId,
        tablets: Vec<LiveTablet>,
        request_capacity: usize,
        durability: Option<WorkerDurability>,
        chunks: WorkerChunks,
        fabric: TabletFabric,
        launch: crate::net::NetLaunch,
    ) -> Result<(Self, SocketAddr), NetStartError> {
        let (request_tx, request_rx) = bounded::<TabletRequest>(request_capacity.max(1));
        let (control_tx, control_rx) = bounded::<WorkerControl>(CONTROL_CAPACITY);
        // The bridge parks on this receiver (event-driven, no poll
        // quantum); every ingress send through the handle's bundles pings
        // the matching sender.
        let (bridge, woken) = BridgeWaker::channel();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let thread = thread::Builder::new()
            .name(format!("kivi-worker-{}", id.as_u64()))
            .spawn(move || {
                crate::net::run_net(
                    id, tablets, request_rx, control_rx, durability, chunks, fabric, launch, woken,
                    ready_tx,
                );
            })
            .map_err(|error| NetStartError::Startup(error.to_string()))?;
        match ready_rx.recv_timeout(crate::net::STARTUP_TIMEOUT) {
            Ok(Ok(addr)) => Ok((
                Self {
                    id,
                    requests: request_tx,
                    control: control_tx,
                    bridge,
                    thread: Some(thread),
                },
                addr,
            )),
            Ok(Err(error)) => {
                let _ = thread.join();
                Err(error)
            }
            Err(_) => Err(NetStartError::Startup(
                "worker startup handshake timed out".to_owned(),
            )),
        }
    }

    /// Whether the thread handle is still unjoined.
    #[must_use]
    pub fn is_joined(&self) -> bool {
        self.thread.is_none()
    }
}

impl fmt::Debug for WorkerHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerHandle")
            .field("id", &self.id)
            .field("joined", &self.is_joined())
            .finish_non_exhaustive()
    }
}

/// Worker thread body: own tablets, serve control first, then requests.
///
/// Channels arrive by value (not reference) because ownership must transfer
/// into the thread: the worker exits when the engine drops every sender.
///
/// In durable mode the loop also pumps the commit coordinator after every
/// wake, and parks boundedly (not indefinitely) while a batch is admitted
/// or in flight — the embedded client waits blocked in its rendezvous, so
/// an indefinite park would deadlock its reply. The park quantum only
/// bounds completion detection, never batching: batches seal on
/// drain/full/deadline/linger regardless.
///
/// Requests drain boundedly per wake (not one per wake) so concurrent
/// embedded clients accumulate into shared batches instead of sealing
/// alone one wake at a time.
#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
fn run(
    id: WorkerId,
    tablets: Vec<LiveTablet>,
    requests: Receiver<TabletRequest>,
    control: Receiver<WorkerControl>,
    mut durability: Option<WorkerDurability>,
    chunks: WorkerChunks,
    mut fabric: TabletFabric,
) {
    let _ = id;
    let metrics = core::cell::Cell::new(WorkerMetrics::default());
    let mut tablets: HashMap<TabletId, LiveTablet> = tablets
        .into_iter()
        .map(|tablet| (tablet.id(), tablet))
        .collect();
    // Pumps the coordinator once (no-op in ephemeral mode) plus amortized
    // fabric maintenance (planning slice, lane submissions, retire sweep).
    let pump = |tablets: &mut HashMap<TabletId, LiveTablet>,
                durability: Option<&mut WorkerDurability>,
                fabric: &mut TabletFabric| {
        if let Some(durable) = durability {
            let namespace = durable.namespace;
            durable.commit.poll(tablets, namespace, fabric);
        }
        fabric.maintenance_if_due();
    };
    loop {
        // Control fast path: administration never waits behind requests.
        if let Ok(message) = control.try_recv() {
            if !handle_control(
                message,
                &mut tablets,
                &metrics,
                durability.as_ref(),
                &mut fabric,
            ) {
                flush_for_shutdown(&mut tablets, durability.as_mut(), &mut fabric);
                break;
            }
            pump(&mut tablets, durability.as_mut(), &mut fabric);
            continue;
        }
        if durability
            .as_ref()
            .is_some_and(|durable| durable.commit.has_pending())
        {
            select! {
                recv(control) -> message => {
                    match message {
                        Ok(message) => {
                            if !handle_control(message, &mut tablets, &metrics, durability.as_ref(), &mut fabric) {
                                flush_for_shutdown(&mut tablets, durability.as_mut(), &mut fabric);
                                break;
                            }
                        }
                        // Engine gone: exit rather than serve a headless worker.
                        Err(_) => break,
                    }
                }
                recv(requests) -> message => {
                    match message {
                        Ok(request) => admit_drained(request, &mut tablets, &metrics, &mut durability, &chunks, &mut fabric, &requests),
                        // All clients gone: exit cleanly.
                        Err(_) => break,
                    }
                }
                default(crate::commit::COMPLETION_PARK) => {}
            }
        } else {
            select! {
                recv(control) -> message => {
                    match message {
                        Ok(message) => {
                            if !handle_control(message, &mut tablets, &metrics, durability.as_ref(), &mut fabric) {
                                flush_for_shutdown(&mut tablets, durability.as_mut(), &mut fabric);
                                break;
                            }
                        }
                        // Engine gone: exit rather than serve a headless worker.
                        Err(_) => break,
                    }
                }
                recv(requests) -> message => {
                    match message {
                        Ok(request) => admit_drained(request, &mut tablets, &metrics, &mut durability, &chunks, &mut fabric, &requests),
                        // All clients gone: exit cleanly.
                        Err(_) => break,
                    }
                }
            }
        }
        pump(&mut tablets, durability.as_mut(), &mut fabric);
    }
}

/// Flushes admitted work synchronously for shutdown: every admitted
/// request is answered (committed or explicitly failed) before the worker
/// exits, so shutdown never strands a blocked client.
fn flush_for_shutdown(
    tablets: &mut HashMap<TabletId, LiveTablet>,
    durability: Option<&mut WorkerDurability>,
    fabric: &mut TabletFabric,
) {
    if let Some(durable) = durability {
        let namespace = durable.namespace;
        durable.commit.flush_blocking(tablets, namespace, fabric);
    }
}

/// Live fabric ids per owned tablet: committed roots plus prepared
/// intent references. Shared by journal GC marking and observability;
/// in-flight batch payloads (sealed but unapplied) join at the call
/// site that can see the commit coordinator.
fn tablet_fabric_roots(
    tablets: &HashMap<TabletId, LiveTablet>,
) -> HashMap<TabletId, std::collections::HashSet<u64>> {
    let mut out: HashMap<TabletId, std::collections::HashSet<u64>> = HashMap::new();
    for live in tablets.values() {
        let entry = out.entry(live.id()).or_default();
        for (_, object) in live.store().snapshot_entries() {
            if let Some(reference) = object.fabric_ref() {
                entry.insert(reference.id);
            }
        }
        for intent in live.store().snapshot_intents() {
            if let kivi_state::TxnWriteKind::PutFabric { fabric_id, .. } = &intent.write.kind {
                entry.insert(*fabric_id);
            }
        }
    }
    out
}

/// Live fabric ids across owned tablets for journal GC: committed
/// roots plus prepared intent references plus open/in-flight batch
/// payloads. The union is exactly what journal GC must keep.
fn fabric_live_roots(
    tablets: &HashMap<TabletId, LiveTablet>,
    durability: Option<&WorkerDurability>,
) -> std::collections::HashSet<u64> {
    // Roots name committed ids; prepared intents name staged ids
    // whose seal rides an upcoming finalize; open/in-flight batch
    // payloads name sealed-but-unapplied ids.
    let mut live: std::collections::HashSet<u64> = tablet_fabric_roots(tablets)
        .into_values()
        .flatten()
        .collect();
    if let Some(durable) = durability {
        live.extend(durable.commit.pending_fabric_ids());
    }
    live
}

/// Builds one worker's fabric observability report: snapshot,
/// per-tablet footprints, and queue depths.
fn fabric_stats_report(
    tablets: &HashMap<TabletId, LiveTablet>,
    fabric: &mut crate::fabric::TabletFabric,
) -> FabricWorkerReport {
    let roots = tablet_fabric_roots(tablets);
    let snapshot = fabric.stats_snapshot();
    let memory = fabric.fabric_mut();
    let tablets_report = roots
        .iter()
        .map(|(tablet, ids)| (*tablet, memory.footprint_for(ids)))
        .collect();
    FabricWorkerReport {
        fabric: snapshot,
        tablets: tablets_report,
        offcore_depth: memory.offcore_depth(),
        move_depth: memory.move_depth(),
        arena_pressure: memory.pressure(),
    }
}

/// Handles one control message. Returns `false` when the worker must exit.
pub(crate) fn handle_control(
    message: WorkerControl,
    tablets: &mut HashMap<TabletId, LiveTablet>,
    metrics: &core::cell::Cell<WorkerMetrics>,
    durability: Option<&WorkerDurability>,
    fabric: &mut crate::fabric::TabletFabric,
) -> bool {
    match message {
        WorkerControl::Shutdown => false,
        WorkerControl::Sweep {
            now,
            limit,
            respond,
        } => {
            // A sweep physically deletes expired keys outside the commit
            // pipeline, so it must skip every key with a staged prediction
            // (queued, open, or in flight): deleting one would diverge the
            // post-durability verification. Skipped keys wait for the next
            // sweep; deletion is reclamation, never correctness.
            let touched = durability.map(|durable| durable.commit.touched_keys());
            let mut reclaimed = 0usize;
            for tablet in tablets.values_mut() {
                reclaimed += tablet
                    .sweep_expired_except(now, limit, touched.as_ref())
                    .len();
            }
            let _ = respond.try_send(reclaimed);
            true
        }
        WorkerControl::Metrics { respond } => {
            let _ = respond.try_send(metrics.get());
            true
        }
        WorkerControl::DurabilityStats { respond } => {
            let _ = respond.try_send(durability.map(WorkerDurability::lane_stats));
            true
        }
        WorkerControl::CommitMetrics { respond } => {
            let _ = respond.try_send(durability.map(|durable| durable.commit.metrics_snapshot()));
            true
        }
        WorkerControl::CaptureTablet { tablet, respond } => {
            let view = tablets
                .get_mut(&tablet)
                .map(|live| live.capture_view(capture_namespace(durability)));
            let _ = respond.try_send(view);
            true
        }
        WorkerControl::TabletCheckpointStatus { respond } => {
            let statuses = tablets
                .values()
                .map(|live| TabletCheckpointStatus {
                    tablet: live.id(),
                    applied: live.applied_commit().as_u64(),
                    dirty: live
                        .dirty_bands()
                        .map_or(0, kivi_checkpoint::BandDirty::dirty_count),
                })
                .collect();
            let _ = respond.try_send(statuses);
            true
        }
        WorkerControl::ChunkedJournal { bounds, respond } => {
            let bounds: std::collections::HashMap<TabletId, u64> = bounds.into_iter().collect();
            let journals = tablets
                .values_mut()
                .filter(|live| bounds.contains_key(&live.id()))
                .map(|live| {
                    let bound = bounds.get(&live.id()).copied().unwrap_or(0);
                    (live.id(), live.collect_chunked_journal(bound))
                })
                .collect();
            let _ = respond.try_send(journals);
            true
        }
        WorkerControl::ListIntents { respond } => {
            let intents = tablets
                .values()
                .flat_map(|live| {
                    live.pending_intents()
                        .into_iter()
                        .map(|intent| (live.id(), intent))
                })
                .collect();
            let _ = respond.try_send(intents);
            true
        }
        WorkerControl::FabricLiveRoots { respond } => {
            let _ = respond.try_send(fabric_live_roots(tablets, durability));
            true
        }
        WorkerControl::FabricStats { respond } => {
            let _ = respond.try_send(fabric_stats_report(tablets, fabric));
            true
        }
        WorkerControl::SetMemoryPolicy { policy, respond } => {
            fabric.set_control_policy(policy);
            let _ = respond.try_send(true);
            true
        }
        WorkerControl::MemorySignals { respond } => {
            let _ = respond.try_send(fabric.drain_access_signals());
            true
        }
    }
}

/// Peeks the tablet-visible base a range patch applies against: one
/// synchronous store read, no mutation. Unknown tablets read as absent —
/// the plan outcome is then unused because downstream answers
/// `UnknownTablet` before preparing anything.
pub(crate) fn peek_range_base(
    tablets: &HashMap<TabletId, LiveTablet>,
    tablet: TabletId,
    key: &kivi_state::Key,
    now: WallTimestamp,
) -> RangeBase {
    let Some(live) = tablets.get(&tablet) else {
        return RangeBase::Absent;
    };
    match live.store().get(key, now) {
        None => RangeBase::Absent,
        Some(object) => match object.value() {
            kivi_state::LogicalValue::Bytes(bytes) => RangeBase::Inline(bytes.clone()),
            kivi_state::LogicalValue::StrictCounter(_)
            | kivi_state::LogicalValue::CommutativeCounter(_)
            | kivi_state::LogicalValue::BoundedCounter(_)
            | kivi_state::LogicalValue::Semaphore(_)
            | kivi_state::LogicalValue::Lease(_)
            | kivi_state::LogicalValue::StreamShard(_) => RangeBase::NonBytes,
            kivi_state::LogicalValue::Chunked(chunked) => RangeBase::Chunked {
                manifest: chunked.manifest,
                logical_len: chunked.logical_len,
            },
            kivi_state::LogicalValue::Fabric(fabric) => RangeBase::Fabric {
                fabric_id: fabric.id,
                logical_len: fabric.logical_len,
                version: fabric.version,
            },
        },
    }
}

/// One staging job's input: a full value to chunk, or a chunked base plus
/// a patch for the lane's prefix-reuse splice.
#[derive(Debug)]
enum StageWork {
    /// Chunk a full value (large `Set`, or a restaged range result).
    Value(bytes::Bytes),
    /// Chunk a full conditional value, preserving its condition/policy.
    ConditionalValue {
        /// Full logical bytes to chunk.
        value: bytes::Bytes,
        /// Presence condition evaluated atomically at prepare.
        condition: kivi_state::SetCondition,
        /// Expiry policy resolved at prepare.
        expiry: kivi_state::ExpiryPolicy,
    },
    /// Splice a patch into a chunked base without materializing it.
    Splice {
        /// Manifest addressing the base value.
        manifest: ManifestId,
        /// Total logical bytes the root claims.
        logical_len: u64,
        /// Logical byte index the patch overwrites from.
        offset: u64,
        /// Bytes written starting at `offset`.
        patch: bytes::Bytes,
    },
}

/// Stages one value-or-splice on the owner's lane (blocking lane call —
/// plain worker threads only) and rewrites it as a small `SetChunked`
/// with its pin guard. Answers the failure and reports `Err(())` with
/// state untouched when staging fails.
fn stage_value_work(
    key: &kivi_state::Key,
    work: StageWork,
    chunks: &WorkerChunks,
    metrics: &core::cell::Cell<WorkerMetrics>,
    respond: &Sender<WorkerResponse>,
) -> Result<(Operation, Option<crate::chunk_lane::PinnedUpload>), ()> {
    fn failed(
        metrics: &core::cell::Cell<WorkerMetrics>,
        respond: &Sender<WorkerResponse>,
        error: WorkerRequestError,
    ) -> Result<(Operation, Option<crate::chunk_lane::PinnedUpload>), ()> {
        let mut snapshot = metrics.get();
        snapshot.channel_ops += 1;
        metrics.set(snapshot);
        let _ = respond.try_send(Err(error));
        Err(())
    }
    match work {
        StageWork::Value(value) => {
            // Pins open before the first append and close after commit
            // (the guard rides the coordinator entry below): GC always
            // sees staged data as pinned or journaled, never neither.
            let mut guard =
                match crate::chunk_lane::PinnedUpload::begin(&chunks.pins, chunks.domain, &value) {
                    Ok(guard) => guard,
                    Err(error) => {
                        return failed(metrics, respond, WorkerRequestError::ChunkStore(error));
                    }
                };
            let staged = stage_plain_value(value, chunks, metrics, respond)?;
            guard.set_manifest(staged.manifest);
            Ok((
                Operation::SetChunked {
                    key: key.clone(),
                    manifest: staged.manifest,
                    logical_len: staged.logical_len,
                },
                Some(guard),
            ))
        }
        StageWork::ConditionalValue {
            value,
            condition,
            expiry,
        } => {
            let mut guard =
                match crate::chunk_lane::PinnedUpload::begin(&chunks.pins, chunks.domain, &value) {
                    Ok(guard) => guard,
                    Err(error) => {
                        return failed(metrics, respond, WorkerRequestError::ChunkStore(error));
                    }
                };
            let staged = stage_plain_value(value, chunks, metrics, respond)?;
            guard.set_manifest(staged.manifest);
            Ok((
                Operation::SetConditionalChunked {
                    key: key.clone(),
                    manifest: staged.manifest,
                    logical_len: staged.logical_len,
                    condition,
                    expiry,
                },
                Some(guard),
            ))
        }
        StageWork::Splice {
            manifest,
            logical_len,
            offset,
            patch,
        } => match chunks
            .lane
            .splice_range_blocking(manifest, logical_len, offset, patch)
        {
            Ok(staged) => {
                let guard = crate::chunk_lane::PinnedUpload::staged(&chunks.pins, &staged);
                // Best-effort redundancy for the spliced sequence (the
                // manifest is always covered; chunks need a lane re-read).
                protect_spliced_value(chunks, &staged);
                // A splice preserves the live expiry like the inline path
                // does: partial writes touch bytes, never the TTL.
                Ok((
                    Operation::SetConditionalChunked {
                        key: key.clone(),
                        manifest: staged.manifest,
                        logical_len: staged.logical_len,
                        condition: SetCondition::Always,
                        expiry: ExpiryPolicy::Keep,
                    },
                    Some(guard),
                ))
            }
            Err(error) => failed(metrics, respond, WorkerRequestError::ChunkStore(error)),
        },
    }
}

/// Stages one plain value on the owner's lane (blocking lane call — plain
/// worker threads only) and submits best-effort protection of the staged
/// bytes to the redundancy lane (fire-and-forget, never blocking). Answers
/// the failure and reports `Err(())` with state untouched when staging
/// fails; fabric misses never fail staging (local durability already proved
/// the bytes).
fn stage_plain_value(
    value: bytes::Bytes,
    chunks: &WorkerChunks,
    metrics: &core::cell::Cell<WorkerMetrics>,
    respond: &Sender<WorkerResponse>,
) -> Result<crate::chunk_lane::StagedValue, ()> {
    // `Bytes` clones share the allocation: the copy below only feeds the
    // best-effort lane submission after staging.
    let bytes = value.clone();
    match chunks.lane.stage_value_blocking(value) {
        Ok(staged) => {
            crate::redundancy::protect_staged_value(
                chunks.redundancy.as_ref(),
                chunks.domain,
                &bytes,
                &staged,
            );
            Ok(staged)
        }
        Err(error) => {
            let mut snapshot = metrics.get();
            snapshot.channel_ops += 1;
            metrics.set(snapshot);
            let _ = respond.try_send(Err(WorkerRequestError::ChunkStore(error)));
            Err(())
        }
    }
}

/// Best-effort redundancy for a spliced chunk sequence: the lane owns the
/// fresh bytes (no caller value survives the splice), so one lane job
/// re-reads the resolved value on the redundancy lane thread and protects
/// from it. The manifest is always protected from its canonical encoding; a
/// re-read miss only skips chunks. Never blocks staging and never fails it:
/// local durability already proved the bytes.
fn protect_spliced_value(chunks: &WorkerChunks, staged: &crate::chunk_lane::StagedValue) {
    let Some(fabric) = chunks.redundancy.as_ref() else {
        return;
    };
    fabric.submit_spliced_value(chunks.lane.clone(), chunks.domain, staged);
}

/// Maps a fabric failure onto the request error surface: saturation is
/// retryable backpressure, everything else fails the single operation
/// closed with logical state untouched.
fn map_fabric_error(error: crate::fabric::FabricError) -> WorkerRequestError {
    match error {
        crate::fabric::FabricError::Overloaded => WorkerRequestError::SessionOverloaded,
        other => WorkerRequestError::Fabric(other),
    }
}

/// Stages one transactional write's medium inline payload into the
/// fabric, rewriting it as a `PutFabric` reference. Carried `PutFabric`
/// references pass through for proving below. Returns the rewritten
/// write plus its seal when staging happened; `None` means the request
/// already answered with state untouched.
#[allow(clippy::too_many_arguments)]
fn stage_txn_write(
    tablet: TabletId,
    write: kivi_state::TxnWrite,
    chunks: &WorkerChunks,
    fabric: &mut crate::fabric::TabletFabric,
    metrics: &core::cell::Cell<WorkerMetrics>,
    respond: &Sender<WorkerResponse>,
) -> Result<Option<(kivi_state::TxnWrite, Option<crate::fabric::StagedSeal>)>, ()> {
    fn failed(
        metrics: &core::cell::Cell<WorkerMetrics>,
        respond: &Sender<WorkerResponse>,
        error: WorkerRequestError,
    ) -> Result<Option<(kivi_state::TxnWrite, Option<crate::fabric::StagedSeal>)>, ()> {
        let mut snapshot = metrics.get();
        snapshot.channel_ops += 1;
        metrics.set(snapshot);
        let _ = respond.try_send(Err(error));
        Err(())
    }
    match write.kind {
        kivi_state::TxnWriteKind::Put(value)
            if value.len() > crate::fabric::FABRIC_INLINE_MAX
                && (value.len() as u64) <= chunks.inline_threshold =>
        {
            let (fabric_id, logical_len) = match fabric.stage(tablet, value.clone(), true) {
                Ok(staged) => staged,
                Err(error) => return failed(metrics, respond, map_fabric_error(error)),
            };
            Ok(Some((
                kivi_state::TxnWrite {
                    key: write.key.clone(),
                    kind: kivi_state::TxnWriteKind::PutFabric {
                        fabric_id,
                        logical_len,
                        version: 0,
                    },
                    expect: write.expect,
                },
                Some(crate::fabric::StagedSeal {
                    fabric_id,
                    key: write.key.clone(),
                    bytes: value,
                }),
            )))
        }
        _ => Ok(Some((write, None))),
    }
}

/// Attaches a commit-finalize's prepared intent bytes as a seal payload:
/// the intent's staged fabric id re-seals under the predicted
/// authoritative version at prepare time, superseding the staging record
/// from `TxnPrepare`. Reads through any valid residence (prepared ids
/// stay pinned arena-resident, so this is synchronous on the common
/// path). Aborts, non-fabric intents, and missing intents attach nothing
/// (prepare resolves those terminally). Returns `None` when the request
/// already answered: a dangling intent reference fails closed, never
/// commits.
fn seal_finalize_intent(
    op: Operation,
    tablets: &HashMap<TabletId, LiveTablet>,
    tablet: TabletId,
    fabric: &mut crate::fabric::TabletFabric,
    metrics: &core::cell::Cell<WorkerMetrics>,
    respond: &Sender<WorkerResponse>,
) -> Option<(
    Operation,
    Vec<crate::chunk_lane::PinnedUpload>,
    Vec<crate::fabric::StagedSeal>,
)> {
    let (commit, key) = match &op {
        Operation::TxnFinalize { commit, key, .. } => (*commit, key.clone()),
        _ => return Some((op, Vec::new(), Vec::new())),
    };
    if !commit {
        // Aborts publish nothing; apply releases the prepared pin.
        return Some((op, Vec::new(), Vec::new()));
    }
    let fabric_id = tablets
        .get(&tablet)
        .and_then(|live| live.store().intent_for_key(&key))
        .and_then(|intent| match &intent.write.kind {
            kivi_state::TxnWriteKind::PutFabric { fabric_id, .. } => Some(*fabric_id),
            _ => None,
        });
    let Some(fabric_id) = fabric_id else {
        // No fabric intent here: prepare resolves missing/conflicting
        // intents terminally without touching the fabric.
        return Some((op, Vec::new(), Vec::new()));
    };
    match fabric.read_anywhere(fabric_id) {
        Ok(bytes) => Some((
            op,
            Vec::new(),
            vec![crate::fabric::StagedSeal {
                fabric_id,
                key: key.clone(),
                bytes,
            }],
        )),
        Err(error) => {
            let mut snapshot = metrics.get();
            snapshot.channel_ops += 1;
            metrics.set(snapshot);
            let _ = respond.try_send(Err(map_fabric_error(error)));
            None
        }
    }
}

/// Proves every fabric reference a transaction carries names live local
/// bytes (staged uploads and committed materializations alike): a
/// dangling root is rejected loudly before any participant may say
/// Prepared, so a later Commit can never meet missing payload.
fn verify_txn_fabric_refs(
    op: &Operation,
    fabric: &mut crate::fabric::TabletFabric,
    metrics: &core::cell::Cell<WorkerMetrics>,
    respond: &Sender<WorkerResponse>,
) -> Result<(), ()> {
    fn failed(
        metrics: &core::cell::Cell<WorkerMetrics>,
        respond: &Sender<WorkerResponse>,
        detail: String,
    ) -> Result<(), ()> {
        let mut snapshot = metrics.get();
        snapshot.channel_ops += 1;
        metrics.set(snapshot);
        let _ = respond.try_send(Err(WorkerRequestError::InvalidRequest { detail }));
        Err(())
    }
    let refs: Vec<u64> = match op {
        Operation::TxnPrepare { write, .. } => match &write.kind {
            kivi_state::TxnWriteKind::PutFabric { fabric_id, .. } => vec![*fabric_id],
            _ => Vec::new(),
        },
        Operation::TxnCommitLocal { writes, .. } => writes
            .iter()
            .filter_map(|write| match &write.kind {
                kivi_state::TxnWriteKind::PutFabric { fabric_id, .. } => Some(*fabric_id),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    for fabric_id in refs {
        if fabric.read_anywhere(fabric_id).is_err() {
            return failed(
                metrics,
                respond,
                format!("transactional fabric write references unavailable payload: {fabric_id}"),
            );
        }
    }
    Ok(())
}

/// Proves every chunked reference a transaction carries names durable
/// local payload (staged uploads and committed packs alike): a dangling
/// root is rejected loudly before any participant may say Prepared, so a
/// later Commit can never meet missing payload.
fn verify_txn_chunk_refs(
    op: &Operation,
    chunks: &WorkerChunks,
    metrics: &core::cell::Cell<WorkerMetrics>,
    respond: &Sender<WorkerResponse>,
) -> Result<(), ()> {
    fn failed(
        metrics: &core::cell::Cell<WorkerMetrics>,
        respond: &Sender<WorkerResponse>,
        detail: String,
    ) -> Result<(), ()> {
        let mut snapshot = metrics.get();
        snapshot.channel_ops += 1;
        metrics.set(snapshot);
        let _ = respond.try_send(Err(WorkerRequestError::InvalidRequest { detail }));
        Err(())
    }
    let refs: Vec<(kivi_types::ManifestId, u64)> = match op {
        Operation::TxnPrepare { write, .. } => match &write.kind {
            kivi_state::TxnWriteKind::PutChunked {
                manifest,
                logical_len,
            } => vec![(*manifest, *logical_len)],
            _ => Vec::new(),
        },
        Operation::TxnCommitLocal { writes, .. } => writes
            .iter()
            .filter_map(|write| match &write.kind {
                kivi_state::TxnWriteKind::PutChunked {
                    manifest,
                    logical_len,
                } => Some((*manifest, *logical_len)),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    for (manifest, logical_len) in refs {
        if let Err(error) = chunks.lane.check_root_blocking(manifest, logical_len) {
            return failed(
                metrics,
                respond,
                format!("transactional chunked write references unavailable payload: {error}"),
            );
        }
    }
    Ok(())
}

/// Stages a restaged range result, preserving expiry exactly like the
/// inline splice does (partial writes touch bytes, never the TTL).
fn stage_restaged(
    key: &kivi_state::Key,
    value: bytes::Bytes,
    chunks: &WorkerChunks,
    metrics: &core::cell::Cell<WorkerMetrics>,
    respond: &Sender<WorkerResponse>,
) -> Result<(Operation, Option<crate::chunk_lane::PinnedUpload>), ()> {
    stage_value_work(
        key,
        StageWork::ConditionalValue {
            value,
            condition: SetCondition::Always,
            expiry: ExpiryPolicy::Keep,
        },
        chunks,
        metrics,
        respond,
    )
}

/// Stages one medium value into the fabric (blocking lane-free call —
/// plain worker threads only; the reactor bridge defers through its
/// continuation queue instead) and rewrites it as a small `SetFabric`
/// root. Answers the failure and reports `Err(())` with state untouched
/// when staging fails (usually bounded-capacity backpressure).
#[allow(clippy::too_many_arguments)]
fn stage_fabric_value(
    tablet: TabletId,
    key: &kivi_state::Key,
    value: bytes::Bytes,
    hot: bool,
    fabric: &mut crate::fabric::TabletFabric,
    metrics: &core::cell::Cell<WorkerMetrics>,
    respond: &Sender<WorkerResponse>,
) -> Result<(Operation, crate::fabric::StagedSeal), ()> {
    fn failed(
        metrics: &core::cell::Cell<WorkerMetrics>,
        respond: &Sender<WorkerResponse>,
        error: WorkerRequestError,
    ) -> Result<(Operation, crate::fabric::StagedSeal), ()> {
        let mut snapshot = metrics.get();
        snapshot.channel_ops += 1;
        metrics.set(snapshot);
        let _ = respond.try_send(Err(error));
        Err(())
    }
    let (fabric_id, logical_len) = match fabric.stage(tablet, value.clone(), hot) {
        Ok(staged) => staged,
        Err(crate::fabric::FabricError::Overloaded) => {
            return failed(metrics, respond, WorkerRequestError::SessionOverloaded);
        }
        Err(_) => {
            return failed(
                metrics,
                respond,
                WorkerRequestError::InvalidRequest {
                    detail: "memory fabric staging unavailable".to_owned(),
                },
            );
        }
    };
    Ok((
        Operation::SetFabric {
            key: key.clone(),
            fabric_id,
            logical_len,
            version: 0,
        },
        crate::fabric::StagedSeal {
            fabric_id,
            key: key.clone(),
            bytes: value,
        },
    ))
}

/// Stages one medium conditional value, preserving condition and expiry
/// policy for atomic evaluation at prepare.
#[allow(clippy::too_many_arguments)]
fn stage_fabric_conditional(
    tablet: TabletId,
    key: &kivi_state::Key,
    value: bytes::Bytes,
    condition: kivi_state::SetCondition,
    expiry: kivi_state::ExpiryPolicy,
    fabric: &mut crate::fabric::TabletFabric,
    metrics: &core::cell::Cell<WorkerMetrics>,
    respond: &Sender<WorkerResponse>,
) -> Result<(Operation, crate::fabric::StagedSeal), ()> {
    fn failed(
        metrics: &core::cell::Cell<WorkerMetrics>,
        respond: &Sender<WorkerResponse>,
        error: WorkerRequestError,
    ) -> Result<(Operation, crate::fabric::StagedSeal), ()> {
        let mut snapshot = metrics.get();
        snapshot.channel_ops += 1;
        metrics.set(snapshot);
        let _ = respond.try_send(Err(error));
        Err(())
    }
    let (fabric_id, logical_len) = match fabric.stage(tablet, value.clone(), true) {
        Ok(staged) => staged,
        Err(crate::fabric::FabricError::Overloaded) => {
            return failed(metrics, respond, WorkerRequestError::SessionOverloaded);
        }
        Err(_) => {
            return failed(
                metrics,
                respond,
                WorkerRequestError::InvalidRequest {
                    detail: "memory fabric staging unavailable".to_owned(),
                },
            );
        }
    };
    Ok((
        Operation::SetConditionalFabric {
            key: key.clone(),
            fabric_id,
            logical_len,
            version: 0,
            condition,
            expiry,
        },
        crate::fabric::StagedSeal {
            fabric_id,
            key: key.clone(),
            bytes: value,
        },
    ))
}

/// Resolves one prepared outcome for the channel path: fabric references
/// serve through the lane (blocking promotion on plain threads), every
/// other outcome passes through untouched. Synchronous with the prepare
/// on this thread, so the root cannot move underneath the resolve; a
/// mismatch fails closed as a retryable error. `GetRange` windows slice
/// the resolved bytes, exactly like the durable fast path.
fn resolve_channel_outcome(
    tablet: TabletId,
    key: &kivi_state::Key,
    now: WallTimestamp,
    op: &Operation,
    outcome: OperationResult,
    tablets: &HashMap<TabletId, LiveTablet>,
    fabric: &mut crate::fabric::TabletFabric,
) -> WorkerResponse {
    let OperationResult::FabricValue {
        fabric_id, version, ..
    } = outcome
    else {
        return Ok(outcome);
    };
    let range = match op {
        Operation::GetRange { offset, len, .. } => Some((*offset, *len)),
        _ => None,
    };
    let current = tablets
        .get(&tablet)
        .and_then(|live| live.store().get(key, now));
    let reference = kivi_state::FabricRef {
        id: fabric_id,
        logical_len: current
            .as_ref()
            .and_then(|object| object.fabric_ref())
            .map_or(0, |live| live.logical_len),
        version,
    };
    match fabric.promote_blocking(&reference, current) {
        Ok(bytes) => {
            let shaped = match range {
                Some((offset, len)) => kivi_state::slice_range(&bytes, offset, len),
                None => bytes,
            };
            Ok(OperationResult::Value(Some(shaped)))
        }
        Err(crate::fabric::FabricError::Overloaded) => Err(WorkerRequestError::SessionOverloaded),
        Err(error) => Err(WorkerRequestError::Fabric(error)),
    }
}

/// Namespace for checkpoint captures (durable workers serve exactly one).
fn capture_namespace(durability: Option<&WorkerDurability>) -> NamespaceId {
    durability.map_or(NamespaceId::from_u64(0), |durable| durable.namespace)
}

/// Executes one request: ephemeral tablets run inline; durable requests
/// join the commit coordinator's admission queue and complete when their
/// batch proves durable. Large legacy `Set`s convert to staged chunked
/// roots first, medium `Set`s to staged fabric roots (blocking calls —
/// plain worker threads only; the reactor bridge defers through its
/// continuation queue instead). Reads may complete as
/// [`ChunkedValue`](OperationResult::ChunkedValue) or
/// [`FabricValue`](OperationResult::FabricValue): the caller resolves
/// those through the lane/fabric (async on the reactor, blocking on
/// caller threads). A dropped responder (client gone) only drops the
/// outcome.
/// Resolves a range base for planning: fabric bases read through
/// directly (blocking direct read on plain threads: no promotion, no
/// placement change), everything else plans as-is. `None` means the
/// request already answered with state untouched.
fn resolve_range_base(
    tablets: &HashMap<TabletId, LiveTablet>,
    tablet: TabletId,
    key: &kivi_state::Key,
    now: WallTimestamp,
    fabric: &mut crate::fabric::TabletFabric,
    metrics: &core::cell::Cell<WorkerMetrics>,
    respond: &Sender<WorkerResponse>,
) -> Option<RangeBase> {
    match peek_range_base(tablets, tablet, key, now) {
        RangeBase::Fabric { fabric_id, .. } => match fabric.read_anywhere(fabric_id) {
            Ok(bytes) => Some(RangeBase::Inline(bytes)),
            Err(error) => {
                let mut snapshot = metrics.get();
                snapshot.channel_ops += 1;
                metrics.set(snapshot);
                let _ = respond.try_send(Err(map_fabric_error(error)));
                None
            }
        },
        other => Some(other),
    }
}

/// Stages one `SetRange` request: resolves a fabric base first (blocking
/// direct read on plain threads: no promotion, no placement change),
/// plans against the resolved bytes, and stages the result back through
/// the size split. `None` means the request already answered with state
/// untouched.
#[allow(clippy::too_many_arguments)]
fn stage_range_request(
    tablet: TabletId,
    key: kivi_state::Key,
    offset: u64,
    patch: bytes::Bytes,
    tablets: &HashMap<TabletId, LiveTablet>,
    now: WallTimestamp,
    chunks: &WorkerChunks,
    fabric: &mut crate::fabric::TabletFabric,
    metrics: &core::cell::Cell<WorkerMetrics>,
    respond: &Sender<WorkerResponse>,
) -> Option<(
    Operation,
    Vec<crate::chunk_lane::PinnedUpload>,
    Vec<crate::fabric::StagedSeal>,
)> {
    let base = resolve_range_base(tablets, tablet, &key, now, fabric, metrics, respond)?;
    match plan_set_range(key, offset, patch, base, chunks.inline_threshold) {
        SetRangePlan::Inline(op) => Some((op, Vec::new(), Vec::new())),
        SetRangePlan::Reject { detail } => {
            let mut snapshot = metrics.get();
            snapshot.channel_ops += 1;
            metrics.set(snapshot);
            let _ = respond.try_send(Err(WorkerRequestError::InvalidRequest { detail }));
            None
        }
        SetRangePlan::RestageSet { key, value } => {
            stage_restaged_set(tablet, &key, value, chunks, fabric, metrics, respond)
        }
        SetRangePlan::Splice {
            key,
            manifest,
            logical_len,
            offset,
            patch,
        } => match stage_value_work(
            &key,
            StageWork::Splice {
                manifest,
                logical_len,
                offset,
                patch,
            },
            chunks,
            metrics,
            respond,
        ) {
            Ok((staged, pin)) => Some((staged, pin.into_iter().collect(), Vec::new())),
            Err(()) => None,
        },
    }
}

/// Stages one restaged range result back through the size split: a
/// medium result stages into the fabric (preserving expiry exactly like
/// the chunked restage does), anything else rides the chunk path.
/// `None` means the request already answered with state untouched.
fn stage_restaged_set(
    tablet: TabletId,
    key: &kivi_state::Key,
    value: bytes::Bytes,
    chunks: &WorkerChunks,
    fabric: &mut crate::fabric::TabletFabric,
    metrics: &core::cell::Cell<WorkerMetrics>,
    respond: &Sender<WorkerResponse>,
) -> Option<(
    Operation,
    Vec<crate::chunk_lane::PinnedUpload>,
    Vec<crate::fabric::StagedSeal>,
)> {
    // Restaged results re-enter the size split so a
    // medium result stages into the fabric (preserving
    // expiry exactly like the chunked restage does).
    if value.len() > crate::fabric::FABRIC_INLINE_MAX
        && (value.len() as u64) <= chunks.inline_threshold
    {
        match stage_fabric_conditional(
            tablet,
            key,
            value,
            SetCondition::Always,
            ExpiryPolicy::Keep,
            fabric,
            metrics,
            respond,
        ) {
            Ok((staged, seal)) => Some((staged, Vec::new(), vec![seal])),
            Err(()) => None,
        }
    } else {
        match stage_restaged(key, value, chunks, metrics, respond) {
            Ok((staged, pin)) => Some((staged, pin.into_iter().collect(), Vec::new())),
            Err(()) => None,
        }
    }
}

/// Stages every write of a same-tablet atomic batch, collecting the
/// fabric seals the later `Commit` must persist. Any single staging
/// failure answers and rejects the whole batch (`None`). Kept beside
/// [`stage_request_representation`] so the admission match stays
/// reviewable.
fn stage_txn_batch(
    tablet: TabletId,
    writes: Vec<kivi_state::TxnWrite>,
    chunks: &WorkerChunks,
    fabric: &mut crate::fabric::TabletFabric,
    metrics: &core::cell::Cell<WorkerMetrics>,
    respond: &Sender<WorkerResponse>,
) -> Option<(Vec<kivi_state::TxnWrite>, Vec<crate::fabric::StagedSeal>)> {
    let mut staged_writes = Vec::with_capacity(writes.len());
    let mut seals = Vec::new();
    for write in writes {
        match stage_txn_write(tablet, write, chunks, fabric, metrics, respond) {
            Ok(Some((write, seal))) => {
                seals.extend(seal);
                staged_writes.push(write);
            }
            Ok(None) | Err(()) => return None,
        }
    }
    Some((staged_writes, seals))
}

/// Stages one request's representation before admission, answering
/// rejections and lane failures inline: `None` means the request already
/// answered with state untouched; `Some` carries the admitted operation,
/// its chunk pins, and its staged fabric seals.
#[allow(clippy::too_many_arguments)]
fn stage_request_representation(
    op: Operation,
    tablets: &mut HashMap<TabletId, LiveTablet>,
    tablet: TabletId,
    now: WallTimestamp,
    chunks: &WorkerChunks,
    fabric: &mut crate::fabric::TabletFabric,
    metrics: &core::cell::Cell<WorkerMetrics>,
    respond: &Sender<WorkerResponse>,
) -> Option<(
    Operation,
    Vec<crate::chunk_lane::PinnedUpload>,
    Vec<crate::fabric::StagedSeal>,
)> {
    // Medium transactional puts stage into the fabric before prepare so
    // a later Commit can never meet a durable root with unavailable
    // payload; chunked references ride the existing chunk path. Staged
    // seals ride the admission.
    let staged = match op {
        Operation::TxnPrepare {
            txn,
            coordinator,
            write,
            digest,
        } => match stage_txn_write(tablet, write, chunks, fabric, metrics, respond) {
            Ok(Some((write, seal))) => {
                let mut seals = Vec::new();
                seals.extend(seal);
                (
                    Operation::TxnPrepare {
                        txn,
                        coordinator,
                        write,
                        digest,
                    },
                    Vec::new(),
                    seals,
                )
            }
            Ok(None) | Err(()) => return None,
        },
        Operation::TxnCommitLocal { txn, writes } => {
            let (staged_writes, seals) =
                stage_txn_batch(tablet, writes, chunks, fabric, metrics, respond)?;
            (
                Operation::TxnCommitLocal {
                    txn,
                    writes: staged_writes,
                },
                Vec::new(),
                seals,
            )
        }
        Operation::Set { key, value }
            if value.len() > crate::fabric::FABRIC_INLINE_MAX
                && (value.len() as u64) <= chunks.inline_threshold =>
        {
            match stage_fabric_value(tablet, &key, value, true, fabric, metrics, respond) {
                Ok((staged, seal)) => (staged, Vec::new(), vec![seal]),
                Err(()) => return None,
            }
        }
        Operation::SetConditional {
            key,
            value,
            condition,
            expiry,
        } if value.len() > crate::fabric::FABRIC_INLINE_MAX
            && (value.len() as u64) <= chunks.inline_threshold =>
        {
            match stage_fabric_conditional(
                tablet, &key, value, condition, expiry, fabric, metrics, respond,
            ) {
                Ok((staged, seal)) => (staged, Vec::new(), vec![seal]),
                Err(()) => return None,
            }
        }
        Operation::SetRange { key, offset, patch } => stage_range_request(
            tablet, key, offset, patch, tablets, now, chunks, fabric, metrics, respond,
        )?,
        op @ Operation::TxnFinalize { .. } => {
            seal_finalize_intent(op, tablets, tablet, fabric, metrics, respond)?
        }
        other => match split_large_set(other, chunks.inline_threshold) {
            LargeSetSplit::Inline(op) => (op, Vec::new(), Vec::new()),
            LargeSetSplit::Stage { key, value } => {
                match stage_value_work(&key, StageWork::Value(value), chunks, metrics, respond) {
                    Ok((staged, pin)) => (staged, pin.into_iter().collect(), Vec::new()),
                    Err(()) => return None,
                }
            }
            LargeSetSplit::StageConditional {
                key,
                value,
                condition,
                expiry,
            } => match stage_value_work(
                &key,
                StageWork::ConditionalValue {
                    value,
                    condition,
                    expiry,
                },
                chunks,
                metrics,
                respond,
            ) {
                Ok((staged, pin)) => (staged, pin.into_iter().collect(), Vec::new()),
                Err(()) => return None,
            },
        },
    };
    Some(staged)
}

pub(crate) fn handle_request(
    request: TabletRequest,
    tablets: &mut HashMap<TabletId, LiveTablet>,
    metrics: &core::cell::Cell<WorkerMetrics>,
    durability: Option<&mut WorkerDurability>,
    chunks: &WorkerChunks,
    fabric: &mut crate::fabric::TabletFabric,
) {
    let TabletRequest {
        tablet,
        op,
        now,
        identity,
        respond,
    } = request;
    // Representation split before anything else: values over the threshold
    // stage (chunks, manifest, one sync barrier) or into the fabric (DRAM
    // residency plus seal payloads) and re-enter as small roots. Range
    // patches plan against the peeked base first (small inline results
    // WAL as splices, chunked bases restage through the lane splice,
    // fabric bases resolve then re-plan). Transactional writes pass
    // through unstaged except medium inline puts, which stage into the
    // fabric here; every carried reference is proven below before
    // admission, so a later Commit can never meet a durable root with
    // unavailable payload. Staging precedes admission, so a lane failure
    // or a rejected range answers here with state untouched.
    let Some((op, pins, staged)) =
        stage_request_representation(op, tablets, tablet, now, chunks, fabric, metrics, &respond)
    else {
        return;
    };
    // Prove every chunked and fabric reference a transaction carries:
    // staged values pass (just proven above and still alive), as do
    // copies of live committed roots; anything else is a dangling root
    // and the prepare is rejected loudly, never committed.
    if let Err(()) = verify_txn_chunk_refs(&op, chunks, metrics, &respond) {
        retire_staged(fabric, staged);
        return;
    }
    if let Err(()) = verify_txn_fabric_refs(&op, fabric, metrics, &respond) {
        retire_staged(fabric, staged);
        return;
    }
    // Attribute escrow ownership: a bounded-counter create takes its
    // initial full share on the tablet executing it (clients send zero;
    // routing already placed the key here, so the holder is exact).
    let mut op = op;
    if let Operation::BoundedCounterCreate { holder, .. } = &mut op {
        *holder = TabletId::from_u64(tablet.as_u64());
    }
    match durability {
        None => match tablets.get_mut(&tablet) {
            None => {
                let _ = respond.try_send(Err(WorkerRequestError::UnknownTablet { tablet }));
            }
            Some(live) => {
                let key = op.key().clone();
                // Pre-image for post-write retirement: when this write
                // supersedes a fabric root, the old materialization
                // retires (or defers while pinned) after success.
                let previous = live
                    .store()
                    .get(&key, now)
                    .and_then(kivi_state::StoredObject::fabric_ref);
                match live.execute(&op, now) {
                    Err(error) => {
                        retire_staged(fabric, staged);
                        let _ = respond.try_send(Err(WorkerRequestError::Tablet(error)));
                    }
                    Ok(result) => {
                        if let Some(post) = tablets
                            .get(&tablet)
                            .and_then(|live| live.store().get(&key, now))
                        {
                            fabric.retire_superseded(previous, post);
                        }
                        match resolve_channel_outcome(
                            tablet, &key, now, &op, result, tablets, fabric,
                        ) {
                            Ok(resolved) => {
                                let _ = respond.try_send(Ok(resolved));
                            }
                            Err(error) => {
                                let _ = respond.try_send(Err(error));
                            }
                        }
                    }
                }
            }
        },
        Some(durable) => {
            durable.commit.admit(
                PendingEntry::new(tablet, op, now, identity, respond)
                    .with_pins(pins)
                    .with_fabric_staged(staged),
            );
            durable.commit.poll(tablets, durable.namespace, fabric);
            // Admission itself is the outcome; the reply arrives through
            // the responder once the batch completes.
            let mut snapshot = metrics.get();
            snapshot.channel_ops += 1;
            metrics.set(snapshot);
            return;
        }
    }
    let mut snapshot = metrics.get();
    snapshot.channel_ops += 1;
    metrics.set(snapshot);
}

/// Retires staged fabric ids that never committed (prepare failure,
/// ephemeral execute failure): they are fresh, unpinned, and referenced
/// nowhere, so retirement always succeeds — failures are ignored.
fn retire_staged(fabric: &mut crate::fabric::TabletFabric, staged: Vec<crate::fabric::StagedSeal>) {
    for seal in staged {
        fabric.retire_or_defer(seal.fabric_id);
    }
}

/// Builds a one-shot response rendezvous for tests and the engine.
#[cfg(test)]
pub(crate) fn response_pair() -> (Sender<WorkerResponse>, Receiver<WorkerResponse>) {
    bounded(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_state::Key;
    use kivi_tablet::{DirectorySnapshot, HashPrefix, PartitionRange};
    use kivi_types::{NamespaceId, TabletAuthority, TabletEpoch, WriteGuardGeneration};

    const NS: NamespaceId = NamespaceId::from_u64(1);

    /// Test chunk access: TempDir-backed lane, shut down explicitly at
    /// the test's end (lane thread joins before the directory drops, so
    /// no open file handles wedge Windows cleanup). Mirrors the engine's
    /// shutdown ordering on a small scale.
    struct TestChunks {
        chunks: WorkerChunks,
        lane: crate::chunk_lane::ChunkLaneHandle,
        guard: crate::chunk_lane::ChunkLaneGuard,
        #[allow(dead_code)]
        dir: tempfile::TempDir,
    }

    impl TestChunks {
        fn open() -> Self {
            use crate::chunk_lane::{DEFAULT_CHUNK_CACHE_BYTES, spawn_lane};
            use kivi_chunk::{ChunkStore, DEFAULT_INLINE_THRESHOLD};
            let dir = tempfile::tempdir().expect("scratch");
            let domain = kivi_types::SecurityDomainId::from_u64(NS.as_u64());
            let (store, _) = ChunkStore::open(
                dir.path(),
                0,
                domain,
                1024 * 1024,
                kivi_types::WallTimestamp::from_micros(1_000_000),
            )
            .expect("test lane opens");
            let (lane, guard) = spawn_lane(WorkerId::from_u64(0), store, DEFAULT_CHUNK_CACHE_BYTES);
            Self {
                chunks: WorkerChunks {
                    lane: lane.clone(),
                    inline_threshold: DEFAULT_INLINE_THRESHOLD,
                    domain,
                    pins: std::sync::Arc::new(crate::chunk_lane::StagingPins::default()),
                    redundancy: None,
                },
                lane,
                guard,
                dir,
            }
        }

        fn shutdown(mut self) {
            self.guard.shutdown(&self.lane);
        }
    }

    fn test_fabric() -> (
        crate::fabric::TabletFabric,
        kivi_memory::OffcoreLaneGuard,
        tempfile::TempDir,
    ) {
        let dir = tempfile::tempdir().expect("scratch");
        let paths = crate::fabric::FabricPaths {
            root: dir.path().to_owned(),
        };
        let (fabric, guard) =
            crate::fabric::TabletFabric::open(WorkerId::from_u64(0), &paths, 1 << 20, 16 << 20)
                .expect("fabric opens");
        (fabric, guard, dir)
    }

    fn live_tablet(id: u64) -> LiveTablet {
        let tablet = TabletId::from_u64(id);
        let authority =
            TabletAuthority::new(tablet, TabletEpoch::INITIAL, WriteGuardGeneration::INITIAL);
        let snapshot = DirectorySnapshot::bootstrap(
            NS,
            tablet,
            PartitionRange::Hash(HashPrefix::new(0, 0).expect("root")),
            TabletEpoch::INITIAL,
            WriteGuardGeneration::INITIAL,
        )
        .expect("genesis");
        let descriptor = snapshot.get(tablet).expect("descriptor").clone();
        LiveTablet::from_descriptor(&descriptor, authority).expect("live")
    }

    fn round_trip(handle: &WorkerHandle, tablet: TabletId, op: Operation) -> WorkerResponse {
        let (respond, receive) = response_pair();
        handle
            .try_send(TabletRequest {
                tablet,
                op,
                now: WallTimestamp::from_micros(1_000_000),
                identity: None,
                respond,
            })
            .expect("admitted");
        receive.recv().expect("response")
    }

    #[test]
    fn worker_executes_requests_and_reports_unknown_tablets() {
        let chunks = TestChunks::open();
        let (fabric, _fabric_guard, _fabric_dir) = test_fabric();
        let mut handle = WorkerHandle::spawn(
            WorkerId::from_u64(0),
            vec![live_tablet(1)],
            16,
            None,
            chunks.chunks.clone(),
            fabric,
        );
        let set = round_trip(
            &handle,
            TabletId::from_u64(1),
            Operation::Set {
                key: Key::from("k"),
                value: bytes::Bytes::from_static(b"v"),
            },
        );
        assert!(matches!(set, Ok(OperationResult::Stored { .. })));
        let missing = round_trip(
            &handle,
            TabletId::from_u64(999),
            Operation::Get {
                key: Key::from("k"),
            },
        );
        assert!(matches!(
            missing,
            Err(WorkerRequestError::UnknownTablet { .. })
        ));
        handle
            .try_control(WorkerControl::Shutdown)
            .expect("control");
        handle.join().expect("clean exit");
        assert!(handle.is_joined());
        chunks.shutdown();
    }

    #[test]
    fn splice_plan_routes_every_base() {
        use crate::chunk_lane::{RangeBase, SetRangePlan, plan_set_range};
        use kivi_chunk::policy::MAX_LEGACY_VALUE_BYTES;
        let key = || Key::from("k");
        // Absent bases: small results splice inline, large restage.
        assert!(matches!(
            plan_set_range(
                key(),
                0,
                bytes::Bytes::from_static(b"v"),
                RangeBase::Absent,
                1024
            ),
            SetRangePlan::Inline(_)
        ));
        assert!(matches!(
            plan_set_range(
                key(),
                0,
                bytes::Bytes::from(vec![0u8; 2048]),
                RangeBase::Absent,
                1024
            ),
            SetRangePlan::RestageSet { .. }
        ));
        // Non-bytes bases pass through: preparation rejects them, not the plan.
        assert!(matches!(
            plan_set_range(
                key(),
                0,
                bytes::Bytes::from_static(b"v"),
                RangeBase::NonBytes,
                1024
            ),
            SetRangePlan::Inline(_)
        ));
        // Chunked bases splice lane-side with no size bound (streaming).
        assert!(matches!(
            plan_set_range(
                key(),
                0,
                bytes::Bytes::from_static(b"v"),
                RangeBase::Chunked {
                    manifest: kivi_types::ManifestId::from_bytes([0x11; 32]),
                    logical_len: MAX_LEGACY_VALUE_BYTES + 1,
                },
                1024
            ),
            SetRangePlan::Splice { .. }
        ));
        // Overflowing arithmetic and past-bound results reject with state
        // untouched (the caller answers `InvalidRequest`).
        assert!(matches!(
            plan_set_range(
                key(),
                u64::MAX,
                bytes::Bytes::from_static(b"v"),
                RangeBase::Absent,
                1024
            ),
            SetRangePlan::Reject { .. }
        ));
        assert!(matches!(
            plan_set_range(
                key(),
                MAX_LEGACY_VALUE_BYTES + 1,
                bytes::Bytes::new(),
                RangeBase::Absent,
                1024
            ),
            SetRangePlan::Reject { .. }
        ));
        // A result one byte past the bound rejects (at the bound it
        // still restages: the cap is enforced, not the boundary).
        assert!(matches!(
            plan_set_range(
                key(),
                MAX_LEGACY_VALUE_BYTES,
                bytes::Bytes::from_static(b"v"),
                RangeBase::Inline(bytes::Bytes::from_static(b"v")),
                1024
            ),
            SetRangePlan::Reject { .. }
        ));
    }

    #[test]
    fn splice_missing_manifest_fails_loudly_without_touching_state() {
        let chunks = TestChunks::open();
        let missing = kivi_types::ManifestId::from_bytes([0xAB; 32]);
        let error = chunks
            .lane
            .splice_range_blocking(missing, 100, 0, bytes::Bytes::from_static(b"x"))
            .expect_err("missing manifest fails");
        assert!(
            matches!(error, kivi_chunk::ChunkError::MissingManifest { .. }),
            "loud missing-manifest, got {error:?}"
        );
        chunks.shutdown();
    }

    #[test]
    fn finalize_empty_upload_stores_an_empty_manifest() {
        let chunks = TestChunks::open();
        let staged = chunks
            .lane
            .finalize_stream_blocking(Vec::new(), 0)
            .expect("empty upload finalizes");
        assert_eq!(staged.logical_len, 0);
        assert!(staged.chunks.is_empty());
        let bytes = chunks
            .lane
            .read_value_blocking(staged.manifest, 0)
            .expect("empty value reads");
        assert!(bytes.is_empty());
        let loaded = chunks
            .lane
            .fetch_manifest_blocking(staged.manifest)
            .expect("manifest fetches");
        assert!(loaded.entries.is_empty());
        chunks.shutdown();
    }

    #[test]
    fn stage_chunks_then_finalize_round_trips() {
        use kivi_codec::integrity::chunk_id;
        let chunks = TestChunks::open();
        let domain = chunks.chunks.domain;
        // Grid-aligned pieces (a full chunk plus a short final), exactly
        // what the upload drain stages: anything else is an invalid
        // manifest and must stay rejected (see below).
        let first = bytes::Bytes::from(vec![0x11u8; kivi_chunk::DEFAULT_CHUNK_SIZE]);
        let second = bytes::Bytes::from(vec![0x22u8; 50]);
        let first_id = chunk_id(domain, &first);
        let second_id = chunk_id(domain, &second);
        chunks
            .lane
            .stage_chunk_blocking(first_id, first.clone())
            .expect("stage");
        // Dedup: re-staging identical bytes is free, never a second record.
        chunks
            .lane
            .stage_chunk_blocking(first_id, first)
            .expect("restage");
        chunks
            .lane
            .stage_chunk_blocking(second_id, second)
            .expect("stage");
        let total = kivi_chunk::DEFAULT_CHUNK_SIZE as u64 + 50;
        let staged = chunks
            .lane
            .finalize_stream_blocking(
                vec![
                    (first_id, kivi_chunk::DEFAULT_CHUNK_SIZE as u64),
                    (second_id, 50),
                ],
                total,
            )
            .expect("finalize");
        assert_eq!(staged.logical_len, total);
        let bytes = chunks
            .lane
            .read_value_blocking(staged.manifest, total)
            .expect("read back");
        assert_eq!(bytes.len() as u64, total);
        assert!(
            bytes[..kivi_chunk::DEFAULT_CHUNK_SIZE]
                .iter()
                .all(|byte| *byte == 0x11)
        );
        assert!(
            bytes[kivi_chunk::DEFAULT_CHUNK_SIZE..]
                .iter()
                .all(|byte| *byte == 0x22)
        );
        // Off-grid entries stay invalid: finalize validates, never stores.
        assert!(
            chunks
                .lane
                .finalize_stream_blocking(vec![(first_id, 100), (second_id, 50)], 150)
                .is_err(),
            "short non-final entry rejected"
        );
        chunks.shutdown();
    }

    #[test]
    fn sweep_control_reclaims_without_request_queue() {
        let chunks = TestChunks::open();
        let (fabric, _fabric_guard, _fabric_dir) = test_fabric();
        let mut handle = WorkerHandle::spawn(
            WorkerId::from_u64(0),
            vec![live_tablet(1)],
            16,
            None,
            chunks.chunks.clone(),
            fabric,
        );
        round_trip(
            &handle,
            TabletId::from_u64(1),
            Operation::Set {
                key: Key::from("k"),
                value: bytes::Bytes::from_static(b"v"),
            },
        )
        .expect("set succeeds");
        round_trip(
            &handle,
            TabletId::from_u64(1),
            Operation::ExpireAt {
                key: Key::from("k"),
                expires_at: WallTimestamp::from_micros(10),
            },
        )
        .expect("expiry attaches");
        let (respond, receive) = bounded::<usize>(1);
        handle
            .try_control(WorkerControl::Sweep {
                now: WallTimestamp::from_micros(100),
                limit: 100,
                respond,
            })
            .expect("control");
        assert_eq!(receive.recv().expect("sweep count"), 1);
        handle
            .try_control(WorkerControl::Shutdown)
            .expect("control");
        handle.join().expect("clean exit");
        chunks.shutdown();
    }
}
