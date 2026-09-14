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
use kivi_types::{ManifestId, MutationIdentity, NamespaceId, TabletId, UnixMicros, WorkerId};

use crate::chunk_lane::{
    ChunkLaneHandle, LargeSetSplit, RangeBase, SetRangePlan, StagingPins, plan_set_range,
    split_large_set,
};
use crate::commit::{BatchPolicy, CommitCoordinator, PendingEntry};
use crate::net::NetStartError;

use crate::tablet::{LiveTablet, TabletError};

/// Capacity of each worker's control channel. Control traffic is rare
/// (shutdown, sweeps); 16 slots cannot realistically fill, and every send
/// site handles `Full` explicitly anyway.
pub const CONTROL_CAPACITY: usize = 16;

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
    requests: &Receiver<TabletRequest>,
) {
    handle_request(first, tablets, metrics, durability.as_mut(), chunks);
    for _ in 0..REQUEST_DRAIN_PER_WAKE {
        match requests.try_recv() {
            Ok(request) => handle_request(request, tablets, metrics, durability.as_mut(), chunks),
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
    pub now: UnixMicros,
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
    ) -> Result<Self, DurabilityError> {
        Ok(Self {
            namespace,
            worker,
            commit: CommitCoordinator::spawn(policy, lane, initial_stats)?,
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
        now: UnixMicros,
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
    thread: Option<JoinHandle<()>>,
}

impl WorkerHandle {
    /// Spawns the worker thread owning `tablets`, with a bounded request
    /// queue of `request_capacity`. `durability` is `None` in ephemeral
    /// mode; in durable mode the worker owns its WAL lane exclusively.
    /// `chunks` is this worker's chunk lane access plus the representation
    /// policy (both modes stage large values through it).
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
    ) -> Self {
        let (request_tx, request_rx) = bounded::<TabletRequest>(request_capacity.max(1));
        let (control_tx, control_rx) = bounded::<WorkerControl>(CONTROL_CAPACITY);
        let thread = thread::Builder::new()
            .name(format!("kivi-worker-{}", id.as_u64()))
            .spawn(move || run(id, tablets, request_rx, control_rx, durability, chunks))
            .expect("worker thread spawns");
        Self {
            id,
            requests: request_tx,
            control: control_tx,
            thread: Some(thread),
        }
    }

    /// Returns the worker identity.
    #[must_use]
    pub const fn id(&self) -> WorkerId {
        self.id
    }

    /// Clones the bounded request sender (engine keeps one per worker and
    /// hands clones to clients; the worker exits once all senders drop).
    #[must_use]
    pub fn sender(&self) -> Sender<TabletRequest> {
        self.requests.clone()
    }

    /// Clones the bounded control sender (metrics queries, sweeps).
    #[must_use]
    pub fn control_sender(&self) -> Sender<WorkerControl> {
        self.control.clone()
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

    /// Sends one control message without blocking.
    ///
    /// # Errors
    ///
    /// Returns the `TrySendError` (full control queue or disconnected
    /// worker) for the caller to handle.
    pub fn try_control(&self, control: WorkerControl) -> Result<(), TrySendError<WorkerControl>> {
        self.control.try_send(control)
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
        launch: crate::net::NetLaunch,
    ) -> Result<(Self, SocketAddr), NetStartError> {
        let (request_tx, request_rx) = bounded::<TabletRequest>(request_capacity.max(1));
        let (control_tx, control_rx) = bounded::<WorkerControl>(CONTROL_CAPACITY);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let thread = thread::Builder::new()
            .name(format!("kivi-worker-{}", id.as_u64()))
            .spawn(move || {
                crate::net::run_net(
                    id, tablets, request_rx, control_rx, durability, chunks, launch, ready_tx,
                );
            })
            .map_err(|error| NetStartError::Startup(error.to_string()))?;
        match ready_rx.recv_timeout(crate::net::STARTUP_TIMEOUT) {
            Ok(Ok(addr)) => Ok((
                Self {
                    id,
                    requests: request_tx,
                    control: control_tx,
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
#[allow(clippy::needless_pass_by_value)]
fn run(
    id: WorkerId,
    tablets: Vec<LiveTablet>,
    requests: Receiver<TabletRequest>,
    control: Receiver<WorkerControl>,
    mut durability: Option<WorkerDurability>,
    chunks: WorkerChunks,
) {
    let _ = id;
    let metrics = core::cell::Cell::new(WorkerMetrics::default());
    let mut tablets: HashMap<TabletId, LiveTablet> = tablets
        .into_iter()
        .map(|tablet| (tablet.id(), tablet))
        .collect();
    // Pumps the coordinator once (no-op in ephemeral mode).
    let pump = |tablets: &mut HashMap<TabletId, LiveTablet>,
                durability: Option<&mut WorkerDurability>| {
        if let Some(durable) = durability {
            let namespace = durable.namespace;
            durable.commit.poll(tablets, namespace);
        }
    };
    loop {
        // Control fast path: administration never waits behind requests.
        if let Ok(message) = control.try_recv() {
            if !handle_control(message, &mut tablets, &metrics, durability.as_ref()) {
                flush_for_shutdown(&mut tablets, durability.as_mut());
                break;
            }
            pump(&mut tablets, durability.as_mut());
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
                            if !handle_control(message, &mut tablets, &metrics, durability.as_ref()) {
                                flush_for_shutdown(&mut tablets, durability.as_mut());
                                break;
                            }
                        }
                        // Engine gone: exit rather than serve a headless worker.
                        Err(_) => break,
                    }
                }
                recv(requests) -> message => {
                    match message {
                        Ok(request) => admit_drained(request, &mut tablets, &metrics, &mut durability, &chunks, &requests),
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
                            if !handle_control(message, &mut tablets, &metrics, durability.as_ref()) {
                                flush_for_shutdown(&mut tablets, durability.as_mut());
                                break;
                            }
                        }
                        // Engine gone: exit rather than serve a headless worker.
                        Err(_) => break,
                    }
                }
                recv(requests) -> message => {
                    match message {
                        Ok(request) => admit_drained(request, &mut tablets, &metrics, &mut durability, &chunks, &requests),
                        // All clients gone: exit cleanly.
                        Err(_) => break,
                    }
                }
            }
        }
        pump(&mut tablets, durability.as_mut());
    }
}

/// Flushes admitted work synchronously for shutdown: every admitted
/// request is answered (committed or explicitly failed) before the worker
/// exits, so shutdown never strands a blocked client.
fn flush_for_shutdown(
    tablets: &mut HashMap<TabletId, LiveTablet>,
    durability: Option<&mut WorkerDurability>,
) {
    if let Some(durable) = durability {
        let namespace = durable.namespace;
        durable.commit.flush_blocking(tablets, namespace);
    }
}

/// Handles one control message. Returns `false` when the worker must exit.
pub(crate) fn handle_control(
    message: WorkerControl,
    tablets: &mut HashMap<TabletId, LiveTablet>,
    metrics: &core::cell::Cell<WorkerMetrics>,
    durability: Option<&WorkerDurability>,
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
    now: UnixMicros,
) -> RangeBase {
    let Some(live) = tablets.get(&tablet) else {
        return RangeBase::Absent;
    };
    match live.store().get(key, now) {
        None => RangeBase::Absent,
        Some(object) => match object.value() {
            kivi_state::LogicalValue::Bytes(bytes) => RangeBase::Inline(bytes.clone()),
            kivi_state::LogicalValue::StrictCounter(_) => RangeBase::Counter,
            kivi_state::LogicalValue::Chunked(chunked) => RangeBase::Chunked {
                manifest: chunked.manifest,
                logical_len: chunked.logical_len,
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
            match chunks.lane.stage_value_blocking(value) {
                Ok(staged) => {
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
                Err(error) => failed(metrics, respond, WorkerRequestError::ChunkStore(error)),
            }
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
            match chunks.lane.stage_value_blocking(value) {
                Ok(staged) => {
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
                Err(error) => failed(metrics, respond, WorkerRequestError::ChunkStore(error)),
            }
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

/// Namespace for checkpoint captures (durable workers serve exactly one).
fn capture_namespace(durability: Option<&WorkerDurability>) -> NamespaceId {
    durability.map_or(NamespaceId::from_u64(0), |durable| durable.namespace)
}

/// Executes one request: ephemeral tablets run inline; durable requests
/// join the commit coordinator's admission queue and complete when their
/// batch proves durable. Large legacy `Set`s convert to staged chunked
/// roots first (blocking lane call — plain worker threads only; the
/// reactor bridge defers through its continuation queue instead). Reads
/// may complete as [`ChunkedValue`](OperationResult::ChunkedValue): the
/// caller resolves those through the lane (async on the reactor, blocking
/// on caller threads). A dropped responder (client gone) only drops the
/// outcome.
pub(crate) fn handle_request(
    request: TabletRequest,
    tablets: &mut HashMap<TabletId, LiveTablet>,
    metrics: &core::cell::Cell<WorkerMetrics>,
    durability: Option<&mut WorkerDurability>,
    chunks: &WorkerChunks,
) {
    let TabletRequest {
        tablet,
        op,
        now,
        identity,
        respond,
    } = request;
    // Representation split before anything else: values over the threshold
    // stage (chunks, manifest, one sync barrier) and re-enter as a small
    // `SetChunked`. Range patches plan against the peeked base first
    // (small inline results WAL as splices, chunked bases restage through
    // the lane splice). Staging precedes admission, so a lane failure or
    // a rejected range answers here with state untouched.
    let (op, pinned) = match op {
        Operation::SetRange { key, offset, patch } => {
            let base = peek_range_base(tablets, tablet, &key, now);
            match plan_set_range(key, offset, patch, base, chunks.inline_threshold) {
                SetRangePlan::Inline(op) => (op, None),
                SetRangePlan::Reject { detail } => {
                    let mut snapshot = metrics.get();
                    snapshot.channel_ops += 1;
                    metrics.set(snapshot);
                    let _ = respond.try_send(Err(WorkerRequestError::InvalidRequest { detail }));
                    return;
                }
                SetRangePlan::RestageSet { key, value } => {
                    match stage_restaged(&key, value, chunks, metrics, &respond) {
                        Ok(staged) => staged,
                        Err(()) => return,
                    }
                }
                SetRangePlan::Splice {
                    key,
                    manifest,
                    logical_len,
                    offset,
                    patch,
                } => {
                    match stage_value_work(
                        &key,
                        StageWork::Splice {
                            manifest,
                            logical_len,
                            offset,
                            patch,
                        },
                        chunks,
                        metrics,
                        &respond,
                    ) {
                        Ok(staged) => staged,
                        Err(()) => return,
                    }
                }
            }
        }
        other => match split_large_set(other, chunks.inline_threshold) {
            LargeSetSplit::Inline(op) => (op, None),
            LargeSetSplit::Stage { key, value } => {
                match stage_value_work(&key, StageWork::Value(value), chunks, metrics, &respond) {
                    Ok(staged) => staged,
                    Err(()) => return,
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
                &respond,
            ) {
                Ok(staged) => staged,
                Err(()) => return,
            },
        },
    };
    let outcome = match durability {
        None => match tablets.get_mut(&tablet) {
            None => Err(WorkerRequestError::UnknownTablet { tablet }),
            Some(live) => live.execute(&op, now).map_err(WorkerRequestError::Tablet),
        },
        Some(durable) => {
            durable
                .commit
                .admit(PendingEntry::new(tablet, op, now, identity, respond).with_pinned(pinned));
            durable.commit.poll(tablets, durable.namespace);
            // Admission itself is the outcome; the reply arrives through
            // the responder once the batch completes.
            let mut snapshot = metrics.get();
            snapshot.channel_ops += 1;
            metrics.set(snapshot);
            return;
        }
    };
    let mut snapshot = metrics.get();
    snapshot.channel_ops += 1;
    metrics.set(snapshot);
    let _ = respond.try_send(outcome);
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
            let (store, _) = ChunkStore::open(dir.path(), 0, domain, 1024 * 1024, 1_000_000)
                .expect("test lane opens");
            let (lane, guard) = spawn_lane(WorkerId::from_u64(0), store, DEFAULT_CHUNK_CACHE_BYTES);
            Self {
                chunks: WorkerChunks {
                    lane: lane.clone(),
                    inline_threshold: DEFAULT_INLINE_THRESHOLD,
                    domain,
                    pins: std::sync::Arc::new(crate::chunk_lane::StagingPins::default()),
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
                now: UnixMicros::from_micros(1_000_000),
                identity: None,
                respond,
            })
            .expect("admitted");
        receive.recv().expect("response")
    }

    #[test]
    fn worker_executes_requests_and_reports_unknown_tablets() {
        let chunks = TestChunks::open();
        let mut handle = WorkerHandle::spawn(
            WorkerId::from_u64(0),
            vec![live_tablet(1)],
            16,
            None,
            chunks.chunks.clone(),
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
        // Counters pass through: preparation rejects them, not the plan.
        assert!(matches!(
            plan_set_range(
                key(),
                0,
                bytes::Bytes::from_static(b"v"),
                RangeBase::Counter,
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
        let mut handle = WorkerHandle::spawn(
            WorkerId::from_u64(0),
            vec![live_tablet(1)],
            16,
            None,
            chunks.chunks.clone(),
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
                expires_at: UnixMicros::from_micros(10),
            },
        )
        .expect("expiry attaches");
        let (respond, receive) = bounded::<usize>(1);
        handle
            .try_control(WorkerControl::Sweep {
                now: UnixMicros::from_micros(100),
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
