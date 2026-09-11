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
use kivi_durability::{
    CommitProof, DurabilityError, DurabilityLevel, DurabilityProvider, LaneStats, LocalWalLane,
    PersistIntent, StorageHealth, WalRecord, WorkerLaneStats,
};
use kivi_state::{Operation, OperationResult};
use kivi_types::{MutationIdentity, NamespaceId, TabletId, UnixMicros, WorkerId};

use crate::net::NetStartError;

use crate::tablet::{DurablePrepared, LiveTablet, TabletError};

/// Capacity of each worker's control channel. Control traffic is rare
/// (shutdown, sweeps); 16 slots cannot realistically fill, and every send
/// site handles `Full` explicitly anyway.
pub const CONTROL_CAPACITY: usize = 16;

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
}

/// One worker's durable state: namespace plus WAL lane access. `None`
/// everywhere means ephemeral mode (no persistence, no dedup).
#[derive(Debug)]
pub struct WorkerDurability {
    /// Namespace served (stamped into every record).
    pub namespace: NamespaceId,
    /// Owning worker (tracing and lane selection).
    pub worker: WorkerId,
    /// Lane access: exclusive per-worker lanes, or one shared lane behind
    /// a mutex (the group-commit experiment arm).
    pub lane: LaneAccess,
}

/// How a worker reaches its WAL lane.
#[derive(Debug)]
pub enum LaneAccess {
    /// Private lane: no synchronization, the production default.
    Exclusive(LocalWalLane),
    /// One lane shared by all workers (experiment arm): appends serialize
    /// on the mutex; recovery and ordering are unchanged.
    Shared(Arc<Mutex<LocalWalLane>>),
}

impl WorkerDurability {
    /// Appends one batch on this worker's lane, syncing before return.
    ///
    /// # Errors
    ///
    /// Returns [`DurabilityError`] when the batch cannot be made durable;
    /// the caller must not mutate logical state.
    pub fn append(&mut self, records: &[WalRecord]) -> Result<CommitProof, DurabilityError> {
        let intent = PersistIntent {
            records,
            level: DurabilityLevel::Sync,
        };
        match &mut self.lane {
            LaneAccess::Exclusive(lane) => lane.append_batch(&intent),
            LaneAccess::Shared(shared) => shared
                .lock()
                .map_err(|_| DurabilityError::Io {
                    op: "lock shared WAL lane",
                    message: "lane mutex poisoned".to_owned(),
                    code: None,
                })?
                .append_batch(&intent),
        }
    }

    /// Current coarse lane health.
    #[must_use]
    pub fn health(&self) -> StorageHealth {
        match &self.lane {
            LaneAccess::Exclusive(lane) => lane.health(),
            LaneAccess::Shared(shared) => shared
                .lock()
                .map_or(StorageHealth::Failed, |lane| lane.health()),
        }
    }

    /// Cumulative lane counters (a poisoned shared lane reports zeros;
    /// poisoning cannot happen — see [`WorkerDurability::append`] — so
    /// this is unreachable defensiveness, not a silent drop).
    #[must_use]
    pub fn stats(&self) -> LaneStats {
        match &self.lane {
            LaneAccess::Exclusive(lane) => lane.stats(),
            LaneAccess::Shared(shared) => {
                shared.lock().map(|lane| lane.stats()).unwrap_or_default()
            }
        }
    }

    /// Full lane snapshot for the admin plane.
    #[must_use]
    pub fn lane_stats(&self) -> WorkerLaneStats {
        match &self.lane {
            LaneAccess::Exclusive(lane) => lane.lane_stats(),
            LaneAccess::Shared(shared) => shared.lock().map_or(
                WorkerLaneStats {
                    lane: u16::MAX,
                    active_segment: 0,
                    segments: 0,
                    health: StorageHealth::Failed,
                    stats: LaneStats::default(),
                },
                |lane| lane.lane_stats(),
            ),
        }
    }
}

/// Durable execution failure: maps onto either channel errors or native
/// statuses at the caller's boundary, never silently.
#[derive(Debug)]
pub(crate) enum DurableFailure {
    /// Tablet logic failure (read-path rejections, commit exhaustion).
    Op(TabletError),
    /// Identity below the session floor.
    Expired,
    /// Session outcome window exhausted.
    Overloaded,
    /// WAL append failure (logical state untouched).
    Storage(DurabilityError),
}

/// Executes one operation through the durable pipeline on an already-owned
/// tablet: prepare (dedup + predict), persist exactly one record, apply +
/// verify (mutations) or install (terminal outcomes), reply.
pub(crate) fn execute_durable(
    live: &mut LiveTablet,
    durable: &mut WorkerDurability,
    op: &Operation,
    opcode: u8,
    identity: Option<&MutationIdentity>,
    now: UnixMicros,
) -> Result<OperationResult, DurableFailure> {
    let authority = live.authority();
    let namespace = durable.namespace;
    let tablet = live.id();
    let epoch = authority.epoch();
    let guard = authority.guard();
    match live
        .prepare_durable(op, identity, now)
        .map_err(DurableFailure::Op)?
    {
        DurablePrepared::Read(result) => Ok(result),
        DurablePrepared::Expired => Err(DurableFailure::Expired),
        DurablePrepared::Overloaded => Err(DurableFailure::Overloaded),
        DurablePrepared::DedupHit { outcome, .. } => complete_outcome(outcome),
        DurablePrepared::Terminal { outcome, identity } => {
            let commit = live.assign_commit().map_err(DurableFailure::Op)?;
            let record = WalRecord::Outcome(kivi_durability::wal::OutcomeRecord {
                namespace,
                tablet,
                epoch,
                guard,
                commit,
                now,
                opcode,
                identity,
                outcome: outcome.clone(),
            });
            durable
                .append(std::slice::from_ref(&record))
                .map_err(DurableFailure::Storage)?;
            let committed = live.commit_terminal(&outcome, identity.as_ref(), commit, opcode);
            complete_outcome(committed)
        }
        DurablePrepared::Persist {
            mutation,
            expected,
            identity,
        } => {
            let commit = live.assign_commit().map_err(DurableFailure::Op)?;
            let is_persist_expiry = matches!(op, Operation::PersistExpiry { .. });
            let record = WalRecord::Mutation(kivi_durability::wal::MutationRecord {
                namespace,
                tablet,
                epoch,
                guard,
                commit,
                now,
                opcode,
                identity,
                mutation: mutation.clone(),
                expected: expected.clone(),
            });
            durable
                .append(std::slice::from_ref(&record))
                .map_err(DurableFailure::Storage)?;
            live.commit_persisted(
                &mutation,
                &expected,
                is_persist_expiry,
                identity.as_ref(),
                commit,
                now,
            )
            .map_err(DurableFailure::Op)
        }
    }
}

/// Maps a stored terminal outcome onto a reply (shared by dedup hits and
/// freshly committed terminal outcomes).
fn complete_outcome(
    outcome: kivi_state::DurableOutcome,
) -> Result<OperationResult, DurableFailure> {
    use kivi_state::DurableOutcome;
    match outcome {
        DurableOutcome::Completed(result) => Ok(result),
        DurableOutcome::Rejected(error) => Err(DurableFailure::Op(TabletError::Op(error))),
        DurableOutcome::VersionExhausted => Err(DurableFailure::Op(TabletError::Apply(
            kivi_state::ApplyError::VersionExhausted,
        ))),
    }
}

/// Derives the originating opcode discriminant from a typed operation
/// (channel path; the native path carries it on the request).
pub(crate) fn operation_opcode(op: &Operation) -> u8 {
    kivi_protocol::operation_opcode(op).as_u8()
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
    ) -> Self {
        let (request_tx, request_rx) = bounded::<TabletRequest>(request_capacity.max(1));
        let (control_tx, control_rx) = bounded::<WorkerControl>(CONTROL_CAPACITY);
        let thread = thread::Builder::new()
            .name(format!("kivi-worker-{}", id.as_u64()))
            .spawn(move || run(id, tablets, request_rx, control_rx, durability))
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
        launch: crate::net::NetLaunch,
    ) -> Result<(Self, SocketAddr), NetStartError> {
        let (request_tx, request_rx) = bounded::<TabletRequest>(request_capacity.max(1));
        let (control_tx, control_rx) = bounded::<WorkerControl>(CONTROL_CAPACITY);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let thread = thread::Builder::new()
            .name(format!("kivi-worker-{}", id.as_u64()))
            .spawn(move || {
                crate::net::run_net(
                    id, tablets, request_rx, control_rx, durability, launch, ready_tx,
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
#[allow(clippy::needless_pass_by_value)]
fn run(
    id: WorkerId,
    tablets: Vec<LiveTablet>,
    requests: Receiver<TabletRequest>,
    control: Receiver<WorkerControl>,
    mut durability: Option<WorkerDurability>,
) {
    let _ = id;
    let metrics = core::cell::Cell::new(WorkerMetrics::default());
    let mut tablets: HashMap<TabletId, LiveTablet> = tablets
        .into_iter()
        .map(|tablet| (tablet.id(), tablet))
        .collect();
    loop {
        // Control fast path: administration never waits behind requests.
        if let Ok(message) = control.try_recv() {
            if !handle_control(message, &mut tablets, &metrics, durability.as_ref()) {
                break;
            }
            continue;
        }
        select! {
            recv(control) -> message => {
                match message {
                    Ok(message) => {
                        if !handle_control(message, &mut tablets, &metrics, durability.as_ref()) {
                            break;
                        }
                    }
                    // Engine gone: exit rather than serve a headless worker.
                    Err(_) => break,
                }
            }
            recv(requests) -> message => {
                match message {
                    Ok(request) => handle_request(request, &mut tablets, &metrics, durability.as_mut()),
                    // All clients gone: exit cleanly.
                    Err(_) => break,
                }
            }
        }
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
            let mut reclaimed = 0usize;
            for tablet in tablets.values_mut() {
                reclaimed += tablet.sweep_expired(now, limit).len();
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
    }
}

/// Executes one request against the owned tablet and responds. A dropped
/// responder (client gone) only drops the outcome.
pub(crate) fn handle_request(
    request: TabletRequest,
    tablets: &mut HashMap<TabletId, LiveTablet>,
    metrics: &core::cell::Cell<WorkerMetrics>,
    durability: Option<&mut WorkerDurability>,
) {
    let TabletRequest {
        tablet,
        op,
        now,
        identity,
        respond,
    } = request;
    let outcome = match tablets.get_mut(&tablet) {
        None => Err(WorkerRequestError::UnknownTablet { tablet }),
        Some(live) => match durability {
            None => live.execute(&op, now).map_err(WorkerRequestError::Tablet),
            Some(durable) => {
                let opcode = operation_opcode(&op);
                match execute_durable(live, durable, &op, opcode, identity.as_ref(), now) {
                    Ok(result) => Ok(result),
                    Err(DurableFailure::Op(error)) => Err(WorkerRequestError::Tablet(error)),
                    Err(DurableFailure::Expired) => Err(WorkerRequestError::DedupExpired),
                    Err(DurableFailure::Overloaded) => Err(WorkerRequestError::SessionOverloaded),
                    Err(DurableFailure::Storage(error)) => Err(WorkerRequestError::Storage(error)),
                }
            }
        },
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
        let mut handle = WorkerHandle::spawn(WorkerId::from_u64(0), vec![live_tablet(1)], 16, None);
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
    }

    #[test]
    fn sweep_control_reclaims_without_request_queue() {
        let mut handle = WorkerHandle::spawn(WorkerId::from_u64(0), vec![live_tablet(1)], 16, None);
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
    }
}
