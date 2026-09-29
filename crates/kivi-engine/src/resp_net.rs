//! The Redis/RESP compatibility edge, served from a worker's own reactor.
//!
//! Connection tasks run on the same reactor as the worker that owns the
//! tablets, sharing state through `Rc<RefCell<_>>`, so there is no engine queue
//! and no thread wakeup on the request path.
//!
//! ```text
//! socket → RESP parse → route → handle_request (same thread) → RESP reply → socket
//! ```
//!
//! # What runs inline, and what does not
//!
//! The inline path answers synchronously. That is true of an ephemeral
//! worker's [`handle_request`], and false in two other cases, both of which
//! would otherwise park the reactor on a reply only the reactor can produce:
//!
//! * **Durable workers.** A durable write is admitted to the commit
//!   coordinator and answered when the device flush completes.
//! * **Payloads that live in the chunk lane.** Reading a chunked root, or
//!   writing a value large enough to stage one, has to wait on a separate lane
//!   thread. Those requests are recognised from the request and the stored
//!   representation *before* execution, so nothing is executed twice.
//!
//! Everything else - the overwhelming majority of traffic, and every point-read
//! profile - executes inline. What does not goes to the owning worker through
//! the engine queue: correctness never depends on which worker holds the
//! connection.
//!
//! # Which worker accepts
//!
//! The worker owning the most tablets takes the listener, so a single-tablet
//! engine - the common deployment - forwards nothing. The choice is made once,
//! at spawn, and reported so the admin plane can name the bound endpoint.

use std::net::SocketAddr;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use compio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use compio::net::{TcpListener, TcpStream};
use kivi_resp::{
    ConnConfig, DrainOutcome, ExecuteError, Executor, RespConnection, RespStats, RespVersion,
};
use kivi_state::{Operation, OperationResult};
use kivi_types::{NamespaceId, TabletId, WallTimestamp, WorkerId};

use crate::LocalClient;
use crate::net::{ActiveGuard, WorkerNet};
use crate::worker::{TabletRequest, WorkerResponse, handle_request};

/// Read buffer per connection. Requests are small; a pipeline shares this, and
/// the buffer is reused for the connection's life.
const READ_CHUNK: usize = 64 * 1024;

/// Reply buffer above which the connection releases it rather than letting one
/// oversized value pin megabytes until the client disconnects.
const REPLY_BUFFER_RECLAIM: usize = 8 * 1024 * 1024;

/// Per-worker launch configuration for the RESP edge.
#[derive(Debug, Clone)]
pub struct RespNetConfig {
    /// Socket to listen on. `None` on every worker but the accepting one.
    pub listen: Option<SocketAddr>,
    /// Namespace this edge serves.
    pub namespace: NamespaceId,
    /// Per-connection resource bounds.
    pub conn: ConnConfig,
    /// Shared frontend statistics.
    pub stats: Arc<RespStats>,
    /// Engine client used for requests this worker does not own. Filled in
    /// once the engine exists; the accept loop waits for it, so no request can
    /// arrive before there is somewhere to forward it.
    pub forward: Arc<std::sync::OnceLock<LocalClient>>,
    /// Where the bind result is reported. Separate from the native ready
    /// channel because both listeners bind independently and a caller must be
    /// able to tell which one failed.
    pub ready: std::sync::mpsc::Sender<Result<RespBinding, String>>,
}

/// Engine-wide RESP edge configuration.
#[derive(Debug, Clone)]
pub struct EngineResp {
    /// Socket the accepting worker binds.
    pub listen: SocketAddr,
    /// Namespace this edge serves.
    pub namespace: NamespaceId,
    /// Per-connection resource bounds.
    pub conn: ConnConfig,
    /// Shared frontend statistics.
    pub stats: Arc<RespStats>,
}

/// What the accepting worker bound, for the admin plane and startup logs.
///
/// The bound address, not the requested one: with port `0` the kernel chooses,
/// and an admin view that reported the request would name a port nobody is
/// listening on.
#[derive(Debug, Clone, Copy)]
pub struct RespBinding {
    /// Worker that owns the listener.
    pub worker: WorkerId,
    /// Bound endpoint.
    pub endpoint: SocketAddr,
}

/// Serves the RESP edge beside the native accept loop on this worker's reactor.
///
/// Binds and reports before accepting, so a caller waiting on the bound address
/// can never race the listener and a bind failure is reported rather than
/// leaving a frontend that silently accepts nothing.
pub(crate) async fn serve_resp(shared: Rc<WorkerNet>, config: RespNetConfig) {
    let Some(listen) = config.listen else {
        return;
    };
    let ready = &config.ready;
    let listener = match TcpListener::bind(listen).await {
        Ok(listener) => listener,
        Err(error) => {
            let _ = ready.send(Err(format!("RESP bind {listen}: {error}")));
            return;
        }
    };
    let bound = match listener.local_addr() {
        Ok(addr) => addr,
        Err(error) => {
            let _ = ready.send(Err(format!("RESP local_addr: {error}")));
            return;
        }
    };
    tracing::info!(worker = %shared.id.as_u64(), %bound, "RESP listener bound");
    let _ = ready.send(Ok(RespBinding {
        worker: shared.id,
        endpoint: bound,
    }));

    // Requests this worker does not own need an engine client, which cannot
    // exist before the workers it routes to. Clients that connect during
    // startup sit in the accept backlog until this resolves - a bounded wait,
    // not a race, and never a connection answered with a false error.
    let forward = loop {
        if shared.shutdown.get() {
            return;
        }
        if let Some(client) = config.forward.get() {
            break client.clone();
        }
        crate::net::YieldNow::default().await;
    };
    let forward = Arc::new(forward); // One pool per worker, shared by every connection it serves. Requests that
    // need another thread are the only ones that reach it, so it stays idle -
    // and out of the way - on a workload the worker can answer itself.
    let helpers = spawn_helpers(&forward, HELPER_THREADS);

    // No accept timeout: cancelling a pending accept leaks the listener on this
    // compio version, which breaks same-port restart. Shutdown observes the
    // flag at the top of the next turn, and a pending accept is woken by the
    // dummy connection the engine already sends on shutdown.
    let mut next_id: u64 = 1;
    loop {
        if shared.shutdown.get() {
            break;
        }
        match listener.accept().await {
            Ok((stream, peer)) => {
                // Request/response traffic is small frames both ways: Nagle
                // would make a reply wait for a delayed ACK, so it is off.
                let _ = stream.set_nodelay(true);
                let id = next_id;
                next_id = next_id.wrapping_add(1);
                let stats = Arc::clone(&config.stats);
                let conn = config.conn;
                let worker = Rc::clone(&shared);
                let forward = Arc::clone(&forward);
                let helpers = Arc::clone(&helpers);
                compio::runtime::spawn(async move {
                    serve_resp_conn(RespConnLaunch {
                        stream,
                        peer,
                        id,
                        shared: worker,
                        conn_config: conn,
                        stats,
                        forward,
                        helpers,
                    })
                    .await;
                })
                .detach();
            }
            Err(error) => {
                tracing::debug!(%error, "RESP accept failed");
                compio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }
}

/// Everything one accepted connection needs, so the serving function takes a
/// bundle rather than eight positional arguments that are easy to transpose.
struct RespConnLaunch {
    stream: TcpStream,
    peer: SocketAddr,
    id: u64,
    shared: Rc<WorkerNet>,
    conn_config: ConnConfig,
    stats: Arc<RespStats>,
    forward: Arc<LocalClient>,
    helpers: Arc<async_channel::Sender<DeferredJob>>,
}

/// Serves one RESP connection on the worker's reactor.
///
/// The turn shape is one read, then everything that read decoded, then one
/// write. A pipelined client therefore costs a read and a write per batch
/// rather than per command.
#[allow(clippy::too_many_lines)]
async fn serve_resp_conn(launch: RespConnLaunch) {
    let RespConnLaunch {
        mut stream,
        peer,
        id,
        shared,
        conn_config,
        stats,
        forward,
        helpers,
    } = launch;
    let _guard = ActiveGuard::hold(&shared.active);
    stats.connections_current.fetch_add(1, Ordering::Relaxed);
    stats.connections_total.fetch_add(1, Ordering::Relaxed);
    stats.resp2_current.fetch_add(1, Ordering::Relaxed);

    let (respond, receive) = crossbeam_channel::bounded(1);
    let executor = WorkerExecutor {
        shared,
        forward,
        helpers,
        respond,
        receive,
    };
    let mut connection = RespConnection::new(executor, conn_config, id);
    let mut read_buf = vec![0u8; READ_CHUNK];
    let mut out: Vec<u8> = Vec::with_capacity(READ_CHUNK);
    let mut version = RespVersion::V2;

    let mut exit = None;
    loop {
        let chunk = Vec::with_capacity(READ_CHUNK);
        let outcome = stream.read(chunk).await;
        let count = match outcome.0 {
            Ok(0) | Err(_) => break,
            Ok(count) => count,
        };
        debug_assert_eq!(count, outcome.1.len());
        // compio's read is buffer-passing and hands back a fresh allocation
        // each call, so its bytes are copied into the connection's reusable
        // buffer rather than parsed from a temporary.
        if read_buf.len() < count {
            read_buf.resize(count, 0);
        }
        read_buf[..count].copy_from_slice(&outcome.1[..count]);
        if connection.push(&read_buf[..count]).is_err() {
            break;
        }
        // Every turn this read produced, then ONE write. A pipeline deeper
        // than the per-turn budget spans several turns, and writing after each
        // one costs a syscall per turn - a 256-deep pipeline paid four writes
        // where the client sent one read. Coalescing makes the write count per
        // read rather than per turn, which is the only ratio that matches what
        // a pipelining client actually sends.
        let mut close_now = false;
        while connection.has_unconsumed() {
            let DrainOutcome { replies, close, .. } = connection.drain_async(&mut out).await;
            if replies == 0 {
                break;
            }
            if close {
                close_now = true;
                break;
            }
        }
        if !out.is_empty() {
            // The buffer is written whole and then emptied, so it only ever
            // holds this read's replies. compio's write takes the buffer by
            // value and hands it back, so the allocation is reused rather than
            // freed and re-taken every read.
            let written = stream.write_all(out).await;
            out = written.1;
            out.clear();
            if written.0.is_err() {
                exit = Some("write");
                break;
            }
        }
        if close_now {
            let _ = stream.flush().await;
            exit = Some("client close");
            break;
        }
        if out.capacity() > REPLY_BUFFER_RECLAIM {
            // One oversized value must not pin its allocation for the rest of
            // the connection's life.
            out = Vec::new();
        }
        if connection.state.version != version {
            match connection.state.version {
                RespVersion::V2 => {
                    stats.resp3_current.fetch_sub(1, Ordering::Relaxed);
                    stats.resp2_current.fetch_add(1, Ordering::Relaxed);
                }
                RespVersion::V3 => {
                    stats.resp2_current.fetch_sub(1, Ordering::Relaxed);
                    stats.resp3_current.fetch_add(1, Ordering::Relaxed);
                }
            }
            version = connection.state.version;
        }
    }
    if let Some(reason) = exit {
        tracing::debug!(%peer, reason, "RESP connection ended");
    }
    stats.connections_current.fetch_sub(1, Ordering::Relaxed);
    match version {
        RespVersion::V2 => {
            stats.resp2_current.fetch_sub(1, Ordering::Relaxed);
        }
        RespVersion::V3 => {
            stats.resp3_current.fetch_sub(1, Ordering::Relaxed);
        }
    }
    stats.fold(&connection.metrics);
}

/// One request the reactor cannot answer itself, handed to a thread that may
/// block.
struct DeferredJob {
    op: Operation,
    reply: async_channel::Sender<Result<OperationResult, ExecuteError>>,
}

/// Starts `threads` helper threads sharing `client`, returning the bounded
/// queue they drain.
///
/// A reactor thread must never park: the tasks that would answer it - the
/// bridge that drains the worker queue, the commit coordinator, the chunk lane
/// - all run on that same thread or are only reachable through it. Blocking
/// there is a deadlock, not a slow path. Helpers have no such constraint: they
/// are ordinary threads that go through the same engine queue an embedded
/// caller uses, and the reactor waits on them without parking.
///
/// Sized small on purpose. Only requests that need another thread arrive here,
/// so a busy helper means the workload is dominated by large values or
/// durability - the cases where the engine is waiting on a device or a lane
/// anyway, and where a longer queue buys nothing.
fn spawn_helpers(client: &LocalClient, threads: usize) -> Arc<async_channel::Sender<DeferredJob>> {
    let (jobs, incoming) = async_channel::bounded::<DeferredJob>(DEFERRED_BACKLOG);
    for index in 0..threads.max(1) {
        let client = client.clone();
        let incoming = incoming.clone();
        // A helper that cannot start is fatal to the edge: without it every
        // deferred request would have nowhere to go. Failing loudly here,
        // inside the accept loop, is better than answering clients with an
        // error nobody can act on.
        let spawned = std::thread::Builder::new()
            .name(format!("kivi-resp-helper-{index}"))
            .spawn(move || {
                while let Ok(job) = incoming.recv_blocking() {
                    let outcome = client.execute_op(job.op).map_err(map_engine_error);
                    // The receiver is waiting; a failure here means the
                    // connection went away mid-flight, and the answer has
                    // nowhere to go.
                    let _ = job.reply.try_send(outcome);
                }
            });
        if let Err(error) = spawned {
            tracing::error!(%error, "RESP helper thread spawn failed");
        }
    }
    Arc::new(jobs)
}

/// Runs one request on a helper and waits for its answer.
///
/// Waited on rather than fired and collected, because a RESP turn is
/// sequential: the command after this one may read what this one wrote, so
/// two helpers must never apply the same turn's commands concurrently.
///
/// A full backlog is refused rather than queued: the alternative is
/// unbounded memory for requests whose answers are device-bound anyway.
async fn run_deferred(
    helpers: &async_channel::Sender<DeferredJob>,
    op: Operation,
) -> Result<OperationResult, ExecuteError> {
    let (reply, receiver) = async_channel::bounded(1);
    helpers
        .try_send(DeferredJob { op, reply })
        .map_err(|_| ExecuteError::Overloaded)?;
    receiver
        .recv()
        .await
        .unwrap_or(Err(ExecuteError::Overloaded))
}

/// Helper threads per worker.
const HELPER_THREADS: usize = 2;

/// Backlog of requests waiting for a helper. Bounded because every queue in
/// Kivi is: an unbounded one turns a saturated device into an out-of-memory.
const DEFERRED_BACKLOG: usize = 64;

/// The facts the inline decision depends on, all gathered before execution.
#[derive(Debug, Clone, Copy)]
struct InlineFacts {
    /// A durable worker answers a write only after the device flush, so no
    /// request it accepts can be answered inline.
    durable: bool,
    /// Largest value the engine keeps inline instead of staging in chunks.
    inline_threshold: u64,
    /// Whether the key's stored root addresses chunk-lane payloads rather than
    /// resident bytes. Only a chunked read needs the lane to produce an answer.
    stored_is_chunked: bool,
}

/// Whether one request can be executed inline without ever waiting on anything
/// other than the thread executing it.
///
/// Each exclusion is a wait the reactor cannot serve - a durability proof, or
/// the chunk lane's separate thread - and every one is decided *before*
/// execution, so nothing is executed twice and nothing is left half-done.
///
/// A pure function of the operation and three facts, so the decision can be
/// tested exhaustively without standing up an engine: this is the whole
/// correctness surface of the fast path.
fn inline_eligible(op: &Operation, facts: InlineFacts) -> bool {
    if facts.durable {
        return false;
    }
    match op {
        // A value above the threshold stages through the chunk lane, which is
        // another thread and therefore another wait.
        Operation::Set { value, .. } | Operation::SetConditional { value, .. } => {
            (value.len() as u64) <= facts.inline_threshold
        }
        // A range patch may have to splice a chunked base on that lane, and
        // transactional writes stage their payloads there too.
        Operation::SetRange { .. }
        | Operation::TxnPrepare { .. }
        | Operation::TxnCommitLocal { .. } => false,
        // Everything else answers from resident state, unless the value it
        // reads is a chunked root.
        _ => !facts.stored_is_chunked,
    }
}

/// Executes RESP operations against the worker that owns the tablets.
///
/// Deliberately `!Send`: this handle is only ever valid on the reactor thread
/// that owns the state it points at, which is exactly the point.
///
/// **A reactor-served connection must use the async turn.** The synchronous
/// `Executor` methods hand anything they cannot answer to a helper and *block*
/// on it, which is correct on an ordinary thread and a deadlock on this one -
/// the helper's answer arrives via the worker queue, and the queue is drained
/// by the task this call is blocking. [`RespConnection::drain_async`] takes the
/// path that awaits instead.
struct WorkerExecutor {
    shared: Rc<WorkerNet>,
    /// Engine client for requests this worker cannot answer itself.
    forward: Arc<LocalClient>,
    /// Threads allowed to block, for those requests.
    helpers: Arc<async_channel::Sender<DeferredJob>>,
    /// Reused for the connection's life. A same-thread rendezvous never
    /// blocks, so this costs one channel send and one `try_recv` with no
    /// wakeup on either side; a fresh channel per request would allocate more
    /// than the rendezvous it replaced.
    respond: crossbeam_channel::Sender<WorkerResponse>,
    receive: crossbeam_channel::Receiver<WorkerResponse>,
}

impl WorkerExecutor {
    /// Routes the operation's key, reporting whether this worker owns the
    /// tablet it landed on.
    fn route(&self, routing: &crate::routing::RoutingSnapshot, op: &Operation) -> Option<TabletId> {
        let tablet = crate::compound::route_point_key(routing, op.key().as_bytes())?;
        (routing.placement().worker_of(tablet) == Some(self.shared.id)).then_some(tablet)
    }

    /// Gathers the facts the inline decision depends on, then asks the pure
    /// predicate.
    ///
    /// The probe is the one extra map lookup a read would have done anyway;
    /// everything else is already in hand.
    fn inline_eligible(&self, op: &Operation, tablet: TabletId) -> bool {
        let now = kivi_core::wall_now_or_max(&kivi_core::SystemClock);
        let stored_is_chunked = self
            .shared
            .tablets
            .borrow()
            .get(tablet)
            .and_then(|live| live.store().get(op.key(), now))
            .is_some_and(kivi_state::StoredObject::is_chunked);
        inline_eligible(
            op,
            InlineFacts {
                durable: self.shared.durability.is_some(),
                inline_threshold: self.shared.chunks.inline_threshold,
                stored_is_chunked,
            },
        )
    }

    /// Executes one operation inline against this worker's own state.
    fn execute_inline(
        &self,
        op: &Operation,
        tablet: TabletId,
        now: WallTimestamp,
    ) -> Result<OperationResult, ExecuteError> {
        // The reply channel is reused for the connection's life, so it is
        // emptied first. `handle_request` sends at most one answer and one is
        // always taken below, so this should find nothing - but a path that
        // answered twice would otherwise hand this request its neighbour's
        // reply, and a wrong answer in the right slot is the one failure a
        // RESP client cannot detect.
        while self.receive.try_recv().is_ok() {}
        let request = TabletRequest {
            tablet,
            op: op.clone(),
            now,
            // A RESP client speaks request/response with no retry identity, so
            // there is nothing to deduplicate: exactly-once covers the native
            // protocol's acknowledged mutations, not this edge.
            identity: None,
            respond: self.respond.clone(),
        };
        {
            let mut tablets = self.shared.tablets.borrow_mut();
            let mut fabric = self.shared.fabric.borrow_mut();
            handle_request(
                request,
                &mut tablets,
                &self.shared.metrics,
                None,
                &self.shared.chunks,
                &mut fabric,
                None,
            );
        }
        // Inline execution answers before returning, so this cannot block.
        // `try_recv` rather than `recv` is deliberate: a missed answer must
        // surface as an error, never as a park on the reactor.
        let response = self
            .receive
            .try_recv()
            .map_err(|_| ExecuteError::Overloaded)?;
        let mut snapshot = self.shared.metrics.get();
        snapshot.direct_ops += 1;
        self.shared.metrics.set(snapshot);
        // No slicing here: the store applies a range read's own window, and
        // applying it twice returns the wrong bytes. A chunked root - the one
        // case where the store names a manifest instead of a value - never
        // reaches this path, because `inline_eligible` forwards it.
        response.map_err(map_worker_error)
    }

    /// Executes a turn, keeping its commands in order across the path split.
    ///
    /// Commands this worker can answer run inline, in order, until the first one
    /// it cannot. From there on every command goes to a helper and is waited on
    /// before the next one starts.
    ///
    /// The switch is one-way on purpose. A RESP client pipelines commands it
    /// expects applied in order, and the engine expands one client command into
    /// more than one operation - `SETRANGE` becomes a write plus a length read,
    /// and that read must observe the write. Answering the length inline while
    /// the write is still on a helper is exactly how that breaks: the read wins
    /// the race and reports 0 for a 7-byte value.
    async fn run_batch(&self, ops: &[Operation]) -> Vec<Result<OperationResult, ExecuteError>> {
        let routing = self.shared.routing.load();
        let now = kivi_core::wall_now_or_max(&kivi_core::SystemClock);
        let mut results = Vec::with_capacity(ops.len());
        // Set once a command needs another thread; from there the turn is
        // sequential, because a pipelined turn's correctness is defined in
        // terms of its order.
        let mut sequential = false;
        for op in ops {
            if !sequential {
                if let Some(tablet) = self.route(&routing, op)
                    && self.inline_eligible(op, tablet)
                {
                    results.push(self.execute_inline(op, tablet, now));
                    continue;
                }
                sequential = true;
            }
            results.push(run_deferred(&self.helpers, op.clone()).await);
        }
        results
    }
}

impl Executor for WorkerExecutor {
    /// Hands the operation to the worker that owns its tablet.
    fn execute(&self, op: &Operation) -> Result<OperationResult, ExecuteError> {
        let routing = self.shared.routing.load();
        let now = kivi_core::wall_now_or_max(&kivi_core::SystemClock);
        match self.route(&routing, op) {
            Some(tablet) if self.inline_eligible(op, tablet) => {
                self.execute_inline(op, tablet, now)
            }
            _ => self
                .forward
                .execute_op(op.clone())
                .map_err(map_engine_error),
        }
    }

    /// The synchronous turn. A request this worker cannot answer itself is
    /// handed to a helper and waited on, which is correct on an ordinary
    /// thread and is why a reactor-served connection uses
    /// [`Self::run_batch`] instead: parking here would park the reactor.
    fn execute_batch(&self, ops: &[Operation]) -> Vec<Result<OperationResult, ExecuteError>> {
        let routing = self.shared.routing.load();
        let now = kivi_core::wall_now_or_max(&kivi_core::SystemClock);
        ops.iter()
            .map(|op| match self.route(&routing, op) {
                Some(tablet) if self.inline_eligible(op, tablet) => {
                    self.execute_inline(op, tablet, now)
                }
                _ => self
                    .forward
                    .execute_op(op.clone())
                    .map_err(map_engine_error),
            })
            .collect()
    }

    fn now(&self) -> WallTimestamp {
        kivi_core::wall_now_or_max(&kivi_core::SystemClock)
    }

    /// The single-tablet route this worker owns, asked once per drain rather
    /// than once per command: a pipeline shares one answer, and asking per
    /// command is two `arc-swap` loads and a version compare multiplied by the
    /// pipeline depth. `None` means the general path routes it.
    fn direct_route(&self) -> Option<kivi_types::TabletRoute> {
        self.shared.local_routes().single()
    }

    /// Answers a common read directly from the tablet a compiled route names.
    ///
    /// No semantics are reimplemented: the value, length, existence answer and
    /// TTL all come from the store's own accessors, so there is no second
    /// authority for what `GET` means. Every `None` is a correctness-preserving
    /// fallback - a key this worker does not own, a chunked or fabric-backed
    /// value whose resolution suspends, or a slot vacated mid-pipeline - and the
    /// general path routes against the new publication.
    fn fast_read(
        &self,
        route: kivi_types::TabletRoute,
        kind: kivi_resp::fast::FastKind,
        key: &[u8],
        now: kivi_types::WallTimestamp,
    ) -> Option<kivi_resp::Reply> {
        // The route arrived already resolved against this worker's slots, so
        // reaching the tablet is a bounds check. The borrow guard is bound to a
        // name rather than dropped at the end of a chain: `self.shared.tablets`
        // is a `RefCell`, so the `borrow()` has to outlive the `&ObjectStore` it
        // hands out.
        let tablets = self.shared.tablets.borrow();
        let live = tablets.by_slot(route.slot())?.store();
        let stored = live.get_borrowed(key, now);
        Some(match (kind, stored) {
            // A value that is not inline bytes cannot be answered without
            // resolving it, which is the fallback's job.
            (kivi_resp::fast::FastKind::Get, Some(object)) => {
                let kivi_state::LogicalValue::Bytes(bytes) = object.value() else {
                    return None;
                };
                kivi_resp::Reply::Bulk(bytes.clone())
            }
            (kivi_resp::fast::FastKind::Get, None) => kivi_resp::Reply::Nil,
            (kivi_resp::fast::FastKind::Exists, found) => {
                kivi_resp::Reply::Int(i64::from(found.is_some()))
            }
            (kivi_resp::fast::FastKind::StrLen, Some(object)) => {
                let kivi_state::LogicalValue::Bytes(bytes) = object.value() else {
                    return None;
                };
                // The same saturation the general path's `Length` mapping uses,
                // so the two agree on a value longer than `i64::MAX`.
                kivi_resp::Reply::Int(i64::try_from(bytes.len()).unwrap_or(i64::MAX))
            }
            (kivi_resp::fast::FastKind::StrLen, None) => kivi_resp::Reply::Int(0),
            // The TTL rounding is `kivi-resp`'s, not a second copy: Redis's
            // round-up-to-seconds and the one-second grace for an expired but
            // unreclaimed value are properties of the command.
            (kivi_resp::fast::FastKind::Ttl, found) => kivi_resp::Reply::Int(
                kivi_resp::translate::ttl_seconds(found.map(kivi_state::StoredObject::expiry), now),
            ),
            (kivi_resp::fast::FastKind::Pttl, found) => kivi_resp::Reply::Int(
                kivi_resp::translate::ttl_millis(found.map(kivi_state::StoredObject::expiry), now),
            ),
        })
    }

    fn execute_batch_async<'a>(
        &'a self,
        ops: &'a [Operation],
    ) -> kivi_resp::translate::BatchFuture<'a> {
        Box::pin(self.run_batch(ops))
    }
}

/// Maps a worker-side failure onto the RESP error surface. RESP must not leak
/// Rust internals, so every arm names a category the reply layer can render.
///
/// Takes the error by value because the caller is handing over an error it will
/// never look at again; a borrow here would only invite it to keep one.
#[allow(clippy::needless_pass_by_value)]
fn map_worker_error(error: crate::worker::WorkerRequestError) -> ExecuteError {
    use crate::TabletError as T;
    use crate::worker::WorkerRequestError as E;
    use kivi_state::{ApplyError as A, OpError as O};
    match error {
        E::Tablet(T::Op(O::WrongType { .. }) | T::Apply(A::TypeMismatch { .. })) => {
            ExecuteError::WrongType
        }
        E::DedupExpired
        | E::Tablet(
            T::Op(O::CounterOverflow)
            | T::Apply(A::CounterOverflow | A::VersionExhausted)
            | T::CommitExhausted { .. },
        ) => ExecuteError::Rejected,
        E::Tablet(T::Op(O::StaleRangeBase)) | E::Storage(_) | E::ChunkStore(_) | E::Fabric(_) => {
            ExecuteError::Overloaded
        }
        _ => ExecuteError::Internal,
    }
}

/// Maps an engine-side failure for the forwarded path. Same categories as
/// [`map_worker_error`]: one place decides what a RESP client is told, so the
/// two paths cannot drift into answering the same failure differently.
#[allow(clippy::needless_pass_by_value)]
fn map_engine_error(error: crate::EngineError) -> ExecuteError {
    use crate::EngineError as E;
    use crate::TabletError as T;
    use kivi_state::{ApplyError as A, OpError as O};
    match error {
        E::Overloaded | E::Tablet(T::Op(O::StaleRangeBase)) => ExecuteError::Overloaded,
        E::InvalidRequest { .. }
        | E::DedupExpired
        | E::SessionOverloaded
        | E::Tablet(
            T::Op(O::CounterOverflow)
            | T::Apply(A::CounterOverflow | A::VersionExhausted)
            | T::CommitExhausted { .. },
        ) => ExecuteError::Rejected,
        E::Tablet(T::Op(O::WrongType { .. }) | T::Apply(A::TypeMismatch { .. })) => {
            ExecuteError::WrongType
        }
        E::Tablet(T::Apply(A::UnresolvableSplice) | T::AuthorityMismatch { .. }) => {
            ExecuteError::Internal
        }
        _ => ExecuteError::Internal,
    }
}

#[cfg(test)]
mod tests {
    use super::{InlineFacts, inline_eligible};
    use bytes::Bytes;
    use kivi_state::{ExpiryPolicy, Key, Operation, SetCondition};

    const NS: kivi_types::NamespaceId = kivi_types::NamespaceId::from_u64(1);

    fn key(name: &str) -> Key {
        Key::from(format!("{NS}:resp-gate:{name}"))
    }

    fn facts() -> InlineFacts {
        InlineFacts {
            durable: false,
            inline_threshold: 1024,
            stored_is_chunked: false,
        }
    }

    fn get(name: &str) -> Operation {
        Operation::Get { key: key(name) }
    }

    fn set(name: &str, value: &[u8]) -> Operation {
        Operation::Set {
            key: key(name),
            value: Bytes::copy_from_slice(value),
        }
    }

    /// The gate is the entire correctness surface of the fast path, so it is
    /// pinned by facts rather than by running a server: each case below is a
    /// way the inline path could have answered something it could not have
    /// waited for.
    #[test]
    fn small_reads_and_writes_take_the_fast_path() {
        assert!(inline_eligible(&get("a"), facts()));
        assert!(inline_eligible(&set("a", b"v"), facts()));
        assert!(inline_eligible(
            &Operation::Delete { key: key("a") },
            facts()
        ));
        assert!(inline_eligible(
            &Operation::Exists { key: key("a") },
            facts()
        ));
        assert!(inline_eligible(
            &Operation::BytesLength { key: key("a") },
            facts()
        ));
    }

    /// A value at exactly the threshold is still inline; one byte more is not.
    /// An off-by-one here would either stall the reactor or push small values
    /// onto the slow path, so the boundary is checked from both sides.
    #[test]
    fn the_value_threshold_is_inclusive() {
        let at = vec![b'x'; 1024];
        let over = vec![b'x'; 1025];
        assert!(inline_eligible(&set("a", &at), facts()));
        assert!(!inline_eligible(&set("a", &over), facts()));
    }

    /// A conditional write is a write: it stages the same way a plain one does.
    #[test]
    fn large_conditional_writes_are_forwarded() {
        let small = Operation::SetConditional {
            key: key("a"),
            value: Bytes::from_static(b"v"),
            condition: SetCondition::Always,
            expiry: ExpiryPolicy::Clear,
        };
        let large = Operation::SetConditional {
            key: key("a"),
            value: Bytes::from(vec![b'x'; 4096]),
            condition: SetCondition::Always,
            expiry: ExpiryPolicy::Clear,
        };
        assert!(inline_eligible(&small, facts()));
        assert!(!inline_eligible(&large, facts()));
    }

    /// Reading a chunked root needs the lane's separate thread to produce the
    /// bytes, so it is forwarded however small the window is.
    #[test]
    fn a_chunked_read_is_forwarded() {
        let chunked = InlineFacts {
            stored_is_chunked: true,
            ..facts()
        };
        assert!(!inline_eligible(&get("a"), chunked));
        assert!(!inline_eligible(
            &Operation::GetRange {
                key: key("a"),
                offset: 0,
                len: 1,
            },
            chunked
        ));
    }

    /// A range patch may have to splice a chunked base, so it never runs
    /// inline regardless of the size of its patch.
    #[test]
    fn range_patches_and_transactions_are_always_forwarded() {
        assert!(!inline_eligible(
            &Operation::SetRange {
                key: key("a"),
                offset: 0,
                patch: Bytes::from_static(b"p"),
            },
            facts()
        ));
    }

    /// A durable worker answers after the flush. One exception here would park
    /// the reactor on a reply only the reactor can produce, so the gate is
    /// checked for every operation shape, not just writes.
    #[test]
    fn durable_workers_forward_everything() {
        let durable = InlineFacts {
            durable: true,
            ..facts()
        };
        for op in [
            get("a"),
            set("a", b"v"),
            Operation::Delete { key: key("a") },
            Operation::Exists { key: key("a") },
            Operation::GetRange {
                key: key("a"),
                offset: 0,
                len: 8,
            },
        ] {
            assert!(
                !inline_eligible(&op, durable),
                "durable worker must forward {op:?}"
            );
        }
    }
}
