//! Native TCP serving on `DataWorker`s: one Compio runtime per worker thread.
//!
//! Each networked worker owns an OS thread running a single-threaded Compio
//! runtime. The accept loop and every connection task execute on that same
//! thread and share tablets through `Rc<RefCell<..>>` — no locks, no
//! cross-thread tablet access, ever. (`spawn` requires only `Future`, never
//! `Send`, which is what makes this sound.)
//!
//! Request flow on the owning worker (the fast path):
//!
//! ```text
//! socket → frame parser → route validation → LiveTablet::execute
//!        → response encode → socket
//! ```
//!
//! No channel hop, no dispatcher, no other worker. The embedded
//! `LocalClient` channel path keeps working alongside (bridged below) for
//! embedded use and tests, and per-path counters prove which path served
//! what.
//!
//! Bounds (all explicit, none unbounded): input bytes per connection,
//! frame size, per-turn frames/bytes with executor-agnostic yields, and
//! synchronous response writes (no response queue exists to bound).
//! Overload policy: malformed input closes the connection; semantic
//! failures answer with stable statuses; deep pipelines are paced by turn
//! budgets, never by unbounded queues.
//!
//! Reactor boundary: Compio types appear only in this module (plus the
//! affinity helper beside it). Swapping reactors later reimplements this
//! file against the same tablet/routing/protocol vocabulary.

use core::cell::{Cell, RefCell};
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use core::time::Duration;
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::rc::Rc;
use std::sync::Arc;

use arc_swap::ArcSwap;
use compio::io::{AsyncRead, AsyncWriteExt};
use compio::net::{TcpListener, TcpStream};
use compio::runtime::Runtime;

use kivi_protocol::{
    Capabilities, ClientHello, DEFAULT_MAX_FRAME, FrameReader, RedirectInfo, Request, Response,
    ServerHello, encode_frame,
};
use kivi_state::{Operation, PartitionHasher};
use kivi_types::{ClusterId, MutationIdentity, NodeId, NodeIncarnation, TabletId, WorkerId};

use crate::affinity::{AffinityError, AffinityMode, pin_current_thread};
use crate::clock::SystemClock;
use crate::routing::RoutingSnapshot;
use crate::tablet::LiveTablet;
use crate::worker::{
    DurableFailure, WorkerControl, WorkerDurability, WorkerMetrics, execute_durable,
    handle_control, handle_request,
};

/// Per-connection input bound default (4 MiB of unparsed bytes).
pub const DEFAULT_MAX_INPUT_BYTES: usize = 4 * 1024 * 1024;
/// Decoded-but-not-executed backlog bound default. Pipelining deeper than
/// this closes the connection instead of queueing without limit.
pub const DEFAULT_MAX_PIPELINED: usize = 1024;
/// Per-turn frame budget default: pace deep pipelines without per-op yields.
pub const DEFAULT_TURN_FRAMES: usize = 32;
/// Per-turn byte budget default (256 KiB of request payload).
pub const DEFAULT_TURN_BYTES: usize = 256 * 1024;
/// Embedded-path park interval: bounds idle CPU and caps pickup latency
/// for channel messages on networked workers. `LocalClient` traffic against a
/// networked worker is an admin/test path (never the fast path), so a short
/// park is the honest trade: guaranteed reactor progress with no busy spin.
/// Channel-only workers keep their blocking `select!` loop with zero added
/// latency; benchmarks of the embedded path use those.
const BRIDGE_PARK_INTERVAL: Duration = Duration::from_millis(10);
/// Accept-loop shutdown poll interval.
const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(100);
/// Startup handshake timeout (bind + runtime creation must report fast).
pub(crate) const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);

/// Connection-level bounds, all explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConnLimits {
    /// Maximum unparsed input bytes buffered per connection.
    pub max_input_bytes: usize,
    /// Maximum accepted frame payload (≤ 64 MiB protocol ceiling).
    pub max_frame: usize,
    /// Maximum decoded-but-not-executed frames held per connection.
    pub max_pipelined: usize,
}

impl Default for ConnLimits {
    /// Sane loopback/LAN defaults; tune per deployment after measuring.
    fn default() -> Self {
        Self {
            max_input_bytes: DEFAULT_MAX_INPUT_BYTES,
            max_frame: DEFAULT_MAX_FRAME,
            max_pipelined: DEFAULT_MAX_PIPELINED,
        }
    }
}

/// Per-turn fairness budgets for one connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnBudget {
    /// Frames executed before yielding to the runtime.
    pub max_frames: usize,
    /// Request payload bytes executed before yielding.
    pub max_bytes: usize,
}

impl Default for TurnBudget {
    /// Batches enough work to stay efficient without monopolizing the worker.
    fn default() -> Self {
        Self {
            max_frames: DEFAULT_TURN_FRAMES,
            max_bytes: DEFAULT_TURN_BYTES,
        }
    }
}

/// Worker endpoint directory: worker id to dialable `"ip:port"`.
pub type EndpointMap = std::collections::BTreeMap<WorkerId, String>;

/// Engine-level network configuration (shared across workers).
#[derive(Debug, Clone)]
pub struct EngineNetwork {
    /// Base TCP port; worker `i` listens on `base_port + i`. `0` selects an
    /// ephemeral port per worker (not `0 + i`). Ignored when `ports` is set.
    pub base_port: u16,
    /// Explicit per-worker ports, bypassing base-port arithmetic. Must name
    /// exactly one port per worker when present (used by tests restarting on
    /// fixed addresses, and by operators pinning ports explicitly).
    pub ports: Vec<u16>,
    /// IP to bind (advertised as 127.0.0.1 when unspecified).
    pub bind_ip: IpAddr,
    /// Engine-wide maximum frame (clamped to the 64 MiB protocol ceiling).
    pub max_frame: usize,
    /// CPU pinning for worker threads.
    pub affinity: AffinityMode,
    /// This node's identity.
    pub node_id: NodeId,
    /// This cluster's identity.
    pub cluster_id: ClusterId,
    /// This incarnation (advanced durably on every restart; `INITIAL` when
    /// ephemeral). Advertised in handshakes and stamped into new segments.
    pub incarnation: NodeIncarnation,
    /// Per-connection bounds.
    pub conn: ConnLimits,
    /// Per-turn fairness budgets.
    pub turn: TurnBudget,
}

/// Per-worker resolved network configuration.
#[derive(Debug, Clone)]
pub struct NetConfig {
    /// This worker's identity.
    pub worker: WorkerId,
    /// Socket to listen on (port may be zero for ephemeral assignment).
    pub listen: SocketAddr,
    /// IP advertised in endpoints (loopback for unspecified binds).
    pub advertise_ip: IpAddr,
    /// Connection bounds.
    pub conn: ConnLimits,
    /// Fairness budgets.
    pub turn: TurnBudget,
    /// Node and cluster identity for handshakes.
    pub node_id: NodeId,
    /// Cluster identity for handshakes.
    pub cluster_id: ClusterId,
    /// This incarnation of the node (advanced durably on every restart).
    pub incarnation: NodeIncarnation,
    /// Capabilities advertised in `ServerHello` (`DURABLE_MUTATION_DEDUP`
    /// exactly when the engine runs durable).
    pub advertised_caps: Capabilities,
    /// Live worker endpoint directory (swapped once binds complete).
    pub endpoints: Arc<ArcSwap<EndpointMap>>,
    /// Live routing publication.
    pub routing: Arc<ArcSwap<RoutingSnapshot>>,
}

/// Networked worker startup failure: affinity, runtime, or bind problems
/// report here before serving begins — never as a silent dead worker.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum NetStartError {
    /// CPU pinning failed.
    #[error("worker affinity failed: {0}")]
    Affinity(#[source] AffinityError),
    /// Runtime creation failed.
    #[error("runtime creation failed: {0}")]
    Runtime(String),
    /// Listen/bind failed.
    #[error("listen/bind failed: {0}")]
    Bind(String),
    /// Startup handshake timed out or the thread died first.
    #[error("worker startup failed: {0}")]
    Startup(String),
}

/// Executor-agnostic cooperative yield: returns `Pending` once (after
/// re-arming the waker) so the reactor can run other tasks. Used for turn
/// budgets and shutdown draining without coupling to any runtime's yield API.
#[derive(Debug, Default)]
struct YieldNow(bool);

impl Future for YieldNow {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.0 {
            Poll::Ready(())
        } else {
            self.0 = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

/// Shared per-worker serving state (single-threaded; `Rc` never crosses threads).
struct WorkerNet {
    id: WorkerId,
    tablets: Rc<RefCell<HashMap<TabletId, LiveTablet>>>,
    metrics: Rc<Cell<WorkerMetrics>>,
    routing: Arc<ArcSwap<RoutingSnapshot>>,
    endpoints: Arc<ArcSwap<EndpointMap>>,
    shutdown: Rc<Cell<bool>>,
    active: Rc<Cell<usize>>,
    net: NetConfig,
    /// Durable state (`None` in ephemeral mode). Shared across this
    /// worker's connection tasks and bridge: single-threaded `Rc`, and
    /// every use is synchronous (no `.await` while borrowed), so tasks
    /// cannot interleave a persist-apply sequence.
    durability: Option<Rc<RefCell<WorkerDurability>>>,
}

/// Launch bundle for one networked worker: everything `spawn_net` needs
/// beyond identity, tablets, and queue depth. Grouped so constructor
/// argument lists stay reviewable as launch context grows.
#[derive(Debug)]
pub struct NetLaunch {
    /// Live routing publication (shared with the engine).
    pub routing: Arc<ArcSwap<RoutingSnapshot>>,
    /// Resolved per-worker network configuration.
    pub net: NetConfig,
    /// CPU pinning for the worker thread.
    pub affinity: AffinityMode,
    /// This worker's index for affinity distribution.
    pub worker_index: usize,
    /// Total worker count for affinity distribution.
    pub worker_count: usize,
}

/// Thread body for a networked `DataWorker`: affinity, runtime, listener,
/// accept loop, and the crossbeam bridge for the embedded path.
pub(crate) fn run_net(
    id: WorkerId,
    tablets: Vec<LiveTablet>,
    requests: crossbeam_channel::Receiver<crate::worker::TabletRequest>,
    control: crossbeam_channel::Receiver<WorkerControl>,
    durability: Option<WorkerDurability>,
    launch: NetLaunch,
    ready: std::sync::mpsc::Sender<Result<SocketAddr, NetStartError>>,
) {
    if let Err(error) =
        pin_current_thread(&launch.affinity, launch.worker_index, launch.worker_count)
    {
        let _ = ready.send(Err(NetStartError::Affinity(error)));
        return;
    }
    let tablet_map: HashMap<TabletId, LiveTablet> = tablets
        .into_iter()
        .map(|tablet| (tablet.id(), tablet))
        .collect();
    let shared = Rc::new(WorkerNet {
        id,
        tablets: Rc::new(RefCell::new(tablet_map)),
        metrics: Rc::new(Cell::new(WorkerMetrics::default())),
        routing: launch.routing,
        endpoints: launch.net.endpoints.clone(),
        shutdown: Rc::new(Cell::new(false)),
        active: Rc::new(Cell::new(0)),
        net: launch.net,
        durability: durability.map(|durable| Rc::new(RefCell::new(durable))),
    });
    let runtime = match Runtime::new() {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = ready.send(Err(NetStartError::Runtime(error.to_string())));
            return;
        }
    };
    runtime.block_on(serve(shared, requests, control, ready));
}

/// Top-level serving future: bind, report, accept until shutdown, drain.
async fn serve(
    shared: Rc<WorkerNet>,
    requests: crossbeam_channel::Receiver<crate::worker::TabletRequest>,
    control: crossbeam_channel::Receiver<WorkerControl>,
    ready: std::sync::mpsc::Sender<Result<SocketAddr, NetStartError>>,
) {
    let listener = match TcpListener::bind(shared.net.listen).await {
        Ok(listener) => listener,
        Err(error) => {
            let _ = ready.send(Err(NetStartError::Bind(error.to_string())));
            return;
        }
    };
    let bound = match listener.local_addr() {
        Ok(addr) => addr,
        Err(error) => {
            let _ = ready.send(Err(NetStartError::Bind(error.to_string())));
            return;
        }
    };
    tracing::info!(worker = %shared.id.as_u64(), addr = %bound, "worker listening");
    let _ = ready.send(Ok(bound));
    // The embedded channel bridge runs beside the accept loop so LocalClient
    // keeps working on networked workers through the same tablets.
    let bridge = {
        let tablets = Rc::clone(&shared.tablets);
        let metrics = Rc::clone(&shared.metrics);
        let shutdown = Rc::clone(&shared.shutdown);
        let durability = shared.durability.clone();
        async move {
            bridge_loop(requests, control, tablets, metrics, durability, shutdown).await;
        }
    };
    compio::runtime::spawn(bridge).detach();
    // Accept loop with shutdown polling: a blocked accept would otherwise
    // delay shutdown indefinitely.
    loop {
        if shared.shutdown.get() {
            break;
        }
        match compio::time::timeout(ACCEPT_POLL_INTERVAL, listener.accept()).await {
            Ok(Ok((stream, peer))) => {
                tracing::debug!(worker = %shared.id.as_u64(), %peer, "accepted connection");
                spawn_conn(stream, peer, Rc::clone(&shared));
            }
            Ok(Err(error)) => {
                tracing::warn!(worker = %shared.id.as_u64(), %error, "accept failed");
                compio::time::sleep(Duration::from_millis(10)).await;
            }
            Err(_) => {}
        }
    }
    // Bounded drain: connections observe shutdown through their read poll
    // (≤ one accept interval) and exit on their own, so the runtime never
    // drops under live sockets. Whatever lingers past the grace period is
    // force-closed by the runtime drop that follows.
    for _ in 0..100 {
        if shared.active.get() == 0 {
            break;
        }
        compio::time::sleep(Duration::from_millis(10)).await;
    }
    tracing::info!(worker = %shared.id.as_u64(), "worker listener stopped");
}

/// Embedded-path bridge: serves the crossbeam request/control channels from
/// inside the runtime thread, sharing its tablets. Bounded per turn like
/// connection I/O; exits on Shutdown or when the engine drops every sender
/// (which also flips the listener shutdown so the thread can join).
async fn bridge_loop(
    requests: crossbeam_channel::Receiver<crate::worker::TabletRequest>,
    control: crossbeam_channel::Receiver<WorkerControl>,
    tablets: Rc<RefCell<HashMap<TabletId, LiveTablet>>>,
    metrics: Rc<Cell<WorkerMetrics>>,
    durability: Option<Rc<RefCell<WorkerDurability>>>,
    shutdown: Rc<Cell<bool>>,
) {
    use crossbeam_channel::TryRecvError;
    let mut requests_dead = false;
    let mut control_dead = false;
    loop {
        loop {
            match control.try_recv() {
                Ok(message) => {
                    let durable = durability.as_ref().map(|cell| cell.borrow());
                    let durable_ref = durable.as_deref();
                    if !handle_control(message, &mut tablets.borrow_mut(), &metrics, durable_ref) {
                        shutdown.set(true);
                        return;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    control_dead = true;
                    break;
                }
            }
        }
        for _ in 0..64 {
            match requests.try_recv() {
                Ok(request) => {
                    let mut durable = durability.as_ref().map(|cell| cell.borrow_mut());
                    let durable_ref = durable.as_deref_mut();
                    handle_request(request, &mut tablets.borrow_mut(), &metrics, durable_ref);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    requests_dead = true;
                    break;
                }
            }
        }
        if requests_dead && control_dead {
            shutdown.set(true);
            return;
        }
        // Always park: an executor-agnostic wake-loop here would spin the
        // thread and can starve every other task on a single-threaded
        // reactor. Parking bounds idle CPU and guarantees connection tasks,
        // timers, and completions all progress.
        compio::time::sleep(BRIDGE_PARK_INTERVAL).await;
    }
}

/// Spawns one connection task sharing this worker's tablets.
fn spawn_conn(stream: TcpStream, peer: SocketAddr, shared: Rc<WorkerNet>) {
    compio::runtime::spawn(serve_conn(stream, peer, shared)).detach();
}

/// Guards the active-connection count against every exit path.
struct ActiveGuard {
    active: Rc<Cell<usize>>,
}

impl ActiveGuard {
    fn hold(active: &Rc<Cell<usize>>) -> Self {
        active.set(active.get() + 1);
        Self {
            active: Rc::clone(active),
        }
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.active.set(self.active.get().saturating_sub(1));
    }
}

/// Per-connection serving state.
struct Conn {
    stream: TcpStream,
    worker: WorkerId,
    tablets: Rc<RefCell<HashMap<TabletId, LiveTablet>>>,
    metrics: Rc<Cell<WorkerMetrics>>,
    routing: Arc<ArcSwap<RoutingSnapshot>>,
    endpoints: Arc<ArcSwap<EndpointMap>>,
    shutdown: Rc<Cell<bool>>,
    node_id: NodeId,
    cluster_id: ClusterId,
    incarnation: NodeIncarnation,
    advertised_caps: Capabilities,
    durability: Option<Rc<RefCell<WorkerDurability>>>,
    reader: FrameReader,
    pending: std::collections::VecDeque<kivi_protocol::Frame>,
    max_input: usize,
    max_pipelined: usize,
    turn: TurnBudget,
    shook_hands: bool,
    frames_this_turn: usize,
    bytes_this_turn: usize,
}

/// Why a connection task ended. `Clean` (EOF) is silent; violations
/// and I/O failures trace at warn/debug respectively.
enum ConnExit {
    Clean,
    Violation(&'static str),
    Io,
}

impl Conn {
    /// Serves one connection from handshake to close.
    async fn run(&mut self) -> ConnExit {
        if let Err(exit) = self.handshake().await {
            return exit;
        }
        loop {
            if self.shutdown.get() {
                return ConnExit::Clean;
            }
            let frame = match self.next_frame().await {
                Ok(frame) => frame,
                Err(exit) => return exit,
            };
            self.frames_this_turn += 1;
            self.bytes_this_turn += frame.payload.len();
            if let Err(exit) = self.handle_frame(frame).await {
                return exit;
            }
            if self.frames_this_turn >= self.turn.max_frames
                || self.bytes_this_turn >= self.turn.max_bytes
            {
                self.frames_this_turn = 0;
                self.bytes_this_turn = 0;
                YieldNow::default().await;
            }
        }
    }

    /// Reads the next frame, enforcing the input and backlog bounds. EOF is
    /// clean; violations and I/O errors end the connection.
    async fn next_frame(&mut self) -> Result<kivi_protocol::Frame, ConnExit> {
        loop {
            if let Some(frame) = self.pending.pop_front() {
                return Ok(frame);
            }
            // Pull more bytes only while the backlog has room: a client
            // pipelining deeper than max_pipelined gets closed, never an
            // unbounded queue. The read carries the accept-loop poll
            // interval so an idle pooled connection still observes shutdown:
            // without it, graceful close would drop the runtime under live
            // sockets and leak the listener binding.
            if self.reader.buffered() >= self.max_input {
                return Err(ConnExit::Violation("input buffer bound"));
            }
            let chunk = Vec::with_capacity(8192);
            let outcome =
                compio::time::timeout(ACCEPT_POLL_INTERVAL, self.stream.read(chunk)).await;
            let (result, chunk) = match outcome {
                Err(_) => {
                    if self.shutdown.get() {
                        return Err(ConnExit::Clean);
                    }
                    continue;
                }
                Ok(outcome) => (outcome.0, outcome.1),
            };
            match result {
                Ok(0) => return Err(ConnExit::Clean),
                Ok(_) => {
                    let frames = self
                        .reader
                        .push(&chunk)
                        .map_err(|_| ConnExit::Violation("frame parse"))?;
                    if self.pending.len() + frames.len() > self.max_pipelined {
                        return Err(ConnExit::Violation("pipelined backlog bound"));
                    }
                    self.pending.extend(frames);
                }
                Err(_) => return Err(ConnExit::Io),
            }
        }
    }

    /// Performs the handshake: exactly one `ClientHello` first, then our
    /// `ServerHello`. Any deviation closes the connection.
    async fn handshake(&mut self) -> Result<(), ConnExit> {
        let frame = self.next_frame().await?;
        if frame.kind != kivi_protocol::FrameKind::ClientHello {
            return Err(ConnExit::Violation("first frame must be ClientHello"));
        }
        let hello = ClientHello::decode(&frame.payload)
            .map_err(|_| ConnExit::Violation("malformed ClientHello"))?;
        // Framing already gated the major/minor versions before this point;
        // the handshake only negotiates capabilities and bounds.
        if !hello.required_caps.unknown_required().is_empty() {
            return Err(ConnExit::Violation("unknown required capabilities"));
        }
        if hello.max_frame == 0 {
            return Err(ConnExit::Violation("zero max frame"));
        }
        let server_max = u32::try_from(self.reader.max_frame()).unwrap_or(u32::MAX);
        let max_frame = hello.max_frame.min(server_max);
        self.reader = FrameReader::new(max_frame as usize);
        let routing = self.routing.load();
        let reply = ServerHello {
            major: kivi_protocol::PROTOCOL_MAJOR,
            minor: kivi_protocol::PROTOCOL_MINOR,
            caps: self.advertised_caps,
            cluster: self.cluster_id,
            node: self.node_id(),
            incarnation: self.incarnation,
            worker: self.worker,
            dir_version: routing.version(),
            max_frame,
            endpoints: self
                .endpoints
                .load()
                .iter()
                .map(|(worker, addr)| (*worker, addr.clone()))
                .collect(),
        };
        drop(routing);
        self.write_frame(kivi_protocol::FrameKind::ServerHello, 0, &reply.encode())
            .await?;
        self.shook_hands = true;
        Ok(())
    }

    /// Handles one post-handshake frame.
    async fn handle_frame(&mut self, frame: kivi_protocol::Frame) -> Result<(), ConnExit> {
        use kivi_protocol::FrameKind;
        match frame.kind {
            FrameKind::Ping => {
                self.write_frame(FrameKind::Pong, frame.request_id, &[])
                    .await?;
                Ok(())
            }
            FrameKind::Request => self.handle_request(frame.request_id, &frame.payload).await,
            _ => Err(ConnExit::Violation("unexpected frame kind")),
        }
    }

    /// Routes, executes, and answers one request directly on this worker.
    async fn handle_request(&mut self, request_id: u64, payload: &[u8]) -> Result<(), ConnExit> {
        use kivi_protocol::{Response, ResponseBody, Status};
        let request =
            Request::decode(payload).map_err(|_| ConnExit::Violation("malformed request"))?;
        let namespace = self.routing.load().namespace();
        if request.namespace != namespace {
            return self
                .respond(
                    request_id,
                    request.opcode,
                    Response {
                        status: Status::InvalidRequest,
                        body: ResponseBody::Diagnostic("unknown namespace".to_owned()),
                    },
                )
                .await;
        }
        let hash = PartitionHasher::V1
            .hash(namespace, &request.key)
            .ok_or(ConnExit::Io)?;
        let (tablet, owner) = match self.route(hash) {
            Route::Execute(tablet) => (tablet, self.worker),
            Route::Redirect(info) => {
                let status = if info.worker == self.worker {
                    Status::NotLocal
                } else {
                    Status::StaleRoute
                };
                // Redirects are rare (once per range per client until cached),
                // so a debug event here is signal, not hot-path noise.
                tracing::debug!(
                    worker = self.worker.as_u64(),
                    tablet = info.tablet.as_u64(),
                    owner = info.worker.as_u64(),
                    dir_version = info.dir_version.as_u64(),
                    status = %status,
                    "route redirect"
                );
                return self
                    .respond(
                        request_id,
                        request.opcode,
                        Response {
                            status,
                            body: ResponseBody::Redirect(info),
                        },
                    )
                    .await;
            }
            Route::Nowhere => {
                return self
                    .respond(
                        request_id,
                        request.opcode,
                        Response {
                            status: Status::NotLocal,
                            body: ResponseBody::Diagnostic("no tablet covers key".to_owned()),
                        },
                    )
                    .await;
            }
        };
        let _ = owner;
        if let Some(info) = self.stale_hint_redirect(tablet, request.hint) {
            return self
                .respond(
                    request_id,
                    request.opcode,
                    Response {
                        status: Status::StaleRoute,
                        body: ResponseBody::Redirect(info),
                    },
                )
                .await;
        }
        self.handle_routed_request(request_id, request, tablet)
            .await
    }

    /// Executes one routed request on this worker: durable pipeline when
    /// the lane exists, direct in-memory execution otherwise.
    async fn handle_routed_request(
        &mut self,
        request_id: u64,
        request: Request,
        tablet: TabletId,
    ) -> Result<(), ConnExit> {
        let identity = request.identity;
        let ack_floor = request.ack_floor;
        let operation = request.into_operation();
        let opcode = kivi_protocol::operation_opcode(&operation);
        if self.durability.is_some() {
            return self
                .handle_request_durable(request_id, identity, ack_floor, tablet, &operation, opcode)
                .await;
        }
        match self.execute_local(tablet, &operation) {
            None => {
                self.respond(
                    request_id,
                    opcode,
                    kivi_protocol::Response {
                        status: kivi_protocol::Status::NotLocal,
                        body: kivi_protocol::ResponseBody::Diagnostic(
                            "tablet not live on owner".to_owned(),
                        ),
                    },
                )
                .await
            }
            Some(Ok(result)) => {
                bump_direct(&self.metrics);
                self.respond_ok(request_id, opcode, &result).await
            }
            Some(Err(error)) => self.respond_op_error(request_id, opcode, &error).await,
        }
    }

    /// Durable write pipeline on the owning worker: dedup check, prepare,
    /// exactly one WAL record, apply + verify (or install), reply. Never
    /// replies success before the record is file-synced. Tablet and lane
    /// borrows both end before any `.await`, so connection tasks on this
    /// thread cannot interleave a persist-apply sequence.
    async fn handle_request_durable(
        &mut self,
        request_id: u64,
        identity: Option<kivi_types::RequestIdentity>,
        ack_floor: kivi_types::RequestSeq,
        tablet: TabletId,
        operation: &Operation,
        opcode: kivi_protocol::Opcode,
    ) -> Result<(), ConnExit> {
        use kivi_protocol::{Response, ResponseBody, Status};
        // Identity rule: mutating requests must carry client identity on a
        // durable server (the safe-retry contract); reads never need it and
        // any identity they carry is ignored.
        let identity = match (identity, opcode.is_mutating()) {
            (Some(identity), true) => Some(MutationIdentity::new(identity, ack_floor)),
            (None, true) => {
                return self
                    .respond(
                        request_id,
                        opcode,
                        Response {
                            status: Status::InvalidRequest,
                            body: ResponseBody::Diagnostic(
                                "mutating request lacks client identity".to_owned(),
                            ),
                        },
                    )
                    .await;
            }
            _ => None,
        };
        let now = SystemClock::wall_now();
        // Resolve the tablet and run the whole pipeline synchronously:
        // both borrows end before any `.await`, so connection tasks on
        // this thread cannot interleave a persist-apply sequence. The
        // double lookup is deliberate: holding any borrow across the
        // response `.await` below would panic a sibling task that
        // borrows while we wait on the socket (same thread, no
        // preemption between the two borrows, so no race).
        if !self.tablets.borrow().contains_key(&tablet) {
            return self
                .respond(
                    request_id,
                    opcode,
                    Response {
                        status: Status::NotLocal,
                        body: ResponseBody::Diagnostic("tablet not live on owner".to_owned()),
                    },
                )
                .await;
        }
        let outcome = {
            let mut tablets = self.tablets.borrow_mut();
            let live = tablets.get_mut(&tablet).expect("presence checked above");
            let durability = self
                .durability
                .as_ref()
                .expect("durable path always has a lane");
            let mut durable = durability.borrow_mut();
            execute_durable(
                live,
                &mut durable,
                operation,
                opcode.as_u8(),
                identity.as_ref(),
                now,
            )
        };
        match outcome {
            Ok(result) => {
                bump_direct(&self.metrics);
                self.respond_ok(request_id, opcode, &result).await
            }
            Err(DurableFailure::Op(error)) => {
                self.respond_op_error(request_id, opcode, &error).await
            }
            Err(DurableFailure::Expired) => {
                self.respond(
                    request_id,
                    opcode,
                    Response {
                        status: Status::DedupExpired,
                        body: ResponseBody::Diagnostic(
                            "mutation identity expired below the session floor".to_owned(),
                        ),
                    },
                )
                .await
            }
            Err(DurableFailure::Overloaded) => {
                self.respond(
                    request_id,
                    opcode,
                    Response {
                        status: Status::SessionOverloaded,
                        body: ResponseBody::Diagnostic(
                            "session outcome window exhausted".to_owned(),
                        ),
                    },
                )
                .await
            }
            Err(DurableFailure::Storage(error)) => {
                self.respond_storage_error(request_id, opcode, &error).await
            }
        }
    }

    /// Answers a WAL append failure: full disks read as resource
    /// exhaustion (reads continue), anything else as internal. The lane
    /// health already flipped inside the provider.
    async fn respond_storage_error(
        &mut self,
        request_id: u64,
        opcode: kivi_protocol::Opcode,
        error: &kivi_durability::DurabilityError,
    ) -> Result<(), ConnExit> {
        use kivi_protocol::{Response, ResponseBody, Status};
        let status = match error {
            kivi_durability::DurabilityError::NoSpace { .. } => Status::ResourceExhausted,
            _ => Status::Internal,
        };
        self.respond(
            request_id,
            opcode,
            Response {
                status,
                body: ResponseBody::Diagnostic("durable write failed".to_owned()),
            },
        )
        .await
    }

    /// Returns a refresh redirect when the client's hint disagrees with the
    /// current authority, even on the right worker.
    fn stale_hint_redirect(
        &self,
        tablet: TabletId,
        hint: Option<kivi_protocol::RouteHint>,
    ) -> Option<RedirectInfo> {
        let hint = hint?;
        let current = self.authority_of(tablet);
        if hint.tablet != tablet
            || Some(hint.epoch) != current.map(kivi_types::TabletAuthority::epoch)
        {
            self.redirect_to(tablet)
        } else {
            None
        }
    }

    /// Executes one operation against a local tablet. The tablet borrow ends
    /// before the caller responds (the socket borrows `self` mutably), so a
    /// routed-but-not-live tablet answers `NotLocal` instead of disconnecting.
    fn execute_local(
        &self,
        tablet: TabletId,
        operation: &Operation,
    ) -> Option<Result<kivi_state::OperationResult, crate::tablet::TabletError>> {
        let now = SystemClock::wall_now();
        let mut tablets = self.tablets.borrow_mut();
        tablets
            .get_mut(&tablet)
            .map(|live| live.execute(operation, now))
    }

    /// Evaluates the current routing snapshot for one hash.
    fn route(&self, hash: kivi_types::PartitionHash) -> Route {
        let routing = self.routing.load();
        let Some(tablet) = routing.directory().lookup_by_hash(hash) else {
            return Route::Nowhere;
        };
        let Some(owner) = routing.placement().worker_of(tablet) else {
            return Route::Nowhere;
        };
        if owner != self.worker {
            if let Some(info) = self.redirect_to_inner(&routing, tablet, owner) {
                return Route::Redirect(info);
            }
            return Route::Nowhere;
        }
        Route::Execute(tablet)
    }

    /// Builds redirect info for a tablet known to this snapshot.
    fn redirect_to(&self, tablet: TabletId) -> Option<RedirectInfo> {
        let routing = self.routing.load();
        let owner = routing.placement().worker_of(tablet)?;
        self.redirect_to_inner(&routing, tablet, owner)
    }

    /// Builds redirect info against one loaded snapshot.
    fn redirect_to_inner(
        &self,
        routing: &RoutingSnapshot,
        tablet: TabletId,
        owner: WorkerId,
    ) -> Option<RedirectInfo> {
        let descriptor = routing.directory().get(tablet)?;
        let endpoint = self
            .endpoints
            .load()
            .get(&owner)
            .cloned()
            .unwrap_or_default();
        RedirectInfo::new(
            routing.version(),
            tablet,
            self.authority_of(tablet).map_or(
                kivi_types::TabletEpoch::INITIAL,
                kivi_types::TabletAuthority::epoch,
            ),
            owner,
            endpoint,
            descriptor.range().clone(),
        )
        .ok()
    }

    /// Current authority triple of a live tablet, if hosted here.
    fn authority_of(&self, tablet: TabletId) -> Option<kivi_types::TabletAuthority> {
        self.tablets
            .borrow()
            .get(&tablet)
            .map(crate::tablet::LiveTablet::authority)
    }

    /// Node identity for handshakes (single-node stage: configured).
    fn node_id(&self) -> NodeId {
        self.node_id
    }

    /// Encodes and writes one frame.
    async fn write_frame(
        &mut self,
        kind: kivi_protocol::FrameKind,
        request_id: u64,
        payload: &[u8],
    ) -> Result<(), ConnExit> {
        let bytes = encode_frame(kind, request_id, payload);
        let outcome = self.stream.write_all(bytes).await;
        outcome.0.map_err(|_| ConnExit::Io)?;
        Ok(())
    }

    /// Answers with an operation outcome mapped onto the status taxonomy.
    async fn respond_ok(
        &mut self,
        request_id: u64,
        opcode: kivi_protocol::Opcode,
        outcome: &kivi_state::OperationResult,
    ) -> Result<(), ConnExit> {
        use kivi_protocol::{Response, ResponseBody, Status};
        use kivi_state::OperationResult as R;
        let response = match outcome {
            R::Value(Some(value)) => Response {
                status: Status::Ok,
                body: ResponseBody::Value(value.to_vec()),
            },
            R::Value(None) | R::Counter(None) | R::Expiry(None) => Response {
                status: Status::NotFound,
                body: ResponseBody::Diagnostic(String::new()),
            },
            R::Stored { version } => Response {
                status: Status::Ok,
                body: ResponseBody::Stored {
                    version: version.as_u64(),
                },
            },
            R::Deleted { existed } => Response {
                status: Status::Ok,
                body: ResponseBody::Deleted { existed: *existed },
            },
            R::Exists(present) => Response {
                status: Status::Ok,
                body: ResponseBody::Exists(*present),
            },
            R::Counter(Some(value)) => Response {
                status: Status::Ok,
                body: ResponseBody::Counter(*value),
            },
            R::CounterUpdated { value, version } => Response {
                status: Status::Ok,
                body: ResponseBody::CounterUpdated {
                    value: *value,
                    version: version.as_u64(),
                },
            },
            R::ExpirySet { applied } => Response {
                status: Status::Ok,
                body: ResponseBody::ExpirySet { applied: *applied },
            },
            R::ExpiryPersisted { removed } => Response {
                status: Status::Ok,
                body: ResponseBody::ExpiryPersisted { removed: *removed },
            },
            R::Expiry(Some(expiry)) => Response {
                status: Status::Ok,
                body: match expiry.as_stamp() {
                    None => ResponseBody::ExpiryNever,
                    Some(stamp) => ResponseBody::ExpiryAt(stamp.as_micros()),
                },
            },
        };
        self.respond(request_id, opcode, response).await
    }

    /// Answers a tablet logic failure with its stable status.
    async fn respond_op_error(
        &mut self,
        request_id: u64,
        opcode: kivi_protocol::Opcode,
        error: &kivi_engine_tablet_error_TabletError,
    ) -> Result<(), ConnExit> {
        use kivi_engine_tablet_error_TabletError as E;
        use kivi_protocol::{Response, ResponseBody, Status};
        let (status, message) = match error {
            E::Op(kivi_state::OpError::WrongType { .. }) => (Status::WrongType, "wrong type"),
            E::Op(kivi_state::OpError::CounterOverflow) => (Status::CounterOverflow, "overflow"),
            E::Apply(kivi_state::ApplyError::VersionExhausted) => {
                (Status::VersionExhausted, "version exhausted")
            }
            E::Apply(kivi_state::ApplyError::CounterOverflow) => {
                (Status::CounterOverflow, "overflow")
            }
            E::Apply(kivi_state::ApplyError::TypeMismatch { .. }) => {
                (Status::WrongType, "wrong type")
            }
            E::AuthorityMismatch { .. } => (Status::Internal, "authority mismatch"),
            E::CommitExhausted { .. } => (Status::VersionExhausted, "commit space exhausted"),
        };
        self.respond(
            request_id,
            opcode,
            Response {
                status,
                body: ResponseBody::Diagnostic(message.to_owned()),
            },
        )
        .await
    }

    /// Writes one typed response.
    async fn respond(
        &mut self,
        request_id: u64,
        opcode: kivi_protocol::Opcode,
        response: Response,
    ) -> Result<(), ConnExit> {
        self.write_frame(
            kivi_protocol::FrameKind::Response,
            request_id,
            &response.encode(opcode),
        )
        .await
    }
}

/// Routing verdict for one hash under the current snapshot.
enum Route {
    /// Execute on the named local tablet.
    Execute(TabletId),
    /// Answer with this redirect (owner elsewhere, or stale hint).
    Redirect(RedirectInfo),
    /// No authority covers the hash.
    Nowhere,
}

/// Maps an operation back to its opcode for response encoding.
/// Bumps the direct-path execution counter.
fn bump_direct(metrics: &Rc<Cell<WorkerMetrics>>) {
    let mut snapshot = metrics.get();
    snapshot.direct_ops += 1;
    metrics.set(snapshot);
}

// Re-exported tablet error under an unambiguous local path for the deeply
// nested response mapping above.
use crate::tablet::TabletError as kivi_engine_tablet_error_TabletError;

async fn serve_conn(stream: TcpStream, peer: SocketAddr, shared: Rc<WorkerNet>) {
    let _guard = ActiveGuard::hold(&shared.active);
    let _ = stream.set_nodelay(true);
    let max_input = shared.net.conn.max_input_bytes;
    let max_pipelined = shared.net.conn.max_pipelined;
    let mut conn = Conn {
        stream,
        worker: shared.id,
        tablets: Rc::clone(&shared.tablets),
        metrics: Rc::clone(&shared.metrics),
        routing: Arc::clone(&shared.routing),
        endpoints: Arc::clone(&shared.endpoints),
        shutdown: Rc::clone(&shared.shutdown),
        node_id: shared.net.node_id,
        cluster_id: shared.net.cluster_id,
        incarnation: shared.net.incarnation,
        advertised_caps: shared.net.advertised_caps,
        durability: shared.durability.clone(),
        reader: FrameReader::new(shared.net.conn.max_frame),
        pending: std::collections::VecDeque::new(),
        max_input,
        max_pipelined,
        turn: shared.net.turn,
        shook_hands: false,
        frames_this_turn: 0,
        bytes_this_turn: 0,
    };
    let exit = conn.run().await;
    match exit {
        ConnExit::Clean => tracing::debug!(%peer, "connection closed"),
        ConnExit::Violation(reason) => {
            tracing::warn!(%peer, reason, "protocol violation; connection closed");
        }
        ConnExit::Io => tracing::debug!(%peer, "connection I/O ended"),
    }
}
