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
use std::collections::{HashMap, VecDeque};
use std::net::{IpAddr, SocketAddr};
use std::rc::Rc;
use std::sync::Arc;

use arc_swap::ArcSwap;
use bytes::Bytes;
use compio::io::{AsyncRead, AsyncWriteExt};
use compio::net::{TcpListener, TcpStream};
use compio::runtime::Runtime;

use kivi_protocol::{
    Capabilities, ClientHello, DEFAULT_MAX_FRAME, FRAME_HEADER_LEN, FrameReader, RedirectInfo,
    Request, Response, ServerHello, StreamAbort, StreamBegin, StreamReady, ValueStreamBegin,
    encode_frame,
};
use kivi_state::{ExpiryPolicy, Key, Operation, OperationResult, SetCondition};
use kivi_types::{
    ClusterId, CommitToken, MutationIdentity, NodeId, NodeIncarnation, ReadContract,
    ReplicaFreshness, TabletId, WorkerId,
};

use crate::affinity::{AffinityError, AffinityMode, pin_current_thread};
use crate::commit::PendingEntry;
use crate::routing::RoutingSnapshot;
use crate::tablet::LiveTablet;
use crate::worker::{
    WorkerChunks, WorkerControl, WorkerDurability, WorkerMetrics, WorkerRequestError,
    WorkerResponse, handle_control, handle_request,
};
use kivi_core::SystemClock;

/// Per-connection unparsed-junk bound default (4 MiB). This caps bytes
/// with no validated meaning yet — never a declared in-progress frame,
/// which [`kivi_protocol::FrameReader::buffer_ceiling`] covers exactly up
/// to `max_frame`. A legal frame therefore always fits.
pub const DEFAULT_MAX_INPUT_BYTES: usize = 4 * 1024 * 1024;
/// Decoded-but-not-executed backlog bound default. Pipelining deeper than
/// this closes the connection instead of queueing without limit.
pub const DEFAULT_MAX_PIPELINED: usize = 1024;
/// Per-turn frame budget default: pace deep pipelines without per-op yields.
pub const DEFAULT_TURN_FRAMES: usize = 32;
/// Per-turn byte budget default (256 KiB of request payload).
pub const DEFAULT_TURN_BYTES: usize = 256 * 1024;
/// Embedded-bridge idle backstop: the bridge parks event-driven on its
/// wakeup channel (every ingress send pings it), so idle workers sleep
/// until traffic arrives with no poll quantum and no latency floor.
/// `LocalClient` traffic against a networked worker carries admin calls,
/// tests, and the RESP compatibility edge (which deliberately reuses local
/// ingress instead of proxying through the native wire protocol) — never
/// the native fast path, which lives on the connection tasks and never
/// waits on this timer. This backstop only bounds a lost-ping bug (every
/// send site pings through its ingress bundle, so it never fires in
/// practice); normal wakes are immediate.
///
/// While the commit coordinator holds admitted work, the bridge parks on
/// the shorter completion quantum instead so embedded replies track batch
/// proofs rather than this interval. The quantum only bounds completion
/// detection, never batching.
const BRIDGE_IDLE_BACKSTOP: Duration = Duration::from_millis(20);
/// Accept-loop shutdown poll interval.
const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(100);
/// Startup handshake timeout (bind + runtime creation must report fast).
pub(crate) const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
/// Maximum stream entries encoded in one `StreamEntries` response page.
/// The store page may hold more; shaping truncates explicitly and the
/// client resumes by offset, so one giant shard can never blow the frame
/// budget. Truncation keeps shard order (a prefix of the page).
const MAX_STREAM_PAGE_ENTRIES: usize = 256;

/// Connection-level bounds, all explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConnLimits {
    /// Maximum unparsed input bytes with no validated frame meaning
    /// buffered per connection. A declared in-progress frame (header
    /// parsed, length within `max_frame`) may buffer exactly its
    /// declared length — see `FrameReader::buffer_ceiling`.
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
    /// exactly when the engine runs durable, plus `STREAMING` in both
    /// modes: every networked worker serves chunk-fabric uploads and
    /// streamed reads).
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
    /// Chunk lane access plus the representation policy. Shared across
    /// connection tasks and bridge like durability above. Staging and
    /// resolution suspend the awaiting task only — never the reactor.
    chunks: crate::worker::WorkerChunks,
    /// Memory Fabric integration. Shared across connection tasks and
    /// bridge like durability above. Staging is synchronous (arena
    /// insert, never blocking); promotion suspends the awaiting task
    /// (connection tasks) or parks on the frontier (bridge) — the
    /// reactor thread itself never blocks on storage.
    fabric: Rc<RefCell<crate::fabric::TabletFabric>>,
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
// Eight parameters: identity, tablets, channels, durability, chunks,
// launch, readiness. Grouping further would hide the startup order the
// surrounding code documents; the engine builds this call once.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_net(
    id: WorkerId,
    tablets: Vec<LiveTablet>,
    requests: crossbeam_channel::Receiver<crate::worker::TabletRequest>,
    control: crossbeam_channel::Receiver<WorkerControl>,
    durability: Option<WorkerDurability>,
    chunks: crate::worker::WorkerChunks,
    fabric: crate::fabric::TabletFabric,
    launch: NetLaunch,
    bridge_wake: async_channel::Receiver<()>,
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
        chunks,
        fabric: Rc::new(RefCell::new(fabric)),
    });
    let runtime = match Runtime::new() {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = ready.send(Err(NetStartError::Runtime(error.to_string())));
            return;
        }
    };
    runtime.block_on(serve(shared, requests, control, bridge_wake, ready));
}

/// Top-level serving future: bind, report, accept until shutdown, drain.
async fn serve(
    shared: Rc<WorkerNet>,
    requests: crossbeam_channel::Receiver<crate::worker::TabletRequest>,
    control: crossbeam_channel::Receiver<WorkerControl>,
    bridge_wake: async_channel::Receiver<()>,
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
    // keeps working on networked workers through the same tablets. Joined
    // explicitly at shutdown (never detached): deterministic teardown with
    // no tasks outliving the runtime they were spawned on.
    let bridge = {
        let tablets = Rc::clone(&shared.tablets);
        let metrics = Rc::clone(&shared.metrics);
        let shutdown = Rc::clone(&shared.shutdown);
        let durability = shared.durability.clone();
        let chunks = shared.chunks.clone();
        let fabric = Rc::clone(&shared.fabric);
        compio::runtime::spawn(async move {
            bridge_loop(BridgeContext {
                requests,
                control,
                bridge_wake,
                tablets,
                metrics,
                durability,
                chunks,
                fabric,
                shutdown,
            })
            .await;
        })
    };
    // Accept loop WITHOUT a timeout: cancelling a pending AcceptEx leaks
    // the listener on this compio version (proven by test — a single
    // cancelled accept keeps the FD bound past runtime drop, breaking
    // same-port restart). The loop parks in `accept()` uncancellable and
    // wakes only on real connections; shutdown wakes it with a dummy
    // connection (see `LocalEngine::shutdown`), which it answers by
    // observing the flag at the top of the next turn.
    loop {
        if shared.shutdown.get() {
            break;
        }
        match listener.accept().await {
            Ok((stream, peer)) => {
                tracing::debug!(worker = %shared.id.as_u64(), %peer, "accepted connection");
                spawn_conn(stream, peer, Rc::clone(&shared));
            }
            Err(error) => {
                tracing::warn!(worker = %shared.id.as_u64(), %error, "accept failed");
                compio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }
    // Bounded drain: connections observe shutdown through their own read
    // polls and exit on their own, so the runtime never drops under live
    // sockets. Whatever lingers past the grace period is force-closed by
    // the runtime drop that follows.
    for _ in 0..100 {
        if shared.active.get() == 0 {
            break;
        }
        compio::time::sleep(Duration::from_millis(10)).await;
    }
    // Flush admitted durable work: proofs already on the lane still need
    // ordered apply so committed state (not just the WAL) reflects every
    // acknowledged write at shutdown. Replies go nowhere — connection
    // tasks already exited — but the state advance is what matters, and
    // recovery would replay it anyway.
    if let Some(cell) = shared.durability.as_ref() {
        let namespace = shared.routing.load().namespace();
        for _ in 0..300 {
            let pending = {
                let mut durable = cell.borrow_mut();
                durable.commit.poll(
                    &mut shared.tablets.borrow_mut(),
                    namespace,
                    &mut shared.fabric.borrow_mut(),
                );
                durable.commit.has_pending()
            };
            if !pending {
                break;
            }
            compio::time::sleep(crate::commit::COMPLETION_PARK).await;
        }
    }
    tracing::info!(worker = %shared.id.as_u64(), "worker listener stopped");
    // Join the bridge explicitly: no detached task may outlive this
    // runtime (a lingering task keeps driver state — and potentially
    // sockets — alive past thread join, breaking same-port restart).
    // The bridge exits on the Shutdown control already sent, so this
    // join is prompt, never a hang.
    let _ = bridge.await;
}

/// Embedded-bridge context: everything the bridge task borrows from the
/// worker. Bundled so the bridge entry point takes one argument.
struct BridgeContext {
    requests: crossbeam_channel::Receiver<crate::worker::TabletRequest>,
    control: crossbeam_channel::Receiver<WorkerControl>,
    bridge_wake: async_channel::Receiver<()>,
    tablets: Rc<RefCell<HashMap<TabletId, LiveTablet>>>,
    metrics: Rc<Cell<WorkerMetrics>>,
    durability: Option<Rc<RefCell<WorkerDurability>>>,
    chunks: crate::worker::WorkerChunks,
    fabric: Rc<RefCell<crate::fabric::TabletFabric>>,
    shutdown: Rc<Cell<bool>>,
}

/// Embedded-path bridge: serves the crossbeam request/control channels from
/// inside the runtime thread, sharing its tablets. Ingress drains yield to
/// the reactor like connection I/O; exits on Shutdown or when the engine
/// drops every sender (which also flips the listener shutdown so the
/// thread can join).
///
/// Handle one bridge control message (`false` = shut down). Split out so
/// the bridge loop stays reviewable; borrows end with the call.
fn bridge_control(
    message: WorkerControl,
    tablets: &Rc<RefCell<HashMap<TabletId, LiveTablet>>>,
    metrics: &Rc<Cell<WorkerMetrics>>,
    durability: Option<&Rc<RefCell<WorkerDurability>>>,
    fabric: &Rc<RefCell<crate::fabric::TabletFabric>>,
) -> bool {
    let durable = durability.map(|cell| cell.borrow());
    let durable_ref = durable.as_deref();
    handle_control(
        message,
        &mut tablets.borrow_mut(),
        metrics,
        durable_ref,
        &mut fabric.borrow_mut(),
    )
}

/// Wakeups are event-driven: every ingress send pings `bridge_wake`, so an
/// idle bridge parks in `recv()` with no poll quantum, no latency floor,
/// and no idle CPU. A short backstop bounds a lost ping (which cannot
/// happen through the ingress bundles — this is defense in depth, not a
/// pacing interval). While admitted work is outstanding the bridge parks
/// on the completion quantum so replies track batch proofs.
async fn bridge_loop(context: BridgeContext) {
    use crossbeam_channel::TryRecvError;
    let BridgeContext {
        requests,
        control,
        bridge_wake,
        tablets,
        metrics,
        durability,
        chunks,
        fabric,
        shutdown,
    } = context;
    let mut requests_dead = false;
    let mut control_dead = false;
    // Large embedded `Set`s parked while the chunk lane stages them (the
    // bridge shares the reactor thread, so it polls completions per turn
    // instead of blocking like plain worker threads do).
    let mut frontier: VecDeque<BridgeStage> = VecDeque::new();
    // Fabric reads parked while the offcore lane promotes them. Same
    // poll-per-turn discipline; resume re-checks residency synchronously
    // before serving so the reactor never blocks on storage.
    let mut fabric_parks: VecDeque<FabricReadPark> = VecDeque::new();
    loop {
        loop {
            match control.try_recv() {
                Ok(message) => {
                    if !bridge_control(message, &tablets, &metrics, durability.as_ref(), &fabric) {
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
        // Full drain (the channel capacity bounds the work): intake is
        // admit-only with no I/O, and a yield every slice keeps connection
        // tasks sharing this single-threaded reactor fairly.
        let mut drained = 0usize;
        loop {
            match requests.try_recv() {
                Ok(request) => {
                    bridge_intake(
                        request,
                        &tablets,
                        &metrics,
                        durability.as_ref(),
                        &chunks,
                        &fabric,
                        &mut frontier,
                        &mut fabric_parks,
                    );
                    drained += 1;
                    if drained.is_multiple_of(64) {
                        YieldNow::default().await;
                    }
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
        // Complete staged large Sets whose lane proofs arrived, resume
        // parked fabric reads whose promotions arrived, then pump the
        // commit coordinator so embedded durable requests keep flowing,
        // then park: short quantum while batches are admitted or in
        // flight (reply latency tracks proofs), event-driven sleep
        // otherwise (bounded CPU, immediate wake on traffic).
        poll_bridge_stages(
            &mut frontier,
            &tablets,
            &metrics,
            durability.as_ref(),
            &fabric,
        );
        poll_fabric_parks(&mut fabric_parks, &tablets, &metrics, &fabric);
        let pending = !frontier.is_empty()
            || !fabric_parks.is_empty()
            || durability.as_ref().is_some_and(|cell| {
                // Synchronous borrow: ends before any `.await` below.
                cell.borrow().commit.has_pending()
            });
        {
            if let Some(cell) = durability.as_ref() {
                let mut durable = cell.borrow_mut();
                let namespace = durable.namespace;
                durable.commit.poll(
                    &mut tablets.borrow_mut(),
                    namespace,
                    &mut fabric.borrow_mut(),
                );
            }
            // Amortized fabric maintenance (planning slice, lane
            // submissions, retire sweep): runs on bridge turns, never
            // per request.
            fabric.borrow_mut().maintenance_if_due();
        }
        if pending {
            compio::time::sleep(crate::commit::COMPLETION_PARK).await;
        } else {
            // Event-driven idle: every ingress send pings this channel, so
            // the bridge sleeps until traffic arrives. The backstop only
            // bounds a lost ping (defense in depth — send sites always
            // ping, and a closed channel here just re-checks the queues).
            let _ = compio::time::timeout(BRIDGE_IDLE_BACKSTOP, bridge_wake.recv()).await;
        }
    }
}

/// One embedded large-`Set` parked on the bridge while its lane staging
/// proves durable. Completion admits the resulting small `SetChunked`
/// into the commit pipeline (durable) or executes it inline (ephemeral)
/// with the original identity and responder, so exactly-once and version
/// semantics match the plain-thread path exactly.
struct BridgeStage {
    /// Owning tablet (routed before parking; placement is static).
    tablet: TabletId,
    /// Target key for the chunked root.
    key: Key,
    /// Client retry identity, preserved across the park.
    identity: Option<MutationIdentity>,
    /// Where the outcome goes.
    respond: crossbeam_channel::Sender<WorkerResponse>,
    /// Lane proof being polled.
    reply: crate::chunk_lane::ChunkReply<
        Result<crate::chunk_lane::StagedValue, kivi_chunk::ChunkError>,
    >,
    /// Staging pins, moved into the coordinator entry on completion.
    pinned: Option<crate::chunk_lane::PinnedUpload>,
    /// Pin set for post-stage pinning (splice jobs only: their addresses
    /// are unknown until the lane replies, so pinning happens at
    /// completion instead of intake).
    pins: Option<Arc<crate::chunk_lane::StagingPins>>,
    /// Conditional store context (`None` for plain `Set` stages): the
    /// condition and policy the completion reattaches as
    /// `SetConditionalChunked`.
    conditional: Option<(kivi_state::SetCondition, kivi_state::ExpiryPolicy)>,
}

/// One embedded fabric read parked while the offcore lane promotes it.
/// Resume re-checks residency synchronously before serving (the reactor
/// never blocks on storage); bounded re-parks absorb races, then the
/// request answers backpressure instead of wedging the bridge.
/// Medium-tier routing outcome: staged, already answered, or not a
/// medium write (caller falls through to the chunk path).
#[derive(Debug)]
enum RoutedMedium {
    /// Staged: admitted op plus its seal payloads (caller attaches its pin).
    Staged(Operation, Vec<crate::fabric::StagedSeal>),
    /// Saturation: the request already answered overload.
    Answered,
    /// Not a medium write: caller continues to the chunk path.
    Pass,
}

struct FabricReadPark {
    /// Owning tablet (routed before parking; placement is static).
    tablet: TabletId,
    /// Full read operation to answer on resume (`Get`/`GetRange`).
    op: Operation,
    /// Logical wall time captured at the client boundary (expiry
    /// re-check on resume: never resurrect expired state).
    now: kivi_types::WallTimestamp,
    /// Where the outcome goes.
    respond: crossbeam_channel::Sender<crate::worker::WorkerResponse>,
    /// Fabric park sequence holding the lane reply.
    park: u64,
}

/// Embedded intake on the reactor bridge: inline ops serve synchronously
/// (no lane traffic, no block); large `Set`s park on the frontier and
/// complete on later turns; medium `Set`s stage into the fabric
/// synchronously (arena insert, never blocking). Fabric reads that miss
/// residency park on the fabric frontier; everything else serves inline.
/// Lane saturation answers `Overloaded` at once.
#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
fn bridge_intake(
    request: crate::worker::TabletRequest,
    tablets: &Rc<RefCell<HashMap<TabletId, LiveTablet>>>,
    metrics: &Rc<Cell<WorkerMetrics>>,
    durability: Option<&Rc<RefCell<WorkerDurability>>>,
    chunks: &WorkerChunks,
    fabric: &Rc<RefCell<crate::fabric::TabletFabric>>,
    frontier: &mut VecDeque<BridgeStage>,
    fabric_parks: &mut VecDeque<FabricReadPark>,
) {
    let crate::worker::TabletRequest {
        tablet,
        op,
        now,
        identity,
        respond,
    } = request;
    // Fabric reads pre-check residency synchronously: a hit flows into
    // the normal path below (whose internal resolve will also hit —
    // check and act share one synchronous turn, so no race); a miss
    // parks here instead of entering `handle_request`, whose blocking
    // resolve must never run on the reactor.
    if let Operation::Get { key } | Operation::GetRange { key, .. } = &op
        && let Some(reference) = tablets
            .borrow()
            .get(&tablet)
            .and_then(|live| live.store().get(key, now))
            .and_then(kivi_state::StoredObject::fabric_ref)
    {
        let resident = fabric
            .borrow_mut()
            .resolve_sync(
                &reference,
                tablets
                    .borrow()
                    .get(&tablet)
                    .and_then(|live| live.store().get(key, now)),
            )
            .ok()
            .and_then(|outcome| outcome);
        if resident.is_none() {
            park_fabric_read(
                tablet,
                op,
                now,
                respond,
                reference,
                tablets,
                metrics,
                fabric,
                fabric_parks,
            );
            return;
        }
    }
    // Range patches plan against the peeked base before the
    // representation split: small inline results serve synchronously
    // below, restaged results re-enter as ordinary large `Set`s, chunked
    // bases park a lane splice on the frontier, fabric bases resolve or
    // park, and absurd ranges reject with state untouched. A parked or
    // rejected patch returns here (its responder is consumed by the
    // frontier or the answer).
    let (planned, respond) = bridge_plan_range(
        tablet, op, now, identity, respond, tablets, metrics, chunks, fabric, frontier,
    );
    let (Some(op), Some(respond)) = (planned, respond) else {
        return;
    };
    // Medium values stage into the fabric synchronously (arena insert,
    // never blocking on the reactor); large values park chunk staging
    // on the frontier below.
    if let Operation::Set { value, .. } | Operation::SetConditional { value, .. } = &op
        && value.len() > crate::fabric::FABRIC_INLINE_MAX
        && (value.len() as u64) <= chunks.inline_threshold
    {
        bridge_stage_fabric(
            tablet, &op, now, identity, respond, tablets, metrics, durability, fabric,
        );
        return;
    }
    match crate::chunk_lane::split_large_set(op, chunks.inline_threshold) {
        crate::chunk_lane::LargeSetSplit::Inline(op) => {
            let mut durable = durability.map(|cell| cell.borrow_mut());
            let durable_ref = durable.as_deref_mut();
            handle_request(
                crate::worker::TabletRequest {
                    tablet,
                    op,
                    now,
                    identity,
                    respond,
                },
                &mut tablets.borrow_mut(),
                metrics,
                durable_ref,
                chunks,
                &mut fabric.borrow_mut(),
            );
        }
        split => bridge_chunk_stage(tablet, split, identity, respond, metrics, chunks, frontier),
    }
}

/// Parks one chunk staging on the bridge frontier: begins the pinned
/// upload, submits the lane stage, and answers lane failures inline.
/// Kept beside [`bridge_intake`] so the intake match stays reviewable.
fn bridge_chunk_stage(
    tablet: TabletId,
    split: crate::chunk_lane::LargeSetSplit,
    identity: Option<MutationIdentity>,
    respond: crossbeam_channel::Sender<crate::worker::WorkerResponse>,
    metrics: &Rc<Cell<WorkerMetrics>>,
    chunks: &WorkerChunks,
    frontier: &mut VecDeque<BridgeStage>,
) {
    #[allow(clippy::too_many_arguments)]
    fn push_stage(
        tablet: TabletId,
        key: Key,
        value: Bytes,
        conditional: Option<(kivi_state::SetCondition, kivi_state::ExpiryPolicy)>,
        identity: Option<MutationIdentity>,
        respond: crossbeam_channel::Sender<crate::worker::WorkerResponse>,
        metrics: &Rc<Cell<WorkerMetrics>>,
        chunks: &WorkerChunks,
        frontier: &mut VecDeque<BridgeStage>,
    ) {
        let pinned =
            match crate::chunk_lane::PinnedUpload::begin(&chunks.pins, chunks.domain, &value) {
                Ok(pinned) => Some(pinned),
                Err(error) => {
                    let mut snapshot = metrics.get();
                    snapshot.channel_ops += 1;
                    metrics.set(snapshot);
                    let _ = respond.try_send(Err(WorkerRequestError::ChunkStore(error)));
                    return;
                }
            };
        match chunks.lane.submit_stage(value) {
            Ok(reply) => frontier.push_back(BridgeStage {
                tablet,
                key,
                identity,
                respond,
                reply,
                pinned,
                pins: None,
                conditional,
            }),
            Err(error) => {
                let mut snapshot = metrics.get();
                snapshot.channel_ops += 1;
                metrics.set(snapshot);
                let _ = respond.try_send(Err(WorkerRequestError::ChunkStore(error)));
            }
        }
    }
    match split {
        crate::chunk_lane::LargeSetSplit::Inline(_) => {
            unreachable!("inline sets serve above")
        }
        crate::chunk_lane::LargeSetSplit::Stage { key, value } => {
            push_stage(
                tablet, key, value, None, identity, respond, metrics, chunks, frontier,
            );
        }
        crate::chunk_lane::LargeSetSplit::StageConditional {
            key,
            value,
            condition,
            expiry,
        } => {
            push_stage(
                tablet,
                key,
                value,
                Some((condition, expiry)),
                identity,
                respond,
                metrics,
                chunks,
                frontier,
            );
        }
    }
}

/// Parks one fabric read on the bridge frontier: submits the lane
/// promotion (non-blocking) and parks the full read for a later turn.
/// The resume path re-checks residency synchronously before serving.
#[allow(clippy::too_many_arguments)]
fn park_fabric_read(
    tablet: TabletId,
    op: Operation,
    now: kivi_types::WallTimestamp,
    respond: crossbeam_channel::Sender<crate::worker::WorkerResponse>,
    reference: kivi_state::FabricRef,
    tablets: &Rc<RefCell<HashMap<TabletId, LiveTablet>>>,
    metrics: &Rc<Cell<WorkerMetrics>>,
    fabric: &Rc<RefCell<crate::fabric::TabletFabric>>,
    fabric_parks: &mut VecDeque<FabricReadPark>,
) {
    let current = tablets
        .borrow()
        .get(&tablet)
        .and_then(|live| live.store().get(op.key(), now).cloned());
    let park = fabric.borrow_mut().submit_parked_promote(
        tablet,
        op.key().clone(),
        &reference,
        current.as_ref(),
    );
    let mut snapshot = metrics.get();
    snapshot.channel_ops += 1;
    metrics.set(snapshot);
    match park {
        Ok(park) => {
            fabric_parks.push_back(FabricReadPark {
                tablet,
                op,
                now,
                respond,
                park,
            });
        }
        Err(error) => {
            let _ = respond.try_send(Err(match error {
                crate::fabric::FabricError::Overloaded => {
                    crate::worker::WorkerRequestError::SessionOverloaded
                }
                other => crate::worker::WorkerRequestError::Fabric(other),
            }));
        }
    }
}

/// Stages one medium `Set`/`SetConditional` into the fabric on the
/// bridge (synchronous arena insert), then admits (durable) or executes
/// (ephemeral) the resulting small root. Staging failures answer at
/// once with state untouched.
#[allow(clippy::too_many_arguments)]
fn bridge_stage_fabric(
    tablet: TabletId,
    op: &Operation,
    now: kivi_types::WallTimestamp,
    identity: Option<MutationIdentity>,
    respond: crossbeam_channel::Sender<crate::worker::WorkerResponse>,
    tablets: &Rc<RefCell<HashMap<TabletId, LiveTablet>>>,
    metrics: &Rc<Cell<WorkerMetrics>>,
    durability: Option<&Rc<RefCell<WorkerDurability>>>,
    fabric: &Rc<RefCell<crate::fabric::TabletFabric>>,
) {
    let (key, value, conditional) = match op {
        Operation::Set { key, value } => (key.clone(), value.clone(), None),
        Operation::SetConditional {
            key,
            value,
            condition,
            expiry,
        } => (key.clone(), value.clone(), Some((*condition, *expiry))),
        _ => unreachable!("bridge stages medium sets only"),
    };
    let staged = fabric.borrow_mut().stage(tablet, value.clone(), true);
    let mut snapshot = metrics.get();
    snapshot.channel_ops += 1;
    metrics.set(snapshot);
    let (fabric_id, logical_len) = match staged {
        Ok(staged) => staged,
        Err(error) => {
            let _ = respond.try_send(Err(match error {
                crate::fabric::FabricError::Overloaded => {
                    crate::worker::WorkerRequestError::SessionOverloaded
                }
                other => crate::worker::WorkerRequestError::Fabric(other),
            }));
            return;
        }
    };
    let staged_op = match conditional {
        None => Operation::SetFabric {
            key: key.clone(),
            fabric_id,
            logical_len,
            version: 0,
        },
        Some((condition, expiry)) => Operation::SetConditionalFabric {
            key: key.clone(),
            fabric_id,
            logical_len,
            version: 0,
            condition,
            expiry,
        },
    };
    let seal = crate::fabric::StagedSeal {
        fabric_id,
        key: key.clone(),
        bytes: value,
    };
    match durability {
        None => {
            let mut tablets = tablets.borrow_mut();
            let outcome = match tablets.get_mut(&tablet) {
                None => Err(crate::worker::WorkerRequestError::UnknownTablet { tablet }),
                Some(live) => {
                    let previous = live
                        .store()
                        .get(&key, now)
                        .and_then(kivi_state::StoredObject::fabric_ref);
                    match live.execute(&staged_op, now) {
                        Err(error) => {
                            fabric.borrow_mut().retire_or_defer(fabric_id);
                            Err(crate::worker::WorkerRequestError::Tablet(error))
                        }
                        Ok(result) => {
                            if let Some(post) = live.store().get(&key, now) {
                                fabric.borrow_mut().retire_superseded(previous, post);
                            }
                            Ok(result)
                        }
                    }
                }
            };
            let _ = respond.try_send(outcome);
        }
        Some(cell) => {
            let mut durable = cell.borrow_mut();
            let namespace = durable.namespace;
            durable.commit.admit(
                crate::commit::PendingEntry::new(tablet, staged_op, now, identity, respond)
                    .with_fabric_staged(vec![seal]),
            );
            durable.commit.poll(
                &mut tablets.borrow_mut(),
                namespace,
                &mut fabric.borrow_mut(),
            );
        }
    }
}
// Ten parameters: the routed request envelope plus the bridge state the
// plan consults (tablets, metrics, lane policy, fabric, frontier).
// Grouping would hide the intake order the surrounding code documents;
// the sole caller builds it explicitly.
#[allow(clippy::too_many_arguments)]
fn bridge_plan_range(
    tablet: TabletId,
    op: Operation,
    now: kivi_types::WallTimestamp,
    identity: Option<MutationIdentity>,
    respond: crossbeam_channel::Sender<crate::worker::WorkerResponse>,
    tablets: &Rc<RefCell<HashMap<TabletId, LiveTablet>>>,
    metrics: &Rc<Cell<WorkerMetrics>>,
    chunks: &WorkerChunks,
    fabric: &Rc<RefCell<crate::fabric::TabletFabric>>,
    frontier: &mut VecDeque<BridgeStage>,
) -> (
    Option<Operation>,
    Option<crossbeam_channel::Sender<crate::worker::WorkerResponse>>,
) {
    let Operation::SetRange { key, offset, patch } = op else {
        return (Some(op), Some(respond));
    };
    let base = crate::worker::peek_range_base(&tablets.borrow(), tablet, &key, now);
    // Fabric bases resolve synchronously when resident (arenas only,
    // never blocking the reactor). A miss answers backpressure for this
    // turn: the client retries once residency returns (promotions
    // complete on later turns), which keeps bridge turns bounded
    // instead of chaining a resume-replan.
    let base = match base {
        crate::chunk_lane::RangeBase::Fabric { fabric_id, .. } => {
            if let Ok(bytes) = fabric.borrow_mut().try_resolve_sync(fabric_id) {
                crate::chunk_lane::RangeBase::Inline(bytes)
            } else {
                let mut snapshot = metrics.get();
                snapshot.channel_ops += 1;
                metrics.set(snapshot);
                let _ = respond.try_send(Err(crate::worker::WorkerRequestError::SessionOverloaded));
                return (None, Some(respond));
            }
        }
        other => other,
    };
    match crate::chunk_lane::plan_set_range(key, offset, patch, base, chunks.inline_threshold) {
        crate::chunk_lane::SetRangePlan::Inline(op) => (Some(op), Some(respond)),
        crate::chunk_lane::SetRangePlan::Reject { detail } => {
            let mut snapshot = metrics.get();
            snapshot.channel_ops += 1;
            metrics.set(snapshot);
            let _ = respond.try_send(Err(WorkerRequestError::InvalidRequest { detail }));
            (None, Some(respond))
        }
        crate::chunk_lane::SetRangePlan::RestageSet { key, value } => {
            // Restaged patches keep the conditional/Keep spelling so a
            // large result preserves expiry exactly like the inline splice.
            (
                Some(Operation::SetConditional {
                    key,
                    value,
                    condition: SetCondition::Always,
                    expiry: ExpiryPolicy::Keep,
                }),
                Some(respond),
            )
        }
        crate::chunk_lane::SetRangePlan::Splice {
            key,
            manifest,
            logical_len,
            offset,
            patch,
        } => match chunks
            .lane
            .submit_splice(manifest, logical_len, offset, patch)
        {
            Ok(reply) => {
                frontier.push_back(BridgeStage {
                    tablet,
                    key,
                    identity,
                    respond,
                    reply,
                    pinned: None,
                    pins: Some(Arc::clone(&chunks.pins)),
                    conditional: Some((SetCondition::Always, ExpiryPolicy::Keep)),
                });
                (None, None)
            }
            Err(error) => {
                let mut snapshot = metrics.get();
                snapshot.channel_ops += 1;
                metrics.set(snapshot);
                let _ = respond.try_send(Err(WorkerRequestError::ChunkStore(error)));
                (None, Some(respond))
            }
        },
    }
}

/// Polls parked fabric reads: ready promotions answer (the park reply
/// already re-fenced id+version and installed a shadow); expiry
/// re-checks against the request's `now` so resumed reads never
/// resurrect; stale roots and lane failures answer closed for client
/// retry. Pending parks stay.
fn poll_fabric_parks(
    parks: &mut VecDeque<FabricReadPark>,
    tablets: &Rc<RefCell<HashMap<TabletId, LiveTablet>>>,
    metrics: &Rc<Cell<WorkerMetrics>>,
    fabric: &Rc<RefCell<crate::fabric::TabletFabric>>,
) {
    let mut cursor = 0usize;
    while cursor < parks.len() {
        let (tablet, key, now, park) = {
            let parked = &parks[cursor];
            (
                parked.tablet,
                parked.op.key().clone(),
                parked.now,
                parked.park,
            )
        };
        let current = tablets
            .borrow()
            .get(&tablet)
            .and_then(|live| live.store().get(&key, now).cloned());
        match fabric.borrow_mut().poll_parked(park, current.as_ref()) {
            Ok(None) => {
                cursor += 1;
            }
            Ok(Some(bytes)) => {
                let parked = parks.remove(cursor).expect("cursor valid");
                let mut snapshot = metrics.get();
                snapshot.channel_ops += 1;
                metrics.set(snapshot);
                // Expiry re-check at the request's `now`: the fenced
                // bytes are only servable while the root is still live.
                let live = tablets
                    .borrow()
                    .get(&tablet)
                    .and_then(|live| live.store().get(&key, now).cloned());
                let answer = match (live, &parked.op) {
                    (None, _) => OperationResult::Value(None),
                    (Some(_), Operation::GetRange { offset, len, .. }) => {
                        OperationResult::Value(Some(kivi_state::slice_range(&bytes, *offset, *len)))
                    }
                    _ => OperationResult::Value(Some(bytes)),
                };
                let _ = parked.respond.try_send(Ok(answer));
            }
            Err(error) => {
                let parked = parks.remove(cursor).expect("cursor valid");
                let mut snapshot = metrics.get();
                snapshot.channel_ops += 1;
                metrics.set(snapshot);
                let _ = parked.respond.try_send(Err(match error {
                    crate::fabric::FabricError::Overloaded => {
                        crate::worker::WorkerRequestError::SessionOverloaded
                    }
                    other => crate::worker::WorkerRequestError::Fabric(other),
                }));
            }
        }
    }
}
/// Polls parked chunk stagings: completions admit (durable) or execute
/// (ephemeral) with the commit timestamp taken at completion — the
/// logical commit happens at admission, not at the original arrival.
fn poll_bridge_stages(
    frontier: &mut VecDeque<BridgeStage>,
    tablets: &Rc<RefCell<HashMap<TabletId, LiveTablet>>>,
    metrics: &Rc<Cell<WorkerMetrics>>,
    durability: Option<&Rc<RefCell<WorkerDurability>>>,
    fabric: &Rc<RefCell<crate::fabric::TabletFabric>>,
) {
    use async_channel::TryRecvError;
    let mut cursor = 0;
    while cursor < frontier.len() {
        let outcome = match frontier[cursor].reply.try_recv() {
            Err(TryRecvError::Empty) => {
                cursor += 1;
                continue;
            }
            // Lane gone: the worker is exiting. Dropping the responder
            // surfaces disconnect downstream; nothing was admitted.
            Err(TryRecvError::Closed) => {
                frontier.remove(cursor);
                continue;
            }
            Ok(outcome) => outcome,
        };
        let staged = frontier.remove(cursor).expect("cursor valid");
        let mut snapshot = metrics.get();
        snapshot.channel_ops += 1;
        metrics.set(snapshot);
        let (op, pinned) = match outcome {
            Err(error) => {
                let _ = staged
                    .respond
                    .try_send(Err(WorkerRequestError::ChunkStore(error)));
                continue;
            }
            Ok(proof) => {
                // Splice jobs pin here (addresses were unknowable at
                // intake); ordinary stages already carry their guard.
                let pinned = staged
                    .pins
                    .as_ref()
                    .map(|pins| crate::chunk_lane::PinnedUpload::staged(pins, &proof))
                    .or(staged.pinned);
                let op = match staged.conditional {
                    None => Operation::SetChunked {
                        key: staged.key,
                        manifest: proof.manifest,
                        logical_len: proof.logical_len,
                    },
                    Some((condition, expiry)) => Operation::SetConditionalChunked {
                        key: staged.key,
                        manifest: proof.manifest,
                        logical_len: proof.logical_len,
                        condition,
                        expiry,
                    },
                };
                (op, pinned)
            }
        };
        let now = kivi_core::wall_now_or_max(&SystemClock);
        match durability {
            None => {
                let outcome = match tablets.borrow_mut().get_mut(&staged.tablet) {
                    None => Err(WorkerRequestError::UnknownTablet {
                        tablet: staged.tablet,
                    }),
                    Some(live) => live.execute(&op, now).map_err(WorkerRequestError::Tablet),
                };
                let _ = staged.respond.try_send(outcome);
            }
            Some(cell) => {
                let mut durable = cell.borrow_mut();
                let namespace = durable.namespace;
                durable.commit.admit(
                    PendingEntry::new(staged.tablet, op, now, staged.identity, staged.respond)
                        .with_pinned(pinned),
                );
                durable.commit.poll(
                    &mut tablets.borrow_mut(),
                    namespace,
                    &mut fabric.borrow_mut(),
                );
            }
        }
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
///
/// The socket is split after the handshake: a detached reader task owns
/// the read half (framing is never cancelled, so pipelined bytes can
/// never be lost to a raced timeout), while this task owns the write
/// half plus the reply outbox. Frames cross by a bounded channel; channel
/// timeouts cancel nothing, so wakeups are free of the swallow hazard
/// that socket-read timeouts carry.
struct Conn {
    /// Write half (post-handshake; `None` only while splitting).
    stream: Option<TcpStream>,
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
    /// Chunk lane access plus the representation policy (large-`Set`
    /// conversion and chunked-read resolution, always off-reactor).
    chunks: WorkerChunks,
    /// Memory Fabric integration (medium-`Set` conversion and
    /// fabric-read resolution). Staging is synchronous (arena insert);
    /// promotion suspends the awaiting task only — never the reactor.
    /// Every borrow is synchronous and ends before any `.await`.
    fabric: Rc<RefCell<crate::fabric::TabletFabric>>,
    /// Parser state (post-handshake; moved to the reader task on split).
    reader: Option<FrameReader>,
    /// Inbound frames from the reader task (bounded by `max_pipelined`).
    frames: async_channel::Receiver<kivi_protocol::Frame>,
    max_input: usize,
    max_pipelined: usize,
    turn: TurnBudget,
    shook_hands: bool,
    frames_this_turn: usize,
    bytes_this_turn: usize,
    /// Admitted durable mutations awaiting their batch proof, in
    /// admission order. Intake never waits on these: each loop turn pumps
    /// the coordinator, replies to whatever proved, then takes more.
    /// Responses match by request id, so barrier completions may answer
    /// out of admission order — the client dispatches by id.
    outbox: std::collections::VecDeque<OutboxEntry>,
    /// Open streaming uploads by client-chosen stream id. Connection-local
    /// by construction: ids never leave this connection, and connection
    /// death drops every half-staged upload with it (staged-but-uncommitted
    /// chunks age out through GC; nothing references them).
    uploads: HashMap<u64, UploadState>,
    /// Maximum `StreamData` payload bytes per frame, negotiated at
    /// handshake from `max_frame` and capped at one chunk: uploads and
    /// downloads chunk to this, so no stream frame ever exceeds framing.
    stream_data_cap: u32,
}

/// One open streaming upload: staged chunk addresses in logical order
/// plus the sub-chunk tail still buffering. Chunks stage incrementally
/// (no per-chunk sync); one barrier at commit proves them together.
struct UploadState {
    /// Target key (its namespace was checked at begin; routing after
    /// that keys off the tablet, exactly like ordinary requests).
    key: Key,
    /// Owning tablet (routed at begin; placement is static).
    tablet: TabletId,
    /// Retry identity for the commit, mirroring requests.
    identity: Option<kivi_types::RequestIdentity>,
    /// Acknowledgement floor for the commit, mirroring requests.
    ack_floor: kivi_types::RequestSeq,
    /// Declared total bytes (`None` = unknown upfront).
    declared: Option<u64>,
    /// Payload bytes accepted so far (staged plus buffered).
    received: u64,
    /// Sub-chunk tail buffering toward the next full piece.
    buf: Vec<u8>,
    /// Staged chunk addresses in logical order with their lengths.
    entries: Vec<(kivi_types::ChunkId, u64)>,
    /// Negotiated per-frame cap echoed for error paths.
    max_data: u32,
}

/// One admitted durable mutation: who to answer once its batch proves.
/// No pin guard rides here: the coordinator entry owns the upload's pins
/// until after its apply journals the commit (which then covers the
/// response window), so dropping this entry unpins nothing.
struct OutboxEntry {
    request_id: u64,
    opcode: kivi_protocol::Opcode,
    /// Owning tablet (response shaping for tablet-naming bodies such as
    /// `TxnPrepared`, which drivers use for route learning).
    tablet: TabletId,
    /// Whether this was a `GetStream` read: completions answer with
    /// `ValueStream*` frames instead of a single `Response`.
    streamed: bool,
    /// `GetRange` window (`None` for every other opcode): applied after
    /// a chunked root resolves, so range reads never materialize more
    /// than the window on the wire (the full payload still resolves
    /// first in this stage; a future lane range-read will fetch only the
    /// required chunks).
    range: Option<(u64, u64)>,
    receive: crossbeam_channel::Receiver<crate::worker::WorkerResponse>,
}

/// How long one loop turn waits for the next frame before re-pumping the
/// coordinator (straggler detection for barrier completions while idle).
/// Fast paths answer in the admit turn itself, so this never paces them.
/// Channel timeouts cancel no socket reads (framing lives on the reader
/// task), so this wait is free of the swallow hazard: wakeups cost a
/// timer, never bytes.
const OUTBOX_IDLE_POLL: Duration = Duration::from_millis(50);
/// Frame wait while this connection (or its worker) holds outstanding
/// work: completions and linger expiries resolve promptly instead of
/// waiting out the idle interval.
const OUTBOX_ACTIVE_POLL: Duration = Duration::from_micros(250);

/// Why a connection task ended. `Clean` (EOF) is silent; violations
/// and I/O failures trace at warn/debug respectively.
#[derive(Debug, Clone, Copy)]
enum ConnExit {
    Clean,
    Violation(&'static str),
    Io,
}

impl Conn {
    /// Serves one connection from handshake to close.
    ///
    /// Intake and replies are decoupled: frames arrive pre-decoded from
    /// the reader task (bounded channel), are admitted eagerly (bounded
    /// by `max_pipelined` across channel plus outbox), and every turn
    /// pumps the coordinator and answers whatever proved. A pipelined
    /// client thus keeps many mutations in flight through one
    /// connection, and concurrent connections genuinely overlap in the
    /// commit pipeline instead of serializing behind one wait at a time.
    /// (The handshake runs first, on the unified stream — see
    /// `serve_conn`.)
    async fn run(&mut self) -> ConnExit {
        loop {
            if self.shutdown.get() {
                // Unanswered outbox entries die with the connection: the
                // client's retry carries the same identity, so exactly-once
                // holds across the shutdown.
                return ConnExit::Clean;
            }
            // Pump first so barrier completions (and fast paths from the
            // last admission) progress even when no new frame arrives.
            self.pump_coordinator();
            if let Err(exit) = self.flush_outbox().await {
                return exit;
            }
            // Active connections (outbox or worker work outstanding)
            // poll frames promptly so linger expiries and completions
            // resolve in hundreds of microseconds; idle connections park
            // longer. Either way this timeout cancels only a channel
            // wait — never a socket read — so no frame is ever at risk.
            let active = !self.outbox.is_empty() || self.coordinator_has_pending();
            let wait = if active {
                OUTBOX_ACTIVE_POLL
            } else {
                OUTBOX_IDLE_POLL
            };
            let frame = match compio::time::timeout(wait, self.next_frame()).await {
                Ok(Ok(frame)) => frame,
                Ok(Err(exit)) => return exit,
                // No frame yet: loop back through pump and flush.
                Err(_) => continue,
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

    /// Whether the shared coordinator holds admitted or in-flight work
    /// (frame wait stays short while it does).
    fn coordinator_has_pending(&self) -> bool {
        self.durability.as_ref().is_some_and(|cell| {
            // Synchronous borrow: ends before any `.await` below.
            cell.borrow().commit.has_pending()
        })
    }

    /// Takes the next pre-decoded frame from the reader task. Reader death
    /// (EOF, violation, I/O) ends the connection the same way a local
    /// read failure would. Channel timeouts cancel nothing.
    async fn next_frame(&mut self) -> Result<kivi_protocol::Frame, ConnExit> {
        self.frames.recv().await.map_err(|_| ConnExit::Clean)
    }

    /// Performs the handshake: exactly one `ClientHello` first, then our
    /// `ServerHello`. Any deviation closes the connection. Runs pre-split
    /// on the unified stream (the only frames ever read synchronously
    /// here); any pipelined frames arriving with the hello are returned
    /// for the reader task's channel so nothing is lost in the handoff.
    /// The handshake carries no outer timeout (one racing hello cannot
    /// lose bytes — nothing else is in flight yet).
    async fn handshake(&mut self) -> Result<Vec<kivi_protocol::Frame>, ConnExit> {
        let (frame, extras) = self.read_handshake_frames().await?;
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
        let server_max = u32::try_from(self.reader.as_ref().expect("parser pre-split").max_frame())
            .unwrap_or(u32::MAX);
        let max_frame = hello.max_frame.min(server_max);
        self.reader = Some(FrameReader::new(max_frame as usize));
        // Stream frames share the framing bound: payloads stay under the
        // header, and one chunk per frame at most keeps upload staging
        // incremental on both sides.
        self.stream_data_cap = max_frame
            .saturating_sub(u32::try_from(FRAME_HEADER_LEN).unwrap_or(u32::MAX))
            .max(1)
            .min(u32::try_from(kivi_chunk::DEFAULT_CHUNK_SIZE).unwrap_or(u32::MAX));
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
        Ok(extras)
    }

    /// Reads handshake frames straight from the socket (pre-split only):
    /// loops until at least one full frame decodes, returning any further
    /// frames that arrived in the same bytes alongside. EOF, violations,
    /// and I/O errors end the connection before it starts.
    async fn read_handshake_frames(
        &mut self,
    ) -> Result<(kivi_protocol::Frame, Vec<kivi_protocol::Frame>), ConnExit> {
        let stream = self.stream.as_mut().expect("unified stream pre-split");
        let reader = self.reader.as_mut().expect("parser pre-split");
        loop {
            // Coherent bound: the input cap applies to unparsed junk, not
            // to a declared in-progress frame (which the ceiling covers
            // exactly). A legal frame up to max_frame always fits.
            if reader.buffered() >= reader.buffer_ceiling(self.max_input) {
                return Err(ConnExit::Violation("input buffer bound"));
            }
            let chunk = Vec::with_capacity(8192);
            let outcome = compio::time::timeout(ACCEPT_POLL_INTERVAL, stream.read(chunk)).await;
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
                    let mut frames: std::collections::VecDeque<kivi_protocol::Frame> = reader
                        .push(&chunk)
                        .map_err(|_| ConnExit::Violation("frame parse"))?
                        .into_iter()
                        .collect();
                    if frames.len() > self.max_pipelined {
                        return Err(ConnExit::Violation("pipelined backlog bound"));
                    }
                    if let Some(first) = frames.pop_front() {
                        return Ok((first, frames.into_iter().collect()));
                    }
                }
                Err(_) => return Err(ConnExit::Io),
            }
        }
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
            FrameKind::StreamBegin => {
                self.handle_stream_begin(frame.request_id, &frame.payload)
                    .await
            }
            FrameKind::StreamData => {
                self.handle_stream_data(frame.request_id, &frame.payload)
                    .await
            }
            FrameKind::StreamCommit => {
                self.handle_stream_commit(frame.request_id, &frame.payload)
                    .await
            }
            FrameKind::StreamAbort => {
                // Best-effort by design: the sender already dropped its
                // side, so a garbled reason still ends in a dropped
                // stream. Never a violation, never a reply.
                if let Ok(abort) = StreamAbort::decode(&frame.payload) {
                    tracing::debug!(stream = frame.request_id, reason = %abort.reason, "peer aborted upload");
                }
                self.uploads.remove(&frame.request_id);
                Ok(())
            }
            _ => Err(ConnExit::Violation("unexpected frame kind")),
        }
    }

    /// Aborts one upload in-band: drops all stream state and tells the
    /// client why. Commits never follow an abort on the same id.
    async fn abort_upload(&mut self, stream: u64, reason: &str) -> Result<(), ConnExit> {
        self.uploads.remove(&stream);
        tracing::debug!(stream, reason, "upload aborted");
        self.write_frame(
            kivi_protocol::FrameKind::StreamAbort,
            stream,
            &StreamAbort {
                reason: reason.to_owned(),
            }
            .encode(),
        )
        .await
    }

    /// Opens one streaming upload: validates, routes, and parks the
    /// assembler, answering `StreamReady` with the per-frame cap.
    async fn handle_stream_begin(&mut self, stream: u64, payload: &[u8]) -> Result<(), ConnExit> {
        let begin = StreamBegin::decode(payload)
            .map_err(|_| ConnExit::Violation("malformed stream-begin"))?;
        if begin.namespace != self.routing.load().namespace() {
            return self.abort_upload(stream, "unknown namespace").await;
        }
        if begin
            .total_len
            .is_some_and(|total| total > kivi_protocol::MAX_STREAM_UPLOAD_BYTES)
        {
            return self
                .abort_upload(stream, "upload exceeds the 1 GiB stream bound")
                .await;
        }
        if self.uploads.contains_key(&stream) {
            return self.abort_upload(stream, "stream id already in use").await;
        }
        let routing = self.routing.load();
        let Some(tablet) =
            crate::compound::route_point_key(routing.directory(), begin.namespace, &begin.key)
        else {
            return self.abort_upload(stream, "no tablet covers key").await;
        };
        let tablet = match self.route_tablet(tablet) {
            Route::Execute(tablet) => tablet,
            Route::Redirect(info) => {
                let endpoint = self
                    .endpoints
                    .load()
                    .get(&info.worker)
                    .cloned()
                    .unwrap_or_default();
                return self
                    .abort_upload(
                        stream,
                        &format!("key lives on worker {} ({endpoint})", info.worker.as_u64()),
                    )
                    .await;
            }
            Route::Nowhere => {
                return self.abort_upload(stream, "no tablet covers key").await;
            }
        };
        self.uploads.insert(
            stream,
            UploadState {
                key: Key::from(begin.key),
                tablet,
                identity: begin.identity,
                ack_floor: begin.ack_floor,
                declared: begin.total_len,
                received: 0,
                buf: Vec::new(),
                entries: Vec::new(),
                max_data: self.stream_data_cap,
            },
        );
        self.write_frame(
            kivi_protocol::FrameKind::StreamReady,
            stream,
            &StreamReady {
                max_data: self.stream_data_cap,
            }
            .encode(),
        )
        .await
    }

    /// Appends one upload frame: stages full chunks incrementally so
    /// server residency stays near one chunk however large the upload.
    async fn handle_stream_data(&mut self, stream: u64, payload: &[u8]) -> Result<(), ConnExit> {
        // Unknown ids echo an abort instead of violating: data already in
        // flight when the server aborts (over-cap, over-declared) is a
        // benign race, not a peer bug.
        if !self.uploads.contains_key(&stream) {
            return self.abort_upload(stream, "unknown stream").await;
        }
        let max_data = self.uploads.get(&stream).expect("checked").max_data;
        if payload.len() as u64 > u64::from(max_data) {
            return self
                .abort_upload(stream, "stream frame exceeds negotiated cap")
                .await;
        }
        // `received` already covers the buffered tail: only the new
        // payload extends the accepted total.
        let accepted = self.uploads.get(&stream).expect("checked").received;
        let total = accepted + payload.len() as u64;
        if total > kivi_protocol::MAX_STREAM_UPLOAD_BYTES {
            return self
                .abort_upload(stream, "upload exceeds the 1 GiB stream bound")
                .await;
        }
        if self
            .uploads
            .get(&stream)
            .expect("checked")
            .declared
            .is_some_and(|declared| total > declared)
        {
            return self
                .abort_upload(stream, "upload exceeds its declared total")
                .await;
        }
        // Queue the bytes, then drain full pieces without holding the
        // state borrow across lane awaits (disjoint field borrows would
        // still tangle with the abort path's `&mut self`).
        let drain = {
            let state = self.uploads.get_mut(&stream).expect("checked");
            state.buf.extend_from_slice(payload);
            state.received += payload.len() as u64;
            state.buf.len() >= kivi_chunk::DEFAULT_CHUNK_SIZE
        };
        if drain {
            self.drain_upload_pieces(stream).await?;
        }
        Ok(())
    }

    /// Stages every full buffered piece for one upload. Split-then-stage:
    /// pieces divide under the state borrow, stage over lane awaits, and
    /// land back under a fresh borrow — the borrow never crosses an await.
    async fn drain_upload_pieces(&mut self, stream: u64) -> Result<(), ConnExit> {
        loop {
            let piece = match self.uploads.get_mut(&stream) {
                None => return self.abort_upload(stream, "unknown stream").await,
                Some(state) => {
                    if state.buf.len() < kivi_chunk::DEFAULT_CHUNK_SIZE {
                        return Ok(());
                    }
                    let piece: Vec<u8> =
                        state.buf.drain(..kivi_chunk::DEFAULT_CHUNK_SIZE).collect();
                    Bytes::from(piece)
                }
            };
            let id = kivi_codec::integrity::chunk_id(self.chunks.domain, &piece);
            if let Err(error) = self.chunks.lane.stage_chunk_async(id, piece).await {
                tracing::warn!(stream, %error, "upload chunk staging failed");
                return self.abort_upload(stream, "chunk staging failed").await;
            }
            match self.uploads.get_mut(&stream) {
                None => return self.abort_upload(stream, "unknown stream").await,
                Some(state) => state
                    .entries
                    .push((id, kivi_chunk::DEFAULT_CHUNK_SIZE as u64)),
            }
        }
    }

    /// Commits one upload: flushes the tail piece, proves the manifest
    /// durable with one sync barrier, and admits the small root — the
    /// same chunks-first ordering as every other write path, with the
    /// stored version answered under the stream id.
    async fn handle_stream_commit(&mut self, stream: u64, payload: &[u8]) -> Result<(), ConnExit> {
        if !payload.is_empty() {
            return Err(ConnExit::Violation("stream-commit carries no payload"));
        }
        let Some(state) = self.uploads.remove(&stream) else {
            return self.abort_upload(stream, "unknown stream").await;
        };
        // `received` counts every accepted payload byte, buffered or
        // staged alike — the commit total needs no adjustment.
        let total = state.received;
        if state.declared.is_some_and(|declared| declared != total) {
            return self
                .abort_upload(stream, "upload length mismatches its declared total")
                .await;
        }
        // Take the owned parts out of the removed state; the state is
        // already gone, so a failure below answers abort with nothing to
        // roll back — the client re-uploads under a fresh id.
        let UploadState {
            key,
            tablet,
            identity,
            ack_floor,
            mut buf,
            mut entries,
            ..
        } = state;
        if !buf.is_empty() {
            let tail = Bytes::from(std::mem::take(&mut buf));
            let id = kivi_codec::integrity::chunk_id(self.chunks.domain, &tail);
            let len = tail.len() as u64;
            if let Err(error) = self.chunks.lane.stage_chunk_async(id, tail).await {
                tracing::warn!(stream, %error, "upload tail staging failed");
                return self.abort_upload(stream, "chunk staging failed").await;
            }
            entries.push((id, len));
        }
        let staged = match self.chunks.lane.finalize_stream_async(entries, total).await {
            Ok(staged) => staged,
            Err(error) => {
                tracing::warn!(stream, %error, "upload finalization failed");
                return self
                    .abort_upload(stream, "upload finalization failed")
                    .await;
            }
        };
        let guard = crate::chunk_lane::PinnedUpload::staged(&self.chunks.pins, &staged);
        let operation = Operation::SetChunked {
            key,
            manifest: staged.manifest,
            logical_len: staged.logical_len,
        };
        if self.durability.is_some() {
            return self
                .handle_request_durable(
                    stream,
                    identity,
                    ack_floor,
                    tablet,
                    &operation,
                    kivi_protocol::Opcode::Set,
                    Some(guard),
                    Vec::new(),
                    false,
                )
                .await;
        }
        match self.execute_local(tablet, &operation) {
            None => {
                self.respond(
                    stream,
                    kivi_protocol::Opcode::Set,
                    kivi_protocol::Response {
                        proof: None,
                        status: kivi_protocol::Status::NotLocal,
                        body: kivi_protocol::ResponseBody::Diagnostic(
                            "tablet not live on owner".to_owned(),
                        ),
                    },
                )
                .await
            }
            Some(Ok(result)) => {
                self.respond_result(
                    stream,
                    kivi_protocol::Opcode::Set,
                    tablet,
                    &result,
                    false,
                    None,
                )
                .await
            }
            Some(Err(error)) => {
                self.respond_op_error(stream, kivi_protocol::Opcode::Set, &error)
                    .await
            }
        }
    }

    /// Routes, executes, and answers one request directly on this worker.
    #[allow(clippy::too_many_lines)]
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
                        proof: None,
                        status: Status::InvalidRequest,
                        body: ResponseBody::Diagnostic("unknown namespace".to_owned()),
                    },
                )
                .await;
        }
        // Compound requests (multi-key scans and batches) route by their
        // dedicated payloads, never by hashing a single key.
        if matches!(
            request.opcode,
            kivi_protocol::Opcode::Scan | kivi_protocol::Opcode::AtomicBatch
        ) {
            let routing = self.routing.load();
            let endpoint_of = |worker: WorkerId| {
                self.endpoints
                    .load()
                    .get(&worker)
                    .cloned()
                    .unwrap_or_default()
            };
            let (response, opcode) = crate::compound::handle_compound(
                &self.tablets,
                &routing,
                self.worker,
                endpoint_of,
                self.durability.as_ref(),
                &self.fabric,
                &request,
            )
            .await;
            return self.respond(request_id, opcode, response).await;
        }
        // Index entries are maintained transactionally: direct single-key
        // writes to the reserved index prefix are rejected (the coherent
        // paths — `AtomicBatch` and `TxnPrepare` — bypass this check).
        // Reads and scans still serve them (index queries are scans).
        if is_direct_single_key_write(request.opcode) && kivi_state::is_index_key(&request.key) {
            return self
                .respond(
                    request_id,
                    request.opcode,
                    Response {
                        proof: None,
                        status: Status::InvalidRequest,
                        body: ResponseBody::Diagnostic(
                            "direct writes to index entries are forbidden; use indexed writes"
                                .to_owned(),
                        ),
                    },
                )
                .await;
        }
        let routing = self.routing.load();
        let Some(tablet) =
            crate::compound::route_point_key(routing.directory(), namespace, &request.key)
        else {
            return self
                .respond(
                    request_id,
                    request.opcode,
                    Response {
                        proof: None,
                        status: Status::NotLocal,
                        body: ResponseBody::Diagnostic("no tablet covers key".to_owned()),
                    },
                )
                .await;
        };
        let (tablet, owner) = match self.route_tablet(tablet) {
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
                            proof: None,
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
                            proof: None,
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
                        proof: None,
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
    /// the lane exists, direct in-memory execution otherwise. Large legacy
    /// `Set`s stage through the chunk lane first (suspending only this
    /// task) and re-enter as small `SetChunked` roots; chunked reads
    /// resolve through [`respond_result`](Self::respond_result).
    #[allow(clippy::too_many_lines)]
    async fn handle_routed_request(
        &mut self,
        request_id: u64,
        request: Request,
        tablet: TabletId,
    ) -> Result<(), ConnExit> {
        let identity = request.identity;
        let ack_floor = request.ack_floor;
        // The response envelope keeps the requested opcode — never the
        // post-split one — so `GetStream` answers stream-shaped even
        // though it executes the same read as `Get`.
        let requested = request.opcode;
        let streamed = requested == kivi_protocol::Opcode::GetStream;
        // Scan and AtomicBatch ride dedicated payloads with their own
        // consistency selectors; a freshness contract on them is a
        // caller bug — reject loudly instead of silently ignoring it.
        if matches!(
            requested,
            kivi_protocol::Opcode::Scan | kivi_protocol::Opcode::AtomicBatch
        ) && request.contract != ReadContract::Latest
        {
            return self
                .respond(
                    request_id,
                    requested,
                    kivi_protocol::Response {
                        proof: None,
                        status: kivi_protocol::Status::InvalidRequest,
                        body: kivi_protocol::ResponseBody::Diagnostic(
                            "freshness contracts ride point reads, not scans or batches".to_owned(),
                        ),
                    },
                )
                .await;
        }
        let contract = request.contract;
        // Scan and AtomicBatch never reach single-key translation (no
        // single key to route); the connection-level dispatcher handles
        // them before routing, so reaching here is a peer bug.
        let Some(operation) = request.into_operation() else {
            return self
                .respond(
                    request_id,
                    requested,
                    kivi_protocol::Response {
                        proof: None,
                        status: kivi_protocol::Status::InvalidRequest,
                        body: kivi_protocol::ResponseBody::Diagnostic(
                            "untranslatable operation".to_owned(),
                        ),
                    },
                )
                .await;
        };
        // Representation split on the owner: medium values stage into
        // the fabric synchronously (arena insert, suspends nothing),
        // large legacy `Set`s stage through the chunk lane first
        // (suspending only this task) and re-enter as small roots, so
        // the WAL only ever carries small roots. Range patches plan
        // against the peeked base first (the peek is synchronous; fabric
        // promotion below suspends only this task, and the split passes
        // the resulting roots straight through). Staging precedes
        // admission, so a lane failure or a rejected range answers here
        // with state untouched. A handled patch (`None`) already answered.
        let Some((operation, pinned, mut seals)) = self
            .plan_routed_range(request_id, tablet, operation)
            .await?
        else {
            return Ok(());
        };
        let Some((operation, pinned, staged)) = self
            .split_routed_set(request_id, tablet, operation, pinned)
            .await?
        else {
            return Ok(());
        };
        seals.extend(staged);
        // Prove every chunked and fabric reference a transaction carries
        // before admission: a dangling root is rejected loudly, never
        // committed. Medium transactional puts stage here (synchronous
        // fabric insert); their seals ride the durable admit below.
        let Some((operation, txn_seals)) =
            self.stage_routed_txn(request_id, tablet, operation).await?
        else {
            return Ok(());
        };
        seals.extend(txn_seals);
        if let Err(detail) = self.verify_txn_chunk_refs(&operation).await {
            return self
                .respond(
                    request_id,
                    requested,
                    kivi_protocol::Response {
                        proof: None,
                        status: kivi_protocol::Status::InvalidRequest,
                        body: kivi_protocol::ResponseBody::Diagnostic(detail),
                    },
                )
                .await;
        }
        if let Err(detail) = self.verify_txn_fabric_refs(&operation) {
            return self
                .respond(
                    request_id,
                    requested,
                    kivi_protocol::Response {
                        proof: None,
                        status: kivi_protocol::Status::InvalidRequest,
                        body: kivi_protocol::ResponseBody::Diagnostic(detail),
                    },
                )
                .await;
        }
        // Attribute escrow ownership: a bounded-counter create takes its
        // initial full share on the serving tablet (clients send zero).
        let mut operation = operation;
        if let Operation::BoundedCounterCreate { holder, .. } = &mut operation {
            *holder = tablet;
        }
        // Single-node contract gate: this replica holds the only copy, so
        // local applied state is definitionally fresh — Latest,
        // BoundedStale, and Any serve directly. AtLeast still validates
        // lineage (a foreign tablet/epoch token never validates) and
        // coverage; a token beyond applied names a write this owner has
        // not committed, which means retry rather than a wait that would
        // stall the tablet (Overloaded, retryable).
        if let ReadContract::AtLeast(token) = contract
            && !operation.is_mutating()
            && let Err(reject) = self.check_at_least(tablet, token)
        {
            use kivi_types::FreshnessReject;
            let (status, detail) = match &reject {
                FreshnessReject::BeyondApplied { .. } => (
                    kivi_protocol::Status::Overloaded,
                    "token beyond local applied state; retry".to_owned(),
                ),
                _ => (kivi_protocol::Status::StaleToken, reject.to_string()),
            };
            return self
                .respond(
                    request_id,
                    requested,
                    kivi_protocol::Response {
                        proof: None,
                        status,
                        body: kivi_protocol::ResponseBody::Diagnostic(detail),
                    },
                )
                .await;
        }
        if self.durability.is_some() {
            return self
                .handle_request_durable(
                    request_id, identity, ack_floor, tablet, &operation, requested, pinned, seals,
                    streamed,
                )
                .await;
        }
        match self.execute_local(tablet, &operation) {
            None => {
                self.respond(
                    request_id,
                    requested,
                    kivi_protocol::Response {
                        proof: None,
                        status: kivi_protocol::Status::NotLocal,
                        body: kivi_protocol::ResponseBody::Diagnostic(
                            "tablet not live on owner".to_owned(),
                        ),
                    },
                )
                .await
            }
            Some(Ok(result)) => {
                // Fabric references resolve here (suspending only this
                // task); superseded materializations retire afterwards.
                // All borrows end before the promotion await below.
                let key = operation.key().clone();
                let now = kivi_core::wall_now_or_max(&SystemClock);
                let (previous, reference) = {
                    let tablets = self.tablets.borrow();
                    let previous = tablets
                        .get(&tablet)
                        .and_then(|live| live.store().get(&key, now))
                        .and_then(kivi_state::StoredObject::fabric_ref);
                    let reference = match &result {
                        OperationResult::FabricValue {
                            fabric_id,
                            logical_len,
                            version,
                        } => Some(kivi_state::FabricRef {
                            id: *fabric_id,
                            logical_len: *logical_len,
                            version: *version,
                        }),
                        _ => None,
                    };
                    (previous, reference)
                };
                let result = match reference {
                    Some(reference) => {
                        match self.promote_value(tablet, &key, now, reference).await {
                            Ok(bytes) => {
                                let shaped = match &operation {
                                    Operation::GetRange { offset, len, .. } => {
                                        kivi_state::slice_range(&bytes, *offset, *len)
                                    }
                                    _ => bytes,
                                };
                                OperationResult::Value(Some(shaped))
                            }
                            Err(error) => {
                                return self
                                    .respond_fabric_error(request_id, requested, &error)
                                    .await;
                            }
                        }
                    }
                    None => result,
                };
                if previous.is_some() {
                    let now = kivi_core::wall_now_or_max(&SystemClock);
                    let post = self
                        .tablets
                        .borrow()
                        .get(&tablet)
                        .and_then(|live| live.store().get(&key, now).cloned());
                    if let Some(post) = post {
                        self.fabric.borrow_mut().retire_superseded(previous, &post);
                    }
                }
                let range = match &operation {
                    Operation::GetRange { offset, len, .. } => Some((*offset, *len)),
                    _ => None,
                };
                self.respond_result(request_id, requested, tablet, &result, streamed, range)
                    .await
            }
            Some(Err(error)) => self.respond_op_error(request_id, requested, &error).await,
        }
    }

    /// Plans one routed range patch against its peeked base: inline
    /// results and restaged `Set`s return for the split below, splices
    /// stage through the lane and return the `SetChunked` pair, and
    /// rejections answer at once. Returns `None` when the request is
    /// already answered (rejected or staging-failed); connection exits
    /// propagate like every other responder.
    async fn plan_routed_range(
        &mut self,
        request_id: u64,
        tablet: TabletId,
        operation: Operation,
    ) -> Result<
        Option<(
            Operation,
            Option<crate::chunk_lane::PinnedUpload>,
            Vec<crate::fabric::StagedSeal>,
        )>,
        ConnExit,
    > {
        let Operation::SetRange { key, offset, patch } = operation else {
            return Ok(Some((operation, None, Vec::new())));
        };
        let now = kivi_core::wall_now_or_max(&SystemClock);
        let base = crate::worker::peek_range_base(&self.tablets.borrow(), tablet, &key, now);
        // Fabric bases resolve synchronously when resident; otherwise the
        // lane promotion suspends only this task, then planning continues
        // against the resolved inline bytes. Each borrow ends before any
        // `.await` below.
        let base = match base {
            crate::chunk_lane::RangeBase::Fabric { fabric_id, .. } => {
                let resolved = self
                    .resolve_range_fabric_base(request_id, tablet, &key, now, fabric_id)
                    .await?;
                let Some(base) = resolved else {
                    return Ok(None);
                };
                base
            }
            other => other,
        };
        match crate::chunk_lane::plan_set_range(
            key,
            offset,
            patch,
            base,
            self.chunks.inline_threshold,
        ) {
            crate::chunk_lane::SetRangePlan::Inline(op) => Ok(Some((op, None, Vec::new()))),
            crate::chunk_lane::SetRangePlan::Reject { detail } => {
                self.respond_diagnostic(
                    request_id,
                    kivi_protocol::Opcode::SetRange,
                    kivi_protocol::Status::InvalidRequest,
                    detail,
                )
                .await?;
                Ok(None)
            }
            crate::chunk_lane::SetRangePlan::RestageSet { key, value } => {
                self.plan_routed_restage(request_id, tablet, key, value)
                    .await
            }
            crate::chunk_lane::SetRangePlan::Splice {
                key,
                manifest,
                logical_len,
                offset,
                patch,
            } => match self
                .chunks
                .lane
                .splice_range_async(manifest, logical_len, offset, patch)
                .await
            {
                Ok(staged) => {
                    let guard = crate::chunk_lane::PinnedUpload::staged(&self.chunks.pins, &staged);
                    // A splice preserves the live expiry like the inline
                    // path does: partial writes touch bytes, never the TTL.
                    Ok(Some((
                        Operation::SetConditionalChunked {
                            key,
                            manifest: staged.manifest,
                            logical_len: staged.logical_len,
                            condition: SetCondition::Always,
                            expiry: ExpiryPolicy::Keep,
                        },
                        Some(guard),
                        Vec::new(),
                    )))
                }
                Err(error) => {
                    self.respond_chunk_error(request_id, kivi_protocol::Opcode::SetRange, &error)
                        .await?;
                    Ok(None)
                }
            },
        }
    }

    /// Answers one executed result, resolving chunked reads through the
    /// chunk lane first. The tablet borrow is long gone by the time the
    /// lane is awaited: resolution suspends only this connection task,
    /// never the reactor, and the tablet stays the sole mutation owner
    /// throughout (§16, §17). `GetStream` completions fan out into
    /// `ValueStream*` frames instead of one response frame.
    /// Validates an `AtLeast` token against local single-node state.
    /// Lineage failures (wrong tablet/epoch/shape) answer `StaleToken`;
    /// a token beyond applied names a write this owner has not committed
    /// and answers retryable `Overloaded`. An unknown tablet falls through
    /// to the normal routing error below.
    ///
    /// # Errors
    ///
    /// Returns the freshness rejection; the caller shapes it onto the wire.
    fn check_at_least(
        &self,
        tablet: TabletId,
        token: CommitToken,
    ) -> Result<(), kivi_types::FreshnessReject> {
        let tablets = self.tablets.borrow();
        let Some(live) = tablets.get(&tablet) else {
            return Ok(());
        };
        // Single-node validation covers lineage and position only: the
        // process is its own fencing domain, so no incarnation participates
        // (covers_token ignores it; the placeholder never authorizes).
        let fresh = ReplicaFreshness::new(
            live.authority(),
            live.applied_commit(),
            NodeIncarnation::INITIAL,
        );
        fresh.covers_token(token)
    }

    async fn respond_result(
        &mut self,
        request_id: u64,
        opcode: kivi_protocol::Opcode,
        tablet: TabletId,
        result: &OperationResult,
        streamed: bool,
        range: Option<(u64, u64)>,
    ) -> Result<(), ConnExit> {
        if streamed {
            return self.respond_stream(request_id, result).await;
        }
        bump_direct(&self.metrics);
        match result {
            OperationResult::ChunkedValue {
                manifest,
                logical_len,
            } => match self
                .chunks
                .lane
                .read_value_async(*manifest, *logical_len)
                .await
            {
                Ok(bytes) => {
                    let value = match range {
                        None => bytes,
                        Some((offset, len)) => kivi_state::slice_range(&bytes, offset, len),
                    };
                    self.respond_ok(
                        request_id,
                        opcode,
                        tablet,
                        &OperationResult::Value(Some(value)),
                    )
                    .await
                }
                Err(error) => self.respond_chunk_error(request_id, opcode, &error).await,
            },
            // Unreachable: every producer resolves fabric references
            // before responding (ephemeral execute resolves inline,
            // durable reads suspend in the coordinator). Fails closed
            // rather than serving an unresolved reference.
            OperationResult::FabricValue { .. } => {
                use kivi_protocol::{Response, ResponseBody, Status};
                self.respond(
                    request_id,
                    opcode,
                    Response {
                        proof: None,
                        status: Status::Internal,
                        body: ResponseBody::Diagnostic("unresolved fabric reference".to_owned()),
                    },
                )
                .await
            }
            other => self.respond_ok(request_id, opcode, tablet, other).await,
        }
    }

    /// Answers one `GetStream` read as `ValueStream*` frames under the
    /// originating request id: `Begin{total}` first, `Data` payloads cut
    /// to the negotiated cap, then one empty `End`. Inline values stream
    /// from memory; chunked values stream entry by entry straight from
    /// packs, so downloads stay flat however large the value. Pre-begin
    /// failures answer as ordinary error responses (the client never saw
    /// a begin); once the begin is out, failures can only abort mid-stream.
    async fn respond_stream(
        &mut self,
        request_id: u64,
        result: &OperationResult,
    ) -> Result<(), ConnExit> {
        use kivi_protocol::{Response, ResponseBody, Status};
        bump_direct(&self.metrics);
        match result {
            OperationResult::Value(Some(bytes)) => self.write_value_stream(request_id, bytes).await,
            OperationResult::Value(None) => {
                self.respond(
                    request_id,
                    kivi_protocol::Opcode::GetStream,
                    Response {
                        proof: None,
                        status: Status::NotFound,
                        body: ResponseBody::Diagnostic(String::new()),
                    },
                )
                .await
            }
            OperationResult::ChunkedValue {
                manifest,
                logical_len,
            } => {
                let loaded = match self.chunks.lane.fetch_manifest_async(*manifest).await {
                    Ok(loaded) => loaded,
                    Err(error) => {
                        return self
                            .respond_chunk_error(
                                request_id,
                                kivi_protocol::Opcode::GetStream,
                                &error,
                            )
                            .await;
                    }
                };
                if loaded.total_len != *logical_len {
                    tracing::error!(%request_id, manifest = %manifest, claimed = logical_len, actual = loaded.total_len, "streamed root disagrees with its manifest");
                    return self
                        .respond(
                            request_id,
                            kivi_protocol::Opcode::GetStream,
                            Response {
                                proof: None,
                                status: Status::Internal,
                                body: ResponseBody::Diagnostic(
                                    "chunked value unavailable".to_owned(),
                                ),
                            },
                        )
                        .await;
                }
                self.write_frame(
                    kivi_protocol::FrameKind::ValueStreamBegin,
                    request_id,
                    &ValueStreamBegin {
                        total_len: *logical_len,
                    }
                    .encode(),
                )
                .await?;
                for entry in loaded.entries {
                    let bytes = match self.chunks.lane.read_chunk_async(entry.id).await {
                        Ok(bytes) => bytes,
                        Err(error) => {
                            tracing::error!(%request_id, %error, "streamed chunk read failed");
                            return self
                                .abort_value_stream(request_id, "chunked value unavailable")
                                .await;
                        }
                    };
                    if bytes.len() as u64 != entry.len {
                        tracing::error!(%request_id, chunk = %entry.id, "streamed chunk disagrees with its manifest entry");
                        return self
                            .abort_value_stream(request_id, "chunked value unavailable")
                            .await;
                    }
                    self.write_stream_data(request_id, &bytes).await?;
                }
                self.write_frame(kivi_protocol::FrameKind::ValueStreamEnd, request_id, &[])
                    .await
            }
            // Unreachable: `Get` executes to values or chunked values
            // (fabric references resolve before responding), or errors,
            // which never reach here. Fail closed, never guess.
            other => {
                tracing::error!(%request_id, result = ?other, "streamed read executed to a non-value");
                self.respond(
                    request_id,
                    kivi_protocol::Opcode::GetStream,
                    Response {
                        proof: None,
                        status: Status::Internal,
                        body: ResponseBody::Diagnostic("chunked value unavailable".to_owned()),
                    },
                )
                .await
            }
        }
    }

    /// Writes one in-memory value as a value stream (begin, capped data
    /// frames, end).
    async fn write_value_stream(
        &mut self,
        request_id: u64,
        bytes: &bytes::Bytes,
    ) -> Result<(), ConnExit> {
        self.write_frame(
            kivi_protocol::FrameKind::ValueStreamBegin,
            request_id,
            &ValueStreamBegin {
                total_len: bytes.len() as u64,
            }
            .encode(),
        )
        .await?;
        self.write_stream_data(request_id, bytes).await?;
        self.write_frame(kivi_protocol::FrameKind::ValueStreamEnd, request_id, &[])
            .await
    }

    /// Writes verbatim bytes as capped `ValueStreamData` frames. Awaiting
    /// each write is the backpressure: a slow client parks this task, not
    /// the reactor, and no frame ever exceeds the negotiated cap.
    async fn write_stream_data(&mut self, request_id: u64, bytes: &[u8]) -> Result<(), ConnExit> {
        let cap = self.stream_data_cap.max(1) as usize;
        for piece in bytes.chunks(cap) {
            self.write_frame(kivi_protocol::FrameKind::ValueStreamData, request_id, piece)
                .await?;
        }
        Ok(())
    }

    /// Ends a broken value stream: the client saw a begin but the packs
    /// stopped verifying, so only an abort (never a forged end) follows.
    async fn abort_value_stream(&mut self, request_id: u64, reason: &str) -> Result<(), ConnExit> {
        self.write_frame(
            kivi_protocol::FrameKind::StreamAbort,
            request_id,
            &StreamAbort {
                reason: reason.to_owned(),
            }
            .encode(),
        )
        .await
    }

    /// Answers a chunk fabric failure with its stable status. Overload and
    /// oversize are retriable signals; missing or corrupt immutable data
    /// is `Internal` (loud, never wrong bytes — later multi-source
    /// recovery will repair what local verification condemns).
    async fn respond_chunk_error(
        &mut self,
        request_id: u64,
        opcode: kivi_protocol::Opcode,
        error: &kivi_chunk::ChunkError,
    ) -> Result<(), ConnExit> {
        use kivi_protocol::{Response, ResponseBody, Status};
        bump_direct(&self.metrics);
        let (status, diagnostic) = match error {
            kivi_chunk::ChunkError::Overloaded => (Status::Overloaded, "chunk lane saturated"),
            kivi_chunk::ChunkError::TooLarge { .. } => (Status::ValueTooLarge, "value too large"),
            kivi_chunk::ChunkError::MissingChunk { .. }
            | kivi_chunk::ChunkError::MissingManifest { .. }
            | kivi_chunk::ChunkError::CorruptChunk { .. }
            | kivi_chunk::ChunkError::CorruptManifest { .. } => {
                tracing::error!(%request_id, %error, "chunked read failed verification");
                (Status::Internal, "chunked value unavailable")
            }
            kivi_chunk::ChunkError::Unsupported { .. }
            | kivi_chunk::ChunkError::Invalid { .. }
            | kivi_chunk::ChunkError::Io { .. } => {
                tracing::warn!(%request_id, %error, "chunk fabric failure");
                (Status::Internal, "chunk fabric failure")
            }
            // `ChunkError` is non-exhaustive across crates: future
            // variants fail loudly retriable-agnostic here by
            // construction (never wrong bytes, never a crash).
            _ => (Status::Internal, "chunk fabric failure"),
        };
        self.respond(
            request_id,
            opcode,
            Response {
                proof: None,
                status,
                body: ResponseBody::Diagnostic(diagnostic.to_owned()),
            },
        )
        .await
    }

    /// Answers a plain diagnostic failure: builds the `Response` so call
    /// sites stay one line. Kept beside [`respond_chunk_error`] so the
    /// routed planners stay reviewable.
    async fn respond_diagnostic(
        &mut self,
        request_id: u64,
        opcode: kivi_protocol::Opcode,
        status: kivi_protocol::Status,
        diagnostic: String,
    ) -> Result<(), ConnExit> {
        self.respond(
            request_id,
            opcode,
            kivi_protocol::Response {
                proof: None,
                status,
                body: kivi_protocol::ResponseBody::Diagnostic(diagnostic),
            },
        )
        .await
    }

    /// Plans one restaged range result: medium results stage into the
    /// fabric, large ones keep the conditional/Keep spelling so expiry
    /// preserves exactly like the inline splice does. Returns `None`
    /// when the request already answered (fabric saturation).
    async fn plan_routed_restage(
        &mut self,
        request_id: u64,
        tablet: TabletId,
        key: Key,
        value: Bytes,
    ) -> Result<
        Option<(
            Operation,
            Option<crate::chunk_lane::PinnedUpload>,
            Vec<crate::fabric::StagedSeal>,
        )>,
        ConnExit,
    > {
        // Restaged patches keep the conditional/Keep spelling so a
        // large result preserves expiry exactly like the inline
        // splice does. Medium results stage into the fabric.
        if value.len() > crate::fabric::FABRIC_INLINE_MAX
            && (value.len() as u64) <= self.chunks.inline_threshold
        {
            let staged = self.fabric.borrow_mut().stage(tablet, value.clone(), true);
            let Ok((fabric_id, logical_len)) = staged else {
                self.respond_diagnostic(
                    request_id,
                    kivi_protocol::Opcode::SetRange,
                    kivi_protocol::Status::SessionOverloaded,
                    "memory fabric saturated; retry".to_owned(),
                )
                .await?;
                return Ok(None);
            };
            Ok(Some((
                Operation::SetConditionalFabric {
                    key: key.clone(),
                    fabric_id,
                    logical_len,
                    version: 0,
                    condition: SetCondition::Always,
                    expiry: ExpiryPolicy::Keep,
                },
                None,
                vec![crate::fabric::StagedSeal {
                    fabric_id,
                    key,
                    bytes: value,
                }],
            )))
        } else {
            Ok(Some((
                Operation::SetConditional {
                    key,
                    value,
                    condition: SetCondition::Always,
                    expiry: ExpiryPolicy::Keep,
                },
                None,
                Vec::new(),
            )))
        }
    }

    /// Resolves one fabric range base: synchronous residency serves
    /// inline, otherwise the lane promotion suspends only this task and
    /// the reply plans against the promoted bytes. Returns `None` when
    /// the request already answered (stale root, saturation, or a lost
    /// base). Split into synchronous phases around the await so no
    /// tablet or fabric borrow is ever held across it.
    async fn resolve_range_fabric_base(
        &mut self,
        request_id: u64,
        tablet: TabletId,
        key: &Key,
        now: kivi_types::WallTimestamp,
        fabric_id: u64,
    ) -> Result<Option<crate::chunk_lane::RangeBase>, ConnExit> {
        let resident = self.fabric.borrow_mut().try_resolve_sync(fabric_id);
        if let Ok(bytes) = resident {
            return Ok(Some(crate::chunk_lane::RangeBase::Inline(bytes)));
        }
        let reference = self
            .tablets
            .borrow()
            .get(&tablet)
            .and_then(|live| live.store().get(key, now).cloned())
            .and_then(|object| object.fabric_ref());
        let Some(reference) = reference else {
            self.respond_diagnostic(
                request_id,
                kivi_protocol::Opcode::SetRange,
                kivi_protocol::Status::Internal,
                "range base changed under admission; retry".to_owned(),
            )
            .await?;
            return Ok(None);
        };
        match self.promote_value(tablet, key, now, reference).await {
            Ok(bytes) => Ok(Some(crate::chunk_lane::RangeBase::Inline(bytes))),
            Err(crate::fabric::FabricError::Overloaded) => {
                self.respond_diagnostic(
                    request_id,
                    kivi_protocol::Opcode::SetRange,
                    kivi_protocol::Status::SessionOverloaded,
                    "offcore lane saturated; retry".to_owned(),
                )
                .await?;
                Ok(None)
            }
            Err(_) => {
                self.respond_diagnostic(
                    request_id,
                    kivi_protocol::Opcode::SetRange,
                    kivi_protocol::Status::Internal,
                    "range base unavailable; retry".to_owned(),
                )
                .await?;
                Ok(None)
            }
        }
    }

    /// Promotes one fabric reference on behalf of this connection task:
    /// synchronous residency serves inline, otherwise the lane promotion
    /// suspends only this task and the reply re-fences id+version before
    /// serving. Split into synchronous phases around the await so no
    /// fabric or tablet borrow is ever held across it.
    async fn promote_value(
        &mut self,
        tablet: TabletId,
        key: &Key,
        now: kivi_types::WallTimestamp,
        reference: kivi_state::FabricRef,
    ) -> Result<Bytes, crate::fabric::FabricError> {
        use crate::fabric::FabricError;
        // Phase 1: synchronous check + submit (borrows end here).
        let reply = {
            let current = self
                .tablets
                .borrow()
                .get(&tablet)
                .and_then(|live| live.store().get(key, now).cloned());
            let mut fabric = self.fabric.borrow_mut();
            match fabric.resolve_sync(&reference, current.as_ref()) {
                Ok(Some(bytes)) => return Ok(bytes),
                Ok(None) => {
                    let live_version = current
                        .as_ref()
                        .and_then(kivi_state::StoredObject::fabric_ref)
                        .map_or(u64::MAX, |live| live.version);
                    fabric.begin_promote(&reference, live_version)?
                }
                Err(error) => return Err(error),
            }
        };
        // Phase 2: suspend only this task (no borrows held).
        let outcome = reply
            .recv_async()
            .await
            .map_err(|_| FabricError::Overloaded)?;
        // Phase 3: re-read the root, fence, and install (synchronous).
        let current = self
            .tablets
            .borrow()
            .get(&tablet)
            .and_then(|live| live.store().get(key, now).cloned());
        self.fabric
            .borrow_mut()
            .finish_promote(&reference, current.as_ref(), outcome)
    }

    /// Stages one medium routed `Set`/`SetConditional` into the fabric
    /// synchronously (arena insert, suspends nothing) and re-enters it
    /// as a small root. Saturation answers overload at once; small and
    /// large values pass through untouched.
    async fn stage_routed_medium(
        &mut self,
        request_id: u64,
        tablet: TabletId,
        operation: &Operation,
    ) -> Result<RoutedMedium, ConnExit> {
        let (key, value, opcode, conditional) = match operation {
            Operation::Set { key, value } => (key, value, kivi_protocol::Opcode::Set, None),
            Operation::SetConditional {
                key,
                value,
                condition,
                expiry,
            } => (
                key,
                value,
                kivi_protocol::Opcode::SetConditional,
                Some((*condition, *expiry)),
            ),
            _ => return Ok(RoutedMedium::Pass),
        };
        if value.len() <= crate::fabric::FABRIC_INLINE_MAX
            || (value.len() as u64) > self.chunks.inline_threshold
        {
            return Ok(RoutedMedium::Pass);
        }
        let staged = self.fabric.borrow_mut().stage(tablet, value.clone(), true);
        let Ok((fabric_id, logical_len)) = staged else {
            self.respond_diagnostic(
                request_id,
                opcode,
                kivi_protocol::Status::SessionOverloaded,
                "memory fabric saturated; retry".to_owned(),
            )
            .await?;
            return Ok(RoutedMedium::Answered);
        };
        let seals = vec![crate::fabric::StagedSeal {
            fabric_id,
            key: key.clone(),
            bytes: value.clone(),
        }];
        let staged = match conditional {
            None => Operation::SetFabric {
                key: key.clone(),
                fabric_id,
                logical_len,
                version: 0,
            },
            Some((condition, expiry)) => Operation::SetConditionalFabric {
                key: key.clone(),
                fabric_id,
                logical_len,
                version: 0,
                condition,
                expiry,
            },
        };
        Ok(RoutedMedium::Staged(staged, seals))
    }

    /// Splits one routed operation by size on the connection task:
    /// medium `Set`s stage into the fabric synchronously (arena insert,
    /// suspends nothing) and re-enter as small `SetFabric` roots; large
    /// values stage through the chunk lane (suspending only this task).
    /// Returns the admitted operation, its chunk pin, and fabric seals.
    /// `None` means already answered (staging failure, state untouched).
    async fn split_routed_set(
        &mut self,
        request_id: u64,
        tablet: TabletId,
        operation: Operation,
        pinned: Option<crate::chunk_lane::PinnedUpload>,
    ) -> Result<
        Option<(
            Operation,
            Option<crate::chunk_lane::PinnedUpload>,
            Vec<crate::fabric::StagedSeal>,
        )>,
        ConnExit,
    > {
        // Medium tier first (synchronous, never suspends). Each stage
        // completes (ending its borrow) before any `.await` below.
        match self
            .stage_routed_medium(request_id, tablet, &operation)
            .await?
        {
            RoutedMedium::Staged(op, seals) => return Ok(Some((op, pinned, seals))),
            RoutedMedium::Answered => return Ok(None),
            RoutedMedium::Pass => {}
        }
        match crate::chunk_lane::split_large_set(operation, self.chunks.inline_threshold) {
            crate::chunk_lane::LargeSetSplit::Inline(op) => Ok(Some((op, pinned, Vec::new()))),
            crate::chunk_lane::LargeSetSplit::Stage { key, value } => {
                let mut guard = match crate::chunk_lane::PinnedUpload::begin(
                    &self.chunks.pins,
                    self.chunks.domain,
                    &value,
                ) {
                    Ok(guard) => guard,
                    Err(error) => {
                        return self
                            .respond_chunk_error(request_id, kivi_protocol::Opcode::Set, &error)
                            .await
                            .map(|()| None);
                    }
                };
                match self.chunks.lane.stage_value_async(value).await {
                    Ok(staged) => {
                        guard.set_manifest(staged.manifest);
                        Ok(Some((
                            Operation::SetChunked {
                                key,
                                manifest: staged.manifest,
                                logical_len: staged.logical_len,
                            },
                            Some(guard),
                            Vec::new(),
                        )))
                    }
                    Err(error) => self
                        .respond_chunk_error(request_id, kivi_protocol::Opcode::Set, &error)
                        .await
                        .map(|()| None),
                }
            }
            crate::chunk_lane::LargeSetSplit::StageConditional {
                key,
                value,
                condition,
                expiry,
            } => {
                let mut guard = match crate::chunk_lane::PinnedUpload::begin(
                    &self.chunks.pins,
                    self.chunks.domain,
                    &value,
                ) {
                    Ok(guard) => guard,
                    Err(error) => {
                        return self
                            .respond_chunk_error(
                                request_id,
                                kivi_protocol::Opcode::SetConditional,
                                &error,
                            )
                            .await
                            .map(|()| None);
                    }
                };
                match self.chunks.lane.stage_value_async(value).await {
                    Ok(staged) => {
                        guard.set_manifest(staged.manifest);
                        Ok(Some((
                            Operation::SetConditionalChunked {
                                key,
                                manifest: staged.manifest,
                                logical_len: staged.logical_len,
                                condition,
                                expiry,
                            },
                            Some(guard),
                            Vec::new(),
                        )))
                    }
                    Err(error) => self
                        .respond_chunk_error(
                            request_id,
                            kivi_protocol::Opcode::SetConditional,
                            &error,
                        )
                        .await
                        .map(|()| None),
                }
            }
        }
    }

    /// Stages medium transactional puts into the fabric on the connection
    /// task (synchronous inserts), rewriting them as `PutFabric`
    /// references. Returns the rewritten operation plus seals. `None`
    /// means already answered (staging failure, state untouched).
    async fn stage_routed_txn(
        &mut self,
        request_id: u64,
        tablet: TabletId,
        operation: Operation,
    ) -> Result<Option<(Operation, Vec<crate::fabric::StagedSeal>)>, ConnExit> {
        match operation {
            Operation::TxnPrepare {
                txn,
                coordinator,
                write,
                digest,
            } => {
                if let Some((write, seal)) = self.stage_routed_txn_write(tablet, write) {
                    Ok(Some((
                        Operation::TxnPrepare {
                            txn,
                            coordinator,
                            write,
                            digest,
                        },
                        seal.into_iter().collect(),
                    )))
                } else {
                    self.respond(
                        request_id,
                        kivi_protocol::Opcode::TxnPrepare,
                        kivi_protocol::Response {
                            proof: None,
                            status: kivi_protocol::Status::SessionOverloaded,
                            body: kivi_protocol::ResponseBody::Diagnostic(
                                "memory fabric saturated; retry".to_owned(),
                            ),
                        },
                    )
                    .await?;
                    Ok(None)
                }
            }
            Operation::TxnCommitLocal { txn, writes } => {
                let mut staged_writes = Vec::with_capacity(writes.len());
                let mut seals = Vec::new();
                for write in writes {
                    if let Some((write, seal)) = self.stage_routed_txn_write(tablet, write) {
                        seals.extend(seal);
                        staged_writes.push(write);
                    } else {
                        self.respond(
                            request_id,
                            kivi_protocol::Opcode::AtomicBatch,
                            kivi_protocol::Response {
                                proof: None,
                                status: kivi_protocol::Status::SessionOverloaded,
                                body: kivi_protocol::ResponseBody::Diagnostic(
                                    "memory fabric saturated; retry".to_owned(),
                                ),
                            },
                        )
                        .await?;
                        return Ok(None);
                    }
                }
                Ok(Some((
                    Operation::TxnCommitLocal {
                        txn,
                        writes: staged_writes,
                    },
                    seals,
                )))
            }
            op @ Operation::TxnFinalize { .. } => {
                self.seal_routed_finalize(request_id, tablet, op).await
            }
            other => Ok(Some((other, Vec::new()))),
        }
    }

    /// Attaches a routed commit-finalize's prepared intent bytes as a seal
    /// payload (reactor twin of the worker path): synchronous residency
    /// serves inline, otherwise the lane promotion suspends only this
    /// task. Aborts and non-fabric intents attach nothing (prepare
    /// resolves those terminally). `None` means already answered: a
    /// dangling intent reference fails closed, never commits.
    async fn seal_routed_finalize(
        &mut self,
        request_id: u64,
        tablet: TabletId,
        op: Operation,
    ) -> Result<Option<(Operation, Vec<crate::fabric::StagedSeal>)>, ConnExit> {
        let now = kivi_core::wall_now_or_max(&SystemClock);
        let finalize_key = op.key().clone();
        let commit = matches!(&op, Operation::TxnFinalize { commit: true, .. });
        let fabric_id = commit
            .then(|| {
                self.tablets
                    .borrow()
                    .get(&tablet)
                    .and_then(|live| live.store().intent_for_key(&finalize_key))
                    .and_then(|intent| match &intent.write.kind {
                        kivi_state::TxnWriteKind::PutFabric { fabric_id, .. } => Some(*fabric_id),
                        _ => None,
                    })
            })
            .flatten();
        let Some(fabric_id) = fabric_id else {
            return Ok(Some((op, Vec::new())));
        };
        let bytes = {
            if let Ok(bytes) = self.fabric.borrow().try_resolve_sync(fabric_id) {
                bytes
            } else {
                let reference = self
                    .tablets
                    .borrow()
                    .get(&tablet)
                    .and_then(|live| live.store().get(&finalize_key, now).cloned())
                    .and_then(|object| object.fabric_ref())
                    .filter(|reference| reference.id == fabric_id);
                let Some(reference) = reference else {
                    return self
                        .respond(
                            request_id,
                            kivi_protocol::Opcode::TxnFinalize,
                            kivi_protocol::Response {
                                proof: None,
                                status: kivi_protocol::Status::InvalidRequest,
                                body: kivi_protocol::ResponseBody::Diagnostic(
                                    "finalize references unavailable payload".to_owned(),
                                ),
                            },
                        )
                        .await
                        .map(|()| None);
                };
                match self
                    .promote_value(tablet, &finalize_key, now, reference)
                    .await
                {
                    Ok(bytes) => bytes,
                    Err(_) => {
                        return self
                            .respond(
                                request_id,
                                kivi_protocol::Opcode::TxnFinalize,
                                kivi_protocol::Response {
                                    proof: None,
                                    status: kivi_protocol::Status::SessionOverloaded,
                                    body: kivi_protocol::ResponseBody::Diagnostic(
                                        "offcore lane saturated; retry".to_owned(),
                                    ),
                                },
                            )
                            .await
                            .map(|()| None);
                    }
                }
            }
        };
        Ok(Some((
            op,
            vec![crate::fabric::StagedSeal {
                fabric_id,
                key: finalize_key,
                bytes,
            }],
        )))
    }

    /// Stages one transactional write's medium inline payload (sync).
    /// Returns the rewritten write plus its seal; `None` on saturation.
    fn stage_routed_txn_write(
        &mut self,
        tablet: TabletId,
        write: kivi_state::TxnWrite,
    ) -> Option<(kivi_state::TxnWrite, Option<crate::fabric::StagedSeal>)> {
        match write.kind {
            kivi_state::TxnWriteKind::Put(value)
                if value.len() > crate::fabric::FABRIC_INLINE_MAX
                    && (value.len() as u64) <= self.chunks.inline_threshold =>
            {
                let (fabric_id, logical_len) = self
                    .fabric
                    .borrow_mut()
                    .stage(tablet, value.clone(), true)
                    .ok()?;
                Some((
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
                ))
            }
            _ => Some((write, None)),
        }
    }

    /// Proves every fabric reference a transaction carries names live
    /// local bytes (synchronous check, never blocks the reactor):
    /// staged uploads and committed materializations alike pass.
    /// Returns the diagnostic for the rejection response.
    fn verify_txn_fabric_refs(&self, operation: &Operation) -> Result<(), String> {
        let refs: Vec<u64> = match operation {
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
        // Availability without I/O: a synchronous residence or a device
        // locator proves the bytes exist locally. Staged uploads pass
        // (just inserted above and still alive).
        let fabric = self.fabric.borrow_mut();
        for fabric_id in refs {
            let resident = fabric.try_resolve_sync(fabric_id).is_ok()
                || fabric.offcore_locator(fabric_id).is_some();
            if !resident {
                return Err(format!(
                    "transactional fabric write references unavailable payload: {fabric_id}"
                ));
            }
        }
        Ok(())
    }

    /// Proves every chunked reference a transaction carries names durable
    /// local payload (async twin of the worker path's blocking check).
    async fn verify_txn_chunk_refs(&self, operation: &Operation) -> Result<(), String> {
        let refs: Vec<(kivi_types::ManifestId, u64)> = match operation {
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
            if let Err(error) = self
                .chunks
                .lane
                .check_root_async(manifest, logical_len)
                .await
            {
                return Err(format!(
                    "transactional chunked write references unavailable payload: {error}"
                ));
            }
        }
        Ok(())
    }

    /// Pumps the shared commit coordinator (no-op in ephemeral mode).
    /// Synchronous borrows only; never held across an `.await`.
    fn pump_coordinator(&mut self) {
        if let Some(cell) = self.durability.as_ref() {
            let mut durable = cell.borrow_mut();
            let namespace = durable.namespace;
            durable.commit.poll(
                &mut self.tablets.borrow_mut(),
                namespace,
                &mut self.fabric.borrow_mut(),
            );
        }
    }

    /// Answers every outbox entry whose batch proved (or failed), in
    /// completion order. Entries answered here leave the outbox; the rest
    /// keep waiting for later turns.
    async fn flush_outbox(&mut self) -> Result<(), ConnExit> {
        let mut ready = Vec::new();
        let mut cursor = 0usize;
        while cursor < self.outbox.len() {
            match self.outbox[cursor].receive.try_recv() {
                Ok(outcome) => {
                    let entry = self.outbox.remove(cursor).expect("cursor valid");
                    ready.push((
                        entry.request_id,
                        entry.opcode,
                        entry.tablet,
                        entry.streamed,
                        entry.range,
                        outcome,
                    ));
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    cursor += 1;
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    // Coordinator gone without answering: the worker is
                    // exiting. Sealed work still proves on the lane and
                    // replays on restart; retries reuse identities.
                    return Err(ConnExit::Io);
                }
            }
        }
        for (request_id, opcode, tablet, streamed, range, outcome) in ready {
            match outcome {
                // `respond_result` resolves chunked reads (bumping itself)
                // and fans `GetStream` completions into value streams;
                // every other outcome answers inline as before.
                Ok(result) => {
                    self.respond_result(request_id, opcode, tablet, &result, streamed, range)
                        .await?;
                }
                Err(error) => {
                    self.respond_worker_error(request_id, opcode, &error)
                        .await?;
                }
            }
        }
        Ok(())
    }

    /// Answers a Memory Fabric failure with its stable status.
    /// Saturation is retryable backpressure (`SessionOverloaded`, like
    /// chunk-lane saturation); corruption fails the single read closed;
    /// unavailability (moved root, retired id) asks for a retry that
    /// re-linearizes.
    async fn respond_fabric_error(
        &mut self,
        request_id: u64,
        opcode: kivi_protocol::Opcode,
        error: &crate::fabric::FabricError,
    ) -> Result<(), ConnExit> {
        use kivi_protocol::{Response, ResponseBody, Status};
        let (status, detail) = match error {
            crate::fabric::FabricError::Overloaded => (
                Status::SessionOverloaded,
                "memory fabric saturated; retry".to_owned(),
            ),
            crate::fabric::FabricError::Corrupt => (
                Status::Internal,
                "fabric record failed verification".to_owned(),
            ),
            crate::fabric::FabricError::Unavailable => (
                Status::Internal,
                "fabric reference unavailable; retry".to_owned(),
            ),
        };
        self.respond(
            request_id,
            opcode,
            Response {
                proof: None,
                status,
                body: ResponseBody::Diagnostic(detail),
            },
        )
        .await
    }

    /// Maps a worker-side request failure onto the wire (shared by inline
    /// and outbox replies so both paths speak identically).
    async fn respond_worker_error(
        &mut self,
        request_id: u64,
        opcode: kivi_protocol::Opcode,
        error: &WorkerRequestError,
    ) -> Result<(), ConnExit> {
        use kivi_protocol::{Response, ResponseBody, Status};
        match error {
            WorkerRequestError::Tablet(error) => {
                self.respond_op_error(request_id, opcode, error).await
            }
            WorkerRequestError::UnknownTablet { .. } => {
                self.respond(
                    request_id,
                    opcode,
                    Response {
                        proof: None,
                        status: Status::NotLocal,
                        body: ResponseBody::Diagnostic("tablet not live on owner".to_owned()),
                    },
                )
                .await
            }
            WorkerRequestError::DedupExpired => {
                self.respond(
                    request_id,
                    opcode,
                    Response {
                        proof: None,
                        status: Status::DedupExpired,
                        body: ResponseBody::Diagnostic(
                            "mutation identity expired below the session floor".to_owned(),
                        ),
                    },
                )
                .await
            }
            WorkerRequestError::SessionOverloaded => {
                self.respond(
                    request_id,
                    opcode,
                    Response {
                        proof: None,
                        status: Status::SessionOverloaded,
                        body: ResponseBody::Diagnostic(
                            "session outcome window exhausted".to_owned(),
                        ),
                    },
                )
                .await
            }
            WorkerRequestError::ChunkStore(error) => {
                self.respond_chunk_error(request_id, opcode, error).await
            }
            WorkerRequestError::Fabric(error) => {
                self.respond_fabric_error(request_id, opcode, error).await
            }
            WorkerRequestError::InvalidRequest { detail } => {
                self.respond(
                    request_id,
                    opcode,
                    Response {
                        proof: None,
                        status: Status::InvalidRequest,
                        body: ResponseBody::Diagnostic(detail.clone()),
                    },
                )
                .await
            }
            WorkerRequestError::Storage(error) => {
                self.respond_storage_error(request_id, opcode, error).await
            }
        }
    }

    /// Durable write intake on the owning worker: validates, admits to the
    /// commit coordinator, and queues the reply slot — never waits.
    /// Never replies success before the record is file-synced: the reply
    /// leaves through [`flush_outbox`](Self::flush_outbox) once the batch
    /// proves. All borrows end before any `.await`, so connection tasks on
    /// this thread cannot interleave a persist-apply sequence.
    // Nine parameters: routing, identity, payload, opcode, the pin guard
    // this stage adds, fabric seals for medium values, plus the stream
    // flag completions answer with. Grouping would hide the intake order
    // the surrounding code documents; the callers build it explicitly.
    #[allow(clippy::too_many_arguments)]
    async fn handle_request_durable(
        &mut self,
        request_id: u64,
        identity: Option<kivi_types::RequestIdentity>,
        ack_floor: kivi_types::RequestSeq,
        tablet: TabletId,
        operation: &Operation,
        opcode: kivi_protocol::Opcode,
        pinned: Option<crate::chunk_lane::PinnedUpload>,
        seals: Vec<crate::fabric::StagedSeal>,
        streamed: bool,
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
                            proof: None,
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
        let now = kivi_core::wall_now_or_max(&SystemClock);
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
                        proof: None,
                        status: Status::NotLocal,
                        body: ResponseBody::Diagnostic("tablet not live on owner".to_owned()),
                    },
                )
                .await;
        }
        let durability = self
            .durability
            .as_ref()
            .expect("durable path always has a lane");
        // Pipelining bound across decode backlog plus outbox: deeper gets
        // closed, never an unbounded queue.
        if self.outbox.len() >= self.max_pipelined {
            return Err(ConnExit::Violation("pipelined backlog bound"));
        }
        let (respond, receive) = crossbeam_channel::bounded::<crate::worker::WorkerResponse>(1);
        {
            let mut durable = durability.borrow_mut();
            let namespace = durable.namespace;
            // The coordinator entry owns the upload's pins and fabric
            // seals until after its apply journals the commit; the
            // journal then covers the response window, so the outbox
            // needs no guard of its own.
            durable.commit.admit(
                crate::commit::PendingEntry::new(tablet, operation.clone(), now, identity, respond)
                    .with_pinned(pinned)
                    .with_fabric_staged(seals),
            );
            durable.commit.poll(
                &mut self.tablets.borrow_mut(),
                namespace,
                &mut self.fabric.borrow_mut(),
            );
        }
        let range = match operation {
            Operation::GetRange { offset, len, .. } => Some((*offset, *len)),
            _ => None,
        };
        self.outbox.push_back(OutboxEntry {
            request_id,
            opcode,
            tablet,
            streamed,
            range,
            receive,
        });
        Ok(())
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
                proof: None,
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
        let now = kivi_core::wall_now_or_max(&SystemClock);
        let mut tablets = self.tablets.borrow_mut();
        tablets
            .get_mut(&tablet)
            .map(|live| live.execute(operation, now))
    }

    /// Evaluates the current routing snapshot for one tablet: local
    /// ownership executes, remote ownership redirects, missing placement
    /// answers nowhere. Key → tablet resolution happens before this
    /// (unified rule in `compound::route_point_key`); this only checks
    /// placement of an already-resolved tablet.
    fn route_tablet(&self, tablet: TabletId) -> Route {
        let routing = self.routing.load();
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
        let stream = self.stream.as_mut().expect("write half held");
        let outcome = stream.write_all(bytes).await;
        outcome.0.map_err(|_| ConnExit::Io)?;
        Ok(())
    }

    /// Answers with an operation outcome mapped onto the status taxonomy.
    /// `tablet` names the serving tablet for tablet-naming bodies (drivers
    /// learn placement from `TxnPrepared` without extra probes).
    #[allow(clippy::too_many_lines)]
    async fn respond_ok(
        &mut self,
        request_id: u64,
        opcode: kivi_protocol::Opcode,
        tablet: TabletId,
        outcome: &kivi_state::OperationResult,
    ) -> Result<(), ConnExit> {
        use kivi_protocol::{Response, ResponseBody, Status};
        use kivi_state::OperationResult as R;
        let response = match outcome {
            R::Value(Some(value)) => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::Value(value.to_vec()),
            },
            R::Value(None)
            | R::Counter(None)
            | R::CommutativeValue(None)
            | R::Expiry(None)
            | R::Length(None)
            | R::Version(None) => Response {
                proof: None,
                status: Status::NotFound,
                body: ResponseBody::Diagnostic(String::new()),
            },
            R::Version(Some(version)) => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::Version(version.as_u64()),
            },
            R::Length(Some(len)) => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::Length(*len),
            },
            R::ConditionalSet { applied, version } => {
                // Restaged range writes execute as conditional/chunked ops
                // internally but were requested as `SetRange`: shape them
                // as the `Stored` the opcode promises (the version is the
                // real post-store version either way). A not-applied
                // conditional under `SetRange` is unreachable (`Always`
                // always applies) and fails closed here, never as a forged
                // `Stored`.
                if *applied && opcode == kivi_protocol::Opcode::SetRange {
                    match version {
                        Some(version) => Response {
                            proof: None,
                            status: Status::Ok,
                            body: ResponseBody::Stored {
                                version: version.as_u64(),
                            },
                        },
                        None => Response {
                            proof: None,
                            status: Status::Internal,
                            body: ResponseBody::Diagnostic(
                                "conditional set applied without a version".to_owned(),
                            ),
                        },
                    }
                } else {
                    Response {
                        proof: None,
                        status: Status::Ok,
                        body: ResponseBody::ConditionalSet {
                            applied: *applied,
                            version: version.map_or(0, kivi_state::ObjectVersion::as_u64),
                        },
                    }
                }
            }
            R::Stored { version } => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::Stored {
                    version: version.as_u64(),
                },
            },
            R::Deleted { existed } => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::Deleted { existed: *existed },
            },
            R::Exists(present) => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::Exists(*present),
            },
            R::Counter(Some(value)) => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::Counter(*value),
            },
            R::CounterUpdated { value, version } => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::CounterUpdated {
                    value: *value,
                    version: version.as_u64(),
                },
            },
            R::ExpirySet { applied } => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::ExpirySet { applied: *applied },
            },
            R::ExpiryPersisted { removed } => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::ExpiryPersisted { removed: *removed },
            },
            R::Expiry(Some(expiry)) => Response {
                proof: None,
                status: Status::Ok,
                body: match expiry.as_stamp() {
                    None => ResponseBody::ExpiryNever,
                    Some(stamp) => ResponseBody::ExpiryAt(stamp.as_micros()),
                },
            },
            R::TxnPrepared => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::TxnPrepared {
                    tablet: tablet.as_u64(),
                    dir_version: self.routing.load().version().as_u64(),
                },
            },
            R::TxnConflict => Response {
                proof: None,
                status: Status::TxnConflict,
                body: ResponseBody::Diagnostic("transaction conflict".to_owned()),
            },
            R::TxnFinalized { applied, version } => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::TxnFinalized {
                    applied: *applied,
                    version: version.map_or(0, kivi_state::ObjectVersion::as_u64),
                    tablet: tablet.as_u64(),
                },
            },
            R::TxnLocalCommitted { versions } => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::AtomicCommitted {
                    versions: versions
                        .iter()
                        .map(|version| version.map(kivi_state::ObjectVersion::as_u64))
                        .collect(),
                },
            },
            R::CommutativeApplied => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::CommutativeApplied,
            },
            R::CommutativeValue(Some(value)) => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::CommutativeValue(*value),
            },
            R::BoundedUpdated { value, version } => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::BoundedUpdated {
                    value: *value,
                    version: version.as_u64(),
                },
            },
            R::BoundedValue {
                value: Some(value),
                capacity: Some(capacity),
                share: Some((share_min, share_max)),
            } => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::BoundedValue {
                    value: *value,
                    capacity: *capacity,
                    share_min: *share_min,
                    share_max: *share_max,
                },
            },
            // Absent bounded counters arrive as `NotFound` (handled with
            // the other absent reads above); a half-present value here is
            // divergent state, never a client answer.
            R::BoundedValue { .. } => Response {
                proof: None,
                status: Status::Internal,
                body: ResponseBody::Diagnostic("bounded counter value without capacity".to_owned()),
            },
            R::SemaphoreAcquired => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::SemaphoreAcquired,
            },
            R::SemaphoreReleased { released } => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::SemaphoreReleased {
                    released: *released,
                },
            },
            R::SemaphoreLoad {
                outstanding,
                capacity,
            } => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::SemaphoreLoad {
                    outstanding: *outstanding,
                    capacity: *capacity,
                },
            },
            R::LeaseAcquired {
                fencing,
                expires_at,
            } => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::LeaseAcquired {
                    fencing: fencing.as_u64(),
                    expires_at: expires_at.as_micros(),
                },
            },
            R::LeaseRenewed {
                fencing,
                expires_at,
            } => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::LeaseRenewed {
                    fencing: fencing.as_u64(),
                    expires_at: expires_at.as_micros(),
                },
            },
            R::LeaseReleased { released } => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::LeaseReleased {
                    released: *released,
                },
            },
            R::LeaseInfo {
                holder,
                next_fencing,
            } => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::LeaseInfo {
                    owner: holder.map(|holder| holder.owner),
                    fencing: holder.map_or(0, |holder| holder.fencing.as_u64()),
                    expires_at: holder.map_or(0, |holder| holder.expires_at.as_micros()),
                    next_fencing: next_fencing.as_u64(),
                },
            },
            R::StreamCreated => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::StreamCreated,
            },
            R::StreamAppended { offset } => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::StreamAppended { offset: *offset },
            },
            R::StreamEntries {
                entries,
                next_offset,
            } => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::StreamEntries {
                    entries: entries
                        .iter()
                        .take(MAX_STREAM_PAGE_ENTRIES)
                        .map(|entry| kivi_protocol::StreamEntryBody {
                            offset: entry.offset,
                            partition: entry.partition.clone(),
                            payload: entry.payload.to_vec(),
                        })
                        .collect(),
                    next_offset: *next_offset,
                },
            },
            R::StreamTrimmed { removed } => Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::StreamTrimmed { removed: *removed },
            },
            // Unreachable by construction: the engine resolves chunked
            // reads through the chunk lane (and fabric reads through the
            // Memory Fabric) before responding, so a reference value here
            // is a missed resolution path. Answer loudly retriable
            // `Internal` (never wrong bytes, never a crash) — and the
            // resolution tests below pin the real paths.
            R::ChunkedValue { .. } => Response {
                proof: None,
                status: Status::Internal,
                body: ResponseBody::Diagnostic(
                    "chunked value reached the wire unresolved".to_owned(),
                ),
            },
            R::FabricValue { .. } => Response {
                proof: None,
                status: Status::Internal,
                body: ResponseBody::Diagnostic(
                    "fabric value reached the wire unresolved".to_owned(),
                ),
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
            // Transient admission race: the range base changed
            // representation under the request. Safe to retry at once —
            // the retry re-plans against the new root.
            E::Op(kivi_state::OpError::StaleRangeBase) => {
                (Status::InvalidRequest, "range base changed; retry")
            }
            E::Op(kivi_state::OpError::TxnConflict) => {
                (Status::TxnConflict, "transaction conflict")
            }
            // Prepare-time and apply-time verdicts share the wire shape:
            // admission validates before the WAL, so an apply-side
            // semantic error means same-log/same-state divergence — fail
            // closed with the same typed status either way.
            E::Op(kivi_state::OpError::BoundedExceeded)
            | E::Apply(kivi_state::ApplyError::BoundedExceeded) => (
                Status::BoundedExceeded,
                "bounded counter would exceed owned rights",
            ),
            E::Op(kivi_state::OpError::SemaphoreExhausted)
            | E::Apply(kivi_state::ApplyError::SemaphoreExhausted) => {
                (Status::SemaphoreExhausted, "semaphore capacity exhausted")
            }
            E::Op(kivi_state::OpError::LeaseConflict) => {
                (Status::LeaseConflict, "lease held by another live owner")
            }
            E::Op(kivi_state::OpError::StaleFencing) => {
                (Status::StaleFencing, "stale fencing token")
            }
            E::Op(kivi_state::OpError::StreamFull)
            | E::Apply(kivi_state::ApplyError::StreamFull) => {
                (Status::StreamFull, "stream shard full")
            }
            E::Op(kivi_state::OpError::NotFound) => (Status::NotFound, "not found"),
            E::Apply(kivi_state::ApplyError::VersionExhausted) => {
                (Status::VersionExhausted, "version exhausted")
            }
            E::Apply(kivi_state::ApplyError::CounterOverflow) => {
                (Status::CounterOverflow, "overflow")
            }
            E::Apply(kivi_state::ApplyError::TypeMismatch { .. }) => {
                (Status::WrongType, "wrong type")
            }
            // Divergent-state splice: a corrupt or forked record met a
            // chunked base. Never a client retry — surface as internal.
            E::Apply(kivi_state::ApplyError::UnresolvableSplice) => {
                (Status::Internal, "splice cannot apply")
            }
            // Transaction divergence: prepare-time validation passed but
            // apply disagreed (same log on same state never does this).
            // Fail closed as internal, never partial intent state.
            E::Apply(kivi_state::ApplyError::TxnDiverged) => {
                (Status::Internal, "transaction diverged")
            }
            // Semantic divergence: same-log/same-state never produces
            // these (admission validates before the WAL). Fail closed.
            E::Apply(kivi_state::ApplyError::LeaseConflict) => {
                (Status::LeaseConflict, "lease grant conflict")
            }
            E::OpScan(_) => (Status::InvalidRequest, "scan rejected"),
            E::AuthorityMismatch { .. } => (Status::Internal, "authority mismatch"),
            E::CommitExhausted { .. } => (Status::VersionExhausted, "commit space exhausted"),
        };
        self.respond(
            request_id,
            opcode,
            Response {
                proof: None,
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

/// Routing verdict for one tablet under the current snapshot.
enum Route {
    /// Execute on the named local tablet.
    Execute(TabletId),
    /// Answer with this redirect (owner elsewhere, or stale hint).
    Redirect(RedirectInfo),
    /// No authority covers the tablet.
    Nowhere,
}

/// Whether `opcode` is a direct single-key mutating write (the only shapes
/// subject to the index-prefix admission rule). Reads, scans, batches, and
/// transaction steps bypass it: batches and transaction steps are the
/// coherent index-maintenance paths, reads never mutate.
fn is_direct_single_key_write(opcode: kivi_protocol::Opcode) -> bool {
    use kivi_protocol::Opcode as O;
    matches!(
        opcode,
        O::Set
            | O::Delete
            | O::CounterAdd
            | O::ExpireAt
            | O::PersistExpiry
            | O::SetRange
            | O::SetConditional
    )
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
    let (frames_tx, frames_rx) =
        async_channel::bounded::<kivi_protocol::Frame>(max_pipelined.max(1));
    let mut conn = Conn {
        stream: Some(stream),
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
        chunks: shared.chunks.clone(),
        fabric: Rc::clone(&shared.fabric),
        reader: Some(FrameReader::new(shared.net.conn.max_frame)),
        frames: frames_rx,
        max_input,
        max_pipelined,
        turn: shared.net.turn,
        shook_hands: false,
        frames_this_turn: 0,
        bytes_this_turn: 0,
        outbox: std::collections::VecDeque::new(),
        uploads: HashMap::new(),
        // Negotiated below in the handshake; this default only covers
        // the pre-handshake window, where stream frames are violations.
        stream_data_cap: 1,
    };
    // Handshake first on the unified stream, then split: the reader task
    // owns framing from here on (never cancelled), the serving task owns
    // the write half plus the reply outbox.
    let extras = match conn.handshake().await {
        Ok(extras) => extras,
        Err(exit) => {
            log_conn_exit(peer, exit);
            return;
        }
    };
    let stream = conn.stream.take().expect("handshake keeps the stream");
    let parser = conn.reader.take().expect("handshake keeps the parser");
    let (read_stream, write_stream) = stream.into_split();
    conn.stream = Some(write_stream);
    for frame in extras {
        if frames_tx.try_send(frame).is_err() {
            log_conn_exit(peer, ConnExit::Violation("pipelined backlog bound"));
            return;
        }
    }
    let reader_shutdown = Rc::clone(&shared.shutdown);
    // Handshake first on the unified stream, then split: the reader task
    // owns framing from here on (never cancelled), the serving task owns
    // the write half plus the reply outbox. Both halves count live: the
    // shutdown drain waits for the serving task AND the reader, so the
    // runtime never drops under a live socket (a dropped runtime with
    // live tasks leaks their FDs past thread join, breaking same-port
    // restart with ghost listeners).
    let reader_guard = ActiveGuard::hold(&shared.active);
    compio::runtime::spawn(async move {
        let _guard = reader_guard;
        conn_read_loop(
            read_stream,
            parser,
            max_input,
            max_pipelined,
            reader_shutdown,
            frames_tx,
            peer,
        )
        .await;
    })
    .detach();
    let exit = conn.run().await;
    log_conn_exit(peer, exit);
}

/// Logs one connection outcome at its level.
fn log_conn_exit(peer: SocketAddr, exit: ConnExit) {
    match exit {
        ConnExit::Clean => tracing::debug!(%peer, "connection closed"),
        ConnExit::Violation(reason) => {
            tracing::warn!(%peer, reason, "protocol violation; connection closed");
        }
        ConnExit::Io => tracing::debug!(%peer, "connection I/O ended"),
    }
}

/// Per-connection reader task: owns the read half and framing, forwards
/// decoded frames over a bounded channel. Reads are never wrapped in an
/// outer timeout, so a racing wakeup can never cancel an in-flight read
/// and lose pipelined bytes. Backpressure is explicit: a full channel
/// (server task behind) closes the connection instead of queueing
/// without limit. Exits on EOF, violation, I/O failure, shutdown, or a
/// dead server task.
async fn conn_read_loop(
    mut stream: TcpStream,
    mut reader: FrameReader,
    max_input: usize,
    max_pipelined: usize,
    shutdown: Rc<Cell<bool>>,
    tx: async_channel::Sender<kivi_protocol::Frame>,
    peer: SocketAddr,
) {
    let mut pending = std::collections::VecDeque::new();
    let exit = loop {
        if shutdown.get() {
            break ConnExit::Clean;
        }
        if let Some(frame) = pending.pop_front() {
            if tx.try_send(frame).is_err() {
                break ConnExit::Violation("pipelined backlog bound");
            }
            continue;
        }
        // Coherent bound (see the handshake loop): unparsed junk is
        // capped at max_input while a declared in-progress frame may
        // buffer exactly its declared length up to max_frame.
        if reader.buffered() >= reader.buffer_ceiling(max_input) {
            break ConnExit::Violation("input buffer bound");
        }
        let chunk = Vec::with_capacity(8192);
        let outcome = compio::time::timeout(ACCEPT_POLL_INTERVAL, stream.read(chunk)).await;
        let (result, chunk) = match outcome {
            Err(_) => {
                if shutdown.get() {
                    break ConnExit::Clean;
                }
                continue;
            }
            Ok(outcome) => (outcome.0, outcome.1),
        };
        match result {
            Ok(0) => break ConnExit::Clean,
            Ok(_) => match reader.push(&chunk) {
                Ok(frames) => {
                    if pending.len() + frames.len() > max_pipelined {
                        break ConnExit::Violation("pipelined backlog bound");
                    }
                    pending.extend(frames);
                }
                Err(_) => break ConnExit::Violation("frame parse"),
            },
            Err(_) => break ConnExit::Io,
        }
    };
    log_conn_exit(peer, exit);
}
