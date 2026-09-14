//! Persistent peer connections on the Compio mesh.
//!
//! One long-lived TCP connection per node pair carries the critical
//! consensus path (election, pre-vote, append, snapshot fragments). The
//! transport owns dialing, handshakes, reconnect with bounded backoff,
//! request/response matching, per-peer queue bounds, and connection stats
//! — `OpenRaft` never touches a socket directly.
//!
//! ```text
//! OpenRaft RPC → router codec → Kivi peer frame (group-multiplexed) → peer TCP
//! ```
//!
//! Every consensus body carries its addressed group, so one mesh serves
//! every Raft group on the node. All tasks run on the owner's Compio
//! reactor (no Tokio anywhere in this crate); cross-thread callers
//! (`call`, `stats`, `suspend_peer`) interact through the `Send + Sync`
//! front only.
//!
//! ## Bounds (no unbounded state)
//!
//! Per peer: queued messages ([`TransportConfig::queue_depth`]), queued
//! bytes ([`TransportConfig::queue_bytes`]), one active connection,
//! backoff capped at [`TransportConfig::backoff_max`]. A full queue fails
//! the send with [`TransportError::QueueFull`] (the caller backs off;
//! nothing blocks, nothing grows).
//!
//! ## Failure discipline
//!
//! Connection failure never crashes the node: the affected link backs off
//! and redials while the rest of the mesh keeps serving. Frames dropped on
//! disconnect are safe to lose: every consensus RPC is idempotent
//! (heartbeats, votes, offset-keyed snapshot fragments) and `OpenRaft`
//! retries on timeout, so a dead peer surfaces as
//! [`TransportError::Unreachable`] or [`TransportError::Timeout`], never a
//! hang and never a phantom success.
//!
//! ## Simultaneous open
//!
//! Both sides dial each other, so two sockets per pair can handshake at
//! once. The tie-break keeps exactly one: the connection initiated by the
//! higher [`NodeId`] wins (deterministic on both sides, no negotiation
//! round trip). A newer generation from the same initiator supersedes its
//! predecessor, which evicts stale-incarnation sockets the moment a
//! restarted peer redials. Connection validity is socket-scoped: once a
//! generation is evicted its reader stops dispatching, so messages from a
//! previous incarnation can never regain validity after supersede.

use std::collections::HashMap;
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
use std::time::Duration;

use compio::io::{AsyncRead as _, AsyncWriteExt as _};
use compio::net::{TcpListener, TcpStream};
use kivi_types::{NodeId, NodeIncarnation};

use crate::peer::{
    FrameDecoder, HandshakeFrame, HandshakeHello, IncarnationTable, PEER_MAX_FRAME_BYTES,
    PEER_PROTOCOL_VERSION, PeerBody, PeerIdentity, PeerRequest, PeerResponse, PeerRpcError,
    REQUIRED_CAPABILITIES, decode_body, decode_handshake, encode_accept, encode_hello,
    encode_reject, encode_request, encode_response, frame_body, validate_hello,
};
use crate::types::ConsensusGroupId;

/// Bounded per-peer outbound queue: message count plus byte sum. Both
/// bounds fail closed with [`TransportError::QueueFull`]. Reservations
/// are released by the writer after each flush or drop, so accounting
/// never leaks across disconnects.
#[derive(Debug)]
struct OutboundQueue {
    bytes: usize,
    queued_bytes: usize,
    sender: async_channel::Sender<Vec<u8>>,
}

impl OutboundQueue {
    /// `sender` is bounded at `queue_depth` by the constructor site; both
    /// this channel bound and `bytes` fail closed with `QueueFull`.
    fn new(bytes: usize, sender: async_channel::Sender<Vec<u8>>) -> Self {
        Self {
            bytes: bytes.max(1024),
            queued_bytes: 0,
            sender,
        }
    }

    /// Admits one framed message without blocking. Both bounds are
    /// checked before the channel send, so a full queue never parks the
    /// caller.
    fn try_send(&mut self, frame: Vec<u8>) -> Result<(), TransportError> {
        if self.sender.is_full() || self.queued_bytes + frame.len() > self.bytes {
            return Err(TransportError::QueueFull);
        }
        self.queued_bytes += frame.len();
        self.sender
            .try_send(frame)
            .map_err(|_| TransportError::QueueFull)?;
        Ok(())
    }

    /// Releases one message's byte reservation after the writer flushes
    /// it or drops it on disconnect.
    fn completed(&mut self, len: usize) {
        self.queued_bytes = self.queued_bytes.saturating_sub(len);
    }

    /// Current reservations (diagnostics).
    fn reserved(&self) -> (usize, usize) {
        (self.sender.len(), self.queued_bytes)
    }
}

/// Transport tuning: every bound explicit, every retry capped.
#[derive(Debug, Clone)]
pub struct TransportConfig {
    /// Queued messages per peer before sends fail.
    pub queue_depth: usize,
    /// Queued bytes per peer before sends fail.
    pub queue_bytes: usize,
    /// TCP connect timeout per attempt.
    pub dial_timeout: Duration,
    /// Handshake read timeout per attempt.
    pub handshake_timeout: Duration,
    /// Default RPC timeout (the router overrides per call with
    /// `OpenRaft`'s TTL).
    pub rpc_timeout: Duration,
    /// Read buffer chunk per socket read.
    pub read_chunk: usize,
    /// First reconnect delay.
    pub backoff_min: Duration,
    /// Reconnect delay ceiling.
    pub backoff_max: Duration,
    /// Open paused: handshake and dial loops wait for [`PeerTransport::start`].
    /// Static bootstrap needs this — every node installs the same joint
    /// membership while the mesh is quiet, so no node can grant a vote
    /// before its own `initialize` runs (which `OpenRaft` would refuse as
    /// `NotAllowed`). Production startup order: recover, classify,
    /// initialize-if-fresh, then start.
    pub start_paused: bool,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            queue_depth: 64,
            queue_bytes: 16 * 1024 * 1024,
            dial_timeout: Duration::from_secs(2),
            handshake_timeout: Duration::from_secs(5),
            rpc_timeout: Duration::from_secs(10),
            read_chunk: 64 * 1024,
            backoff_min: Duration::from_millis(100),
            backoff_max: Duration::from_secs(5),
            start_paused: false,
        }
    }
}

/// Why a peer send or call failed. Transport failures are retryable
/// signals, never process-fatal.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TransportError {
    /// The peer's bounded queue is full (depth or bytes).
    #[error("peer queue full")]
    QueueFull,
    /// No such peer is configured.
    #[error("unknown peer {node}")]
    UnknownPeer {
        /// The addressed node.
        node: u64,
    },
    /// The peer is unreachable (refused, reset, handshake rejected).
    #[error("peer unreachable: {detail}")]
    Unreachable {
        /// Human-readable cause.
        detail: String,
    },
    /// No response arrived before the deadline.
    #[error("peer RPC timed out")]
    Timeout,
    /// The transport is shutting down.
    #[error("peer transport shut down")]
    ShuttingDown,
    /// The remote Raft layer refused the RPC.
    #[error("remote peer refused RPC: {detail}")]
    RemoteRefused {
        /// Remote cause.
        detail: String,
    },
}

/// Per-peer connection diagnostics for the admin plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerStats {
    /// Whether a validated connection is currently active.
    pub connected: bool,
    /// Whether the link is administratively suspended (partition).
    pub suspended: bool,
    /// Incarnation of the active connection, if any.
    pub incarnation: Option<u64>,
    /// Queued outbound messages.
    pub queue_depth: usize,
    /// Queued outbound bytes.
    pub queue_bytes: usize,
    /// Reconnect attempts so far (dial attempts plus accepted inbound).
    pub reconnects: u64,
    /// Framed bytes written on all connections to this peer.
    pub bytes_sent: u64,
    /// Framed bytes read on all connections from this peer.
    pub bytes_received: u64,
}

/// Serves incoming consensus RPCs for one group. The transport only
/// routes bytes; the returned response must match the request family
/// (vote answers vote); mismatches trip a debug assertion and fail the
/// caller rather than misframing the stream.
///
/// The future stays `Send`: implementations forward to the owner thread
/// over channels (the `!Send` Raft handle never leaves its reactor).
pub trait PeerHandler: Send + Sync + 'static {
    /// Serves one request from `from` for `group`.
    fn handle(
        &self,
        from: NodeId,
        group: ConsensusGroupId,
        request: PeerRequest,
    ) -> Pin<Box<dyn Future<Output = Result<PeerResponse, PeerRpcError>> + Send + '_>>;
}

#[derive(Debug)]
struct ActiveConn {
    generation: u64,
    initiator: NodeId,
}

#[derive(Debug)]
struct PeerLink {
    addr: SocketAddr,
    queue_rx: futures::lock::Mutex<Option<async_channel::Receiver<Vec<u8>>>>,
    queue_state: futures::lock::Mutex<OutboundQueue>,
    slot: futures::lock::Mutex<Option<ActiveConn>>,
    incarnation: futures::lock::Mutex<Option<u64>>,
    connected: AtomicBool,
    /// Live connection tasks (reader+writer pairs) on this link.
    /// `suspend_peer` waits for zero, so returns mean no task can
    /// still send or receive on behalf of a dropped generation —
    /// a cleared slot alone would still leave a dying writer holding
    /// an open socket for up to one read quantum.
    active: AtomicUsize,
    /// Operational partition switch (partition tests, drain tooling):
    /// suspended links neither dial nor accept; queued RPCs stay bounded
    /// and fail by timeout while suspended.
    suspended: AtomicBool,
    generation: AtomicU64,
    reconnects: AtomicU64,
    bytes_sent: AtomicU64,
    bytes_received: AtomicU64,
}

struct TransportInner {
    local: PeerIdentity,
    config: TransportConfig,
    links: HashMap<NodeId, Arc<PeerLink>>,
    incarnations: futures::lock::Mutex<IncarnationTable>,
    pending: futures::lock::Mutex<
        HashMap<u64, futures::channel::oneshot::Sender<Result<PeerResponse, TransportError>>>,
    >,
    next_id: AtomicU64,
    running: AtomicBool,
    ready: AtomicBool,
    handler: Arc<dyn PeerHandler>,
}

/// Persistent peer mesh over Compio TCP. Cloneable (and `Send + Sync`):
/// the accept loop, dial loops, and connection tasks share one inner.
/// Connection tasks themselves run on the owner's Compio reactor; the
/// front's async methods are callable from any thread.
#[derive(Clone)]
pub struct PeerTransport {
    inner: Arc<TransportInner>,
}

impl std::fmt::Debug for PeerTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeerTransport")
            .field("local", &self.inner.local)
            .field("peers", &self.inner.links.len())
            .finish_non_exhaustive()
    }
}

impl PeerTransport {
    /// Opens the mesh: binds the peer listener, spawns the accept loop
    /// plus one dial loop per configured peer, and returns the transport
    /// with its bound address. The `peers` map holds every node
    /// (including the local entry, whose address is the bind address —
    /// port `0` selects an ephemeral port); only non-local entries are
    /// dialed.
    ///
    /// # Errors
    ///
    /// Returns [`TransportError::Unreachable`] when the listener cannot
    /// bind. Dial failures never fail `open` (links back off and retry).
    ///
    /// # Panics
    ///
    /// Panics when the loopback fallback bind address fails to parse
    /// (a literal, so only if the literal itself is edited into garbage).
    pub async fn open(
        config: TransportConfig,
        local: PeerIdentity,
        peers: &HashMap<NodeId, SocketAddr>,
        handler: Arc<dyn PeerHandler>,
    ) -> Result<(Self, SocketAddr), TransportError> {
        let listen_addr = peers
            .get(&local.node)
            .copied()
            .unwrap_or_else(|| "127.0.0.1:0".parse().expect("loopback bind address parses"));
        let listener =
            TcpListener::bind(listen_addr)
                .await
                .map_err(|error| TransportError::Unreachable {
                    detail: format!("peer listener bind failed: {error}"),
                })?;
        Self::open_with_listener(config, local, peers, handler, listener)
    }

    /// Opens the mesh on an already-bound listener (race-free port
    /// discovery for tests: bind first, read the address, then open every
    /// node with the complete map). Synchronous: it only spawns tasks, so
    /// callers must already run on a Compio reactor (the consensus owner
    /// thread in production, the test reactor in tests).
    ///
    /// # Errors
    ///
    /// Returns [`TransportError::Unreachable`] when the bound address is
    /// unreadable.
    pub fn open_with_listener(
        config: TransportConfig,
        local: PeerIdentity,
        peers: &HashMap<NodeId, SocketAddr>,
        handler: Arc<dyn PeerHandler>,
        listener: TcpListener,
    ) -> Result<(Self, SocketAddr), TransportError> {
        let bound = listener
            .local_addr()
            .map_err(|error| TransportError::Unreachable {
                detail: format!("peer listener address unreadable: {error}"),
            })?;
        let mut links = HashMap::new();
        for (node, addr) in peers {
            if *node == local.node {
                continue;
            }
            let (queue_tx, queue_rx) = async_channel::bounded(config.queue_depth.max(1));
            links.insert(
                *node,
                Arc::new(PeerLink {
                    addr: *addr,
                    queue_rx: futures::lock::Mutex::new(Some(queue_rx)),
                    queue_state: futures::lock::Mutex::new(OutboundQueue::new(
                        config.queue_bytes,
                        queue_tx,
                    )),
                    slot: futures::lock::Mutex::new(None),
                    incarnation: futures::lock::Mutex::new(None),
                    connected: AtomicBool::new(false),
                    suspended: AtomicBool::new(false),
                    active: AtomicUsize::new(0),
                    generation: AtomicU64::new(0),
                    reconnects: AtomicU64::new(0),
                    bytes_sent: AtomicU64::new(0),
                    bytes_received: AtomicU64::new(0),
                }),
            );
        }
        let paused = config.start_paused;
        let transport = Self {
            inner: Arc::new(TransportInner {
                local,
                config,
                links,
                incarnations: futures::lock::Mutex::new(IncarnationTable::new()),
                pending: futures::lock::Mutex::new(HashMap::new()),
                next_id: AtomicU64::new(1),
                running: AtomicBool::new(true),
                ready: AtomicBool::new(!paused),
                handler,
            }),
        };
        {
            let transport = transport.clone();
            // NOTE: compio cancels a task when its `JoinHandle` drops
            // (unlike Tokio's detach-on-drop), so every fire-and-forget
            // task must `detach()` explicitly.
            compio::runtime::spawn(async move { transport.serve(listener).await }).detach();
        }
        for (node, link) in transport.inner.links.clone() {
            let transport = transport.clone();
            compio::runtime::spawn(async move { transport.dial_loop(node, link).await }).detach();
        }
        Ok((transport, bound))
    }

    /// Calls one peer RPC for one group: registers the response slot,
    /// enqueues the framed request on the peer's bounded queue, and awaits
    /// the answer or the deadline. Queue-full and timeout fail without
    /// touching the connection (other peers and later retries are
    /// unaffected). The returned future is `Send`: it holds only the
    /// shared front plus channel ends.
    ///
    /// # Errors
    ///
    /// Returns [`TransportError`] on unknown peers, full queues,
    /// disconnects, remote refusals, timeouts, or shutdown.
    pub async fn call(
        &self,
        target: NodeId,
        group: ConsensusGroupId,
        request: PeerRequest,
        timeout: Duration,
    ) -> Result<PeerResponse, TransportError> {
        let link = self
            .inner
            .links
            .get(&target)
            .ok_or(TransportError::UnknownPeer {
                node: target.as_u64(),
            })?;
        if !self.inner.running.load(Ordering::SeqCst) {
            return Err(TransportError::ShuttingDown);
        }
        let id = self.inner.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = futures::channel::oneshot::channel();
        {
            let mut pending = self.inner.pending.lock().await;
            pending.insert(id, tx);
        }
        let frame = frame_body(&encode_request(id, group.tablet().as_u64(), &request));
        {
            let mut queue = link.queue_state.lock().await;
            if let Err(error) = queue.try_send(frame) {
                self.inner.pending.lock().await.remove(&id);
                return Err(error);
            }
        }
        match compio::time::timeout(timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(TransportError::ShuttingDown),
            Err(_) => {
                self.inner.pending.lock().await.remove(&id);
                Err(TransportError::Timeout)
            }
        }
    }

    /// Snapshots per-peer diagnostics (admin plane; lock-free counters,
    /// brief mutexes for identity state).
    pub async fn stats(&self) -> HashMap<NodeId, PeerStats> {
        let mut out = HashMap::new();
        for (node, link) in &self.inner.links {
            let (depth, bytes) = link.queue_state.lock().await.reserved();
            out.insert(
                *node,
                PeerStats {
                    connected: link.connected.load(Ordering::SeqCst),
                    suspended: link.suspended.load(Ordering::SeqCst),
                    incarnation: *link.incarnation.lock().await,
                    queue_depth: depth,
                    queue_bytes: bytes,
                    reconnects: link.reconnects.load(Ordering::SeqCst),
                    bytes_sent: link.bytes_sent.load(Ordering::SeqCst),
                    bytes_received: link.bytes_received.load(Ordering::SeqCst),
                },
            );
        }
        out
    }

    /// Releases paused handshakes and dial loops (static bootstrap calls
    /// this after every node installed the joint membership). Idempotent:
    /// already-started transports ignore it.
    pub fn start(&self) {
        self.inner.ready.store(true, Ordering::SeqCst);
    }

    /// Suspends one peer link (partition test hook, drain tooling):
    /// drops the active connection and pauses dials and accepts until
    /// [`PeerTransport::resume_peer`]. Queued RPCs stay bounded and fail
    /// by timeout; nothing blocks, nothing grows.
    pub async fn suspend_peer(&self, peer: NodeId) {
        let Some(link) = self.inner.links.get(&peer) else {
            return;
        };
        link.suspended.store(true, Ordering::SeqCst);
        self.fail_all_generations(peer).await;
        // Wait for the evicted tasks to actually exit (bounded): only
        // then are their sockets closed and the partition airtight for
        // new proposals. A cleared slot alone still leaves dying tasks
        // holding open sockets for up to one read quantum.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while link.active.load(Ordering::SeqCst) > 0 {
            if std::time::Instant::now() >= deadline {
                break;
            }
            compio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Resumes a suspended peer link: drops any stale generation so the
    /// link revalidates from scratch, then dial loops redial and inbound
    /// handshakes validate again (incarnation discipline unchanged).
    pub async fn resume_peer(&self, peer: NodeId) {
        if let Some(link) = self.inner.links.get(&peer) {
            link.suspended.store(false, Ordering::SeqCst);
            self.fail_all_generations(peer).await;
        }
    }

    /// Stops dial loops, closes connections, and fails pending RPCs.
    pub async fn shutdown(&self) {
        self.inner.running.store(false, Ordering::SeqCst);
        let mut pending = self.inner.pending.lock().await;
        for (_, tx) in pending.drain() {
            let _ = tx.send(Err(TransportError::ShuttingDown));
        }
    }

    /// Returns the local peer identity.
    #[must_use]
    pub fn local(&self) -> PeerIdentity {
        self.inner.local
    }

    async fn serve(&self, listener: TcpListener) {
        loop {
            if !self.inner.running.load(Ordering::SeqCst) {
                break;
            }
            let accepted =
                compio::time::timeout(Duration::from_millis(200), listener.accept()).await;
            let Ok(Ok((socket, _))) = accepted else {
                continue;
            };
            let transport = self.clone();
            compio::runtime::spawn(async move {
                transport.serve_inbound(socket).await;
            })
            .detach();
        }
    }

    /// Waits for [`PeerTransport::start`] (or shutdown), so paused
    /// transports hold connections without handshaking.
    async fn wait_ready(&self) -> bool {
        loop {
            if self.inner.ready.load(Ordering::SeqCst) {
                return true;
            }
            if !self.inner.running.load(Ordering::SeqCst) {
                return false;
            }
            compio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    async fn serve_inbound(&self, mut socket: TcpStream) {
        if !self.wait_ready().await {
            return;
        }
        // Hello read failure (timeout, EOF, corrupt): drop the socket;
        // the dialer backs off and retries.
        let Ok(hello) = read_hello(&mut socket, self.inner.config.handshake_timeout).await else {
            return;
        };
        let node = NodeId::from_u64(hello.node);
        let incarnation = NodeIncarnation::from_u64(hello.incarnation);
        let Some(link) = self.inner.links.get(&node) else {
            return;
        };
        if link.suspended.load(Ordering::SeqCst) {
            // Partitioned: close without answering (the dialer backs off;
            // resume revalidates the handshake from scratch).
            return;
        }
        let decision = {
            let table = self.inner.incarnations.lock().await;
            validate_hello(&hello, &self.inner.local, &table)
        };
        match decision {
            Ok(accept) => {
                if write_frame(&mut socket, &encode_accept(&accept))
                    .await
                    .is_err()
                {
                    return;
                }
                // Record only validated generations.
                {
                    let mut table = self.inner.incarnations.lock().await;
                    table.observe(node, incarnation);
                }
                if let Some(link) = self.inner.links.get(&node) {
                    link.reconnects.fetch_add(1, Ordering::SeqCst);
                }
                // The initiator is the dialer (the remote node here), so
                // both sides attribute this socket identically and the
                // higher-initiator tie-break converges on one socket.
                self.attach(node, incarnation, socket, node).await;
            }
            Err(reject) => {
                let _ = write_frame(&mut socket, &encode_reject(&reject)).await;
            }
        }
    }

    async fn dial_loop(&self, node: NodeId, link: Arc<PeerLink>) {
        let mut backoff = self.inner.config.backoff_min;
        loop {
            if !self.inner.running.load(Ordering::SeqCst) {
                break;
            }
            if !self.wait_ready().await {
                break;
            }
            // Suspended links stay down until resumed (partition).
            if link.suspended.load(Ordering::SeqCst) {
                compio::time::sleep(Duration::from_millis(200)).await;
                continue;
            }
            // A healthy link needs no dial.
            if link.connected.load(Ordering::SeqCst) {
                compio::time::sleep(Duration::from_millis(200)).await;
                continue;
            }
            match compio::time::timeout(
                self.inner.config.dial_timeout,
                TcpStream::connect(link.addr),
            )
            .await
            {
                Ok(Ok(socket)) => {
                    link.reconnects.fetch_add(1, Ordering::SeqCst);
                    if self.dial_handshake(node, socket).await.is_ok() {
                        backoff = self.inner.config.backoff_min;
                    } else {
                        sleep_capped(&mut backoff, self.inner.config.backoff_max).await;
                    }
                }
                _ => {
                    sleep_capped(&mut backoff, self.inner.config.backoff_max).await;
                }
            }
        }
    }

    async fn dial_handshake(&self, node: NodeId, mut socket: TcpStream) -> Result<(), ()> {
        let hello = HandshakeHello {
            version: PEER_PROTOCOL_VERSION,
            cluster: self.inner.local.cluster.as_u128(),
            node: self.inner.local.node.as_u64(),
            incarnation: self.inner.local.incarnation.as_u64(),
            capabilities: REQUIRED_CAPABILITIES,
        };
        if write_frame(&mut socket, &encode_hello(&hello))
            .await
            .is_err()
        {
            return Err(());
        }
        let Ok(body) = read_frame(&mut socket, self.inner.config.handshake_timeout).await else {
            return Err(());
        };
        let Ok(HandshakeFrame::Accept(accept)) = decode_handshake(&body) else {
            return Err(());
        };
        if accept.cluster != self.inner.local.cluster.as_u128()
            || accept.capabilities & REQUIRED_CAPABILITIES != REQUIRED_CAPABILITIES
        {
            return Err(());
        }
        let incarnation = NodeIncarnation::from_u64(accept.incarnation);
        {
            let mut table = self.inner.incarnations.lock().await;
            // The accept completes our own dial: its generation must not
            // be OLDER than observed (a restarted-away generation), but
            // EQUAL is the normal concurrent-handshake echo — the inbound
            // hello from this same generation is routinely recorded
            // first, and strict-greater here would reject every
            // simultaneous open. `INVALID` is never valid anywhere.
            let stale = incarnation == NodeIncarnation::INVALID
                || table
                    .observed(node)
                    .is_some_and(|known| incarnation.as_u64() < known.as_u64());
            if stale {
                return Err(());
            }
            table.observe(node, incarnation);
        }
        self.attach(node, incarnation, socket, self.inner.local.node)
            .await;
        Ok(())
    }

    /// Attaches a validated socket to its peer link, evicting any
    /// superseded connection per the higher-initiator tie-break, then
    /// pumps the reader and writer until the socket dies or is evicted.
    async fn attach(
        &self,
        node: NodeId,
        incarnation: NodeIncarnation,
        socket: TcpStream,
        initiator: NodeId,
    ) {
        let Some(link) = self.inner.links.get(&node) else {
            return;
        };
        // Choke point for handshakes already in flight when a suspend
        // lands: a suspended link never activates, so no racing
        // handshake can resurrect a partition after the suspend closed
        // it (the dial and accept paths check earlier, but only this
        // check is atomic with slot installation).
        if link.suspended.load(Ordering::SeqCst) {
            return;
        }
        let generation = link.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let preferred = {
            let mut slot = link.slot.lock().await;
            let keep_new = match slot.as_ref() {
                None => true,
                Some(active) => {
                    initiator.as_u64() > active.initiator.as_u64()
                        || (initiator == active.initiator && generation > active.generation)
                }
            };
            if keep_new {
                *slot = Some(ActiveConn {
                    generation,
                    initiator,
                });
            }
            keep_new
        };
        if !preferred {
            return;
        }
        link.active.fetch_add(1, Ordering::SeqCst);
        link.connected.store(true, Ordering::SeqCst);
        *link.incarnation.lock().await = Some(incarnation.as_u64());
        let (reader, writer) = socket.into_split();
        // The evicted predecessor returns the queue receiver on exit
        // (bounded wait: eviction trips its generation check within one
        // read quantum). Taking it without waiting would fail our own
        // generation on a benign handoff race.
        let receiver = {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            loop {
                if let Some(receiver) = link.queue_rx.lock().await.take() {
                    break Some(receiver);
                }
                if std::time::Instant::now() >= deadline
                    || !self.inner.running.load(Ordering::SeqCst)
                    || link.suspended.load(Ordering::SeqCst)
                {
                    break None;
                }
                compio::time::sleep(Duration::from_millis(5)).await;
            }
        };
        let Some(receiver) = receiver else {
            link.active.fetch_sub(1, Ordering::SeqCst);
            self.fail_generation(node, generation).await;
            return;
        };
        // A suspend that landed while waiting for the receiver must
        // still stand down (the pre-wait choke is stale by now).
        if link.suspended.load(Ordering::SeqCst) {
            *link.queue_rx.lock().await = Some(receiver);
            link.active.fetch_sub(1, Ordering::SeqCst);
            return;
        }
        let transport = self.clone();
        let read_handle = compio::runtime::spawn(async move {
            transport.read_loop(node, generation, reader).await;
        });
        let transport = self.clone();
        let write_handle = compio::runtime::spawn(async move {
            transport
                .write_loop(node, generation, writer, Some(receiver))
                .await;
        });
        let _ = futures::join!(read_handle, write_handle);
        // This generation ended: clear the slot only if it is still ours
        // (a superseding generation keeps serving).
        {
            let mut slot = link.slot.lock().await;
            if slot
                .as_ref()
                .is_some_and(|active| active.generation == generation)
            {
                *slot = None;
            }
            link.connected.store(slot.is_some(), Ordering::SeqCst);
            if slot.is_none() {
                *link.incarnation.lock().await = None;
            }
        }
        link.active.fetch_sub(1, Ordering::SeqCst);
    }

    async fn read_loop(&self, node: NodeId, generation: u64, mut reader: TcpStream) {
        let Some(link) = self.inner.links.get(&node) else {
            return;
        };
        let mut decoder = FrameDecoder::new();
        loop {
            if !self.is_current(node, generation).await {
                break;
            }
            // Bounded read quantum (mirrors the writer's recv timeout):
            // eviction and shutdown are observed within one quantum even
            // when the peer sends nothing and never closes. Without this
            // the reader would park forever and pin a dead generation.
            let Ok(res) = compio::time::timeout(
                Duration::from_millis(200),
                reader.read(vec![0u8; self.inner.config.read_chunk]),
            )
            .await
            else {
                continue;
            };
            let count = match res.0 {
                // EOF or read error ends OUR side of the generation too:
                // without this the writer would idle forever on a
                // half-dead socket (its slot still current), pinning a
                // stale connection that rejects fresh handshakes.
                Ok(0) | Err(_) => {
                    self.fail_generation(node, generation).await;
                    break;
                }
                Ok(count) => count,
            };
            let buf = res.1;
            link.bytes_received
                .fetch_add(count as u64, Ordering::SeqCst);
            decoder.push(&buf[..count]);
            loop {
                match decoder.take_frame() {
                    Ok(Some(body)) => self.dispatch(node, body).await,
                    Ok(None) => break,
                    Err(_) => {
                        self.fail_generation(node, generation).await;
                        return;
                    }
                }
            }
            if decoder.buffered() > PEER_MAX_FRAME_BYTES {
                self.fail_generation(node, generation).await;
                return;
            }
        }
    }

    async fn dispatch(&self, from: NodeId, body: Vec<u8>) {
        match decode_body(&body) {
            Ok(PeerBody::Request { id, group, request }) => {
                let kind = match &request {
                    PeerRequest::Vote(_) => crate::peer::KIND_VOTE_REQUEST,
                    PeerRequest::PreVote(_) => crate::peer::KIND_PRE_VOTE_REQUEST,
                    PeerRequest::Append(_) => crate::peer::KIND_APPEND_REQUEST,
                    PeerRequest::Snapshot(_) => crate::peer::KIND_SNAPSHOT_REQUEST,
                };
                // Served on a dedicated task: application work (snapshot
                // install can take a while) must never head-of-line-block
                // the read loop behind it.
                let transport = self.clone();
                // Fire-and-forget service task: must `detach` (compio
                // cancels on handle drop).
                compio::runtime::spawn(async move {
                    let group = ConsensusGroupId::of_tablet(kivi_types::TabletId::from_u64(group));
                    let response = transport.inner.handler.handle(from, group, request).await;
                    debug_assert!(
                        response.as_ref().is_ok_and(|response| matches!(
                            (kind, response),
                            (crate::peer::KIND_VOTE_REQUEST, PeerResponse::Vote(_))
                                | (crate::peer::KIND_PRE_VOTE_REQUEST, PeerResponse::PreVote(_))
                                | (crate::peer::KIND_APPEND_REQUEST, PeerResponse::Append(_))
                                | (
                                    crate::peer::KIND_SNAPSHOT_REQUEST,
                                    PeerResponse::Snapshot(_)
                                )
                        )),
                        "peer handler must answer the requested family"
                    );
                    let frame = frame_body(&encode_response(
                        id,
                        group.tablet().as_u64(),
                        &response,
                        kind,
                    ));
                    transport.send_raw(from, frame).await;
                })
                .detach();
            }
            Ok(PeerBody::Response { id, result }) => {
                let sender = self.inner.pending.lock().await.remove(&id);
                if let Some(sender) = sender {
                    let outcome = result.map_err(|error| TransportError::RemoteRefused {
                        detail: error.detail,
                    });
                    let _ = sender.send(outcome);
                }
            }
            Err(_) => {
                // A malformed body on a validated connection means a
                // version-skewed or corrupt peer: the reader's connection
                // is failed by the caller observing repeated faults. (One
                // bad body alone only drops that body; the frame layer
                // already guaranteed integrity.)
            }
        }
    }

    async fn send_raw(&self, target: NodeId, frame: Vec<u8>) {
        let Some(link) = self.inner.links.get(&target) else {
            return;
        };
        let mut queue = link.queue_state.lock().await;
        let _ = queue.try_send(frame);
        // Responses never fail the node: a full queue drops the reply and
        // the caller times out and retries (bounded).
    }

    async fn write_loop(
        &self,
        node: NodeId,
        generation: u64,
        mut writer: TcpStream,
        receiver: Option<async_channel::Receiver<Vec<u8>>>,
    ) {
        let Some(receiver) = receiver else {
            self.fail_generation(node, generation).await;
            return;
        };
        let Some(link) = self.inner.links.get(&node) else {
            return;
        };
        loop {
            if !self.is_current(node, generation).await {
                break;
            }
            let frame = compio::time::timeout(Duration::from_millis(200), receiver.recv()).await;
            let frame = match frame {
                Ok(Ok(frame)) => frame,
                // Senders live as long as the link: `Err` (closed) is a
                // lifecycle bug that must end the writer loudly.
                Ok(Err(_)) => break,
                Err(_) => continue,
            };
            let len = frame.len();
            let res = writer.write_all(frame).await;
            if res.0.is_err() {
                // Dropped on disconnect (safe: consensus RPCs are
                // idempotent and retried on timeout).
                link.queue_state.lock().await.completed(len);
                self.fail_generation(node, generation).await;
                break;
            }
            link.bytes_sent.fetch_add(len as u64, Ordering::SeqCst);
            link.queue_state.lock().await.completed(len);
        }
        // Return the receiver so the next generation keeps the backlog.
        *link.queue_rx.lock().await = Some(receiver);
    }

    async fn is_current(&self, node: NodeId, generation: u64) -> bool {
        let Some(link) = self.inner.links.get(&node) else {
            return false;
        };
        link.slot
            .lock()
            .await
            .as_ref()
            .is_some_and(|active| active.generation == generation)
    }

    async fn fail_generation(&self, node: NodeId, generation: u64) {
        if let Some(link) = self.inner.links.get(&node) {
            let mut slot = link.slot.lock().await;
            if slot
                .as_ref()
                .is_some_and(|active| active.generation == generation)
            {
                *slot = None;
            }
            link.connected.store(slot.is_some(), Ordering::SeqCst);
        }
    }

    /// Drops every generation on a link (suspend path): readers and
    /// writers observe the cleared slot and exit, closing their sockets.
    async fn fail_all_generations(&self, node: NodeId) {
        if let Some(link) = self.inner.links.get(&node) {
            link.slot.lock().await.take();
            link.connected.store(false, Ordering::SeqCst);
            *link.incarnation.lock().await = None;
        }
    }
}

async fn sleep_capped(backoff: &mut Duration, max: Duration) {
    compio::time::sleep(*backoff).await;
    // Doubling stays far below overflow (delays are capped at `max`,
    // seconds-scale) and the ceiling keeps the bound exact.
    *backoff = (*backoff * 2).min(max).max(Duration::from_millis(1));
}

async fn read_exact_timeout(
    reader: &mut TcpStream,
    count: usize,
    timeout: Duration,
) -> Result<Vec<u8>, ()> {
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let want = count - out.len();
        let res = compio::time::timeout(timeout, reader.read(vec![0u8; want]))
            .await
            .map_err(|_| ())?;
        match res.0 {
            Ok(0) | Err(_) => return Err(()),
            Ok(n) => {
                let mut buf = res.1;
                buf.truncate(n);
                out.append(&mut buf);
            }
        }
    }
    Ok(out)
}

async fn read_frame(socket: &mut TcpStream, timeout: Duration) -> Result<Vec<u8>, ()> {
    let mut decoder = FrameDecoder::new();
    loop {
        // Handshake frames are small; byte-at-a-time reads keep the
        // decoder exact without over-reading into post-handshake bytes.
        let chunk = read_exact_timeout(socket, 1, timeout).await?;
        decoder.push(&chunk);
        match decoder.take_frame() {
            Ok(Some(body)) => return Ok(body),
            Ok(None) => {}
            Err(_) => return Err(()),
        }
    }
}

async fn read_hello(socket: &mut TcpStream, timeout: Duration) -> Result<HandshakeHello, ()> {
    let body = read_frame(socket, timeout).await?;
    match decode_handshake(&body) {
        Ok(HandshakeFrame::Hello(hello)) => Ok(hello),
        _ => Err(()),
    }
}

async fn write_frame(socket: &mut TcpStream, body: &[u8]) -> Result<(), ()> {
    let res = socket.write_all(frame_body(body)).await;
    res.0.map_err(|_| ())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    use kivi_types::{ClusterId, NodeId, NodeIncarnation};

    use super::{PeerHandler, PeerTransport, TransportConfig, TransportError};
    use crate::peer::{PeerIdentity, PeerRequest, PeerResponse, PeerRpcError, PeerVoteRequest};
    use crate::types::ConsensusGroupId;

    const CLUSTER: u128 = 0x0C10_57E2;

    fn identity(node: u64, incarnation: u64) -> PeerIdentity {
        PeerIdentity {
            cluster: ClusterId::from_u128(CLUSTER),
            node: NodeId::from_u64(node),
            incarnation: NodeIncarnation::from_u64(incarnation),
        }
    }

    /// Test handler: echoes votes with a fixed vote, stalls all other
    /// families (timeout-path coverage).
    struct Echo {
        vote: crate::peer::PeerVote,
        stall: bool,
    }

    impl PeerHandler for Echo {
        fn handle(
            &self,
            _from: NodeId,
            _group: ConsensusGroupId,
            request: PeerRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<PeerResponse, PeerRpcError>> + Send + '_>,
        > {
            let vote = self.vote;
            let stall = self.stall;
            Box::pin(async move {
                match request {
                    PeerRequest::Vote(request) => {
                        Ok(PeerResponse::Vote(crate::peer::PeerVoteResponse {
                            vote,
                            granted: true,
                            last_log: request.last_log,
                        }))
                    }
                    _ if stall => std::future::pending().await,
                    _ => Err(PeerRpcError {
                        detail: "echo serves votes only".to_owned(),
                    }),
                }
            })
        }
    }

    fn echo(vote: crate::peer::PeerVote) -> Arc<dyn PeerHandler> {
        Arc::new(Echo { vote, stall: false })
    }

    fn block_on<F, T>(future: F) -> T
    where
        F: Future<Output = T>,
    {
        compio::runtime::Runtime::new()
            .expect("compio test runtime")
            .block_on(future)
    }

    /// Opens two loopback transports with race-free port discovery (bind
    /// first, then open with the complete map). Returns both transports
    /// with node 1's and node 2's listener addresses.
    async fn pair() -> (
        PeerTransport,
        PeerTransport,
        std::net::SocketAddr,
        std::net::SocketAddr,
    ) {
        use compio::net::TcpListener;
        let vote = crate::peer::PeerVote {
            term: 1,
            node: 1,
            committed: true,
        };
        let listener_a = TcpListener::bind("127.0.0.1:0").await.expect("binds");
        let listener_b = TcpListener::bind("127.0.0.1:0").await.expect("binds");
        let addr_a = listener_a.local_addr().expect("addr");
        let addr_b = listener_b.local_addr().expect("addr");
        let peers: HashMap<NodeId, std::net::SocketAddr> =
            [(NodeId::from_u64(1), addr_a), (NodeId::from_u64(2), addr_b)]
                .into_iter()
                .collect();
        let (first, _) = PeerTransport::open_with_listener(
            TransportConfig::default(),
            identity(1, 1),
            &peers,
            echo(vote),
            listener_a,
        )
        .expect("first opens");
        let (second, _) = PeerTransport::open_with_listener(
            TransportConfig::default(),
            identity(2, 1),
            &peers,
            echo(vote),
            listener_b,
        )
        .expect("second opens");
        (first, second, addr_a, addr_b)
    }

    async fn wait_connected(transport: &PeerTransport, peer: NodeId) {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let stats = transport.stats().await;
            if stats.get(&peer).is_some_and(|stats| stats.connected) {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "peer {peer} never connected"
            );
            compio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    fn vote_request() -> PeerRequest {
        PeerRequest::Vote(PeerVoteRequest {
            vote: crate::peer::PeerVote {
                term: 1,
                node: 1,
                committed: false,
            },
            last_log: None,
            leadership_transfer: false,
        })
    }

    #[test]
    fn mesh_connects_and_serves_votes() {
        block_on(async {
            let (first, second, _addr_a, _addr_b) = pair().await;
            wait_connected(&first, NodeId::from_u64(2)).await;
            wait_connected(&second, NodeId::from_u64(1)).await;
            let group = ConsensusGroupId::of_tablet(kivi_types::TabletId::from_u64(9));
            let response = first
                .call(
                    NodeId::from_u64(2),
                    group,
                    vote_request(),
                    Duration::from_secs(5),
                )
                .await
                .expect("vote served");
            assert!(matches!(response, PeerResponse::Vote(_)));
            first.shutdown().await;
            second.shutdown().await;
        });
    }

    #[test]
    fn unknown_peer_fails_fast() {
        block_on(async {
            let (first, second, _addr_a, _addr_b) = pair().await;
            // Unknown peer fails without touching the mesh.
            let group = ConsensusGroupId::of_tablet(kivi_types::TabletId::from_u64(9));
            let error = first
                .call(
                    NodeId::from_u64(99),
                    group,
                    vote_request(),
                    Duration::from_secs(5),
                )
                .await
                .expect_err("unknown peer refuses");
            assert_eq!(
                error,
                TransportError::UnknownPeer { node: 99 },
                "unknown peer fails fast"
            );
            first.shutdown().await;
            second.shutdown().await;
        });
    }

    #[test]
    fn suspended_links_time_out_and_heal() {
        block_on(async {
            let (first, second, _addr_a, _addr_b) = pair().await;
            let group = ConsensusGroupId::of_tablet(kivi_types::TabletId::from_u64(9));
            wait_connected(&first, NodeId::from_u64(2)).await;
            first.suspend_peer(NodeId::from_u64(2)).await;
            let error = first
                .call(
                    NodeId::from_u64(2),
                    group,
                    vote_request(),
                    Duration::from_millis(300),
                )
                .await
                .expect_err("suspended link times out");
            assert_eq!(error, TransportError::Timeout);
            first.resume_peer(NodeId::from_u64(2)).await;
            wait_connected(&first, NodeId::from_u64(2)).await;
            let response = first
                .call(
                    NodeId::from_u64(2),
                    group,
                    vote_request(),
                    Duration::from_secs(5),
                )
                .await
                .expect("healed link serves");
            assert!(matches!(response, PeerResponse::Vote(_)));
            first.shutdown().await;
            second.shutdown().await;
        });
    }
}
