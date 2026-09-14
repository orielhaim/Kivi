//! QUIC/H3 peer mesh (replaces the former TCP/KVSP transport outright).
//!
//! One QUIC endpoint per node. Per node pair, **two** H3 connections:
//!
//! ```text
//! peer-control H3 connection — votes, heartbeats, append metadata,
//!                               snapshot fragments (small bodies)
//! peer-bulk H3 connection    — manifests, chunks (bulk bodies)
//! ```
//!
//! Two connections (not one, not a custom priority scheduler) because both
//! H3 streams on one QUIC connection still share connection-level
//! congestion and flow control. Separate connections give control traffic
//! an independent congestion/flow domain, so a saturated 64 MiB transfer
//! cannot starve heartbeats or force an election. QUIC already removes
//! TCP head-of-line blocking *across streams*; the split removes the
//! remaining *connection-level* coupling.
//!
//! H3 request/response streams are the RPC layer (§6):
//!
//! ```text
//! POST /_kivi/raft/{tablet}/vote        (body: vote request codec)
//! POST /_kivi/raft/{tablet}/prevote
//! POST /_kivi/raft/{tablet}/append
//! POST /_kivi/raft/{tablet}/snapshot-frag
//! GET  /_kivi/immutable/manifest/{hex}  (body: canonical manifest bytes)
//! GET  /_kivi/immutable/chunk/{hex}     (body: raw chunk bytes)
//! ```
//!
//! Bodies are compact Kivi-owned binary formats (never JSON). HTTP carries
//! only method, resource identity, status, and stream lifetime. One request
//! carries one payload — no second framing layer inside H3 bodies.
//!
//! ## Identity
//!
//! TLS (rustls via compio-quic, ALPN `h3`) encrypts and authenticates
//! endpoints against statically pinned peer certificates (trust anchors;
//! see [`crate::tls`]). Kivi identity (cluster, node, incarnation,
//! capabilities) rides authenticated request headers validated per
//! request; stale incarnations are refused. Socket tuples are never
//! identity. Only the higher [`NodeId`] of a pair dials (deterministic,
//! two connections per pair, no simultaneous-open tie-break); the lower
//! side serves. Suspended links drop connections, block dials, and refuse
//! inbound requests for the suspended peer.
//!
//! ## Runtime
//!
//! All QUIC/H3 I/O lives on the consensus owner thread's Compio reactor.
//! The public front stays `Send + Sync` (bounded channels + oneshots), so
//! callers on any runtime keep ordinary `Send` futures. `h3` pulls `tokio`
//! transitively for synchronization primitives only; no Tokio runtime or
//! Tokio network execution exists anywhere in this crate (`cargo tree`
//! shows `tokio` under `h3` alone).
//!
//! ## Bounds
//!
//! No unbounded state: per-request body caps (16 MiB Raft, 8 MiB bulk),
//! bounded concurrent streams per connection (QUIC transport tuning),
//! bounded dial backoff, per-RPC timeouts. Timeouts drop the H3 stream,
//! which resets it — the cancellation primitive for truncated suffixes,
//! superseded snapshots, and shutdown.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::Duration;

use compio::quic::{
    ClientBuilder, ClientConfig, Connection, Endpoint, IdleTimeout, ServerConfig,
    TransportConfig as QuicTransportConfig, VarInt,
};
use kivi_types::{ClusterId, NodeId};

use crate::peer::{
    IncarnationTable, PeerIdentity, PeerRequest, PeerResponse, PeerRpcError, REQUIRED_CAPABILITIES,
};
use crate::tls::NodeCert;
use crate::types::ConsensusGroupId;

/// Media type for Raft RPC bodies (Kivi-owned binary codecs).
const MEDIA_RAFT: &str = "application/vnd.kivi.raft";
/// Media type for manifest bodies (canonical manifest bytes).
const MEDIA_MANIFEST: &str = "application/vnd.kivi.manifest";
/// Media type for chunk bodies (raw logical chunk bytes).
const MEDIA_CHUNK: &str = "application/vnd.kivi.chunk";

/// Authenticated Kivi request headers (application identity over the
/// TLS-authenticated channel).
const HDR_CLUSTER: &str = "kivi-cluster";
/// Speaking node.
const HDR_NODE: &str = "kivi-node";
/// Speaking process generation.
const HDR_INCARNATION: &str = "kivi-incarnation";
/// Offered capability bits.
const HDR_CAPS: &str = "kivi-caps";

/// Maximum Raft RPC body in bytes (append batches stay small: the Raft log
/// carries tiny roots, never bulk).
pub const MAX_RAFT_BODY_BYTES: usize = 16 * 1024 * 1024;
/// Maximum sidecar body in bytes (chunk bodies never exceed the 8 MiB
/// stored-body cap; manifests peak near 2.6 MiB).
pub const MAX_BULK_BODY_BYTES: usize = 8 * 1024 * 1024;

/// Saturating `u64` → QUIC varint for window tuning.
fn varint_saturating(value: u64) -> VarInt {
    VarInt::from_u64(value).unwrap_or(VarInt::MAX)
}

/// Which H3 connection of a pair carries a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Lane {
    /// Votes, heartbeats, appends, snapshot fragments.
    Control,
    /// Manifests, chunks.
    Bulk,
}

/// Transport tuning: every bound explicit, every retry capped.
#[derive(Debug, Clone)]
pub struct TransportConfig {
    /// Concurrent inbound bidirectional H3 streams per connection.
    pub max_concurrent_bidi_streams: u64,
    /// Per-stream receive window in bytes.
    pub stream_receive_window: u64,
    /// Connection-level receive window in bytes.
    pub receive_window: u64,
    /// Send window in bytes.
    pub send_window: u64,
    /// Fair queuing across same-priority send streams.
    pub send_fairness: bool,
    /// Idle timeout before a quiet connection closes.
    pub idle_timeout: Duration,
    /// Keep-alive interval (`None` disables).
    pub keep_alive_interval: Option<Duration>,
    /// Default RPC timeout (the router overrides per call).
    pub rpc_timeout: Duration,
    /// Bulk fetch timeout (sidecar acquisition).
    pub bulk_timeout: Duration,
    /// First reconnect delay.
    pub backoff_min: Duration,
    /// Reconnect delay ceiling.
    pub backoff_max: Duration,
    /// Open paused: dial and serve loops wait for [`PeerTransport::start`].
    /// Static bootstrap needs this — every node installs the same joint
    /// membership while the mesh is quiet, so no node can grant a vote
    /// before its own `initialize` runs.
    pub start_paused: bool,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            max_concurrent_bidi_streams: 128,
            stream_receive_window: 2 * 1024 * 1024,
            receive_window: 32 * 1024 * 1024,
            send_window: 32 * 1024 * 1024,
            send_fairness: true,
            idle_timeout: Duration::from_secs(30),
            keep_alive_interval: Some(Duration::from_secs(10)),
            rpc_timeout: Duration::from_secs(10),
            bulk_timeout: Duration::from_secs(30),
            backoff_min: Duration::from_millis(100),
            backoff_max: Duration::from_secs(5),
            start_paused: false,
        }
    }
}

impl TransportConfig {
    /// Maps Kivi tuning onto the QUIC state machine. Defaults suit a
    /// 100 Mbps / 100 ms link; the per-stream window stays well under the
    /// connection window so one bulk stream cannot monopolize buffers
    /// while control streams need data.
    fn quic_transport(&self) -> QuicTransportConfig {
        let mut config = QuicTransportConfig::default();
        config.max_concurrent_bidi_streams(VarInt::from_u32(
            u32::try_from(self.max_concurrent_bidi_streams).unwrap_or(u32::MAX),
        ));
        // H3 needs remotely-initiated unidirectional streams (control,
        // QPACK): zero would deadlock the H3 handshake itself. Request
        // streams are bidirectional and bounded above; uni streams stay
        // small (a handful per connection in practice).
        config.max_concurrent_uni_streams(VarInt::from_u32(32));
        config.stream_receive_window(varint_saturating(self.stream_receive_window));
        config.receive_window(varint_saturating(self.receive_window));
        config.send_window(self.send_window);
        config.send_fairness(self.send_fairness);
        config.max_idle_timeout(Some(
            self.idle_timeout
                .try_into()
                .unwrap_or_else(|_| IdleTimeout::from(VarInt::from_u32(30_000))),
        ));
        config.keep_alive_interval(self.keep_alive_interval);
        config
    }
}

/// Why a peer send or call failed. Transport failures are retryable
/// signals, never process-fatal.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TransportError {
    /// No such peer is configured.
    #[error("unknown peer {node}")]
    UnknownPeer {
        /// The addressed node.
        node: u64,
    },
    /// The peer is unreachable (TLS/auth failure, reset, refused,
    /// suspended, no connection).
    #[error("peer unreachable: {detail}")]
    Unreachable {
        /// Human-readable cause.
        detail: String,
    },
    /// No response arrived before the deadline (the H3 stream was reset).
    #[error("peer RPC timed out")]
    Timeout,
    /// The transport is shutting down.
    #[error("peer transport shut down")]
    ShuttingDown,
    /// The remote application refused the RPC (decode fault, wrong group,
    /// member refusal). Back off and retry per `OpenRaft` policy.
    #[error("remote peer refused RPC: {detail}")]
    RemoteRefused {
        /// Remote cause.
        detail: String,
    },
}

/// Per-peer diagnostics for the admin plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerStats {
    /// Whether the control H3 connection is currently established.
    pub connected_control: bool,
    /// Whether the bulk H3 connection is currently established.
    pub connected_bulk: bool,
    /// Whether the link is administratively suspended (partition).
    pub suspended: bool,
    /// Incarnation last validated on this link, if any.
    pub incarnation: Option<u64>,
    /// Reconnect (redial) attempts so far.
    pub reconnects: u64,
    /// Control requests issued.
    pub control_requests: u64,
    /// Bulk requests issued.
    pub bulk_requests: u64,
    /// Control bytes sent.
    pub control_bytes_sent: u64,
    /// Control bytes received.
    pub control_bytes_received: u64,
    /// Bulk bytes sent.
    pub bulk_bytes_sent: u64,
    /// Bulk bytes received.
    pub bulk_bytes_received: u64,
    /// Last observed control RTT in milliseconds, if connected.
    pub rtt_ms: Option<u64>,
    /// Currently active H3 streams.
    pub active_streams: u64,
}

/// Serves incoming peer RPCs for one group. The transport only routes
/// bytes; the returned response must match the request family.
pub trait PeerHandler: Send + Sync + 'static {
    /// Serves one request from `from` for `group`.
    fn handle(
        &self,
        from: NodeId,
        group: ConsensusGroupId,
        request: PeerRequest,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<PeerResponse, PeerRpcError>> + Send + '_>,
    >;
}

/// TLS material for the mesh: this node's certificate plus how to verify
/// everyone else. Exactly one of `peer_certs` / `insecure_skip_verify`
/// authenticates outgoing connections.
#[derive(Debug, Clone)]
pub struct TlsMaterial {
    /// This node's certificate and key.
    pub cert: NodeCert,
    /// Expected peer certificates (trust anchors) by node.
    pub peer_certs: HashMap<NodeId, Vec<u8>>,
    /// Test-only escape hatch for `kivi-lab` (ephemeral ports and fresh
    /// data directories per run make static pins impractical there).
    /// Never set in production: without verification any network peer
    /// could impersonate a member.
    pub insecure_skip_verify: bool,
}

/// Internal driver command: one H3 request to execute on the owner reactor.
/// The front deadline-bounds the reply channel; the driver never sees the
/// timeout (dropping the reply on timeout resets the stream).
struct CallCommand {
    target: NodeId,
    lane: Lane,
    group: ConsensusGroupId,
    request: PeerRequest,
    reply: futures::channel::oneshot::Sender<Result<PeerResponse, TransportError>>,
}

/// Work for the owner-reactor mesh driver.
enum DriverCommand {
    /// Execute one H3 request stream.
    Call(Box<CallCommand>),
    /// Drop a peer's connections (suspend path).
    DropPeer(NodeId),
    /// Close the endpoint and stop the driver (shutdown path).
    Shutdown {
        /// Acked after the endpoint closed, so restarts rebind promptly.
        reply: futures::channel::oneshot::Sender<()>,
    },
}

#[derive(Debug, Default)]
struct LaneCounters {
    requests: AtomicU64,
    bytes_sent: AtomicU64,
    bytes_received: AtomicU64,
}

#[derive(Debug)]
struct PeerLink {
    addr: SocketAddr,
    suspended: AtomicBool,
    reconnects: AtomicU64,
    incarnation: futures::lock::Mutex<Option<u64>>,
    connected_control: AtomicBool,
    connected_bulk: AtomicBool,
    /// Last control RTT in millis (`u64::MAX` = none yet).
    rtt_ms: AtomicU64,
    /// In-flight H3 streams both directions (partition drain waits on it).
    active: AtomicU64,
    control: LaneCounters,
    bulk: LaneCounters,
}

impl PeerLink {
    fn new(addr: SocketAddr) -> Self {
        Self {
            addr,
            suspended: AtomicBool::new(false),
            reconnects: AtomicU64::new(0),
            incarnation: futures::lock::Mutex::new(None),
            connected_control: AtomicBool::new(false),
            connected_bulk: AtomicBool::new(false),
            rtt_ms: AtomicU64::new(u64::MAX),
            active: AtomicU64::new(0),
            control: LaneCounters::default(),
            bulk: LaneCounters::default(),
        }
    }

    /// RTT in millis, if ever observed.
    fn rtt(&self) -> Option<u64> {
        let rtt = self.rtt_ms.load(Ordering::SeqCst);
        (rtt != u64::MAX).then_some(rtt)
    }
}

struct TransportInner {
    local: PeerIdentity,
    cluster: ClusterId,
    config: TransportConfig,
    links: HashMap<NodeId, Arc<PeerLink>>,
    incarnations: futures::lock::Mutex<IncarnationTable>,
    running: AtomicBool,
    ready: AtomicBool,
    handler: Arc<dyn PeerHandler>,
    driver_tx: async_channel::Sender<DriverCommand>,
    active_streams: AtomicU64,
}

/// QUIC/H3 peer mesh. Cloneable (`Send + Sync`): connection tasks run on
/// the owner's Compio reactor; the front's async methods are callable from
/// any thread.
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
    /// Opens the mesh: binds one UDP endpoint, spawns the accept loop plus
    /// one dial loop per higher-addressed peer, and returns the transport
    /// with its bound address. `peers` holds every node (including local,
    /// whose address is the bind address — port `0` selects an ephemeral
    /// port); only higher-`NodeId` entries are dialed.
    ///
    /// # Errors
    ///
    /// Returns [`TransportError::Unreachable`] when the endpoint cannot
    /// bind or TLS material is incoherent. Dial failures never fail `open`.
    ///
    /// # Panics
    ///
    /// Panics when the loopback fallback bind address fails to parse (a
    /// literal, so only if the literal itself is edited into garbage).
    #[allow(clippy::too_many_lines)]
    pub async fn open(
        config: TransportConfig,
        local: PeerIdentity,
        cluster: ClusterId,
        peers: &HashMap<NodeId, SocketAddr>,
        tls: TlsMaterial,
        handler: Arc<dyn PeerHandler>,
    ) -> Result<(Self, SocketAddr), TransportError> {
        use TransportError as Fault;
        // Process-default crypto provider for rustls (ring): required
        // before any client/server config builds. Idempotent process-wide;
        // already-set is fine.
        let _ = rustls::crypto::ring::default_provider().install_default();
        if !tls.insecure_skip_verify {
            for peer in peers.keys() {
                if *peer != local.node && !tls.peer_certs.contains_key(peer) {
                    return Err(Fault::Unreachable {
                        detail: format!("no pinned peer certificate for node {}", peer.as_u64()),
                    });
                }
            }
        }
        let bind_addr = peers
            .get(&local.node)
            .copied()
            .unwrap_or_else(|| "127.0.0.1:0".parse().expect("loopback bind address parses"));
        // QUIC state-machine tuning shared by incoming and outgoing
        // connections (stream concurrency, windows, fairness, timeouts).
        let quic_transport = Arc::new(config.quic_transport());
        let rustls_server = rustls_server_config_for_builder(&tls.cert)?;
        let mut server_config = ServerConfig::with_crypto(Arc::new(
            compio::quic::crypto::rustls::QuicServerConfig::try_from(rustls_server).map_err(
                |error| Fault::Unreachable {
                    detail: format!("peer QUIC server crypto: {error}"),
                },
            )?,
        ));
        server_config.transport_config(Arc::clone(&quic_transport));
        // The endpoint binds through the client builder (which sets the
        // default client config for dials); the tuned server config
        // installs right after for incoming connections. Binds retry
        // through release races: a previous generation's UDP socket frees
        // asynchronously in the QUIC worker, and parallel formations may
        // steal a probed port — both fail loudly after the window instead
        // of serving a half mesh.
        let endpoint = {
            let deadline = std::time::Instant::now() + Duration::from_secs(15);
            loop {
                match client_builder_for(&tls)?
                    .with_alpn_protocols(&["h3"])
                    .bind(bind_addr)
                    .await
                {
                    Ok(endpoint) => break endpoint,
                    Err(error) => {
                        if std::time::Instant::now() >= deadline {
                            return Err(Fault::Unreachable {
                                detail: format!(
                                    "peer endpoint bind failed on {bind_addr}: {error}"
                                ),
                            });
                        }
                        compio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
            }
        };
        endpoint.set_server_config(Some(server_config));
        let bound = endpoint.local_addr().map_err(|error| Fault::Unreachable {
            detail: format!("peer listener address unreadable: {error}"),
        })?;
        let mut links = HashMap::new();
        for (node, addr) in peers {
            if *node == local.node {
                continue;
            }
            links.insert(*node, Arc::new(PeerLink::new(*addr)));
        }
        let paused = config.start_paused;
        let (driver_tx, driver_rx) = async_channel::bounded::<DriverCommand>(256);
        let transport = Self {
            inner: Arc::new(TransportInner {
                local,
                cluster,
                config,
                links,
                incarnations: futures::lock::Mutex::new(IncarnationTable::new()),
                running: AtomicBool::new(true),
                ready: AtomicBool::new(!paused),
                handler,
                driver_tx,
                active_streams: AtomicU64::new(0),
            }),
        };
        {
            // Tuned rustls client config for per-dial QUIC configs: pinned
            // anchors normally, the test-only insecure verifier for lab.
            let rustls_client = Arc::new(if tls.insecure_skip_verify {
                crate::tls::client_config_insecure()
            } else {
                crate::tls::client_config_pinned(&tls.peer_certs).map_err(|detail| {
                    TransportError::Unreachable {
                        detail: format!("peer TLS client config: {detail}"),
                    }
                })?
            });
            let driver = MeshDriver {
                transport: transport.clone(),
                endpoint,
                tls: Arc::new(tls),
                rustls_client,
                quic_transport: Arc::new(transport.inner.config.quic_transport()),
                outgoing: std::rc::Rc::new(std::cell::RefCell::new(HashMap::new())),
                incoming: std::rc::Rc::new(std::cell::RefCell::new(HashMap::new())),
                shutdown_ack: std::rc::Rc::new(std::cell::RefCell::new(None)),
            };
            compio::runtime::spawn(async move { driver.run(driver_rx).await }).detach();
        }
        Ok((transport, bound))
    }

    /// Issues one control-lane RPC (votes, appends, snapshot fragments).
    ///
    /// # Errors
    ///
    /// Returns [`TransportError`] on unknown peers, timeouts, disconnects,
    /// remote refusals, or shutdown.
    pub async fn call(
        &self,
        target: NodeId,
        group: ConsensusGroupId,
        request: PeerRequest,
        timeout: Duration,
    ) -> Result<PeerResponse, TransportError> {
        self.dispatch(target, Lane::Control, group, request, timeout)
            .await
    }

    /// Issues one bulk-lane RPC (manifest/chunk sidecars) on the bulk H3
    /// connection, independent of control congestion/flow domains.
    ///
    /// # Errors
    ///
    /// Returns [`TransportError`] on unknown peers, timeouts, disconnects,
    /// remote refusals, or shutdown.
    pub async fn call_bulk(
        &self,
        target: NodeId,
        group: ConsensusGroupId,
        request: PeerRequest,
        timeout: Duration,
    ) -> Result<PeerResponse, TransportError> {
        self.dispatch(target, Lane::Bulk, group, request, timeout)
            .await
    }

    async fn dispatch(
        &self,
        target: NodeId,
        lane: Lane,
        group: ConsensusGroupId,
        request: PeerRequest,
        timeout: Duration,
    ) -> Result<PeerResponse, TransportError> {
        if !self.inner.links.contains_key(&target) {
            return Err(TransportError::UnknownPeer {
                node: target.as_u64(),
            });
        }
        if !self.inner.running.load(Ordering::SeqCst) {
            return Err(TransportError::ShuttingDown);
        }
        let (tx, rx) = futures::channel::oneshot::channel();
        self.inner
            .driver_tx
            .send(DriverCommand::Call(Box::new(CallCommand {
                target,
                lane,
                group,
                request,
                reply: tx,
            })))
            .await
            .map_err(|_| TransportError::ShuttingDown)?;
        match compio::time::timeout(timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(TransportError::ShuttingDown),
            Err(_) => Err(TransportError::Timeout),
        }
    }

    /// Snapshots per-peer diagnostics (admin plane).
    pub async fn stats(&self) -> HashMap<NodeId, PeerStats> {
        let mut out = HashMap::new();
        for (node, link) in &self.inner.links {
            out.insert(
                *node,
                PeerStats {
                    connected_control: link.connected_control.load(Ordering::SeqCst),
                    connected_bulk: link.connected_bulk.load(Ordering::SeqCst),
                    suspended: link.suspended.load(Ordering::SeqCst),
                    incarnation: *link.incarnation.lock().await,
                    reconnects: link.reconnects.load(Ordering::SeqCst),
                    control_requests: link.control.requests.load(Ordering::SeqCst),
                    bulk_requests: link.bulk.requests.load(Ordering::SeqCst),
                    control_bytes_sent: link.control.bytes_sent.load(Ordering::SeqCst),
                    control_bytes_received: link.control.bytes_received.load(Ordering::SeqCst),
                    bulk_bytes_sent: link.bulk.bytes_sent.load(Ordering::SeqCst),
                    bulk_bytes_received: link.bulk.bytes_received.load(Ordering::SeqCst),
                    rtt_ms: link.rtt(),
                    active_streams: self.inner.active_streams.load(Ordering::SeqCst),
                },
            );
        }
        out
    }

    /// Releases paused dial/serve loops. Idempotent.
    pub fn start(&self) {
        self.inner.ready.store(true, Ordering::SeqCst);
    }

    /// Suspends one peer link (partition hook): drops connections, blocks
    /// dials, and refuses inbound requests for the peer until resumed.
    /// Returns only after the link's in-flight streams drained (bounded),
    /// so the partition is airtight for fencing verdicts.
    pub async fn suspend_peer(&self, peer: NodeId) {
        let Some(link) = self.inner.links.get(&peer) else {
            return;
        };
        link.suspended.store(true, Ordering::SeqCst);
        let _ = self
            .inner
            .driver_tx
            .send(DriverCommand::DropPeer(peer))
            .await;
        // Bounded drain: evicted tasks observe the cleared connections and
        // suspended flag within one quantum and exit, closing their
        // streams. A cleared flag alone would still leave a dying stream
        // holding a verdict for up to one RPC timeout.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while link.active.load(Ordering::SeqCst) > 0 {
            if std::time::Instant::now() >= deadline {
                break;
            }
            compio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Resumes a suspended peer link.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn resume_peer(&self, peer: NodeId) {
        if let Some(link) = self.inner.links.get(&peer) {
            link.suspended.store(false, Ordering::SeqCst);
        }
    }

    /// Stops dial loops, closes the endpoint and all connections, and
    /// fails pending RPCs. Waits for the driver to ack the endpoint
    /// close, so the UDP socket releases before the caller rebinds it on
    /// restart (no `TIME_WAIT` dance, just ordering).
    pub async fn shutdown(&self) {
        self.inner.running.store(false, Ordering::SeqCst);
        let (tx, rx) = futures::channel::oneshot::channel();
        if self
            .inner
            .driver_tx
            .send(DriverCommand::Shutdown { reply: tx })
            .await
            .is_err()
        {
            return;
        }
        // Bounded teardown wait: the driver acks after its loops exited
        // and the endpoint shut down. A stuck teardown must never hang
        // node shutdown forever; restart rebinds retry through lingering
        // sockets.
        match compio::time::timeout(Duration::from_secs(5), rx).await {
            Ok(_) => {}
            Err(_) => {
                tracing::warn!("peer mesh teardown timed out; continuing shutdown");
            }
        }
    }

    /// Returns the local peer identity.
    #[must_use]
    pub fn local(&self) -> PeerIdentity {
        self.inner.local
    }
}

/// Builds the rustls server config the endpoint builder needs (single
/// cert + `h3` ALPN).
fn rustls_server_config_for_builder(
    cert: &NodeCert,
) -> Result<rustls::ServerConfig, TransportError> {
    let mut config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(
            vec![rustls::pki_types::CertificateDer::from(
                cert.cert_der.clone(),
            )],
            rustls::pki_types::PrivateKeyDer::try_from(cert.key_der.clone()).map_err(|error| {
                TransportError::Unreachable {
                    detail: format!("peer key invalid: {error}"),
                }
            })?,
        )
        .map_err(|error| TransportError::Unreachable {
            detail: format!("peer TLS server config: {error}"),
        })?;
    config.alpn_protocols = vec![b"h3".to_vec()];
    Ok(config)
}

/// Builds the client builder for the endpoint bind: pinned trust anchors
/// in normal mode, the test-only insecure verifier when `kivi-lab` asks
/// for it. The builder's config becomes the endpoint default; dials pass
/// a tuned per-connection config (see [`MeshDriver::dial_client_config`]).
fn client_builder_for(
    tls: &TlsMaterial,
) -> Result<ClientBuilder<rustls::ClientConfig>, TransportError> {
    if tls.insecure_skip_verify {
        Ok(ClientBuilder::new_with_no_server_verification())
    } else {
        let rustls_config =
            crate::tls::client_config_pinned(&tls.peer_certs).map_err(|detail| {
                TransportError::Unreachable {
                    detail: format!("peer TLS client config: {detail}"),
                }
            })?;
        Ok(ClientBuilder::new_with_rustls_client_config(rustls_config))
    }
}

/// One lane's outgoing H3 client: the QUIC connection plus its request
/// sender. The H3 driver future runs detached alongside; when it ends the
/// entry is dropped and the next call redials.
type H3Sender = compio::quic::h3::client::SendRequest<compio::quic::h3::OpenStreams, H3Bytes>;

/// Buffer type for H3 stream bodies (single-segment chunk units).
type H3Bytes = compio::buf::bytes::Bytes;

/// One lane's inbound H3 request stream body reader lives inline at the
/// single server call site (client and server stream types differ).
type H3ClientStream =
    compio::quic::h3::client::RequestStream<compio::quic::h3::BidiStream<H3Bytes>, H3Bytes>;

/// Server-side request stream type for refusal responses.
type H3ServerStream =
    compio::quic::h3::server::RequestStream<compio::quic::h3::BidiStream<H3Bytes>, H3Bytes>;

/// Server-side request resolver type for inbound H3 requests (bound to
/// the QUIC connection; streams resolve per request).
type H3Resolver = compio::quic::h3::server::RequestResolver<compio::quic::Connection, H3Bytes>;

struct Outgoing {
    send: H3Sender,
    /// Dial generation that created this entry (stale driver ends must
    /// not evict a newer entry). The H3 driver task owns the QUIC
    /// connection lifetime; this entry only routes senders.
    generation: u64,
}

/// Owner-reactor mesh driver: owns the endpoint, dials, serves, and
/// executes H3 streams. Everything `!Send` stays here; the front talks to
/// it through [`CallCommand`]s only. Cloneable across the driver's own
/// same-reactor tasks through `Rc`-shared connection state.
/// One tracked incoming QUIC connection: per-lane activity is learned
/// from the first validated request of each lane (the dialer opens one
/// connection per lane, so flags stay exact in steady state).
struct IncomingConn {
    conn: Connection,
    control: bool,
    bulk: bool,
}

#[derive(Clone)]
struct MeshDriver {
    transport: PeerTransport,
    endpoint: Endpoint,
    tls: Arc<TlsMaterial>,
    /// Tuned rustls client config for per-dial QUIC configs (pinned
    /// anchors, or the test-only insecure verifier): transport windows
    /// plus no 0-RTT in every mode.
    rustls_client: Arc<rustls::ClientConfig>,
    quic_transport: Arc<QuicTransportConfig>,
    outgoing: std::rc::Rc<std::cell::RefCell<HashMap<(NodeId, Lane), Outgoing>>>,
    /// Incoming QUIC connections by validated peer (suspend drops them).
    incoming: std::rc::Rc<std::cell::RefCell<HashMap<NodeId, Vec<IncomingConn>>>>,
    /// Shutdown ack, answered only after every driver loop exited and the
    /// endpoint dropped — so the socket releases before the caller rebinds.
    shutdown_ack: std::rc::Rc<std::cell::RefCell<Option<futures::channel::oneshot::Sender<()>>>>,
}

impl MeshDriver {
    async fn run(self, commands: async_channel::Receiver<DriverCommand>) {
        let accept = self.accept_loop();
        let dial = self.dial_loops();
        let serve = self.serve_commands(commands);
        let _ = futures::join!(accept, dial, serve);
        // Every loop exited: shut the endpoint down (its worker closes
        // the UDP socket; plain drop would leave it to worker teardown
        // timing) and only then ack shutdown — restarts rebind promptly.
        // (Partial move of owned `self`: disjoint fields, no live borrows.)
        let _ = self.endpoint.shutdown().await;
        if let Some(reply) = self.shutdown_ack.borrow_mut().take() {
            let _ = reply.send(());
        }
    }

    /// Waits for readiness (static-bootstrap discipline).
    async fn wait_ready(&self) -> bool {
        loop {
            if self.transport.inner.ready.load(Ordering::SeqCst) {
                return true;
            }
            if !self.transport.inner.running.load(Ordering::SeqCst) {
                return false;
            }
            compio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    async fn accept_loop(&self) {
        if !self.wait_ready().await {
            return;
        }
        while self.transport.inner.running.load(Ordering::SeqCst) {
            let Some(incoming) = self.endpoint.wait_incoming().await else {
                break;
            };
            let driver = self.clone();
            compio::runtime::spawn(async move {
                let Ok(conn) = incoming.await else {
                    return;
                };
                driver.serve_quic_conn(conn).await;
            })
            .detach();
        }
    }

    async fn dial_loops(&self) {
        if !self.wait_ready().await {
            return;
        }
        // Only the higher NodeId dials: two connections per pair, no
        // simultaneous-open coordination needed.
        let mut tasks = Vec::new();
        for (node, link) in self.transport.inner.links.clone() {
            if node.as_u64() < self.transport.inner.local.node.as_u64() {
                continue;
            }
            let driver = self.clone();
            tasks.push(compio::runtime::spawn(async move {
                driver.dial_loop(node, link).await;
            }));
        }
        let _ = futures::future::join_all(tasks).await;
    }

    async fn serve_commands(&self, commands: async_channel::Receiver<DriverCommand>) {
        while let Ok(command) = commands.recv().await {
            match command {
                DriverCommand::Call(command) => {
                    if !self.transport.inner.running.load(Ordering::SeqCst) {
                        let _ = command.reply.send(Err(TransportError::ShuttingDown));
                        continue;
                    }
                    let driver = self.clone();
                    compio::runtime::spawn(async move {
                        driver.execute_command(*command).await;
                    })
                    .detach();
                }
                DriverCommand::DropPeer(peer) => {
                    self.drop_peer_connections(peer);
                }
                DriverCommand::Shutdown { reply } => {
                    self.endpoint.close(VarInt::from_u32(0), b"shutdown");
                    *self.shutdown_ack.borrow_mut() = Some(reply);
                    break;
                }
            }
        }
    }

    /// Drops every connection to `peer` (suspend path): outgoing cache
    /// entries plus tracked incoming connections. In-flight streams reset;
    /// per-link activity drains through the suspend wait.
    fn drop_peer_connections(&self, peer: NodeId) {
        self.outgoing
            .borrow_mut()
            .retain(|(node, _), _| *node != peer);
        let mut incoming = self.incoming.borrow_mut();
        if let Some(conns) = incoming.remove(&peer) {
            for tracked in conns {
                tracked.conn.close(VarInt::from_u32(0), b"suspended");
            }
        }
        if let Some(link) = self.transport.inner.links.get(&peer) {
            link.connected_control.store(false, Ordering::SeqCst);
            link.connected_bulk.store(false, Ordering::SeqCst);
        }
    }

    /// Records first validated inbound activity of `lane` from `node`
    /// (connection identity comes from the QUIC/TLS binding or, in
    /// test-only insecure mode, the validated headers).
    fn note_inbound_lane(&self, node: NodeId, lane: Lane, conn: &Connection) {
        let mut incoming = self.incoming.borrow_mut();
        let conns = incoming.entry(node).or_default();
        let id = conn.stable_id();
        if !conns.iter().any(|tracked| tracked.conn.stable_id() == id) {
            conns.push(IncomingConn {
                conn: conn.clone(),
                control: false,
                bulk: false,
            });
        }
        if let Some(tracked) = conns
            .iter_mut()
            .find(|tracked| tracked.conn.stable_id() == id)
        {
            match lane {
                Lane::Control => tracked.control = true,
                Lane::Bulk => tracked.bulk = true,
            }
        }
        let control = conns.iter().any(|tracked| tracked.control);
        let bulk = conns.iter().any(|tracked| tracked.bulk);
        if let Some(link) = self.transport.inner.links.get(&node) {
            link.connected_control.store(control, Ordering::SeqCst);
            link.connected_bulk.store(bulk, Ordering::SeqCst);
            let rtt_ms = conn.rtt().as_millis();
            link.rtt_ms
                .store(u64::try_from(rtt_ms).unwrap_or(u64::MAX), Ordering::SeqCst);
        }
    }

    /// Forgets a closed incoming connection, recomputing lane flags from
    /// survivors.
    fn forget_incoming(&self, node: Option<NodeId>, conn: &Connection) {
        let Some(node) = node else {
            return;
        };
        let mut incoming = self.incoming.borrow_mut();
        let Some(conns) = incoming.get_mut(&node) else {
            return;
        };
        let id = conn.stable_id();
        conns.retain(|tracked| tracked.conn.stable_id() != id);
        let control = conns.iter().any(|tracked| tracked.control);
        let bulk = conns.iter().any(|tracked| tracked.bulk);
        if conns.is_empty() {
            incoming.remove(&node);
        }
        if let Some(link) = self.transport.inner.links.get(&node) {
            link.connected_control.store(control, Ordering::SeqCst);
            link.connected_bulk.store(bulk, Ordering::SeqCst);
        }
    }

    /// Builds the tuned per-dial QUIC client config: transport windows
    /// plus no 0-RTT in every mode (replayable handshakes must never
    /// carry mutating Raft RPCs; all peer RPCs are idempotent besides,
    /// but the safe default stands).
    fn dial_client_config(&self) -> Option<ClientConfig> {
        let crypto = compio::quic::crypto::rustls::QuicClientConfig::try_from(Arc::clone(
            &self.rustls_client,
        ))
        .ok()?;
        let mut config = ClientConfig::new(Arc::new(crypto));
        config.transport_config(Arc::clone(&self.quic_transport));
        Some(config)
    }
}

impl MeshDriver {
    /// Executes one driver command: runs one H3 request stream to
    /// completion and answers the caller. The per-call deadline is
    /// enforced by the front; dropping the stream on timeout resets it
    /// (cancellation for truncated suffixes and shutdown).
    async fn execute_command(&self, command: CallCommand) {
        let result = self.execute_inner(&command).await;
        let _ = command.reply.send(result);
    }

    async fn execute_inner(&self, command: &CallCommand) -> Result<PeerResponse, TransportError> {
        use TransportError as Fault;
        let inner = &self.transport.inner;
        let link = inner.links.get(&command.target).ok_or(Fault::UnknownPeer {
            node: command.target.as_u64(),
        })?;
        if link.suspended.load(Ordering::SeqCst) {
            // Suspended links fail fast as timeouts (partition tests and
            // drain tooling expect `Timeout`, never a hang).
            return Err(Fault::Timeout);
        }
        let max_body = match command.lane {
            Lane::Control => MAX_RAFT_BODY_BYTES,
            Lane::Bulk => MAX_BULK_BODY_BYTES,
        };
        let (path, body) =
            crate::peer::encode_h3_request(&command.request, command.group.tablet().as_u64());
        let media = crate::peer::h3_media_type(&command.request);
        let counters = match command.lane {
            Lane::Control => &link.control,
            Lane::Bulk => &link.bulk,
        };
        counters.requests.fetch_add(1, Ordering::SeqCst);
        inner.active_streams.fetch_add(1, Ordering::SeqCst);
        let result = self
            .h3_round_trip(command, link, &path, media, &body, max_body)
            .await;
        inner.active_streams.fetch_sub(1, Ordering::SeqCst);
        let response_body = result?;
        let response =
            crate::peer::decode_h3_response(&command.request, &response_body).map_err(|error| {
                Fault::RemoteRefused {
                    detail: format!("peer response undecodable: {error}"),
                }
            })?;
        // Content verification stays in the sidecar store (trust
        // boundary): it recomputes every id before indexing and never
        // installs unverified bytes. The transport's job here is bounded,
        // segmented receive (no 64 MiB buffers) plus caps.
        counters
            .bytes_received
            .fetch_add(response_body.len() as u64, Ordering::SeqCst);
        Ok(response)
    }

    /// One H3 request/response round trip on the lane's connection,
    /// dialing on demand. Bodies stream with per-segment caps; dropping
    /// the stream resets it (cancellation).
    #[allow(clippy::too_many_lines)]
    async fn h3_round_trip(
        &self,
        command: &CallCommand,
        link: &Arc<PeerLink>,
        path: &str,
        media: &str,
        body: &[u8],
        max_body: usize,
    ) -> Result<Vec<u8>, TransportError> {
        use TransportError as Fault;
        let inner = &self.transport.inner;
        let mut sender = self.outgoing(command.target, command.lane, link).await?;
        // H3 requires absolute request targets: the authority names the
        // dialed peer (routing uses the path; the authority is never
        // trusted for identity — TLS plus Kivi headers decide that).
        let uri = format!(
            "https://{}{}",
            crate::tls::server_name_for(command.target),
            path
        );
        let request = http::Request::builder()
            .method(match command.lane {
                Lane::Control => http::Method::POST,
                Lane::Bulk => http::Method::GET,
            })
            .uri(uri)
            .header("content-type", media)
            .header(HDR_CLUSTER, inner.cluster.as_u128().to_string())
            .header(HDR_NODE, inner.local.node.as_u64().to_string())
            .header(
                HDR_INCARNATION,
                inner.local.incarnation.as_u64().to_string(),
            )
            .header(HDR_CAPS, REQUIRED_CAPABILITIES.to_string())
            .body(())
            .map_err(|error| Fault::Unreachable {
                detail: format!("peer request build failed: {error}"),
            })?;
        // Connection health tracks whether any HTTP response arrived: an
        // app-level refusal (400/500) proves the connection is FINE and
        // must not drop it (dropping on refusals redial-storms: every
        // missing sidecar would kill a healthy connection). Only stream
        // transport failures drop the cached client.
        let mut conn_healthy = false;
        let outcome = async {
            let mut stream =
                sender
                    .send_request(request)
                    .await
                    .map_err(|error| Fault::Unreachable {
                        detail: format!("peer request rejected: {error}"),
                    })?;
            if !body.is_empty() {
                stream
                    .send_data(compio::buf::bytes::Bytes::from(body.to_vec()))
                    .await
                    .map_err(|error| Fault::Unreachable {
                        detail: format!("peer request body failed: {error}"),
                    })?;
            }
            stream.finish().await.map_err(|error| Fault::Unreachable {
                detail: format!("peer request finish failed: {error}"),
            })?;
            let response = stream
                .recv_response()
                .await
                .map_err(|error| Fault::Unreachable {
                    detail: format!("peer response missing: {error}"),
                })?;
            // Headers arrived: the connection served this stream.
            conn_healthy = true;
            match response.status() {
                http::StatusCode::OK => {}
                http::StatusCode::SERVICE_UNAVAILABLE => return Err(Fault::Timeout),
                http::StatusCode::FORBIDDEN => {
                    return Err(Fault::Unreachable {
                        detail: "peer authentication refused".to_owned(),
                    });
                }
                status => {
                    let detail = read_client_capped(&mut stream, 4096)
                        .await
                        .unwrap_or_default();
                    return Err(Fault::RemoteRefused {
                        detail: format!(
                            "peer refused with {status}: {}",
                            String::from_utf8_lossy(&detail)
                        ),
                    });
                }
            }
            let received = read_client_capped(&mut stream, max_body)
                .await
                .map_err(|()| Fault::RemoteRefused {
                    detail: "peer response exceeds body cap".to_owned(),
                })?;
            Ok(received)
        }
        .await;
        if !conn_healthy {
            self.drop_outgoing(command.target, command.lane);
        }
        let counters = match command.lane {
            Lane::Control => &link.control,
            Lane::Bulk => &link.bulk,
        };
        counters
            .bytes_sent
            .fetch_add(body.len() as u64, Ordering::SeqCst);
        outcome
    }

    /// Returns the lane's outgoing H3 sender, dialing when absent. Both
    /// lanes redial independently so one lane's failure never blocks the
    /// other; failures surface per call and the next call retries.
    async fn outgoing(
        &self,
        target: NodeId,
        lane: Lane,
        link: &Arc<PeerLink>,
    ) -> Result<H3Sender, TransportError> {
        use TransportError as Fault;
        if let Some(sender) = self
            .outgoing
            .borrow()
            .get(&(target, lane))
            .map(|out| out.send.clone())
        {
            return Ok(sender);
        }
        self.dial_lane(target, lane, link).await?;
        self.outgoing
            .borrow()
            .get(&(target, lane))
            .map(|out| out.send.clone())
            .ok_or(Fault::Unreachable {
                detail: format!("peer {} {lane:?} unavailable after dial", target.as_u64()),
            })
    }

    /// Dials one lane's QUIC connection and completes the H3 handshake,
    /// caching the sender. Runs the H3 driver detached; its end drops the
    /// entry so the next call redials.
    async fn dial_lane(
        &self,
        target: NodeId,
        lane: Lane,
        link: &Arc<PeerLink>,
    ) -> Result<(), TransportError> {
        use TransportError as Fault;
        let addr = link.addr;
        let name = crate::tls::server_name_for(target);
        let quic = self
            .endpoint
            .connect(addr, &name, self.dial_client_config())
            .map_err(|error| Fault::Unreachable {
                detail: format!("peer {} dial refused: {error}", target.as_u64()),
            })?
            .await
            .map_err(|error| Fault::Unreachable {
                detail: format!("peer {} connect failed: {error}", target.as_u64()),
            })?;
        link.reconnects.fetch_add(1, Ordering::SeqCst);
        let rtt_ms = quic.rtt().as_millis();
        link.rtt_ms
            .store(u64::try_from(rtt_ms).unwrap_or(u64::MAX), Ordering::SeqCst);
        match lane {
            Lane::Control => link.connected_control.store(true, Ordering::SeqCst),
            Lane::Bulk => link.connected_bulk.store(true, Ordering::SeqCst),
        }
        // `client::new` takes the QUIC connection; the spawned H3 driver
        // below owns its lifetime (QUIC handles are refcounted clones).
        let (mut driver, send) =
            compio::quic::h3::client::new(quic)
                .await
                .map_err(|error| Fault::Unreachable {
                    detail: format!("peer {} H3 handshake failed: {error}", target.as_u64()),
                })?;
        let generation = link.reconnects.load(Ordering::SeqCst);
        self.outgoing
            .borrow_mut()
            .insert((target, lane), Outgoing { send, generation });
        let outgoing = self.outgoing.clone();
        compio::runtime::spawn(async move {
            let _ = driver.wait_idle().await;
            // Driver ended: drop the entry so the next call redials. A
            // newer generation's entry must survive.
            let mut cache = outgoing.borrow_mut();
            if cache
                .get(&(target, lane))
                .is_some_and(|out| out.generation == generation)
            {
                cache.remove(&(target, lane));
            }
        })
        .detach();
        Ok(())
    }

    /// Drops a cached outgoing client (failure path) so the next call
    /// redials instead of reusing a broken stream.
    fn drop_outgoing(&self, target: NodeId, lane: Lane) {
        self.outgoing.borrow_mut().remove(&(target, lane));
        if let Some(link) = self.transport.inner.links.get(&target) {
            match lane {
                Lane::Control => link.connected_control.store(false, Ordering::SeqCst),
                Lane::Bulk => link.connected_bulk.store(false, Ordering::SeqCst),
            }
        }
    }

    /// Dial loop for one higher-addressed peer: keeps both lanes warm
    /// with bounded backoff until shutdown. Suspended links stay down;
    /// on-demand dials in [`MeshDriver::outgoing`] cover followers and
    /// post-resume healing without waiting for the loop tick.
    async fn dial_loop(&self, node: NodeId, link: Arc<PeerLink>) {
        let inner = &self.transport.inner;
        let mut backoff = inner.config.backoff_min;
        loop {
            if !inner.running.load(Ordering::SeqCst) {
                break;
            }
            if link.suspended.load(Ordering::SeqCst) {
                compio::time::sleep(Duration::from_millis(200)).await;
                continue;
            }
            let missing = [Lane::Control, Lane::Bulk]
                .into_iter()
                .any(|lane| !self.outgoing.borrow().contains_key(&(node, lane)));
            if !missing {
                compio::time::sleep(Duration::from_millis(200)).await;
                continue;
            }
            let mut ok = true;
            for lane in [Lane::Control, Lane::Bulk] {
                if self.outgoing.borrow().contains_key(&(node, lane)) {
                    continue;
                }
                if self.dial_lane(node, lane, &link).await.is_err() {
                    ok = false;
                    break;
                }
            }
            if ok {
                backoff = inner.config.backoff_min;
            } else {
                sleep_capped(&mut backoff, inner.config.backoff_max).await;
            }
        }
    }

    /// Serves one accepted QUIC connection: binds the peer certificate to
    /// a configured node (unknown certificates serve nothing), then runs
    /// the H3 server accept loop with one task per request. Tracked
    /// connections drop on suspend.
    async fn serve_quic_conn(&self, conn: Connection) {
        let bound = self.incoming_peer(&conn);
        // Unbound (test-only insecure) connections still serve: identity
        // rests on validated headers alone (loopback lab traffic only).
        if !self.tls.insecure_skip_verify && bound.is_none() {
            return;
        }
        let Ok(mut h3) = compio::quic::h3::server::builder()
            .build::<_, compio::buf::bytes::Bytes>(conn.clone())
            .await
        else {
            return;
        };
        loop {
            let Ok(Some(resolver)) = h3.accept().await else {
                break;
            };
            let driver = self.clone();
            let conn = conn.clone();
            compio::runtime::spawn(async move {
                driver.serve_h3_request(bound, &conn, resolver).await;
            })
            .detach();
        }
        self.forget_incoming(bound, &conn);
    }

    /// Binds an incoming QUIC connection's peer certificate to a
    /// configured node. Unknown certificates serve nothing: the handshake
    /// TLS already rejected untrusted peers in normal mode, and this maps
    /// the trusted certificate to its Kivi identity so a request claiming
    /// another node is refused. In test-only insecure mode there is no pin
    /// to bind (loopback lab traffic); identity then rests on the
    /// validated headers alone.
    fn incoming_peer(&self, conn: &Connection) -> Option<NodeId> {
        if self.tls.peer_certs.is_empty() {
            return None;
        }
        let chain = conn.peer_identity()?;
        let leaf = chain.first()?;
        self.tls
            .peer_certs
            .iter()
            .find_map(|(node, der)| (*der == leaf.as_ref()).then_some(*node))
    }

    /// Serves one H3 request: validates auth headers, enforces suspension
    /// and body caps, routes through the peer handler, and answers. Every
    /// failure is a status code, never a hang; unknown routes and decode
    /// faults refuse with backoff-safe statuses.
    #[allow(clippy::too_many_lines)]
    async fn serve_h3_request(
        &self,
        bound: Option<NodeId>,
        conn: &Connection,
        resolver: H3Resolver,
    ) {
        let inner = &self.transport.inner;
        let Ok((request, mut stream)) = resolver.resolve_request().await else {
            return;
        };
        let headers = request.headers();
        let parse = |name: &str| -> Option<String> {
            headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned)
        };
        let Some((cluster, node, incarnation, caps)) = parse(HDR_CLUSTER)
            .zip(parse(HDR_NODE))
            .zip(parse(HDR_INCARNATION))
            .zip(parse(HDR_CAPS))
            .map(|(((cluster, node), incarnation), caps)| (cluster, node, incarnation, caps))
        else {
            serve_refuse(
                &mut stream,
                http::StatusCode::BAD_REQUEST,
                "missing kivi headers",
            )
            .await;
            return;
        };
        let Some((cluster, node, incarnation, caps)) = cluster
            .parse::<u128>()
            .ok()
            .zip(node.parse::<u64>().ok())
            .zip(incarnation.parse::<u64>().ok())
            .zip(caps.parse::<u64>().ok())
            .map(|(((cluster, node), incarnation), caps)| (cluster, node, incarnation, caps))
        else {
            serve_refuse(
                &mut stream,
                http::StatusCode::BAD_REQUEST,
                "bad kivi headers",
            )
            .await;
            return;
        };
        // Certificate-bound connections must claim their own node.
        if bound.is_some_and(|bound| bound.as_u64() != node) {
            serve_refuse(&mut stream, http::StatusCode::FORBIDDEN, "node mismatch").await;
            return;
        }
        let from = NodeId::from_u64(node);
        if inner
            .links
            .get(&from)
            .is_some_and(|link| link.suspended.load(Ordering::SeqCst))
        {
            serve_refuse(
                &mut stream,
                http::StatusCode::SERVICE_UNAVAILABLE,
                "suspended",
            )
            .await;
            return;
        }
        let validated = {
            let table = inner.incarnations.lock().await;
            match crate::peer::validate_peer_headers(
                cluster,
                node,
                incarnation,
                caps,
                &inner.local,
                &table,
            ) {
                Ok(validated) => validated,
                Err(error) => {
                    serve_refuse(&mut stream, http::StatusCode::FORBIDDEN, &error.to_string())
                        .await;
                    return;
                }
            }
        };
        {
            let mut table = inner.incarnations.lock().await;
            table.observe(validated.node, validated.incarnation);
        }
        if let Some(link) = inner.links.get(&from) {
            *link.incarnation.lock().await = Some(incarnation);
        }
        // Only Raft POST bodies and empty bulk GETs exist; cap everything.
        let path = request.uri().path().to_owned();
        let is_bulk = path.starts_with("/_kivi/immutable/");
        let cap = if is_bulk {
            MAX_BULK_BODY_BYTES
        } else {
            MAX_RAFT_BODY_BYTES
        };
        let mut body = Vec::new();
        loop {
            let chunk = match stream.recv_data().await {
                Ok(Some(chunk)) => chunk,
                Ok(None) => break,
                Err(_) => {
                    serve_refuse(
                        &mut stream,
                        http::StatusCode::BAD_REQUEST,
                        "request body unreadable",
                    )
                    .await;
                    return;
                }
            };
            if !drain_chunk(&mut body, chunk, cap) {
                serve_refuse(
                    &mut stream,
                    http::StatusCode::PAYLOAD_TOO_LARGE,
                    "request body exceeds cap",
                )
                .await;
                return;
            }
        }
        let decoded = match crate::peer::decode_h3_request(&path, &body) {
            Ok(decoded) => decoded,
            Err(error) => {
                serve_refuse(
                    &mut stream,
                    http::StatusCode::NOT_FOUND,
                    &format!("unknown peer route: {error}"),
                )
                .await;
                return;
            }
        };
        let group = ConsensusGroupId::of_tablet(kivi_types::TabletId::from_u64(decoded.tablet));
        // First validated request of a lane marks inbound connectivity
        // (the dialer opens one QUIC connection per lane).
        let lane = match &decoded.request {
            PeerRequest::Manifest(_) | PeerRequest::Chunk(_) => Lane::Bulk,
            _ => Lane::Control,
        };
        self.note_inbound_lane(from, lane, conn);
        if let Some(link) = inner.links.get(&from) {
            link.active.fetch_add(1, Ordering::SeqCst);
        }
        inner.active_streams.fetch_add(1, Ordering::SeqCst);
        let answer = inner.handler.handle(from, group, decoded.request).await;
        if let Some(link) = inner.links.get(&from) {
            link.active.fetch_sub(1, Ordering::SeqCst);
        }
        inner.active_streams.fetch_sub(1, Ordering::SeqCst);
        match answer {
            Ok(response) => {
                let media = match &response {
                    PeerResponse::Vote(_) | PeerResponse::PreVote(_) | PeerResponse::Append(_) => {
                        MEDIA_RAFT
                    }
                    PeerResponse::Snapshot(_) => MEDIA_RAFT,
                    PeerResponse::Manifest(_) => MEDIA_MANIFEST,
                    PeerResponse::Chunk(_) => MEDIA_CHUNK,
                };
                let bytes = crate::peer::encode_h3_response(&response);
                let http_response = http::Response::builder()
                    .status(http::StatusCode::OK)
                    .header("content-type", media)
                    .header("content-length", bytes.len().to_string())
                    .body(())
                    .expect("ok response builds");
                if stream.send_response(http_response).await.is_err() {
                    return;
                }
                // Bulk bodies stream in bounded segments (never one giant
                // write call holding the whole value unnecessarily long —
                // 1 MiB chunks stay single-segment in practice).
                for piece in bytes.chunks(256 * 1024) {
                    if stream.send_data(piece.to_vec().into()).await.is_err() {
                        return;
                    }
                }
                let _ = stream.finish().await;
            }
            Err(error) => {
                serve_refuse(
                    &mut stream,
                    http::StatusCode::INTERNAL_SERVER_ERROR,
                    &error.detail,
                )
                .await;
            }
        }
    }
}

/// Sends an H3 refusal response (status + plain-text detail).
async fn serve_refuse(stream: &mut H3ServerStream, status: http::StatusCode, detail: &str) {
    let response = http::Response::builder()
        .status(status)
        .header("content-type", "text/plain")
        .body(())
        .expect("refusal response builds");
    let _ = stream.send_response(response).await;
    let _ = stream.send_data(detail.as_bytes().to_vec().into()).await;
    let _ = stream.finish().await;
}

/// Bounded reconnect nap with doubling.
async fn sleep_capped(backoff: &mut Duration, max: Duration) {
    compio::time::sleep(*backoff).await;
    // Doubling stays far below overflow (delays are capped at `max`,
    // seconds-scale) and the ceiling keeps the bound exact.
    *backoff = (*backoff * 2).min(max).max(Duration::from_millis(1));
}

/// Appends one received segment into `out` without ever exceeding `cap`.
/// Returns `false` when the segment would break the cap (caller refuses).
fn drain_chunk<B>(out: &mut Vec<u8>, mut chunk: B, cap: usize) -> bool
where
    B: bytes::Buf,
{
    while chunk.has_remaining() {
        if out.len() >= cap {
            return false;
        }
        let take = (cap - out.len()).min(chunk.remaining());
        out.extend_from_slice(&chunk.chunk()[..take]);
        chunk.advance(take);
    }
    true
}

/// Bounded client-stream body receive: per-segment accumulation with a
/// hard cap checked before every append, so a hostile peer cannot force
/// a giant allocation.
///
/// # Errors
///
/// Returns `()` when the stream errors or the cap would break.
async fn read_client_capped(stream: &mut H3ClientStream, cap: usize) -> Result<Vec<u8>, ()> {
    let mut out = Vec::new();
    loop {
        let chunk = stream.recv_data().await.map_err(|_| ())?;
        let Some(chunk) = chunk else {
            break;
        };
        if !drain_chunk(&mut out, chunk, cap) {
            return Err(());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::future::Future;
    use std::sync::Arc;
    use std::time::Duration;

    use kivi_types::{ClusterId, NodeId, NodeIncarnation};

    use super::{PeerHandler, PeerTransport, TlsMaterial, TransportConfig, TransportError};
    use crate::peer::{PeerIdentity, PeerRequest, PeerResponse, PeerRpcError, PeerVoteRequest};
    use crate::tls::NodeCert;
    use crate::types::ConsensusGroupId;

    const CLUSTER: u128 = 0x0C10_57E2;

    fn identity(node: u64, incarnation: u64) -> PeerIdentity {
        PeerIdentity {
            cluster: ClusterId::from_u128(CLUSTER),
            node: NodeId::from_u64(node),
            incarnation: NodeIncarnation::from_u64(incarnation),
        }
    }

    /// In-memory sidecar origin plus a fixed vote answer. `hang_once`
    /// pends the next request forever (a `Send`-safe stall: compio timers
    /// are reactor-local and cannot cross the `PeerHandler` boundary) to
    /// force deterministic client timeouts while later requests serve.
    struct Echo {
        vote: crate::peer::PeerVote,
        manifests: HashMap<[u8; 32], Vec<u8>>,
        chunks: HashMap<[u8; 32], Vec<u8>>,
        hang_once: Arc<std::sync::atomic::AtomicBool>,
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
            let manifests = self.manifests.clone();
            let chunks = self.chunks.clone();
            let hang_once = Arc::clone(&self.hang_once);
            Box::pin(async move {
                if hang_once.swap(false, std::sync::atomic::Ordering::SeqCst) {
                    std::future::pending::<()>().await;
                }
                match request {
                    PeerRequest::Vote(request) => {
                        Ok(PeerResponse::Vote(crate::peer::PeerVoteResponse {
                            vote,
                            granted: true,
                            last_log: request.last_log,
                        }))
                    }
                    PeerRequest::Manifest(request) => manifests
                        .get(&request.manifest)
                        .map(|canonical| {
                            PeerResponse::Manifest(crate::peer::PeerManifestResponse {
                                manifest: request.manifest,
                                canonical: canonical.clone(),
                            })
                        })
                        .ok_or_else(|| PeerRpcError {
                            detail: "no such manifest".to_owned(),
                        }),
                    PeerRequest::Chunk(request) => chunks
                        .get(&request.chunk)
                        .map(|bytes| {
                            PeerResponse::Chunk(crate::peer::PeerChunkResponse {
                                chunk: request.chunk,
                                bytes: bytes.clone(),
                            })
                        })
                        .ok_or_else(|| PeerRpcError {
                            detail: "no such chunk".to_owned(),
                        }),
                    PeerRequest::Snapshot(request) => {
                        Ok(PeerResponse::Snapshot(crate::peer::PeerSnapshotResponse {
                            vote: request.vote,
                        }))
                    }
                    _ => Err(PeerRpcError {
                        detail: "echo serves votes, snapshots, and sidecars only".to_owned(),
                    }),
                }
            })
        }
    }

    fn echo() -> Arc<dyn PeerHandler> {
        echo_hang(false)
    }

    fn echo_hang(hang_once: bool) -> Arc<dyn PeerHandler> {
        Arc::new(Echo {
            vote: crate::peer::PeerVote {
                term: 1,
                node: 1,
                committed: true,
            },
            manifests: [([0x11; 32], vec![7u8; 78])].into_iter().collect(),
            chunks: [([0x22; 32], vec![9u8; 1024])].into_iter().collect(),
            hang_once: Arc::new(std::sync::atomic::AtomicBool::new(hang_once)),
        })
    }

    /// Test-only TLS: fresh self-signed certs, no verification (loopback).
    fn insecure_tls(dir: &std::path::Path, node: u64) -> TlsMaterial {
        TlsMaterial {
            cert: NodeCert::load_or_generate(dir, NodeId::from_u64(node))
                .expect("test cert generates"),
            peer_certs: HashMap::new(),
            insecure_skip_verify: true,
        }
    }

    fn block_on<F, T>(future: F) -> T
    where
        F: Future<Output = T>,
    {
        compio::runtime::Runtime::new()
            .expect("compio test runtime")
            .block_on(future)
    }

    /// Probed loopback UDP ports (QUIC has no `TIME_WAIT`; a stolen port
    /// fails the open loudly and the test retries by rerunning).
    fn probe_udp() -> std::net::SocketAddr {
        let socket = std::net::UdpSocket::bind("127.0.0.1:0").expect("probe binds");
        socket.local_addr().expect("probe addr")
    }

    /// Opens two loopback transports sharing one runtime generation each
    /// (each `open` spawns its driver on the calling reactor). Returns the
    /// scratch directory guard alongside so parallel tests never share
    /// certificate paths.
    async fn pair() -> (PeerTransport, PeerTransport, tempfile::TempDir) {
        pair_with(echo(), echo()).await
    }

    async fn pair_with(
        handler_a: Arc<dyn PeerHandler>,
        handler_b: Arc<dyn PeerHandler>,
    ) -> (PeerTransport, PeerTransport, tempfile::TempDir) {
        let addr_a = probe_udp();
        let addr_b = probe_udp();
        let peers: HashMap<NodeId, std::net::SocketAddr> =
            [(NodeId::from_u64(1), addr_a), (NodeId::from_u64(2), addr_b)]
                .into_iter()
                .collect();
        let dir = tempfile::tempdir().expect("scratch");
        let (first, _) = PeerTransport::open(
            TransportConfig::default(),
            identity(1, 1),
            ClusterId::from_u128(CLUSTER),
            &peers,
            insecure_tls(&dir.path().join("a"), 1),
            handler_a,
        )
        .await
        .expect("first opens");
        let (second, _) = PeerTransport::open(
            TransportConfig::default(),
            identity(2, 1),
            ClusterId::from_u128(CLUSTER),
            &peers,
            insecure_tls(&dir.path().join("b"), 2),
            handler_b,
        )
        .await
        .expect("second opens");
        (first, second, dir)
    }

    /// Waits for the waiter's side to observe the peer (outbound dial for
    /// the dialing side, first validated inbound request for the serving
    /// side — so pair this with traffic, not bare dialer-only waits).
    async fn wait_connected(transport: &PeerTransport, peer: NodeId) {
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        loop {
            let stats = transport.stats().await;
            if stats
                .get(&peer)
                .is_some_and(|stats| stats.connected_control || stats.connected_bulk)
            {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "peer {peer} never connected"
            );
            compio::time::sleep(Duration::from_millis(50)).await;
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
            let (first, second, _dir) = pair().await;
            wait_connected(&first, NodeId::from_u64(2)).await;
            let group = ConsensusGroupId::of_tablet(kivi_types::TabletId::from_u64(9));
            let response = first
                .call(
                    NodeId::from_u64(2),
                    group,
                    vote_request(),
                    Duration::from_secs(10),
                )
                .await
                .expect("vote served");
            assert!(matches!(response, PeerResponse::Vote(_)));
            // The server side learns connectivity from validated inbound
            // requests; the reverse RPC exercises on-demand dial from the
            // non-dialing side.
            wait_connected(&second, NodeId::from_u64(1)).await;
            let reverse = second
                .call(
                    NodeId::from_u64(1),
                    group,
                    vote_request(),
                    Duration::from_secs(10),
                )
                .await
                .expect("reverse vote served");
            assert!(matches!(reverse, PeerResponse::Vote(_)));
            first.shutdown().await;
            second.shutdown().await;
        });
    }

    #[test]
    fn bulk_sidecars_stream_over_their_own_connection() {
        block_on(async {
            let (first, second, _dir) = pair().await;
            wait_connected(&first, NodeId::from_u64(2)).await;
            let group = ConsensusGroupId::of_tablet(kivi_types::TabletId::from_u64(9));
            let manifest = first
                .call_bulk(
                    NodeId::from_u64(2),
                    group,
                    PeerRequest::Manifest(crate::peer::PeerManifestRequest {
                        manifest: [0x11; 32],
                    }),
                    Duration::from_secs(10),
                )
                .await
                .expect("manifest served");
            assert!(
                matches!(manifest, PeerResponse::Manifest(ref answer) if answer.canonical == vec![7u8; 78])
            );
            let chunk = first
                .call_bulk(
                    NodeId::from_u64(2),
                    group,
                    PeerRequest::Chunk(crate::peer::PeerChunkRequest { chunk: [0x22; 32] }),
                    Duration::from_secs(10),
                )
                .await
                .expect("chunk served");
            assert!(
                matches!(chunk, PeerResponse::Chunk(ref answer) if answer.bytes == vec![9u8; 1024])
            );
            // Missing objects refuse (retry-safe), never hang.
            let missing = first
                .call_bulk(
                    NodeId::from_u64(2),
                    group,
                    PeerRequest::Chunk(crate::peer::PeerChunkRequest { chunk: [0x33; 32] }),
                    Duration::from_secs(10),
                )
                .await
                .expect_err("missing chunk refuses");
            assert!(matches!(missing, TransportError::RemoteRefused { .. }));
            first.shutdown().await;
            second.shutdown().await;
        });
    }

    #[test]
    fn shutdown_releases_the_port_for_rebind() {
        block_on(async {
            let addr = probe_udp();
            let peers: HashMap<NodeId, std::net::SocketAddr> =
                [(NodeId::from_u64(1), addr)].into_iter().collect();
            let dir = tempfile::tempdir().expect("scratch");
            let open = |node: u64| {
                let peers = peers.clone();
                let dir = dir.path().join(format!("n{node}"));
                async move {
                    PeerTransport::open(
                        TransportConfig::default(),
                        identity(node, 1),
                        ClusterId::from_u128(CLUSTER),
                        &peers,
                        insecure_tls(&dir, node),
                        echo(),
                    )
                    .await
                }
            };
            let (first, bound) = open(1).await.expect("first opens");
            assert_eq!(bound, addr);
            first.shutdown().await;
            // Rebinding the same port must succeed immediately: shutdown
            // acked only after the endpoint shut down and freed the socket.
            let (second, rebound) = open(1).await.expect("rebind opens");
            assert_eq!(rebound, addr);
            second.shutdown().await;
        });
    }

    /// Many simultaneous H3 streams share two connections: Raft RPCs
    /// and chunk GETs multiplex without head-of-line blocking.
    #[test]
    fn concurrent_streams_all_succeed() {
        block_on(async {
            let (first, second, _dir) = pair().await;
            wait_connected(&first, NodeId::from_u64(2)).await;
            let group = ConsensusGroupId::of_tablet(kivi_types::TabletId::from_u64(9));
            let mut tasks = Vec::new();
            for _ in 0..32 {
                let first = first.clone();
                tasks.push(compio::runtime::spawn(async move {
                    first
                        .call(
                            NodeId::from_u64(2),
                            group,
                            vote_request(),
                            Duration::from_secs(15),
                        )
                        .await
                }));
            }
            for _ in 0..8 {
                let first = first.clone();
                tasks.push(compio::runtime::spawn(async move {
                    first
                        .call_bulk(
                            NodeId::from_u64(2),
                            group,
                            PeerRequest::Chunk(crate::peer::PeerChunkRequest { chunk: [0x22; 32] }),
                            Duration::from_secs(15),
                        )
                        .await
                }));
            }
            for task in tasks {
                let result = task.await.expect("task joins");
                assert!(result.is_ok(), "concurrent stream succeeds");
            }
            first.shutdown().await;
            second.shutdown().await;
        });
    }

    /// A wrong-cluster peer is refused at the auth layer (never served,
    /// never a hang): TLS is insecure here, so the Kivi cluster header
    /// carries the verdict.
    #[test]
    fn wrong_cluster_identity_is_refused() {
        block_on(async {
            let addr_a = probe_udp();
            let addr_b = probe_udp();
            let peers: HashMap<NodeId, std::net::SocketAddr> =
                [(NodeId::from_u64(1), addr_a), (NodeId::from_u64(2), addr_b)]
                    .into_iter()
                    .collect();
            let dir = tempfile::tempdir().expect("scratch");
            let foreign = PeerIdentity {
                cluster: ClusterId::from_u128(0xDEAD),
                node: NodeId::from_u64(2),
                incarnation: NodeIncarnation::from_u64(1),
            };
            let (first, _) = PeerTransport::open(
                TransportConfig::default(),
                identity(1, 1),
                ClusterId::from_u128(CLUSTER),
                &peers,
                insecure_tls(&dir.path().join("a"), 1),
                echo(),
            )
            .await
            .expect("first opens");
            // Same addresses, different cluster: the responder's headers
            // name a foreign cluster.
            let (second, _) = PeerTransport::open(
                TransportConfig::default(),
                foreign,
                ClusterId::from_u128(0xDEAD),
                &peers,
                insecure_tls(&dir.path().join("b"), 2),
                echo(),
            )
            .await
            .expect("second opens");
            let group = ConsensusGroupId::of_tablet(kivi_types::TabletId::from_u64(9));
            let error = first
                .call(
                    NodeId::from_u64(2),
                    group,
                    vote_request(),
                    Duration::from_secs(8),
                )
                .await
                .expect_err("foreign cluster refuses");
            assert!(
                matches!(error, TransportError::Unreachable { .. }),
                "wrong cluster is unreachable, got {error:?}"
            );
            first.shutdown().await;
            second.shutdown().await;
        });
    }

    /// Oversized bodies refuse on both sides: the sender checks caps
    /// before transmitting, the receiver caps while accumulating.
    #[test]
    fn oversized_bodies_refuse_without_allocation() {
        block_on(async {
            let (first, second, _dir) = pair().await;
            wait_connected(&first, NodeId::from_u64(2)).await;
            let group = ConsensusGroupId::of_tablet(kivi_types::TabletId::from_u64(9));
            // A 20 MiB snapshot fragment exceeds the 16 MiB Raft cap.
            let big = PeerRequest::Snapshot(crate::peer::PeerSnapshotRequest {
                vote: crate::peer::PeerVote {
                    term: 1,
                    node: 1,
                    committed: true,
                },
                meta: crate::peer::PeerSnapshotMeta {
                    last_log: None,
                    voters: vec![1, 2],
                    nodes: vec![],
                    membership_log: None,
                    checkpoint: vec![],
                },
                transfer: 1,
                offset: 0,
                data: vec![0xAA; 20 * 1024 * 1024],
                done: true,
            });
            let error = first
                .call(NodeId::from_u64(2), group, big, Duration::from_secs(15))
                .await
                .expect_err("oversize refuses");
            // The server guards its cap by resetting the stream, which
            // surfaces as a retryable transport failure (indistinguishable
            // from a network reset by design); either way nothing oversize
            // is served and the lane stays usable.
            assert!(
                matches!(
                    error,
                    TransportError::Unreachable { .. } | TransportError::RemoteRefused { .. }
                ),
                "oversize fails retryably, got {error:?}"
            );
            first.shutdown().await;
            second.shutdown().await;
        });
    }

    /// A starved deadline times out without poisoning the lane: the next
    /// call on the same connection succeeds (stream cancellation works).
    #[test]
    fn tiny_timeout_cancels_without_poisoning_the_lane() {
        block_on(async {
            // The responder hangs forever; a 500 ms deadline starves
            // deterministically even on loopback.
            let (first, second, _dir) = pair_with(echo(), echo_hang(true)).await;
            wait_connected(&first, NodeId::from_u64(2)).await;
            let group = ConsensusGroupId::of_tablet(kivi_types::TabletId::from_u64(9));
            let error = first
                .call(
                    NodeId::from_u64(2),
                    group,
                    vote_request(),
                    Duration::from_millis(500),
                )
                .await
                .expect_err("hang starves the deadline");
            assert_eq!(error, TransportError::Timeout);
            let response = first
                .call(
                    NodeId::from_u64(2),
                    group,
                    vote_request(),
                    Duration::from_secs(10),
                )
                .await
                .expect("lane reusable after cancel");
            assert!(matches!(response, PeerResponse::Vote(_)));
            first.shutdown().await;
            second.shutdown().await;
        });
    }

    /// Shutdown with active streams terminates every in-flight call
    /// (no hang): a hanging responder plus shutdown on both ends still
    /// resolves every pending RPC.
    #[test]
    fn shutdown_with_active_streams_terminates_calls() {
        block_on(async {
            let (first, second, _dir) = pair_with(echo(), echo_hang(true)).await;
            wait_connected(&first, NodeId::from_u64(2)).await;
            let group = ConsensusGroupId::of_tablet(kivi_types::TabletId::from_u64(9));
            let first_clone = first.clone();
            let pending = compio::runtime::spawn(async move {
                first_clone
                    .call(
                        NodeId::from_u64(2),
                        group,
                        vote_request(),
                        Duration::from_secs(30),
                    )
                    .await
            });
            compio::time::sleep(Duration::from_millis(200)).await;
            first.shutdown().await;
            second.shutdown().await;
            let outcome = compio::time::timeout(Duration::from_secs(10), pending).await;
            assert!(
                outcome.is_ok(),
                "in-flight call terminates on shutdown, never hangs"
            );
        });
    }

    #[test]
    fn unknown_peer_fails_fast() {
        block_on(async {
            let (first, second, _dir) = pair().await;
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
            let (first, second, _dir) = pair().await;
            let group = ConsensusGroupId::of_tablet(kivi_types::TabletId::from_u64(9));
            wait_connected(&first, NodeId::from_u64(2)).await;
            first.suspend_peer(NodeId::from_u64(2)).await;
            let error = first
                .call(
                    NodeId::from_u64(2),
                    group,
                    vote_request(),
                    Duration::from_secs(5),
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
                    Duration::from_secs(10),
                )
                .await
                .expect("healed link serves");
            assert!(matches!(response, PeerResponse::Vote(_)));
            first.shutdown().await;
            second.shutdown().await;
        });
    }

    #[test]
    fn pinned_tls_rejects_foreign_certificates() {
        block_on(async {
            let addr_a = probe_udp();
            let addr_b = probe_udp();
            let peers: HashMap<NodeId, std::net::SocketAddr> =
                [(NodeId::from_u64(1), addr_a), (NodeId::from_u64(2), addr_b)]
                    .into_iter()
                    .collect();
            // Node 1 pins a certificate node 2 never presents (wrong trust
            // anchor): the QUIC handshake itself must fail, surfacing as
            // unreachable rather than a hang.
            let dir = tempfile::tempdir().expect("scratch");
            let cert_a = NodeCert::load_or_generate(&dir.path().join("a"), NodeId::from_u64(1))
                .expect("cert a");
            // A certificate the pinned set never names: presenting anything
            // else must fail verification (a distinct key, not a re-read).
            let foreign =
                NodeCert::load_or_generate(&dir.path().join("foreign"), NodeId::from_u64(99))
                    .expect("foreign");
            let _ = foreign;
            let pinned: HashMap<NodeId, Vec<u8>> = [(NodeId::from_u64(2), cert_a.cert_der.clone())]
                .into_iter()
                .collect();
            let (first, _) = PeerTransport::open(
                TransportConfig::default(),
                identity(1, 1),
                ClusterId::from_u128(CLUSTER),
                &peers,
                TlsMaterial {
                    cert: cert_a,
                    peer_certs: pinned,
                    insecure_skip_verify: false,
                },
                echo(),
            )
            .await
            .expect("first opens");
            let dir_b = tempfile::tempdir().expect("scratch");
            let (second, _) = PeerTransport::open(
                TransportConfig::default(),
                identity(2, 1),
                ClusterId::from_u128(CLUSTER),
                &peers,
                insecure_tls(dir_b.path(), 2),
                echo(),
            )
            .await
            .expect("second opens");
            let group = ConsensusGroupId::of_tablet(kivi_types::TabletId::from_u64(9));
            let error = first
                .call(
                    NodeId::from_u64(2),
                    group,
                    vote_request(),
                    Duration::from_secs(8),
                )
                .await
                .expect_err("foreign certificate refuses");
            assert!(
                matches!(
                    error,
                    TransportError::Unreachable { .. } | TransportError::Timeout
                ),
                "TLS mismatch fails cleanly, got {error:?}"
            );
            first.shutdown().await;
            second.shutdown().await;
        });
    }
}
