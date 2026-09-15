//! Replicated cluster mode: many tablet groups per process set.
//!
//! This module is the explicit cluster configuration. Each process opens
//! one [`ConsensusNode`] — many tablet
//! replicas on fixed Compio workers sharing one H3 peer mesh and one
//! shared WAL writer — and serves:
//!
//! * the native Kivi protocol on Tokio (handshake, requests, redirects),
//! * a read-only admin plane with per-tablet diagnostics,
//! * an optional RESP edge translating Redis commands into Kivi semantics.
//!
//! No single-node `DataWorker` exists in this process: the isolated Tokio
//! runtime hosts consensus fronts and serving only, so the state
//! libraries stay runtime-agnostic and the single-node path keeps its
//! reactor.
//!
//! ## Native routing contract
//!
//! Every operation routes by key: `PartitionHash` → `DirectorySnapshot` →
//! `TabletId` → local replica. Only the tablet's leader proposes.
//! Followers answer `StaleRoute` with a `Redirect` carrying the tablet's
//! real range (plus directory version, tablet, and leader native
//! endpoint) — the existing client machinery (sparse range cache,
//! same-identity retries, bounded redirect budget) then carries the
//! request to the leader without any protocol change and without
//! exposing `OpenRaft` concepts. Different tablets have different
//! leaders; there is no global current leader. Unknown leaders map to
//! retriable `Overloaded` (election in flight: back off and retry).
//!
//! ## RESP behavior
//!
//! RESP stays an edge adapter: no `MOVED`/`ASK`/hash slots. Commands
//! translate into Kivi semantics, route to their tablet like native
//! traffic, and execute against the local replica (reads serve applied
//! state) or its proposal path (writes). A non-leader answers a
//! retryable Redis `BUSY` error; retry-capable clients retry (ideally
//! against the leader endpoint discovered via admin `/v1/tablets`, which
//! names it per tablet). Redis clients never learn consensus concepts,
//! and no unsafe fallback ever executes a strong write off-leader.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use kivi_consensus::{
    ConsensusError, ConsensusGroupId, ConsensusNode, MultiNodeConfig, NodeStatus, ProposeError,
    ProposeOutcome, ReadError,
};
use kivi_protocol::{
    Capabilities, ClientHello, FrameKind, FrameReader, RedirectInfo, Request, Response,
    ResponseBody, ServerHello, Status, StreamAbort, StreamBegin, StreamReady, ValueStreamBegin,
    encode_frame,
};
#[cfg(feature = "redis-compat")]
use kivi_state::Operation;
use kivi_state::{DurableOutcome, OpError, OperationResult};
use kivi_tablet::DirectorySnapshot;
use kivi_types::{
    ClusterId, MutationIdentity, NamespaceId, NodeId, ReadContract, TabletEpoch, TabletId,
    UnixMicros, WorkerId, WriteGuardGeneration,
};
use serde::Serialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

/// Worker identity reported on cluster native connections. The cluster
/// exposes one native endpoint per node (not per consensus worker);
/// every connection reports worker `0`, and redirects name worker `0`,
/// so the client's worker validation stays consistent across dials and
/// hints. Consensus worker assignment is local-physical (which reactor
/// drives a group) and never crosses the client protocol.
const CLUSTER_WORKER: u64 = 0;
/// Chunk staging granularity for cluster uploads (matches the chunk
/// fabric default; bounds per-frame memory and pack records).
const CLUSTER_CHUNK_SIZE: usize = 1_048_576;
/// Maximum `StreamData` payload per frame on cluster uploads (bounded
/// bulk framing; well under the peer and native ceilings).
const CLUSTER_STREAM_MAX_DATA: u32 = 1_048_576;
/// Admin per-request ceiling.
const ADMIN_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
/// Admin body limit.
const MAX_ADMIN_BODY_BYTES: usize = 64 * 1024;

/// Static cluster server configuration (parsed/validated in `main`).
#[derive(Debug, Clone)]
pub struct ClusterServeConfig {
    /// Cluster identity (must match every member).
    pub cluster: ClusterId,
    /// This process's node identity.
    pub node: NodeId,
    /// Tablet count: the static directory tiles the whole hash space with
    /// this many tablets (any nonzero count).
    pub tablet_count: usize,
    /// Consensus worker (Compio reactor) count: tablet groups stripe
    /// across these deterministically.
    pub worker_count: usize,
    /// Namespace served.
    pub namespace: NamespaceId,
    /// Data-directory root.
    pub data_dir: std::path::PathBuf,
    /// Static peer endpoints per node id.
    pub peers: Vec<(NodeId, SocketAddr)>,
    /// Static native endpoints per node id (redirect targets).
    pub natives: Vec<(NodeId, SocketAddr)>,
    /// Native listen address (may be `:0`).
    pub native_bind: SocketAddr,
    /// Admin listen address (may be `:0`).
    pub admin_bind: SocketAddr,
    /// Optional RESP listen address.
    #[cfg(feature = "redis-compat")]
    pub redis_bind: Option<SocketAddr>,
    /// WAL segment rotation target in bytes.
    pub segment_target_bytes: u64,
    /// Maximum native frame in bytes.
    pub max_frame: usize,
    /// Static peer TLS certificates per node id (`node → DER path`).
    pub peer_certs: Vec<(NodeId, std::path::PathBuf)>,
}

/// Shared cluster serving state: the multi-tablet node, its static
/// directory for key → tablet routing, plus the static address maps
/// redirects resolve through.
#[derive(Debug, Clone)]
pub struct ClusterShared {
    /// The multi-tablet consensus node.
    pub node: Arc<ConsensusNode>,
    /// Static tablet directory: every key deterministically maps to
    /// exactly one active tablet (no gaps, no overlaps).
    pub directory: DirectorySnapshot,
    /// Native endpoint per node id (redirect targets).
    pub natives: Arc<std::collections::HashMap<NodeId, SocketAddr>>,
    /// This node's native endpoint (advertised in hellos).
    pub native: SocketAddr,
    /// Namespace served.
    pub namespace: NamespaceId,
}

/// Why key → tablet routing failed (small error enum: the wire
/// `Response` bodies are built at the call sites).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouteError {
    /// The partition hash algorithm is unavailable in this build.
    UnsupportedHash,
    /// No active tablet owns the key (unreachable on a static tiling —
    /// loud internal, never a silent misroute).
    NoTablet,
}

/// Routes key bytes to their tablet through the static directory.
/// Every key maps to exactly one active tablet (the tiling covers the
/// whole hash space).
fn route_key(shared: &ClusterShared, key: &[u8]) -> Result<TabletId, RouteError> {
    use kivi_state::PartitionHasher;
    let hash = PartitionHasher::V1
        .hash(shared.namespace, key)
        .ok_or(RouteError::UnsupportedHash)?;
    shared
        .directory
        .lookup_by_hash(hash)
        .ok_or(RouteError::NoTablet)
}

/// Shapes a routing failure as a wire response.
fn route_error(error: RouteError) -> Response {
    match error {
        RouteError::UnsupportedHash => Response {
            status: Status::InvalidRequest,
            body: ResponseBody::Diagnostic("partition hash unsupported".to_owned()),
        },
        RouteError::NoTablet => Response {
            status: Status::Internal,
            body: ResponseBody::Diagnostic("no tablet owns key".to_owned()),
        },
    }
}

/// Builds the cluster serving configuration from CLI args, failing
/// loudly on missing or incoherent cluster flags. Natives default to
/// the peer endpoints when `--cluster-natives` is absent (single-address
/// deployments); explicit natives let operators advertise dialable
/// client addresses distinct from the mesh.
pub fn run_from_args(args: &super::Args) -> anyhow::Result<()> {
    let Some(data_dir) = args.data_dir.clone() else {
        anyhow::bail!("cluster mode requires --data-dir <path>");
    };
    let Some(cluster) = args.cluster_id else {
        anyhow::bail!("cluster mode requires --cluster-id <u128>");
    };
    let Some(node) = args.node_id else {
        anyhow::bail!("cluster mode requires --node-id <u64>");
    };
    if node == 0 {
        anyhow::bail!("node id 0 is reserved");
    }
    if args.cluster_peers.is_empty() {
        anyhow::bail!("cluster mode requires --cluster-peers node=host:port,...");
    }
    // Natives are required explicitly: defaulting them to the peer
    // endpoints would redirect native clients into the peer mesh (wrong
    // protocol, failed handshakes). Peer and native listeners always
    // need distinct ports.
    if args.cluster_natives.is_empty() {
        anyhow::bail!("cluster mode requires --cluster-natives node=host:port,...");
    }
    let natives = args.cluster_natives.clone();
    let config = ClusterServeConfig {
        cluster: ClusterId::from_u128(cluster),
        node: NodeId::from_u64(node),
        tablet_count: args.cluster_tablets,
        worker_count: args.cluster_workers,
        namespace: NamespaceId::from_u64(1),
        data_dir,
        peers: args
            .cluster_peers
            .iter()
            .map(|(id, addr)| (NodeId::from_u64(*id), *addr))
            .collect(),
        natives: natives
            .iter()
            .map(|(id, addr)| (NodeId::from_u64(*id), *addr))
            .collect(),
        native_bind: args.cluster_native,
        admin_bind: args.admin,
        #[cfg(feature = "redis-compat")]
        redis_bind: args.redis_listen,
        segment_target_bytes: args.wal_segment_target,
        max_frame: args.max_frame,
        peer_certs: args
            .cluster_peer_certs
            .iter()
            .map(|(id, path)| (NodeId::from_u64(*id), path.clone()))
            .collect(),
    };
    // The server's Tokio runtime drives the consensus front (whose
    // futures stay `Send` across runtimes) and the Tokio-native edges
    // only. Consensus itself runs on its own Compio owner thread inside
    // `ReplicatedNode` — no Tokio exists in `kivi-consensus`.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("kivi-server")
        .build()
        .context("server runtime failed to start")?;
    runtime.block_on(run(config))
}

/// Binds one address with retries through `TIME_WAIT` release (restarts
/// rebind their static ports; a freshly killed socket may briefly hold
/// the port on some platforms). Failing to bind after the window is a
/// real conflict and fails loudly.
async fn bind_with_retry(addr: SocketAddr, what: &str) -> anyhow::Result<std::net::TcpListener> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    loop {
        match std::net::TcpListener::bind(addr) {
            Ok(listener) => return Ok(listener),
            Err(error) => {
                if tokio::time::Instant::now() >= deadline {
                    anyhow::bail!("{what} bind failed on {addr}: {error}");
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

/// Builds the static directory: the whole hash space tiled into
/// `tablet_count` tablets (validated transitions only; see
/// [`DirectorySnapshot::static_tiles`]).
fn build_directory(config: &ClusterServeConfig) -> anyhow::Result<DirectorySnapshot> {
    DirectorySnapshot::static_tiles(
        config.namespace,
        config.tablet_count,
        TabletEpoch::INITIAL,
        WriteGuardGeneration::INITIAL,
    )
    .context("static tablet directory failed to build")
}

/// Builds the static topology: every configured peer with its native
/// redirect target, plus one assignment per directory tablet replicating
/// on every static node (independent Raft leadership per tablet).
fn build_topology(
    config: &ClusterServeConfig,
    tablets: &[TabletId],
) -> kivi_consensus::ClusterTopology {
    use kivi_consensus::{ClusterTopology, NodeDescriptor, TabletAssignment};

    ClusterTopology {
        cluster: config.cluster,
        nodes: config
            .peers
            .iter()
            .map(|(node, peer)| {
                let native = config
                    .natives
                    .iter()
                    .find_map(|(id, addr)| (*id == *node).then_some(*addr))
                    .unwrap_or(*peer);
                NodeDescriptor {
                    node: *node,
                    peer: *peer,
                    native,
                }
            })
            .collect(),
        tablets: tablets
            .iter()
            .map(|tablet| TabletAssignment {
                tablet: *tablet,
                replicas: config.peers.iter().map(|(node, _)| *node).collect(),
            })
            .collect(),
    }
}

/// Opens the multi-tablet node. The QUIC endpoint binds its UDP socket
/// inside the consensus mesh thread (no pre-bound listener, no
/// `TIME_WAIT` dance — QUIC has no TCP listener backlog); a bind
/// conflict fails loudly so the harness retries formation. Peer
/// certificates come from `--cluster-peer-certs`; without them the node
/// opens only when `KIVI_INSECURE_PEER_TLS=1` (lab tests).
async fn open_node(
    config: &ClusterServeConfig,
    directory: &DirectorySnapshot,
) -> anyhow::Result<Arc<ConsensusNode>> {
    let tablets: Vec<TabletId> = directory
        .tablets()
        .iter()
        .filter(|tablet| tablet.state().is_writable())
        .map(kivi_tablet::TabletDescriptor::id)
        .collect();
    let topology = build_topology(config, &tablets);
    let mut peer_certs = std::collections::HashMap::new();
    for (node, path) in &config.peer_certs {
        let der = std::fs::read(path)
            .with_context(|| format!("peer cert for node {} unreadable", node.as_u64()))?;
        peer_certs.insert(*node, der);
    }
    let insecure_peer_tls = std::env::var("KIVI_INSECURE_PEER_TLS").is_ok_and(|value| value == "1");
    if peer_certs.is_empty() && !insecure_peer_tls {
        anyhow::bail!(
            "peer TLS needs --cluster-peer-certs or KIVI_INSECURE_PEER_TLS=1 (lab tests only)"
        );
    }
    // Correctness probe (lab tests only): `KIVI_DISABLE_PREFLIGHT=1`
    // forces the append-gate fallback path, proving preflight is only an
    // optimization. Production always preflights.
    let preflight_enabled =
        !std::env::var("KIVI_DISABLE_PREFLIGHT").is_ok_and(|value| value == "1");
    let node = Arc::new(
        ConsensusNode::open(MultiNodeConfig {
            data_dir: config.data_dir.clone(),
            namespace: config.namespace,
            tablets,
            local: config.node,
            topology,
            segment_target_bytes: config.segment_target_bytes,
            transport: kivi_consensus::TransportConfig::default(),
            peer_certs,
            insecure_peer_tls,
            preflight_enabled,
            worker_count: config.worker_count,
            durability: kivi_consensus::SharedDurabilityConfig::default(),
        })
        .await
        .context("replicated node failed to open")?,
    );
    Ok(node)
}

/// Runs the cluster server until Ctrl-C: builds the static directory,
/// opens the multi-tablet node, serves native, admin, and optional RESP,
/// prints the `KIVI_READY` line with bound addresses, and shuts down
/// orderly.
pub async fn run(config: ClusterServeConfig) -> anyhow::Result<()> {
    let directory = build_directory(&config)?;
    let node = open_node(&config, &directory).await?;
    let native_std = bind_with_retry(config.native_bind, "native").await?;
    native_std
        .set_nonblocking(true)
        .context("native listener nonblocking")?;
    let native_listener =
        tokio::net::TcpListener::from_std(native_std).context("native listener adopt")?;
    let native_addr = native_listener
        .local_addr()
        .context("native listener address")?;
    let admin_std = bind_with_retry(config.admin_bind, "admin").await?;
    admin_std
        .set_nonblocking(true)
        .context("admin listener nonblocking")?;
    let admin_listener =
        tokio::net::TcpListener::from_std(admin_std).context("admin listener adopt")?;
    let admin_addr = admin_listener
        .local_addr()
        .context("admin listener address")?;
    let natives: std::collections::HashMap<NodeId, SocketAddr> =
        config.natives.iter().copied().collect();
    let shared = ClusterShared {
        node: Arc::clone(&node),
        directory,
        natives: Arc::new(natives),
        native: native_addr,
        namespace: config.namespace,
    };
    // Native serving (Tokio tasks per connection).
    let native = {
        let shared = shared.clone();
        tokio::spawn(async move { serve_native(native_listener, shared, config.max_frame).await })
    };
    // Read-only admin plane.
    let admin = {
        let shared = shared.clone();
        tokio::spawn(async move { serve_admin(admin_listener, shared).await })
    };
    // Optional RESP edge.
    #[cfg(feature = "redis-compat")]
    let (resp_admin, resp_shutdown) = match config.redis_bind {
        None => (None, None),
        Some(addr) => {
            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .with_context(|| format!("redis bind failed on {addr}"))?;
            let endpoint = listener.local_addr().context("redis listener address")?;
            let stats = Arc::new(super::resp::RespStats::default());
            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
            tokio::spawn(super::resp::serve_with(
                listener,
                ClusterExecutor::new(
                    Arc::clone(&node),
                    shared.directory.clone(),
                    config.namespace,
                ),
                Arc::clone(&stats),
                shutdown_rx,
            ));
            (
                Some(super::resp::RespAdmin {
                    endpoint,
                    namespace: config.namespace.as_u64(),
                    stats,
                }),
                Some(shutdown_tx),
            )
        }
    };
    #[cfg(feature = "redis-compat")]
    let redis_part = resp_admin
        .as_ref()
        .map_or(String::new(), |admin| format!(" redis={}", admin.endpoint));
    #[cfg(not(feature = "redis-compat"))]
    let redis_part = String::new();
    // NOTE: the token is `consensus_workers=`, never `workers=`: the
    // harness readiness parser treats `workers=<a>,<b>` as the single-node
    // native list and prefers it over `native=` — a bare `workers=2`
    // would parse as the native endpoint "2".
    println!(
        "KIVI_READY native={native_addr} peer={} admin={admin_addr}{redis_part} node={} cluster={} incarnation={} tablets={} consensus_workers={} dir_version={}",
        node.peer_addr(),
        node.node().as_u64(),
        node.cluster().as_u128(),
        node.incarnation().as_u64(),
        node.tablets().len(),
        node.worker_count(),
        shared.directory.version().as_u64(),
    );
    let _ = std::io::Write::flush(&mut std::io::stdout());
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::warn!(%error, "shutdown signal watch failed; exiting");
    }
    #[cfg(feature = "redis-compat")]
    drop(resp_shutdown);
    #[cfg(feature = "redis-compat")]
    drop(resp_admin);
    native.abort();
    admin.abort();
    node.shutdown().await;
    tracing::info!("kivi-server (cluster) stopped");
    Ok(())
}

/// Serves native connections until aborted: handshake, then one request
/// loop per connection.
async fn serve_native(listener: tokio::net::TcpListener, shared: ClusterShared, max_frame: usize) {
    loop {
        let Ok((socket, _)) = listener.accept().await else {
            continue;
        };
        if socket.set_nodelay(true).is_err() {
            continue;
        }
        let shared = shared.clone();
        tokio::spawn(async move {
            serve_native_conn(socket, shared, max_frame).await;
        });
    }
}

/// Serves one native connection: exactly one `ClientHello`, one
/// `ServerHello`, then requests until EOF or violation.
#[allow(clippy::too_many_lines)]
async fn serve_native_conn(socket: tokio::net::TcpStream, shared: ClusterShared, max_frame: usize) {
    let (mut reader, mut writer) = socket.into_split();
    let mut frames = FrameReader::new(max_frame);
    let mut buf = vec![0u8; 64 * 1024];
    // Handshake first (mirrors the single-node negotiation exactly so
    // the same clients connect unchanged).
    let hello = loop {
        let count = match reader.read(&mut buf).await {
            Ok(count) if count > 0 => count,
            _ => return,
        };
        match frames.push(&buf[..count]) {
            Ok(mut parsed) => {
                if let Some(frame) = parsed.pop() {
                    break frame;
                }
            }
            Err(_) => return,
        }
    };
    if hello.kind != FrameKind::ClientHello {
        return;
    }
    let Ok(client) = ClientHello::decode(&hello.payload) else {
        return;
    };
    if !client.required_caps.unknown_required().is_empty() || client.max_frame == 0 {
        return;
    }
    let negotiated = (client.max_frame as usize).min(max_frame);
    let reply = ServerHello {
        major: kivi_protocol::PROTOCOL_MAJOR,
        minor: kivi_protocol::PROTOCOL_MINOR,
        caps: Capabilities::BASE_V1
            | Capabilities::DURABLE_MUTATION_DEDUP
            | Capabilities::STREAMING,
        cluster: shared.node.cluster(),
        node: shared.node.node(),
        incarnation: shared.node.incarnation(),
        worker: WorkerId::from_u64(CLUSTER_WORKER),
        dir_version: shared.directory.version(),
        max_frame: u32::try_from(negotiated).unwrap_or(u32::MAX),
        endpoints: vec![(
            WorkerId::from_u64(CLUSTER_WORKER),
            shared.native.to_string(),
        )],
    };
    if writer
        .write_all(&encode_frame(FrameKind::ServerHello, 0, &reply.encode()))
        .await
        .is_err()
    {
        return;
    }
    frames = FrameReader::new(negotiated);
    let mut uploads: std::collections::HashMap<u64, ClusterUpload> =
        std::collections::HashMap::new();
    loop {
        let count = match reader.read(&mut buf).await {
            Ok(count) if count > 0 => count,
            _ => return,
        };
        let Ok(parsed) = frames.push(&buf[..count]) else {
            return;
        };
        for frame in parsed {
            match frame.kind {
                FrameKind::Request => {
                    // `GetStream` streams `ValueStream*` frames under the
                    // request id; all other opcodes answer with one
                    // `Response` (chunked roots resolved via sidecars).
                    if is_get_stream(&frame.payload) {
                        if handle_get_stream(&shared, &mut writer, &frame.payload, frame.request_id)
                            .await
                            .is_err()
                        {
                            return;
                        }
                        continue;
                    }
                    let Some((response, opcode)) = handle_request(&shared, &frame.payload).await
                    else {
                        // Malformed requests close the connection (same
                        // contract as the single-node edge).
                        return;
                    };
                    if writer
                        .write_all(&encode_frame(
                            FrameKind::Response,
                            frame.request_id,
                            &response.encode(opcode),
                        ))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                FrameKind::Ping => {
                    if writer
                        .write_all(&encode_frame(FrameKind::Pong, frame.request_id, &[]))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                FrameKind::StreamBegin => {
                    if handle_stream_begin(
                        &shared,
                        &mut writer,
                        &mut uploads,
                        &frame.payload,
                        frame.request_id,
                        negotiated,
                    )
                    .await
                    .is_err()
                    {
                        return;
                    }
                }
                FrameKind::StreamData => {
                    if handle_stream_data(&shared, &mut uploads, &frame.payload, frame.request_id)
                        .await
                        .is_err()
                    {
                        let _ = writer
                            .write_all(&encode_frame(
                                FrameKind::StreamAbort,
                                frame.request_id,
                                &StreamAbort {
                                    reason: "upload overflow or unknown stream".to_owned(),
                                }
                                .encode(),
                            ))
                            .await;
                        uploads.remove(&frame.request_id);
                    }
                }
                FrameKind::StreamCommit => {
                    if handle_stream_commit(&shared, &mut writer, &mut uploads, frame.request_id)
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                FrameKind::StreamAbort => {
                    uploads.remove(&frame.request_id);
                }
                _ => return,
            }
        }
    }
}

/// Whether a request payload names the `GetStream` opcode (delivery
/// differs: `ValueStream*` frames instead of one `Response`).
fn is_get_stream(payload: &[u8]) -> bool {
    Request::decode(payload).is_ok_and(|request| request.opcode == kivi_protocol::Opcode::GetStream)
}

/// One cluster upload assembling on its connection task.
///
/// Chunks stage incrementally (one pack record per full 1 MiB piece, no
/// per-chunk sync); the manifest + one barrier complete on commit. Staged
/// chunks pin implicitly via content addressing until the commit proposes;
/// a dropped connection leaves harmless orphans for deferred GC.
struct ClusterUpload {
    key: Vec<u8>,
    identity: Option<kivi_types::RequestIdentity>,
    ack_floor: kivi_types::RequestSeq,
    declared: Option<u64>,
    received: u64,
    buf: Vec<u8>,
    chunks: Vec<(kivi_types::ChunkId, u64)>,
}

/// Handles `StreamBegin`: validates, registers the upload, answers
/// `StreamReady` (or `StreamAbort` on definite rejection, pre-commit).
async fn handle_stream_begin(
    shared: &ClusterShared,
    writer: &mut tokio::net::tcp::OwnedWriteHalf,
    uploads: &mut std::collections::HashMap<u64, ClusterUpload>,
    payload: &[u8],
    stream: u64,
    negotiated: usize,
) -> anyhow::Result<()> {
    use tokio::io::AsyncWriteExt;
    let abort = |reason: &str| {
        encode_frame(
            FrameKind::StreamAbort,
            stream,
            &StreamAbort {
                reason: reason.to_owned(),
            }
            .encode(),
        )
    };
    let Ok(begin) = StreamBegin::decode(payload) else {
        writer.write_all(&abort("malformed stream-begin")).await?;
        return Ok(());
    };
    if begin.namespace != shared.namespace {
        writer.write_all(&abort("unknown namespace")).await?;
        return Ok(());
    }
    if begin.key.is_empty() {
        writer.write_all(&abort("empty key")).await?;
        return Ok(());
    }
    if begin
        .total_len
        .is_some_and(|total| total > kivi_protocol::MAX_STREAM_UPLOAD_BYTES)
    {
        writer
            .write_all(&abort("upload exceeds the stream bound"))
            .await?;
        return Ok(());
    }
    if uploads.contains_key(&stream) {
        writer.write_all(&abort("duplicate stream id")).await?;
        return Ok(());
    }
    uploads.insert(
        stream,
        ClusterUpload {
            key: begin.key,
            identity: begin.identity,
            ack_floor: begin.ack_floor,
            declared: begin.total_len,
            received: 0,
            buf: Vec::new(),
            chunks: Vec::new(),
        },
    );
    let max_data = u32::try_from(negotiated.saturating_sub(256))
        .unwrap_or(CLUSTER_STREAM_MAX_DATA)
        .clamp(1, CLUSTER_STREAM_MAX_DATA);
    writer
        .write_all(&encode_frame(
            FrameKind::StreamReady,
            stream,
            &StreamReady { max_data }.encode(),
        ))
        .await?;
    Ok(())
}

/// Handles `StreamData`: appends, stages full 1 MiB pieces (no sync yet,
/// so residency stays ~1 chunk, never the full 64 MiB).
/// Returns `Err` on overflow/unknown stream (caller aborts the stream).
async fn handle_stream_data(
    shared: &ClusterShared,
    uploads: &mut std::collections::HashMap<u64, ClusterUpload>,
    payload: &[u8],
    stream: u64,
) -> anyhow::Result<()> {
    let Some(upload) = uploads.get_mut(&stream) else {
        anyhow::bail!("unknown stream");
    };
    if u32::try_from(payload.len()).unwrap_or(u32::MAX) > CLUSTER_STREAM_MAX_DATA {
        anyhow::bail!("stream data exceeds negotiated max");
    }
    upload.received = upload
        .received
        .checked_add(payload.len() as u64)
        .ok_or_else(|| anyhow::anyhow!("upload overflow"))?;
    if upload.received > kivi_protocol::MAX_STREAM_UPLOAD_BYTES {
        anyhow::bail!("upload exceeds the stream bound");
    }
    if upload.declared.is_some_and(|total| upload.received > total) {
        anyhow::bail!("upload exceeds its declared total");
    }
    upload.buf.extend_from_slice(payload);
    // Incremental staging: residency stays ~1 chunk, never the full value.
    drain_upload_pieces(shared, upload).await?;
    Ok(())
}

/// Stages full buffered pieces for an upload (shared by data/commit).
async fn drain_upload_pieces(
    shared: &ClusterShared,
    upload: &mut ClusterUpload,
) -> anyhow::Result<()> {
    while upload.buf.len() >= CLUSTER_CHUNK_SIZE {
        let piece: Vec<u8> = upload.buf.drain(..CLUSTER_CHUNK_SIZE).collect();
        let id = kivi_codec::integrity::chunk_id(
            kivi_types::SecurityDomainId::from_u64(shared.namespace.as_u64()),
            &piece,
        );
        // Stage without sync; the commit barriers once. Synchronous
        // verification happens inside `stage_chunk` (id recompute).
        shared
            .node
            .sidecar()
            .stage_chunk(id, piece)
            .await
            .map_err(|error| anyhow::anyhow!("stage chunk: {error}"))?;
        upload.chunks.push((id, CLUSTER_CHUNK_SIZE as u64));
    }
    Ok(())
}

/// Handles `StreamCommit`: stages the tail, builds the manifest, syncs
/// once, proposes the tiny root, and answers `Response::Stored` under the
/// stream id (or `StreamAbort` on failure, pre-apply).
#[allow(clippy::too_many_lines)]
async fn handle_stream_commit(
    shared: &ClusterShared,
    writer: &mut tokio::net::tcp::OwnedWriteHalf,
    uploads: &mut std::collections::HashMap<u64, ClusterUpload>,
    stream: u64,
) -> anyhow::Result<()> {
    use tokio::io::AsyncWriteExt;
    let Some(mut upload) = uploads.remove(&stream) else {
        return Ok(());
    };
    let fail = |reason: String| {
        encode_frame(
            FrameKind::StreamAbort,
            stream,
            &StreamAbort { reason }.encode(),
        )
    };
    if upload
        .declared
        .is_some_and(|total| upload.received != total)
    {
        writer
            .write_all(&fail(
                "upload length mismatches its declared total".to_owned(),
            ))
            .await?;
        return Ok(());
    }
    // Stage full pieces buffered so far (commit path stages the tail
    // below, so every byte is chunked exactly once).
    if let Err(error) = drain_upload_pieces(shared, &mut upload).await {
        writer.write_all(&fail(error.to_string())).await?;
        return Ok(());
    }
    // Tail piece (possibly empty when the value aligns exactly).
    if !upload.buf.is_empty() {
        let tail: Vec<u8> = std::mem::take(&mut upload.buf);
        let id = kivi_codec::integrity::chunk_id(
            kivi_types::SecurityDomainId::from_u64(shared.namespace.as_u64()),
            &tail,
        );
        if let Err(error) = shared.node.sidecar().stage_chunk(id, tail.clone()).await {
            writer
                .write_all(&fail(format!("stage tail: {error}")))
                .await?;
            return Ok(());
        }
        upload.chunks.push((id, tail.len() as u64));
    }
    // Empty value: no chunks, empty manifest (total 0).
    let domain = kivi_types::SecurityDomainId::from_u64(shared.namespace.as_u64());
    let entries: Vec<kivi_chunk::ChunkEntry> = upload
        .chunks
        .iter()
        .map(|(id, len)| kivi_chunk::ChunkEntry { id: *id, len: *len })
        .collect();
    let total = upload.received;
    let (manifest_obj, manifest) = match kivi_chunk::build_manifest(
        domain,
        kivi_chunk::Chunking::DEFAULT,
        kivi_chunk::ChunkCodecId::NONE,
        total,
        entries,
    ) {
        Ok(built) => built,
        Err(error) => {
            writer
                .write_all(&fail(format!("build manifest: {error}")))
                .await?;
            return Ok(());
        }
    };
    let mut canonical = Vec::new();
    manifest_obj.encode_canonical(&mut canonical);
    if let Err(error) = shared
        .node
        .sidecar()
        .stage_manifest(manifest, canonical)
        .await
    {
        writer
            .write_all(&fail(format!("stage manifest: {error}")))
            .await?;
        return Ok(());
    }
    if let Err(error) = shared.node.sidecar().sync().await {
        writer
            .write_all(&fail(format!("sidecar sync: {error}")))
            .await?;
        return Ok(());
    }
    // Pin for the proposal lifetime; ownership transfers to live state on
    // commit (pins release on reply/dedup-hit paths inside propose).
    {
        let chunk_ids: Vec<kivi_types::ChunkId> = upload.chunks.iter().map(|(id, _)| *id).collect();
        let mut pins = shared.node.sidecar().pins().lock().await;
        pins.pin_root(manifest, &chunk_ids);
    }
    let operation = kivi_state::Operation::SetChunked {
        key: kivi_state::Key::new(upload.key.clone()),
        manifest,
        logical_len: total,
    };
    let identity = upload.identity.map(|client| MutationIdentity {
        client,
        ack_floor: upload.ack_floor,
    });
    let now = wall_now();
    // Streamed values route like any other write: the key selects the
    // tablet whose group commits the tiny root (preflight then moves the
    // staged sidecars before that root proposes).
    let tablet = match route_key(shared, &upload.key) {
        Ok(tablet) => tablet,
        Err(error) => {
            writer
                .write_all(&encode_frame(
                    FrameKind::Response,
                    stream,
                    &route_error(error).encode(kivi_protocol::Opcode::Set),
                ))
                .await?;
            return Ok(());
        }
    };
    let response = match shared
        .node
        .propose(tablet, &operation, identity, None, now)
        .await
    {
        Ok(outcome) => match outcome {
            ProposeOutcome::Applied { outcome, .. }
            | ProposeOutcome::Duplicate { outcome }
            | ProposeOutcome::Read { outcome } => {
                // Release the proposal pin: live state now references the
                // root (or the dedup hit answered from retained state).
                {
                    let chunk_ids: Vec<kivi_types::ChunkId> =
                        upload.chunks.iter().map(|(id, _)| *id).collect();
                    let mut pins = shared.node.sidecar().pins().lock().await;
                    pins.unpin_root(&manifest, &chunk_ids);
                }
                shape_result(kivi_protocol::Opcode::Set, &outcome)
            }
            ProposeOutcome::Rejected { outcome } => {
                {
                    let chunk_ids: Vec<kivi_types::ChunkId> =
                        upload.chunks.iter().map(|(id, _)| *id).collect();
                    let mut pins = shared.node.sidecar().pins().lock().await;
                    pins.unpin_root(&manifest, &chunk_ids);
                }
                shape_durable(&outcome, kivi_protocol::Opcode::Set)
            }
        },
        Err(error) => {
            {
                let chunk_ids: Vec<kivi_types::ChunkId> =
                    upload.chunks.iter().map(|(id, _)| *id).collect();
                let mut pins = shared.node.sidecar().pins().lock().await;
                pins.unpin_root(&manifest, &chunk_ids);
            }
            shape_propose_error(shared, tablet, &error)
        }
    };
    writer
        .write_all(&encode_frame(
            FrameKind::Response,
            stream,
            &response.encode(kivi_protocol::Opcode::Set),
        ))
        .await?;
    Ok(())
}

/// Handles `GetStream`: linearizable read on the leader, then streams
/// `ValueStreamBegin/Data/End` under the request id (or a terminal
/// `Response` for absent/redirect/error, which the client loops on).
#[allow(clippy::too_many_lines)]
async fn handle_get_stream(
    shared: &ClusterShared,
    writer: &mut tokio::net::tcp::OwnedWriteHalf,
    payload: &[u8],
    stream: u64,
) -> anyhow::Result<()> {
    use tokio::io::AsyncWriteExt;
    let Ok(request) = Request::decode(payload) else {
        return Ok(());
    };
    if request.namespace != shared.namespace {
        let response = Response {
            status: Status::InvalidRequest,
            body: ResponseBody::Diagnostic("unknown namespace".to_owned()),
        };
        writer
            .write_all(&encode_frame(
                FrameKind::Response,
                stream,
                &response.encode(kivi_protocol::Opcode::GetStream),
            ))
            .await?;
        return Ok(());
    }
    let operation = request.into_operation();
    let now = wall_now();
    let tablet = match route_key(shared, operation.key().as_bytes()) {
        Ok(tablet) => tablet,
        Err(error) => {
            writer
                .write_all(&encode_frame(
                    FrameKind::Response,
                    stream,
                    &route_error(error).encode(kivi_protocol::Opcode::GetStream),
                ))
                .await?;
            return Ok(());
        }
    };
    let outcome = match shared
        .node
        .read(tablet, &operation, ReadContract::Latest, now)
        .await
    {
        Ok(outcome) => outcome,
        Err(error) => {
            let response = shape_read_error(shared, tablet, &error);
            writer
                .write_all(&encode_frame(
                    FrameKind::Response,
                    stream,
                    &response.encode(kivi_protocol::Opcode::GetStream),
                ))
                .await?;
            return Ok(());
        }
    };
    // Resolve bytes (inline or chunked via sidecars).
    let bytes = match outcome {
        OperationResult::Value(Some(value)) => Some(value.to_vec()),
        OperationResult::Value(None) => None,
        OperationResult::ChunkedValue {
            manifest,
            logical_len,
        } => {
            if let Ok(bytes) = shared
                .node
                .sidecar()
                .read_value(manifest, logical_len)
                .await
            {
                Some(bytes)
            } else {
                let abort = StreamAbort {
                    reason: "chunked value unavailable".to_owned(),
                };
                writer
                    .write_all(&encode_frame(
                        FrameKind::StreamAbort,
                        stream,
                        &abort.encode(),
                    ))
                    .await?;
                return Ok(());
            }
        }
        _ => {
            let response = shape_result(kivi_protocol::Opcode::GetStream, &outcome);
            writer
                .write_all(&encode_frame(
                    FrameKind::Response,
                    stream,
                    &response.encode(kivi_protocol::Opcode::GetStream),
                ))
                .await?;
            return Ok(());
        }
    };
    let Some(bytes) = bytes else {
        let response = Response {
            status: Status::NotFound,
            body: ResponseBody::Diagnostic(String::new()),
        };
        writer
            .write_all(&encode_frame(
                FrameKind::Response,
                stream,
                &response.encode(kivi_protocol::Opcode::GetStream),
            ))
            .await?;
        return Ok(());
    };
    writer
        .write_all(&encode_frame(
            FrameKind::ValueStreamBegin,
            stream,
            &ValueStreamBegin {
                total_len: bytes.len() as u64,
            }
            .encode(),
        ))
        .await?;
    for piece in bytes.chunks(CLUSTER_CHUNK_SIZE) {
        writer
            .write_all(&encode_frame(FrameKind::ValueStreamData, stream, piece))
            .await?;
    }
    writer
        .write_all(&encode_frame(FrameKind::ValueStreamEnd, stream, &[]))
        .await?;
    Ok(())
}

/// Handles one decoded request against the replicated node. Returns
/// `None` for malformed requests (the connection closes, mirroring the
/// single-node edge); otherwise the response with the request's opcode
/// (which the encoding embeds, so decoders never guess shapes).
async fn handle_request(
    shared: &ClusterShared,
    payload: &[u8],
) -> Option<(Response, kivi_protocol::Opcode)> {
    let Ok(request) = Request::decode(payload) else {
        return None;
    };
    let opcode = request.opcode;
    let namespace = request.namespace;
    let identity = request.identity;
    let ack_floor = request.ack_floor;
    if namespace != shared.namespace {
        return Some((
            Response {
                status: Status::InvalidRequest,
                body: ResponseBody::Diagnostic("unknown namespace".to_owned()),
            },
            opcode,
        ));
    }
    // Real multi-tablet routing: every operation maps key → tablet, then
    // proposes/reads against that tablet's group. Request hints accelerate
    // nothing and authorize nothing, so they are ignored; sparse route
    // discovery flows through `StaleRoute` redirects carrying the tablet's
    // real range. `GetStream` never reaches here (the connection loop
    // streams `ValueStream*` frames instead of one `Response`).
    let operation = request.into_operation();
    let tablet = match route_key(shared, operation.key().as_bytes()) {
        Ok(tablet) => tablet,
        Err(error) => return Some((route_error(error), opcode)),
    };
    let now = wall_now();
    if opcode.is_mutating() {
        let identity = identity.map(|client| MutationIdentity { client, ack_floor });
        Some((
            match shared
                .node
                .propose(tablet, &operation, identity, None, now)
                .await
            {
                Ok(outcome) => match outcome {
                    ProposeOutcome::Applied { outcome, .. }
                    | ProposeOutcome::Duplicate { outcome }
                    | ProposeOutcome::Read { outcome } => {
                        resolve_result(shared, opcode, &operation, &outcome).await
                    }
                    ProposeOutcome::Rejected { outcome } => {
                        resolve_durable(shared, &operation, &outcome, opcode).await
                    }
                },
                Err(error) => shape_propose_error(shared, tablet, &error),
            },
            opcode,
        ))
    } else {
        // Native reads are linearizable in cluster mode: the barrier runs
        // on the tablet's leader; followers redirect. Chunked roots
        // resolve via sidecars (same bytes as single-node, never wrong
        // bytes).
        Some((
            match shared
                .node
                .read(tablet, &operation, ReadContract::Latest, now)
                .await
            {
                Ok(outcome) => resolve_result(shared, opcode, &operation, &outcome).await,
                Err(error) => shape_read_error(shared, tablet, &error),
            },
            opcode,
        ))
    }
}

/// Resolves a deterministic outcome, fetching chunked bytes via sidecars
/// when the outcome references immutable state.
async fn resolve_result(
    shared: &ClusterShared,
    opcode: kivi_protocol::Opcode,
    operation: &kivi_state::Operation,
    outcome: &OperationResult,
) -> Response {
    use kivi_state::OperationResult as R;
    match outcome {
        R::ChunkedValue {
            manifest,
            logical_len,
        } => match opcode {
            kivi_protocol::Opcode::Get => {
                if *logical_len > kivi_chunk::policy::MAX_LEGACY_VALUE_BYTES {
                    return Response {
                        status: Status::ValueTooLarge,
                        body: ResponseBody::Diagnostic(
                            "value exceeds legacy GET; use get_stream".to_owned(),
                        ),
                    };
                }
                match shared
                    .node
                    .sidecar()
                    .read_value(*manifest, *logical_len)
                    .await
                {
                    Ok(bytes) => Response {
                        status: Status::Ok,
                        body: ResponseBody::Value(bytes),
                    },
                    Err(_) => Response {
                        status: Status::Internal,
                        body: ResponseBody::Diagnostic("chunked value unavailable".to_owned()),
                    },
                }
            }
            kivi_protocol::Opcode::GetRange => {
                let (offset, len) = match operation {
                    kivi_state::Operation::GetRange { offset, len, .. } => (*offset, *len),
                    _ => (0, u64::MAX),
                };
                match shared
                    .node
                    .sidecar()
                    .read_value(*manifest, *logical_len)
                    .await
                {
                    Ok(bytes) => Response {
                        status: Status::Ok,
                        body: ResponseBody::Value(slice_range(&bytes, offset, len)),
                    },
                    Err(_) => Response {
                        status: Status::Internal,
                        body: ResponseBody::Diagnostic("chunked value unavailable".to_owned()),
                    },
                }
            }
            _ => shape_result(opcode, outcome),
        },
        _ => shape_result(opcode, outcome),
    }
}

/// Resolves a terminal outcome, fetching chunked bytes when completed.
async fn resolve_durable(
    shared: &ClusterShared,
    operation: &kivi_state::Operation,
    outcome: &DurableOutcome,
    opcode: kivi_protocol::Opcode,
) -> Response {
    match outcome {
        DurableOutcome::Completed(result) => {
            resolve_result(shared, opcode, operation, result).await
        }
        _ => shape_durable(outcome, opcode),
    }
}

/// Slices `[offset, offset+len)` clamped to the value (empty past the end).
fn slice_range(value: &[u8], offset: u64, len: u64) -> Vec<u8> {
    let offset = usize::try_from(offset).unwrap_or(usize::MAX);
    let len = usize::try_from(len).unwrap_or(usize::MAX);
    if offset >= value.len() {
        return Vec::new();
    }
    let end = offset.saturating_add(len).min(value.len());
    value[offset..end].to_vec()
}

/// Shapes a deterministic outcome like the single-node edge (same bodies
/// for the same results, so clients cannot tell the modes apart).
fn shape_result(opcode: kivi_protocol::Opcode, outcome: &OperationResult) -> Response {
    use kivi_state::OperationResult as R;
    match outcome {
        R::Value(Some(value)) => Response {
            status: Status::Ok,
            body: ResponseBody::Value(value.to_vec()),
        },
        R::Value(None) | R::Counter(None) | R::Expiry(None) | R::Length(None) => Response {
            status: Status::NotFound,
            body: ResponseBody::Diagnostic(String::new()),
        },
        R::Length(Some(len)) => Response {
            status: Status::Ok,
            body: ResponseBody::Length(*len),
        },
        R::ConditionalSet { applied, version } => {
            // `SetRange` restages internally but promises `Stored`.
            if *applied && opcode == kivi_protocol::Opcode::SetRange {
                match version {
                    Some(version) => Response {
                        status: Status::Ok,
                        body: ResponseBody::Stored {
                            version: version.as_u64(),
                        },
                    },
                    None => Response {
                        status: Status::Internal,
                        body: ResponseBody::Diagnostic(
                            "conditional set applied without a version".to_owned(),
                        ),
                    },
                }
            } else {
                Response {
                    status: Status::Ok,
                    body: ResponseBody::ConditionalSet {
                        applied: *applied,
                        version: version.map_or(0, kivi_state::ObjectVersion::as_u64),
                    },
                }
            }
        }
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
        // Chunked roots never replicate in this stage; reaching the wire
        // unresolved is a serving bug, answered loudly, never wrong bytes.
        R::ChunkedValue { .. } => Response {
            status: Status::Internal,
            body: ResponseBody::Diagnostic("chunked value reached the wire unresolved".to_owned()),
        },
    }
}

/// Shapes a terminal (non-replicating) outcome: completed results shape
/// normally under the requesting opcode; rejections shape as their
/// stable statuses.
fn shape_durable(outcome: &DurableOutcome, opcode: kivi_protocol::Opcode) -> Response {
    match outcome {
        DurableOutcome::Completed(result) => shape_result(opcode, result),
        DurableOutcome::Rejected(OpError::WrongType { .. }) => Response {
            status: Status::WrongType,
            body: ResponseBody::Diagnostic("wrong type".to_owned()),
        },
        DurableOutcome::Rejected(OpError::CounterOverflow) => Response {
            status: Status::CounterOverflow,
            body: ResponseBody::Diagnostic("overflow".to_owned()),
        },
        DurableOutcome::Rejected(OpError::StaleRangeBase) => Response {
            status: Status::InvalidRequest,
            body: ResponseBody::Diagnostic("range base changed; retry".to_owned()),
        },
        DurableOutcome::VersionExhausted => Response {
            status: Status::VersionExhausted,
            body: ResponseBody::Diagnostic("version exhausted".to_owned()),
        },
    }
}

/// Extracts the node from a replica identity (named function so call
/// sites satisfy `Option::map` without redundant closures).
fn replica_node(replica: kivi_consensus::ReplicaId) -> NodeId {
    replica.node()
}

/// Shapes proposal failures: routing becomes redirects/retriable
/// statuses (never `OpenRaft` concepts on the wire), validation becomes
/// stable semantic statuses.
fn shape_propose_error(shared: &ClusterShared, tablet: TabletId, error: &ProposeError) -> Response {
    match error {
        ProposeError::Consensus(ConsensusError::NotLeader { hint }) => {
            redirect_or_overloaded(shared, tablet, hint.leader.map(replica_node))
        }
        ProposeError::Consensus(ConsensusError::LeaderUnknown) => Response {
            status: Status::Overloaded,
            body: ResponseBody::Diagnostic("leader unknown; retry".to_owned()),
        },
        ProposeError::Consensus(ConsensusError::Unavailable { .. }) => Response {
            status: Status::Internal,
            body: ResponseBody::Diagnostic("consensus unavailable".to_owned()),
        },
        ProposeError::Consensus(ConsensusError::ShuttingDown) => Response {
            status: Status::Internal,
            body: ResponseBody::Diagnostic("replica shutting down".to_owned()),
        },
        ProposeError::Expired => Response {
            status: Status::DedupExpired,
            body: ResponseBody::Diagnostic("mutation identity expired".to_owned()),
        },
        ProposeError::Overloaded => Response {
            status: Status::SessionOverloaded,
            body: ResponseBody::Diagnostic("session outcome window exhausted".to_owned()),
        },
        ProposeError::Unsupported { reason } => Response {
            status: Status::Unsupported,
            body: ResponseBody::Diagnostic(reason.clone()),
        },
        ProposeError::Op(OpError::WrongType { .. }) => Response {
            status: Status::WrongType,
            body: ResponseBody::Diagnostic("wrong type".to_owned()),
        },
        ProposeError::Op(OpError::CounterOverflow) => Response {
            status: Status::CounterOverflow,
            body: ResponseBody::Diagnostic("overflow".to_owned()),
        },
        ProposeError::Op(OpError::StaleRangeBase) => Response {
            status: Status::InvalidRequest,
            body: ResponseBody::Diagnostic("range base changed; retry".to_owned()),
        },
        // Forward-compatibility: future proposal failures fail closed as
        // internal rather than mis-shaping on the wire.
        _ => Response {
            status: Status::Internal,
            body: ResponseBody::Diagnostic("proposal failed".to_owned()),
        },
    }
}

/// Shapes read failures with the same routing contract as proposals.
fn shape_read_error(shared: &ClusterShared, tablet: TabletId, error: &ReadError) -> Response {
    match error {
        kivi_consensus::ReadError::Consensus(ConsensusError::NotLeader { hint }) => {
            redirect_or_overloaded(shared, tablet, hint.leader.map(replica_node))
        }
        kivi_consensus::ReadError::Consensus(ConsensusError::LeaderUnknown) => Response {
            status: Status::Overloaded,
            body: ResponseBody::Diagnostic("leader unknown; retry".to_owned()),
        },
        kivi_consensus::ReadError::Consensus(ConsensusError::Unavailable { .. }) => Response {
            status: Status::Internal,
            body: ResponseBody::Diagnostic("consensus unavailable".to_owned()),
        },
        kivi_consensus::ReadError::Consensus(ConsensusError::ShuttingDown) => Response {
            status: Status::Internal,
            body: ResponseBody::Diagnostic("replica shutting down".to_owned()),
        },
        kivi_consensus::ReadError::Op(OpError::WrongType { .. }) => Response {
            status: Status::WrongType,
            body: ResponseBody::Diagnostic("wrong type".to_owned()),
        },
        kivi_consensus::ReadError::Op(OpError::CounterOverflow) => Response {
            status: Status::CounterOverflow,
            body: ResponseBody::Diagnostic("overflow".to_owned()),
        },
        kivi_consensus::ReadError::Op(OpError::StaleRangeBase) => Response {
            status: Status::InvalidRequest,
            body: ResponseBody::Diagnostic("range base changed; retry".to_owned()),
        },
        kivi_consensus::ReadError::NotARead => Response {
            status: Status::InvalidRequest,
            body: ResponseBody::Diagnostic("not a read operation".to_owned()),
        },
        kivi_consensus::ReadError::CoverageTimeout => Response {
            status: Status::Overloaded,
            body: ResponseBody::Diagnostic("applied coverage timed out; retry".to_owned()),
        },
        // Forward-compatibility: future read failures fail closed as
        // internal rather than mis-shaping on the wire.
        _ => Response {
            status: Status::Internal,
            body: ResponseBody::Diagnostic("read failed".to_owned()),
        },
    }
}

/// Answers `NotLeader` as `StaleRoute` + leader redirect when the leader
/// endpoint is known, else retriable `Overloaded` (election in flight).
/// The redirect carries the tablet's real range (plus directory version
/// and tablet), so the client's sparse range cache learns per-tablet
/// routes: different tablets have different leaders.
fn redirect_or_overloaded(
    shared: &ClusterShared,
    tablet: TabletId,
    leader: Option<kivi_types::NodeId>,
) -> Response {
    let endpoint = leader.and_then(|node| shared.natives.get(&node).copied());
    let range = shared
        .directory
        .get(tablet)
        .map(|descriptor| descriptor.range().clone());
    match (endpoint, range) {
        (Some(addr), Some(range)) => Response {
            status: Status::StaleRoute,
            body: ResponseBody::Redirect(RedirectInfo {
                dir_version: shared.directory.version(),
                tablet,
                epoch: TabletEpoch::INITIAL,
                worker: WorkerId::from_u64(CLUSTER_WORKER),
                endpoint: addr.to_string(),
                range,
            }),
        },
        _ => Response {
            status: Status::Overloaded,
            body: ResponseBody::Diagnostic("leader unknown; retry".to_owned()),
        },
    }
}

/// Wall-clock Unix microseconds (client-boundary capture only; replicas
/// apply the leader's materialized stamp, never their own clock).
fn wall_now() -> UnixMicros {
    let elapsed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    UnixMicros::from_micros(elapsed.as_micros().try_into().unwrap_or(u64::MAX))
}

/// Read-only cluster admin plane: per-tablet diagnostics for operators
/// and the lab harness.
#[derive(Debug, Clone, Serialize)]
struct TabletDto {
    group: u64,
    replica: u64,
    role: String,
    term: u64,
    leader: Option<u64>,
    last_log: u64,
    committed: u64,
    applied: u64,
    applied_commit: u64,
    snapshot: Option<u64>,
    purged: Option<u64>,
    healthy: bool,
    detail: Option<String>,
    /// Tablet hash-range prefix bits as 32 lowercase hex chars (the full
    /// `u128` tile origin). `None` for non-hash layouts (never in static
    /// clusters, which always tile the hash space).
    range_bits: Option<String>,
    /// Tablet hash-range prefix length (addresses covered:
    /// `2^(128-len)`).
    range_len: Option<u8>,
}

/// Per-peer QUIC/H3 diagnostics (control + bulk connections).
#[derive(Debug, Clone, Serialize)]
struct PeerDto {
    node: u64,
    connected_control: bool,
    connected_bulk: bool,
    suspended: bool,
    incarnation: Option<u64>,
    reconnects: u64,
    control_requests: u64,
    bulk_requests: u64,
    control_bytes_sent: u64,
    control_bytes_received: u64,
    bulk_bytes_sent: u64,
    bulk_bytes_received: u64,
    rtt_ms: Option<u64>,
    active_streams: u64,
}

/// Immutable sidecar replication diagnostics.
#[derive(Debug, Clone, Serialize)]
struct SidecarDto {
    bulk_bytes_sent: u64,
    bulk_bytes_received: u64,
    manifest_requests: u64,
    chunk_requests: u64,
    cache_hits: u64,
    cache_misses: u64,
    deduped_bytes: u64,
    bulk_errors: u64,
    verification_failures: u64,
    pending_gated: u64,
    inflight: u64,
}

fn tablet_dto(status: &NodeStatus, directory: &DirectorySnapshot) -> TabletDto {
    // Static clusters always tile the hash space; non-hash layouts (a
    // future concern) encode as nulls, never wrong ranges.
    let (range_bits, range_len) = directory
        .get(TabletId::from_u64(status.group.tablet().as_u64()))
        .and_then(|descriptor| match descriptor.range() {
            kivi_tablet::PartitionRange::Hash(prefix) => Some((
                Some(format!("{:032x}", prefix.bits())),
                Some(prefix.prefix_len()),
            )),
            kivi_tablet::PartitionRange::Ordered(_) => None,
        })
        .unwrap_or((None, None));
    TabletDto {
        group: status.group.tablet().as_u64(),
        replica: status.replica.node().as_u64(),
        role: status.role.to_string(),
        term: status.term.get(),
        leader: status.leader.map(|leader| replica_node(leader).as_u64()),
        last_log: status.last_log.get(),
        committed: status.committed.get(),
        applied: status.applied.get(),
        applied_commit: status.applied_commit.as_u64(),
        snapshot: status.snapshot.map(kivi_consensus::ConsensusLogIndex::get),
        purged: status.purged.map(kivi_consensus::ConsensusLogIndex::get),
        healthy: status.healthy,
        detail: status.detail.clone(),
        range_bits,
        range_len,
    }
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok", "version": env!("CARGO_PKG_VERSION") }))
}

async fn ready(State(shared): State<ClusterShared>) -> Json<serde_json::Value> {
    let statuses = shared.node.status_all().await;
    // Multi-tablet readiness: process alive is not enough. A node is
    // usable once its peer mesh started, every configured local replica
    // loaded, and every group healthy — partial failure is reported
    // openly (counts), never hidden behind one boolean.
    let total = statuses.len();
    let healthy = statuses.iter().filter(|status| status.healthy).count();
    let leaders_known = statuses
        .iter()
        .filter(|status| status.leader.is_some())
        .count();
    Json(serde_json::json!({
        "ready": total > 0 && healthy == total,
        "initialized": true,
        "transport": true,
        "membership_loaded": true,
        "tablets_total": total,
        "tablets_healthy": healthy,
        "leaders_known": leaders_known,
        "election_pending": leaders_known != total,
        "leader_known": leaders_known == total,
        "healthy": healthy == total,
    }))
}

async fn node_info(State(shared): State<ClusterShared>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "node": shared.node.node().as_u64(),
        "cluster": shared.node.cluster().as_u128(),
        "incarnation": shared.node.incarnation().as_u64(),
        "namespace": shared.namespace.as_u64(),
        "tablets": shared.node.tablets().iter().map(|tablet| tablet.as_u64()).collect::<Vec<_>>(),
        "tablet_count": shared.node.tablets().len(),
        "workers": shared.node.worker_count(),
        "dir_version": shared.directory.version().as_u64(),
        "mode": "replicated",
        "cluster_mode": "replicated",
        "peer": shared.node.peer_addr().to_string(),
        "native": shared.native.to_string(),
        "client_endpoint": shared.native.to_string(),
        "peer_endpoint": shared.node.peer_addr().to_string(),
    }))
}

/// Per-tablet diagnostics for every local group, sorted by tablet.
async fn tablets(State(shared): State<ClusterShared>) -> Json<Vec<TabletDto>> {
    let mut out: Vec<TabletDto> = shared
        .node
        .status_all()
        .await
        .iter()
        .map(|status| tablet_dto(status, &shared.directory))
        .collect();
    out.sort_by_key(|tablet| tablet.group);
    Json(out)
}

/// Single-tablet compatibility: serves the only group when exactly one
/// tablet is configured; multi-tablet nodes answer `400` directing
/// callers at `/v1/tablets` (no silent partial view).
async fn tablet(State(shared): State<ClusterShared>) -> (StatusCode, Json<serde_json::Value>) {
    let mut statuses = shared.node.status_all().await;
    if statuses.len() != 1 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "multi-tablet node; use /v1/tablets" })),
        );
    }
    let status = statuses.pop().expect("exactly one status");
    match serde_json::to_value(tablet_dto(&status, &shared.directory)) {
        Ok(value) => (StatusCode::OK, Json(value)),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": error.to_string() })),
        ),
    }
}

async fn peers(State(shared): State<ClusterShared>) -> Json<Vec<PeerDto>> {
    // Peer stats are node-wide (one shared mesh): any group's status
    // carries them; fall back to empty when no group exists (unreachable:
    // at least one tablet is always configured).
    let statuses = shared.node.status_all().await;
    let Some(status) = statuses.first() else {
        return Json(Vec::new());
    };
    let mut out: Vec<PeerDto> = status
        .peers
        .iter()
        .map(|(node, stats)| PeerDto {
            node: node.as_u64(),
            connected_control: stats.connected_control,
            connected_bulk: stats.connected_bulk,
            suspended: stats.suspended,
            incarnation: stats.incarnation,
            reconnects: stats.reconnects,
            control_requests: stats.control_requests,
            bulk_requests: stats.bulk_requests,
            control_bytes_sent: stats.control_bytes_sent,
            control_bytes_received: stats.control_bytes_received,
            bulk_bytes_sent: stats.bulk_bytes_sent,
            bulk_bytes_received: stats.bulk_bytes_received,
            rtt_ms: stats.rtt_ms,
            active_streams: stats.active_streams,
        })
        .collect();
    out.sort_by_key(|peer| peer.node);
    Json(out)
}

async fn sidecar(State(shared): State<ClusterShared>) -> Json<SidecarDto> {
    // Sidecar metrics are node-wide (one shared content store).
    let snapshot = shared.node.sidecar().metrics().snapshot();
    Json(SidecarDto {
        bulk_bytes_sent: snapshot.bulk_bytes_sent,
        bulk_bytes_received: snapshot.bulk_bytes_received,
        manifest_requests: snapshot.manifest_requests,
        chunk_requests: snapshot.chunk_requests,
        cache_hits: snapshot.cache_hits,
        cache_misses: snapshot.cache_misses,
        deduped_bytes: snapshot.deduped_bytes,
        bulk_errors: snapshot.bulk_errors,
        verification_failures: snapshot.verification_failures,
        pending_gated: snapshot.pending_gated,
        inflight: snapshot.inflight,
    })
}

/// Shared physical durability metrics: physical batches, records/bytes
/// per batch, barriers, groups per batch, queue depth, barrier latency.
/// Evidence that cross-group batching actually happens.
async fn durability(State(shared): State<ClusterShared>) -> Json<serde_json::Value> {
    let metrics = shared.node.durability_metrics();
    Json(serde_json::json!({
        "physical_batches": metrics.physical_batches,
        "records": metrics.records,
        "bytes": metrics.bytes,
        "barriers": metrics.barriers,
        "group_appearances": metrics.group_appearances,
        "max_groups_per_batch": metrics.max_groups_per_batch,
        "max_records_per_batch": metrics.max_records_per_batch,
        "queue_depth": metrics.queue_depth,
        "queue_bytes": metrics.queue_bytes,
        "barrier_latency_us": metrics.barrier_latency_us,
    }))
}

/// Sidecar preflight metrics: attempts, quorum-ready rounds, append-gate
/// fallbacks, ready replies, total latency.
async fn preflight(State(shared): State<ClusterShared>) -> Json<serde_json::Value> {
    let metrics = shared.node.preflight_metrics();
    Json(serde_json::json!({
        "attempts": metrics.attempts,
        "quorum_ready": metrics.quorum_ready,
        "fallbacks": metrics.fallbacks,
        "ready_replies": metrics.ready_replies,
        "latency_us": metrics.latency_us,
    }))
}

/// Suspends one peer link (partition test hook and drain tooling).
/// Body: `{ "node": <u64> }`. Bounded: the transport drops the link and
/// backs off with bounded reconnects; the peer is NOT removed from
/// membership, so healing via `/v1/peers/resume` restores replication.
async fn suspend_peer(
    State(shared): State<ClusterShared>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Some(node) = body.get("node").and_then(serde_json::Value::as_u64) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "body must be {\"node\": <u64>}" })),
        );
    };
    if node == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "node id 0 is reserved" })),
        );
    }
    shared.node.suspend_peer(NodeId::from_u64(node)).await;
    (
        StatusCode::OK,
        Json(serde_json::json!({ "suspended": node })),
    )
}

/// Resumes a suspended peer link.
async fn resume_peer(
    State(shared): State<ClusterShared>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Some(node) = body.get("node").and_then(serde_json::Value::as_u64) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "body must be {\"node\": <u64>}" })),
        );
    };
    shared.node.resume_peer(NodeId::from_u64(node)).await;
    (StatusCode::OK, Json(serde_json::json!({ "resumed": node })))
}

/// Seals a checkpoint snapshot and purges one group's Raft log through
/// its base. Used by operators and the lab harness to force snapshot
/// catch-up: a lagging follower behind the purge point recovers via
/// snapshot install, then resumes log replication. Body: `{ "tablet":
/// <u64> }`, optional when exactly one tablet is configured (otherwise
/// required — snapshots stay per tablet, never one giant node image).
async fn snapshot(
    State(shared): State<ClusterShared>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let tablet = if let Some(id) = body.get("tablet").and_then(serde_json::Value::as_u64) {
        TabletId::from_u64(id)
    } else {
        let tablets = shared.node.tablets();
        if tablets.len() != 1 {
            return (
                StatusCode::BAD_REQUEST,
                Json(
                    serde_json::json!({ "error": "multi-tablet node; body must be {\"tablet\": <u64>}" }),
                ),
            );
        }
        tablets[0]
    };
    match shared
        .node
        .snapshot_and_purge(ConsensusGroupId::of_tablet(tablet))
        .await
    {
        Ok(base) => (
            StatusCode::OK,
            Json(serde_json::json!({ "snapshot": base.get(), "tablet": tablet.as_u64() })),
        ),
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        ),
    }
}

/// Serves the cluster admin plane until aborted.
async fn serve_admin(listener: tokio::net::TcpListener, shared: ClusterShared) {
    let router = axum::Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/v1/node", get(node_info))
        .route("/v1/tablet", get(tablet))
        .route("/v1/tablets", get(tablets))
        .route("/v1/peers", get(peers))
        .route("/v1/sidecar", get(sidecar))
        .route("/v1/durability", get(durability))
        .route("/v1/preflight", get(preflight))
        .route("/v1/peers/suspend", post(suspend_peer))
        .route("/v1/peers/resume", post(resume_peer))
        .route("/v1/snapshot", post(snapshot))
        .layer(
            tower::ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(RequestBodyLimitLayer::new(MAX_ADMIN_BODY_BYTES))
                .layer(TimeoutLayer::with_status_code(
                    StatusCode::SERVICE_UNAVAILABLE,
                    ADMIN_REQUEST_TIMEOUT,
                )),
        )
        .with_state(shared);
    if let Err(error) = axum::serve(listener, router).await {
        tracing::warn!(%error, "cluster admin server failed");
    }
}

/// RESP executor over the replicated node (task P edge adapter).
/// Anonymous identity throughout (Redis carries no sessions); leader
/// routing failures surface as retryable `Overloaded` (`BUSY`, safe to
/// retry with backoff). The `ExecuteError` surface carries no dynamic
/// leader endpoint (fixed renderings by design), so cluster RESP
/// clients discover the leader out of band (admin `/v1/tablet` names
/// it) or simply retry: the error is retryable, never `MOVED`/`ASK`
/// (there is no Redis Cluster here), and Redis clients never learn
/// consensus concepts.
#[cfg(feature = "redis-compat")]
#[derive(Debug, Clone)]
pub struct ClusterExecutor {
    node: Arc<ConsensusNode>,
    directory: DirectorySnapshot,
    namespace: NamespaceId,
    handle: tokio::runtime::Handle,
}

#[cfg(feature = "redis-compat")]
impl ClusterExecutor {
    /// Binds the executor to its node (call inside the serving runtime
    /// so the handle is current).
    #[must_use]
    pub fn new(
        node: Arc<ConsensusNode>,
        directory: DirectorySnapshot,
        namespace: NamespaceId,
    ) -> Self {
        Self {
            node,
            directory,
            namespace,
            handle: tokio::runtime::Handle::current(),
        }
    }

    /// Routes one operation to its tablet through the static directory
    /// (same key → tablet mapping as the native edge).
    fn route(&self, op: &Operation) -> Result<TabletId, kivi_resp::ExecuteError> {
        use kivi_resp::ExecuteError as E;
        use kivi_state::PartitionHasher;
        let hash = PartitionHasher::V1
            .hash(self.namespace, op.key().as_bytes())
            .ok_or(E::Internal)?;
        self.directory.lookup_by_hash(hash).ok_or(E::Internal)
    }

    /// Executes one typed operation against its tablet's replica:
    /// reads serve local applied state, mutations propose anonymously.
    fn execute_inner(&self, op: &Operation) -> Result<OperationResult, kivi_resp::ExecuteError> {
        use kivi_resp::ExecuteError as E;
        // `block_on` from a blocking context: `RespConnection` drains on
        // the blocking pool (`spawn_blocking`), where no async worker
        // runs, so this never re-enters the runtime.
        let tablet = self.route(op)?;
        if op_is_read(op) {
            self.handle
                .block_on(self.node.read(tablet, op, ReadContract::Any, wall_now()))
                .map_err(|_| E::Internal)
        } else {
            match self
                .handle
                .block_on(self.node.propose(tablet, op, None, None, wall_now()))
            {
                Ok(outcome) => match outcome {
                    ProposeOutcome::Applied { outcome, .. }
                    | ProposeOutcome::Duplicate { outcome }
                    | ProposeOutcome::Read { outcome } => Ok(outcome),
                    ProposeOutcome::Rejected { outcome } => match outcome {
                        DurableOutcome::Completed(result) => Ok(result),
                        DurableOutcome::Rejected(OpError::WrongType { .. }) => Err(E::WrongType),
                        DurableOutcome::Rejected(_) | DurableOutcome::VersionExhausted => {
                            Err(E::Rejected)
                        }
                    },
                },
                Err(ProposeError::Op(OpError::WrongType { .. })) => Err(E::WrongType),
                Err(ProposeError::Op(_)) => Err(E::Rejected),
                Err(ProposeError::Unsupported { .. }) => Err(E::Internal),
                // Routing, dedup-window, and future proposal failures
                // surface as retryable, never silent.
                Err(_) => Err(E::Overloaded),
            }
        }
    }
}

/// Whether an operation serves as a read (local applied state, no
/// proposal) on the RESP edge.
#[cfg(feature = "redis-compat")]
fn op_is_read(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Get { .. }
            | Operation::Exists { .. }
            | Operation::CounterGet { .. }
            | Operation::GetExpiry { .. }
            | Operation::GetRange { .. }
            | Operation::BytesLength { .. }
    )
}

#[cfg(feature = "redis-compat")]
impl kivi_resp::Executor for ClusterExecutor {
    fn execute(&self, op: &Operation) -> Result<OperationResult, kivi_resp::ExecuteError> {
        self.execute_inner(op)
    }

    fn now_micros(&self) -> u64 {
        wall_now().as_micros()
    }
}
