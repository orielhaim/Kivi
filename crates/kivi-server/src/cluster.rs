//! Replicated cluster mode: one tablet group per process set.
//!
//! This module is the explicit cluster configuration (task AF keeps
//! single-node modes intact; nothing here runs unless `--cluster` is
//! passed). Each process opens one [`ReplicatedNode`]
//! (durable log + Kivi state machine + peer mesh + `OpenRaft`) and serves:
//!
//! * the native Kivi protocol on Tokio (handshake, requests, redirects),
//! * a read-only admin plane with replicated-tablet diagnostics,
//! * an optional RESP edge translating Redis commands into Kivi semantics.
//!
//! No Compio `DataWorker` exists in this process: the isolated Tokio
//! runtime hosts consensus and serving only, so the state libraries stay
//! runtime-agnostic and the single-node path keeps its reactor.
//!
//! ## Native routing contract (tasks N, O)
//!
//! Only the leader proposes. Followers answer `StaleRoute` with a
//! `Redirect` at the leader's native endpoint — the existing client
//! machinery (route cache, same-identity retries, bounded redirect
//! budget) then carries the request to the leader without any protocol
//! change and without exposing `OpenRaft` concepts. Unknown leaders map
//! to retriable `Overloaded` (election in flight: back off and retry).
//!
//! ## RESP behavior (task P)
//!
//! RESP stays an edge adapter: no `MOVED`/`ASK`/hash slots. Commands
//! translate into Kivi semantics and execute against the local replica
//! (reads serve applied state) or its proposal path (writes). A
//! non-leader answers a retryable Redis `BUSY` error; retry-capable
//! clients retry (ideally against the leader endpoint discovered via
//! admin `/v1/tablet`, which names it). Redis clients never learn
//! consensus concepts, and no unsafe fallback ever executes a strong
//! write off-leader.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::get;
use kivi_consensus::{
    ConsensusError, NodeConfig, NodeStatus, ProposeError, ProposeOutcome, ReadError, ReplicatedNode,
};
use kivi_protocol::{
    Capabilities, ClientHello, FrameKind, FrameReader, RedirectInfo, Request, Response,
    ResponseBody, ServerHello, Status, encode_frame,
};
#[cfg(feature = "redis-compat")]
use kivi_state::Operation;
use kivi_state::{DurableOutcome, OpError, OperationResult};
use kivi_tablet::{DirectoryVersion, HashPrefix, PartitionRange};
use kivi_types::{
    ClusterId, MutationIdentity, NamespaceId, NodeId, ReadContract, TabletAuthority, TabletEpoch,
    TabletId, UnixMicros, WorkerId, WriteGuardGeneration,
};
use serde::Serialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

/// Worker identity reported on cluster native connections. The cluster
/// has no per-worker endpoints (one tablet, one replica per node); every
/// connection reports worker `0`, and redirects name worker `0`, so the
/// client's worker validation stays consistent across dials and hints.
const CLUSTER_WORKER: u64 = 0;
/// Directory version reported in hellos and redirects (the cluster has
/// no multi-tablet directory yet; the value only needs to be stable).
const CLUSTER_DIR_VERSION: u64 = 1;
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
    /// Tablet replicated.
    pub tablet: TabletId,
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
}

/// Shared cluster serving state: the replicated node plus the static
/// address maps redirects resolve through.
#[derive(Debug, Clone)]
pub struct ClusterShared {
    /// The replicated node.
    pub node: Arc<ReplicatedNode>,
    /// Native endpoint per node id (redirect targets).
    pub natives: Arc<std::collections::HashMap<NodeId, SocketAddr>>,
    /// This node's native endpoint (advertised in hellos).
    pub native: SocketAddr,
    /// Tablet replicated.
    pub tablet: TabletId,
    /// Namespace served.
    pub namespace: NamespaceId,
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
        tablet: TabletId::from_u64(args.cluster_tablet),
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

/// Builds the static topology: every configured peer with its native
/// redirect target (defaulting to the peer endpoint).
fn build_topology(config: &ClusterServeConfig) -> kivi_consensus::ClusterTopology {
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
        tablets: vec![TabletAssignment {
            tablet: config.tablet,
            replicas: config.peers.iter().map(|(node, _)| *node).collect(),
        }],
    }
}

/// Opens the replicated node: pre-binds the peer listener (with
/// `TIME_WAIT` retries) before recovery, so restarts rebind their
/// static port and the topology other members dial never changes shape
/// across restarts.
async fn open_node(config: &ClusterServeConfig) -> anyhow::Result<Arc<ReplicatedNode>> {
    let topology = build_topology(config);
    let peer_bind = topology
        .node(config.node)
        .map(|descriptor| descriptor.peer)
        .with_context(|| format!("no peer endpoint for node {}", config.node.as_u64()))?;
    let peer_std = bind_with_retry(peer_bind, "peer").await?;
    peer_std
        .set_nonblocking(true)
        .context("peer listener nonblocking")?;
    let node = Arc::new(
        ReplicatedNode::open(NodeConfig {
            data_dir: config.data_dir.clone(),
            namespace: config.namespace,
            tablet: config.tablet,
            authority: TabletAuthority::new(
                config.tablet,
                TabletEpoch::INITIAL,
                WriteGuardGeneration::INITIAL,
            ),
            local: config.node,
            topology,
            segment_target_bytes: config.segment_target_bytes,
            transport: kivi_consensus::TransportConfig::default(),
            peer_listener: Some(peer_std),
        })
        .await
        .context("replicated node failed to open")?,
    );
    Ok(node)
}

/// Runs the cluster server until Ctrl-C: opens the replicated node,
/// serves native, admin, and optional RESP, prints the `KIVI_READY` line
/// with bound addresses, and shuts down orderly.
pub async fn run(config: ClusterServeConfig) -> anyhow::Result<()> {
    let node = open_node(&config).await?;
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
        natives: Arc::new(natives),
        native: native_addr,
        tablet: config.tablet,
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
                ClusterExecutor::new(Arc::clone(&node)),
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
    println!(
        "KIVI_READY native={native_addr} peer={} admin={admin_addr}{redis_part} node={} cluster={} incarnation={}",
        node.peer_addr(),
        node.node().as_u64(),
        node.cluster().as_u128(),
        node.incarnation().as_u64(),
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
        caps: Capabilities::BASE_V1 | Capabilities::DURABLE_MUTATION_DEDUP,
        cluster: shared.node.cluster(),
        node: shared.node.node(),
        incarnation: shared.node.incarnation(),
        worker: WorkerId::from_u64(CLUSTER_WORKER),
        dir_version: DirectoryVersion::from_u64(CLUSTER_DIR_VERSION),
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
                // Chunk-fabric streaming has no replicated substrate yet
                // (task AK): close rather than misserve bulk uploads.
                _ => return,
            }
        }
    }
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
    // The single replicated tablet owns every key in this stage; hints
    // accelerate nothing and authorize nothing, so they are ignored.
    let operation = request.into_operation();
    let now = wall_now();
    if opcode == kivi_protocol::Opcode::GetStream {
        return Some((
            Response {
                status: Status::Unsupported,
                body: ResponseBody::Diagnostic(
                    "streaming reads are unsupported in cluster mode".to_owned(),
                ),
            },
            opcode,
        ));
    }
    if opcode.is_mutating() {
        let identity = identity.map(|client| MutationIdentity { client, ack_floor });
        Some((
            match shared.node.propose(&operation, identity, None, now).await {
                Ok(outcome) => match outcome {
                    ProposeOutcome::Applied { outcome, .. }
                    | ProposeOutcome::Duplicate { outcome }
                    | ProposeOutcome::Read { outcome } => shape_result(opcode, &outcome),
                    ProposeOutcome::Rejected { outcome } => shape_durable(&outcome, opcode),
                },
                Err(error) => shape_propose_error(shared, &error),
            },
            opcode,
        ))
    } else {
        // Native reads are linearizable in cluster mode (task Q): the
        // barrier runs on the leader; followers redirect.
        Some((
            match shared
                .node
                .read(&operation, ReadContract::Latest, now)
                .await
            {
                Ok(outcome) => shape_result(opcode, &outcome),
                Err(error) => shape_read_error(shared, &error),
            },
            opcode,
        ))
    }
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
fn shape_propose_error(shared: &ClusterShared, error: &ProposeError) -> Response {
    match error {
        ProposeError::Consensus(ConsensusError::NotLeader { hint }) => {
            redirect_or_overloaded(shared, hint.leader.map(replica_node))
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
fn shape_read_error(shared: &ClusterShared, error: &ReadError) -> Response {
    match error {
        kivi_consensus::ReadError::Consensus(ConsensusError::NotLeader { hint }) => {
            redirect_or_overloaded(shared, hint.leader.map(replica_node))
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
fn redirect_or_overloaded(shared: &ClusterShared, leader: Option<kivi_types::NodeId>) -> Response {
    let endpoint = leader.and_then(|node| shared.natives.get(&node).copied());
    match endpoint {
        Some(addr) => Response {
            status: Status::StaleRoute,
            body: ResponseBody::Redirect(RedirectInfo {
                dir_version: DirectoryVersion::from_u64(CLUSTER_DIR_VERSION),
                tablet: shared.tablet,
                epoch: TabletEpoch::INITIAL,
                worker: WorkerId::from_u64(CLUSTER_WORKER),
                endpoint: addr.to_string(),
                range: PartitionRange::Hash(HashPrefix::new(0, 0).expect("root range valid")),
            }),
        },
        None => Response {
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

/// Read-only cluster admin plane: replicated-tablet diagnostics for
/// operators and the lab harness (task AE).
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
}

/// Per-peer connection diagnostics.
#[derive(Debug, Clone, Serialize)]
struct PeerDto {
    node: u64,
    connected: bool,
    suspended: bool,
    incarnation: Option<u64>,
    queue_depth: usize,
    queue_bytes: usize,
    reconnects: u64,
    bytes_sent: u64,
    bytes_received: u64,
}

fn tablet_dto(status: &NodeStatus) -> TabletDto {
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
    }
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok", "version": env!("CARGO_PKG_VERSION") }))
}

async fn ready(State(shared): State<ClusterShared>) -> Json<serde_json::Value> {
    let status = shared.node.status().await;
    Json(serde_json::json!({
        "ready": status.healthy,
        "role": status.role.to_string(),
        "leader": status.leader.map(|leader| leader.node().as_u64()),
    }))
}

async fn node_info(State(shared): State<ClusterShared>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "node": shared.node.node().as_u64(),
        "cluster": shared.node.cluster().as_u128(),
        "incarnation": shared.node.incarnation().as_u64(),
        "namespace": shared.namespace.as_u64(),
        "tablet": shared.tablet.as_u64(),
        "mode": "replicated",
        "native": shared.native.to_string(),
    }))
}

async fn tablet(State(shared): State<ClusterShared>) -> Json<TabletDto> {
    Json(tablet_dto(&shared.node.status().await))
}

async fn peers(State(shared): State<ClusterShared>) -> Json<Vec<PeerDto>> {
    let status = shared.node.status().await;
    let mut out: Vec<PeerDto> = status
        .peers
        .iter()
        .map(|(node, stats)| PeerDto {
            node: node.as_u64(),
            connected: stats.connected,
            suspended: stats.suspended,
            incarnation: stats.incarnation,
            queue_depth: stats.queue_depth,
            queue_bytes: stats.queue_bytes,
            reconnects: stats.reconnects,
            bytes_sent: stats.bytes_sent,
            bytes_received: stats.bytes_received,
        })
        .collect();
    out.sort_by_key(|peer| peer.node);
    Json(out)
}

/// Serves the cluster admin plane until aborted.
async fn serve_admin(listener: tokio::net::TcpListener, shared: ClusterShared) {
    let router = axum::Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/v1/node", get(node_info))
        .route("/v1/tablet", get(tablet))
        .route("/v1/peers", get(peers))
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
    node: Arc<ReplicatedNode>,
    handle: tokio::runtime::Handle,
}

#[cfg(feature = "redis-compat")]
impl ClusterExecutor {
    /// Binds the executor to its node (call inside the serving runtime
    /// so the handle is current).
    #[must_use]
    pub fn new(node: Arc<ReplicatedNode>) -> Self {
        Self {
            node,
            handle: tokio::runtime::Handle::current(),
        }
    }

    /// Executes one typed operation against the replicated tablet:
    /// reads serve local applied state, mutations propose anonymously.
    fn execute_inner(&self, op: &Operation) -> Result<OperationResult, kivi_resp::ExecuteError> {
        use kivi_resp::ExecuteError as E;
        // `block_on` from a blocking context: `RespConnection` drains on
        // the blocking pool (`spawn_blocking`), where no async worker
        // runs, so this never re-enters the runtime.
        if op_is_read(op) {
            self.handle
                .block_on(self.node.read(op, ReadContract::Any, wall_now()))
                .map_err(|_| E::Internal)
        } else {
            match self
                .handle
                .block_on(self.node.propose(op, None, None, wall_now()))
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
