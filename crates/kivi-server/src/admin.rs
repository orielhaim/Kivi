//! Admin/control HTTP plane: Tokio + Axum, read-only.
//!
//! Architectural boundary: this module is the ONLY place Tokio, Axum,
//! Tower, or Serde exist. The Compio native data plane (`kivi-engine`,
//! `LiveTablet`, `DataWorker` execution) never sees them — Axum interacts
//! with the engine exclusively through [`kivi_engine::AdminHandle`]
//! (immutable snapshots plus the bounded control channel), and Serde/JSON
//! exist only for these admin DTOs, never for the Mutation IR, the native
//! protocol, or any durable format.
//!
//! Process shape:
//!
//! ```text
//! kivi-server
//! ├── Data Plane: pinned Compio DataWorkers, native Kivi protocol
//! └── Admin Plane (here): Tokio multi-thread runtime, Axum + Tower
//! ```
//!
//! Endpoints are read-only (`GET`); mutation/admin-control operations are a
//! later stage. The listener binds loopback by default (see `--admin`).

use std::net::SocketAddr;
use std::time::Duration;

use anyhow::Context;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::get;
use kivi_engine::AdminHandle;
use kivi_types::{ClusterId, NodeId, NodeIncarnation};
use serde::Serialize;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

/// Bodies above this are rejected before routing (the API is read-only;
/// legitimate requests are a few hundred bytes).
const MAX_ADMIN_BODY_BYTES: usize = 64 * 1024;
/// Per-request ceiling: admin reads are local snapshots, never slow work.
const ADMIN_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// Shared admin state: node identity (operator-configured, same values the
/// engine serves in native handshakes) plus the engine query handle.
#[derive(Debug, Clone)]
pub struct AdminState {
    /// This node's identity.
    pub node: NodeId,
    /// This cluster's identity.
    pub cluster: ClusterId,
    /// Read-only engine query handle (`Send + Sync`, Tokio-safe).
    pub engine: AdminHandle,
}

/// Liveness: the process is up and the admin plane serves.
#[derive(Debug, Clone, Serialize)]
struct HealthDto {
    status: &'static str,
    version: &'static str,
}

/// Readiness: the data plane is constructed with workers and tablets.
#[derive(Debug, Clone, Serialize)]
struct ReadyDto {
    ready: bool,
    workers: usize,
    tablets: usize,
    directory_version: u64,
}

/// Node identity and topology summary.
#[derive(Debug, Clone, Serialize)]
struct NodeDto {
    node: u64,
    cluster: u128,
    incarnation: u64,
    namespace: u64,
    workers: usize,
    tablets: usize,
    directory_version: u64,
}

/// One worker: endpoint, owned tablets, and arrival-path counters (`None`
/// once the worker has exited).
#[derive(Debug, Clone, Serialize)]
struct WorkerDto {
    worker: u64,
    endpoint: Option<String>,
    tablets: Vec<u64>,
    channel_ops: Option<u64>,
    direct_ops: Option<u64>,
}

/// One tablet: directory facts plus its placed owner, if any.
#[derive(Debug, Clone, Serialize)]
struct TabletDto {
    tablet: u64,
    state: String,
    epoch: u64,
    guard: u64,
    range: String,
    owner: Option<u64>,
}

async fn health() -> Json<HealthDto> {
    Json(HealthDto {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

async fn ready(State(state): State<AdminState>) -> Json<ReadyDto> {
    let routing = state.engine.routing();
    Json(ReadyDto {
        ready: true,
        workers: state.engine.worker_count(),
        tablets: routing
            .directory()
            .tablets()
            .iter()
            .filter(|tablet| tablet.state().is_writable())
            .count(),
        directory_version: routing.version().as_u64(),
    })
}

async fn node(State(state): State<AdminState>) -> Json<NodeDto> {
    let routing = state.engine.routing();
    Json(NodeDto {
        node: state.node.as_u64(),
        cluster: state.cluster.as_u128(),
        incarnation: NodeIncarnation::INITIAL.as_u64(),
        namespace: state.engine.namespace().as_u64(),
        workers: state.engine.worker_count(),
        tablets: routing
            .directory()
            .tablets()
            .iter()
            .filter(|tablet| tablet.state().is_writable())
            .count(),
        directory_version: routing.version().as_u64(),
    })
}

async fn workers(State(state): State<AdminState>) -> Result<Json<Vec<WorkerDto>>, StatusCode> {
    // Control-channel rendezvous blocks: keep it off the async executor.
    let engine = state.engine.clone();
    let metrics = tokio::task::spawn_blocking(move || engine.worker_metrics_snapshot())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let routing = state.engine.routing();
    let by_worker: std::collections::HashMap<u64, (Option<u64>, Option<u64>)> = metrics
        .into_iter()
        .map(|(id, m)| (id.as_u64(), (Some(m.channel_ops), Some(m.direct_ops))))
        .collect();
    let mut out = Vec::new();
    for index in 0..state.engine.worker_count() {
        let id = u64::try_from(index).unwrap_or(u64::MAX);
        let worker = kivi_types::WorkerId::from_u64(id);
        let tablets: Vec<u64> = routing
            .directory()
            .tablets()
            .iter()
            .filter(|tablet| routing.placement().worker_of(tablet.id()) == Some(worker))
            .map(|tablet| tablet.id().as_u64())
            .collect();
        let (channel_ops, direct_ops) = by_worker
            .get(&worker.as_u64())
            .copied()
            .unwrap_or((None, None));
        out.push(WorkerDto {
            worker: worker.as_u64(),
            endpoint: state
                .engine
                .endpoints()
                .iter()
                .find(|(id, _)| *id == worker)
                .map(|(_, addr)| addr.to_string()),
            tablets,
            channel_ops,
            direct_ops,
        });
    }
    Ok(Json(out))
}

async fn tablets(State(state): State<AdminState>) -> Json<Vec<TabletDto>> {
    let routing = state.engine.routing();
    Json(
        routing
            .directory()
            .tablets()
            .iter()
            .map(|tablet| TabletDto {
                tablet: tablet.id().as_u64(),
                state: tablet.state().to_string(),
                epoch: tablet.epoch().as_u64(),
                guard: tablet.guard().as_u64(),
                range: tablet.range().to_string(),
                owner: routing
                    .placement()
                    .worker_of(tablet.id())
                    .map(kivi_types::WorkerId::as_u64),
            })
            .collect(),
    )
}

/// Builds the admin router with its Tower middleware: request tracing,
/// a small body limit (read-only API), and a per-request timeout.
pub fn router(state: AdminState) -> axum::Router {
    axum::Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/v1/node", get(node))
        .route("/v1/workers", get(workers))
        .route("/v1/tablets", get(tablets))
        .layer(
            tower::ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(RequestBodyLimitLayer::new(MAX_ADMIN_BODY_BYTES))
                .layer(TimeoutLayer::with_status_code(
                    StatusCode::SERVICE_UNAVAILABLE,
                    ADMIN_REQUEST_TIMEOUT,
                )),
        )
        .with_state(state)
}

/// Serves the admin plane until `shutdown` resolves, then returns for
/// orderly data-plane teardown by the caller.
pub async fn serve(
    listener: tokio::net::TcpListener,
    state: AdminState,
    shutdown: impl core::future::Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let addr: SocketAddr = listener.local_addr().context("admin listener address")?;
    tracing::info!(endpoint = %addr, "admin plane listening");
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await
        .context("admin server failed")?;
    tracing::info!("admin plane stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_engine::{EngineConfig, LocalEngine, Placement};
    use kivi_tablet::{DirectorySnapshot, HashPrefix, PartitionRange};
    use kivi_types::{NamespaceId, TabletEpoch, TabletId, WorkerId, WriteGuardGeneration};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    const NS: NamespaceId = NamespaceId::from_u64(1);

    /// One active root tablet on worker 0 of two (worker 1 idles: valid,
    /// and it exercises the empty-tablet-list path).
    fn test_engine() -> LocalEngine {
        let directory = DirectorySnapshot::bootstrap(
            NS,
            TabletId::from_u64(1),
            PartitionRange::Hash(HashPrefix::new(0, 0).expect("root")),
            TabletEpoch::INITIAL,
            WriteGuardGeneration::INITIAL,
        )
        .and_then(|snapshot| snapshot.stage(TabletId::from_u64(1)))
        .and_then(|snapshot| snapshot.activate(TabletId::from_u64(1)))
        .expect("active root");
        LocalEngine::start(EngineConfig {
            namespace: NS,
            directory,
            placement: Placement::new([(TabletId::from_u64(1), WorkerId::from_u64(0))]),
            worker_count: 2,
            request_capacity: 16,
            network: None,
        })
        .expect("test engine starts")
    }

    fn test_state(engine: &LocalEngine) -> AdminState {
        // Channel-only engines expose no sockets; the admin plane still
        // reports topology and counters.
        AdminState {
            node: NodeId::from_u64(7),
            cluster: ClusterId::from_u128(0xC1),
            engine: engine.admin_handle(),
        }
    }

    /// Runs the admin plane on an ephemeral port. Returns the port, the
    /// shutdown trigger, and the engine (held alive for the test).
    async fn spawn_admin() -> (u16, tokio::sync::oneshot::Sender<()>, LocalEngine) {
        let engine = test_engine();
        let state = test_state(&engine);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("admin binds");
        let port = listener.local_addr().expect("addr").port();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(async move {
            serve(listener, state, async move {
                rx.await.ok();
            })
            .await
            .expect("serve runs");
        });
        // Let the listener start accepting before the first probe lands.
        tokio::time::sleep(Duration::from_millis(50)).await;
        (port, tx, engine)
    }

    /// Minimal HTTP/1.1 exchange over a fresh connection (`connection:
    /// close`, so the response ends at EOF). Returns status + body bytes.
    async fn exchange(port: u16, head: &str, body: &[u8]) -> (u16, Vec<u8>) {
        let mut stream = tokio::time::timeout(
            Duration::from_secs(10),
            tokio::net::TcpStream::connect(format!("127.0.0.1:{port}")),
        )
        .await
        .expect("connect is fast")
        .expect("connect");
        let request = format!("{head}\r\nhost: x\r\nconnection: close\r\n\r\n");
        stream.write_all(request.as_bytes()).await.expect("head");
        // Best effort: oversized bodies may meet an early RST.
        let _ = stream.write_all(body).await;
        let mut raw = Vec::new();
        tokio::time::timeout(Duration::from_secs(10), stream.read_to_end(&mut raw))
            .await
            .expect("response is fast")
            .expect("read");
        let text = String::from_utf8_lossy(&raw);
        let status: u16 = text
            .lines()
            .next()
            .expect("status line")
            .split_whitespace()
            .nth(1)
            .expect("status code")
            .parse()
            .expect("numeric status");
        let body = raw
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map_or(Vec::new(), |at| raw[at + 4..].to_vec());
        (status, body)
    }

    async fn get(port: u16, path: &str) -> (u16, serde_json::Value) {
        let (status, body) = exchange(port, &format!("GET {path} HTTP/1.1"), &[]).await;
        let json: serde_json::Value = serde_json::from_slice(&body).expect("admin bodies are JSON");
        (status, json)
    }

    #[tokio::test]
    async fn health_reports_ok_with_version() {
        let (port, _shutdown, _engine) = spawn_admin().await;
        let (status, json) = get(port, "/health").await;
        assert_eq!(status, 200);
        assert_eq!(json["status"], "ok");
        assert_eq!(json["version"], env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn node_ready_workers_tablets_shapes() {
        let (port, _shutdown, _engine) = spawn_admin().await;
        let (status, ready) = get(port, "/ready").await;
        assert_eq!(status, 200);
        assert_eq!(ready["ready"], true);
        assert_eq!(ready["workers"], 2);
        assert_eq!(ready["tablets"], 1);
        let (status, node) = get(port, "/v1/node").await;
        assert_eq!(status, 200);
        assert_eq!(node["node"], 7);
        assert_eq!(node["cluster"], 0xC1);
        assert_eq!(node["namespace"], 1);
        assert_eq!(node["workers"], 2);
        let (status, workers) = get(port, "/v1/workers").await;
        assert_eq!(status, 200);
        let workers = workers.as_array().expect("worker list");
        assert_eq!(workers.len(), 2);
        assert_eq!(workers[0]["worker"], 0);
        assert_eq!(workers[0]["tablets"], serde_json::json!([1]));
        assert_eq!(workers[1]["tablets"], serde_json::json!([]));
        // Channel-only test engine: no native endpoints, zero counters.
        assert_eq!(workers[0]["endpoint"], serde_json::Value::Null);
        assert_eq!(workers[0]["channel_ops"], 0);
        assert_eq!(workers[0]["direct_ops"], 0);
        let (status, tablets) = get(port, "/v1/tablets").await;
        assert_eq!(status, 200);
        let tablets = tablets.as_array().expect("tablet list");
        assert_eq!(tablets.len(), 1);
        assert_eq!(tablets[0]["tablet"], 1);
        assert_eq!(tablets[0]["state"], "active");
        assert_eq!(tablets[0]["range"], "hash:*");
        assert_eq!(tablets[0]["owner"], 0);
    }

    #[tokio::test]
    async fn unknown_paths_are_404() {
        let (port, _shutdown, _engine) = spawn_admin().await;
        let (status, _) = exchange(port, "GET /v1/nope HTTP/1.1", &[]).await;
        assert_eq!(status, 404);
    }

    #[tokio::test]
    async fn oversized_bodies_are_rejected_before_routing() {
        let (port, _shutdown, _engine) = spawn_admin().await;
        let big = vec![b'x'; 128 * 1024];
        let mut stream = tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .expect("connect");
        let head = "POST /health HTTP/1.1\r\ncontent-length: 131072\
            \r\nhost: x\r\nconnection: close\r\n\r\n";
        stream.write_all(head.as_bytes()).await.expect("head");
        let _ = stream.write_all(&big).await;
        let mut raw = Vec::new();
        // Early rejection can RST the connection while body bytes are still
        // in flight (unread data on close): a 413 or a bare abort both
        // prove the request never reached a handler. Anything else fails.
        match tokio::time::timeout(Duration::from_secs(10), stream.read_to_end(&mut raw))
            .await
            .expect("server answers promptly")
        {
            Ok(_) => {
                let text = String::from_utf8_lossy(&raw);
                let status: u16 = text
                    .lines()
                    .next()
                    .expect("status line")
                    .split_whitespace()
                    .nth(1)
                    .expect("status code")
                    .parse()
                    .expect("numeric status");
                assert_eq!(status, 413);
            }
            Err(error)
                if error.kind() == std::io::ErrorKind::ConnectionAborted
                    || error.kind() == std::io::ErrorKind::ConnectionReset =>
            {
                assert!(raw.is_empty(), "aborted before any response bytes");
            }
            Err(error) => panic!("unexpected read error: {error}"),
        }
    }

    #[tokio::test]
    async fn graceful_shutdown_stops_serving() {
        let (port, shutdown, _engine) = spawn_admin().await;
        let (status, _) = get(port, "/health").await;
        assert_eq!(status, 200);
        shutdown.send(()).expect("trigger shutdown");
        // The listener closes promptly; refused connections prove it.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            let refused = tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
                .await
                .is_err();
            if refused || tokio::time::Instant::now() > deadline {
                assert!(refused, "admin listener closed after shutdown");
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}
