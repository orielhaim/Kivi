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

/// Shared admin state: node identity (operator-configured in ephemeral
/// mode, data-directory-persisted in durable mode — always the same values
/// the engine serves in native handshakes) plus the engine query handle.
#[derive(Debug, Clone)]
pub struct AdminState {
    /// This node's identity.
    pub node: NodeId,
    /// This cluster's identity.
    pub cluster: ClusterId,
    /// This process's incarnation (durable: advanced at startup).
    pub incarnation: NodeIncarnation,
    /// Read-only engine query handle (`Send + Sync`, Tokio-safe).
    pub engine: AdminHandle,
    /// RESP frontend view (`None` when compiled without the feature or
    /// running native-only).
    #[cfg(feature = "redis-compat")]
    pub resp: Option<super::resp::RespAdmin>,
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
    durability_mode: &'static str,
    data_dir: Option<String>,
    redis_compat: RedisCompatSummary,
}

/// RESP compatibility summary embedded in `/v1/node`.
#[derive(Debug, Clone, Serialize)]
struct RedisCompatSummary {
    compiled: bool,
    enabled: bool,
    endpoint: Option<String>,
    namespace: Option<u64>,
}

/// Full RESP frontend diagnostics (`GET /v1/redis`).
#[derive(Debug, Clone, Serialize)]
struct RedisDto {
    compiled: bool,
    enabled: bool,
    endpoint: Option<String>,
    namespace: Option<u64>,
    profile: &'static str,
    connections: u64,
    connections_total: u64,
    requests: u64,
    bytes_in: u64,
    bytes_out: u64,
    protocol_errors: u64,
    unsupported_commands: u64,
    resp2_connections: u64,
    resp3_connections: u64,
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

/// Durability posture: mode, identity, lanes, commit batching, health,
/// and what recovery did.
#[derive(Debug, Clone, Serialize)]
struct DurabilityDto {
    mode: &'static str,
    data_dir: Option<String>,
    node: u64,
    cluster: u128,
    incarnation: u64,
    lanes: Vec<LaneDto>,
    commit: Vec<CommitDto>,
    recovery: Option<RecoveryDto>,
}

/// One WAL lane: position, health, and cumulative counters (`None` fields
/// in ephemeral mode, which has no lanes).
#[derive(Debug, Clone, Serialize)]
struct LaneDto {
    worker: u64,
    lane: Option<u16>,
    active_segment: Option<u64>,
    segments: Option<u64>,
    health: Option<String>,
    batches: Option<u64>,
    records: Option<u64>,
    bytes: Option<u64>,
    fsyncs: Option<u64>,
}

/// What startup recovery replayed.
#[derive(Debug, Clone, Serialize)]
struct RecoveryDto {
    records_replayed: u64,
    segments_scanned: u64,
    tail_truncated_bytes: u64,
    tablets_recovered: u64,
    checkpoint_tablets_loaded: u64,
    checkpoint_bands_verified: u64,
    checkpoint_bytes_restored: u64,
    checkpoint_obsolete_skipped: u64,
    checkpoint_fell_back: u64,
}

/// One worker's commit pipeline: the logical-writes-per-fsync story
/// (`None` fields in ephemeral mode, which has no pipeline).
#[derive(Debug, Clone, Serialize)]
struct CommitDto {
    worker: u64,
    logical_mutations: Option<u64>,
    physical_batches: Option<u64>,
    barriers: Option<u64>,
    bytes_durable: Option<u64>,
    failed_batches: Option<u64>,
    batch_size_avg: Option<f64>,
    batch_size_max: Option<u64>,
    batch_size_p50: Option<u64>,
    batch_size_p95: Option<u64>,
    batch_size_p99: Option<u64>,
    oldest_wait_avg_us: Option<u64>,
    oldest_wait_max_us: Option<u64>,
    barrier_latency_avg_us: Option<u64>,
    barrier_latency_max_us: Option<u64>,
    queue_depth: Option<usize>,
    queue_depth_max: Option<usize>,
    in_flight: Option<bool>,
}

/// Checkpoints: installed current/previous per tablet plus WAL retention.
#[derive(Debug, Clone, Serialize)]
struct CheckpointsDto {
    tablets: Vec<CheckpointTabletDto>,
    wal_retained_bytes: u64,
    wal_reclaimable_bytes: u64,
    checkpoints_completed: u64,
    last_error: Option<String>,
}

/// One retained checkpoint (current or previous) of one tablet.
#[derive(Debug, Clone, Serialize)]
struct CheckpointTabletDto {
    tablet: u64,
    which: &'static str,
    cut: u64,
    manifest: Option<String>,
    created_wall_micros: u64,
    bands: usize,
    bands_reused: usize,
    stored_bytes: u64,
    duration_ms: u64,
    fell_back: bool,
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
    let durability = state.engine.durability();
    Json(NodeDto {
        node: state.node.as_u64(),
        cluster: state.cluster.as_u128(),
        incarnation: state.incarnation.as_u64(),
        namespace: state.engine.namespace().as_u64(),
        workers: state.engine.worker_count(),
        tablets: routing
            .directory()
            .tablets()
            .iter()
            .filter(|tablet| tablet.state().is_writable())
            .count(),
        directory_version: routing.version().as_u64(),
        durability_mode: durability_mode(durability),
        data_dir: durability.map(|info| info.data_dir.display().to_string()),
        redis_compat: redis_summary(&state),
    })
}

/// Builds the `/v1/node`-embedded RESP summary (compiled/enabled split so
/// operators can tell "not built" from "built but native-only").
fn redis_summary(state: &AdminState) -> RedisCompatSummary {
    #[cfg(feature = "redis-compat")]
    {
        match &state.resp {
            None => RedisCompatSummary {
                compiled: true,
                enabled: false,
                endpoint: None,
                namespace: None,
            },
            Some(resp) => RedisCompatSummary {
                compiled: true,
                enabled: true,
                endpoint: Some(resp.endpoint.to_string()),
                namespace: Some(resp.namespace),
            },
        }
    }
    #[cfg(not(feature = "redis-compat"))]
    {
        let _ = state;
        RedisCompatSummary {
            compiled: false,
            enabled: false,
            endpoint: None,
            namespace: None,
        }
    }
}

/// Full RESP diagnostics (`GET /v1/redis`).
async fn redis(State(state): State<AdminState>) -> Json<RedisDto> {
    #[cfg(feature = "redis-compat")]
    {
        match &state.resp {
            None => Json(RedisDto {
                compiled: true,
                enabled: false,
                endpoint: None,
                namespace: None,
                profile: kivi_resp::PROFILE_VERSION,
                connections: 0,
                connections_total: 0,
                requests: 0,
                bytes_in: 0,
                bytes_out: 0,
                protocol_errors: 0,
                unsupported_commands: 0,
                resp2_connections: 0,
                resp3_connections: 0,
            }),
            Some(resp) => {
                let snapshot = resp.stats.snapshot();
                Json(RedisDto {
                    compiled: true,
                    enabled: true,
                    endpoint: Some(resp.endpoint.to_string()),
                    namespace: Some(resp.namespace),
                    profile: kivi_resp::PROFILE_VERSION,
                    connections: snapshot.connections_current,
                    connections_total: snapshot.connections_total,
                    requests: snapshot.requests,
                    bytes_in: snapshot.bytes_in,
                    bytes_out: snapshot.bytes_out,
                    protocol_errors: snapshot.protocol_errors,
                    unsupported_commands: snapshot.unsupported,
                    resp2_connections: snapshot.resp2_current,
                    resp3_connections: snapshot.resp3_current,
                })
            }
        }
    }
    #[cfg(not(feature = "redis-compat"))]
    {
        let _ = state;
        Json(RedisDto {
            compiled: false,
            enabled: false,
            endpoint: None,
            namespace: None,
            profile: "Kivi RESP Compatibility Profile v1",
            connections: 0,
            connections_total: 0,
            requests: 0,
            bytes_in: 0,
            bytes_out: 0,
            protocol_errors: 0,
            unsupported_commands: 0,
            resp2_connections: 0,
            resp3_connections: 0,
        })
    }
}

fn durability_mode(durability: Option<&kivi_engine::EngineDurability>) -> &'static str {
    if durability.is_some() {
        "durable"
    } else {
        "ephemeral"
    }
}

// One linear DTO assembly (lanes → commit → checkpoints): the fields
// read top to bottom in response order; splitting would scatter the shape
// of the admin contract.
#[allow(clippy::too_many_lines)]
async fn durability(State(state): State<AdminState>) -> Result<Json<DurabilityDto>, StatusCode> {
    let engine = state.engine.clone();
    // Lane stats arrive over the blocking control channel.
    let lanes = tokio::task::spawn_blocking(move || engine.lane_stats_snapshot())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let engine = state.engine.clone();
    let commit = tokio::task::spawn_blocking(move || engine.commit_metrics_snapshot())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let by_worker: std::collections::HashMap<u64, kivi_durability::WorkerLaneStats> = lanes
        .into_iter()
        .filter_map(|(id, stats)| stats.map(|stats| (id.as_u64(), stats)))
        .collect();
    let by_commit: std::collections::HashMap<u64, kivi_engine::CommitMetricsSnapshot> = commit
        .into_iter()
        .filter_map(|(id, stats)| stats.map(|stats| (id.as_u64(), stats)))
        .collect();
    let mut out = Vec::new();
    let mut commit_out = Vec::new();
    for index in 0..state.engine.worker_count() {
        let id = u64::try_from(index).unwrap_or(u64::MAX);
        match by_worker.get(&id) {
            Some(lane) => out.push(LaneDto {
                worker: id,
                lane: Some(lane.lane),
                active_segment: Some(lane.active_segment),
                segments: Some(lane.segments),
                health: Some(lane.health.to_string()),
                batches: Some(lane.stats.batches),
                records: Some(lane.stats.records),
                bytes: Some(lane.stats.bytes),
                fsyncs: Some(lane.stats.fsyncs),
            }),
            None => out.push(LaneDto {
                worker: id,
                lane: None,
                active_segment: None,
                segments: None,
                health: None,
                batches: None,
                records: None,
                bytes: None,
                fsyncs: None,
            }),
        }
        match by_commit.get(&id) {
            Some(metrics) => commit_out.push(CommitDto {
                worker: id,
                logical_mutations: Some(metrics.logical_mutations),
                physical_batches: Some(metrics.physical_batches),
                barriers: Some(metrics.barriers),
                bytes_durable: Some(metrics.bytes_durable),
                failed_batches: Some(metrics.failed_batches),
                batch_size_avg: Some(metrics.batch_size_avg),
                batch_size_max: Some(metrics.batch_size_max),
                batch_size_p50: Some(metrics.batch_size_p50),
                batch_size_p95: Some(metrics.batch_size_p95),
                batch_size_p99: Some(metrics.batch_size_p99),
                oldest_wait_avg_us: Some(
                    metrics
                        .oldest_wait_avg
                        .as_micros()
                        .try_into()
                        .unwrap_or(u64::MAX),
                ),
                oldest_wait_max_us: Some(
                    metrics
                        .oldest_wait_max
                        .as_micros()
                        .try_into()
                        .unwrap_or(u64::MAX),
                ),
                barrier_latency_avg_us: Some(
                    metrics
                        .barrier_latency_avg
                        .as_micros()
                        .try_into()
                        .unwrap_or(u64::MAX),
                ),
                barrier_latency_max_us: Some(
                    metrics
                        .barrier_latency_max
                        .as_micros()
                        .try_into()
                        .unwrap_or(u64::MAX),
                ),
                queue_depth: Some(metrics.queue_depth),
                queue_depth_max: Some(metrics.queue_depth_max),
                in_flight: Some(metrics.in_flight),
            }),
            None => commit_out.push(CommitDto {
                worker: id,
                logical_mutations: None,
                physical_batches: None,
                barriers: None,
                bytes_durable: None,
                failed_batches: None,
                batch_size_avg: None,
                batch_size_max: None,
                batch_size_p50: None,
                batch_size_p95: None,
                batch_size_p99: None,
                oldest_wait_avg_us: None,
                oldest_wait_max_us: None,
                barrier_latency_avg_us: None,
                barrier_latency_max_us: None,
                queue_depth: None,
                queue_depth_max: None,
                in_flight: None,
            }),
        }
    }
    let info = state.engine.durability();
    Ok(Json(DurabilityDto {
        mode: durability_mode(info),
        data_dir: info.map(|info| info.data_dir.display().to_string()),
        node: state.node.as_u64(),
        cluster: state.cluster.as_u128(),
        incarnation: state.incarnation.as_u64(),
        lanes: out,
        commit: commit_out,
        recovery: info.map(|info| RecoveryDto {
            records_replayed: info.recovery.records_replayed,
            segments_scanned: info.recovery.segments_scanned,
            tail_truncated_bytes: info.recovery.tail_truncated_bytes,
            tablets_recovered: info.recovery.tablets_recovered,
            checkpoint_tablets_loaded: info.checkpoint_recovery.tablets_loaded,
            checkpoint_bands_verified: info.checkpoint_recovery.bands_verified,
            checkpoint_bytes_restored: info.checkpoint_recovery.bytes_restored,
            checkpoint_obsolete_skipped: info.checkpoint_recovery.obsolete_skipped,
            checkpoint_fell_back: info.checkpoint_recovery.fell_back,
        }),
    }))
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

/// Checkpoints: installed current/previous per tablet plus WAL retention.
/// Pure status reads (locks a brief status mutex, never the data path).
async fn checkpoints(State(state): State<AdminState>) -> Json<CheckpointsDto> {
    let engine = state.engine.clone();
    let snapshot = tokio::task::spawn_blocking(move || engine.checkpoint_state())
        .await
        .unwrap_or_default();
    let mut tablets: Vec<CheckpointTabletDto> = Vec::new();
    let mut ids: Vec<u64> = snapshot
        .latest
        .keys()
        .chain(snapshot.previous.keys())
        .map(|tablet| tablet.as_u64())
        .collect();
    ids.sort_unstable();
    ids.dedup();
    for id in ids {
        let tablet = kivi_types::TabletId::from_u64(id);
        if let Some(info) = snapshot.latest.get(&tablet) {
            tablets.push(CheckpointTabletDto {
                tablet: id,
                which: "current",
                cut: info.cut,
                manifest: info
                    .manifest
                    .as_ref()
                    .map(kivi_checkpoint::ArtifactHash::hex),
                created_wall_micros: info.created_wall_micros,
                bands: info.bands,
                bands_reused: info.bands_reused,
                stored_bytes: info.stored_bytes,
                duration_ms: info.duration.as_millis().try_into().unwrap_or(u64::MAX),
                fell_back: info.fell_back,
            });
        }
        if let Some(info) = snapshot.previous.get(&tablet) {
            tablets.push(CheckpointTabletDto {
                tablet: id,
                which: "previous",
                cut: info.cut,
                manifest: info
                    .manifest
                    .as_ref()
                    .map(kivi_checkpoint::ArtifactHash::hex),
                created_wall_micros: info.created_wall_micros,
                bands: info.bands,
                bands_reused: info.bands_reused,
                stored_bytes: info.stored_bytes,
                duration_ms: info.duration.as_millis().try_into().unwrap_or(u64::MAX),
                fell_back: info.fell_back,
            });
        }
    }
    Json(CheckpointsDto {
        tablets,
        wal_retained_bytes: snapshot.wal_retained_bytes,
        wal_reclaimable_bytes: snapshot.wal_reclaimable_bytes,
        checkpoints_completed: snapshot.checkpoints_completed,
        last_error: snapshot.last_error,
    })
}

/// Per-tablet Memory Fabric footprint.
#[derive(Serialize)]
struct FabricTabletDto {
    tablet: u64,
    objects: usize,
    inline_bytes: u64,
    dram_bytes: u64,
    compressed_bytes: u64,
    nvme_bytes: u64,
}

/// Per-worker Memory Fabric report: counters, gauges, queue depths, and
/// per-tablet footprints. Answers "why is this object in `NVMe`"
/// (residence breakdown) and "why can this memory not be reclaimed"
/// (`pinned`, `retire_pending`, `locators`).
#[derive(Serialize)]
struct FabricWorkerDto {
    worker: u64,
    objects: usize,
    inline_bytes: u64,
    dram_bytes: u64,
    compressed_bytes: u64,
    nvme_bytes: u64,
    hits: u64,
    promotion_misses: u64,
    demotions: u64,
    promotions: u64,
    migration_bytes: u64,
    compression_ratio_bps: u64,
    cancelled: u64,
    compactions: u64,
    pinned: usize,
    parked: usize,
    retire_pending: usize,
    stale_completions: u64,
    parked_resumes: u64,
    admission_rejects: u64,
    locators: usize,
    offcore_queued: usize,
    offcore_inflight: usize,
    moves_queued: usize,
    arena_pressure: f64,
    tablets: Vec<FabricTabletDto>,
}

/// Memory Fabric observability rollup across workers plus summed totals.
#[derive(Serialize)]
struct FabricDto {
    workers: Vec<FabricWorkerDto>,
    total_objects: usize,
    total_dram_bytes: u64,
    total_compressed_bytes: u64,
    total_nvme_bytes: u64,
    total_pinned: usize,
    total_parked: usize,
    total_admission_rejects: u64,
}

async fn fabric(State(state): State<AdminState>) -> Result<Json<FabricDto>, StatusCode> {
    // Control-channel rendezvous blocks: keep it off the async executor.
    let engine = state.engine.clone();
    let reports = tokio::task::spawn_blocking(move || engine.fabric_stats_snapshot())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let mut workers = Vec::with_capacity(reports.len());
    let mut total_objects = 0usize;
    let mut total_dram_bytes = 0u64;
    let mut total_compressed_bytes = 0u64;
    let mut total_nvme_bytes = 0u64;
    let mut total_pinned = 0usize;
    let mut total_parked = 0usize;
    let mut total_admission_rejects = 0u64;
    for (id, report) in reports {
        let snap = &report.fabric;
        total_objects += snap.objects;
        total_dram_bytes += snap.dram_bytes;
        total_compressed_bytes += snap.compressed_bytes;
        total_nvme_bytes += snap.nvme_bytes;
        total_pinned += snap.pinned;
        total_parked += snap.parked;
        total_admission_rejects += snap.admission_rejects;
        workers.push(FabricWorkerDto {
            worker: id.as_u64(),
            objects: snap.objects,
            inline_bytes: snap.inline_bytes,
            dram_bytes: snap.dram_bytes,
            compressed_bytes: snap.compressed_bytes,
            nvme_bytes: snap.nvme_bytes,
            hits: snap.hits,
            promotion_misses: snap.promotion_misses,
            demotions: snap.demotions,
            promotions: snap.promotions,
            migration_bytes: snap.migration_bytes,
            compression_ratio_bps: snap.compression_ratio_bps,
            cancelled: snap.cancelled,
            compactions: snap.compactions,
            pinned: snap.pinned,
            parked: snap.parked,
            retire_pending: snap.retire_pending,
            stale_completions: snap.stale_completions,
            parked_resumes: snap.parked_resumes,
            admission_rejects: snap.admission_rejects,
            locators: snap.locators,
            offcore_queued: report.offcore_depth.0,
            offcore_inflight: report.offcore_depth.1,
            moves_queued: report.move_depth,
            arena_pressure: report.arena_pressure,
            tablets: report
                .tablets
                .iter()
                .map(|(tablet, footprint)| FabricTabletDto {
                    tablet: tablet.as_u64(),
                    objects: footprint.objects,
                    inline_bytes: footprint.inline_bytes,
                    dram_bytes: footprint.dram_bytes,
                    compressed_bytes: footprint.compressed_bytes,
                    nvme_bytes: footprint.nvme_bytes,
                })
                .collect(),
        });
    }
    Ok(Json(FabricDto {
        workers,
        total_objects,
        total_dram_bytes,
        total_compressed_bytes,
        total_nvme_bytes,
        total_pinned,
        total_parked,
        total_admission_rejects,
    }))
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
        .route("/v1/durability", get(durability))
        .route("/v1/checkpoints", get(checkpoints))
        .route("/v1/fabric", get(fabric))
        .route("/v1/redis", get(redis))
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
    use kivi_engine::{DurabilityMode, EngineConfig, LocalEngine, Placement};
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
            chunks: kivi_engine::ChunkFabricConfig::default(),
            fabric: kivi_engine::FabricConfig::default(),
            network: None,
            durability: DurabilityMode::Ephemeral,
        })
        .expect("test engine starts")
    }

    fn test_state(engine: &LocalEngine) -> AdminState {
        // Channel-only engines expose no sockets; the admin plane still
        // reports topology and counters.
        AdminState {
            node: NodeId::from_u64(7),
            cluster: ClusterId::from_u128(0xC1),
            incarnation: NodeIncarnation::INITIAL,
            engine: engine.admin_handle(),
            #[cfg(feature = "redis-compat")]
            resp: None,
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
        assert_eq!(node["incarnation"], 1);
        assert_eq!(node["durability_mode"], "ephemeral");
        assert_eq!(node["data_dir"], serde_json::Value::Null);
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
    async fn durability_endpoint_reports_ephemeral_shape() {
        let (port, _shutdown, _engine) = spawn_admin().await;
        let (status, durability) = get(port, "/v1/durability").await;
        assert_eq!(status, 200);
        assert_eq!(durability["mode"], "ephemeral");
        assert_eq!(durability["data_dir"], serde_json::Value::Null);
        assert_eq!(durability["node"], 7);
        assert_eq!(durability["incarnation"], 1);
        assert_eq!(durability["recovery"], serde_json::Value::Null);
        let lanes = durability["lanes"].as_array().expect("lane list");
        assert_eq!(lanes.len(), 2);
        assert_eq!(lanes[0]["lane"], serde_json::Value::Null);
        assert_eq!(lanes[0]["health"], serde_json::Value::Null);
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
