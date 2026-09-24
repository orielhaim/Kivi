//! Distributed redundancy admin surface: `/v1/redundancy/*`.
//!
//! Thin HTTP adapter over
//! [`kivi_consensus::distcoord::RedundancyCoordinator`]. Every handler is
//! bounded (60 s request ceiling, 8 MiB decoded image cap, admin body limit
//! shared with the rest of the plane), verifies bytes locally before exposing
//! them, and never touches Raft: a fabric read, repair, or restore heals
//! immutable bytes and cache state only. No handler here orders mutable
//! state, no handler acknowledges replication, and no handler weakens the
//! sidecar durability gate.
//!
//! ## Threading
//!
//! Kivi's async architecture makes reactor futures `!Send` (consensus and
//! mesh work lives on Compio owner threads), while axum handlers must be
//! `Send`. Rather than adding a runtime or forcing `Send` through the mesh,
//! each handler runs its coordinator future to completion on one Tokio
//! blocking thread inside a fresh Compio runtime through [`bridge`] - the
//! same synchronous bridge the coordinator itself uses for control reads.
//! Bounded by the coordinator's RPC and commit-wait deadlines; tablet
//! workers are never involved.
//!
//! Shapes follow the existing admin conventions: `Json` responses, HTTP
//! status carrying the coordinator's failure class, additive-only DTOs.

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;

use kivi_redundancy::{AssetId, AssetKind, SchemeParams};
use kivi_types::{ChunkId, SecurityDomainId};

use crate::cluster::ClusterShared;

/// Maximum decoded image bytes accepted by one call (matches the fragment
/// wire cap; base64 inflates 33% and the admin body limit still applies).
const MAX_REDUNDANCY_BYTES: usize = 8 * 1024 * 1024;

/// Standard base64 alphabet decode (padded) with strict length and character
/// checks. Rejects whitespace and the URL-safe alphabet so one image has
/// exactly one accepted encoding.
fn decode_b64(text: &str) -> Option<Vec<u8>> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = text.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for quad in bytes.chunks(4) {
        let mut values = [0u8; 4];
        let mut padding = 0usize;
        for (index, byte) in quad.iter().enumerate() {
            if *byte == b'=' {
                if index < 2 {
                    return None;
                }
                padding += 1;
                values[index] = 0;
                continue;
            }
            if padding > 0 {
                return None;
            }
            // Sextets are 0..=63 by construction, so this cast is exact.
            #[allow(clippy::cast_possible_truncation)]
            let value = TABLE.iter().position(|item| item == byte)? as u8;
            values[index] = value;
        }
        let triple = (u32::from(values[0]) << 18)
            | (u32::from(values[1]) << 12)
            | (u32::from(values[2]) << 6)
            | u32::from(values[3]);
        // Shifts select one byte each; truncation is exact by construction.
        #[allow(clippy::cast_possible_truncation)]
        {
            out.push((triple >> 16) as u8);
            if padding < 2 {
                out.push((triple >> 8) as u8);
            }
            if padding < 1 {
                out.push(triple as u8);
            }
        }
    }
    Some(out)
}

/// Encodes bytes as padded standard base64.
fn encode_b64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let mut block = [0u8; 3];
        block[..chunk.len()].copy_from_slice(chunk);
        let packed = (u32::from(block[0]) << 16) | (u32::from(block[1]) << 8) | u32::from(block[2]);
        out.push(TABLE[((packed >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((packed >> 12) & 0x3F) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[((packed >> 6) & 0x3F) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(packed & 0x3F) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// Renders one 32-byte identity as lowercase hex.
fn hex32(id: &[u8; 32]) -> String {
    use core::fmt::Write as _;
    let mut out = String::with_capacity(64);
    for byte in id {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Parses a 32-byte hex identity (either case).
fn parse_hex32(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (index, slot) in out.iter_mut().enumerate() {
        let pair = text.get(index * 2..index * 2 + 2)?;
        *slot = u8::from_str_radix(pair, 16).ok()?;
    }
    Some(out)
}

/// Maps a coordinator error onto an HTTP status: bounded overloads and
/// transport faults are retryable (503), fencing is a conflict (409),
/// unrecoverable loss is terminal for this request (422), everything else is
/// a bad request (400).
fn status_for(error: &kivi_redundancy::RedundancyError) -> StatusCode {
    use kivi_redundancy::RedundancyError as Fault;
    match error {
        Fault::Overloaded { .. } | Fault::Unreachable { .. } | Fault::Unreadable { .. } => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        Fault::StaleGeneration { .. } | Fault::StaleIncarnation { .. } => StatusCode::CONFLICT,
        Fault::Unrecoverable { .. } => StatusCode::UNPROCESSABLE_ENTITY,
        _ => StatusCode::BAD_REQUEST,
    }
}

/// Body for `protect`: either inline bytes or a sidecar chunk reference.
#[derive(Debug, Clone, Deserialize)]
struct ProtectBody {
    /// Asset family discriminant (see [`AssetKind`]).
    kind: u8,
    /// Security domain (chunk/manifest families; `0` for checkpoint).
    #[serde(default)]
    domain: u64,
    /// Inline image, standard base64 (exclusive with `chunk_hex`).
    #[serde(default)]
    bytes_b64: Option<String>,
    /// Sidecar chunk identity read from the local sidecar store.
    #[serde(default)]
    chunk_hex: Option<String>,
    /// Expected chunk length (required with `chunk_hex`).
    #[serde(default)]
    len: Option<u64>,
    /// Independent-failure tolerance the planner must satisfy.
    #[serde(default)]
    tolerance: u8,
}

/// Body for asset-addressed actions (`repair`).
#[derive(Debug, Clone, Deserialize)]
struct AssetBody {
    /// Asset family discriminant.
    kind: u8,
    /// Security domain.
    #[serde(default)]
    domain: u64,
    /// Content identity, lowercase hex.
    hash_hex: String,
}

/// Body for `drain`.
#[derive(Debug, Clone, Deserialize)]
struct DrainBody {
    /// Node losing its fragments.
    node: u64,
}

/// Body for `transition`.
#[derive(Debug, Clone, Deserialize)]
struct TransitionBody {
    /// Asset family discriminant.
    kind: u8,
    /// Security domain.
    #[serde(default)]
    domain: u64,
    /// Content identity, lowercase hex.
    hash_hex: String,
    /// Target scheme: `{"replication":{"copies":3}}` or
    /// `{"reed_solomon":{"data":8,"parity":3,"fragment_len":524288}}`.
    scheme: serde_json::Value,
}

/// Body for `restore-chunk`.
#[derive(Debug, Clone, Deserialize)]
struct RestoreBody {
    /// Sidecar chunk identity, lowercase hex.
    chunk_hex: String,
    /// Security domain the identity binds.
    #[serde(default)]
    domain: u64,
    /// Exact chunk length.
    len: u64,
}

/// Parses a target scheme from JSON. Widths stay server policy; this exists
/// so operators can request a transition explicitly.
fn scheme_from_json(value: &serde_json::Value) -> Result<SchemeParams, String> {
    use kivi_redundancy::{ReplicationParams, RsParams};
    let object = value
        .as_object()
        .ok_or_else(|| "scheme must be an object".to_owned())?;
    if let Some(params) = object.get("replication") {
        let copies = params
            .get("copies")
            .and_then(serde_json::Value::as_u64)
            .and_then(|copies| u8::try_from(copies).ok())
            .ok_or_else(|| "replication.copies must be 1..=8".to_owned())?;
        let scheme = SchemeParams::Replication(ReplicationParams { copies });
        scheme.validate().map_err(|error| error.to_string())?;
        return Ok(scheme);
    }
    if let Some(params) = object.get("reed_solomon") {
        let data = params
            .get("data")
            .and_then(serde_json::Value::as_u64)
            .and_then(|data| u8::try_from(data).ok())
            .ok_or_else(|| "reed_solomon.data must be 1..=32".to_owned())?;
        let parity = params
            .get("parity")
            .and_then(serde_json::Value::as_u64)
            .and_then(|parity| u8::try_from(parity).ok())
            .ok_or_else(|| "reed_solomon.parity must be 1..=8".to_owned())?;
        let fragment_len = params
            .get("fragment_len")
            .and_then(serde_json::Value::as_u64)
            .and_then(|len| u32::try_from(len).ok())
            .ok_or_else(|| "reed_solomon.fragment_len must be 1..=8 MiB".to_owned())?;
        let scheme = SchemeParams::ReedSolomon(RsParams {
            data,
            parity,
            fragment_len,
        });
        scheme.validate().map_err(|error| error.to_string())?;
        return Ok(scheme);
    }
    Err("scheme must name replication or reed_solomon".to_owned())
}

/// Resolves the coordinator or answers 503 (the plane is optional).
fn coordinator_arc(
    shared: &ClusterShared,
) -> Option<Arc<kivi_consensus::distcoord::RedundancyCoordinator>> {
    shared.redundancy.clone()
}

/// Standard 503 for a disabled plane.
fn plane_disabled() -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({ "error": "redundancy plane disabled" })),
    )
}

/// Builds an asset identity from request fields, answering 400 on bad input.
fn asset_from(
    kind: u8,
    domain: u64,
    hash_hex: &str,
) -> Result<AssetId, (StatusCode, Json<serde_json::Value>)> {
    let kind = AssetKind::from_u8(kind).map_err(|error| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": error.to_string() })),
        )
    })?;
    let hash = parse_hex32(hash_hex).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "hash_hex must be 64 hex characters" })),
        )
    })?;
    AssetId::new(kind, domain, hash).map_err(|error| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": error.to_string() })),
        )
    })
}

/// Reads `kind`/`domain`/`hash_hex` from a query string.
fn asset_from_query(
    params: &HashMap<String, String>,
) -> Result<AssetId, (StatusCode, Json<serde_json::Value>)> {
    let kind = params
        .get("kind")
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or(5);
    let domain = params
        .get("domain")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    let Some(hash_hex) = params.get("hash_hex") else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "hash_hex required" })),
        ));
    };
    asset_from(kind, domain, hash_hex)
}

/// Admin-plane failure: either the blocking bridge/runtime failed or the
/// coordinator reported a fabric fault. Both render to JSON with the right
/// status; the bridge failure is retryable (503).
enum AdminFault {
    /// Blocking bridge or Compio runtime failure (retryable).
    Bridge(String),
    /// Coordinator-reported fabric fault.
    Fabric(kivi_redundancy::RedundancyError),
}

impl AdminFault {
    /// Renders the fault as an HTTP response.
    fn into_response(self) -> (StatusCode, Json<serde_json::Value>) {
        match self {
            Self::Bridge(detail) => (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": detail })),
            ),
            Self::Fabric(error) => (
                status_for(&error),
                Json(serde_json::json!({ "error": error.to_string() })),
            ),
        }
    }
}

/// Bridges one coordinator future: Tokio blocking thread → fresh Compio
/// runtime → coordinator future. Kivi's data plane is Compio-only and its
/// futures are `!Send`, so this is the single documented crossing point
/// between the HTTP surface and the redundancy plane. Bounded by the
/// coordinator's RPC and commit-wait deadlines.
async fn bridge<T, F, Fut>(operation: &'static str, task: F) -> Result<T, AdminFault>
where
    T: Send + 'static,
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<T, kivi_redundancy::RedundancyError>>,
{
    let joined: Result<T, kivi_redundancy::RedundancyError> =
        tokio::task::spawn_blocking(move || {
            let runtime = match compio::runtime::Runtime::new() {
                Ok(runtime) => runtime,
                Err(error) => {
                    return Err(AdminFault::Bridge(format!(
                        "{operation} compio runtime: {error}"
                    )));
                }
            };
            Ok(runtime.block_on(task()))
        })
        .await
        .map_err(|error| AdminFault::Bridge(format!("{operation} task failed: {error}")))??;
    joined.map_err(AdminFault::Fabric)
}

/// Inline bytes or a sidecar chunk, as the protect endpoint accepts both.
enum ProtectRequest {
    /// Inline image bytes (base64 in the request body).
    Bytes(Vec<u8>),
    /// Sidecar chunk coordinates plus its declared length.
    Chunk(ChunkId, SecurityDomainId, u64),
}

/// POST `/v1/redundancy/protect`: protects inline bytes or a sidecar chunk
/// and returns the published generation plus its actual fragment holders.
async fn protect(
    State(shared): State<ClusterShared>,
    Json(body): Json<ProtectBody>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Some(coordinator) = coordinator_arc(&shared) else {
        return plane_disabled();
    };
    let request = match (body.bytes_b64.as_deref(), body.chunk_hex.as_deref()) {
        (Some(_), Some(_)) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "bytes_b64 and chunk_hex are exclusive" })),
            );
        }
        (Some(encoded), None) => {
            let Some(bytes) = decode_b64(encoded) else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": "bytes_b64 is not standard base64" })),
                );
            };
            if bytes.len() > MAX_REDUNDANCY_BYTES {
                return (
                    StatusCode::PAYLOAD_TOO_LARGE,
                    Json(serde_json::json!({ "error": "image exceeds 8 MiB" })),
                );
            }
            ProtectRequest::Bytes(bytes)
        }
        (None, Some(chunk_hex)) => {
            let (Some(hash), Some(len)) = (parse_hex32(chunk_hex), body.len) else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": "chunk_hex + len required" })),
                );
            };
            ProtectRequest::Chunk(
                ChunkId::from_bytes(hash),
                SecurityDomainId::from_u64(body.domain),
                len,
            )
        }
        (None, None) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "bytes_b64 or chunk_hex required" })),
            );
        }
    };
    let (kind, domain, tolerance) = (body.kind, body.domain, body.tolerance);
    let job = Arc::clone(&coordinator);
    let published = match bridge("protect", move || async move {
        match request {
            ProtectRequest::Bytes(bytes) => {
                job.protect_bytes(kind, domain, &bytes, tolerance).await
            }
            ProtectRequest::Chunk(chunk, security, len) => {
                job.protect_chunk(chunk, security, len, tolerance).await
            }
        }
    })
    .await
    {
        Ok(published) => published,
        Err(fault) => return fault.into_response(),
    };
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "asset": hex32(&published.layout.asset.hash),
            "kind": published.layout.asset.kind.as_u8(),
            "domain": published.layout.asset.domain,
            "control_generation": published.control_generation,
            "generation": published.layout.generation,
            "scheme": published.layout.params.scheme().name(),
            "params": format!("{:?}", published.layout.params),
            "logical_len": published.layout.logical_len,
            "fragments": published.layout.fragments.iter()
                .map(|record| serde_json::json!({
                    "index": record.id.index,
                    "node": record.node.as_u64(),
                    "role": record.id.role.as_u8(),
                    "stored_len": record.stored_len,
                }))
                .collect::<Vec<_>>(),
        })),
    )
}

/// GET `/v1/redundancy/read`: reconstructs through the published layout
/// (replica or RS) and returns the content-verified image.
async fn read(
    State(shared): State<ClusterShared>,
    Query(params): Query<HashMap<String, String>>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Some(coordinator) = coordinator_arc(&shared) else {
        return plane_disabled();
    };
    let asset = match asset_from_query(&params) {
        Ok(asset) => asset,
        Err(response) => return response,
    };
    match bridge("read", {
        let job = Arc::clone(&coordinator);
        move || {
            let owned = Arc::clone(&job);
            async move { owned.read_bytes(&asset).await }
        }
    })
    .await
    {
        Ok((bytes, degraded)) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "asset": hex32(&asset.hash),
                "len": bytes.len(),
                "bytes_b64": encode_b64(&bytes),
                "degraded": degraded,
            })),
        ),
        Err(fault) => fault.into_response(),
    }
}

/// POST `/v1/redundancy/repair`: rebuilds explicit deficits from minimum
/// pieces and reinstalls them; idempotent and generation-fenced.
async fn repair(
    State(shared): State<ClusterShared>,
    Json(body): Json<AssetBody>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Some(coordinator) = coordinator_arc(&shared) else {
        return plane_disabled();
    };
    let asset = match asset_from(body.kind, body.domain, &body.hash_hex) {
        Ok(asset) => asset,
        Err(response) => return response,
    };
    match bridge("repair", {
        let job = Arc::clone(&coordinator);
        move || {
            let owned = Arc::clone(&job);
            async move { owned.repair(&asset).await }
        }
    })
    .await
    {
        Ok(report) => (
            StatusCode::OK,
            Json(serde_json::json!({ "report": format!("{report:?}") })),
        ),
        Err(fault) => fault.into_response(),
    }
}

/// POST `/v1/redundancy/drain`: moves every fragment off one node
/// (replacement verified and published before old fragments retire).
async fn drain(
    State(shared): State<ClusterShared>,
    Json(body): Json<DrainBody>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Some(coordinator) = coordinator_arc(&shared) else {
        return plane_disabled();
    };
    let node = kivi_types::NodeId::from_u64(body.node);
    match bridge("drain", {
        let job = Arc::clone(&coordinator);
        move || {
            let owned = Arc::clone(&job);
            async move { owned.drain(node).await }
        }
    })
    .await
    {
        Ok(report) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "moved": report.moved,
                "assets": report.assets.iter()
                    .map(|asset| hex32(&asset.hash))
                    .collect::<Vec<_>>(),
            })),
        ),
        Err(fault) => fault.into_response(),
    }
}

/// POST `/v1/redundancy/transition`: builds, verifies, and publishes a new
/// scheme generation while the current layout keeps serving.
async fn transition(
    State(shared): State<ClusterShared>,
    Json(body): Json<TransitionBody>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Some(coordinator) = coordinator_arc(&shared) else {
        return plane_disabled();
    };
    let asset = match asset_from(body.kind, body.domain, &body.hash_hex) {
        Ok(asset) => asset,
        Err(response) => return response,
    };
    let target = match scheme_from_json(&body.scheme) {
        Ok(target) => target,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": error })),
            );
        }
    };
    match bridge("transition", {
        let job = Arc::clone(&coordinator);
        move || {
            let owned = Arc::clone(&job);
            async move { owned.transition(&asset, target).await }
        }
    })
    .await
    {
        Ok(published) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "generation": published.layout.generation,
                "control_generation": published.control_generation,
                "scheme": published.layout.params.scheme().name(),
                "params": format!("{:?}", published.layout.params),
            })),
        ),
        Err(fault) => fault.into_response(),
    }
}

/// GET `/v1/redundancy/health`: desired vs actual placement plus the
/// Healthy/Degraded/Critical/Unrecoverable verdict and its evidence.
async fn health(
    State(shared): State<ClusterShared>,
    Query(params): Query<HashMap<String, String>>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Some(coordinator) = coordinator_arc(&shared) else {
        return plane_disabled();
    };
    let asset = match asset_from_query(&params) {
        Ok(asset) => asset,
        Err(response) => return response,
    };
    let health = match bridge("health", {
        let job = Arc::clone(&coordinator);
        move || {
            let owned = Arc::clone(&job);
            async move { Ok::<_, kivi_redundancy::RedundancyError>(owned.health(&asset).await) }
        }
    })
    .await
    {
        Ok(Some(health)) => health,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": "no published layout" })),
            );
        }
        Err(fault) => return fault.into_response(),
    };
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "asset": hex32(&asset.hash),
            "health": health.health.name(),
            "generation": health.generation,
            "control_generation": health.control_generation,
            "usable": health.usable,
            "required": health.required,
            "total": health.total,
            "survivable": health.survivable,
            "reconstructable": health.reconstructable,
            "desired": health.desired.iter()
                .map(|(index, node)| serde_json::json!({ "index": index, "node": node.as_u64() }))
                .collect::<Vec<_>>(),
            "actual": health.actual.iter()
                .map(|(index, node, healthy)| serde_json::json!({
                    "index": index, "node": node.as_u64(), "healthy": healthy,
                }))
                .collect::<Vec<_>>(),
        })),
    )
}

/// GET `/v1/redundancy/metrics`: fabric counters, local orphans, per-holder
/// desired bytes, current generations, and background-lane counters.
async fn metrics(State(shared): State<ClusterShared>) -> (StatusCode, Json<serde_json::Value>) {
    let Some(coordinator) = coordinator_arc(&shared) else {
        return plane_disabled();
    };
    let (metrics, lane) = match bridge("metrics", {
        let job = Arc::clone(&coordinator);
        move || {
            let owned = Arc::clone(&job);
            async move {
                let metrics = owned.metrics().await;
                Ok::<_, kivi_redundancy::RedundancyError>((metrics, owned.lane_stats()))
            }
        }
    })
    .await
    {
        Ok(pair) => pair,
        Err(fault) => return fault.into_response(),
    };
    let snapshot = metrics.snapshot;
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "assets_replicated": metrics.replicated_assets,
            "assets_coded": metrics.coded_assets,
            "logical_bytes": metrics.logical_bytes,
            "physical_bytes": metrics.physical_bytes,
            "amplification": metrics.amplification(),
            "healthy": snapshot.healthy,
            "degraded": snapshot.degraded,
            "critical": snapshot.critical,
            "unrecoverable": snapshot.unrecoverable,
            "pending_repairs": snapshot.pending_repairs,
            "repair_bytes_pending": snapshot.repair_bytes_pending,
            "repair_bytes_done": snapshot.repair_bytes_done,
            "encodes": snapshot.encodes,
            "decodes": snapshot.decodes,
            "corruption_detected": snapshot.corruption_detected,
            "reconstruction_failures": snapshot.reconstruction_failures,
            "stale_rejected": snapshot.stale_rejected,
            "transitions": snapshot.transitions,
            "remote_bytes_sent": snapshot.remote_bytes_sent,
            "remote_bytes_received": snapshot.remote_bytes_received,
            "local_bytes_written": snapshot.local_bytes_written,
            "local_bytes_read": snapshot.local_bytes_read,
            "fetch_count": snapshot.fetch_count,
            "fetch_error_count": snapshot.fetch_error_count,
            "fetch_latency_us_sum": snapshot.fetch_latency_us_sum,
            "fetch_latency_us_max": snapshot.fetch_latency_us_max,
            "degraded_reads": snapshot.degraded_reads,
            "orphans_seen": snapshot.orphans_seen,
            "orphans_swept": snapshot.orphans_swept,
            "incarnation_refusals": snapshot.incarnation_refusals,
            "desired_vs_actual_mismatches": snapshot.desired_vs_actual_mismatches,
            "orphans_local": metrics.orphans,
            "by_node_bytes": metrics.by_node_bytes.iter()
                .map(|(node, bytes)| serde_json::json!({ "node": node, "bytes": bytes }))
                .collect::<Vec<_>>(),
            "current_gens": metrics.current_gens.iter()
                .map(|(asset, generation)| serde_json::json!({
                    "asset": hex32(&asset.hash),
                    "kind": asset.kind.as_u8(),
                    "domain": asset.domain,
                    "control_generation": generation,
                }))
                .collect::<Vec<_>>(),
            "lane": format!("{lane:?}"),
        })),
    )
}

/// GET `/v1/redundancy/catalog`: every published layout with its generations.
async fn catalog(State(shared): State<ClusterShared>) -> (StatusCode, Json<serde_json::Value>) {
    let Some(coordinator) = coordinator_arc(&shared) else {
        return plane_disabled();
    };
    let entries = match bridge("catalog", {
        let job = Arc::clone(&coordinator);
        move || {
            let owned = Arc::clone(&job);
            async move { Ok::<_, kivi_redundancy::RedundancyError>(owned.catalog_list().await) }
        }
    })
    .await
    {
        Ok(entries) => entries,
        Err(fault) => return fault.into_response(),
    };
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "assets": entries.iter()
                .map(|entry| serde_json::json!({
                    "asset": hex32(&entry.asset.hash),
                    "kind": entry.asset.kind.as_u8(),
                    "domain": entry.asset.domain,
                    "control_generation": entry.control_generation,
                    "generation": entry.generation,
                    "scheme": entry.params.scheme().name(),
                    "logical_len": entry.logical_len,
                }))
                .collect::<Vec<_>>(),
        })),
    )
}

/// POST `/v1/redundancy/restore-chunk`: heals the local sidecar cache from
/// remote redundancy. Never proposes Raft state; the gate still governs any
/// later acknowledgement.
async fn restore_chunk(
    State(shared): State<ClusterShared>,
    Json(body): Json<RestoreBody>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Some(coordinator) = coordinator_arc(&shared) else {
        return plane_disabled();
    };
    let Some(hash) = parse_hex32(&body.chunk_hex) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "chunk_hex must be 64 hex characters" })),
        );
    };
    let chunk = ChunkId::from_bytes(hash);
    let domain = SecurityDomainId::from_u64(body.domain);
    match bridge("restore-chunk", {
        let job = Arc::clone(&coordinator);
        move || {
            let owned = Arc::clone(&job);
            async move { owned.restore_chunk(chunk, domain, body.len).await }
        }
    })
    .await
    {
        Ok(restored) => (
            StatusCode::OK,
            Json(serde_json::json!({ "restored": restored })),
        ),
        Err(fault) => fault.into_response(),
    }
}

/// POST `/v1/redundancy/sweep-orphans`: reclaims local staged fragments that
/// never became authoritative (disk hygiene; never affects protection).
async fn sweep_orphans(
    State(shared): State<ClusterShared>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Some(coordinator) = coordinator_arc(&shared) else {
        return plane_disabled();
    };
    match bridge("sweep-orphans", {
        let job = Arc::clone(&coordinator);
        move || {
            let owned = Arc::clone(&job);
            async move { owned.sweep_orphans().await }
        }
    })
    .await
    {
        Ok(report) => {
            let unreachable: Vec<u64> = report
                .unreachable
                .iter()
                .map(|node| node.as_u64())
                .collect();
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "swept": report.swept,
                    "unreachable": unreachable,
                })),
            )
        }
        Err(fault) => fault.into_response(),
    }
}

/// Router fragment merged into the server admin plane.
pub fn router() -> Router<ClusterShared> {
    Router::new()
        .route("/v1/redundancy/protect", post(protect))
        .route("/v1/redundancy/read", get(read))
        .route("/v1/redundancy/health", get(health))
        .route("/v1/redundancy/metrics", get(metrics))
        .route("/v1/redundancy/catalog", get(catalog))
        .route("/v1/redundancy/repair", post(repair))
        .route("/v1/redundancy/drain", post(drain))
        .route("/v1/redundancy/transition", post(transition))
        .route("/v1/redundancy/restore-chunk", post(restore_chunk))
        .route("/v1/redundancy/sweep-orphans", post(sweep_orphans))
}

#[cfg(test)]
mod tests {
    use super::{decode_b64, encode_b64, parse_hex32, scheme_from_json};

    #[test]
    fn base64_round_trips_every_length() {
        for len in 0..32usize {
            let bytes: Vec<u8> = (0..len)
                .map(|index| u8::try_from(index * 7 + 3).unwrap_or(0))
                .collect();
            let encoded = encode_b64(&bytes);
            assert_eq!(
                decode_b64(&encoded).as_deref(),
                Some(bytes.as_slice()),
                "len {len}"
            );
        }
    }

    #[test]
    fn base64_rejects_malformed() {
        assert!(decode_b64("A").is_none());
        assert!(decode_b64("A?==").is_none());
        assert!(decode_b64("=AAA").is_none());
        assert!(decode_b64("AA=A").is_none());
    }

    #[test]
    fn hex_round_trips() {
        let id = [0xABu8; 32];
        assert_eq!(parse_hex32(&super::hex32(&id)), Some(id));
        assert!(parse_hex32("zz").is_none());
    }

    #[test]
    fn scheme_parsing_validates_bounds() {
        let ok =
            serde_json::json!({ "reed_solomon": { "data": 2, "parity": 1, "fragment_len": 64 } });
        assert!(scheme_from_json(&ok).is_ok());
        let wide =
            serde_json::json!({ "reed_solomon": { "data": 64, "parity": 1, "fragment_len": 64 } });
        assert!(scheme_from_json(&wide).is_err());
        let copies = serde_json::json!({ "replication": { "copies": 3 } });
        assert!(scheme_from_json(&copies).is_ok());
        assert!(scheme_from_json(&serde_json::json!({})).is_err());
    }
}
