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
    WallTimestamp, WorkerId, WriteGuardGeneration,
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
/// Admin per-request ceiling (60 s: redundancy protects/uploads plus Raft
/// commits exceed the old 5 s budget; ordinary diagnostics stay fast).
const ADMIN_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
/// Admin body limit (16 MiB: redundancy `protect` carries base64 images
/// whose decoded form caps at 8 MiB; base64 inflates 33%, so ~10.7 MiB
/// encoded fits with headroom; every redundancy handler additionally
/// enforces its own decoded-size cap before allocating).
const MAX_ADMIN_BODY_BYTES: usize = 16 * 1024 * 1024;
/// Maximum stream entries encoded in one `StreamEntries` response page
/// (mirrors the embedded engine's bound; shaping truncates explicitly and
/// the client resumes by offset, keeping shard order).
const MAX_STREAM_PAGE_ENTRIES: usize = 256;

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
    /// Namespace physical layout (`hash` or `ordered`): selects the
    /// genesis tiling (hash-prefix buddy tiles vs ordered byte tiles).
    /// Post-genesis splits/merges follow each tablet's range layout.
    pub layout: String,
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
    /// Admin endpoints per node id (reconciler forwarding targets).
    pub admins: Vec<(NodeId, SocketAddr)>,
    /// Control-plane voter ids (initial product: first three nodes;
    /// added data nodes join control as learners).
    pub control_voters: Vec<NodeId>,
    /// Control-plane seeds proving the cluster already exists (joiners
    /// only; empty on founders and ignored on restart).
    pub control_seeds: Vec<SocketAddr>,
}

/// Shared cluster serving state: the multi-tablet node, its dynamic
/// directory for key → tablet routing, plus the static address maps
/// redirects resolve through.
///
/// The directory starts as the static genesis tiling and evolves through
/// online split/merge cutovers: each cutover publishes one validated
/// snapshot atomically (RCU via `ArcSwap`), so readers never observe a
/// gap or double authority. Restarts rebuild the same directory by
/// replaying completed topology plans from the replicated control image
/// over the genesis tiling.
#[derive(Debug, Clone)]
pub struct ClusterShared {
    /// The multi-tablet consensus node.
    pub node: Arc<ConsensusNode>,
    /// Dynamic tablet directory: every key deterministically maps to
    /// exactly one active tablet (no gaps, no overlaps) at every
    /// published version.
    pub directory: Arc<arc_swap::ArcSwap<DirectorySnapshot>>,
    /// Native endpoint per node id (redirect targets).
    pub natives: Arc<std::collections::HashMap<NodeId, SocketAddr>>,
    /// This node's native endpoint (advertised in hellos).
    pub native: SocketAddr,
    /// Namespace served.
    pub namespace: NamespaceId,
    /// Live reconciler policy (operator-tunable via
    /// `POST /v1/control/policy`; the reconciler clones it every pass).
    pub policy: Arc<std::sync::Mutex<super::control::ReconcilePolicy>>,
    /// Distributed redundancy coordinator (`None` when its lane failed to
    /// open; every redundancy endpoint then answers 503).
    pub redundancy: Option<Arc<kivi_consensus::distcoord::RedundancyCoordinator>>,
}

impl ClusterShared {
    /// Loads the current directory snapshot (lock-free RCU read).
    pub fn directory_snapshot(&self) -> arc_swap::Guard<Arc<DirectorySnapshot>> {
        self.directory.load()
    }

    /// Publishes a new directory snapshot after validation. Returns the
    /// published version. Retries are safe (same snapshot publishes
    /// idempotently when versions match).
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason when the snapshot fails
    /// [`DirectorySnapshot::validate`].
    pub fn publish_directory(&self, snapshot: DirectorySnapshot) -> Result<u64, String> {
        snapshot
            .validate()
            .map_err(|error| format!("invalid directory snapshot: {error}"))?;
        let version = snapshot.version().as_u64();
        self.directory.store(Arc::new(snapshot));
        Ok(version)
    }
}

/// Why key → tablet routing failed (small error enum: the wire
/// `Response` bodies are built at the call sites).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouteError {
    /// The partition hash algorithm is unavailable in this build.
    UnsupportedHash,
    /// No active tablet owns the key (unreachable on a static tiling —
    /// loud internal, never a silent misroute).
    NoTablet,
}

/// Routes key bytes to their tablet through the current directory by the
/// unified rule (mirrored in `kivi-engine/src/compound.rs` and
/// `kivi-server/src/compound.rs`; keep the three in sync):
/// record-key fiat, projection/index-primary stripping, then layout routing.
///
/// Returns the retired tablet itself when the key's current owner is a
/// tombstone: the caller redirects through
/// [`successor_ranges`](kivi_tablet::DirectorySnapshot::successor_ranges)
/// instead of treating the request as unroutable.
pub(crate) fn route_key(shared: &ClusterShared, key: &[u8]) -> Result<TabletId, RouteError> {
    let snapshot = shared.directory_snapshot();
    if let Some((coordinator, _)) = kivi_state::parse_txn_record_key(key)
        && snapshot
            .get(coordinator)
            .is_some_and(|descriptor| descriptor.state().is_writable())
    {
        return Ok(coordinator);
    }
    if let Some(primary) = kivi_state::parse_projection_primary(key) {
        return route_normal_key(&snapshot, shared.namespace, primary);
    }
    if let Some(primary) = kivi_state::parse_index_primary(key) {
        return route_normal_key(&snapshot, shared.namespace, &primary);
    }
    route_normal_key(&snapshot, shared.namespace, key)
}

/// Redirects a request naming a retired tablet through its live
/// successors: resolves the tombstone chain transitively to the active
/// successor owning `key` (split children partition the parent range;
/// chained split→merge→split lineages resolve through intermediate
/// tombstones). Returns `None` when no live successor covers the key in
/// this snapshot, letting the caller fall back to a retryable answer
/// instead of guessing from untrustworthy local state.
async fn retired_redirect(
    shared: &ClusterShared,
    tablet: TabletId,
    key: &[u8],
) -> Option<Response> {
    let directory = shared.directory_snapshot();
    let (successor, range) = directory.resolve_successor(tablet, key)?;
    let endpoint = successor_endpoint(shared, successor).await?;
    Some(Response {
        proof: None,
        status: Status::StaleRoute,
        body: ResponseBody::Redirect(RedirectInfo {
            dir_version: directory.version(),
            tablet: successor,
            epoch: TabletEpoch::INITIAL,
            worker: WorkerId::from_u64(CLUSTER_WORKER),
            endpoint: endpoint.to_string(),
            range: range.clone(),
        }),
    })
}

/// Dialable native endpoint for one tablet's serving replicas: live
/// migration sources first (voting until the terminal retire step),
/// then desired voters, then static founders. Mirrors the candidate
/// order of [`control_plane_redirect`](self::control_plane_redirect):
/// a migrating successor's desired set names a learner that can never
/// lead, so naming it would bounce the client into `Overloaded`.
async fn successor_endpoint(shared: &ClusterShared, tablet: TabletId) -> Option<SocketAddr> {
    let state = shared.node.control_state().await?;
    let desired = state.desired(tablet)?;
    let mut candidates: Vec<NodeId> = Vec::new();
    for plan in state.live_plans_for(tablet) {
        if !candidates.contains(&plan.from) {
            candidates.push(plan.from);
        }
    }
    for voter in &desired.replicas {
        if !candidates.contains(voter) {
            candidates.push(*voter);
        }
    }
    candidates.into_iter().find_map(|voter| {
        state
            .node(voter)
            .map(|record| record.native)
            .or_else(|| shared.natives.get(&voter).copied())
    })
}

/// Layout routing for ordinary keys.
fn route_normal_key(
    snapshot: &kivi_tablet::DirectorySnapshot,
    namespace: NamespaceId,
    key: &[u8],
) -> Result<TabletId, RouteError> {
    use kivi_state::PartitionHasher;
    if let Some(tablet) = snapshot.lookup_by_key(key) {
        return Ok(tablet);
    }
    let hash = PartitionHasher::V1
        .hash(namespace, key)
        .ok_or(RouteError::UnsupportedHash)?;
    snapshot.lookup_by_hash(hash).ok_or(RouteError::NoTablet)
}

/// Shapes a routing failure as a wire response.
fn route_error(error: RouteError) -> Response {
    match error {
        RouteError::UnsupportedHash => Response {
            proof: None,
            status: Status::InvalidRequest,
            body: ResponseBody::Diagnostic("partition hash unsupported".to_owned()),
        },
        RouteError::NoTablet => Response {
            proof: None,
            status: Status::Internal,
            body: ResponseBody::Diagnostic("no tablet owns key".to_owned()),
        },
    }
}

/// Whether `opcode` is a direct single-key mutating write subject to the
/// index-prefix admission rule (reads, scans, batches, and transaction
/// steps bypass it).
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
    if args.cluster_admins.is_empty() {
        anyhow::bail!("cluster mode requires --cluster-admins node=host:port,...");
    }
    let natives = args.cluster_natives.clone();
    // Control voters default to the three lowest peer ids (initial
    // product: the first three nodes stay the control voters).
    let mut peer_ids: Vec<u64> = args.cluster_peers.iter().map(|(id, _)| *id).collect();
    peer_ids.sort_unstable();
    peer_ids.dedup();
    let control_voters: Vec<NodeId> = if args.control_voters.is_empty() {
        peer_ids.into_iter().take(3).map(NodeId::from_u64).collect()
    } else {
        let mut voters = args.control_voters.clone();
        voters.sort_unstable();
        voters.dedup();
        voters.into_iter().map(NodeId::from_u64).collect()
    };
    let config = ClusterServeConfig {
        cluster: ClusterId::from_u128(cluster),
        node: NodeId::from_u64(node),
        tablet_count: args.cluster_tablets,
        worker_count: args.cluster_workers,
        namespace: NamespaceId::from_u64(1),
        layout: args.layout.clone(),
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
        admins: args
            .cluster_admins
            .iter()
            .map(|(id, addr)| (NodeId::from_u64(*id), *addr))
            .collect(),
        control_voters,
        control_seeds: args.control_seeds.clone(),
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

/// Builds the static directory: hash-prefix buddy tiles or ordered
/// first-byte tiles, per the configured layout (validated transitions
/// only).
fn build_directory(config: &ClusterServeConfig) -> anyhow::Result<DirectorySnapshot> {
    if config.layout == "ordered" {
        let mut split_keys = Vec::new();
        for index in 1..config.tablet_count.max(1) {
            #[allow(clippy::cast_possible_truncation)]
            let boundary = (index * 256 / config.tablet_count.max(1)) as u8;
            if boundary != 0
                && split_keys
                    .last()
                    .is_none_or(|last: &Vec<u8>| *last != vec![boundary])
            {
                split_keys.push(vec![boundary]);
            }
        }
        return DirectorySnapshot::static_ordered_tiles(config.namespace, &split_keys)
            .context("static ordered directory failed to build");
    }
    DirectorySnapshot::static_tiles(
        config.namespace,
        config.tablet_count,
        TabletEpoch::INITIAL,
        WriteGuardGeneration::INITIAL,
    )
    .context("static tablet directory failed to build")
}

/// One committed topology cutover awaiting directory replay (splits and
/// merges share the control `next_plan_id` sequence, so one sorted pass
/// replays split→merge→split lineages with parents present).
enum ReplayOp {
    Split(kivi_control::SplitPlan),
    Merge(kivi_control::MergePlan),
}

impl ReplayOp {
    fn id(&self) -> u64 {
        match self {
            ReplayOp::Split(plan) => plan.id.as_u64(),
            ReplayOp::Merge(plan) => plan.id.as_u64(),
        }
    }
}

/// Replays completed topology cutovers over the genesis tiling so a full
/// cluster restart recovers the post-split/merge directory. Only plans at
/// or past `CutoverCommitted` affect the directory (earlier phases keep
/// children inactive and the parent authoritative); replay applies them in
/// global plan-id order (splits and merges interleave: split→merge→split
/// lineages only rebuild when each cutover sees its parents), deterministically
/// rebuilding the same ranges. Old parents stay retired with redirects —
/// they never reappear as active.
#[allow(clippy::too_many_lines)]
async fn replay_topology_from_control(shared: &ClusterShared) {
    replay_committed_topology(&shared.node, &shared.directory).await;
}

/// Replays committed split/merge cutovers into `directory` from the
/// replicated control image. Idempotent: snapshots that already contain a
/// cutover re-apply as no-ops, and publication happens only when the
/// snapshot actually changed.
///
/// Runs at startup (restart recovery) AND on every reconciler pass on
/// every node: a freshly admitted member starts from its genesis tiling
/// and missed every past cutover, and a member down during a cutover push
/// misses it — without continuous replay their serving directories diverge
/// permanently and produce contradictory tombstone redirects. The admin
/// cutover push stays the fast path (same-pass convergence); this is the
/// correctness backstop that makes "every member eventually applies every
/// committed cutover" true rather than hoped-for.
pub(crate) async fn replay_committed_topology(
    node: &ConsensusNode,
    directory: &Arc<arc_swap::ArcSwap<DirectorySnapshot>>,
) {
    let Some(state) = node.control_state().await else {
        return;
    };
    let mut ops: Vec<ReplayOp> = Vec::new();
    for plan in state.splits().cloned() {
        if matches!(
            plan.phase,
            kivi_control::SplitPhase::CutoverCommitted
                | kivi_control::SplitPhase::ParentRetiring
                | kivi_control::SplitPhase::Completed
        ) {
            ops.push(ReplayOp::Split(plan));
        }
    }
    for plan in state.merges().cloned() {
        if matches!(
            plan.phase,
            kivi_control::MergePhase::CutoverCommitted
                | kivi_control::MergePhase::ParentsRetiring
                | kivi_control::MergePhase::Completed
        ) {
            ops.push(ReplayOp::Merge(plan));
        }
    }
    // Global creation order: split and merge ids share the control
    // `next_plan_id` sequence, so one sorted pass replays lineage chains
    // (split→merge→split) with parents present. A fixpoint retry covers
    // any residual ordering skew without masking real gaps.
    ops.sort_by_key(ReplayOp::id);
    let mut snapshot = directory.load().as_ref().clone();
    let mut changed = false;
    for _ in 0..ops.len().max(1) {
        let mut progress = false;
        let mut pending: Vec<ReplayOp> = Vec::new();
        for op in ops.drain(..) {
            let result = match &op {
                ReplayOp::Split(plan) => apply_completed_split(&snapshot, plan)
                    .map_err(|reason| (plan.id.as_u64(), plan.parent.as_u64(), reason)),
                ReplayOp::Merge(plan) => apply_completed_merge(&snapshot, plan)
                    .map_err(|reason| (plan.id.as_u64(), plan.merged.as_u64(), reason)),
            };
            match result {
                Ok(next) => {
                    // Idempotent replays return the same snapshot; only
                    // count real transitions as progress to terminate.
                    if next != snapshot {
                        snapshot = next;
                        changed = true;
                    }
                    progress = true;
                }
                Err((id, tablet, reason)) => {
                    // Parent unknown yet (later pass may create it);
                    // defer one round before warning.
                    pending.push(op);
                    tracing::debug!(
                        plan = id,
                        tablet,
                        %reason,
                        "topology replay deferred cutover"
                    );
                }
            }
        }
        ops = pending;
        if ops.is_empty() {
            break;
        }
        if !progress {
            for op in &ops {
                match op {
                    ReplayOp::Split(plan) => tracing::warn!(
                        plan = plan.id.as_u64(),
                        parent = plan.parent.as_u64(),
                        "topology replay skipped split cutover"
                    ),
                    ReplayOp::Merge(plan) => tracing::warn!(
                        plan = plan.id.as_u64(),
                        merged = plan.merged.as_u64(),
                        "topology replay skipped merge cutover"
                    ),
                }
            }
            break;
        }
    }
    if changed {
        if let Err(reason) = snapshot.validate() {
            tracing::warn!(%reason, "topology replay publish failed");
            return;
        }
        let version = snapshot.version().as_u64();
        directory.store(Arc::new(snapshot));
        tracing::info!(dir_version = version, "topology replay published directory");
    }
}

/// Derives ordered/hash children from a parent range and a plan boundary:
/// hash splits verify the midpoint, ordered splits verify the real key is
/// strictly interior (`start < key < end`).
fn split_child_ranges(
    parent_range: &kivi_tablet::PartitionRange,
    split_hash: u128,
    split_key: Option<&[u8]>,
) -> Result<(kivi_tablet::PartitionRange, kivi_tablet::PartitionRange), String> {
    use kivi_tablet::PartitionRange;
    match (parent_range, split_key) {
        (PartitionRange::Hash(prefix), None) => {
            let (left_prefix, right_prefix, expect_hash) =
                prefix.split_midpoint().map_err(|error| error.to_string())?;
            if expect_hash != split_hash {
                return Err(format!(
                    "split boundary mismatch: plan {split_hash:x} != range {expect_hash:x}"
                ));
            }
            Ok((
                PartitionRange::Hash(left_prefix),
                PartitionRange::Hash(right_prefix),
            ))
        }
        (PartitionRange::Ordered(range), Some(key)) => {
            let (left, right) = range.split_at(key).map_err(|error| error.to_string())?;
            Ok((
                PartitionRange::Ordered(left),
                PartitionRange::Ordered(right),
            ))
        }
        (PartitionRange::Hash(_), Some(_)) => Err("hash parent with key boundary".to_owned()),
        (PartitionRange::Ordered(_), None) => Err("ordered parent needs a split key".to_owned()),
    }
}

/// Applies one completed split cutover to `directory`: derives the
/// children from the parent range (verifying the persisted boundary),
/// then allocate → stage → seal → atomic cutover.
fn apply_completed_split(
    directory: &DirectorySnapshot,
    plan: &kivi_control::SplitPlan,
) -> Result<DirectorySnapshot, String> {
    // Idempotent: already cut over (parent tombstoned, children active).
    let parent_done = directory
        .get(plan.parent)
        .is_some_and(|descriptor| descriptor.state() == kivi_tablet::TabletState::Tombstone);
    let children_active = [plan.left, plan.right].iter().all(|child| {
        directory
            .get(*child)
            .is_some_and(|descriptor| descriptor.state() == kivi_tablet::TabletState::Active)
    });
    if parent_done && children_active {
        return Ok(directory.clone());
    }
    let parent_range = directory
        .get(plan.parent)
        .map(|descriptor| descriptor.range().clone())
        .ok_or_else(|| format!("split parent {} unknown", plan.parent.as_u64()))?;
    let (left_range, _right_range) =
        split_child_ranges(&parent_range, plan.split_hash, plan.split_key.as_deref())?;
    let mut next = directory
        .allocate(
            plan.left,
            left_range,
            TabletEpoch::INITIAL,
            WriteGuardGeneration::INITIAL,
        )
        .or_else(|error| {
            // Already allocated (replay after partial progress): continue.
            if directory.get(plan.left).is_some() {
                Ok(directory.clone())
            } else {
                Err(error)
            }
        })
        .map_err(|error| error.to_string())?;
    if next.get(plan.right).is_none() {
        // Recompute (allocate borrows nothing; ranges are deterministic).
        let (_, right_range) =
            split_child_ranges(&parent_range, plan.split_hash, plan.split_key.as_deref())?;
        next = next
            .allocate(
                plan.right,
                right_range,
                TabletEpoch::INITIAL,
                WriteGuardGeneration::INITIAL,
            )
            .map_err(|error| error.to_string())?;
    }
    for child in [plan.left, plan.right] {
        if next
            .get(child)
            .is_some_and(|descriptor| descriptor.state() == kivi_tablet::TabletState::Allocated)
        {
            next = next.stage(child).map_err(|error| error.to_string())?;
        }
    }
    if next
        .get(plan.parent)
        .is_some_and(|descriptor| descriptor.state() == kivi_tablet::TabletState::Active)
    {
        next = next.seal(plan.parent).map_err(|error| error.to_string())?;
    }
    next = next
        .cutover_split(plan.parent, plan.left, plan.right)
        .map_err(|error| error.to_string())?;
    Ok(next)
}

/// Applies one split cutover to a local directory snapshot from explicit
/// plan fields (admin endpoint path): idempotent, deterministic, one
/// validated atomic transition. Used by the reconciler on every node and
/// by restart replay (via [`apply_completed_split`]).
pub(crate) fn apply_cutover_split_for_reconciler(
    directory: &DirectorySnapshot,
    parent: TabletId,
    left: TabletId,
    right: TabletId,
    split_hash: u128,
    split_key: Option<&[u8]>,
) -> Result<DirectorySnapshot, String> {
    apply_cutover_split_local(directory, parent, left, right, split_hash, split_key)
}

/// Applies one split cutover to a local directory snapshot from explicit
/// plan fields (admin endpoint path): idempotent, deterministic, one
/// validated atomic transition. Used by the reconciler on every node and
/// by restart replay (via [`apply_completed_split`]).
fn apply_cutover_split_local(
    directory: &DirectorySnapshot,
    parent: TabletId,
    left: TabletId,
    right: TabletId,
    split_hash: u128,
    split_key: Option<&[u8]>,
) -> Result<DirectorySnapshot, String> {
    let parent_done = directory
        .get(parent)
        .is_some_and(|descriptor| descriptor.state() == kivi_tablet::TabletState::Tombstone);
    let children_active = [left, right].iter().all(|child| {
        directory
            .get(*child)
            .is_some_and(|descriptor| descriptor.state() == kivi_tablet::TabletState::Active)
    });
    if parent_done && children_active {
        return Ok(directory.clone());
    }
    let parent_range = directory
        .get(parent)
        .map(|descriptor| descriptor.range().clone())
        .ok_or_else(|| format!("split parent {} unknown", parent.as_u64()))?;
    let (left_range, _right_range) = split_child_ranges(&parent_range, split_hash, split_key)?;
    let mut next = if directory.get(left).is_none() {
        directory
            .allocate(
                left,
                left_range,
                TabletEpoch::INITIAL,
                WriteGuardGeneration::INITIAL,
            )
            .map_err(|error| error.to_string())?
    } else {
        directory.clone()
    };
    if next.get(right).is_none() {
        let (_, right_range) = split_child_ranges(&parent_range, split_hash, split_key)?;
        next = next
            .allocate(
                right,
                right_range,
                TabletEpoch::INITIAL,
                WriteGuardGeneration::INITIAL,
            )
            .map_err(|error| error.to_string())?;
    }
    for child in [left, right] {
        if next
            .get(child)
            .is_some_and(|descriptor| descriptor.state() == kivi_tablet::TabletState::Allocated)
        {
            next = next.stage(child).map_err(|error| error.to_string())?;
        }
    }
    if next
        .get(parent)
        .is_some_and(|descriptor| descriptor.state() == kivi_tablet::TabletState::Active)
    {
        next = next.seal(parent).map_err(|error| error.to_string())?;
    }
    next.cutover_split(parent, left, right)
        .map_err(|error| error.to_string())
}

/// Derives a merge target range from two parent ranges: hash parents must
/// be midpoint siblings; ordered parents must be exactly adjacent
/// (`left.end == right.start`, same namespace layout by construction).
fn merge_parent_ranges(
    left_range: &kivi_tablet::PartitionRange,
    right_range: &kivi_tablet::PartitionRange,
) -> Result<kivi_tablet::PartitionRange, String> {
    use kivi_tablet::{HashPrefix, PartitionRange};
    match (left_range, right_range) {
        (PartitionRange::Hash(left_prefix), PartitionRange::Hash(right_prefix)) => {
            if left_prefix.prefix_len() != right_prefix.prefix_len()
                || left_prefix.prefix_len() == 0
            {
                return Err("merge parents are not siblings".to_owned());
            }
            let len = left_prefix.prefix_len();
            let half = 1u128 << (128 - len);
            if left_prefix.bits() | half != right_prefix.bits() || left_prefix.bits() & half != 0 {
                return Err("merge parents are not adjacent siblings".to_owned());
            }
            let merged_prefix =
                HashPrefix::new(left_prefix.bits(), len - 1).map_err(|error| error.to_string())?;
            Ok(PartitionRange::Hash(merged_prefix))
        }
        (PartitionRange::Ordered(left), PartitionRange::Ordered(right)) => {
            let merged = left.merge_with(right).map_err(|error| error.to_string())?;
            Ok(PartitionRange::Ordered(merged))
        }
        _ => Err("merge parents mix layouts".to_owned()),
    }
}

/// Applies one merge cutover to a local directory snapshot from explicit
/// plan fields (admin endpoint path): idempotent, deterministic, one
/// validated atomic transition.
pub(crate) fn apply_cutover_merge_for_reconciler(
    directory: &DirectorySnapshot,
    left: TabletId,
    right: TabletId,
    merged: TabletId,
) -> Result<DirectorySnapshot, String> {
    apply_cutover_merge_local(directory, left, right, merged)
}

/// Applies one merge cutover to a local directory snapshot from explicit
/// plan fields (admin endpoint path): idempotent, deterministic, one
/// validated atomic transition.
fn apply_cutover_merge_local(
    directory: &DirectorySnapshot,
    left: TabletId,
    right: TabletId,
    merged: TabletId,
) -> Result<DirectorySnapshot, String> {
    if directory
        .get(merged)
        .is_some_and(|descriptor| descriptor.state() == kivi_tablet::TabletState::Active)
    {
        return Ok(directory.clone());
    }
    let left_range = directory
        .get(left)
        .map(|descriptor| descriptor.range().clone())
        .ok_or_else(|| format!("merge left {} unknown", left.as_u64()))?;
    let right_range = directory
        .get(right)
        .map(|descriptor| descriptor.range().clone())
        .ok_or_else(|| format!("merge right {} unknown", right.as_u64()))?;
    let merged_range = merge_parent_ranges(&left_range, &right_range)?;
    let mut next = if directory.get(merged).is_none() {
        directory
            .allocate(
                merged,
                merged_range,
                TabletEpoch::INITIAL,
                WriteGuardGeneration::INITIAL,
            )
            .map_err(|error| error.to_string())?
    } else {
        directory.clone()
    };
    if next
        .get(merged)
        .is_some_and(|descriptor| descriptor.state() == kivi_tablet::TabletState::Allocated)
    {
        next = next.stage(merged).map_err(|error| error.to_string())?;
    }
    for parent in [left, right] {
        if next
            .get(parent)
            .is_some_and(|descriptor| descriptor.state() == kivi_tablet::TabletState::Active)
        {
            next = next.seal(parent).map_err(|error| error.to_string())?;
        }
    }
    next.cutover_merge(left, right, merged)
        .map_err(|error| error.to_string())
}

/// Applies one completed merge cutover to `directory`: derives the union
/// range from the sibling parents, then allocate → stage → seal →
/// atomic cutover.
fn apply_completed_merge(
    directory: &DirectorySnapshot,
    plan: &kivi_control::MergePlan,
) -> Result<DirectorySnapshot, String> {
    let merged_done = directory
        .get(plan.merged)
        .is_some_and(|descriptor| descriptor.state() == kivi_tablet::TabletState::Active);
    if merged_done {
        return Ok(directory.clone());
    }
    let left_range = directory
        .get(plan.left)
        .map(|descriptor| descriptor.range().clone())
        .ok_or_else(|| format!("merge left {} unknown", plan.left.as_u64()))?;
    let right_range = directory
        .get(plan.right)
        .map(|descriptor| descriptor.range().clone())
        .ok_or_else(|| format!("merge right {} unknown", plan.right.as_u64()))?;
    let merged_range = merge_parent_ranges(&left_range, &right_range)?;
    let mut next = if directory.get(plan.merged).is_none() {
        directory
            .allocate(
                plan.merged,
                merged_range,
                TabletEpoch::INITIAL,
                WriteGuardGeneration::INITIAL,
            )
            .map_err(|error| error.to_string())?
    } else {
        directory.clone()
    };
    if next
        .get(plan.merged)
        .is_some_and(|descriptor| descriptor.state() == kivi_tablet::TabletState::Allocated)
    {
        next = next.stage(plan.merged).map_err(|error| error.to_string())?;
    }
    for parent in [plan.left, plan.right] {
        if next
            .get(parent)
            .is_some_and(|descriptor| descriptor.state() == kivi_tablet::TabletState::Active)
        {
            next = next.seal(parent).map_err(|error| error.to_string())?;
        }
    }
    next = next
        .cutover_merge(plan.left, plan.right, plan.merged)
        .map_err(|error| error.to_string())?;
    Ok(next)
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
    // System control participation: every node joins the control group
    // (voter when listed, learner otherwise). Voter dial info rides the
    // control config; seeds are explicit operator intent
    // (`--control-seeds`, joiners only): founding voters boot without
    // them and initialize exactly once, joiners wait for `add_learner`,
    // restarts ignore them.
    let peer_by_node: std::collections::HashMap<NodeId, SocketAddr> =
        config.peers.iter().copied().collect();
    let mut control_voters: std::collections::BTreeSet<u64> = config
        .control_voters
        .iter()
        .map(|node| node.as_u64())
        .collect();
    if control_voters.is_empty() {
        control_voters = peer_by_node.keys().map(|node| node.as_u64()).collect();
    }
    let control_peer_addrs: std::collections::BTreeMap<u64, String> = control_voters
        .iter()
        .filter_map(|voter| {
            peer_by_node
                .get(&NodeId::from_u64(*voter))
                .map(|addr| (*voter, addr.to_string()))
        })
        .collect();
    let control_seeds: Vec<SocketAddr> = config.control_seeds.clone();
    // Joiners (fresh nodes outside the voter set) start data-empty: the
    // control plane assigns their replicas through migration, so no
    // static tablet list is required at join time (§9).
    let joiner = !control_voters.contains(&config.node.as_u64())
        && !config.data_dir.join("consensus-sm").exists();
    let data_tablets: Vec<TabletId> = if joiner { Vec::new() } else { tablets };
    let node = Arc::new(
        ConsensusNode::open(MultiNodeConfig {
            data_dir: config.data_dir.clone(),
            namespace: config.namespace,
            tablets: data_tablets,
            local: config.node,
            topology,
            segment_target_bytes: config.segment_target_bytes,
            transport: kivi_consensus::TransportConfig::default(),
            peer_certs,
            insecure_peer_tls,
            preflight_enabled,
            lease_params: kivi_types::LeaseParams::default(),
            worker_count: config.worker_count,
            durability: kivi_consensus::SharedDurabilityConfig::default(),
            control: Some(kivi_consensus::ControlGroupConfig {
                voters: control_voters,
                peer_addrs: control_peer_addrs,
                seeds: control_seeds,
            }),
            // Node-local redundancy fragments live under
            // `<data_dir>/redundancy/fragments` (sibling of `sidecar/`,
            // `consensus-sm/`, ...). Restart reopens and re-indexes the
            // store while the catalog recovers through control-log replay:
            // the two-source recovery documented on the coordinator.
            fragment_store_root: Some(config.data_dir.join("redundancy").join("fragments")),
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
#[allow(clippy::too_many_lines)]
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
    let policy = Arc::new(std::sync::Mutex::new(policy_from_env()));
    // Distributed redundancy plane: the coordinator reuses this node's
    // existing mesh (identity/runtime/dials), the replicated control group
    // as layout authority, and the node-local fragment store opened above.
    // A coordinator that cannot open leaves the plane disabled (503) and
    // never blocks startup: durability of Raft-referenced sidecars stays
    // local-first, redundancy only adds protection.
    let redundancy = match kivi_consensus::distcoord::RedundancyCoordinator::open(
        Arc::clone(&node),
        kivi_consensus::distcoord::CoordinatorConfig::conservative(),
    ) {
        Ok(coordinator) => Some(Arc::new(coordinator)),
        Err(error) => {
            tracing::warn!(%error, "redundancy plane disabled");
            None
        }
    };
    let shared = ClusterShared {
        node: Arc::clone(&node),
        directory: Arc::new(arc_swap::ArcSwap::from_pointee(directory)),
        natives: Arc::new(natives),
        native: native_addr,
        namespace: config.namespace,
        policy: Arc::clone(&policy),
        redundancy,
    };
    // Replay completed topology plans over the genesis tiling so a full
    // cluster restart recovers the post-split/merge directory (old parents
    // stay retired with redirects; they never reappear as active).
    replay_topology_from_control(&shared).await;
    // Control-plane genesis + reconciler (background Tokio tasks; both
    // idle on non-leaders and exit with the server).
    let seed = genesis_seed(&config, &shared.directory_snapshot());
    let (genesis, reconciler) = spawn_control_loops(
        &node,
        seed,
        Arc::clone(&shared.directory),
        Arc::clone(&policy),
    );
    let redundancy_healer = shared.redundancy.as_ref().map(spawn_redundancy_healer);
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
    // Optional RESP edge (edge adapter; never `MOVED`/`ASK`).
    #[cfg(feature = "redis-compat")]
    let (resp_admin, resp_shutdown) =
        start_resp_edge(&node, &shared, config.redis_bind, config.namespace).await?;
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
        shared.directory_snapshot().version().as_u64(),
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
    genesis.abort();
    reconciler.abort();
    if let Some(coordinator) = shared.redundancy.as_ref() {
        coordinator.stop_healing();
    }
    if let Some(healer) = redundancy_healer {
        healer.abort();
    }
    node.shutdown().await;
    tracing::info!("kivi-server (cluster) stopped");
    Ok(())
}

/// Starts the optional RESP edge against the replicated node. Absent
/// bind means native-only even when compiled in.
#[cfg(feature = "redis-compat")]
async fn start_resp_edge(
    node: &Arc<ConsensusNode>,
    shared: &ClusterShared,
    bind: Option<SocketAddr>,
    namespace: NamespaceId,
) -> anyhow::Result<(
    Option<super::resp::RespAdmin>,
    Option<tokio::sync::watch::Sender<bool>>,
)> {
    let Some(addr) = bind else {
        return Ok((None, None));
    };
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("redis bind failed on {addr}"))?;
    let endpoint = listener.local_addr().context("redis listener address")?;
    let stats = Arc::new(super::resp::RespStats::default());
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(super::resp::serve_with(
        listener,
        ClusterExecutor::new(Arc::clone(node), Arc::clone(&shared.directory), namespace),
        Arc::clone(&stats),
        shutdown_rx,
    ));
    Ok((
        Some(super::resp::RespAdmin {
            endpoint,
            namespace: namespace.as_u64(),
            stats,
        }),
        Some(shutdown_tx),
    ))
}

/// Resolves the reconciler policy from the environment:
///
/// * `KIVI_AUTO_SPLIT=1` enables automatic splits (default off).
/// * `KIVI_AUTO_MERGE=1` enables automatic merges (default off).
/// * `KIVI_SPLIT_OBJECTS` overrides the split object-count threshold.
/// * `KIVI_MERGE_OBJECTS` overrides the merge object-count threshold.
fn policy_from_env() -> super::control::ReconcilePolicy {
    let mut policy = super::control::ReconcilePolicy::default();
    if std::env::var("KIVI_AUTO_SPLIT").is_ok_and(|value| value == "1") {
        policy.auto_split = true;
    }
    if std::env::var("KIVI_AUTO_MERGE").is_ok_and(|value| value == "1") {
        policy.auto_merge = true;
    }
    if let Some(threshold) = std::env::var("KIVI_SPLIT_OBJECTS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|threshold| *threshold >= 10)
    {
        policy.split_object_threshold = threshold;
    }
    if let Some(threshold) = std::env::var("KIVI_MERGE_OBJECTS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|threshold| *threshold >= 2)
    {
        policy.merge_object_threshold = threshold;
    }
    policy
}

/// Spawns control-plane genesis plus the reconciler (migrations,
/// repairs, splits, merges; idle on non-leaders; both abort with the
/// server). The reconciler clones the shared policy every pass so
/// `POST /v1/control/policy` retunes the live loop without a restart.
fn spawn_control_loops(
    node: &Arc<ConsensusNode>,
    seed: super::control::GenesisSeed,
    directory: Arc<arc_swap::ArcSwap<DirectorySnapshot>>,
    policy: Arc<std::sync::Mutex<super::control::ReconcilePolicy>>,
) -> (tokio::task::JoinHandle<()>, tokio::task::JoinHandle<()>) {
    let genesis = {
        let node = Arc::clone(node);
        tokio::spawn(async move { super::control::genesis_loop(node, seed).await })
    };
    let reconciler = {
        let node = Arc::clone(node);
        tokio::spawn(async move {
            super::control::reconcile_loop_shared(node, policy, directory).await;
        })
    };
    (genesis, reconciler)
}

fn spawn_redundancy_healer(
    coordinator: &Arc<kivi_consensus::distcoord::RedundancyCoordinator>,
) -> tokio::task::JoinHandle<()> {
    let coordinator = Arc::clone(coordinator);
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(1));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            ticker.tick().await;
            if coordinator.healing_stopped() {
                break;
            }
            let owned = Arc::clone(&coordinator);
            let result = tokio::task::spawn_blocking(move || {
                let runtime = compio::runtime::Runtime::new().map_err(|error| error.to_string())?;
                runtime
                    .block_on(owned.maintenance_tick(false))
                    .map_err(|error| error.to_string())
            })
            .await;
            match result {
                Ok(Ok(_)) => {}
                Ok(Err(error)) => tracing::warn!(%error, "redundancy maintenance tick deferred"),
                Err(error) => tracing::warn!(%error, "redundancy maintenance task failed"),
            }
        }
    })
}

/// Builds the genesis seed from static startup flags: every configured
/// peer becomes a registry record (fingerprints from pin files when
/// present, unenforced zeros in lab-insecure mode), and every writable
/// directory tablet starts fully replicated across all seed nodes.
/// After bootstrap the replicated control plane owns placement;
/// restarts ignore these flags.
fn genesis_seed(
    config: &ClusterServeConfig,
    directory: &DirectorySnapshot,
) -> super::control::GenesisSeed {
    use std::collections::HashMap;
    let ders: HashMap<NodeId, Vec<u8>> = config
        .peer_certs
        .iter()
        .filter_map(|(node, path)| std::fs::read(path).ok().map(|der| (*node, der)))
        .collect();
    let natives: HashMap<NodeId, SocketAddr> = config.natives.iter().copied().collect();
    let admins: HashMap<NodeId, SocketAddr> = config.admins.iter().copied().collect();
    let nodes = config
        .peers
        .iter()
        .map(|(node, peer)| super::control::SeedNode {
            node: *node,
            peer: *peer,
            native: natives.get(node).copied().unwrap_or(*peer),
            admin: admins.get(node).copied().unwrap_or(config.admin_bind),
            fingerprint: ders.get(node).map_or([0u8; 32], |der| {
                kivi_codec::integrity::blake3_256(der.as_slice())
            }),
            domain: String::new(),
        })
        .collect();
    let tablets: Vec<TabletId> = directory
        .tablets()
        .iter()
        .filter(|tablet| tablet.state().is_writable())
        .map(kivi_tablet::TabletDescriptor::id)
        .collect();
    super::control::GenesisSeed {
        nodes,
        tablets,
        namespace: config.namespace,
        layout: config.layout.clone(),
    }
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
        dir_version: shared.directory_snapshot().version(),
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

fn schedule_sidecar_protection(
    shared: &ClusterShared,
    manifest: kivi_types::ManifestId,
    chunks: Vec<(kivi_types::ChunkId, u64)>,
    _logical_len: u64,
) {
    let Some(coordinator) = shared.redundancy.clone() else {
        return;
    };
    let domain = kivi_types::SecurityDomainId::from_u64(shared.namespace.as_u64());
    tokio::spawn(async move {
        let owned = Arc::clone(&coordinator);
        let result = tokio::task::spawn_blocking(move || {
            let runtime = compio::runtime::Runtime::new().map_err(|error| error.to_string())?;
            runtime.block_on(async move {
                for (chunk, len) in chunks {
                    if let Err(error) = owned.protect_chunk(chunk, domain, len, 1).await {
                        tracing::warn!(%error, ?chunk, "sidecar chunk protection deferred");
                        return Err::<(), String>(error.to_string());
                    }
                }
                if let Err(error) = owned.protect_manifest_auto(manifest, domain).await {
                    tracing::warn!(%error, ?manifest, "sidecar manifest protection deferred");
                    return Err::<(), String>(error.to_string());
                }
                Ok::<(), String>(())
            })
        })
        .await;
        if let Ok(Err(error)) = result {
            tracing::warn!(%error, "sidecar protection worker stopped");
        }
    });
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
                schedule_sidecar_protection(shared, manifest, upload.chunks.clone(), total);
                shape_result(kivi_protocol::Opcode::Set, tablet, &outcome)
            }
            ProposeOutcome::Rejected { outcome } => {
                {
                    let chunk_ids: Vec<kivi_types::ChunkId> =
                        upload.chunks.iter().map(|(id, _)| *id).collect();
                    let mut pins = shared.node.sidecar().pins().lock().await;
                    pins.unpin_root(&manifest, &chunk_ids);
                }
                shape_durable(&outcome, kivi_protocol::Opcode::Set, tablet)
            }
        },
        Err(error) => {
            {
                let chunk_ids: Vec<kivi_types::ChunkId> =
                    upload.chunks.iter().map(|(id, _)| *id).collect();
                let mut pins = shared.node.sidecar().pins().lock().await;
                pins.unpin_root(&manifest, &chunk_ids);
            }
            shape_propose_error(shared, tablet, &error).await
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

/// Handles `GetStream`: read under the request contract, then streams
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
            proof: None,
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
    // Streamed reads honor the request contract like point reads (same
    // visibility boundary for the root); the streamed frames carry no
    // proof trailer, so `AtLeast` chaining after a stream needs a point
    // read for its receipt. Captured before `into_operation` consumes the
    // request.
    let stream_contract = request.contract;
    let Some(operation) = request.into_operation() else {
        let response = Response {
            proof: None,
            status: Status::InvalidRequest,
            body: ResponseBody::Diagnostic("untranslatable operation".to_owned()),
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
    let ctx = read_ctx();
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
    // Large streamed values ride the same committed visibility boundary
    // as point reads: the manifest resolves from the served applied
    // state, so chunk delivery cannot cross it. Point-read eligibility
    // applies (one tablet, one key).
    let outcome = match shared
        .node
        .read(
            tablet,
            &operation,
            stream_contract,
            ctx,
            kivi_types::LeaseEligibility::Eligible,
        )
        .await
    {
        Ok(served) => served.outcome,
        Err(error) => {
            let response = shape_read_error(shared, tablet, &error).await;
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
            let response = shape_result(kivi_protocol::Opcode::GetStream, tablet, &outcome);
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
            proof: None,
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
    // The freshness contract is honored on every read opcode below and
    // stamped onto the response proof; mutating opcodes ignore it. Scans
    // and batches ride dedicated payloads with their own selectors, so a
    // non-Latest contract on them is a caller bug, rejected loudly.
    let contract = request.contract;
    if matches!(
        opcode,
        kivi_protocol::Opcode::Scan | kivi_protocol::Opcode::AtomicBatch
    ) && contract != ReadContract::Latest
    {
        return Some((
            Response {
                proof: None,
                status: Status::InvalidRequest,
                body: ResponseBody::Diagnostic(
                    "freshness contracts ride point reads, not scans or batches".to_owned(),
                ),
            },
            opcode,
        ));
    }
    if namespace != shared.namespace {
        return Some((
            Response {
                proof: None,
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
    // streams `ValueStream*` frames instead of one `Response`). `Scan`
    // and `AtomicBatch` dispatch on dedicated payloads before operation
    // translation (no single key to route).
    if opcode == kivi_protocol::Opcode::Scan {
        return Some(super::compound::handle_scan(shared, &request).await);
    }
    if opcode == kivi_protocol::Opcode::AtomicBatch {
        return Some(super::compound::handle_batch(shared, &request).await);
    }
    // Index entries are maintained transactionally: direct single-key
    // writes to the reserved index prefix are rejected (the coherent
    // paths — `AtomicBatch` and `TxnPrepare` — bypass this check).
    if is_direct_single_key_write(opcode) && kivi_state::is_index_key(&request.key) {
        return Some((
            Response {
                proof: None,
                status: Status::InvalidRequest,
                body: ResponseBody::Diagnostic(
                    "direct writes to index entries are forbidden; use indexed writes".to_owned(),
                ),
            },
            opcode,
        ));
    }
    let Some(operation) = request.into_operation() else {
        return Some((
            Response {
                proof: None,
                status: Status::InvalidRequest,
                body: ResponseBody::Diagnostic("untranslatable operation".to_owned()),
            },
            opcode,
        ));
    };
    let tablet = match route_key(shared, operation.key().as_bytes()) {
        Ok(tablet) => tablet,
        Err(error) => return Some((route_error(error), opcode)),
    };
    let now = wall_now();
    if opcode.is_mutating() {
        let identity = identity.map(|client| MutationIdentity { client, ack_floor });
        Some(handle_write(shared, tablet, &operation, identity, opcode, now).await)
    } else {
        Some(handle_read(shared, tablet, &operation, contract, opcode).await)
    }
}

/// Serves one mutating request: propose through the tablet's group and
/// shape the outcome (prepare outcomes name their tablet here, where the
/// tablet is known rather than in the store). A request naming a retired
/// tablet redirects through its live successors instead of failing: stale
/// routes are normal distributed-system behavior after split/merge
/// cutover, never a malformed connection.
async fn handle_write(
    shared: &ClusterShared,
    tablet: TabletId,
    operation: &kivi_state::Operation,
    identity: Option<MutationIdentity>,
    opcode: kivi_protocol::Opcode,
    now: WallTimestamp,
) -> (Response, kivi_protocol::Opcode) {
    // Attribute escrow ownership: a bounded-counter create takes its
    // initial full share on the serving tablet (clients send zero).
    let mut operation = operation.clone();
    if let kivi_state::Operation::BoundedCounterCreate { holder, .. } = &mut operation {
        *holder = tablet;
    }
    let response = match shared
        .node
        .propose(tablet, &operation, identity, None, now)
        .await
    {
        Ok(outcome) => match outcome {
            ProposeOutcome::Applied { outcome, .. }
            | ProposeOutcome::Duplicate { outcome }
            | ProposeOutcome::Read { outcome } => {
                // Tablet-naming prepare outcomes shape here (the
                // tablet is known at this layer, not in the store).
                if opcode == kivi_protocol::Opcode::TxnPrepare
                    && matches!(outcome, OperationResult::TxnPrepared)
                {
                    Response {
                        proof: None,
                        status: Status::Ok,
                        body: ResponseBody::TxnPrepared {
                            tablet: tablet.as_u64(),
                            dir_version: shared.directory_snapshot().version().as_u64(),
                        },
                    }
                } else {
                    resolve_result(shared, opcode, tablet, &operation, &outcome).await
                }
            }
            ProposeOutcome::Rejected { outcome } => {
                resolve_durable(shared, tablet, &operation, &outcome, opcode).await
            }
        },
        Err(error) => {
            // Retired-tablet redirects take precedence over the generic
            // migration answer: the directory names the live successors,
            // so the client converges in one hop with transport intact.
            if let ProposeError::Consensus(ConsensusError::Unavailable { reason }) = &error
                && reason.contains("not served by this node")
                && shared
                    .directory_snapshot()
                    .get(tablet)
                    .is_none_or(|descriptor| !descriptor.state().is_writable())
                && let Some(redirect) =
                    retired_redirect(shared, tablet, operation.key().as_bytes()).await
            {
                redirect
            } else {
                shape_propose_error(shared, tablet, &error).await
            }
        }
    };
    (response, opcode)
}

/// Serves one point read under the request contract: `Latest` barriers on
/// the tablet's leader (followers redirect), `AtLeast` serves from any
/// sufficiently applied replica, `BoundedStale` serves from proven-fresh
/// state or escalates, `Any` serves locally. Chunked roots resolve via
/// sidecars under the same visibility boundary (the manifest comes from
/// the served state, never newer). The served proof rides the response
/// for `AtLeast` chaining. Reads naming a retired tablet redirect through
/// its live successors like writes.
async fn handle_read(
    shared: &ClusterShared,
    tablet: TabletId,
    operation: &kivi_state::Operation,
    contract: kivi_types::ReadContract,
    opcode: kivi_protocol::Opcode,
) -> (Response, kivi_protocol::Opcode) {
    let ctx = read_ctx();
    // Directory-vs-serving cross-check: the routing view and the commit
    // view must name the same authority. Any drift on a hosted tablet
    // (stale directory, half-applied topology change) fails closed before
    // any state is read — evidence and tokens never validate across it.
    // Unhosted tablets skip the check: the read path below answers "not
    // served by this node" with a forward redirect (normal mid-migration),
    // which a mismatch error here would wrongly shadow.
    if let Some(recorded) = shared
        .directory_snapshot()
        .get(tablet)
        .and_then(kivi_tablet::TabletDescriptor::authority)
    {
        let drifted = match shared.node.authority_for(tablet) {
            Some(serving) => kivi_types::reconcile_authority(&serving, Some(&recorded)).is_err(),
            None => false,
        };
        if drifted {
            return (
                Response {
                    proof: None,
                    status: Status::Internal,
                    body: ResponseBody::Diagnostic("tablet authority mismatch; retry".to_owned()),
                },
                opcode,
            );
        }
    }
    let response = match shared
        .node
        .read(
            tablet,
            operation,
            contract,
            ctx,
            kivi_types::LeaseEligibility::Eligible,
        )
        .await
    {
        Ok(served) => {
            let mut response =
                resolve_result(shared, opcode, tablet, operation, &served.outcome).await;
            response.proof = Some(served.receipt);
            response
        }
        Err(error) => {
            if let kivi_consensus::ReadError::Consensus(ConsensusError::Unavailable { reason }) =
                &error
                && reason.contains("not served by this node")
                && shared
                    .directory_snapshot()
                    .get(tablet)
                    .is_none_or(|descriptor| !descriptor.state().is_writable())
                && let Some(redirect) =
                    retired_redirect(shared, tablet, operation.key().as_bytes()).await
            {
                redirect
            } else {
                shape_read_error(shared, tablet, &error).await
            }
        }
    };
    (response, opcode)
}

/// Resolves a deterministic outcome, fetching chunked bytes via sidecars
/// when the outcome references immutable state.
async fn resolve_result(
    shared: &ClusterShared,
    opcode: kivi_protocol::Opcode,
    tablet: TabletId,
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
                        proof: None,
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
                        proof: None,
                        status: Status::Ok,
                        body: ResponseBody::Value(bytes),
                    },
                    Err(_) => Response {
                        proof: None,
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
                        proof: None,
                        status: Status::Ok,
                        body: ResponseBody::Value(slice_range(&bytes, offset, len)),
                    },
                    Err(_) => Response {
                        proof: None,
                        status: Status::Internal,
                        body: ResponseBody::Diagnostic("chunked value unavailable".to_owned()),
                    },
                }
            }
            _ => shape_result(opcode, tablet, outcome),
        },
        _ => shape_result(opcode, tablet, outcome),
    }
}

/// Resolves a terminal outcome, fetching chunked bytes when completed.
async fn resolve_durable(
    shared: &ClusterShared,
    tablet: TabletId,
    operation: &kivi_state::Operation,
    outcome: &DurableOutcome,
    opcode: kivi_protocol::Opcode,
) -> Response {
    match outcome {
        DurableOutcome::Completed(result) => {
            resolve_result(shared, opcode, tablet, operation, result).await
        }
        _ => shape_durable(outcome, opcode, tablet),
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
#[allow(clippy::too_many_lines)]
fn shape_result(
    opcode: kivi_protocol::Opcode,
    tablet: TabletId,
    outcome: &OperationResult,
) -> Response {
    use kivi_state::OperationResult as R;
    match outcome {
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
            // `SetRange` restages internally but promises `Stored`.
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
        // Chunked roots never replicate in this stage; reaching the wire
        // unresolved is a serving bug, answered loudly, never wrong bytes.
        // Fabric references never replicate at any stage (worker-local
        // ids); reaching the wire is likewise a serving bug.
        R::ChunkedValue { .. } => Response {
            proof: None,
            status: Status::Internal,
            body: ResponseBody::Diagnostic("chunked value reached the wire unresolved".to_owned()),
        },
        R::FabricValue { .. } => Response {
            proof: None,
            status: Status::Internal,
            body: ResponseBody::Diagnostic("fabric value reached the wire unresolved".to_owned()),
        },
        // Tablet-naming prepare outcomes shape at the call site that knows
        // the tablet (see `handle_request`); reaching here is a serving
        // bug, answered loudly.
        R::TxnPrepared => Response {
            proof: None,
            status: Status::Internal,
            body: ResponseBody::Diagnostic("prepare reached the wire unnamed".to_owned()),
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
    }
}

/// Shapes a terminal (non-replicating) outcome: completed results shape
/// normally under the requesting opcode; rejections shape as their
/// stable statuses.
pub(crate) fn shape_durable(
    outcome: &DurableOutcome,
    opcode: kivi_protocol::Opcode,
    tablet: TabletId,
) -> Response {
    match outcome {
        DurableOutcome::Completed(result) => shape_result(opcode, tablet, result),
        DurableOutcome::Rejected(OpError::WrongType { .. }) => Response {
            proof: None,
            status: Status::WrongType,
            body: ResponseBody::Diagnostic("wrong type".to_owned()),
        },
        DurableOutcome::Rejected(OpError::CounterOverflow) => Response {
            proof: None,
            status: Status::CounterOverflow,
            body: ResponseBody::Diagnostic("overflow".to_owned()),
        },
        DurableOutcome::Rejected(OpError::StaleRangeBase) => Response {
            proof: None,
            status: Status::InvalidRequest,
            body: ResponseBody::Diagnostic("range base changed; retry".to_owned()),
        },
        DurableOutcome::Rejected(OpError::TxnConflict) => Response {
            proof: None,
            status: Status::TxnConflict,
            body: ResponseBody::Diagnostic("transaction conflict".to_owned()),
        },
        DurableOutcome::Rejected(OpError::BoundedExceeded) => Response {
            proof: None,
            status: Status::BoundedExceeded,
            body: ResponseBody::Diagnostic("bounded counter would exceed owned rights".to_owned()),
        },
        DurableOutcome::Rejected(OpError::SemaphoreExhausted) => Response {
            proof: None,
            status: Status::SemaphoreExhausted,
            body: ResponseBody::Diagnostic("semaphore capacity exhausted".to_owned()),
        },
        DurableOutcome::Rejected(OpError::LeaseConflict) => Response {
            proof: None,
            status: Status::LeaseConflict,
            body: ResponseBody::Diagnostic("lease held by another live owner".to_owned()),
        },
        DurableOutcome::Rejected(OpError::StaleFencing) => Response {
            proof: None,
            status: Status::StaleFencing,
            body: ResponseBody::Diagnostic("stale fencing token".to_owned()),
        },
        DurableOutcome::Rejected(OpError::StreamFull) => Response {
            proof: None,
            status: Status::StreamFull,
            body: ResponseBody::Diagnostic("stream shard full".to_owned()),
        },
        DurableOutcome::Rejected(OpError::NotFound) => Response {
            proof: None,
            status: Status::NotFound,
            body: ResponseBody::Diagnostic(String::new()),
        },
        DurableOutcome::VersionExhausted => Response {
            proof: None,
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

/// Redirects a request this node cannot serve to a desired replica from
/// replicated control state (migration-aware routing, §32). Returns
/// `None` when no desired replica is known, letting the caller fall
/// back to a retryable answer.
async fn desired_redirect(shared: &ClusterShared, tablet: TabletId) -> Option<Response> {
    if let Some(response) = control_plane_redirect(shared, tablet).await {
        return Some(response);
    }
    // No control image or no desired placement yet (a freshly admitted,
    // still data-empty member): redirect to a static founder instead of
    // answering `Overloaded`. The founders host every genesis tablet and
    // every split/merge child inherits a founder replica set until an
    // explicit migration moves it, so a founder always routes closer to
    // the authority than this node can. This keeps a non-serving member
    // from becoming a routing black hole that burns client redirect
    // budgets: the client follows one authoritative hop and converges.
    static_founder_redirect(shared, tablet)
}

/// Redirects through the replicated desired placement (registry native
/// endpoints first, static flags second).
///
/// Candidate order is serving truth, not placement goals: when a live
/// migration plan moves `tablet`, the desired set already names the
/// target (a learner that can never lead) while the source voter still
/// serves. So the plan's source leads, then desired voters, so redirects
/// always name a replica that can actually reach the leader instead of a
/// future voter that answers `Overloaded` and burns client budgets.
async fn control_plane_redirect(shared: &ClusterShared, tablet: TabletId) -> Option<Response> {
    let state = shared.node.control_state().await?;
    let desired = state.desired(tablet)?;
    let mut candidates: Vec<NodeId> = Vec::new();
    // Live migration sources first: they hold voting replicas until the
    // terminal retire step (which completes the plan immediately after).
    for plan in state.live_plans_for(tablet) {
        if !candidates.contains(&plan.from) {
            candidates.push(plan.from);
        }
    }
    for voter in &desired.replicas {
        if !candidates.contains(voter) {
            candidates.push(*voter);
        }
    }
    candidates.into_iter().find_map(|voter| {
        let endpoint = state
            .node(voter)
            .map(|record| record.native)
            .or_else(|| shared.natives.get(&voter).copied())?;
        let directory = shared.directory_snapshot();
        let range = directory
            .get(tablet)
            .map(|descriptor| descriptor.range().clone())?;
        Some(Response {
            proof: None,
            status: Status::StaleRoute,
            body: ResponseBody::Redirect(RedirectInfo {
                dir_version: directory.version(),
                tablet,
                epoch: TabletEpoch::INITIAL,
                worker: WorkerId::from_u64(CLUSTER_WORKER),
                endpoint: endpoint.to_string(),
                range,
            }),
        })
    })
}

/// Redirects to a statically configured founder (lowest node id first,
/// never self): the admission-time backstop when this node has no
/// control image or desired placement for `tablet`. Founders host the
/// genesis tiling and inherit into every child replica set, so they hold
/// (or authoritatively redirect from) any tablet this node cannot serve.
/// The range comes from the local directory (possibly a genesis view);
/// the founder re-routes by its converged directory, so a stale range
/// costs one extra hop, never a wrong authority: the client's
/// version-aware progress treats the founder's newer generation as
/// progress and converges.
fn static_founder_redirect(shared: &ClusterShared, tablet: TabletId) -> Option<Response> {
    let mut founders: Vec<(NodeId, SocketAddr)> = shared
        .natives
        .iter()
        .map(|(node, addr)| (*node, *addr))
        .collect();
    founders.sort_by_key(|(node, _)| node.as_u64());
    let directory = shared.directory_snapshot();
    let range = directory
        .get(tablet)
        .map(|descriptor| descriptor.range().clone())?;
    founders.into_iter().find_map(|(node, endpoint)| {
        if node == shared.node.node() {
            return None;
        }
        Some(Response {
            proof: None,
            status: Status::StaleRoute,
            body: ResponseBody::Redirect(RedirectInfo {
                dir_version: directory.version(),
                tablet,
                epoch: TabletEpoch::INITIAL,
                worker: WorkerId::from_u64(CLUSTER_WORKER),
                endpoint: endpoint.to_string(),
                range: range.clone(),
            }),
        })
    })
}

/// Shapes proposal failures: routing becomes redirects/retriable
/// statuses (never `OpenRaft` concepts on the wire), validation becomes
/// stable semantic statuses.
pub(crate) async fn shape_propose_error(
    shared: &ClusterShared,
    tablet: TabletId,
    error: &ProposeError,
) -> Response {
    match error {
        ProposeError::Consensus(ConsensusError::NotLeader { hint }) => {
            redirect_or_overloaded(shared, tablet, hint.leader.map(replica_node))
        }
        ProposeError::Consensus(ConsensusError::LeaderUnknown) => Response {
            proof: None,
            status: Status::Overloaded,
            body: ResponseBody::Diagnostic("leader unknown; retry".to_owned()),
        },
        ProposeError::Consensus(ConsensusError::Unavailable { reason })
            if reason.contains("not served by this node") =>
        {
            // Normal during migration: this node holds no replica.
            // Route through desired placement instead of a generic miss.
            desired_redirect(shared, tablet).await.unwrap_or(Response {
                proof: None,
                status: Status::Overloaded,
                body: ResponseBody::Diagnostic("tablet migrating; retry".to_owned()),
            })
        }
        ProposeError::Consensus(ConsensusError::Unavailable { reason }) => Response {
            proof: None,
            status: Status::Internal,
            body: ResponseBody::Diagnostic(format!("consensus unavailable: {reason}")),
        },
        ProposeError::Consensus(ConsensusError::ShuttingDown) => Response {
            proof: None,
            status: Status::Internal,
            body: ResponseBody::Diagnostic("replica shutting down".to_owned()),
        },
        ProposeError::Expired => Response {
            proof: None,
            status: Status::DedupExpired,
            body: ResponseBody::Diagnostic("mutation identity expired".to_owned()),
        },
        ProposeError::Overloaded => Response {
            proof: None,
            status: Status::SessionOverloaded,
            body: ResponseBody::Diagnostic("session outcome window exhausted".to_owned()),
        },
        ProposeError::Unsupported { reason } => Response {
            proof: None,
            status: Status::Unsupported,
            body: ResponseBody::Diagnostic(reason.clone()),
        },
        ProposeError::Op(OpError::WrongType { .. }) => Response {
            proof: None,
            status: Status::WrongType,
            body: ResponseBody::Diagnostic("wrong type".to_owned()),
        },
        ProposeError::Op(OpError::CounterOverflow) => Response {
            proof: None,
            status: Status::CounterOverflow,
            body: ResponseBody::Diagnostic("overflow".to_owned()),
        },
        ProposeError::Op(OpError::StaleRangeBase) => Response {
            proof: None,
            status: Status::InvalidRequest,
            body: ResponseBody::Diagnostic("range base changed; retry".to_owned()),
        },
        ProposeError::Op(OpError::TxnConflict) => Response {
            proof: None,
            status: Status::TxnConflict,
            body: ResponseBody::Diagnostic("transaction conflict".to_owned()),
        },
        ProposeError::Fenced { .. } => Response {
            proof: None,
            status: Status::Overloaded,
            body: ResponseBody::Diagnostic("tablet fenced for topology cutover; retry".to_owned()),
        },
        ProposeError::ResponderCoverage { detail } => Response {
            proof: None,
            status: Status::CoverageUncertain,
            body: ResponseBody::Diagnostic(format!(
                "responder coverage timed out; outcome uncertain ({detail})"
            )),
        },
        // Forward-compatibility: future proposal failures fail closed as
        // internal rather than mis-shaping on the wire.
        _ => Response {
            proof: None,
            status: Status::Internal,
            body: ResponseBody::Diagnostic("proposal failed".to_owned()),
        },
    }
}

/// Shapes read failures with the same routing contract as proposals.
pub(crate) async fn shape_read_error(
    shared: &ClusterShared,
    tablet: TabletId,
    error: &ReadError,
) -> Response {
    match error {
        kivi_consensus::ReadError::Consensus(ConsensusError::NotLeader { hint }) => {
            redirect_or_overloaded(shared, tablet, hint.leader.map(replica_node))
        }
        kivi_consensus::ReadError::Consensus(ConsensusError::LeaderUnknown) => Response {
            proof: None,
            status: Status::Overloaded,
            body: ResponseBody::Diagnostic("leader unknown; retry".to_owned()),
        },
        kivi_consensus::ReadError::Consensus(ConsensusError::Unavailable { reason })
            if reason.contains("not served by this node") =>
        {
            desired_redirect(shared, tablet).await.unwrap_or(Response {
                proof: None,
                status: Status::Overloaded,
                body: ResponseBody::Diagnostic("tablet migrating; retry".to_owned()),
            })
        }
        kivi_consensus::ReadError::Consensus(ConsensusError::Unavailable { reason }) => Response {
            proof: None,
            status: Status::Internal,
            body: ResponseBody::Diagnostic(format!("consensus unavailable: {reason}")),
        },
        kivi_consensus::ReadError::Consensus(ConsensusError::ShuttingDown) => Response {
            proof: None,
            status: Status::Internal,
            body: ResponseBody::Diagnostic("replica shutting down".to_owned()),
        },
        kivi_consensus::ReadError::Op(OpError::WrongType { .. }) => Response {
            proof: None,
            status: Status::WrongType,
            body: ResponseBody::Diagnostic("wrong type".to_owned()),
        },
        kivi_consensus::ReadError::Op(OpError::CounterOverflow) => Response {
            proof: None,
            status: Status::CounterOverflow,
            body: ResponseBody::Diagnostic("overflow".to_owned()),
        },
        kivi_consensus::ReadError::Op(OpError::StaleRangeBase) => Response {
            proof: None,
            status: Status::InvalidRequest,
            body: ResponseBody::Diagnostic("range base changed; retry".to_owned()),
        },
        kivi_consensus::ReadError::NotARead => Response {
            proof: None,
            status: Status::InvalidRequest,
            body: ResponseBody::Diagnostic("not a read operation".to_owned()),
        },
        kivi_consensus::ReadError::StaleToken { detail } => Response {
            proof: None,
            status: Status::StaleToken,
            body: ResponseBody::Diagnostic(detail.clone()),
        },
        kivi_consensus::ReadError::CoverageTimeout => Response {
            proof: None,
            status: Status::Overloaded,
            body: ResponseBody::Diagnostic("applied coverage timed out; retry".to_owned()),
        },
        // Forward-compatibility: future read failures fail closed as
        // internal rather than mis-shaping on the wire.
        _ => Response {
            proof: None,
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
    let directory = shared.directory_snapshot();
    let range = directory
        .get(tablet)
        .map(|descriptor| descriptor.range().clone());
    match (endpoint, range) {
        (Some(addr), Some(range)) => Response {
            proof: None,
            status: Status::StaleRoute,
            body: ResponseBody::Redirect(RedirectInfo {
                dir_version: directory.version(),
                tablet,
                epoch: TabletEpoch::INITIAL,
                worker: WorkerId::from_u64(CLUSTER_WORKER),
                endpoint: addr.to_string(),
                range,
            }),
        },
        _ => Response {
            proof: None,
            status: Status::Overloaded,
            body: ResponseBody::Diagnostic("leader unknown; retry".to_owned()),
        },
    }
}

/// Wall-clock timestamp (client-boundary capture only; replicas apply
/// the leader's materialized stamp, never their own clock). One shared
/// production read, failing closed to `MAX` so a broken clock expires
/// rather than resurrects.
pub(crate) fn wall_now() -> WallTimestamp {
    kivi_core::wall_now_or_max(&kivi_core::SystemClock)
}

/// Default bounded wait for `AtLeast` catch-up on one read. The
/// consistency layer caps it again at its own hard bound; this only sets
/// the server's ask.
const DEFAULT_READ_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

/// Process monotonic microseconds since first call: the staleness/lease
/// clock for read contexts. Wall time measures expiry; this measures ages
/// — the two are never equated (see `kivi_types::consistency`).
pub(crate) fn mono_now() -> kivi_types::Ticks {
    use std::sync::OnceLock;
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    let start = START.get_or_init(std::time::Instant::now);
    kivi_types::Ticks::from_micros(start.elapsed().as_micros().try_into().unwrap_or(u64::MAX))
}

/// Builds one read execution context: wall time for expiry, monotonic
/// time for proof/lease ages, bounded wait for catch-up.
pub(crate) fn read_ctx() -> kivi_types::ReadContext {
    kivi_types::ReadContext::new(wall_now(), mono_now(), DEFAULT_READ_WAIT)
}

/// Intent age for admin diagnostics: `now - prepared_at` floored at zero
/// (a skewed clock never reports a negative age).
fn intent_age_micros(now: WallTimestamp, prepared_at: WallTimestamp) -> u64 {
    u64::try_from(
        now.as_micros()
            .saturating_sub(prepared_at.as_micros())
            .max(0),
    )
    .unwrap_or(u64::MAX)
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
    /// Ordered range start as lowercase hex (`None` for hash layouts).
    range_start: Option<String>,
    /// Ordered range end as lowercase hex (`None` for hash layouts or
    /// unbounded `+∞`).
    range_end: Option<String>,
    /// Directory state of this tablet (`Active`, `Inactive`, `Fenced`,
    /// `Tombstone`, or `unknown` when absent from the directory).
    /// Coverage checks use `Active` only: tombstones overlap their
    /// successors by design (redirects, never authorities).
    state: String,
    /// Prepared transaction intents on this replica (stuck-intent
    /// detection for tests and operators).
    intent_count: usize,
    /// Age of the oldest prepared intent in micros (`None` when no
    /// intents are prepared): with `intent_count` answers "what is
    /// blocking resolution" at a glance.
    oldest_prepared_age_micros: Option<u64>,
    /// Consistency-layer counters for this tablet: reads per contract
    /// `(latest, at_least, bounded_stale, any)`.
    consistency_reads: (u64, u64, u64, u64),
    /// Reads served without a fresh barrier vs after one.
    consistency_local_serves: u64,
    /// Reads served after a fresh authority barrier.
    consistency_authority_serves: u64,
    /// `AtLeast` waits, wait timeouts, and lineage rejects.
    consistency_at_least_waits: u64,
    /// `AtLeast` waits that timed out instead of covering.
    consistency_at_least_timeouts: u64,
    /// `AtLeast` tokens rejected for lineage.
    consistency_at_least_rejects: u64,
    /// `BoundedStale` proof hits, escalations, and unprovable bounds.
    consistency_bounded_hits: u64,
    /// `BoundedStale` reads escalated to the authority path.
    consistency_bounded_escalations: u64,
    /// Reads failed because no freshness proof could be established.
    consistency_unprovable: u64,
    /// Strong-cache hits, fallbacks, and authority invalidations.
    consistency_cache_hits: u64,
    /// Strong-cache fallbacks.
    consistency_cache_fallbacks: u64,
    /// Strong-cache invalidations on authority change.
    consistency_cache_invalidations: u64,
    /// Lazy-ALR batches formed.
    consistency_alr_batches: u64,
    /// Reads executed inside Lazy-ALR batches.
    consistency_alr_reads: u64,
    /// ALR batches that committed their own fence entry.
    consistency_alr_syncs: u64,
    /// ALR batches subsumed by a concurrent write.
    consistency_alr_subsumed: u64,
    /// ALR batches abandoned to the conservative barrier.
    consistency_alr_fallbacks: u64,
    /// Roster-lease fast-path hits.
    consistency_roster_hits: u64,
    /// Roster-lease reads behind the safety floor (hold-or-fallback).
    consistency_roster_holds: u64,
    /// Roster-lease fallbacks.
    consistency_roster_fallbacks: u64,
    /// Roster pairings lapsed on local timers.
    consistency_roster_expiries: u64,
    /// Strong writes gated on explicit responder coverage.
    consistency_coverage_waits: u64,
    /// Coverage gates that timed out instead of acknowledging.
    consistency_coverage_timeouts: u64,
    /// Responders currently covered by the write gate.
    consistency_covered_responders: u64,
    /// Stable-roster summary (`None` when no roster is stable).
    consistency_roster: Option<RosterDto>,
    /// Authority/fencing rejects on the read path.
    consistency_fencing_rejects: u64,
}

/// Copy-safe stable-roster summary for status surfaces.
#[derive(Debug, Clone, Serialize)]
struct RosterDto {
    /// Roster generation served under.
    generation: u64,
    /// Consensus term the roster is bound to.
    term: u64,
    /// Designated responders.
    responders: u64,
    /// Grants backing stability.
    grants: u64,
    /// Majority count required.
    required: u64,
    /// Safety floor (inclusive commit position).
    floor: u64,
    /// Local applied position at snapshot time.
    applied: u64,
    /// Whether the lease engine is quarantined.
    quarantined: bool,
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

fn tablet_dto(
    status: &NodeStatus,
    directory: &DirectorySnapshot,
    intent_count: usize,
    oldest_prepared_age_micros: Option<u64>,
) -> TabletDto {
    // Static clusters always tile the hash space; non-hash layouts (a
    // future concern) encode as nulls, never wrong ranges.
    let (range_bits, range_len, range_start, range_end, state) = directory
        .get(TabletId::from_u64(status.group.tablet().as_u64()))
        .map_or(
            (None, None, None, None, "unknown".to_owned()),
            |descriptor| {
                let ranges = match descriptor.range() {
                    kivi_tablet::PartitionRange::Hash(prefix) => (
                        Some(format!("{:032x}", prefix.bits())),
                        Some(prefix.prefix_len()),
                        None,
                        None,
                    ),
                    kivi_tablet::PartitionRange::Ordered(range) => (
                        None,
                        None,
                        Some(hex_bytes(range.start())),
                        range.end().map(hex_bytes),
                    ),
                };
                (
                    ranges.0,
                    ranges.1,
                    ranges.2,
                    ranges.3,
                    format!("{:?}", descriptor.state()),
                )
            },
        );
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
        range_start,
        range_end,
        state,
        intent_count,
        oldest_prepared_age_micros,
        consistency_reads: status.consistency.reads_by_contract,
        consistency_local_serves: status.consistency.local_serves,
        consistency_authority_serves: status.consistency.authority_serves,
        consistency_at_least_waits: status.consistency.at_least_waits,
        consistency_at_least_timeouts: status.consistency.at_least_timeouts,
        consistency_at_least_rejects: status.consistency.at_least_lineage_rejects,
        consistency_bounded_hits: status.consistency.bounded_stale_hits,
        consistency_bounded_escalations: status.consistency.bounded_stale_escalations,
        consistency_unprovable: status.consistency.freshness_unprovable,
        consistency_cache_hits: status.consistency.strong_cache_hits,
        consistency_cache_fallbacks: status.consistency.strong_cache_fallbacks,
        consistency_cache_invalidations: status.consistency.strong_cache_invalidations,
        consistency_alr_batches: status.consistency.alr_batches,
        consistency_alr_reads: status.consistency.alr_reads,
        consistency_alr_syncs: status.consistency.alr_syncs,
        consistency_alr_subsumed: status.consistency.alr_subsumed,
        consistency_alr_fallbacks: status.consistency.alr_fallbacks,
        consistency_roster_hits: status.consistency.roster_hits,
        consistency_roster_holds: status.consistency.roster_holds,
        consistency_roster_fallbacks: status.consistency.roster_fallbacks,
        consistency_roster_expiries: status.consistency.roster_expiries,
        consistency_coverage_waits: status.consistency.coverage_waits,
        consistency_coverage_timeouts: status.consistency.coverage_timeouts,
        consistency_covered_responders: status.consistency.covered_responders,
        consistency_roster: status.consistency.roster.map(|summary| RosterDto {
            generation: summary.generation,
            term: summary.term,
            responders: summary.responders,
            grants: summary.grants,
            required: summary.required,
            floor: summary.floor.as_u64(),
            applied: summary.applied.as_u64(),
            quarantined: summary.quarantined,
        }),
        consistency_fencing_rejects: status.consistency.fencing_rejects,
    }
}

/// Lowercase hex encoding for admin range bounds (human-readable, exact).
fn hex_bytes(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok", "version": env!("CARGO_PKG_VERSION") }))
}

async fn ready(State(shared): State<ClusterShared>) -> Json<serde_json::Value> {
    let statuses = shared.node.status_all().await;
    // Multi-tablet readiness: process alive is not enough. A node is
    // usable once its peer mesh started, every configured local replica
    // loaded, and every group healthy — partial failure is reported
    // openly (counts), never hidden behind one boolean. A cluster can be
    // ready while rebalancing: migrations never gate readiness (§44).
    let total = statuses.len();
    let healthy = statuses.iter().filter(|status| status.healthy).count();
    let leaders_known = statuses
        .iter()
        .filter(|status| status.leader.is_some())
        .count();
    // Control-plane summary for operators: registry counts by lifecycle
    // plus migration progress. Absent when this node hosts no control
    // replica yet.
    let control = shared.node.control_state().await.map(|state| {
        let mut active = 0u64;
        let mut joining = 0u64;
        let mut suspect = 0u64;
        let mut unavailable = 0u64;
        let mut draining = 0u64;
        let mut drained = 0u64;
        for record in state.nodes() {
            match record.state {
                kivi_control::NodeState::Active => active += 1,
                kivi_control::NodeState::Joining => joining += 1,
                kivi_control::NodeState::Suspect => suspect += 1,
                kivi_control::NodeState::Unavailable => unavailable += 1,
                kivi_control::NodeState::Draining => draining += 1,
                kivi_control::NodeState::Drained => drained += 1,
                kivi_control::NodeState::Removed => {}
            }
        }
        let live = state
            .migrations()
            .filter(|plan| !plan.phase.is_terminal())
            .count() as u64;
        let tablets_migrating = state
            .placements()
            .filter(|desired| !state.live_plans_for(desired.tablet).is_empty())
            .count() as u64;
        serde_json::json!({
            "present": true,
            "placement_version": state.placement_version().as_u64(),
            "nodes_active": active,
            "nodes_joining": joining,
            "nodes_suspect": suspect,
            "nodes_unavailable": unavailable,
            "nodes_draining": draining,
            "nodes_drained": drained,
            "migrations_live": live,
            "tablets_migrating": tablets_migrating,
        })
    });
    // A data-empty joiner (no tablets yet, control present) is usable:
    // readiness reflects node/control usability, never zero migrations
    // or zero local tablets.
    let control_present = control
        .as_ref()
        .and_then(|control| control.get("present").and_then(serde_json::Value::as_bool))
        .unwrap_or(false);
    Json(serde_json::json!({
        "ready": healthy == total && (total > 0 || control_present),
        "initialized": true,
        "transport": true,
        "membership_loaded": true,
        "tablets_total": total,
        "tablets_healthy": healthy,
        "leaders_known": leaders_known,
        "election_pending": leaders_known != total,
        "leader_known": leaders_known == total,
        "healthy": healthy == total,
        "control": control.unwrap_or(serde_json::json!({ "present": false })),
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
        "dir_version": shared.directory_snapshot().version().as_u64(),
        "mode": "replicated",
        "cluster_mode": "replicated",
        "peer": shared.node.peer_addr().to_string(),
        "native": shared.native.to_string(),
        "client_endpoint": shared.native.to_string(),
        "peer_endpoint": shared.node.peer_addr().to_string(),
    }))
}

/// Per-tablet diagnostics for every local group, sorted by tablet,
/// enriched with desired vs actual placement (§43): desired voters from
/// replicated control state when present, actual voters/learners from
/// live membership, and a placement verdict per tablet.
async fn tablets(State(shared): State<ClusterShared>) -> Json<Vec<serde_json::Value>> {
    let control = shared.node.control_state().await;
    let statuses = shared.node.status_all().await;
    let mut observed = std::collections::BTreeMap::new();
    for status in &statuses {
        let tablet = TabletId::from_u64(status.group.tablet().as_u64());
        if let Ok(obs) = shared.node.observe_membership(tablet).await {
            observed.insert(tablet.as_u64(), obs);
        }
    }
    let health: std::collections::HashMap<u64, super::control::PlacementHealth> = control
        .as_ref()
        .map(|state| super::control::placement_health(state, &observed))
        .map(|list| {
            list.into_iter()
                .map(|item| (item.tablet.as_u64(), item))
                .collect()
        })
        .unwrap_or_default();
    let directory = shared.directory_snapshot();
    let mut out = Vec::new();
    for status in &statuses {
        let id = status.group.tablet().as_u64();
        out.push(tablet_row(&shared, status, &directory, health.get(&id)).await);
    }
    out.sort_by_key(|value| {
        value
            .get("group")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(u64::MAX)
    });
    Json(out)
}

/// Renders one tablet's admin row: base DTO plus placement, health, and
/// intent age. Split from the `/v1/tablets` handler so the endpoint
/// stays a gather-and-sort loop.
async fn tablet_row(
    shared: &ClusterShared,
    status: &NodeStatus,
    directory: &DirectorySnapshot,
    item: Option<&super::control::PlacementHealth>,
) -> serde_json::Value {
    let id = status.group.tablet().as_u64();
    let intents = shared
        .node
        .tablet_intents(TabletId::from_u64(id))
        .await
        .unwrap_or_default();
    let now = wall_now();
    let oldest_prepared_age = intents
        .iter()
        .map(|intent| intent.prepared_at)
        .min()
        .map(|oldest| intent_age_micros(now, oldest));
    // Bounded per-intent identity for stuck-intent forensics (which
    // transaction, which key, which coordinator, how old): the count
    // alone cannot attribute a wedge to its creator.
    let intent_keys: Vec<serde_json::Value> = intents
        .iter()
        .take(8)
        .map(|intent| {
            serde_json::json!({
                "key": hex_bytes(intent.key.as_bytes()),
                "coordinator": intent.coordinator.as_u64(),
                "age_micros": intent_age_micros(now, intent.prepared_at),
            })
        })
        .collect();
    let mut value = serde_json::to_value(tablet_dto(
        status,
        directory,
        intents.len(),
        oldest_prepared_age,
    ))
    .unwrap_or(serde_json::Value::Null);
    if let serde_json::Value::Object(ref mut map) = value {
        map.insert(
            "voters".to_owned(),
            item.and_then(|item| item.actual.clone())
                .map_or(serde_json::Value::Null, |voters| {
                    serde_json::json!(voters.iter().map(|node| node.as_u64()).collect::<Vec<_>>())
                }),
        );
        map.insert(
            "learners".to_owned(),
            item.map_or(serde_json::json!([]), |item| {
                serde_json::json!(
                    item.learners
                        .iter()
                        .map(|node| node.as_u64())
                        .collect::<Vec<_>>()
                )
            }),
        );
        map.insert(
            "desired".to_owned(),
            item.map_or(serde_json::Value::Null, |item| {
                serde_json::json!(
                    item.desired
                        .iter()
                        .map(|node| node.as_u64())
                        .collect::<Vec<_>>()
                )
            }),
        );
        map.insert(
            "placement".to_owned(),
            serde_json::Value::String(
                item.map_or("unknown", |item| match item.status {
                    kivi_control::TabletHealth::Healthy => "healthy",
                    kivi_control::TabletHealth::Converging => "converging",
                    kivi_control::TabletHealth::UnderReplicated => "under-replicated",
                    kivi_control::TabletHealth::Unavailable => "unavailable",
                    kivi_control::TabletHealth::OverReplicated => "over-replicated",
                })
                .to_owned(),
            ),
        );
        map.insert(
            "healthy".to_owned(),
            serde_json::Value::Bool(item.is_some_and(|item| item.healthy)),
        );
        map.insert(
            "migrating".to_owned(),
            serde_json::Value::Bool(item.is_some_and(|item| item.migrating)),
        );
        map.insert(
            "intent_keys".to_owned(),
            serde_json::Value::Array(intent_keys),
        );
    }
    value
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
    let directory = shared.directory_snapshot();
    let intents = shared
        .node
        .tablet_intents(TabletId::from_u64(status.group.tablet().as_u64()))
        .await
        .unwrap_or_default();
    let now = wall_now();
    let oldest_prepared_age_micros = intents
        .iter()
        .map(|intent| intent.prepared_at)
        .min()
        .map(|oldest| intent_age_micros(now, oldest));
    match serde_json::to_value(tablet_dto(
        &status,
        &directory,
        intents.len(),
        oldest_prepared_age_micros,
    )) {
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

/// Adds a dialable peer link for a dynamically admitted node. Body:
/// `{ "node": <u64>, "addr": "<peer endpoint>" }`. The reconciler's
/// mesh sync performs this automatically from registry commits; the
/// endpoint exists for manual repair and tests.
async fn peers_add(
    State(shared): State<ClusterShared>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let (Some(node), Some(addr)) = (
        body.get("node").and_then(serde_json::Value::as_u64),
        body.get("addr").and_then(serde_json::Value::as_str),
    ) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::json!({ "error": "body must be {\"node\": <u64>, \"addr\": \"<peer>\"}" }),
            ),
        );
    };
    if node == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "node id 0 is reserved" })),
        );
    }
    let Ok(addr) = addr.parse::<SocketAddr>() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "addr must be a socket address" })),
        );
    };
    let added = shared.node.add_peer_dial(NodeId::from_u64(node), addr);
    (
        StatusCode::OK,
        Json(serde_json::json!({ "ok": true, "added": added })),
    )
}

/// Pins a dynamically admitted peer's certificate DER for TLS
/// verification. Body: `{ "node": <u64>, "cert_der_hex": "<hex>" }`.
/// Static anchors keep precedence; the reconciler cannot distribute
/// DERs itself (the registry carries fingerprints), so operators push
/// them over this endpoint after approving the admission.
async fn peers_trust(
    State(shared): State<ClusterShared>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let (Some(node), Some(hex)) = (
        body.get("node").and_then(serde_json::Value::as_u64),
        body.get("cert_der_hex").and_then(serde_json::Value::as_str),
    ) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::json!({ "error": "body must be {\"node\": <u64>, \"cert_der_hex\": \"<hex>\"}" }),
            ),
        );
    };
    if node == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "node id 0 is reserved" })),
        );
    }
    if hex.len() > 8192 || !hex.len().is_multiple_of(2) {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::json!({ "error": "cert_der_hex must be even-length hex under 8 KiB" }),
            ),
        );
    }
    let mut der = Vec::with_capacity(hex.len() / 2);
    for index in 0..hex.len() / 2 {
        match u8::from_str_radix(&hex[2 * index..2 * index + 2], 16) {
            Ok(byte) => der.push(byte),
            Err(_) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": "cert_der_hex is not hex" })),
                );
            }
        }
    }
    shared.node.trust_peer(NodeId::from_u64(node), der);
    (StatusCode::OK, Json(serde_json::json!({ "ok": true })))
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

/// Hex rendering for certificate fingerprints in admin output.
fn hex32(bytes: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Parses a 64-hex-char fingerprint (empty/absent means unenforced:
/// lab-insecure only).
fn parse_fingerprint(value: Option<&str>) -> Result<[u8; 32], String> {
    let Some(hex) = value else {
        return Ok([0u8; 32]);
    };
    if hex.len() != 64 {
        return Err(format!(
            "fingerprint must be 64 hex chars, got {}",
            hex.len()
        ));
    }
    let mut out = [0u8; 32];
    for (index, chunk) in out.iter_mut().enumerate() {
        *chunk = u8::from_str_radix(&hex[2 * index..2 * index + 2], 16)
            .map_err(|_| format!("fingerprint is not hex at byte {index}"))?;
    }
    Ok(out)
}

/// Replicated control image for operators: registry, desired
/// placements, migration plans, and versions. `present=false` when this
/// node hosts no control replica (pure data host without a control
/// learner yet).
async fn control(State(shared): State<ClusterShared>) -> Json<serde_json::Value> {
    let Some(state) = shared.node.control_state().await else {
        return Json(serde_json::json!({ "present": false }));
    };
    let nodes: Vec<serde_json::Value> = state
        .nodes()
        .map(|record| {
            serde_json::json!({
                "node": record.node.as_u64(),
                "peer": record.peer.to_string(),
                "native": record.native.to_string(),
                "admin": record.admin.to_string(),
                "fingerprint": hex32(&record.cert_fingerprint),
                "domain": record.failure_domain,
                "weight": record.weight,
                "state": format!("{:?}", record.state),
            })
        })
        .collect();
    let placements: Vec<serde_json::Value> = state
        .placements()
        .map(|desired| {
            serde_json::json!({
                "tablet": desired.tablet.as_u64(),
                "replicas": desired.replicas.iter().map(|node| node.as_u64()).collect::<Vec<_>>(),
                "version": desired.version.as_u64(),
            })
        })
        .collect();
    let migrations: Vec<serde_json::Value> = state
        .migrations()
        .map(|plan| {
            serde_json::json!({
                "id": plan.id.as_u64(),
                "tablet": plan.tablet.as_u64(),
                "generation": plan.generation.as_u64(),
                "from": plan.from.as_u64(),
                "to": plan.to.as_u64(),
                "desired": plan.desired_voters.iter().map(|node| node.as_u64()).collect::<Vec<_>>(),
                "phase": format!("{:?}", plan.phase),
            })
        })
        .collect();
    Json(serde_json::json!({
        "present": true,
        "placement_version": state.placement_version().as_u64(),
        "generation": state.generation().as_u64(),
        "next_plan_id": state.next_plan_id().as_u64(),
        "nodes": nodes,
        "placements": placements,
        "migrations": migrations,
    }))
}

/// Live migration plans (same objects as `/v1/control`, plan-focused).
async fn migrations(State(shared): State<ClusterShared>) -> Json<serde_json::Value> {
    let Some(state) = shared.node.control_state().await else {
        return Json(serde_json::json!({ "present": false, "migrations": [] }));
    };
    let live = state
        .migrations()
        .filter(|plan| !plan.phase.is_terminal())
        .count();
    Json(serde_json::json!({
        "present": true,
        "live": live,
        "migrations": state.migrations().map(|plan| serde_json::json!({
            "id": plan.id.as_u64(),
            "tablet": plan.tablet.as_u64(),
            "from": plan.from.as_u64(),
            "to": plan.to.as_u64(),
            "phase": format!("{:?}", plan.phase),
        })).collect::<Vec<_>>(),
    }))
}

/// Split plans with phases, ranges, and replicas (operator visibility).
async fn control_splits(State(shared): State<ClusterShared>) -> Json<serde_json::Value> {
    let Some(state) = shared.node.control_state().await else {
        return Json(serde_json::json!({ "present": false, "splits": [] }));
    };
    let directory = shared.directory_snapshot();
    Json(serde_json::json!({
        "present": true,
        "splits": state.splits().map(|plan| {
            let parent_range = directory.get(plan.parent).map(|descriptor| descriptor.range().to_string());
            serde_json::json!({
                "id": plan.id.as_u64(),
                "parent": plan.parent.as_u64(),
                "parent_range": parent_range,
                "split_hash": format!("{:x}", plan.split_hash),
                "left": plan.left.as_u64(),
                "right": plan.right.as_u64(),
                "replicas": plan.replicas.iter().map(|node| node.as_u64()).collect::<Vec<_>>(),
                "generation": plan.generation.as_u64(),
                "phase": format!("{:?}", plan.phase),
            })
        }).collect::<Vec<_>>(),
    }))
}

/// Creates a split plan for one populated tablet through the persisted
/// reconciler pathway (never a metadata-only edit). Body:
/// `{ "tablet": <u64> }`. The boundary is the parent midpoint; children
/// inherit the parent replica set; the reconciler seeds, fences, cuts
/// over, and retires. One straight-line validation chain per layout.
#[allow(clippy::too_many_lines)]
async fn control_splits_create(
    State(shared): State<ClusterShared>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Some(tablet) = body.get("tablet").and_then(serde_json::Value::as_u64) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "body must be {\"tablet\": <u64>}" })),
        );
    };
    let parent = TabletId::from_u64(tablet);
    let Some(state) = shared.node.control_state().await else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": "no local control image" })),
        );
    };
    let directory = shared.directory_snapshot();
    let Some(descriptor) = directory.get(parent) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "unknown tablet" })),
        );
    };
    if descriptor.state() != kivi_tablet::TabletState::Active {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "only active tablets split" })),
        );
    }
    let kivi_tablet::PartitionRange::Hash(prefix) = descriptor.range().clone() else {
        // Ordered parents split at a real key: explicit `split_key` wins,
        // otherwise the leader-local median balances logical bytes.
        let kivi_tablet::PartitionRange::Ordered(range) = descriptor.range().clone() else {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "unknown range layout" })),
            );
        };
        let split_key: Option<Vec<u8>> = body.get("split_key").and_then(|value| {
            value.as_array().map(|bytes| {
                bytes
                    .iter()
                    .filter_map(serde_json::Value::as_u64)
                    .filter_map(|byte| u8::try_from(byte).ok())
                    .collect()
            })
        });
        let split_key = match split_key {
            Some(key) => key,
            None => match shared.node.tablet_median_key(parent).await {
                Some(key) => key,
                None => {
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        Json(
                            serde_json::json!({ "error": "no interior split key (too few keys)" }),
                        ),
                    );
                }
            },
        };
        if range.split_at(&split_key).is_err() {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "split key must satisfy start < key < end" })),
            );
        }
        let fresh = state.fresh_tablet_ids(2);
        let (Some(left), Some(right)) = (fresh.first().copied(), fresh.get(1).copied()) else {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": "tablet allocator exhausted" })),
            );
        };
        return match super::control::create_split_plan_full(
            &shared.node,
            parent,
            0,
            Some(split_key.clone()),
            left,
            right,
        )
        .await
        {
            Ok(id) => (
                StatusCode::OK,
                Json(serde_json::json!({
                    "ok": true,
                    "plan": id.as_u64(),
                    "parent": parent.as_u64(),
                    "left": left.as_u64(),
                    "right": right.as_u64(),
                    "split_key": split_key,
                })),
            ),
            Err(reason) => (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": reason })),
            ),
        };
    };
    let Ok((_, _, split_hash)) = prefix.split_midpoint() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "tablet range unsplittable" })),
        );
    };
    let fresh = state.fresh_tablet_ids(2);
    let (Some(left), Some(right)) = (fresh.first().copied(), fresh.get(1).copied()) else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": "tablet allocator exhausted" })),
        );
    };
    match super::control::create_split_plan_at(&shared.node, parent, split_hash, left, right).await
    {
        Ok(id) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "ok": true,
                "plan": id.as_u64(),
                "parent": parent.as_u64(),
                "left": left.as_u64(),
                "right": right.as_u64(),
                "split_hash": format!("{:x}", split_hash),
            })),
        ),
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        ),
    }
}

/// Namespace catalog (operator visibility): id plus physical layout.
async fn control_namespaces(State(shared): State<ClusterShared>) -> Json<serde_json::Value> {
    let Some(state) = shared.node.control_state().await else {
        return Json(serde_json::json!({ "present": false, "namespaces": [] }));
    };
    Json(serde_json::json!({
        "present": true,
        "namespaces": state.namespaces().map(|record| serde_json::json!({
            "id": record.id.as_u64(),
            "layout": format!("{:?}", record.layout),
        })).collect::<Vec<_>>(),
    }))
}

/// Registers a namespace (`{ "id": <u64>, "layout": "hash"|"ordered" }`).
/// Idempotent for identical records; conflicting layouts are rejected.
async fn control_namespaces_register(
    State(shared): State<ClusterShared>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let (Some(id), Some(layout)) = (
        body.get("id").and_then(serde_json::Value::as_u64),
        body.get("layout").and_then(serde_json::Value::as_str),
    ) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::json!({ "error": "body must be {\"id\": <u64>, \"layout\": \"hash\"|\"ordered\"}" }),
            ),
        );
    };
    let layout = match layout {
        "hash" => kivi_control::CatalogLayout::Hash,
        "ordered" => kivi_control::CatalogLayout::Ordered,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "layout must be \"hash\" or \"ordered\"" })),
            );
        }
    };
    match super::control::register_namespace(&shared.node, NamespaceId::from_u64(id), layout).await
    {
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({ "ok": true, "id": id })),
        ),
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        ),
    }
}

/// Secondary index catalog (operator visibility).
async fn control_indexes(State(shared): State<ClusterShared>) -> Json<serde_json::Value> {
    let Some(state) = shared.node.control_state().await else {
        return Json(serde_json::json!({ "present": false, "indexes": [] }));
    };
    Json(serde_json::json!({
        "present": true,
        "indexes": state.indexes().map(|record| serde_json::json!({
            "id": record.id,
            "primary": record.primary.as_u64(),
            "kind": format!("{:?}", record.kind),
            "state": format!("{:?}", record.state),
        })).collect::<Vec<_>>(),
    }))
}

/// Creates a secondary index definition
/// (`{ "primary": <u64>, "kind": "unique"|"non-unique" }`).
/// Starts `Building`; advance to `Ready` once backfilled.
async fn control_indexes_create(
    State(shared): State<ClusterShared>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let (Some(primary), Some(kind)) = (
        body.get("primary").and_then(serde_json::Value::as_u64),
        body.get("kind").and_then(serde_json::Value::as_str),
    ) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::json!({ "error": "body must be {\"primary\": <u64>, \"kind\": \"unique\"|\"non-unique\"}" }),
            ),
        );
    };
    let kind = match kind {
        "unique" => kivi_control::CatalogIndexKind::Unique,
        "non-unique" | "nonunique" => kivi_control::CatalogIndexKind::NonUnique,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "kind must be \"unique\" or \"non-unique\"" })),
            );
        }
    };
    match super::control::create_index(&shared.node, NamespaceId::from_u64(primary), kind).await {
        Ok(id) => (
            StatusCode::OK,
            Json(serde_json::json!({ "ok": true, "id": id })),
        ),
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        ),
    }
}

/// Advances an index lifecycle state (`{ "state": "ready"|"dropping" }`).
async fn control_indexes_advance(
    State(shared): State<ClusterShared>,
    axum::extract::Path(id): axum::extract::Path<u64>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Some(state) = body.get("state").and_then(serde_json::Value::as_str) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::json!({ "error": "body must be {\"state\": \"ready\"|\"dropping\"}" }),
            ),
        );
    };
    let state = match state {
        "building" => kivi_control::CatalogIndexState::Building,
        "ready" => kivi_control::CatalogIndexState::Ready,
        "dropping" => kivi_control::CatalogIndexState::Dropping,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(
                    serde_json::json!({ "error": "state must be \"building\", \"ready\", or \"dropping\"" }),
                ),
            );
        }
    };
    match super::control::advance_index(&shared.node, id, state).await {
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({ "ok": true, "id": id })),
        ),
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        ),
    }
}

/// Drops an index definition (must be `Dropping` first: advance, remove
/// entries, then drop the definition).
async fn control_indexes_drop(
    State(shared): State<ClusterShared>,
    axum::extract::Path(id): axum::extract::Path<u64>,
) -> (StatusCode, Json<serde_json::Value>) {
    match super::control::drop_index(&shared.node, id).await {
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({ "ok": true, "id": id })),
        ),
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        ),
    }
}

/// Merge plans with phases (operator visibility).
async fn control_merges(State(shared): State<ClusterShared>) -> Json<serde_json::Value> {
    let Some(state) = shared.node.control_state().await else {
        return Json(serde_json::json!({ "present": false, "merges": [] }));
    };
    Json(serde_json::json!({
        "present": true,
        "merges": state.merges().map(|plan| serde_json::json!({
            "id": plan.id.as_u64(),
            "left": plan.left.as_u64(),
            "right": plan.right.as_u64(),
            "merged": plan.merged.as_u64(),
            "replicas": plan.replicas.iter().map(|node| node.as_u64()).collect::<Vec<_>>(),
            "generation": plan.generation.as_u64(),
            "phase": format!("{:?}", plan.phase),
        })).collect::<Vec<_>>(),
    }))
}

/// Creates a merge plan for two adjacent tablets. Body:
/// `{ "left": <u64>, "right": <u64> }`. Requires adjacency, compatible
/// placement (equal desired voters), and healthy parents.
async fn control_merges_create(
    State(shared): State<ClusterShared>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let (Some(left), Some(right)) = (
        body.get("left").and_then(serde_json::Value::as_u64),
        body.get("right").and_then(serde_json::Value::as_u64),
    ) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::json!({ "error": "body must be {\"left\": <u64>, \"right\": <u64>}" }),
            ),
        );
    };
    match super::control::create_merge_plan(
        &shared.node,
        TabletId::from_u64(left),
        TabletId::from_u64(right),
    )
    .await
    {
        Ok(id) => (
            StatusCode::OK,
            Json(
                serde_json::json!({ "ok": true, "plan": id.as_u64(), "left": left, "right": right }),
            ),
        ),
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        ),
    }
}

/// Repair status: under-replicated / unavailable tablets plus active
/// repair-aged migration plans (same pathway as manual moves).
async fn control_repair(State(shared): State<ClusterShared>) -> Json<serde_json::Value> {
    let Some(state) = shared.node.control_state().await else {
        return Json(serde_json::json!({ "present": false }));
    };
    let policy = shared.policy.lock().map_or_else(
        |_| super::control::ReconcilePolicy::default(),
        |policy| policy.clone(),
    );
    let intents = super::control::repair_intents(&state, &policy);
    let mut under_replicated = Vec::new();
    let mut unavailable = Vec::new();
    for desired in state.placements() {
        let usable = desired
            .replicas
            .iter()
            .filter(|node| {
                state.node(**node).is_some_and(|record| {
                    matches!(
                        record.state,
                        kivi_control::NodeState::Active
                            | kivi_control::NodeState::Suspect
                            | kivi_control::NodeState::Draining
                    )
                })
            })
            .count();
        let health = kivi_control::classify_tablet(
            desired.replicas.len(),
            usable,
            !state.live_plans_for(desired.tablet).is_empty(),
            policy.planner.replication_factor,
        );
        match health {
            kivi_control::TabletHealth::UnderReplicated => {
                under_replicated.push(desired.tablet.as_u64());
            }
            kivi_control::TabletHealth::Unavailable => {
                unavailable.push(desired.tablet.as_u64());
            }
            kivi_control::TabletHealth::Healthy
            | kivi_control::TabletHealth::Converging
            | kivi_control::TabletHealth::OverReplicated => {}
        }
    }
    under_replicated.sort_unstable();
    unavailable.sort_unstable();
    Json(serde_json::json!({
        "present": true,
        "under_replicated": under_replicated,
        "unavailable": unavailable,
        "planned_repairs": intents.len(),
        "repairs": intents.iter().map(|intent| serde_json::json!({
            "tablet": intent.tablet.as_u64(),
            "from": intent.from.as_u64(),
            "to": intent.to.as_u64(),
        })).collect::<Vec<_>>(),
    }))
}

/// Topology overview: tablet count, split/merge plans, directory version,
/// placement summary, and node liveness.
async fn control_topology(State(shared): State<ClusterShared>) -> Json<serde_json::Value> {
    let directory = shared.directory_snapshot();
    let active_tablets = directory
        .tablets()
        .iter()
        .filter(|descriptor| descriptor.state() == kivi_tablet::TabletState::Active)
        .count();
    let Some(state) = shared.node.control_state().await else {
        return Json(serde_json::json!({
            "present": false,
            "tablets": active_tablets,
            "dir_version": directory.version().as_u64(),
        }));
    };
    let mut suspect = 0u64;
    let mut unavailable = 0u64;
    for record in state.nodes() {
        match record.state {
            kivi_control::NodeState::Suspect => suspect += 1,
            kivi_control::NodeState::Unavailable => unavailable += 1,
            kivi_control::NodeState::Joining
            | kivi_control::NodeState::Active
            | kivi_control::NodeState::Draining
            | kivi_control::NodeState::Drained
            | kivi_control::NodeState::Removed => {}
        }
    }
    Json(serde_json::json!({
        "present": true,
        "tablets": active_tablets,
        "dir_version": directory.version().as_u64(),
        "placement_version": state.placement_version().as_u64(),
        "splits_live": state.splits().filter(|plan| !plan.phase.is_terminal()).count(),
        "splits_total": state.splits().count(),
        "merges_live": state.merges().filter(|plan| !plan.phase.is_terminal()).count(),
        "merges_total": state.merges().count(),
        "migrations_live": state.migrations().filter(|plan| !plan.phase.is_terminal()).count(),
        "nodes_suspect": suspect,
        "nodes_unavailable": unavailable,
    }))
}

/// Live reconciler policy (operator visibility).
async fn control_policy(State(shared): State<ClusterShared>) -> Json<serde_json::Value> {
    let policy = shared.policy.lock().map_or_else(
        |_| super::control::ReconcilePolicy::default(),
        |policy| policy.clone(),
    );
    Json(serde_json::json!({
        "auto_repair": policy.auto_repair,
        "auto_split": policy.auto_split,
        "auto_merge": policy.auto_merge,
        "split_object_threshold": policy.split_object_threshold,
        "split_bytes_threshold": policy.split_bytes_threshold,
        "merge_object_threshold": policy.merge_object_threshold,
        "max_live_plans": policy.max_live_plans,
        "max_live_splits": policy.scheduler.max_live_splits,
        "max_live_merges": policy.scheduler.max_live_merges,
        "replication_factor": policy.planner.replication_factor,
        "suspicion_threshold": policy.failure.suspicion_threshold,
        "repair_grace_misses": policy.failure.repair_grace_misses,
    }))
}

/// Retunes the live reconciler policy without a restart. Body accepts any
/// subset of `{ "auto_split": <bool>, "auto_merge": <bool>,
/// "split_object_threshold": <u64>, "split_bytes_threshold": <u64>,
/// "merge_object_threshold": <u64> }`.
async fn control_policy_update(
    State(shared): State<ClusterShared>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Ok(mut policy) = shared.policy.lock() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": "policy lock poisoned" })),
        );
    };
    if let Some(auto_split) = body.get("auto_split").and_then(serde_json::Value::as_bool) {
        policy.auto_split = auto_split;
    }
    if let Some(auto_merge) = body.get("auto_merge").and_then(serde_json::Value::as_bool) {
        policy.auto_merge = auto_merge;
    }
    // Lab floors admit tiny demonstration thresholds; production stays
    // conservative through the defaults (200k/20k). Hysteresis
    // (`merge << split`) is the operator's duty here; the merge path
    // independently refuses above-threshold tablets.
    if let Some(threshold) = body
        .get("split_object_threshold")
        .and_then(serde_json::Value::as_u64)
        && threshold >= 10
    {
        policy.split_object_threshold = usize::try_from(threshold).unwrap_or(usize::MAX);
    }
    // Byte threshold drives ordered auto-splits (logical value bytes).
    if let Some(threshold) = body
        .get("split_bytes_threshold")
        .and_then(serde_json::Value::as_u64)
        && threshold >= 1024
    {
        policy.split_bytes_threshold = threshold;
    }
    if let Some(threshold) = body
        .get("merge_object_threshold")
        .and_then(serde_json::Value::as_u64)
        && threshold >= 2
    {
        policy.merge_object_threshold = usize::try_from(threshold).unwrap_or(usize::MAX);
    }
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "ok": true,
            "auto_split": policy.auto_split,
            "auto_merge": policy.auto_merge,
            "split_object_threshold": policy.split_object_threshold,
            "merge_object_threshold": policy.merge_object_threshold,
        })),
    )
}

/// Observed membership for one tablet group: actual voters, learners,
/// leader, term, joint flag, and replication lag. Powers reconciler
/// forwarding and operator inspection (§43).
async fn tablet_membership(
    State(shared): State<ClusterShared>,
    axum::extract::Path(id): axum::extract::Path<u64>,
) -> (StatusCode, Json<serde_json::Value>) {
    if id == 0 || id == kivi_consensus::CONTROL_TABLET_RAW {
        // The control group's membership is served by id too (operators
        // manage control voters through the same endpoints).
    }
    match shared.node.observe_membership(TabletId::from_u64(id)).await {
        Ok(observed) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "tablet": id,
                "voters": observed.voters.into_iter().collect::<Vec<_>>(),
                "learners": observed.learners.into_iter().collect::<Vec<_>>(),
                "leader": observed.leader.map(|replica| replica.node().as_u64()),
                "term": observed.term.get(),
                "joint": observed.is_joint,
                "lag": observed.lag,
            })),
        ),
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        ),
    }
}

/// Registers a learner on one group's leader. Body: `{ "node": <u64>,
/// "addr": "<peer endpoint>", "blocking": <bool> }`. Non-blocking in
/// production: catch-up is polled through replication lag.
async fn tablet_learners(
    State(shared): State<ClusterShared>,
    axum::extract::Path(id): axum::extract::Path<u64>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let (Some(node), Some(addr)) = (
        body.get("node").and_then(serde_json::Value::as_u64),
        body.get("addr").and_then(serde_json::Value::as_str),
    ) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::json!({ "error": "body must be {\"node\": <u64>, \"addr\": \"<peer>\"}" }),
            ),
        );
    };
    if node == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "node id 0 is reserved" })),
        );
    }
    let blocking = body
        .get("blocking")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    match shared
        .node
        .add_learner(
            TabletId::from_u64(id),
            NodeId::from_u64(node),
            addr.to_owned(),
            blocking,
        )
        .await
    {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({ "ok": true }))),
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        ),
    }
}

/// Replaces one group's voter set through joint consensus. Body:
/// `{ "voters": [<u64>], "retain": <bool> }`. New voters must already
/// be learners; planned migration uses `retain=false`.
async fn tablet_members(
    State(shared): State<ClusterShared>,
    axum::extract::Path(id): axum::extract::Path<u64>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Some(voters) = body.get("voters").and_then(serde_json::Value::as_array) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "body must be {\"voters\": [<u64>]}" })),
        );
    };
    let voters: std::collections::BTreeSet<u64> = voters
        .iter()
        .filter_map(serde_json::Value::as_u64)
        .collect();
    if voters.is_empty() || voters.contains(&0) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "voter set must be nonempty with no zero ids" })),
        );
    }
    let retain = body
        .get("retain")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let tablet = TabletId::from_u64(id);
    if let Err(reason) = shared
        .node
        .change_membership(tablet, voters.clone(), retain)
        .await
    {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        );
    }
    // Converge the joint: the committed joint config needs its uniform
    // successor, proposed here on the tablet leader (the reconciler
    // covers migration plans the same way). Poll briefly, re-proposing
    // while joint persists — re-proposing the same desired set after a
    // committed joint is safe (propose-after-commit holds) and lands
    // directly on the uniform config. Stays inside the admin deadline;
    // callers retry on 503.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(4);
    loop {
        match shared.node.observe_membership(tablet).await {
            Ok(observed) if !observed.is_joint => {
                break (StatusCode::OK, Json(serde_json::json!({ "ok": true })));
            }
            Ok(_) => {
                if tokio::time::Instant::now() >= deadline {
                    break (
                        StatusCode::SERVICE_UNAVAILABLE,
                        Json(serde_json::json!({ "error": "joint membership pending; retry" })),
                    );
                }
                let _ = shared
                    .node
                    .change_membership(tablet, voters.clone(), retain)
                    .await;
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            Err(_) => break (StatusCode::OK, Json(serde_json::json!({ "ok": true }))),
        }
    }
}

/// Hands group leadership to a retained voter. Body: `{ "to": <u64> }`.
/// Fire-and-forget: a non-leader ignores it; the reconciler
/// re-observes and continues.
async fn tablet_leader(
    State(shared): State<ClusterShared>,
    axum::extract::Path(id): axum::extract::Path<u64>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Some(to) = body.get("to").and_then(serde_json::Value::as_u64) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "body must be {\"to\": <u64>}" })),
        );
    };
    match shared
        .node
        .transfer_leader(TabletId::from_u64(id), NodeId::from_u64(to))
        .await
    {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({ "ok": true }))),
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        ),
    }
}

/// Creates an empty local replica for a migration target. Body:
/// `{ "node": <u64>, "voters": [<u64>], "generation": <u64> }`. The
/// `node` names the intended target: a mismatch fails loudly instead
/// of creating a replica on the wrong member after a stale registry
/// read.
async fn tablet_ensure(
    State(shared): State<ClusterShared>,
    axum::extract::Path(id): axum::extract::Path<u64>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let (Some(node), Some(voters), Some(generation)) = (
        body.get("node").and_then(serde_json::Value::as_u64),
        body.get("voters").and_then(serde_json::Value::as_array),
        body.get("generation").and_then(serde_json::Value::as_u64),
    ) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::json!({ "error": "body must be {\"node\": <u64>, \"voters\": [<u64>], \"generation\": <u64>}" }),
            ),
        );
    };
    if node != shared.node.node().as_u64() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "ensure targets another node; refusing" })),
        );
    }
    let voters: std::collections::BTreeSet<u64> = voters
        .iter()
        .filter_map(serde_json::Value::as_u64)
        .collect();
    match shared
        .node
        .ensure_group(TabletId::from_u64(id), voters, generation)
        .await
    {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({ "ok": true }))),
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        ),
    }
}

/// Retires the local replica after membership no longer requires it.
/// Body: `{ "node": <u64>, "generation": <u64> }`. The `node` names
/// the intended source: a mismatch fails loudly instead of retiring
/// the wrong member's replica after a stale registry read.
async fn tablet_retire(
    State(shared): State<ClusterShared>,
    axum::extract::Path(id): axum::extract::Path<u64>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let (Some(node), Some(generation)) = (
        body.get("node").and_then(serde_json::Value::as_u64),
        body.get("generation").and_then(serde_json::Value::as_u64),
    ) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::json!({ "error": "body must be {\"node\": <u64>, \"generation\": <u64>}" }),
            ),
        );
    };
    if node != shared.node.node().as_u64() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "retire targets another node; refusing" })),
        );
    }
    match shared
        .node
        .retire_group(TabletId::from_u64(id), generation)
        .await
    {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({ "ok": true }))),
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        ),
    }
}

/// Initializes one new tablet group on this replica (split-child /
/// merge-target founder). Body: `{ "voters": [<u64>], "peer_addrs":
/// { "<u64>": "<addr>" } }`. Idempotent: an already-initialized group
/// answers `{ "ok": true }`.
async fn tablet_initialize(
    State(shared): State<ClusterShared>,
    axum::extract::Path(id): axum::extract::Path<u64>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Some(voters) = body.get("voters").and_then(serde_json::Value::as_array) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::json!({ "error": "body must be {\"voters\": [<u64>], \"peer_addrs\": {...}}" }),
            ),
        );
    };
    let voters: std::collections::BTreeSet<u64> = voters
        .iter()
        .filter_map(serde_json::Value::as_u64)
        .collect();
    let mut peer_addrs = std::collections::BTreeMap::new();
    if let Some(map) = body
        .get("peer_addrs")
        .and_then(serde_json::Value::as_object)
    {
        for (key, value) in map {
            if let (Ok(node), Some(addr)) = (key.parse::<u64>(), value.as_str()) {
                peer_addrs.insert(node, addr.to_owned());
            }
        }
    }
    match shared
        .node
        .initialize_group(TabletId::from_u64(id), voters, peer_addrs)
        .await
    {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({ "ok": true }))),
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        ),
    }
}

/// Sets or clears the split/merge cutover fence on the local replica.
/// Body: `{ "fenced": <bool> }`. Idempotent.
async fn tablet_fence(
    State(shared): State<ClusterShared>,
    axum::extract::Path(id): axum::extract::Path<u64>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Some(fenced) = body.get("fenced").and_then(serde_json::Value::as_bool) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "body must be {\"fenced\": <bool>}" })),
        );
    };
    match shared
        .node
        .set_tablet_fenced(TabletId::from_u64(id), fenced)
        .await
    {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({ "ok": true }))),
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        ),
    }
}

/// Seeds one split child from the local parent replica (colocated base,
/// no bulk network copy). Path id is the child; body:
/// `{ "parent": <u64>, "split_hash": "<u128 decimal>", "split_key": [<u8>],
/// "left": <bool> }` (`split_key` present for ordered splits, absent for
/// hash). Installs the child's half plus the full parent session set
/// (chunked roots reference the same immutable manifests). Idempotent.
async fn tablet_seed_split(
    State(shared): State<ClusterShared>,
    axum::extract::Path(id): axum::extract::Path<u64>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let (Some(parent), Some(split_hash), Some(left)) = (
        body.get("parent").and_then(serde_json::Value::as_u64),
        body.get("split_hash").and_then(|value| {
            value
                .as_str()
                .and_then(|text| text.parse::<u128>().ok())
                .or_else(|| {
                    // Accept raw JSON numbers when they fit (small test hashes).
                    value.as_u64().map(u128::from)
                })
        }),
        body.get("left").and_then(serde_json::Value::as_bool),
    ) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::json!({ "error": "body must be {\"parent\": <u64>, \"split_hash\": \"<u128>\", \"left\": <bool>}" }),
            ),
        );
    };
    let split_key: Option<Vec<u8>> = body.get("split_key").and_then(|value| {
        value.as_array().map(|bytes| {
            bytes
                .iter()
                .filter_map(serde_json::Value::as_u64)
                .filter_map(|byte| u8::try_from(byte).ok())
                .collect()
        })
    });
    let seeded = match split_key {
        Some(key) => {
            shared
                .node
                .seed_split_child_ordered(
                    TabletId::from_u64(parent),
                    TabletId::from_u64(id),
                    key,
                    left,
                )
                .await
        }
        None => {
            shared
                .node
                .seed_split_child(
                    TabletId::from_u64(parent),
                    TabletId::from_u64(id),
                    split_hash,
                    left,
                )
                .await
        }
    };
    match seeded {
        Ok((objects, sessions)) => (
            StatusCode::OK,
            Json(serde_json::json!({ "ok": true, "objects": objects, "sessions": sessions })),
        ),
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        ),
    }
}

/// Seeds a merge target from two local parents (union, ranges disjoint).
/// Path id is the merged tablet; body: `{ "left": <u64>, "right": <u64> }`.
/// Idempotent.
async fn tablet_seed_merge(
    State(shared): State<ClusterShared>,
    axum::extract::Path(id): axum::extract::Path<u64>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let (Some(left), Some(right)) = (
        body.get("left").and_then(serde_json::Value::as_u64),
        body.get("right").and_then(serde_json::Value::as_u64),
    ) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::json!({ "error": "body must be {\"left\": <u64>, \"right\": <u64>}" }),
            ),
        );
    };
    match shared
        .node
        .seed_merge_target(
            TabletId::from_u64(left),
            TabletId::from_u64(right),
            TabletId::from_u64(id),
        )
        .await
    {
        Ok((objects, sessions)) => (
            StatusCode::OK,
            Json(serde_json::json!({ "ok": true, "objects": objects, "sessions": sessions })),
        ),
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        ),
    }
}

/// Serving directory snapshot for operators and semantic waiters:
/// version, namespace, and every tablet with state, range, and tombstone
/// successors. Tests wait on exact tiling agreement across members
/// (same version + same active ranges), never on wall-clock sleeps.
async fn directory(State(shared): State<ClusterShared>) -> Json<serde_json::Value> {
    let snapshot = shared.directory_snapshot();
    let tablets: Vec<serde_json::Value> = snapshot
        .tablets()
        .iter()
        .map(|descriptor| {
            let (range_kind, range_detail) = match descriptor.range() {
                kivi_tablet::PartitionRange::Hash(prefix) => (
                    "hash",
                    serde_json::json!({
                        "bits": format!("{:032x}", prefix.bits()),
                        "len": prefix.prefix_len(),
                    }),
                ),
                kivi_tablet::PartitionRange::Ordered(range) => (
                    "ordered",
                    serde_json::json!({
                        "start": hex_bytes(range.start()),
                        "end": range.end().map(hex_bytes),
                    }),
                ),
            };
            serde_json::json!({
                "tablet": descriptor.id().as_u64(),
                "state": format!("{:?}", descriptor.state()),
                "range_kind": range_kind,
                "range": range_detail,
                "successors": descriptor
                    .redirect()
                    .map(|redirect| redirect
                        .successors()
                        .iter()
                        .map(|tablet| tablet.as_u64())
                        .collect::<Vec<_>>())
                    .unwrap_or_default(),
            })
        })
        .collect();
    Json(serde_json::json!({
        "version": snapshot.version().as_u64(),
        "namespace": snapshot.namespace().as_u64(),
        "ordered_coverage_complete": snapshot.ordered_coverage_complete(),
        "tablets": tablets,
    }))
}

/// Applies one deterministic topology cutover to the local directory
/// (idempotent). Split body:
/// `{ "kind": "split", "parent": <u64>, "left": <u64>, "right": <u64>,
/// "split_hash": "<u128>" }`; merge body:
/// `{ "kind": "merge", "left": <u64>, "right": <u64>, "merged": <u64> }`.
/// The control reconciler calls this on every node after fencing and the
/// final tail install; the single validated snapshot transition keeps
/// clients from ever observing a gap or double authority.
async fn directory_cutover(
    State(shared): State<ClusterShared>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let kind = body
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let current = shared.directory_snapshot().as_ref().clone();
    let next: Result<DirectorySnapshot, String> = match kind {
        "split" => {
            let (Some(parent), Some(left), Some(right), Some(split_hash)) = (
                body.get("parent").and_then(serde_json::Value::as_u64),
                body.get("left").and_then(serde_json::Value::as_u64),
                body.get("right").and_then(serde_json::Value::as_u64),
                body.get("split_hash").and_then(|value| {
                    value
                        .as_str()
                        .and_then(|text| text.parse::<u128>().ok())
                        .or_else(|| value.as_u64().map(u128::from))
                }),
            ) else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(
                        serde_json::json!({ "error": "split cutover needs parent/left/right/split_hash" }),
                    ),
                );
            };
            let split_key: Option<Vec<u8>> = body.get("split_key").and_then(|value| {
                value.as_array().map(|bytes| {
                    bytes
                        .iter()
                        .filter_map(serde_json::Value::as_u64)
                        .filter_map(|byte| u8::try_from(byte).ok())
                        .collect()
                })
            });
            apply_cutover_split_local(
                &current,
                TabletId::from_u64(parent),
                TabletId::from_u64(left),
                TabletId::from_u64(right),
                split_hash,
                split_key.as_deref(),
            )
        }
        "merge" => {
            let (Some(left), Some(right), Some(merged)) = (
                body.get("left").and_then(serde_json::Value::as_u64),
                body.get("right").and_then(serde_json::Value::as_u64),
                body.get("merged").and_then(serde_json::Value::as_u64),
            ) else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": "merge cutover needs left/right/merged" })),
                );
            };
            apply_cutover_merge_local(
                &current,
                TabletId::from_u64(left),
                TabletId::from_u64(right),
                TabletId::from_u64(merged),
            )
        }
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "body kind must be \"split\" or \"merge\"" })),
            );
        }
    };
    match next {
        Ok(next) => match shared.publish_directory(next) {
            Ok(version) => (
                StatusCode::OK,
                Json(serde_json::json!({ "ok": true, "dir_version": version })),
            ),
            Err(reason) => (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": reason })),
            ),
        },
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        ),
    }
}

/// Admits a node: commits `RegisterNode` (as `Joining`) then activates
/// it. Body: `{ "node": <u64>, "peer": "<addr>", "native": "<addr>",
/// "admin": "<addr>", "fingerprint": "<64 hex>"?, "domain": "<s>"? }`.
/// Effective only after the replicated commit (§8: no implicit trust).
async fn control_node_add(
    State(shared): State<ClusterShared>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let (Some(node), Some(peer), Some(native), Some(admin)) = (
        body.get("node").and_then(serde_json::Value::as_u64),
        body.get("peer").and_then(serde_json::Value::as_str),
        body.get("native").and_then(serde_json::Value::as_str),
        body.get("admin").and_then(serde_json::Value::as_str),
    ) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::json!({ "error": "body must be {\"node\", \"peer\", \"native\", \"admin\"}" }),
            ),
        );
    };
    if node == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "node id 0 is reserved" })),
        );
    }
    let parse = |value: &str| value.parse::<SocketAddr>();
    let (Ok(peer), Ok(native), Ok(admin)) = (parse(peer), parse(native), parse(admin)) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "peer/native/admin must be socket addresses" })),
        );
    };
    let fingerprint =
        match parse_fingerprint(body.get("fingerprint").and_then(serde_json::Value::as_str)) {
            Ok(fingerprint) => fingerprint,
            Err(reason) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": reason })),
                );
            }
        };
    let domain = body
        .get("domain")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let record = kivi_control::NodeRecord {
        node: NodeId::from_u64(node),
        peer,
        native,
        admin,
        cert_fingerprint: fingerprint,
        failure_domain: domain,
        weight: 1,
        state: kivi_control::NodeState::Joining,
    };
    if let Err(reason) = shared
        .node
        .propose_control(kivi_control::ControlMutation::RegisterNode { record })
        .await
    {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        );
    }
    if let Err(reason) = shared
        .node
        .propose_control(kivi_control::ControlMutation::SetNodeState {
            node: NodeId::from_u64(node),
            state: kivi_control::NodeState::Active,
        })
        .await
    {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        );
    }
    (
        StatusCode::OK,
        Json(serde_json::json!({ "ok": true, "node": node })),
    )
}

/// Starts draining a node: marks it `Draining` (excluded from new
/// placements) and creates replacement migration plans for every tablet
/// desiring it. Leaderships transfer away through the normal plan flow.
async fn control_node_drain(
    State(shared): State<ClusterShared>,
    axum::extract::Path(id): axum::extract::Path<u64>,
) -> (StatusCode, Json<serde_json::Value>) {
    let node = NodeId::from_u64(id);
    if let Err(reason) = shared
        .node
        .propose_control(kivi_control::ControlMutation::SetNodeState {
            node,
            state: kivi_control::NodeState::Draining,
        })
        .await
    {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        );
    }
    // Plans derive from fresh state (the drain mark must be visible).
    let mut attempts = 0;
    loop {
        let Some(state) = shared.node.control_state().await else {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": "no local control image" })),
            );
        };
        if state
            .node(node)
            .is_some_and(|record| record.state == kivi_control::NodeState::Draining)
        {
            let policy = super::control::ReconcilePolicy::default();
            let intents = super::control::drain_intents(&state, node, &policy);
            return match super::control::create_plans(&shared.node, &state, &intents).await {
                Ok(ids) => (
                    StatusCode::OK,
                    Json(
                        serde_json::json!({ "ok": true, "plans": ids.iter().map(|id| id.as_u64()).collect::<Vec<_>>() }),
                    ),
                ),
                Err(reason) => (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(serde_json::json!({ "error": reason })),
                ),
            };
        }
        attempts += 1;
        if attempts > 40 {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": "drain mark not visible; retry" })),
            );
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

/// Removes a node after safe drain. Fails loudly when the node still
/// desires tablets, still votes anywhere observed locally, or still
/// votes in the control group — never silently deletes an active
/// voter (§29). No force mode.
async fn control_node_remove(
    State(shared): State<ClusterShared>,
    axum::extract::Path(id): axum::extract::Path<u64>,
) -> (StatusCode, Json<serde_json::Value>) {
    let node = NodeId::from_u64(id);
    let Some(state) = shared.node.control_state().await else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": "no local control image" })),
        );
    };
    if !state.removable(node) {
        return (
            StatusCode::CONFLICT,
            Json(
                serde_json::json!({ "error": "node still draining or desired; drain first and wait for Drained" }),
            ),
        );
    }
    // Refuse while the node still votes in the control group (replace
    // control membership first through the tablet endpoints).
    if let Ok(observed) = shared
        .node
        .observe_membership(TabletId::from_u64(kivi_consensus::CONTROL_TABLET_RAW))
        .await
        && observed.is_voter(node)
    {
        return (
            StatusCode::CONFLICT,
            Json(
                serde_json::json!({ "error": "node still votes in the control group; shrink control membership first" }),
            ),
        );
    }
    match shared
        .node
        .propose_control(kivi_control::ControlMutation::SetNodeState {
            node,
            state: kivi_control::NodeState::Removed,
        })
        .await
    {
        Ok(_) => (StatusCode::OK, Json(serde_json::json!({ "ok": true }))),
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        ),
    }
}

/// Computes desired placements from current topology and creates
/// migration plans, bounded by the reconciler policy (no unbounded
/// snapshot storms, §25–26).
async fn control_rebalance(
    State(shared): State<ClusterShared>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Some(state) = shared.node.control_state().await else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": "no local control image" })),
        );
    };
    let policy = super::control::ReconcilePolicy::default();
    let intents = super::control::rebalance_intents(&state, &policy);
    match super::control::create_plans(&shared.node, &state, &intents).await {
        Ok(ids) => (
            StatusCode::OK,
            Json(
                serde_json::json!({ "ok": true, "plans": ids.iter().map(|id| id.as_u64()).collect::<Vec<_>>() }),
            ),
        ),
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": reason })),
        ),
    }
}

/// Rebalances tablet leadership explicitly (§53): graceful handoffs
/// away from overrepresented leaders, never continuous (no flapping).
async fn control_leadership(
    State(shared): State<ClusterShared>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Some(state) = shared.node.control_state().await else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": "no local control image" })),
        );
    };
    let transfers = super::control::rebalance_leadership(&shared.node, &state).await;
    (
        StatusCode::OK,
        Json(serde_json::json!({ "ok": true, "leadership_transfers": transfers })),
    )
}

/// Manual tablet move through the same persisted plan pathway as auto
/// rebalance (no second migration pathway, §46). Body:
/// `{ "tablet": <u64>, "from": <u64>, "to": <u64> }`.
async fn control_migrations_create(
    State(shared): State<ClusterShared>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let (Some(tablet), Some(from), Some(to)) = (
        body.get("tablet").and_then(serde_json::Value::as_u64),
        body.get("from").and_then(serde_json::Value::as_u64),
        body.get("to").and_then(serde_json::Value::as_u64),
    ) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "body must be {\"tablet\", \"from\", \"to\"}" })),
        );
    };
    if from == to || from == 0 || to == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "from and to must differ and be nonzero" })),
        );
    }
    let Some(state) = shared.node.control_state().await else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": "no local control image" })),
        );
    };
    let tablet = TabletId::from_u64(tablet);
    let from = NodeId::from_u64(from);
    let to = NodeId::from_u64(to);
    let Some(current) = state.desired(tablet) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "tablet has no desired placement" })),
        );
    };
    if !current.contains(from) {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": "source is not a desired voter of the tablet" })),
        );
    }
    if current.contains(to) {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": "target already desires the tablet" })),
        );
    }
    if state
        .node(to)
        .is_none_or(|record| record.state != kivi_control::NodeState::Active)
    {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": "target must be an Active node" })),
        );
    }
    if !state.live_plans_for(tablet).is_empty() {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": "tablet already has a live migration plan" })),
        );
    }
    let mut voters: Vec<NodeId> = current
        .replicas
        .iter()
        .copied()
        .filter(|node| *node != from)
        .collect();
    voters.push(to);
    let intent = kivi_control::MigrationIntent {
        tablet,
        from,
        to,
        desired_voters: voters,
    };
    match super::control::create_plans(&shared.node, &state, &[intent]).await {
        Ok(ids) => (
            StatusCode::OK,
            Json(
                serde_json::json!({ "ok": true, "plans": ids.iter().map(|id| id.as_u64()).collect::<Vec<_>>() }),
            ),
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
        .merge(crate::redundancy_admin::router())
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
        .route("/v1/peers/add", post(peers_add))
        .route("/v1/peers/trust", post(peers_trust))
        .route("/v1/snapshot", post(snapshot))
        .route("/v1/control", get(control))
        .route("/v1/control/migrations", get(migrations))
        .route(
            "/v1/control/migrations/create",
            post(control_migrations_create),
        )
        .route("/v1/control/splits", get(control_splits))
        .route("/v1/control/splits/create", post(control_splits_create))
        .route("/v1/control/merges", get(control_merges))
        .route("/v1/control/merges/create", post(control_merges_create))
        .route("/v1/control/repair", get(control_repair))
        .route("/v1/control/topology", get(control_topology))
        .route("/v1/control/policy", get(control_policy))
        .route("/v1/control/policy/update", post(control_policy_update))
        .route("/v1/control/nodes", post(control_node_add))
        .route("/v1/control/nodes/{id}/drain", post(control_node_drain))
        .route("/v1/control/nodes/{id}/remove", post(control_node_remove))
        .route("/v1/control/rebalance", post(control_rebalance))
        .route("/v1/control/leadership", post(control_leadership))
        .route("/v1/control/namespaces", get(control_namespaces))
        .route(
            "/v1/control/namespaces/register",
            post(control_namespaces_register),
        )
        .route("/v1/control/indexes", get(control_indexes))
        .route("/v1/control/indexes/create", post(control_indexes_create))
        .route(
            "/v1/control/indexes/{id}/advance",
            post(control_indexes_advance),
        )
        .route("/v1/control/indexes/{id}/drop", post(control_indexes_drop))
        .route("/v1/tablets/{id}/membership", get(tablet_membership))
        .route("/v1/tablets/{id}/learners", post(tablet_learners))
        .route("/v1/tablets/{id}/members", post(tablet_members))
        .route("/v1/tablets/{id}/leader", post(tablet_leader))
        .route("/v1/tablets/{id}/ensure", post(tablet_ensure))
        .route("/v1/tablets/{id}/retire", post(tablet_retire))
        .route("/v1/tablets/{id}/initialize", post(tablet_initialize))
        .route("/v1/tablets/{id}/fence", post(tablet_fence))
        .route("/v1/tablets/{id}/seed-split", post(tablet_seed_split))
        .route("/v1/tablets/{id}/seed-merge", post(tablet_seed_merge))
        .route("/v1/directory/cutover", post(directory_cutover))
        .route("/v1/directory", get(directory))
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
    directory: Arc<arc_swap::ArcSwap<DirectorySnapshot>>,
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
        directory: Arc<arc_swap::ArcSwap<DirectorySnapshot>>,
        namespace: NamespaceId,
    ) -> Self {
        Self {
            node,
            directory,
            namespace,
            handle: tokio::runtime::Handle::current(),
        }
    }

    /// Routes one operation to its tablet through the current directory
    /// (same key → tablet mapping as the native edge).
    fn route(&self, op: &Operation) -> Result<TabletId, kivi_resp::ExecuteError> {
        use kivi_resp::ExecuteError as E;
        use kivi_state::PartitionHasher;
        let hash = PartitionHasher::V1
            .hash(self.namespace, op.key().as_bytes())
            .ok_or(E::Internal)?;
        self.directory
            .load()
            .lookup_by_hash(hash)
            .ok_or(E::Internal)
    }

    /// Executes one typed operation against its tablet's replica:
    /// reads serve local applied state, mutations propose anonymously.
    /// RESP reads stay `Any` (the RESP edge promises no freshness
    /// contract); native reads name theirs explicitly.
    fn execute_inner(&self, op: &Operation) -> Result<OperationResult, kivi_resp::ExecuteError> {
        use kivi_resp::ExecuteError as E;
        // `block_on` from a blocking context: `RespConnection` drains on
        // the blocking pool (`spawn_blocking`), where no async worker
        // runs, so this never re-enters the runtime.
        let tablet = self.route(op)?;
        if op_is_read(op) {
            self.handle
                .block_on(self.node.read(
                    tablet,
                    op,
                    kivi_types::ReadContract::Any,
                    read_ctx(),
                    kivi_types::LeaseEligibility::Eligible,
                ))
                .map(|served| served.outcome)
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

    fn now(&self) -> WallTimestamp {
        wall_now()
    }
}
