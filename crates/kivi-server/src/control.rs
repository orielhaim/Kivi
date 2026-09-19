//! Replicated control-plane integration: genesis bootstrap, the
//! migration reconciler, and the tiny admin HTTP client it forwards
//! through.
//!
//! Only the active control-plane leader performs orchestration side
//! effects. Every step is idempotent: after a control-leader crash the
//! new leader reads persisted plans, observes actual group state, and
//! continues safely with no in-memory workflow continuation.
//!
//! Remote steps run through each node's admin plane (loopback-trusted,
//! like the existing admin surface): the control leader drives tablet
//! RPCs on whichever node hosts the tablet leader, and target-side
//! creation on the joining node. Local steps call the node front
//! directly.

use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use kivi_consensus::{ConsensusGroupId, ConsensusNode};
use kivi_control::{
    ControlMutation, ControlState, DesiredReplicaSet, MergePhase, MergePlan, MigrationIntent,
    MigrationPhase, MigrationPlan, MigrationPlanId, NodeRecord, NodeState, PlacementVersion,
    PlanSchedulerConfig, PlannerConfig, SplitPhase, SplitPlan, intents_to_plans, plan_drain,
    plan_rebalance,
};
use kivi_types::{NodeId, TabletId};

/// One seed node for genesis bootstrap.
#[derive(Debug, Clone)]
pub struct SeedNode {
    /// Stable identity.
    pub node: NodeId,
    /// Consensus (H3 peer) endpoint.
    pub peer: SocketAddr,
    /// Native client endpoint (redirect target).
    pub native: SocketAddr,
    /// Admin endpoint.
    pub admin: SocketAddr,
    /// BLAKE3 of the peer-TLS certificate DER (`[0; 32]` when unknown:
    /// lab-insecure only, never production).
    pub fingerprint: [u8; 32],
    /// Failure-domain label (empty when unknown).
    pub domain: String,
}

/// Genesis bootstrap input: the initial registry plus the initial
/// desired placement (every seed node hosts every seed tablet, the
/// preserved all-nodes-host-all-tablets starting shape).
#[derive(Debug, Clone)]
pub struct GenesisSeed {
    /// Initial nodes (admitted as `Joining`, then activated).
    pub nodes: Vec<SeedNode>,
    /// Initial tablets with full replication across all seed nodes.
    pub tablets: Vec<TabletId>,
    /// Namespace served (registered in control state with its layout).
    pub namespace: kivi_types::NamespaceId,
    /// Namespace physical layout (`hash` or `ordered`).
    pub layout: String,
}

/// Runs genesis bootstrap until the replicated control image holds the
/// seed registry and placements, then returns. Idempotent: founding
/// voters race the same batch (register is insert-if-absent, placements
/// overwrite identically, activation is a legal self-transition), and
/// only the control leader's proposals commit — followers and joiners
/// retry until the image appears through replication.
pub async fn genesis_loop(node: Arc<ConsensusNode>, seed: GenesisSeed) {
    loop {
        if genesis_done(&node, &seed).await {
            return;
        }
        // Best-effort batch; any failure retries next pass.
        let mut ok = true;
        // The served namespace is registered first (idempotent): the
        // control plane owns namespace/layout truth from genesis on.
        {
            let layout = match seed.layout.as_str() {
                "ordered" => kivi_control::CatalogLayout::Ordered,
                _ => kivi_control::CatalogLayout::Hash,
            };
            let mutation = ControlMutation::RegisterNamespace {
                record: kivi_control::NamespaceRecord {
                    id: seed.namespace,
                    layout,
                },
            };
            if node.propose_control(mutation).await.is_err() {
                ok = false;
            }
        }
        for record in &seed.nodes {
            let mutation = ControlMutation::RegisterNode {
                record: NodeRecord {
                    node: record.node,
                    peer: record.peer,
                    native: record.native,
                    admin: record.admin,
                    cert_fingerprint: record.fingerprint,
                    failure_domain: record.domain.clone(),
                    weight: 1,
                    state: NodeState::Joining,
                },
            };
            if node.propose_control(mutation).await.is_err() {
                ok = false;
                break;
            }
        }
        if ok {
            for tablet in &seed.tablets {
                let desired = DesiredReplicaSet::new(
                    *tablet,
                    seed.nodes.iter().map(|seed| seed.node).collect(),
                    PlacementVersion::INITIAL,
                );
                let Ok(desired) = desired else {
                    ok = false;
                    break;
                };
                if node
                    .propose_control(ControlMutation::SetDesiredPlacement { desired })
                    .await
                    .is_err()
                {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            for record in &seed.nodes {
                if node
                    .propose_control(ControlMutation::SetNodeState {
                        node: record.node,
                        state: NodeState::Active,
                    })
                    .await
                    .is_err()
                {
                    ok = false;
                    break;
                }
            }
        }
        if ok && genesis_done(&node, &seed).await {
            tracing::info!("control-plane genesis committed");
            return;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// Whether the control image already holds every seed node as `Active`
/// with every seed tablet placed and the namespace registered.
async fn genesis_done(node: &ConsensusNode, seed: &GenesisSeed) -> bool {
    let Some(state) = node.control_state().await else {
        return false;
    };
    if state.namespace(seed.namespace).is_none() {
        return false;
    }
    for record in &seed.nodes {
        if state
            .node(record.node)
            .is_none_or(|known| known.state != NodeState::Active)
        {
            return false;
        }
    }
    for tablet in &seed.tablets {
        if state.desired(*tablet).is_none() {
            return false;
        }
    }
    true
}

/// Reconciler tuning.
#[derive(Debug, Clone)]
pub struct ReconcilePolicy {
    /// Delay between passes.
    pub poll_interval: Duration,
    /// Placement policy (replication factor, per-round bounds).
    pub planner: PlannerConfig,
    /// Maximum live (non-terminal) plans cluster-wide. New plans wait
    /// while catch-up moves gigabytes.
    pub max_live_plans: usize,
    /// Failure-detector thresholds (suspicion, repair grace, cooldown).
    pub failure: kivi_control::FailureDetectorConfig,
    /// Whether the control leader automatically plans repairs for
    /// `Unavailable` nodes. Defaults on: thresholds are conservative, so
    /// short outages never re-replicate, and repair reuses the standard
    /// migration path.
    pub auto_repair: bool,
    /// Cross-plan scheduler bounds (migrations vs splits vs merges).
    pub scheduler: PlanSchedulerConfig,
    /// Whether the control leader automatically plans splits for hot /
    /// large tablets. Defaults off for manual-testing conservatism;
    /// enabling it is a supported feature (see `SplitPolicy`).
    pub auto_split: bool,
    /// Whether the control leader automatically plans merges for small /
    /// cold adjacent tablets. Defaults off; enabling it is supported.
    pub auto_merge: bool,
    /// Object-count threshold that triggers an automatic split (with
    /// hysteresis: merge threshold must stay far below this).
    pub split_object_threshold: usize,
    /// Logical-byte threshold that triggers an automatic ordered split
    /// (ordered tablets split by bytes, not object count).
    pub split_bytes_threshold: u64,
    /// Object-count threshold below which BOTH adjacent tablets must sit
    /// before an automatic merge is planned.
    pub merge_object_threshold: usize,
}

impl Default for ReconcilePolicy {
    /// Production shaping: brisk passes, RF=3, bounded churn, conservative
    /// failure detection with automatic repair. Automatic topology
    /// adaptation defaults off (manual split/merge supported); enabling
    /// it is a real feature with hysteresis (`split >> merge`).
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_millis(500),
            planner: PlannerConfig::default_rf3(),
            max_live_plans: 32,
            failure: kivi_control::FailureDetectorConfig::conservative(),
            auto_repair: true,
            scheduler: PlanSchedulerConfig::conservative(),
            auto_split: false,
            auto_merge: false,
            split_object_threshold: 200_000,
            split_bytes_threshold: 64 << 20,
            merge_object_threshold: 20_000,
        }
    }
}

/// Runs the reconciler until aborted: every pass syncs the peer mesh
/// from the replicated registry (all nodes), then — only on the control
/// leader — drives migrations, repairs, splits, and merges: reads
/// persisted plans, observes actual membership through
/// [`kivi_consensus::select_authoritative`], and advances one idempotent
/// step per live plan.
///
/// One source of desired-state reconciliation: no second topology
/// controller races this loop. Plan kinds serialize per tablet (one live
/// plan per tablet) and globally for topology (one live split, one live
/// merge) via [`PlanSchedulerConfig`].
///
/// The policy is re-read every pass from `policy` (updated live via
/// `POST /v1/control/policy`), while the failure detector keeps its
/// counters across passes: a new control leader starts empty (failover
/// never inherits stale suspicion), but a policy retune never wipes
/// live suspicion history (only thresholds update).
pub async fn reconcile_loop_shared(
    node: Arc<ConsensusNode>,
    policy: Arc<std::sync::Mutex<ReconcilePolicy>>,
    directory: Arc<arc_swap::ArcSwap<kivi_tablet::DirectorySnapshot>>,
) {
    let initial = policy
        .lock()
        .map_or_else(|_| ReconcilePolicy::default(), |policy| policy.clone());
    let mut detector = kivi_control::FailureDetector::new(initial.failure.clone());
    loop {
        let snapshot = policy
            .lock()
            .map_or_else(|_| ReconcilePolicy::default(), |policy| policy.clone());
        detector.set_config(snapshot.failure.clone());
        tokio::time::sleep(snapshot.poll_interval).await;
        mesh_sync(&node).await;
        // Transaction resolution runs on every node (not just the control
        // leader): each node resolves intents on tablets it leads.
        // Bounded per pass; failures simply retry next pass.
        resolve_transactions(&node).await;
        // Ordered-index enforcement likewise runs everywhere: each replica
        // maintains its own index, driven from the namespace layout.
        ensure_ordered_indexing(&node, &directory).await;
        if let Err(reason) = reconcile_once(&node, &snapshot, &mut detector, &directory).await {
            tracing::debug!(%reason, "reconciler pass skipped");
        }
    }
}

/// Syncs dialable mesh links from the replicated registry on every
/// node (not just the control leader): statically unknown admitted
/// nodes become dialable without restarts. Links only add, never
/// replace; removals leave dormant links (harmless, no traffic).
async fn mesh_sync(node: &Arc<ConsensusNode>) {
    let Some(state) = node.control_state().await else {
        return;
    };
    for record in state.nodes() {
        if record.node == node.node() {
            continue;
        }
        // Suspect nodes stay meshed (a transient miss must not partition
        // them); Unavailable nodes stay known but are not re-admitted
        // (their links go dormant, no traffic, until they return).
        if matches!(
            record.state,
            NodeState::Joining | NodeState::Active | NodeState::Suspect | NodeState::Draining
        ) && node.add_peer_dial(record.node, record.peer)
        {
            tracing::info!(node = record.node.as_u64(), peer = %record.peer, "mesh admitted peer");
        }
    }
}

/// One reconciler pass. `Ok` means the pass completed (plans may still
/// be in flight); `Err` is a skip reason (not leader, no control
/// replica, transient observation failure).
async fn reconcile_once(
    node: &Arc<ConsensusNode>,
    policy: &ReconcilePolicy,
    detector: &mut kivi_control::FailureDetector,
    directory: &Arc<arc_swap::ArcSwap<kivi_tablet::DirectorySnapshot>>,
) -> Result<(), String> {
    // Only the control leader orchestrates.
    let status = node.status_for(ConsensusGroupId::control()).await;
    if status.role != kivi_consensus::ReplicaRole::Leader {
        return Err("not the control leader".to_owned());
    }
    let Some(state) = node.control_state().await else {
        return Err("no local control image".to_owned());
    };
    // Admit Active data nodes to control as learners so they observe
    // placement for routing (votes stay with the control voters).
    ensure_control_learners(node, &state, &control_voters_of(node).await).await;
    // Liveness first (ephemeral probes -> replicated transitions), then
    // existing plans, then new automated work. New topology actions
    // (liveness transitions, drain top-ups, repairs) require control-plane
    // quorum: the leader role alone proves a quorum elected us, but a
    // partition could have since isolated us — gate on reachable control
    // voters so a lone leader never re-replicates the cluster alone.
    let liveness = probe_liveness(node, &state).await;
    let quorum = control_quorum(node, &state, &liveness).await;
    // Drive every live plan one step, concurrently: plans touch
    // independent tablet groups, so sequential passes would serialize
    // dozens of migrations behind one slow step. Each task owns its
    // plan snapshot; phase advances commit through Raft (serialized
    // there) and every step is idempotent, so overlap is safe. A stuck
    // step (e.g., a commit awaiting lost quorum) must never wedge the
    // pass: each plan gets a bounded slice, then the pass moves on and
    // retries next round.
    let live: Vec<MigrationPlan> = state
        .migrations()
        .filter(|plan| !plan.phase.is_terminal())
        .cloned()
        .collect();
    let mut set = tokio::task::JoinSet::new();
    for plan in live {
        let node = Arc::clone(node);
        let state = state.clone();
        set.spawn(async move {
            // Bound every plan step: a hung step (commit awaiting lost
            // quorum, wedged peer) ends here instead of leaking a
            // detached task that piles onto every future pass. The
            // underlying Raft op, if any, continues server-side and is
            // idempotent on retry next pass.
            let _ = tokio::time::timeout(Duration::from_secs(30), drive_plan(&node, &state, &plan))
                .await;
        });
    }
    while set.join_next().await.is_some() {}
    // Drive live topology plans (splits/merges) one step each,
    // sequentially: at most one live split and one live merge exist
    // cluster-wide, so concurrency buys nothing and serialization keeps
    // the directory cutover easy to reason about.
    {
        let splits: Vec<SplitPlan> = state
            .splits()
            .filter(|plan| !plan.phase.is_terminal())
            .cloned()
            .collect();
        for plan in splits {
            let _ = tokio::time::timeout(
                Duration::from_secs(60),
                drive_split(node, &state, &plan, directory),
            )
            .await;
        }
        let merges: Vec<MergePlan> = state
            .merges()
            .filter(|plan| !plan.phase.is_terminal())
            .cloned()
            .collect();
        for plan in merges {
            let _ = tokio::time::timeout(
                Duration::from_secs(60),
                drive_merge(node, &state, &plan, directory),
            )
            .await;
        }
    }
    if quorum {
        drive_liveness(node, &state, detector, &liveness).await;
        // Re-read: liveness transitions may have changed node states that
        // repair planning reads (a fresh Unavailable must repair now).
        let fresh = node.control_state().await.unwrap_or(state);
        // Top up drain plans as capacity frees: draining nodes need EVERY
        // tablet moved, but plan creation stays bounded per round, so the
        // reconciler refills until nothing desires the draining node.
        // (Rebalance stays operator-driven per round; drain completion is
        // operator-initiated and reconciler-finished.)
        topup_drains(node, &fresh, policy).await;
        // Automatic repair (safety before optimization): under-replicated
        // tablets gain replacements through the same migration path.
        drive_repairs(node, &fresh, policy).await;
        // Automatic topology optimization (hysteresis-guarded, off by
        // default): split hot/large tablets, merge small/cold siblings.
        let planned = node.control_state().await.unwrap_or(fresh);
        drive_topology_auto(node, &planned, policy, directory).await;
        // Complete drains whose migrations all finished.
        let done = node.control_state().await.unwrap_or(planned);
        complete_drains(node, &done).await;
        // Bound tombstone/history growth: keep only the newest terminal
        // plans per kind (retention for fencing/debug), removing older
        // ones. Tombstones needed for fencing/restart stay (retired
        // tablet tombstones live in `consensus-sm`, not here); removed
        // plans are terminal, unreferenced by the directory (cutover
        // already published), and past any reconciler retry.
        let pruned = node.control_state().await.unwrap_or(done);
        prune_terminal_plans(node, &pruned).await;
    } else {
        tracing::debug!("control quorum unreachable; deferring automated topology actions");
    }
    Ok(())
}

/// Enforces the ordered key index from the namespace layout on every
/// locally hosted replica: ordered-range tablets maintain it from birth
/// (scans, median splits, and index maintenance read it); hash tablets
/// keep the fast path. Late enablement rebuilds deterministically from
/// committed objects, so this converges tablets hosted before the flag
/// existed, after restores, and after any layout change. Flag reads make
/// the steady state free (one lock + bool per tablet per pass).
async fn ensure_ordered_indexing(
    node: &Arc<ConsensusNode>,
    directory: &Arc<arc_swap::ArcSwap<kivi_tablet::DirectorySnapshot>>,
) {
    let snapshot = directory.load();
    for tablet in node.tablets() {
        let ordered = snapshot.get(tablet).is_some_and(|descriptor| {
            matches!(descriptor.range(), kivi_tablet::PartitionRange::Ordered(_))
        });
        if !ordered {
            continue;
        }
        if node.ordered_index_enabled(tablet).await != Some(true) {
            let _ = node.set_ordered_indexing(tablet, true).await;
        }
    }
}

/// Resolves abandoned transaction intents on every locally hosted tablet.
///
/// Runs on all nodes (not just the control leader): each node resolves
/// tablets it leads, skipping others via `NotLeader`. Per intent: a durable
/// Commit/Abort decision finalizes accordingly; undecided records past
/// the lease CAS-abort through the record (never unilaterally — the CAS
/// proves no commit won the race); live intents wait. Bounded per pass;
/// the next pass retries the rest.
async fn resolve_transactions(node: &Arc<ConsensusNode>) {
    let namespace = node.namespace();
    for tablet in node.tablets() {
        let intents = node.tablet_intents(tablet).await.unwrap_or_default();
        if intents.is_empty() {
            continue;
        }
        for intent in intents {
            resolve_one_intent(node, namespace, tablet, &intent).await;
        }
    }
}

/// Resolves one intent: follows the coordinator record's durable
/// decision, or CAS-aborts an expired undecided transaction. Decided
/// records converge only intents older than [`kivi_state::RESOLVE_GRACE_MICROS`]:
/// fresher intents belong to a live driver still waving its finalizes,
/// and converging them from here races that wave (fatal to the node).
async fn resolve_one_intent(
    node: &Arc<ConsensusNode>,
    namespace: kivi_types::NamespaceId,
    tablet: TabletId,
    intent: &kivi_state::TxnIntent,
) {
    use kivi_state::{Operation, OperationResult, TxnRecord, TxnState};
    let _ = namespace;
    let record_key = kivi_state::txn_record_key(intent.coordinator, intent.id);
    let now = super::cluster::wall_now();
    let ctx = super::cluster::read_ctx();
    // Read the decision record (fiat-routed to the coordinator tablet;
    // followers redirect and this pass skips). Lease-ineligible: 2PC
    // decisions integrate with final participant outcomes, never with a
    // roster fast path.
    let record = match node
        .read(
            intent.coordinator,
            &Operation::Get {
                key: record_key.clone(),
            },
            kivi_types::ReadContract::Latest,
            ctx,
            kivi_types::LeaseEligibility::ConservativeOnly,
        )
        .await
    {
        Ok(served) => {
            if let OperationResult::Value(Some(bytes)) = served.outcome {
                match TxnRecord::decode(&bytes) {
                    Ok(record) => Some(record),
                    Err(_) => return,
                }
            } else {
                // Missing record with an expired lease: fence first, then
                // discard. The fence CASes an Aborted decision on absence;
                // a slow driver racing to decide CASes the same key, so
                // exactly one wins and the loser follows the winner on
                // re-read (next pass / re-drive). Without the fence a slow
                // commit could land after this discard and report success
                // for a write that never applied.
                if lease_expired(intent.prepared_at, now)
                    && cas_abort_absent_record(node, intent.coordinator, intent.id).await
                {
                    let _ = node
                        .propose(
                            tablet,
                            &Operation::TxnFinalize {
                                txn: intent.id,
                                key: intent.key.clone(),
                                commit: false,
                                digest: intent.digest,
                            },
                            None,
                            None,
                            now,
                        )
                        .await;
                }
                return;
            }
        }
        Err(_) => return,
    };
    let Some(record) = record else {
        return;
    };
    match record.state {
        TxnState::Committed => {
            if intent_fresh(intent.prepared_at, now) {
                return;
            }
            let _ = node
                .propose(
                    tablet,
                    &Operation::TxnFinalize {
                        txn: intent.id,
                        key: intent.key.clone(),
                        commit: true,
                        digest: intent.digest,
                    },
                    None,
                    None,
                    now,
                )
                .await;
        }
        TxnState::Aborted => {
            if intent_fresh(intent.prepared_at, now) {
                return;
            }
            let _ = node
                .propose(
                    tablet,
                    &Operation::TxnFinalize {
                        txn: intent.id,
                        key: intent.key.clone(),
                        commit: false,
                        digest: intent.digest,
                    },
                    None,
                    None,
                    now,
                )
                .await;
        }
        TxnState::Begun => {
            if lease_expired(intent.prepared_at, now) {
                cas_abort_record(node, &record).await;
            }
        }
    }
}

/// Whether a prepare lease expired (`prepared_at` micros + max lifetime
/// below `now`).
fn lease_expired(prepared_at: u64, now: kivi_types::UnixMicros) -> bool {
    prepared_at.saturating_add(kivi_state::MAX_TXN_LIFETIME_MICROS) < now.as_micros()
}

/// Whether an intent is younger than the resolver grace (its driver is
/// presumably still live: hands off).
fn intent_fresh(prepared_at: u64, now: kivi_types::UnixMicros) -> bool {
    now.as_micros().saturating_sub(prepared_at) < kivi_state::RESOLVE_GRACE_MICROS
}

/// CAS-aborts a record-absent expired transaction: an absence-guarded
/// prepare + commit-finalize of a skeletal Aborted record on the
/// coordinator tablet. Wins only when no driver decided concurrently (a
/// concurrent commit conflicts instead, and the resolver follows that
/// decision next pass). Readers only match on `state`, so the skeletal
/// payload (no participants, zero digest) is safe: Aborted carries no
/// obligations beyond "discard intents", which every pass converges.
/// Best-effort: any failure simply retries later.
async fn cas_abort_absent_record(
    node: &Arc<ConsensusNode>,
    coordinator: kivi_types::TabletId,
    txn: kivi_state::TxnId,
) -> bool {
    use kivi_consensus::ProposeOutcome;
    use kivi_state::{
        Operation, OperationResult, TxnExpect, TxnRecord, TxnState, TxnWrite, TxnWriteKind,
    };
    let record_key = kivi_state::txn_record_key(coordinator, txn);
    let aborted = TxnRecord {
        id: txn,
        coordinator,
        participants: Vec::new(),
        state: TxnState::Aborted,
        dir_version: 0,
        digest: [0u8; 32],
    };
    // Resolver salt (2) never aliases driver decide-transactions
    // (commit 0 / abort 1): distinct intents, genuine OCC on the key.
    let abort_txn = kivi_state::TxnId::derive(u128::from_le_bytes(txn.as_bytes()), 0, 2);
    let now = super::cluster::wall_now();
    let prepare = Operation::TxnPrepare {
        txn: abort_txn,
        coordinator,
        write: TxnWrite {
            key: record_key.clone(),
            kind: TxnWriteKind::Put(bytes::Bytes::from(aborted.encode())),
            expect: TxnExpect::Absent,
        },
        // Record-key steps bind the zero digest (no user write set).
        digest: [0u8; 32],
    };
    if !matches!(
        node.propose(coordinator, &prepare, None, None, now).await,
        Ok(ProposeOutcome::Applied {
            outcome: OperationResult::TxnPrepared,
            ..
        } | ProposeOutcome::Duplicate {
            outcome: OperationResult::TxnPrepared,
            ..
        },)
    ) {
        return false;
    }
    let finalize = Operation::TxnFinalize {
        txn: abort_txn,
        key: record_key,
        commit: true,
        digest: [0u8; 32],
    };
    matches!(
        node.propose(coordinator, &finalize, None, None, now).await,
        Ok(ProposeOutcome::Applied {
            outcome: OperationResult::TxnFinalized { applied: true, .. },
            ..
        } | ProposeOutcome::Duplicate {
            outcome: OperationResult::TxnFinalized { applied: true, .. },
            ..
        },)
    )
}

/// CAS-aborts an expired undecided transaction: a version-guarded record
/// write wins only when no commit landed concurrently (losers re-read the
/// decision and follow it next pass). Runs as a guarded single-key batch
/// on the record tablet (same fast-path machinery, driven internally).
async fn cas_abort_record(node: &Arc<ConsensusNode>, record: &kivi_state::TxnRecord) {
    let record_key = kivi_state::txn_record_key(record.coordinator, record.id);
    let ctx = super::cluster::read_ctx();
    let Ok(served) = node
        .read(
            record.coordinator,
            &kivi_state::Operation::GetVersion {
                key: record_key.clone(),
            },
            kivi_types::ReadContract::Latest,
            ctx,
            kivi_types::LeaseEligibility::ConservativeOnly,
        )
        .await
    else {
        return;
    };
    let kivi_state::OperationResult::Version(Some(version)) = served.outcome else {
        return;
    };
    let mut aborted = record.clone();
    aborted.state = kivi_state::TxnState::Aborted;
    let () = super::compound::handle_guard_abort(
        node,
        record.coordinator,
        record_key,
        aborted.encode(),
        version,
    )
    .await;
}

/// Probes liveness for every registry node (ephemeral, never persisted):
/// self is always alive; peers get one bounded TCP dial to their admin
/// plane per pass, in parallel. Any successful control/tablet admin RPC
/// later in the pass also counts as alive implicitly (the detector only
/// needs a best-effort signal; thresholds absorb the noise).
async fn probe_liveness(node: &Arc<ConsensusNode>, state: &ControlState) -> BTreeMap<u64, bool> {
    use tokio::task::JoinSet;
    let mut out = BTreeMap::new();
    let mut set = JoinSet::new();
    for record in state.nodes() {
        if record.node == node.node() {
            out.insert(record.node.as_u64(), true);
            continue;
        }
        // Removed nodes are gone (forget below); only probe live lifecycle
        // states. Probing keeps no connections: one dial, immediate close.
        if !matches!(
            record.state,
            NodeState::Joining
                | NodeState::Active
                | NodeState::Suspect
                | NodeState::Unavailable
                | NodeState::Draining
        ) {
            continue;
        }
        let admin = record.admin;
        let id = record.node.as_u64();
        set.spawn(async move {
            let dial = tokio::net::TcpStream::connect(admin);
            let alive = tokio::time::timeout(Duration::from_millis(800), dial)
                .await
                .is_ok_and(|result| result.is_ok());
            (id, alive)
        });
    }
    while let Some(joined) = set.join_next().await {
        if let Ok((id, alive)) = joined {
            out.insert(id, alive);
        }
    }
    out
}

/// Whether the control leader currently holds quorum among reachable
/// control voters (self + probed-alive voters >= majority). Automated
/// topology actions require this; driving already-persisted plans does
/// not (their Raft proposals fail safely without quorum anyway).
async fn control_quorum(
    node: &Arc<ConsensusNode>,
    _state: &ControlState,
    liveness: &BTreeMap<u64, bool>,
) -> bool {
    let voters = control_voters_of(node).await;
    if voters.is_empty() {
        // No control replica locally (should not happen on the leader):
        // fail closed, take no automated actions.
        return false;
    }
    let reachable = voters
        .iter()
        .filter(|id| **id == node.node().as_u64() || liveness.get(*id).copied().unwrap_or(false))
        .count();
    reachable * 2 > voters.len()
}

/// Drives replicated liveness transitions from ephemeral probe results.
/// One transition per node per pass at most (Active -> Suspect ->
/// Unavailable, either -> Active on recovery). Failures only defer to
/// the next pass; the detector counters persist in memory across passes.
async fn drive_liveness(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    detector: &mut kivi_control::FailureDetector,
    liveness: &BTreeMap<u64, bool>,
) {
    for record in state.nodes() {
        if record.node == node.node() {
            continue;
        }
        if record.state == NodeState::Removed || record.state == NodeState::Drained {
            detector.forget(record.node);
            continue;
        }
        // Operator-driven states are never auto-transitioned out from
        // under the operator, except the documented Unavailable paths.
        if matches!(record.state, NodeState::Joining | NodeState::Draining) {
            continue;
        }
        let alive = liveness
            .get(&record.node.as_u64())
            .copied()
            .unwrap_or(false);
        if let Some(next) = detector.observe(record.node, alive, record.state) {
            tracing::info!(
                node_id = record.node.as_u64(),
                from = ?record.state,
                to = ?next,
                alive,
                "node liveness transition"
            );
            if let Err(reason) = node
                .propose_control(ControlMutation::SetNodeState {
                    node: record.node,
                    state: next,
                })
                .await
            {
                tracing::debug!(node = record.node.as_u64(), %reason, "liveness transition deferred");
            }
        }
    }
}

/// Plans automatic repairs for `Unavailable` nodes through the same
/// `create_plans` pathway as manual moves (one mechanism, same learner /
/// membership / retirement steps). Honors the global live-plan cap and
/// the planner's per-source/per-target repair bounds.
async fn drive_repairs(node: &Arc<ConsensusNode>, state: &ControlState, policy: &ReconcilePolicy) {
    if !policy.auto_repair {
        return;
    }
    let live = state
        .migrations()
        .filter(|plan| !plan.phase.is_terminal())
        .count();
    if live >= policy.max_live_plans {
        return;
    }
    let intents = kivi_control::plan_repair(state, &policy.planner);
    if intents.is_empty() {
        return;
    }
    let budget = policy.max_live_plans.saturating_sub(live);
    let intents: Vec<MigrationIntent> = intents.into_iter().take(budget.max(1)).collect();
    let Some(fresh) = node.control_state().await else {
        return;
    };
    match create_plans(node, &fresh, &intents).await {
        Ok(ids) => {
            tracing::info!(plans = ids.len(), "automatic repair round created");
        }
        Err(reason) => {
            tracing::debug!(%reason, "automatic repair deferred");
        }
    }
}

/// Advances one split plan's persisted phase (generation-fenced).
async fn advance_split(node: &Arc<ConsensusNode>, plan: &SplitPlan, phase: SplitPhase) {
    tracing::info!(
        node_id = node.node().as_u64(),
        plan_id = plan.id.as_u64(),
        parent = plan.parent.as_u64(),
        left = plan.left.as_u64(),
        right = plan.right.as_u64(),
        ?phase,
        "split plan advanced"
    );
    let generation = node
        .control_state()
        .await
        .and_then(|state| state.split(plan.id).map(|plan| plan.generation))
        .unwrap_or(PlacementVersion::INITIAL);
    if let Err(reason) = node
        .propose_control(ControlMutation::AdvanceSplit {
            plan: plan.id.into(),
            phase,
            generation,
        })
        .await
    {
        tracing::debug!(plan = plan.id.as_u64(), %reason, "split advance deferred");
    }
}

/// Advances one merge plan's persisted phase (generation-fenced).
async fn advance_merge(
    node: &Arc<ConsensusNode>,
    plan: &kivi_control::MergePlan,
    phase: MergePhase,
) {
    tracing::info!(
        node_id = node.node().as_u64(),
        plan_id = plan.id.as_u64(),
        left = plan.left.as_u64(),
        right = plan.right.as_u64(),
        merged = plan.merged.as_u64(),
        ?phase,
        "merge plan advanced"
    );
    let generation = node
        .control_state()
        .await
        .and_then(|state| state.merge(plan.id).map(|plan| plan.generation))
        .unwrap_or(PlacementVersion::INITIAL);
    if let Err(reason) = node
        .propose_control(ControlMutation::AdvanceMerge {
            plan: plan.id.into(),
            phase,
            generation,
        })
        .await
    {
        tracing::debug!(plan = plan.id.as_u64(), %reason, "merge advance deferred");
    }
}

/// Whether one parent tablet is under-replicated (a degraded parent
/// never splits or merges: safety before optimization).
fn parent_degraded(state: &ControlState, parent: TabletId) -> bool {
    state.desired(parent).is_some_and(|desired| {
        desired.replicas.iter().any(|replica| {
            state.node(*replica).is_some_and(|record| {
                matches!(record.state, NodeState::Unavailable | NodeState::Removed)
            })
        })
    })
}

/// Returns the parents in `parents` holding unresolved intents: intents
/// never migrate, so any blocked parent defers drain, fence, seal, and
/// retire alike.
async fn blocked_parents(node: &Arc<ConsensusNode>, parents: &[TabletId]) -> Vec<TabletId> {
    let mut blocked = Vec::new();
    for parent in parents {
        if node
            .tablet_intent_count(*parent)
            .await
            .is_some_and(|count| count > 0)
        {
            blocked.push(*parent);
        }
    }
    blocked
}

/// Drives one split plan a single idempotent step.
///
/// Repair wins over split: a parent with a live migration plan (repair,
/// drain, or rebalance) defers the split until redundancy is healthy.
async fn drive_split(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    plan: &SplitPlan,
    directory: &Arc<arc_swap::ArcSwap<kivi_tablet::DirectorySnapshot>>,
) {
    if plan.phase.is_terminal() {
        return;
    }
    if !state.live_plans_for(plan.parent).is_empty() {
        tracing::debug!(
            plan = plan.id.as_u64(),
            parent = plan.parent.as_u64(),
            "split deferred: parent has a live migration (repair wins)"
        );
        return;
    }
    // A degraded parent never splits: safety before optimization.
    if parent_degraded(state, plan.parent) {
        tracing::debug!(
            plan = plan.id.as_u64(),
            parent = plan.parent.as_u64(),
            "split deferred: parent under-replicated"
        );
        return;
    }
    match plan.phase {
        SplitPhase::Planned => {
            if ensure_split_children(node, state, plan).await
                && initialize_split_children(node, state, plan).await
            {
                advance_split(node, plan, SplitPhase::ChildrenAllocated).await;
            }
        }
        SplitPhase::ChildrenAllocated => {
            if seed_split_children(node, state, plan).await {
                advance_split(node, plan, SplitPhase::BaseSeeded).await;
            }
        }
        SplitPhase::BaseSeeded => {
            // Drain before fence: unresolved intents block the seal (they
            // never migrate — topology copies carry objects and sessions
            // only), and finalizes flow only while unfenced. Fencing with
            // live intents would strand them: the fence blocks the very
            // finalizes that could resolve them.
            if !blocked_parents(node, &[plan.parent]).await.is_empty() {
                tracing::debug!(
                    plan = plan.id.as_u64(),
                    parent = plan.parent.as_u64(),
                    "split deferred: parent holds unresolved intents"
                );
                return;
            }
            let fenced_at = std::time::Instant::now();
            if fence_tablets(node, state, &[plan.parent], true).await
                && seed_split_children(node, state, plan).await
                && snapshot_split_children(node, state, plan).await
            {
                tracing::info!(
                    plan = plan.id.as_u64(),
                    parent = plan.parent.as_u64(),
                    fence_ms = u64::try_from(fenced_at.elapsed().as_millis()).unwrap_or(u64::MAX),
                    "split final tail installed under fence"
                );
                advance_split(node, plan, SplitPhase::Fenced).await;
            } else {
                let _ = fence_tablets(node, state, &[plan.parent], false).await;
            }
        }
        SplitPhase::Fenced => {
            // Seal gate: a prepare admitted between the drain check and the
            // fence would strand at cutover (intents never migrate), so the
            // seal re-verifies zero intents under fence. On a race the seal
            // unfences and regresses to BaseSeeded (finalizes flow again,
            // the next pass re-drains and re-fences) rather than stranding
            // transactional state in a retired lineage.
            if !blocked_parents(node, &[plan.parent]).await.is_empty() {
                tracing::info!(
                    plan = plan.id.as_u64(),
                    parent = plan.parent.as_u64(),
                    "split seal deferred: intents raced the fence; unfencing"
                );
                let _ = fence_tablets(node, state, &[plan.parent], false).await;
                advance_split(node, plan, SplitPhase::BaseSeeded).await;
                return;
            }
            if publish_split_cutover(node, state, plan, directory).await {
                advance_split(node, plan, SplitPhase::CutoverCommitted).await;
            }
        }
        SplitPhase::CutoverCommitted => {
            // Retirement reclaims the parent lineage: never retire under
            // unresolved intents (a stranded intent could never resolve
            // once its group is gone).
            if !blocked_parents(node, &[plan.parent]).await.is_empty() {
                tracing::debug!(
                    plan = plan.id.as_u64(),
                    parent = plan.parent.as_u64(),
                    "split retire deferred: parent holds unresolved intents"
                );
                return;
            }
            if retire_tablets(node, state, &[plan.parent], plan.generation.as_u64()).await {
                advance_split(node, plan, SplitPhase::ParentRetiring).await;
            }
        }
        SplitPhase::ParentRetiring => {
            advance_split(node, plan, SplitPhase::Completed).await;
        }
        SplitPhase::Completed | SplitPhase::Failed => {}
    }
}

/// Defers a merge step while any parent holds unresolved intents,
/// logging each blocked parent: `true` means all clear (drain, seal, and
/// retire share the gate — intents never migrate, so the step retries on
/// a later pass). Seal races log at info (`notable`); routine drains at
/// debug.
async fn merge_parents_clear(
    node: &Arc<ConsensusNode>,
    plan: &kivi_control::MergePlan,
    cause: &str,
    notable: bool,
) -> bool {
    let blocked = blocked_parents(node, &[plan.left, plan.right]).await;
    for parent in &blocked {
        if notable {
            tracing::info!(plan = plan.id.as_u64(), parent = parent.as_u64(), "{cause}");
        } else {
            tracing::debug!(plan = plan.id.as_u64(), parent = parent.as_u64(), "{cause}");
        }
    }
    blocked.is_empty()
}

/// Installs the merge final tail under fence: seed, snapshot, advance —
/// or unfence on any failure so finalizes flow again.
async fn merge_install_tail(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    plan: &kivi_control::MergePlan,
) {
    let fenced_at = std::time::Instant::now();
    if fence_tablets(node, state, &[plan.left, plan.right], true).await
        && seed_merge_target_nodes(node, state, plan).await
        && snapshot_merge_target(node, state, plan).await
    {
        tracing::info!(
            plan = plan.id.as_u64(),
            fence_ms = u64::try_from(fenced_at.elapsed().as_millis()).unwrap_or(u64::MAX),
            "merge final tail installed under fence"
        );
        advance_merge(node, plan, MergePhase::Fenced).await;
    } else {
        let _ = fence_tablets(node, state, &[plan.left, plan.right], false).await;
    }
}

/// Drives one merge plan a single idempotent step (inverse of split).
async fn drive_merge(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    plan: &kivi_control::MergePlan,
    directory: &Arc<arc_swap::ArcSwap<kivi_tablet::DirectorySnapshot>>,
) {
    if plan.phase.is_terminal() {
        return;
    }
    if !state.live_plans_for(plan.left).is_empty() || !state.live_plans_for(plan.right).is_empty() {
        tracing::debug!(
            plan = plan.id.as_u64(),
            "merge deferred: parent has a live migration (repair wins)"
        );
        return;
    }
    for parent in [plan.left, plan.right] {
        if parent_degraded(state, parent) {
            tracing::debug!(
                plan = plan.id.as_u64(),
                parent = parent.as_u64(),
                "merge deferred: parent under-replicated"
            );
            return;
        }
    }
    match plan.phase {
        MergePhase::Planned => {
            if ensure_merge_target(node, state, plan).await
                && initialize_merge_target(node, state, plan).await
            {
                advance_merge(node, plan, MergePhase::TargetAllocated).await;
            }
        }
        MergePhase::TargetAllocated => {
            if seed_merge_target_nodes(node, state, plan).await {
                advance_merge(node, plan, MergePhase::BaseSeeded).await;
            }
        }
        MergePhase::BaseSeeded => {
            // Drain before fence (see the split seal gate): unresolved
            // intents never migrate, and finalizes flow only while
            // unfenced.
            if !merge_parents_clear(
                node,
                plan,
                "merge deferred: parent holds unresolved intents",
                false,
            )
            .await
            {
                return;
            }
            merge_install_tail(node, state, plan).await;
        }
        MergePhase::Fenced => {
            // Seal gate (see the split seal gate): re-verify zero intents
            // under fence; on a race, unfence and regress rather than
            // strand transactional state in a retired lineage.
            if !merge_parents_clear(
                node,
                plan,
                "merge seal deferred: intents raced the fence; unfencing",
                true,
            )
            .await
            {
                let _ = fence_tablets(node, state, &[plan.left, plan.right], false).await;
                advance_merge(node, plan, MergePhase::BaseSeeded).await;
                return;
            }
            if publish_merge_cutover(node, state, plan, directory).await {
                advance_merge(node, plan, MergePhase::CutoverCommitted).await;
            }
        }
        MergePhase::CutoverCommitted => {
            // Retirement reclaims the parent lineages: never retire under
            // unresolved intents.
            if !merge_parents_clear(
                node,
                plan,
                "merge retire deferred: parent holds unresolved intents",
                false,
            )
            .await
            {
                return;
            }
            if retire_tablets(
                node,
                state,
                &[plan.left, plan.right],
                plan.generation.as_u64(),
            )
            .await
            {
                advance_merge(node, plan, MergePhase::ParentsRetiring).await;
            }
        }
        MergePhase::ParentsRetiring => {
            advance_merge(node, plan, MergePhase::Completed).await;
        }
        MergePhase::Completed | MergePhase::Failed => {}
    }
}

/// Replicas eligible for topology steps right now: skips repair-worthy
/// (`Unavailable`) and tombstoned (`Removed`) nodes. A killed-then-marked
/// replica stops blocking splits/merges once the detector marks it; it
/// heals through the standard learner path after it restarts (post-cutover
/// the children are ordinary tablets).
fn live_replicas(state: &ControlState, replicas: &[NodeId]) -> Vec<NodeId> {
    replicas
        .iter()
        .copied()
        .filter(|replica| {
            state.node(*replica).is_none_or(|record| {
                !matches!(record.state, NodeState::Unavailable | NodeState::Removed)
            })
        })
        .collect()
}

/// Ensures both split children on every live replica node.
async fn ensure_split_children(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    plan: &SplitPlan,
) -> bool {
    for child in [plan.left, plan.right] {
        for replica in live_replicas(state, &plan.replicas) {
            if !ensure_one(node, state, child, replica, &plan.replicas, plan.generation).await {
                return false;
            }
        }
    }
    true
}

/// Initializes both split children on the founder (lowest replica id).
/// Other replicas join through learner replication, never a second
/// initialize (that would fork history). Idempotent: re-initialize
/// answers success. Defers while the founder is repair-worthy: only the
/// deterministic founder may initialize.
async fn initialize_split_children(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    plan: &SplitPlan,
) -> bool {
    let Some(founder) = plan
        .replicas
        .iter()
        .min_by_key(|node| node.as_u64())
        .copied()
    else {
        return false;
    };
    if !live_replicas(state, std::slice::from_ref(&founder)).contains(&founder) {
        tracing::debug!(
            plan = plan.id.as_u64(),
            founder = founder.as_u64(),
            "split initialize deferred: founder repair-worthy"
        );
        return false;
    }
    for child in [plan.left, plan.right] {
        if !initialize_one(node, state, child, founder, &plan.replicas).await {
            return false;
        }
    }
    true
}

/// Seeds both children from the local parent replica on every live
/// replica node (colocated base, chunk roots referenced, never copied).
async fn seed_split_children(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    plan: &SplitPlan,
) -> bool {
    for replica in live_replicas(state, &plan.replicas) {
        for (child, left) in [(plan.left, true), (plan.right, false)] {
            if !seed_split_one(
                node,
                state,
                replica,
                plan.parent,
                child,
                plan.split_hash,
                plan.split_key.clone(),
                left,
            )
            .await
            {
                return false;
            }
        }
    }
    true
}

/// Ensures the merge target on every replica node.
async fn ensure_merge_target(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    plan: &kivi_control::MergePlan,
) -> bool {
    for replica in live_replicas(state, &plan.replicas) {
        if !ensure_one(
            node,
            state,
            plan.merged,
            replica,
            &plan.replicas,
            plan.generation,
        )
        .await
        {
            return false;
        }
    }
    true
}

/// Initializes the merge target on the founder replica (founder-only,
/// defers while the founder is repair-worthy).
async fn initialize_merge_target(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    plan: &kivi_control::MergePlan,
) -> bool {
    let Some(founder) = plan
        .replicas
        .iter()
        .min_by_key(|node| node.as_u64())
        .copied()
    else {
        return false;
    };
    if !live_replicas(state, std::slice::from_ref(&founder)).contains(&founder) {
        return false;
    }
    initialize_one(node, state, plan.merged, founder, &plan.replicas).await
}

/// Seeds the merge target from both local parents on every live replica.
async fn seed_merge_target_nodes(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    plan: &kivi_control::MergePlan,
) -> bool {
    for replica in live_replicas(state, &plan.replicas) {
        if !seed_merge_one(node, state, replica, plan.left, plan.right, plan.merged).await {
            return false;
        }
    }
    true
}

/// Ensures one tablet replica on one node (local front or admin plane).
async fn ensure_one(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    tablet: TabletId,
    target: NodeId,
    voters: &[NodeId],
    generation: PlacementVersion,
) -> bool {
    let voter_set: BTreeSet<u64> = voters.iter().map(|node| node.as_u64()).collect();
    if target == node.node() {
        return node
            .ensure_group(tablet, voter_set, generation.as_u64())
            .await
            .is_ok();
    }
    let Some(admin) = state.node(target).map(|record| record.admin) else {
        return false;
    };
    admin_post_ensure(admin, tablet, target, &voter_set, generation.as_u64()).await
}

/// Initializes one tablet group on the founder node.
async fn initialize_one(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    tablet: TabletId,
    founder: NodeId,
    voters: &[NodeId],
) -> bool {
    let voter_set: BTreeSet<u64> = voters.iter().map(|node| node.as_u64()).collect();
    let peer_addrs: BTreeMap<u64, String> = voters
        .iter()
        .filter_map(|replica| {
            state
                .node(*replica)
                .map(|record| (replica.as_u64(), record.peer.to_string()))
        })
        .collect();
    if founder == node.node() {
        return node
            .initialize_group(tablet, voter_set, peer_addrs)
            .await
            .is_ok();
    }
    let Some(admin) = state.node(founder).map(|record| record.admin) else {
        return false;
    };
    admin_post_initialize(admin, tablet, &voter_set, &peer_addrs).await
}

/// Seeds one split child on one replica node.
#[allow(clippy::too_many_arguments)]
async fn seed_split_one(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    target: NodeId,
    parent: TabletId,
    child: TabletId,
    split_hash: u128,
    split_key: Option<Vec<u8>>,
    left: bool,
) -> bool {
    if target == node.node() {
        let seeded = match split_key {
            Some(key) => node
                .seed_split_child_ordered(parent, child, key, left)
                .await
                .is_ok(),
            None => node
                .seed_split_child(parent, child, split_hash, left)
                .await
                .is_ok(),
        };
        return seeded;
    }
    let Some(admin) = state.node(target).map(|record| record.admin) else {
        return false;
    };
    admin_post_seed_split(admin, child, parent, split_hash, split_key.as_deref(), left).await
}

/// Seeds the merge target on one replica node.
async fn seed_merge_one(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    target: NodeId,
    left: TabletId,
    right: TabletId,
    merged: TabletId,
) -> bool {
    if target == node.node() {
        return node.seed_merge_target(left, right, merged).await.is_ok();
    }
    let Some(admin) = state.node(target).map(|record| record.admin) else {
        return false;
    };
    admin_post_seed_merge(admin, merged, left, right).await
}

/// Snapshots one tablet replica (local front or admin plane): seals its
/// current state into a snapshot and purges the log through the base.
/// Split/merge targets hold out-of-band seeded state that exists nowhere
/// in their Raft log — without this snapshot a hard restart before the
/// first automatic snapshot would resurrect them empty (seeded base
/// lost; only post-seed log entries would replay).
async fn snapshot_one(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    target: NodeId,
    tablet: TabletId,
) -> bool {
    if target == node.node() {
        return node
            .snapshot_and_purge(ConsensusGroupId::of_tablet(tablet))
            .await
            .is_ok();
    }
    let Some(admin) = state.node(target).map(|record| record.admin) else {
        return false;
    };
    admin_post_snapshot(admin, tablet).await
}

/// Forwards one per-tablet snapshot+purge to a replica's admin plane.
async fn admin_post_snapshot(admin: SocketAddr, tablet: TabletId) -> bool {
    admin_post(
        admin,
        "/v1/snapshot",
        &serde_json::json!({ "tablet": tablet.as_u64() }),
    )
    .await
    .is_ok_and(|value| {
        value
            .get("snapshot")
            .and_then(serde_json::Value::as_u64)
            .is_some()
    })
}

/// Snapshots both split children on every live replica (durable seeding:
/// the Fenced advance below means every replica can resurrect its child
/// from disk).
async fn snapshot_split_children(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    plan: &SplitPlan,
) -> bool {
    for replica in live_replicas(state, &plan.replicas) {
        for child in [plan.left, plan.right] {
            if !snapshot_one(node, state, replica, child).await {
                return false;
            }
        }
    }
    true
}

/// Snapshots the merge target on every live replica (durable seeding).
async fn snapshot_merge_target(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    plan: &kivi_control::MergePlan,
) -> bool {
    for replica in live_replicas(state, &plan.replicas) {
        if !snapshot_one(node, state, replica, plan.merged).await {
            return false;
        }
    }
    true
}

/// Sets or clears the cutover fence on every live replica of `tablets`.
/// Repair-worthy hosts are skipped (they stop blocking the cutover once
/// marked; they heal as ordinary tablets afterwards). Returns false when
/// any live replica refuses (next pass retries).
async fn fence_tablets(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    tablets: &[TabletId],
    fenced: bool,
) -> bool {
    let mut hosts: BTreeSet<NodeId> = BTreeSet::new();
    for tablet in tablets {
        if let Some(desired) = state.desired(*tablet) {
            hosts.extend(live_replicas(state, &desired.replicas));
        }
    }
    for host in &hosts {
        for tablet in tablets {
            let ok = if *host == node.node() {
                node.set_tablet_fenced(*tablet, fenced).await.is_ok()
            } else {
                let Some(admin) = state.node(*host).map(|record| record.admin) else {
                    return false;
                };
                admin_post_fence(admin, *tablet, fenced).await
            };
            if !ok {
                return false;
            }
        }
    }
    true
}

/// Retires every tablet replica in `tablets` on all of its live desired
/// hosts (repair-worthy hosts are skipped; see `live_replicas`).
async fn retire_tablets(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    tablets: &[TabletId],
    generation: u64,
) -> bool {
    for tablet in tablets {
        let hosts: Vec<NodeId> = state
            .desired(*tablet)
            .map(|desired| live_replicas(state, &desired.replicas))
            .unwrap_or_default();
        for host in hosts {
            let ok = if host == node.node() {
                node.retire_group(*tablet, generation).await.is_ok()
            } else {
                let Some(admin) = state.node(host).map(|record| record.admin) else {
                    return false;
                };
                admin_post_retire(admin, *tablet, host, generation).await
            };
            if !ok {
                return false;
            }
        }
    }
    true
}

/// Publishes the split directory cutover locally and on every serving
/// node (idempotent, one validated atomic transition per node).
async fn publish_split_cutover(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    plan: &SplitPlan,
    directory: &Arc<arc_swap::ArcSwap<kivi_tablet::DirectorySnapshot>>,
) -> bool {
    let body = serde_json::json!({
        "kind": "split",
        "parent": plan.parent.as_u64(),
        "left": plan.left.as_u64(),
        "right": plan.right.as_u64(),
        "split_hash": plan.split_hash.to_string(),
        "split_key": plan.split_key,
    });
    // Local publish first (the control leader routes correctly even
    // before peers converge).
    if !publish_cutover_local(directory, &body) {
        return false;
    }
    for record in state.nodes() {
        if record.node == node.node() {
            continue;
        }
        if !matches!(
            record.state,
            NodeState::Active | NodeState::Suspect | NodeState::Draining
        ) {
            continue;
        }
        if !admin_post_cutover(record.admin, &body).await {
            return false;
        }
    }
    true
}

/// Publishes the merge directory cutover locally and on every serving node.
async fn publish_merge_cutover(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    plan: &kivi_control::MergePlan,
    directory: &Arc<arc_swap::ArcSwap<kivi_tablet::DirectorySnapshot>>,
) -> bool {
    let body = serde_json::json!({
        "kind": "merge",
        "left": plan.left.as_u64(),
        "right": plan.right.as_u64(),
        "merged": plan.merged.as_u64(),
    });
    if !publish_cutover_local(directory, &body) {
        return false;
    }
    for record in state.nodes() {
        if record.node == node.node() {
            continue;
        }
        if !matches!(
            record.state,
            NodeState::Active | NodeState::Suspect | NodeState::Draining
        ) {
            continue;
        }
        if !admin_post_cutover(record.admin, &body).await {
            return false;
        }
    }
    true
}

/// Applies a cutover body to the local directory snapshot.
fn publish_cutover_local(
    directory: &Arc<arc_swap::ArcSwap<kivi_tablet::DirectorySnapshot>>,
    body: &serde_json::Value,
) -> bool {
    let current = directory.load();
    let next: Result<kivi_tablet::DirectorySnapshot, String> = (|| {
        let kind = body
            .get("kind")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        match kind {
            "split" => {
                let parent = TabletId::from_u64(
                    body.get("parent")
                        .and_then(serde_json::Value::as_u64)
                        .ok_or("missing parent")?,
                );
                let left = TabletId::from_u64(
                    body.get("left")
                        .and_then(serde_json::Value::as_u64)
                        .ok_or("missing left")?,
                );
                let right = TabletId::from_u64(
                    body.get("right")
                        .and_then(serde_json::Value::as_u64)
                        .ok_or("missing right")?,
                );
                let split_hash = body
                    .get("split_hash")
                    .and_then(|value| {
                        value
                            .as_str()
                            .and_then(|text| text.parse::<u128>().ok())
                            .or_else(|| value.as_u64().map(u128::from))
                    })
                    .ok_or("missing split_hash")?;
                // Ordered splits carry `split_key` as a byte array;
                // absence means a hash split (key must be interior —
                // validated by the range operation, not here).
                let split_key: Option<Vec<u8>> = body.get("split_key").and_then(|value| {
                    value.as_array().map(|bytes| {
                        bytes
                            .iter()
                            .filter_map(serde_json::Value::as_u64)
                            .filter_map(|byte| u8::try_from(byte).ok())
                            .collect()
                    })
                });
                crate::cluster::apply_cutover_split_for_reconciler(
                    &current,
                    parent,
                    left,
                    right,
                    split_hash,
                    split_key.as_deref(),
                )
            }
            "merge" => {
                let left = TabletId::from_u64(
                    body.get("left")
                        .and_then(serde_json::Value::as_u64)
                        .ok_or("missing left")?,
                );
                let right = TabletId::from_u64(
                    body.get("right")
                        .and_then(serde_json::Value::as_u64)
                        .ok_or("missing right")?,
                );
                let merged = TabletId::from_u64(
                    body.get("merged")
                        .and_then(serde_json::Value::as_u64)
                        .ok_or("missing merged")?,
                );
                crate::cluster::apply_cutover_merge_for_reconciler(&current, left, right, merged)
            }
            _ => Err("unknown cutover kind".to_owned()),
        }
    })();
    match next {
        Ok(next) => {
            if let Err(reason) = next.validate() {
                tracing::warn!(%reason, "cutover snapshot invalid");
                return false;
            }
            directory.store(Arc::new(next));
            true
        }
        Err(reason) => {
            tracing::debug!(%reason, "cutover deferred");
            false
        }
    }
}

/// Automatic split/merge planning (conservative, hysteresis-guarded).
/// Splits fire when a healthy tablet exceeds `split_object_threshold`;
/// merges fire when BOTH adjacent siblings sit below
/// `merge_object_threshold` (`split >> merge`, no flapping). Degraded
/// tablets never split or merge (repair wins). At most one live split
/// and one live merge exist cluster-wide.
async fn drive_topology_auto(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    policy: &ReconcilePolicy,
    directory: &Arc<arc_swap::ArcSwap<kivi_tablet::DirectorySnapshot>>,
) {
    if policy.auto_split {
        drive_auto_split(node, state, policy, directory).await;
    }
    if policy.auto_merge {
        let fresh = node.control_state().await.unwrap_or_else(|| state.clone());
        drive_auto_merge(node, &fresh, policy, directory).await;
    }
}

/// Plans at most one automatic split per pass for the largest eligible
/// tablet above threshold: hash tablets split by object count at the hash
/// midpoint; ordered tablets split by logical bytes at the approximate
/// median key (leader-local, balancing bytes rather than object count).
/// One straight-line pass over placements; splitting it would obscure the
/// shared eligibility gating.
#[allow(clippy::too_many_lines)]
async fn drive_auto_split(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    policy: &ReconcilePolicy,
    directory: &Arc<arc_swap::ArcSwap<kivi_tablet::DirectorySnapshot>>,
) {
    let live_splits = state
        .splits()
        .filter(|plan| !plan.phase.is_terminal())
        .count();
    if live_splits >= policy.scheduler.max_live_splits {
        return;
    }
    let snapshot = directory.load();
    // Best hash candidate by count, best ordered candidate by bytes.
    let mut best_hash: Option<(usize, TabletId)> = None;
    let mut best_ordered: Option<(u64, TabletId)> = None;
    for desired in state.placements() {
        let tablet = desired.tablet;
        if state.has_live_topology(tablet) {
            continue;
        }
        if is_degraded(state, &desired.replicas) {
            continue;
        }
        let Some(descriptor) = snapshot.get(tablet) else {
            continue;
        };
        if descriptor.state() != kivi_tablet::TabletState::Active {
            continue;
        }
        // Tablets with unresolved intents do not split (resolve first,
        // then cut over — see the intent migration rule).
        if node
            .tablet_intent_count(tablet)
            .await
            .is_some_and(|count| count > 0)
        {
            continue;
        }
        match descriptor.range() {
            kivi_tablet::PartitionRange::Hash(prefix) => {
                if prefix.prefix_len() >= kivi_tablet::range::MAX_PREFIX_LEN {
                    continue;
                }
                let count = node.tablet_object_count(tablet).await.unwrap_or(0);
                if count < policy.split_object_threshold {
                    continue;
                }
                if best_hash.is_none_or(|(best_count, _)| count > best_count) {
                    best_hash = Some((count, tablet));
                }
            }
            kivi_tablet::PartitionRange::Ordered(_) => {
                let now = super::cluster::wall_now();
                let bytes = node
                    .tablet_stats(tablet, now)
                    .await
                    .map_or(0, |stats| stats.logical_value_bytes);
                if bytes < policy.split_bytes_threshold {
                    continue;
                }
                if best_ordered.is_none_or(|(best_bytes, _)| bytes > best_bytes) {
                    best_ordered = Some((bytes, tablet));
                }
            }
        }
    }
    if let Some((count, tablet)) = best_hash {
        tracing::info!(
            tablet = tablet.as_u64(),
            objects = count,
            "automatic hash split triggered"
        );
        // Derive the midpoint boundary and fresh children from the same
        // directory view + persisted allocator the manual path uses.
        let state = node.control_state().await;
        let snapshot = directory.load();
        let Some((split_hash, left, right)) = state.as_ref().and_then(|state| {
            snapshot
                .get(tablet)
                .and_then(|descriptor| match descriptor.range() {
                    kivi_tablet::PartitionRange::Hash(prefix) => {
                        prefix.split_midpoint().ok().map(|(_, _, split_hash)| {
                            let fresh = state.fresh_tablet_ids(2);
                            (
                                split_hash,
                                fresh
                                    .first()
                                    .copied()
                                    .unwrap_or(TabletId::from_u64(u64::MAX)),
                                fresh
                                    .get(1)
                                    .copied()
                                    .unwrap_or(TabletId::from_u64(u64::MAX)),
                            )
                        })
                    }
                    kivi_tablet::PartitionRange::Ordered(_) => None,
                })
        }) else {
            return;
        };
        let _ = create_split_plan_at(node, tablet, split_hash, left, right).await;
    }
    if let Some((bytes, tablet)) = best_ordered {
        // Median-key pivot from the leader-local ordered index.
        let Some(split_key) = node.tablet_median_key(tablet).await else {
            return;
        };
        tracing::info!(
            tablet = tablet.as_u64(),
            bytes = bytes,
            "automatic ordered split triggered"
        );
        let state = node.control_state().await;
        let snapshot = directory.load();
        let Some((left, right)) = state.as_ref().and_then(|state| {
            snapshot
                .get(tablet)
                .and_then(|descriptor| match descriptor.range() {
                    kivi_tablet::PartitionRange::Ordered(range) => {
                        range.split_at(&split_key).ok().map(|_| {
                            let fresh = state.fresh_tablet_ids(2);
                            (
                                fresh
                                    .first()
                                    .copied()
                                    .unwrap_or(TabletId::from_u64(u64::MAX)),
                                fresh
                                    .get(1)
                                    .copied()
                                    .unwrap_or(TabletId::from_u64(u64::MAX)),
                            )
                        })
                    }
                    kivi_tablet::PartitionRange::Hash(_) => None,
                })
        }) else {
            return;
        };
        let _ = create_split_plan_full(node, tablet, 0, Some(split_key), left, right).await;
    }
}

/// Plans at most one automatic merge per pass for the smallest eligible
/// adjacent sibling pair below threshold (hash siblings share a parent
/// prefix; ordered siblings satisfy `left.end == right.start`). One
/// straight-line pass per layout; splitting further would obscure the
/// shared eligibility gating.
#[allow(clippy::too_many_lines)]
async fn drive_auto_merge(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    policy: &ReconcilePolicy,
    directory: &Arc<arc_swap::ArcSwap<kivi_tablet::DirectorySnapshot>>,
) {
    let live_merges = state
        .merges()
        .filter(|plan| !plan.phase.is_terminal())
        .count();
    if live_merges >= policy.scheduler.max_live_merges {
        return;
    }
    let snapshot = directory.load();
    // Adjacent siblings share a parent prefix of len-1; collect active
    // hash tablets sorted by (len desc, bits) so siblings neighbor.
    let mut tablets: Vec<(u128, u8, TabletId)> = Vec::new();
    for desired in state.placements() {
        let tablet = desired.tablet;
        if state.has_live_topology(tablet) {
            continue;
        }
        if is_degraded(state, &desired.replicas) {
            continue;
        }
        let Some(descriptor) = snapshot.get(tablet) else {
            continue;
        };
        if descriptor.state() != kivi_tablet::TabletState::Active {
            continue;
        }
        let kivi_tablet::PartitionRange::Hash(prefix) = descriptor.range().clone() else {
            continue;
        };
        tablets.push((prefix.bits(), prefix.prefix_len(), tablet));
    }
    tablets.sort();
    for window in tablets.windows(2) {
        let [(left_bits, left_len, left), (right_bits, right_len, right)] = window else {
            continue;
        };
        if left_len != right_len || *left_len == 0 {
            continue;
        }
        let half = 1u128 << (128 - left_len);
        if left_bits | half != *right_bits || left_bits & half != 0 {
            continue;
        }
        let left_count = node.tablet_object_count(*left).await.unwrap_or(usize::MAX);
        let right_count = node.tablet_object_count(*right).await.unwrap_or(usize::MAX);
        if left_count > policy.merge_object_threshold || right_count > policy.merge_object_threshold
        {
            continue;
        }
        tracing::info!(
            left = left.as_u64(),
            right = right.as_u64(),
            "automatic merge triggered"
        );
        let _ = create_merge_plan(node, *left, *right).await;
        return;
    }
    // Ordered siblings, sorted by range start so siblings neighbor.
    let mut ordered: Vec<(Vec<u8>, Option<Vec<u8>>, TabletId)> = Vec::new();
    for desired in state.placements() {
        let tablet = desired.tablet;
        if state.has_live_topology(tablet) {
            continue;
        }
        if is_degraded(state, &desired.replicas) {
            continue;
        }
        let Some(descriptor) = snapshot.get(tablet) else {
            continue;
        };
        if descriptor.state() != kivi_tablet::TabletState::Active {
            continue;
        }
        let kivi_tablet::PartitionRange::Ordered(range) = descriptor.range().clone() else {
            continue;
        };
        // Tablets with unresolved intents do not merge (resolve first).
        if node
            .tablet_intent_count(tablet)
            .await
            .is_some_and(|count| count > 0)
        {
            continue;
        }
        ordered.push((
            range.start().to_vec(),
            range.end().map(<[u8]>::to_vec),
            tablet,
        ));
    }
    ordered.sort();
    for window in ordered.windows(2) {
        let [(_, left_end, left), (right_start, _, right)] = window else {
            continue;
        };
        if left_end.as_deref() != Some(right_start.as_slice()) {
            continue;
        }
        let left_count = node.tablet_object_count(*left).await.unwrap_or(usize::MAX);
        let right_count = node.tablet_object_count(*right).await.unwrap_or(usize::MAX);
        if left_count > policy.merge_object_threshold || right_count > policy.merge_object_threshold
        {
            continue;
        }
        tracing::info!(
            left = left.as_u64(),
            right = right.as_u64(),
            "automatic ordered merge triggered"
        );
        let _ = create_merge_plan(node, *left, *right).await;
        return;
    }
}

/// Whether any desired replica is repair-worthy (split/merge ineligible).
fn is_degraded(state: &ControlState, replicas: &[NodeId]) -> bool {
    replicas.iter().any(|replica| {
        state.node(*replica).is_none_or(|record| {
            matches!(
                record.state,
                NodeState::Unavailable | NodeState::Removed | NodeState::Joining
            )
        })
    })
}

/// Creates a split plan for `parent` at an explicit midpoint boundary
/// with explicit children: midpoint boundary, fresh child ids from the
/// persisted allocator, replicas inherited from the parent. One shared
/// pathway for manual splits (admin/CLI, which reads the directory) and
/// automatic splits (the reconciler, which owns a directory view).
pub async fn create_split_plan_at(
    node: &Arc<ConsensusNode>,
    parent: TabletId,
    split_hash: u128,
    left: TabletId,
    right: TabletId,
) -> Result<kivi_control::SplitPlanId, String> {
    create_split_plan_full(node, parent, split_hash, None, left, right).await
}

/// Creates a split plan with an explicit boundary: `split_key` set for
/// ordered parents (real-key boundary), `split_hash` for hash parents
/// (midpoint). Exactly one boundary style applies per plan.
pub async fn create_split_plan_full(
    node: &Arc<ConsensusNode>,
    parent: TabletId,
    split_hash: u128,
    split_key: Option<Vec<u8>>,
    left: TabletId,
    right: TabletId,
) -> Result<kivi_control::SplitPlanId, String> {
    let state = node
        .control_state()
        .await
        .ok_or_else(|| "no local control image".to_owned())?;
    if state.has_live_topology(parent)
        || state.has_live_topology(left)
        || state.has_live_topology(right)
    {
        return Err(format!(
            "tablet {} (or its children) already has a live topology plan",
            parent.as_u64()
        ));
    }
    let desired = state
        .desired(parent)
        .ok_or_else(|| format!("tablet {} has no desired placement", parent.as_u64()))?
        .clone();
    if desired.replicas.iter().any(|replica| {
        state.node(*replica).is_some_and(|record| {
            matches!(record.state, NodeState::Unavailable | NodeState::Removed)
        })
    }) {
        return Err("parent under-replicated; repair first".to_owned());
    }
    let generation = state
        .placement_version()
        .next()
        .map_err(|error| error.to_string())?;
    let plan = SplitPlan::new(
        kivi_control::SplitPlanId::from_u64(state.next_plan_id().as_u64()),
        parent,
        split_hash,
        split_key,
        left,
        right,
        desired.replicas.clone(),
        generation,
    )
    .map_err(|error| error.to_string())?;
    let id = plan.id;
    node.propose_control(ControlMutation::CreateSplit { plan })
        .await?;
    tracing::info!(
        plan = id.as_u64(),
        parent = parent.as_u64(),
        left = left.as_u64(),
        right = right.as_u64(),
        "split plan created"
    );
    Ok(id)
}

/// Registers a namespace in replicated control state (idempotent for
/// identical records; conflicting layouts are rejected).
///
/// # Errors
///
/// Returns a human-readable reason when control is unreachable.
pub async fn register_namespace(
    node: &Arc<ConsensusNode>,
    id: kivi_types::NamespaceId,
    layout: kivi_control::CatalogLayout,
) -> Result<(), String> {
    node.propose_control(ControlMutation::RegisterNamespace {
        record: kivi_control::NamespaceRecord { id, layout },
    })
    .await
    .map(|_| ())
}

/// Creates a secondary index definition (starts `Building`).
///
/// # Errors
///
/// Returns a human-readable reason when control is unreachable or the
/// allocated id collides (retry: the allocator re-reads).
pub async fn create_index(
    node: &Arc<ConsensusNode>,
    primary: kivi_types::NamespaceId,
    kind: kivi_control::CatalogIndexKind,
) -> Result<u64, String> {
    let state = node
        .control_state()
        .await
        .ok_or_else(|| "no local control image".to_owned())?;
    if state.namespace(primary).is_none() {
        return Err(format!("unknown namespace {}", primary.as_u64()));
    }
    let id = state.indexes().map(|record| record.id).max().unwrap_or(0) + 1;
    node.propose_control(ControlMutation::CreateIndex {
        record: kivi_control::IndexRecord {
            id,
            primary,
            kind,
            state: kivi_control::CatalogIndexState::Building,
        },
    })
    .await?;
    Ok(id)
}

/// Advances an index lifecycle state.
///
/// # Errors
///
/// Returns a human-readable reason when control is unreachable or the
/// index is unknown.
pub async fn advance_index(
    node: &Arc<ConsensusNode>,
    id: u64,
    state: kivi_control::CatalogIndexState,
) -> Result<(), String> {
    node.propose_control(ControlMutation::AdvanceIndex {
        id,
        state: state.encode_byte(),
    })
    .await
    .map(|_| ())
}

/// Removes a `Dropping` index definition.
///
/// # Errors
///
/// Returns a human-readable reason when control is unreachable, the
/// index is unknown, or it is not `Dropping`.
pub async fn drop_index(node: &Arc<ConsensusNode>, id: u64) -> Result<(), String> {
    node.propose_control(ControlMutation::RemoveIndex { id })
        .await
        .map(|_| ())
}

/// Creates a merge plan for adjacent `left` + `right` into a fresh merged
/// tablet: replicas default to the left parent's desired voters (callers
/// may rebalance the merged tablet afterwards as an ordinary tablet).
/// Requires compatible placement (equal desired voter sets); otherwise
/// the operator must first migrate one parent onto the other's nodes.
pub async fn create_merge_plan(
    node: &Arc<ConsensusNode>,
    left: TabletId,
    right: TabletId,
) -> Result<kivi_control::MergePlanId, String> {
    let state = node
        .control_state()
        .await
        .ok_or_else(|| "no local control image".to_owned())?;
    if state.has_live_topology(left) || state.has_live_topology(right) {
        return Err("a parent already has a live topology plan".to_owned());
    }
    let left_desired = state
        .desired(left)
        .ok_or_else(|| format!("tablet {} has no desired placement", left.as_u64()))?
        .clone();
    let right_desired = state
        .desired(right)
        .ok_or_else(|| format!("tablet {} has no desired placement", right.as_u64()))?
        .clone();
    if left_desired.replicas != right_desired.replicas {
        return Err(
            "parents have different desired voters; migrate to a common set first".to_owned(),
        );
    }
    for replica in &left_desired.replicas {
        if state.node(*replica).is_some_and(|record| {
            matches!(record.state, NodeState::Unavailable | NodeState::Removed)
        }) {
            return Err("parent under-replicated; repair first".to_owned());
        }
    }
    let generation = state
        .placement_version()
        .next()
        .map_err(|error| error.to_string())?;
    let merged = state
        .fresh_tablet_ids(1)
        .into_iter()
        .next()
        .unwrap_or(TabletId::from_u64(u64::MAX));
    let plan = kivi_control::MergePlan::new(
        kivi_control::MergePlanId::from_u64(state.next_plan_id().as_u64()),
        left,
        right,
        merged,
        left_desired.replicas.clone(),
        generation,
    )
    .map_err(|error| error.to_string())?;
    let id = plan.id;
    node.propose_control(ControlMutation::CreateMerge { plan })
        .await?;
    tracing::info!(
        plan = id.as_u64(),
        left = left.as_u64(),
        right = right.as_u64(),
        merged = merged.as_u64(),
        "merge plan created"
    );
    Ok(id)
}

/// Maximum retained terminal plans per kind (migrations, splits,
/// merges). Bounds control-state and snapshot growth on long-running
/// clusters with frequent topology changes; live plans are never
/// touched.
const MAX_RETAINED_TERMINAL_PLANS: usize = 128;

/// Removes terminal plans beyond the retention cap (oldest first).
/// Idempotent: `Remove*` only accepts terminal plans, and a concurrent
/// pass proposing the same removal is a safe duplicate.
async fn prune_terminal_plans(node: &Arc<ConsensusNode>, state: &ControlState) {
    let mut migrations: Vec<u64> = state
        .migrations()
        .filter(|plan| plan.phase.is_terminal())
        .map(|plan| plan.id.as_u64())
        .collect();
    migrations.sort_unstable();
    if let Some(drop) = migrations.len().checked_sub(MAX_RETAINED_TERMINAL_PLANS)
        && drop > 0
    {
        for id in migrations.into_iter().take(drop) {
            let mutation = ControlMutation::RemoveMigration {
                plan: MigrationPlanId::from_u64(id).into(),
            };
            if node.propose_control(mutation).await.is_err() {
                return;
            }
        }
    }
    let mut splits: Vec<u64> = state
        .splits()
        .filter(|plan| plan.phase.is_terminal())
        .map(|plan| plan.id.as_u64())
        .collect();
    splits.sort_unstable();
    if let Some(drop) = splits.len().checked_sub(MAX_RETAINED_TERMINAL_PLANS)
        && drop > 0
    {
        for id in splits.into_iter().take(drop) {
            let mutation = ControlMutation::RemoveSplit {
                plan: kivi_control::SplitPlanId::from_u64(id).into(),
            };
            if node.propose_control(mutation).await.is_err() {
                return;
            }
        }
    }
    let mut merges: Vec<u64> = state
        .merges()
        .filter(|plan| plan.phase.is_terminal())
        .map(|plan| plan.id.as_u64())
        .collect();
    merges.sort_unstable();
    if let Some(drop) = merges.len().checked_sub(MAX_RETAINED_TERMINAL_PLANS)
        && drop > 0
    {
        for id in merges.into_iter().take(drop) {
            let mutation = ControlMutation::RemoveMerge {
                plan: kivi_control::MergePlanId::from_u64(id).into(),
            };
            if node.propose_control(mutation).await.is_err() {
                return;
            }
        }
    }
}

/// Creates bounded top-up migration rounds for draining nodes that
/// still desire tablets, honoring the global live-plan cap.
async fn topup_drains(node: &Arc<ConsensusNode>, state: &ControlState, policy: &ReconcilePolicy) {
    let live = state
        .migrations()
        .filter(|plan| !plan.phase.is_terminal())
        .count();
    if live >= policy.max_live_plans {
        return;
    }
    for record in state.nodes() {
        if record.state != NodeState::Draining {
            continue;
        }
        if state.tablets_desiring(record.node).is_empty() {
            continue;
        }
        let intents = drain_intents(state, record.node, policy);
        if intents.is_empty() {
            continue;
        }
        // Cap this refill to the remaining global budget.
        let live_now = node.control_state().await.map_or(live, |fresh| {
            fresh
                .migrations()
                .filter(|plan| !plan.phase.is_terminal())
                .count()
        });
        if live_now >= policy.max_live_plans {
            return;
        }
        let budget = policy.max_live_plans - live_now;
        let intents: Vec<MigrationIntent> = intents.into_iter().take(budget.max(1)).collect();
        // Re-read state for plan creation (generations must be fresh).
        let Some(fresh) = node.control_state().await else {
            return;
        };
        match create_plans(node, &fresh, &intents).await {
            Ok(ids) => {
                tracing::info!(
                    node = record.node.as_u64(),
                    plans = ids.len(),
                    "drain top-up round created"
                );
            }
            Err(reason) => {
                tracing::debug!(node = record.node.as_u64(), %reason, "drain top-up deferred");
            }
        }
        return;
    }
}

/// Proposes `add_learner` on the control group for every `Active`
/// non-voter with a known peer endpoint (data nodes observe placement
/// for routing without voting). Idempotent: existing learners re-add
/// harmlessly; failures retry next pass.
async fn ensure_control_learners(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    voters: &BTreeSet<u64>,
) {
    let observed = node
        .observe_membership(control_tablet())
        .await
        .map(|observed| observed.learners)
        .unwrap_or_default();
    for record in state.nodes() {
        if record.state != NodeState::Active || voters.contains(&record.node.as_u64()) {
            continue;
        }
        if observed.contains(&record.node.as_u64()) {
            continue;
        }
        let addr = record.peer.to_string();
        if let Err(reason) = node
            .add_learner(control_tablet(), record.node, addr, false)
            .await
        {
            tracing::debug!(node = record.node.as_u64(), %reason, "control learner add deferred");
        }
    }
}

/// Current control voter set from live membership (empty when the
/// local node hosts no control replica). Used to keep data nodes as
/// learners rather than voters.
async fn control_voters_of(node: &Arc<ConsensusNode>) -> BTreeSet<u64> {
    node.observe_membership(control_tablet())
        .await
        .map(|observed| observed.voters)
        .unwrap_or_default()
}

/// Drives one migration plan a single idempotent step.
///
/// Learner registration never blocks a pass: `add_learner` runs
/// non-blocking and catch-up is polled through replication lag
/// ([`caught_up`](kivi_consensus::caught_up), the same threshold the
/// blocking learner waits on). A pass that cannot complete its step
/// simply retries next pass.
async fn drive_plan(node: &Arc<ConsensusNode>, state: &ControlState, plan: &MigrationPlan) {
    let Some(observed) = observe_tablet(node, state, plan.tablet).await else {
        tracing::debug!(
            tablet = plan.tablet.as_u64(),
            "no observation; retry next pass"
        );
        return;
    };
    // Replication lag is only authoritative on the tablet leader
    // (followers report unknown): refresh the view from the leader
    // before any lag-gated step, so catch-up is polled against the
    // leader's match indexes rather than an empty follower view.
    let observed = refresh_from_leader(node, state, &observed).await;
    match kivi_consensus::next_step(plan, &observed) {
        kivi_consensus::ReconcileStep::EnsureTargetReplica { tablet, target } => {
            if exec_ensure(node, state, tablet, target, plan).await {
                advance(node, plan, MigrationPhase::TargetStarting).await;
            }
        }
        kivi_consensus::ReconcileStep::AddLearner { tablet, target } => {
            if exec_learner(node, state, &observed, tablet, target).await {
                advance(node, plan, MigrationPhase::LearnerAdded).await;
            }
        }
        kivi_consensus::ReconcileStep::WaitCatchUp { tablet, target } => {
            // Re-assert learner registration ONLY when absent: every
            // `add_learner` commits a log entry, so re-issuing it each
            // pass would flood the log with no-op membership entries
            // while catch-up (minutes for large tablets) runs.
            if !observed.is_learner(target) && !observed.is_voter(target) {
                let _ = exec_learner(node, state, &observed, tablet, target).await;
            }
            if kivi_consensus::caught_up(&observed, target) {
                advance(node, plan, MigrationPhase::Ready).await;
            } else {
                tracing::debug!(
                    tablet = tablet.as_u64(),
                    target = target.as_u64(),
                    lag = observed.lag_of(target),
                    "learner catching up"
                );
            }
        }
        kivi_consensus::ReconcileStep::ChangeMembership { tablet, desired } => {
            // Issue the membership change until actual == desired: joint
            // configs need their uniform successor re-proposed
            // explicitly, and residue sets (source lingering) need the
            // replacement re-proposed. A converged uniform needs no new
            // entry, so re-issues stop the pass convergence lands (a
            // duplicate or two may append while the commit propagates;
            // identical uniform entries are harmless).
            let source = plan.from;
            let target = plan.to;
            let converged =
                !observed.is_voter(source) && observed.is_voter(target) && !observed.is_joint;
            if !converged && exec_members(node, state, &observed, tablet, &desired).await {
                advance(node, plan, MigrationPhase::MembershipChanging).await;
            }
        }
        kivi_consensus::ReconcileStep::WaitJoint { tablet } => {
            // The joint entry committed but the uniform successor has
            // not landed (leadership flap between OpenRaft's internal
            // two steps, or a lost second step): re-proposing the same
            // desired set is safe (the joint is committed, so
            // propose-after-commit holds) and converges directly to
            // the uniform config. Never wait passively. Repetitive
            // per-pass state (not a transition): `debug`, not `info`.
            tracing::debug!(
                plan = plan.id.as_u64(),
                tablet = tablet.as_u64(),
                "joint pending; re-proposing uniform membership"
            );
            let _ = exec_members(node, state, &observed, tablet, &plan.desired_voters).await;
        }
        kivi_consensus::ReconcileStep::TransferLeadership { tablet, to } => {
            exec_transfer(node, state, &observed, tablet, to).await;
        }
        kivi_consensus::ReconcileStep::RetireSource { tablet, source } => {
            // Retirement is durable once executed (local replica
            // unregistered plus tombstone, awaited; a remote 200
            // likewise answers only after completion), and this step
            // only fires once membership converged — so success
            // completes the plan. A repair-worthy dead source
            // (`Unavailable`/`Removed`) never answers again: skip its
            // retirement RPC and complete — the open-time desired oracle
            // tombstones its stale replica when (if) it returns, so no
            // zombie replica can resurrect.
            let source_gone = state.node(source).is_some_and(|record| {
                matches!(record.state, NodeState::Unavailable | NodeState::Removed)
            });
            if source_gone {
                tracing::info!(
                    plan = plan.id.as_u64(),
                    tablet = tablet.as_u64(),
                    source = source.as_u64(),
                    "source repair-worthy dead; completing without retirement RPC"
                );
                advance(node, plan, MigrationPhase::Completed).await;
            } else if exec_retire(node, state, tablet, source, plan).await {
                advance(node, plan, MigrationPhase::Completed).await;
            }
        }
        kivi_consensus::ReconcileStep::AdvancePlan { plan, phase } => {
            advance_by_id(node, plan, phase).await;
        }
        kivi_consensus::ReconcileStep::Done => {}
    }
}

/// Creates the target replica on the joining node: directly when it is
/// us, else through its admin plane.
async fn exec_ensure(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    tablet: TabletId,
    target: NodeId,
    plan: &MigrationPlan,
) -> bool {
    let voters: BTreeSet<u64> = plan
        .desired_voters
        .iter()
        .map(|node| node.as_u64())
        .collect();
    if target == node.node() {
        return node
            .ensure_group(tablet, voters, plan.generation.as_u64())
            .await
            .map_err(|reason| {
                tracing::debug!(tablet = tablet.as_u64(), %reason, "ensure deferred");
                reason
            })
            .is_ok();
    }
    match step_target(node, state, target) {
        Some(StepTarget::Local(handle)) => handle
            .ensure_group(tablet, voters, plan.generation.as_u64())
            .await
            .is_ok(),
        Some(StepTarget::Remote(admin)) => {
            admin_post_ensure(admin, tablet, target, &voters, plan.generation.as_u64()).await
        }
        None => false,
    }
}

/// Retires the source replica on the leaving node. Returns whether
/// the retirement executed (local success or remote `200`).
async fn exec_retire(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    tablet: TabletId,
    source: NodeId,
    plan: &MigrationPlan,
) -> bool {
    if source == node.node() {
        return node
            .retire_group(tablet, plan.generation.as_u64())
            .await
            .map_err(|reason| {
                tracing::debug!(tablet = tablet.as_u64(), %reason, "retire deferred");
                reason
            })
            .is_ok();
    }
    match step_target(node, state, source) {
        Some(StepTarget::Local(handle)) => handle
            .retire_group(tablet, plan.generation.as_u64())
            .await
            .is_ok(),
        Some(StepTarget::Remote(admin)) => {
            admin_post_retire(admin, tablet, source, plan.generation.as_u64()).await
        }
        None => false,
    }
}

/// Advances one plan's persisted phase (generation-fenced).
async fn advance(node: &Arc<ConsensusNode>, plan: &MigrationPlan, phase: MigrationPhase) {
    tracing::info!(
        node_id = node.node().as_u64(),
        plan_id = plan.id.as_u64(),
        tablet_id = plan.tablet.as_u64(),
        from = plan.from.as_u64(),
        to = plan.to.as_u64(),
        ?phase,
        "migration plan advanced"
    );
    advance_by_id(node, plan.id, phase).await;
}

/// Advances one plan's persisted phase by id.
async fn advance_by_id(node: &Arc<ConsensusNode>, id: MigrationPlanId, phase: MigrationPhase) {
    // The generation rides from the stored plan; a stale advance is a
    // safe rejection the next pass re-derives.
    let generation = node
        .control_state()
        .await
        .and_then(|state| state.migration(id).map(|plan| plan.generation))
        .unwrap_or(PlacementVersion::INITIAL);
    let mutation = ControlMutation::AdvanceMigration {
        plan: id.into(),
        phase,
        generation,
    };
    if let Err(reason) = node.propose_control(mutation).await {
        tracing::debug!(plan = id.as_u64(), %reason, "plan advance deferred");
    }
}

/// Observes one tablet's actual membership through
/// [`kivi_consensus::select_authoritative`]: a member-local view wins,
/// else the first answering member peer wins, else the non-member local
/// placeholder is a last resort.
///
/// See [`kivi_consensus::ObservationAuthority`]: an embryonic local group
/// (empty membership, no leader, not yet voter/learner) must never shadow
/// an authoritative member observation. All reconciler paths (migration,
/// repair, split-child, merge-target creation) share this function.
async fn observe_tablet(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    tablet: TabletId,
) -> Option<kivi_consensus::ObservedMembership> {
    let local = node.observe_membership(tablet).await.ok();
    // Fast path: authoritative member-local view, no network.
    if matches!(
        kivi_consensus::observation_authority(local.as_ref(), node.node()),
        Some(kivi_consensus::ObservationAuthority::Member)
    ) {
        return local;
    }
    // Remote observation through hosting peers' admin planes.
    let mut candidates: Vec<NodeId> = state
        .desired(tablet)
        .map(|desired| desired.replicas.clone())
        .unwrap_or_default();
    for plan in state.live_plans_for(tablet) {
        if !candidates.contains(&plan.from) {
            candidates.push(plan.from);
        }
        if !candidates.contains(&plan.to) {
            candidates.push(plan.to);
        }
    }
    // Split/merge/repair plans name additional hosts; consult them so
    // child/target groups observe through their real members, never an
    // embryonic local placeholder (see `ObservationAuthority`).
    for plan in state.splits_for(tablet) {
        for host in plan.hosts() {
            if !candidates.contains(&host) {
                candidates.push(host);
            }
        }
    }
    for plan in state.merges_for(tablet) {
        for host in plan.hosts() {
            if !candidates.contains(&host) {
                candidates.push(host);
            }
        }
    }
    for candidate in candidates {
        if candidate == node.node() {
            continue;
        }
        let Some(admin) = state.node(candidate).map(|record| record.admin) else {
            continue;
        };
        if let Ok(remote) = admin_get_membership(admin, tablet).await {
            return kivi_consensus::select_authoritative(local, node.node(), Some(remote));
        }
    }
    kivi_consensus::select_authoritative(local, node.node(), None)
}

/// Refreshes an observation from the tablet leader's own metrics when
/// we are not the leader: only the leader's view carries authoritative
/// replication lag (followers report unknown, which would gate
/// promotion forever). Falls back to the original view when the leader
/// is unknown, is us, or does not answer.
async fn refresh_from_leader(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    observed: &kivi_consensus::ObservedMembership,
) -> kivi_consensus::ObservedMembership {
    let Some(leader) = observed.leader.map(kivi_consensus::ReplicaId::node) else {
        return observed.clone();
    };
    if leader == node.node() {
        return observed.clone();
    }
    let Some(admin) = state.node(leader).map(|record| record.admin) else {
        return observed.clone();
    };
    let tablet = observed.group.tablet();
    match admin_get_membership(admin, tablet).await {
        Ok(fresh) => fresh,
        Err(_) => observed.clone(),
    }
}

/// Handle for executing one remote-or-local tablet step.
enum StepTarget {
    /// Run against the local front.
    Local(Arc<ConsensusNode>),
    /// Run through a peer's admin plane.
    Remote(SocketAddr),
}

/// Resolves where a step for `target` executes: locally when it is us,
/// else through the registry's admin endpoint.
fn step_target(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    target: NodeId,
) -> Option<StepTarget> {
    if target == node.node() {
        return Some(StepTarget::Local(Arc::clone(node)));
    }
    state
        .node(target)
        .map(|record| StepTarget::Remote(record.admin))
}

/// Registers the target as learner on the tablet leader (non-blocking:
/// catch-up is polled through replication lag). Runs on the observed
/// leader when remote, locally when we lead. Leadership churn between
/// observation and RPC is expected under load: one stale-leader miss
/// re-observes fresh and retries once before deferring to next pass.
async fn exec_learner(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    observed: &kivi_consensus::ObservedMembership,
    tablet: TabletId,
    target: NodeId,
) -> bool {
    let addr = state
        .node(target)
        .map(|record| record.peer.to_string())
        .unwrap_or_default();
    if addr.is_empty() {
        return false;
    }
    for attempt in 0..2 {
        let Some(view) = refresh_view(node, state, observed, tablet, attempt).await else {
            return false;
        };
        let Some(leader) = view.leader.map(kivi_consensus::ReplicaId::node) else {
            // A leaderless first view may just be stale (leadership
            // churn under load): spend the second attempt on a fresh
            // re-observation before deferring to the next pass.
            if attempt == 0 {
                continue;
            }
            tracing::debug!(
                tablet = tablet.as_u64(),
                target = target.as_u64(),
                "migration step waits: tablet has no leader"
            );
            return false;
        };
        let ok = if leader == node.node() {
            node.add_learner(tablet, target, addr.clone(), false)
                .await
                .is_ok()
        } else {
            match step_target(node, state, leader) {
                Some(StepTarget::Local(handle)) => handle
                    .add_learner(tablet, target, addr.clone(), false)
                    .await
                    .is_ok(),
                Some(StepTarget::Remote(admin)) => {
                    admin_post_learner(admin, tablet, target, &addr).await
                }
                None => false,
            }
        };
        if ok {
            return true;
        }
        tracing::debug!(
            tablet = tablet.as_u64(),
            target = target.as_u64(),
            leader = leader.as_u64(),
            attempt,
            "add_learner missed (stale leader?); re-observing"
        );
    }
    false
}

/// Fresh observation for a retry: re-observe from scratch, then take
/// the leader's own view (replication lag is authoritative there).
/// `None` only when nothing hosts the tablet.
async fn refresh_view(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    observed: &kivi_consensus::ObservedMembership,
    tablet: TabletId,
    attempt: u32,
) -> Option<kivi_consensus::ObservedMembership> {
    if attempt == 0 {
        return Some(refresh_from_leader(node, state, observed).await);
    }
    let fresh = observe_tablet(node, state, tablet).await?;
    Some(refresh_from_leader(node, state, &fresh).await)
}

/// Promotes the caught-up learner through joint consensus on the tablet
/// leader. `retain=false`: the retired source leaves the cluster. One
/// stale-leader miss re-observes fresh and retries once, like
/// [`exec_learner`].
async fn exec_members(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    observed: &kivi_consensus::ObservedMembership,
    tablet: TabletId,
    desired: &[NodeId],
) -> bool {
    let voters: BTreeSet<u64> = desired.iter().map(|node| node.as_u64()).collect();
    for attempt in 0..2 {
        let Some(view) = refresh_view(node, state, observed, tablet, attempt).await else {
            return false;
        };
        let Some(leader) = view.leader.map(kivi_consensus::ReplicaId::node) else {
            if attempt == 0 {
                continue;
            }
            return false;
        };
        let ok = if leader == node.node() {
            node.change_membership(tablet, voters.clone(), false)
                .await
                .is_ok()
        } else {
            match step_target(node, state, leader) {
                Some(StepTarget::Local(handle)) => handle
                    .change_membership(tablet, voters.clone(), false)
                    .await
                    .is_ok(),
                Some(StepTarget::Remote(admin)) => admin_post_members(admin, tablet, &voters).await,
                None => false,
            }
        };
        if ok {
            return true;
        }
        tracing::debug!(
            tablet = tablet.as_u64(),
            leader = leader.as_u64(),
            attempt,
            "change_membership missed (stale leader?); re-observing"
        );
    }
    false
}

/// Hands leadership away from a departing source replica.
async fn exec_transfer(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    observed: &kivi_consensus::ObservedMembership,
    tablet: TabletId,
    to: NodeId,
) {
    let Some(leader) = observed.leader.map(kivi_consensus::ReplicaId::node) else {
        return;
    };
    if leader == node.node() {
        if let Err(reason) = node.transfer_leader(tablet, to).await {
            tracing::debug!(tablet = tablet.as_u64(), %reason, "transfer deferred");
        }
        return;
    }
    if let Some(StepTarget::Remote(admin)) = step_target(node, state, leader) {
        admin_post_transfer(admin, tablet, to).await;
    }
}

/// Marks nodes `Drained` once nothing desires them and no live plan
/// touches them (plan completion already proved each source left its
/// voter set, so this is a safe commit, never a guess).
async fn complete_drains(node: &Arc<ConsensusNode>, state: &ControlState) {
    for record in state.nodes() {
        if record.state != NodeState::Draining {
            continue;
        }
        if !state.tablets_desiring(record.node).is_empty() {
            continue;
        }
        let touched = state
            .migrations()
            .filter(|plan| !plan.phase.is_terminal())
            .any(|plan| plan.from == record.node || plan.to == record.node);
        if touched {
            continue;
        }
        if let Err(reason) = node
            .propose_control(ControlMutation::SetNodeState {
                node: record.node,
                state: NodeState::Drained,
            })
            .await
        {
            tracing::debug!(node = record.node.as_u64(), %reason, "drain completion deferred");
        }
    }
}

/// Creates migration plans for `intents` at one placement generation:
/// publishes each tablet's new desired set, then persists its plan. One
/// shared pathway for auto-rebalance and manual moves (§46: no second
/// migration pathway).
pub async fn create_plans(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    intents: &[MigrationIntent],
) -> Result<Vec<MigrationPlanId>, String> {
    let generation = state
        .placement_version()
        .next()
        .map_err(|error| error.to_string())?;
    // Desired first (the goal), then the plan that reconciles toward it.
    for intent in intents {
        if intent.from == intent.to {
            continue;
        }
        let desired =
            DesiredReplicaSet::new(intent.tablet, intent.desired_voters.clone(), generation)
                .map_err(|error| error.to_string())?;
        node.propose_control(ControlMutation::SetDesiredPlacement { desired })
            .await?;
    }
    let plans = intents_to_plans(intents, state.next_plan_id(), generation);
    let mut ids = Vec::with_capacity(plans.len());
    for plan in plans {
        ids.push(plan.id);
        node.propose_control(ControlMutation::CreateMigration { plan })
            .await?;
    }
    Ok(ids)
}

/// Computes rebalance intents honoring the global live-plan cap.
#[must_use]
pub fn rebalance_intents(state: &ControlState, policy: &ReconcilePolicy) -> Vec<MigrationIntent> {
    let live = state
        .migrations()
        .filter(|plan| !plan.phase.is_terminal())
        .count();
    if live >= policy.max_live_plans {
        return Vec::new();
    }
    plan_rebalance(state, &policy.planner)
}

/// Computes drain intents for `draining` honoring the live-plan cap.
#[must_use]
pub fn drain_intents(
    state: &ControlState,
    draining: NodeId,
    policy: &ReconcilePolicy,
) -> Vec<MigrationIntent> {
    let live = state
        .migrations()
        .filter(|plan| !plan.phase.is_terminal())
        .count();
    if live >= policy.max_live_plans {
        return Vec::new();
    }
    plan_drain(state, draining, &policy.planner)
}

/// Computes automatic repair intents honoring the live-plan cap.
/// Safety outranks optimization: callers plan repairs before rebalances.
#[must_use]
pub fn repair_intents(state: &ControlState, policy: &ReconcilePolicy) -> Vec<MigrationIntent> {
    if !policy.auto_repair {
        return Vec::new();
    }
    let live = state
        .migrations()
        .filter(|plan| !plan.phase.is_terminal())
        .count();
    if live >= policy.max_live_plans {
        return Vec::new();
    }
    kivi_control::plan_repair(state, &policy.planner)
}

/// Rebalances tablet leadership (explicit operator action, never
/// continuous: no flapping). When one node leads materially more
/// tablets than another (beyond an eighth of the tablets), hands half
/// the excess to the least-represented voters through graceful
/// `transfer_leader` — never forced elections. Tablets with live
/// migration plans are skipped (they manage their own leadership).
/// Returns the number of transfers issued.
pub async fn rebalance_leadership(node: &Arc<ConsensusNode>, state: &ControlState) -> usize {
    let mut tablets: Vec<TabletId> = state.placements().map(|desired| desired.tablet).collect();
    tablets.sort_by_key(|tablet| tablet.as_u64());
    // Observe leadership per tablet (local first, peer admin fallback).
    let mut leaders: Vec<(TabletId, NodeId, Vec<NodeId>)> = Vec::new();
    for tablet in tablets {
        if !state.live_plans_for(tablet).is_empty() {
            continue;
        }
        let Some(observed) = observe_tablet(node, state, tablet).await else {
            continue;
        };
        let Some(leader) = observed.leader.map(kivi_consensus::ReplicaId::node) else {
            continue;
        };
        let voters: Vec<NodeId> = observed
            .voters
            .iter()
            .map(|id| NodeId::from_u64(*id))
            .collect();
        leaders.push((tablet, leader, voters));
    }
    if leaders.is_empty() {
        return 0;
    }
    let mut counts: BTreeMap<u64, usize> = BTreeMap::new();
    for (_, leader, _) in &leaders {
        *counts.entry(leader.as_u64()).or_default() += 1;
    }
    let total = leaders.len();
    let (max_node, max_count) = counts
        .iter()
        .max_by_key(|(_, count)| **count)
        .map_or((0, 0), |(node, count)| (*node, *count));
    let min_count = counts.values().copied().min().unwrap_or(0);
    // Hysteresis: act only on material imbalance (an eighth of the
    // tablets), and move half the excess per round.
    if max_count.saturating_sub(min_count) <= total / 8 || max_count == 0 {
        return 0;
    }
    let min_nodes: Vec<u64> = counts
        .iter()
        .filter(|(_, count)| **count == min_count)
        .map(|(node, _)| *node)
        .collect::<Vec<_>>();
    let mut moves = (max_count - min_count) / 2;
    let mut issued = 0usize;
    for (tablet, leader, voters) in &leaders {
        if moves == 0 {
            break;
        }
        if leader.as_u64() != max_node {
            continue;
        }
        // Hand to the least-represented node that actually votes here
        // (a non-voter can never take over).
        let Some(to) = min_nodes
            .iter()
            .filter(|node| voters.contains(&NodeId::from_u64(**node)))
            .min_by_key(|node| counts.get(*node).copied().unwrap_or(0))
            .copied()
        else {
            continue;
        };
        if to == max_node {
            continue;
        }
        let Some(observed) = observe_tablet(node, state, *tablet).await else {
            continue;
        };
        exec_transfer(node, state, &observed, *tablet, NodeId::from_u64(to)).await;
        *counts.entry(max_node).or_default() -= 1;
        *counts.entry(to).or_default() += 1;
        moves -= 1;
        issued += 1;
        if issued >= 16 {
            break;
        }
    }
    if issued > 0 {
        tracing::info!(transfers = issued, "leadership rebalance issued");
    }
    issued
}

/// Placement health for one tablet: desired voters, actual voters (when
/// observed), usability against node liveness, and the classified
/// [`kivi_control::TabletHealth`].
#[derive(Debug, Clone)]
pub struct PlacementHealth {
    /// Tablet assessed.
    pub tablet: TabletId,
    /// Desired voters from control state.
    pub desired: Vec<NodeId>,
    /// Actual voters from live membership (`None` when unobserved).
    pub actual: Option<Vec<NodeId>>,
    /// Learners catching up.
    pub learners: Vec<NodeId>,
    /// Live migration plan, if any.
    pub migrating: bool,
    /// Converged (desired == actual, no live plan).
    pub healthy: bool,
    /// Classified health (healthy/converging/under-replicated/unavailable/
    /// over-replicated) from usable desired voters and liveness.
    pub status: kivi_control::TabletHealth,
}

/// Assesses placement health for every desired tablet.
///
/// Usability counts desired voters whose registry state is
/// `Active`/`Suspect`/`Draining` (expected to serve); `Unavailable`
/// /`Removed`/`Joining`/unknown desired voters are unusable. Unobserved
/// tablets report by desired usability alone (no actual view to confirm).
#[must_use]
pub fn placement_health(
    state: &ControlState,
    observed: &BTreeMap<u64, kivi_consensus::ObservedMembership>,
) -> Vec<PlacementHealth> {
    let mut out = Vec::new();
    let mut tablets: Vec<TabletId> = state.placements().map(|desired| desired.tablet).collect();
    tablets.sort_by_key(|tablet| tablet.as_u64());
    for tablet in tablets {
        let desired = state
            .desired(tablet)
            .map(|set| set.replicas.clone())
            .unwrap_or_default();
        let migrating = !state.live_plans_for(tablet).is_empty();
        let (actual, learners) = observed
            .get(&tablet.as_u64())
            .map_or((None, Vec::new()), |obs| {
                (
                    Some(obs.voters.iter().map(|id| NodeId::from_u64(*id)).collect()),
                    obs.learners
                        .iter()
                        .map(|id| NodeId::from_u64(*id))
                        .collect(),
                )
            });
        let healthy = !migrating && actual.as_ref().is_some_and(|voters| *voters == desired);
        let usable = desired
            .iter()
            .copied()
            .filter(|node| {
                state.node(*node).is_some_and(|record| {
                    matches!(
                        record.state,
                        NodeState::Active | NodeState::Suspect | NodeState::Draining
                    )
                })
            })
            .count();
        let replication_factor = 3;
        let status =
            kivi_control::classify_tablet(desired.len(), usable, migrating, replication_factor);
        out.push(PlacementHealth {
            tablet,
            desired,
            actual,
            learners,
            migrating,
            healthy,
            status,
        });
    }
    out
}

/// The system control tablet id.
const fn control_tablet() -> TabletId {
    TabletId::from_u64(kivi_consensus::CONTROL_TABLET_RAW)
}

// ---------------------------------------------------------------------------
// Tiny admin HTTP client (hand-rolled HTTP/1.1, no new dependencies).
// The admin plane is loopback-trusted; this client speaks only to
// registry-advertised admin endpoints for reconciler forwarding.
// ---------------------------------------------------------------------------

/// Decodes one observed-membership admin response into the reconciler's
/// observation type (total: absent fields default, never an error).
fn decode_observed(
    tablet: TabletId,
    value: &serde_json::Value,
) -> kivi_consensus::ObservedMembership {
    let voters = value
        .get("voters")
        .and_then(serde_json::Value::as_array)
        .map(|ids| ids.iter().filter_map(serde_json::Value::as_u64).collect())
        .unwrap_or_default();
    let learners = value
        .get("learners")
        .and_then(serde_json::Value::as_array)
        .map(|ids| ids.iter().filter_map(serde_json::Value::as_u64).collect())
        .unwrap_or_default();
    let leader = value
        .get("leader")
        .and_then(serde_json::Value::as_u64)
        .map(|id| kivi_consensus::ReplicaId::of_node(NodeId::from_u64(id)));
    let term = value
        .get("term")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let is_joint = value
        .get("joint")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let lag = value
        .get("lag")
        .and_then(serde_json::Value::as_object)
        .map(|map| {
            map.iter()
                .filter_map(|(id, lag)| id.parse::<u64>().ok().zip(lag.as_u64()))
                .collect()
        })
        .unwrap_or_default();
    kivi_consensus::ObservedMembership::with_lag(
        ConsensusGroupId::of_tablet(tablet),
        voters,
        learners,
        leader,
        kivi_consensus::ConsensusTerm::new(term),
        is_joint,
        lag,
    )
}

/// Performs one admin GET and parses the JSON body.
async fn admin_get(admin: SocketAddr, path: &str) -> Result<serde_json::Value, String> {
    let body = http_roundtrip(admin, "GET", path, None).await?;
    serde_json::from_slice(&body).map_err(|error| format!("admin GET {path}: {error}"))
}

/// Performs one admin POST with a JSON body and parses the JSON reply.
async fn admin_post(
    admin: SocketAddr,
    path: &str,
    body: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let raw = serde_json::to_vec(body).map_err(|error| error.to_string())?;
    let reply = http_roundtrip(admin, "POST", path, Some(&raw)).await?;
    serde_json::from_slice(&reply).map_err(|error| format!("admin POST {path}: {error}"))
}

/// Minimal HTTP/1.1 round trip over a fresh connection (Content-Length
/// framing both ways; the admin plane always replies with one).
/// Bounded end to end (10 s): a wedged peer fails the step instead of
/// hanging the reconciler pass.
async fn http_roundtrip(
    admin: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
) -> Result<Vec<u8>, String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    async fn roundtrip(
        admin: SocketAddr,
        method: &str,
        path: &str,
        body: Option<&[u8]>,
    ) -> Result<Vec<u8>, String> {
        let mut stream = tokio::net::TcpStream::connect(admin)
            .await
            .map_err(|error| format!("admin dial {admin}: {error}"))?;
        let len = body.map_or(0, <[u8]>::len);
        let head = format!(
            "{method} {path} HTTP/1.1\r\nhost: {admin}\r\ncontent-type: application/json\r\ncontent-length: {len}\r\nconnection: close\r\n\r\n"
        );
        stream
            .write_all(head.as_bytes())
            .await
            .map_err(|error| error.to_string())?;
        if let Some(body) = body {
            stream
                .write_all(body)
                .await
                .map_err(|error| error.to_string())?;
        }
        let mut raw = Vec::new();
        stream
            .read_to_end(&mut raw)
            .await
            .map_err(|error| error.to_string())?;
        let header_end = raw
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .ok_or_else(|| "admin reply without headers".to_owned())?;
        Ok(raw[header_end + 4..].to_vec())
    }
    match tokio::time::timeout(
        Duration::from_secs(10),
        roundtrip(admin, method, path, body),
    )
    .await
    {
        Ok(result) => result,
        Err(_) => Err(format!("admin {method} {path} timed out")),
    }
}

/// Fetches one tablet's observed membership from a peer's admin plane.
async fn admin_get_membership(
    admin: SocketAddr,
    tablet: TabletId,
) -> Result<kivi_consensus::ObservedMembership, String> {
    let value = admin_get(
        admin,
        &format!("/v1/tablets/{}/membership", tablet.as_u64()),
    )
    .await?;
    Ok(decode_observed(tablet, &value))
}

/// Forwards `add_learner` (non-blocking; catch-up polled) to the tablet
/// leader's admin plane.
async fn admin_post_learner(
    admin: SocketAddr,
    tablet: TabletId,
    target: NodeId,
    addr: &str,
) -> bool {
    admin_post(
        admin,
        &format!("/v1/tablets/{}/learners", tablet.as_u64()),
        &serde_json::json!({ "node": target.as_u64(), "addr": addr, "blocking": false }),
    )
    .await
    .is_ok_and(|value| {
        value
            .get("ok")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    })
}

/// Forwards target-replica creation to the joining node's admin plane.
async fn admin_post_ensure(
    admin: SocketAddr,
    tablet: TabletId,
    target: NodeId,
    voters: &BTreeSet<u64>,
    generation: u64,
) -> bool {
    let voters: Vec<u64> = voters.iter().copied().collect();
    admin_post(
        admin,
        &format!("/v1/tablets/{}/ensure", tablet.as_u64()),
        &serde_json::json!({ "node": target.as_u64(), "voters": voters, "generation": generation }),
    )
    .await
    .is_ok_and(|value| {
        value
            .get("ok")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    })
}

/// Forwards source retirement to the leaving node's admin plane.
/// Returns whether the remote endpoint reported success.
async fn admin_post_retire(
    admin: SocketAddr,
    tablet: TabletId,
    source: NodeId,
    generation: u64,
) -> bool {
    admin_post(
        admin,
        &format!("/v1/tablets/{}/retire", tablet.as_u64()),
        &serde_json::json!({ "node": source.as_u64(), "generation": generation }),
    )
    .await
    .is_ok_and(|value| {
        value
            .get("ok")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    })
}

/// Forwards `change_membership` to the tablet leader's admin plane.
async fn admin_post_members(admin: SocketAddr, tablet: TabletId, voters: &BTreeSet<u64>) -> bool {
    let voters: Vec<u64> = voters.iter().copied().collect();
    admin_post(
        admin,
        &format!("/v1/tablets/{}/members", tablet.as_u64()),
        &serde_json::json!({ "voters": voters, "retain": false }),
    )
    .await
    .is_ok_and(|value| {
        value
            .get("ok")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    })
}

/// Forwards `transfer_leader` to the tablet leader's admin plane.
async fn admin_post_transfer(admin: SocketAddr, tablet: TabletId, to: NodeId) {
    let _ = admin_post(
        admin,
        &format!("/v1/tablets/{}/leader", tablet.as_u64()),
        &serde_json::json!({ "to": to.as_u64() }),
    )
    .await;
}

/// Forwards group initialization to the founder replica's admin plane.
async fn admin_post_initialize(
    admin: SocketAddr,
    tablet: TabletId,
    voters: &BTreeSet<u64>,
    peer_addrs: &BTreeMap<u64, String>,
) -> bool {
    let voters: Vec<u64> = voters.iter().copied().collect();
    admin_post(
        admin,
        &format!("/v1/tablets/{}/initialize", tablet.as_u64()),
        &serde_json::json!({ "voters": voters, "peer_addrs": peer_addrs }),
    )
    .await
    .is_ok_and(|value| {
        value
            .get("ok")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    })
}

/// Forwards the cutover fence to one replica's admin plane.
async fn admin_post_fence(admin: SocketAddr, tablet: TabletId, fenced: bool) -> bool {
    admin_post(
        admin,
        &format!("/v1/tablets/{}/fence", tablet.as_u64()),
        &serde_json::json!({ "fenced": fenced }),
    )
    .await
    .is_ok_and(|value| {
        value
            .get("ok")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    })
}

/// Forwards one split-child seed to the replica's admin plane (colocated
/// base, no bulk network copy). Ordered splits carry the real-key
/// boundary; hash splits carry the midpoint hash.
async fn admin_post_seed_split(
    admin: SocketAddr,
    child: TabletId,
    parent: TabletId,
    split_hash: u128,
    split_key: Option<&[u8]>,
    left: bool,
) -> bool {
    admin_post(
        admin,
        &format!("/v1/tablets/{}/seed-split", child.as_u64()),
        &serde_json::json!({
            "parent": parent.as_u64(),
            "split_hash": split_hash.to_string(),
            "split_key": split_key,
            "left": left,
        }),
    )
    .await
    .is_ok_and(|value| {
        value
            .get("ok")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    })
}

/// Forwards one merge-target seed to the replica's admin plane.
async fn admin_post_seed_merge(
    admin: SocketAddr,
    merged: TabletId,
    left: TabletId,
    right: TabletId,
) -> bool {
    admin_post(
        admin,
        &format!("/v1/tablets/{}/seed-merge", merged.as_u64()),
        &serde_json::json!({ "left": left.as_u64(), "right": right.as_u64() }),
    )
    .await
    .is_ok_and(|value| {
        value
            .get("ok")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    })
}

/// Forwards one directory cutover to a peer's admin plane (idempotent).
async fn admin_post_cutover(admin: SocketAddr, body: &serde_json::Value) -> bool {
    admin_post(admin, "/v1/directory/cutover", body)
        .await
        .is_ok_and(|value| {
            value
                .get("ok")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        })
}
