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
    ControlMutation, ControlState, DesiredReplicaSet, MigrationIntent, MigrationPhase,
    MigrationPlan, MigrationPlanId, NodeRecord, NodeState, PlacementVersion, PlannerConfig,
    intents_to_plans, plan_drain, plan_rebalance,
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
/// with every seed tablet placed.
async fn genesis_done(node: &ConsensusNode, seed: &GenesisSeed) -> bool {
    let Some(state) = node.control_state().await else {
        return false;
    };
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
}

impl Default for ReconcilePolicy {
    /// Production shaping: brisk passes, RF=3, bounded churn.
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_millis(500),
            planner: PlannerConfig::default_rf3(),
            max_live_plans: 32,
        }
    }
}

/// Runs the migration reconciler until aborted: every pass syncs the
/// peer mesh from the replicated registry (all nodes), then — only on
/// the control leader — reads persisted plans, observes actual
/// membership, and advances one idempotent step per live plan.
pub async fn reconcile_loop(node: Arc<ConsensusNode>, policy: ReconcilePolicy) {
    loop {
        tokio::time::sleep(policy.poll_interval).await;
        mesh_sync(&node).await;
        if let Err(reason) = reconcile_once(&node, &policy).await {
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
        if matches!(
            record.state,
            NodeState::Joining | NodeState::Active | NodeState::Draining
        ) && node.add_peer_dial(record.node, record.peer)
        {
            tracing::info!(node = record.node.as_u64(), peer = %record.peer, "mesh admitted peer");
        }
    }
}

/// One reconciler pass. `Ok` means the pass completed (plans may still
/// be in flight); `Err` is a skip reason (not leader, no control
/// replica, transient observation failure).
async fn reconcile_once(node: &Arc<ConsensusNode>, policy: &ReconcilePolicy) -> Result<(), String> {
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
    // Top up drain plans as capacity frees: draining nodes need EVERY
    // tablet moved, but plan creation stays bounded per round, so the
    // reconciler refills until nothing desires the draining node.
    // (Rebalance stays operator-driven per round; drain completion is
    // operator-initiated and reconciler-finished.)
    topup_drains(node, &state, policy).await;
    // Complete drains whose migrations all finished.
    complete_drains(node, &state).await;
    Ok(())
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
            // the uniform config. Never wait passively.
            tracing::info!(
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
            // completes the plan. The open-time desired oracle
            // backstops the corner where the source never answers
            // again (it tombstones on return).
            if exec_retire(node, state, tablet, source, plan).await {
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
        plan = plan.id.as_u64(),
        tablet = plan.tablet.as_u64(),
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

/// Observes one tablet's actual membership: locally when this node is a
/// real member (voter or learner), else through a hosting peer's admin
/// plane (desired voters first, then the migration source).
///
/// A locally-ensured but not-yet-admitted worker (migration target
/// before `add_learner`) runs an embryonic Raft with empty membership
/// and no leader: trusting that view shadows the real cluster view and
/// wedges the plan (every pass sees "no leader" and defers). A
/// non-member local view is therefore only a last-resort fallback when
/// no member answers.
async fn observe_tablet(
    node: &Arc<ConsensusNode>,
    state: &ControlState,
    tablet: TabletId,
) -> Option<kivi_consensus::ObservedMembership> {
    let local = node.observe_membership(tablet).await.ok();
    if let Some(ref observed) = local
        && (observed.is_voter(node.node()) || observed.is_learner(node.node()))
    {
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
    for candidate in candidates {
        if candidate == node.node() {
            continue;
        }
        let Some(admin) = state.node(candidate).map(|record| record.admin) else {
            continue;
        };
        if let Ok(observed) = admin_get_membership(admin, tablet).await {
            return Some(observed);
        }
    }
    local
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
            tracing::info!(
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
        tracing::info!(
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
        tracing::info!(
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
/// observed), and whether the two converged with no live plan.
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
}

/// Assesses placement health for every desired tablet.
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
        out.push(PlacementHealth {
            tablet,
            desired,
            actual,
            learners,
            migrating,
            healthy,
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
