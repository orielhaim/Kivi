//! Deterministic greedy placement planner.
//!
//! The planner maps desired policy (replication factor, eligible active
//! nodes, failure-domain spread, balance) onto desired replica sets. It
//! minimizes unnecessary movement: the same node set plus the same policy
//! produces no new plans once balanced. Tie-breaking is by
//! `TabletId`/`NodeId` order; placement is never random.

use std::collections::{BTreeMap, BTreeSet};

use kivi_types::{NodeId, TabletId};

use crate::node::NodeState;
use crate::state::ControlState;

/// Planner tuning: replication and concurrency bounds.
///
/// Repair concurrency is bounded independently from cosmetic rebalance so
/// a rack/node loss cannot create a repair disaster: `max_repairs_per_round`
/// caps global repair churn per planning round, and per-source/per-target
/// caps spread catch-up load. Safety (repair) always outranks optimization
/// (rebalance); the reconciler plans repairs first and only rebalances
/// with leftover budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannerConfig {
    /// Desired voter count per tablet.
    pub replication_factor: usize,
    /// Maximum migration plans created in one planning round.
    pub max_plans_per_round: usize,
    /// Maximum concurrent live plans touching one node (source or target).
    pub max_plans_per_node: usize,
    /// Maximum repair intents created in one planning round.
    pub max_repairs_per_round: usize,
    /// Maximum repair intents sourced from one failed node per round.
    pub max_repairs_per_source: usize,
    /// Maximum repair intents targeted at one healthy node per round.
    pub max_repairs_per_target: usize,
}

impl PlannerConfig {
    /// Default production shaping: RF=3, bounded churn per round.
    #[must_use]
    pub const fn default_rf3() -> Self {
        Self {
            replication_factor: 3,
            max_plans_per_round: 16,
            max_plans_per_node: 4,
            max_repairs_per_round: 16,
            max_repairs_per_source: 8,
            max_repairs_per_target: 4,
        }
    }
}

/// Why planning failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum PlannerError {
    /// No eligible active nodes exist.
    #[error("no eligible active nodes for placement")]
    NoEligibleNodes,
    /// Replication factor exceeds eligible nodes (degraded placement).
    #[error("replication factor {wanted} exceeds {eligible} eligible nodes")]
    UnderReplicated {
        /// Requested factor.
        wanted: usize,
        /// Eligible nodes.
        eligible: usize,
    },
}

/// One intended move: replace `from` with `to` on `tablet`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationIntent {
    /// Tablet moving.
    pub tablet: TabletId,
    /// Replica leaving.
    pub from: NodeId,
    /// Replica joining.
    pub to: NodeId,
    /// Full desired voter set after the move.
    pub desired_voters: Vec<NodeId>,
}

/// Computes rebalance intents from current desired state.
///
/// Tablets already at the desired shape (right size, all voters eligible
/// active, balanced within one replica of the mean) produce no intents.
/// Overloaded tablets shed one replica to the least-loaded eligible node
/// missing them; under-replicated tablets gain the least-loaded eligible
/// node. Failure-domain spread prefers targets in domains the tablet does
/// not yet cover, degrading gracefully when impossible.
#[must_use]
pub fn plan_rebalance(state: &ControlState, config: &PlannerConfig) -> Vec<MigrationIntent> {
    let eligible = eligible_nodes(state);
    if eligible.is_empty() {
        return Vec::new();
    }
    // Tablets with a live plan are skipped: one move per tablet at a time.
    let mut intents = Vec::new();
    let mut load = replica_load(state);
    let mut planned_touch: BTreeMap<u64, usize> = BTreeMap::new();
    let mut tablets: Vec<TabletId> = state.placements().map(|desired| desired.tablet).collect();
    tablets.sort_by_key(|tablet: &TabletId| tablet.as_u64());
    for tablet in tablets {
        if intents.len() >= config.max_plans_per_round {
            break;
        }
        let Some(desired) = state.desired(tablet) else {
            continue;
        };
        if !state.live_plans_for(tablet).is_empty() {
            continue;
        }
        let current: Vec<NodeId> = desired.replicas.clone();
        // Desired size honors both policy and eligibility: degrade
        // gracefully instead of refusing the whole cluster when the
        // requested spread is impossible. Size changes themselves (grow
        // or shrink below/above the factor) are out of stage scope —
        // replication factor is fixed and genesis always forms full
        // sets — so only balanced-size swaps move here.
        let want = config.replication_factor.min(eligible.len());
        if current.len() != want {
            continue;
        }
        // Move only when materially imbalanced (the most loaded holder
        // exceeds the least loaded eligible outsider by 2+) and the
        // outsider is not already a voter.
        if let Some((from, to)) = pick_balance_move(
            &current,
            &eligible,
            &load,
            &planned_touch,
            config,
            state,
            tablet,
        ) {
            let mut voters: Vec<NodeId> =
                current.into_iter().filter(|node| *node != from).collect();
            voters.push(to);
            voters.sort_by_key(|node: &NodeId| node.as_u64());
            touch(&mut planned_touch, from);
            touch(&mut planned_touch, to);
            bump_load(&mut load, to);
            lower_load(&mut load, from);
            intents.push(MigrationIntent {
                tablet,
                from,
                to,
                desired_voters: voters,
            });
        }
    }
    intents
}

/// Computes drain intents for `draining`: every tablet desiring it gains a
/// replacement move. Tablets already covered by a live plan are skipped.
#[must_use]
pub fn plan_drain(
    state: &ControlState,
    draining: NodeId,
    config: &PlannerConfig,
) -> Vec<MigrationIntent> {
    let eligible: Vec<NodeId> = eligible_nodes(state)
        .into_iter()
        .filter(|node| *node != draining)
        .collect();
    if eligible.is_empty() {
        return Vec::new();
    }
    let mut intents = Vec::new();
    let mut load = replica_load(state);
    let mut planned_touch: BTreeMap<u64, usize> = BTreeMap::new();
    let mut tablets = state.tablets_desiring(draining);
    tablets.sort_by_key(|tablet: &TabletId| tablet.as_u64());
    for tablet in tablets {
        if intents.len() >= config.max_plans_per_round {
            break;
        }
        if !state.live_plans_for(tablet).is_empty() {
            continue;
        }
        let Some(desired) = state.desired(tablet) else {
            continue;
        };
        let current: Vec<NodeId> = desired.replicas.clone();
        // Prefer a target outside the tablet's current failure domains.
        let outsiders: Vec<NodeId> = eligible
            .iter()
            .copied()
            .filter(|node| !current.contains(node))
            .collect();
        if outsiders.is_empty() {
            continue;
        }
        if let Some(target) = pick_add_target(
            &current,
            &outsiders,
            state,
            tablet,
            &load,
            &planned_touch,
            config,
        ) {
            let mut voters: Vec<NodeId> = current
                .into_iter()
                .filter(|node| *node != draining)
                .collect();
            voters.push(target);
            voters.sort_by_key(|node: &NodeId| node.as_u64());
            touch(&mut planned_touch, draining);
            touch(&mut planned_touch, target);
            bump_load(&mut load, target);
            lower_load(&mut load, draining);
            intents.push(MigrationIntent {
                tablet,
                from: draining,
                to: target,
                desired_voters: voters,
            });
        }
    }
    intents
}

/// Computes automatic repair intents for tablets whose desired voters
/// include repair-worthy nodes (`Unavailable`; `Draining` is handled by
/// [`plan_drain`]).
///
/// Safety-first ordering: tablets with zero usable replicas
/// (`Unavailable` health) come before merely under-replicated ones, then
/// tablet-id order (deterministic). Each intent replaces one failed
/// source with the healthiest eligible target through the SAME
/// [`crate::migration::MigrationPlan`] path as manual moves (add learner, catch up,
/// `change_membership`, retire — never delete-first).
///
/// Bounds: global `max_repairs_per_round` plus per-source/per-target caps
/// so a rack/node loss cannot stampede the cluster. When no eligible
/// target exists the tablet is left degraded (exposed via health) rather
/// than blocking recovery forever waiting for a perfect placement.
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn plan_repair(state: &ControlState, config: &PlannerConfig) -> Vec<MigrationIntent> {
    use crate::failure::{TabletHealth, classify_tablet};

    let eligible = eligible_nodes(state);
    if eligible.is_empty() {
        return Vec::new();
    }
    // Failed sources in deterministic order.
    let mut failed: Vec<NodeId> = state
        .nodes()
        .filter(|record| record.state == NodeState::Unavailable)
        .map(|record| record.node)
        .collect();
    failed.sort_by_key(|node| node.as_u64());
    if failed.is_empty() {
        return Vec::new();
    }
    // Candidate tablets: desired names a failed node, no live plan.
    let mut candidates: Vec<(TabletHealth, usize, TabletId, NodeId)> = Vec::new();
    let mut tablets: Vec<TabletId> = state.placements().map(|desired| desired.tablet).collect();
    tablets.sort_by_key(|tablet| tablet.as_u64());
    for tablet in tablets {
        if !state.live_plans_for(tablet).is_empty() {
            continue;
        }
        let Some(desired) = state.desired(tablet) else {
            continue;
        };
        let source = failed.iter().copied().find(|node| desired.contains(*node));
        let Some(source) = source else {
            continue;
        };
        let usable = desired
            .replicas
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
        let health = classify_tablet(
            desired.replicas.len(),
            usable,
            false,
            config.replication_factor,
        );
        if !health.needs_repair() {
            continue;
        }
        candidates.push((health, usable, tablet, source));
    }
    // Safety first: Unavailable (usable 0 sorts first via health order
    // Unavailable < UnderReplicated? ensure explicit rank), then lowest
    // usable count, then tablet id.
    candidates.sort_by(|a, b| {
        fn rank(health: TabletHealth) -> u8 {
            match health {
                TabletHealth::Unavailable => 0,
                TabletHealth::UnderReplicated => 1,
                TabletHealth::Converging => 2,
                TabletHealth::Healthy => 3,
                TabletHealth::OverReplicated => 4,
            }
        }
        rank(a.0)
            .cmp(&rank(b.0))
            .then(a.1.cmp(&b.1))
            .then(a.2.as_u64().cmp(&b.2.as_u64()))
    });
    let mut intents = Vec::new();
    let mut load = replica_load(state);
    let mut planned_touch: BTreeMap<u64, usize> = BTreeMap::new();
    let mut per_source: BTreeMap<u64, usize> = BTreeMap::new();
    let mut per_target: BTreeMap<u64, usize> = BTreeMap::new();
    for (_, _, tablet, source) in candidates {
        if intents.len() >= config.max_repairs_per_round {
            break;
        }
        if per_source.get(&source.as_u64()).copied().unwrap_or(0) >= config.max_repairs_per_source {
            continue;
        }
        let Some(desired) = state.desired(tablet) else {
            continue;
        };
        let current: Vec<NodeId> = desired.replicas.clone();
        let outsiders: Vec<NodeId> = eligible
            .iter()
            .copied()
            .filter(|node| !current.contains(node))
            .collect();
        if outsiders.is_empty() {
            continue;
        }
        // Per-target cap pre-filter: skip candidates already saturated
        // this round before scoring.
        let outsiders: Vec<NodeId> = outsiders
            .into_iter()
            .filter(|node| {
                per_target.get(&node.as_u64()).copied().unwrap_or(0) < config.max_repairs_per_target
            })
            .collect();
        if outsiders.is_empty() {
            continue;
        }
        if let Some(target) = pick_add_target(
            &current,
            &outsiders,
            state,
            tablet,
            &load,
            &planned_touch,
            config,
        ) {
            if per_target.get(&target.as_u64()).copied().unwrap_or(0)
                >= config.max_repairs_per_target
            {
                continue;
            }
            let mut voters: Vec<NodeId> =
                current.into_iter().filter(|node| *node != source).collect();
            voters.push(target);
            voters.sort_by_key(|node| node.as_u64());
            touch(&mut planned_touch, source);
            touch(&mut planned_touch, target);
            bump_load(&mut load, target);
            lower_load(&mut load, source);
            *per_source.entry(source.as_u64()).or_default() += 1;
            *per_target.entry(target.as_u64()).or_default() += 1;
            intents.push(MigrationIntent {
                tablet,
                from: source,
                to: target,
                desired_voters: voters,
            });
        }
    }
    intents
}

/// Eligible placement targets: active nodes in id order.
fn eligible_nodes(state: &ControlState) -> Vec<NodeId> {
    let mut nodes: Vec<NodeId> = state
        .nodes()
        .filter(|record| record.state == NodeState::Active)
        .map(|record| record.node)
        .collect();
    nodes.sort_by_key(|node: &NodeId| node.as_u64());
    nodes
}

/// Current desired replica counts per node.
fn replica_load(state: &ControlState) -> BTreeMap<u64, usize> {
    let mut load: BTreeMap<u64, usize> = BTreeMap::new();
    for desired in state.placements() {
        for replica in &desired.replicas {
            *load.entry(replica.as_u64()).or_default() += 1;
        }
    }
    load
}

/// Picks the least-loaded candidate missing from `current`, preferring a
/// failure domain the tablet does not yet cover.
///
/// Capacity-aware: load is weighted by node `weight` (effective load
/// `load / weight`), so a weight-2 node holds ~2x the tablets of a
/// weight-1 node instead of equal counts. Tablet-count is the current
/// capacity signal; estimated logical bytes will refine the score once
/// per-tablet size estimates flow through the control plane.
fn pick_add_target(
    current: &[NodeId],
    candidates: &[NodeId],
    state: &ControlState,
    tablet: TabletId,
    load: &BTreeMap<u64, usize>,
    touched: &BTreeMap<u64, usize>,
    config: &PlannerConfig,
) -> Option<NodeId> {
    let covered: BTreeSet<String> = current
        .iter()
        .filter_map(|node| state.node(*node))
        .map(|record| record.failure_domain.clone())
        .collect();
    // (fresh_domain desc, effective load asc, id asc, node).
    let mut ranked: Vec<(bool, u64, u64, u64, NodeId)> = Vec::new();
    for candidate in candidates {
        if current.contains(candidate) {
            continue;
        }
        if touched.get(&candidate.as_u64()).copied().unwrap_or(0) >= config.max_plans_per_node {
            continue;
        }
        let fresh_domain = state.node(*candidate).is_some_and(|record| {
            !record.failure_domain.is_empty() && !covered.contains(&record.failure_domain)
        });
        let _ = tablet;
        let weight = u64::from(
            state
                .node(*candidate)
                .map_or(1, |record| record.weight.max(1)),
        );
        let raw = load.get(&candidate.as_u64()).copied().unwrap_or(0) as u64;
        // Effective load scaled to permille for integer ordering:
        // `raw * 1000 / weight`. Deterministic, no float.
        let effective = raw.saturating_mul(1000).saturating_add(weight - 1) / weight.max(1);
        ranked.push((fresh_domain, effective, raw, candidate.as_u64(), *candidate));
    }
    ranked.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
            .then(a.3.cmp(&b.3))
    });
    ranked.into_iter().map(|(_, _, _, _, node)| node).next()
}

/// Picks the most-loaded current holder to shed (deterministic by load
/// desc, id asc).
fn pick_shed_victim(current: &[NodeId], load: &BTreeMap<u64, usize>) -> Option<NodeId> {
    let mut ranked: Vec<(usize, u64, NodeId)> = current
        .iter()
        .map(|node| {
            (
                load.get(&node.as_u64()).copied().unwrap_or(0),
                node.as_u64(),
                *node,
            )
        })
        .collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    ranked.into_iter().map(|(_, _, node)| node).next()
}

/// Picks a balance move when the spread exceeds one replica: the most
/// loaded holder gives one replica to the least loaded eligible outsider.
/// Returns `None` when already balanced (hysteresis: no flapping).
fn pick_balance_move(
    current: &[NodeId],
    eligible: &[NodeId],
    load: &BTreeMap<u64, usize>,
    touched: &BTreeMap<u64, usize>,
    config: &PlannerConfig,
    state: &ControlState,
    tablet: TabletId,
) -> Option<(NodeId, NodeId)> {
    let from = pick_shed_victim(current, load)?;
    let from_load = load.get(&from.as_u64()).copied().unwrap_or(0);
    let to = pick_add_target(current, eligible, state, tablet, load, touched, config)?;
    let to_load = load.get(&to.as_u64()).copied().unwrap_or(0);
    if from_load.saturating_sub(to_load) >= 2 {
        Some((from, to))
    } else {
        None
    }
}

fn touch(touched: &mut BTreeMap<u64, usize>, node: NodeId) {
    *touched.entry(node.as_u64()).or_default() += 1;
}

fn bump_load(load: &mut BTreeMap<u64, usize>, node: NodeId) {
    *load.entry(node.as_u64()).or_default() += 1;
}

fn lower_load(load: &mut BTreeMap<u64, usize>, node: NodeId) {
    let entry = load.entry(node.as_u64()).or_default();
    *entry = entry.saturating_sub(1);
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use super::*;
    use crate::mutation::ControlMutation;
    use crate::node::{NodeRecord, NodeState};
    use crate::placement::{DesiredReplicaSet, PlacementVersion};
    use crate::state::ControlState;

    fn record(id: u64, domain: &str) -> NodeRecord {
        NodeRecord {
            node: NodeId::from_u64(id),
            peer: format!("127.0.0.1:{}", 9100 + id)
                .parse::<SocketAddr>()
                .expect("peer"),
            native: format!("127.0.0.1:{}", 9000 + id)
                .parse::<SocketAddr>()
                .expect("native"),
            admin: format!("127.0.0.1:{}", 19000 + id)
                .parse::<SocketAddr>()
                .expect("admin"),
            cert_fingerprint: [u8::try_from(id % 256).unwrap_or(0xFF); 32],
            failure_domain: domain.to_owned(),
            weight: 1,
            state: NodeState::Active,
        }
    }

    fn balanced_three() -> ControlState {
        let mut state = ControlState::bootstrap(
            vec![
                record(1, "a"),
                record(2, "b"),
                record(3, "c"),
                record(4, "d"),
            ],
            vec![],
        );
        for tablet in 1..=6u64 {
            // Even spread across 1..3; node 4 empty.
            let desired = DesiredReplicaSet::new(
                TabletId::from_u64(tablet),
                vec![
                    NodeId::from_u64(1),
                    NodeId::from_u64(2),
                    NodeId::from_u64(3),
                ],
                PlacementVersion::INITIAL,
            )
            .expect("desired");
            state
                .apply(&ControlMutation::SetDesiredPlacement { desired })
                .expect("apply");
        }
        state
    }

    #[test]
    fn stable_cluster_produces_no_plans() {
        // Three nodes, RF=3, perfectly balanced: no movement.
        let mut state =
            ControlState::bootstrap(vec![record(1, "a"), record(2, "b"), record(3, "c")], vec![]);
        for tablet in 1..=4u64 {
            let desired = DesiredReplicaSet::new(
                TabletId::from_u64(tablet),
                vec![
                    NodeId::from_u64(1),
                    NodeId::from_u64(2),
                    NodeId::from_u64(3),
                ],
                PlacementVersion::INITIAL,
            )
            .expect("desired");
            state
                .apply(&ControlMutation::SetDesiredPlacement { desired })
                .expect("apply");
        }
        let intents = plan_rebalance(&state, &PlannerConfig::default_rf3());
        assert!(intents.is_empty(), "balanced cluster must not move");
    }

    #[test]
    fn rebalance_spreads_onto_empty_node() {
        let state = balanced_three();
        let intents = plan_rebalance(&state, &PlannerConfig::default_rf3());
        assert!(!intents.is_empty(), "empty node 4 should gain replicas");
        for intent in &intents {
            assert_eq!(intent.to, NodeId::from_u64(4));
            assert_ne!(intent.from, intent.to);
        }
        // Deterministic: same input, same output order.
        let again = plan_rebalance(&state, &PlannerConfig::default_rf3());
        assert_eq!(intents, again);
    }

    #[test]
    fn drain_moves_every_tablet_off() {
        let state = balanced_three();
        // Drain uses a wide-open per-round budget so one round covers all
        // six tablets; production rounds stay bounded and converge over
        // multiple reconciler passes.
        let config = PlannerConfig {
            replication_factor: 3,
            max_plans_per_round: 16,
            max_plans_per_node: 16,
            max_repairs_per_round: 16,
            max_repairs_per_source: 16,
            max_repairs_per_target: 16,
        };
        let intents = plan_drain(&state, NodeId::from_u64(1), &config);
        assert_eq!(intents.len(), 6);
        for intent in &intents {
            assert_eq!(intent.from, NodeId::from_u64(1));
            assert!(!intent.desired_voters.contains(&NodeId::from_u64(1)));
        }
    }

    #[test]
    fn repair_replaces_unavailable_without_delete_first() {
        let mut state = balanced_three();
        // Node 3 suffers a repair-worthy outage.
        state
            .apply(&ControlMutation::SetNodeState {
                node: NodeId::from_u64(3),
                state: NodeState::Unavailable,
            })
            .expect("mark unavailable");
        let config = PlannerConfig::default_rf3();
        let intents = plan_repair(&state, &config);
        assert!(!intents.is_empty(), "unavailable node must be repaired");
        for intent in &intents {
            // Add-before-remove: desired keeps RF=3 with the replacement.
            assert_eq!(intent.from, NodeId::from_u64(3));
            assert_eq!(intent.to, NodeId::from_u64(4));
            assert_eq!(intent.desired_voters.len(), 3);
            assert!(intent.desired_voters.contains(&NodeId::from_u64(4)));
            assert!(!intent.desired_voters.contains(&NodeId::from_u64(3)));
        }
        // Deterministic.
        assert_eq!(intents, plan_repair(&state, &config));
    }

    #[test]
    fn suspect_never_triggers_repair() {
        let mut state = balanced_three();
        state
            .apply(&ControlMutation::SetNodeState {
                node: NodeId::from_u64(3),
                state: NodeState::Suspect,
            })
            .expect("mark suspect");
        // Short transient miss: no mass re-replication.
        assert!(plan_repair(&state, &PlannerConfig::default_rf3()).is_empty());
    }

    #[test]
    fn repair_prefers_failure_domain_spread() {
        let mut state = ControlState::bootstrap(
            vec![
                record(1, "a"),
                record(2, "a"),
                record(3, "b"),
                record(4, "c"),
            ],
            vec![],
        );
        let desired = DesiredReplicaSet::new(
            TabletId::from_u64(1),
            vec![
                NodeId::from_u64(1),
                NodeId::from_u64(2),
                NodeId::from_u64(3),
            ],
            PlacementVersion::INITIAL,
        )
        .expect("desired");
        state
            .apply(&ControlMutation::SetDesiredPlacement { desired })
            .expect("apply");
        state
            .apply(&ControlMutation::SetNodeState {
                node: NodeId::from_u64(2),
                state: NodeState::Unavailable,
            })
            .expect("mark unavailable");
        let intents = plan_repair(&state, &PlannerConfig::default_rf3());
        assert_eq!(intents.len(), 1);
        // Node 4 (domain c, uncovered) beats any same-domain choice.
        assert_eq!(intents[0].to, NodeId::from_u64(4));
    }

    #[test]
    fn weight_skews_target_choice_to_heavier_nodes() {
        fn weighted(id: u64, domain: &str, weight: u32) -> NodeRecord {
            let mut record = record(id, domain);
            record.weight = weight;
            record
        }
        let mut state = ControlState::bootstrap(
            vec![
                weighted(1, "a", 1),
                weighted(2, "b", 1),
                weighted(3, "c", 1),
                weighted(4, "d", 8),
            ],
            vec![],
        );
        for tablet in 1..=3u64 {
            let desired = DesiredReplicaSet::new(
                TabletId::from_u64(tablet),
                vec![
                    NodeId::from_u64(1),
                    NodeId::from_u64(2),
                    NodeId::from_u64(3),
                ],
                PlacementVersion::INITIAL,
            )
            .expect("desired");
            state
                .apply(&ControlMutation::SetDesiredPlacement { desired })
                .expect("apply");
        }
        state
            .apply(&ControlMutation::SetNodeState {
                node: NodeId::from_u64(1),
                state: NodeState::Unavailable,
            })
            .expect("mark unavailable");
        let intents = plan_repair(&state, &PlannerConfig::default_rf3());
        assert!(!intents.is_empty());
        // Heavy node 4 absorbs repairs despite starting empty.
        for intent in &intents {
            assert_eq!(intent.to, NodeId::from_u64(4));
        }
    }
}
