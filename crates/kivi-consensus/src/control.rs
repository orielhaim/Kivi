//! Replicated control-plane orchestration over the system control group.
//!
//! The control group is one dedicated Raft group
//! ([`ConsensusGroupId::control`](crate::types::ConsensusGroupId::control))
//! committing typed [`ControlMutation`](kivi_control::ControlMutation)s
//! through the same [`ConsensusCommand`](crate::command::ConsensusCommand)
//! envelope, mesh, and durability as every tablet group. This module owns
//! the orchestration vocabulary: what the reconciler observes about a
//! tablet group and which idempotent step it takes next.
//!
//! Only the active control-plane leader performs orchestration side
//! effects. Every step is idempotent: after a control-leader crash the
//! new leader reads persisted plans, observes actual group state, and
//! continues safely with no in-memory workflow continuation.

use std::collections::BTreeSet;

use kivi_types::{NodeId, TabletId};

use crate::types::{ConsensusGroupId, ConsensusTerm, ReplicaId};

/// Replication-lag threshold (entries) at or below which a learner
/// counts as caught up and safe to promote.
///
/// Mirrors `openraft::Config::replication_lag_threshold` (default
/// `5000`), the exact criterion the blocking `add_learner` waits on:
/// polling this threshold is the same gate without holding a reconciler
/// pass hostage for gigabytes of catch-up.
pub const CATCH_UP_LAG_THRESHOLD: u64 = 5000;

/// What the reconciler observed about one tablet group's actual Raft
/// state. Voters come from live membership (which distinguishes uniform
/// from joint through `is_joint`); learners, leader, term, and
/// replication lag come from the same metrics poll, so a step decision
/// never mixes two eras.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedMembership {
    /// Addressed group.
    pub group: ConsensusGroupId,
    /// Current voter ids.
    pub voters: BTreeSet<u64>,
    /// Current learner ids (catching up, not voting).
    pub learners: BTreeSet<u64>,
    /// Best-known leader, if any.
    pub leader: Option<ReplicaId>,
    /// Best-known term.
    pub term: ConsensusTerm,
    /// Whether the observed membership is a joint config (a
    /// membership transition is mid-flight; the reconciler must wait
    /// rather than propose a competing change).
    pub is_joint: bool,
    /// Replication lag per known node (leader's `last_log` minus the
    /// node's matched index; `u64::MAX` when unknown, e.g. observed on
    /// a follower or before any replication). Only the leader's view
    /// is authoritative.
    pub lag: std::collections::BTreeMap<u64, u64>,
}

impl ObservedMembership {
    /// Builds an observation for tests and diagnostics.
    #[must_use]
    pub fn new(
        group: ConsensusGroupId,
        voters: BTreeSet<u64>,
        learners: BTreeSet<u64>,
        leader: Option<ReplicaId>,
        term: ConsensusTerm,
        is_joint: bool,
    ) -> Self {
        Self {
            group,
            voters,
            learners,
            leader,
            term,
            is_joint,
            lag: std::collections::BTreeMap::new(),
        }
    }

    /// Builds an observation with replication lag (production path).
    #[must_use]
    pub fn with_lag(
        group: ConsensusGroupId,
        voters: BTreeSet<u64>,
        learners: BTreeSet<u64>,
        leader: Option<ReplicaId>,
        term: ConsensusTerm,
        is_joint: bool,
        lag: std::collections::BTreeMap<u64, u64>,
    ) -> Self {
        Self {
            group,
            voters,
            learners,
            leader,
            term,
            is_joint,
            lag,
        }
    }

    /// Whether `node` is currently a voter.
    #[must_use]
    pub fn is_voter(&self, node: NodeId) -> bool {
        self.voters.contains(&node.as_u64())
    }

    /// Whether `node` is currently a learner.
    #[must_use]
    pub fn is_learner(&self, node: NodeId) -> bool {
        self.learners.contains(&node.as_u64())
    }

    /// Replication lag for `node` (`u64::MAX` when unknown).
    #[must_use]
    pub fn lag_of(&self, node: NodeId) -> u64 {
        self.lag.get(&node.as_u64()).copied().unwrap_or(u64::MAX)
    }
}

/// Whether `target` is sufficiently caught up to promote: present as a
/// learner (or already a voter) with replication lag within
/// [`CATCH_UP_LAG_THRESHOLD`]. Unknown lag never promotes — promoting a
/// lagging node can stall quorum progress.
#[must_use]
pub fn caught_up(observed: &ObservedMembership, target: NodeId) -> bool {
    if observed.is_voter(target) {
        return true;
    }
    observed.is_learner(target) && observed.lag_of(target) <= CATCH_UP_LAG_THRESHOLD
}

/// The next idempotent orchestration step for one migration plan, given
/// its persisted phase and the observed actual membership.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconcileStep {
    /// Create the target replica locally (empty `TabletReplica`, group
    /// store, state machine) so it can receive learner replication.
    EnsureTargetReplica {
        /// Tablet to create.
        tablet: TabletId,
        /// Node joining.
        target: NodeId,
    },
    /// Register the target as learner on the tablet leader
    /// (`add_learner(target, blocking=true)`).
    AddLearner {
        /// Tablet migrating.
        tablet: TabletId,
        /// Node joining.
        target: NodeId,
    },
    /// Wait for learner catch-up (blocking learner not yet line-rate).
    /// No RPC: re-observe on the next reconciler pass.
    WaitCatchUp {
        /// Tablet migrating.
        tablet: TabletId,
        /// Node catching up.
        target: NodeId,
    },
    /// Promote the caught-up learner (`change_membership` to the
    /// desired voter set, `retain=false`).
    ChangeMembership {
        /// Tablet migrating.
        tablet: TabletId,
        /// Desired voters after the move.
        desired: Vec<NodeId>,
    },
    /// Wait for a joint membership to settle into its uniform
    /// successor. No RPC: `change_membership` flattens internally.
    WaitJoint {
        /// Tablet migrating.
        tablet: TabletId,
    },
    /// Move leadership away from the departing replica
    /// (`transfer_leader` to a retained voter) before it leaves.
    TransferLeadership {
        /// Tablet migrating.
        tablet: TabletId,
        /// Preferred successor (an eligible caught-up retained voter).
        to: NodeId,
    },
    /// Tombstone the retired source replica locally (unregister the
    /// active Raft group, stop serving, retain durable data).
    RetireSource {
        /// Tablet migrating.
        tablet: TabletId,
        /// Node leaving.
        source: NodeId,
    },
    /// Advance the persisted plan phase (control-plane commit).
    AdvancePlan {
        /// Affected plan.
        plan: kivi_control::MigrationPlanId,
        /// New phase.
        phase: kivi_control::MigrationPhase,
    },
    /// Nothing to do (terminal phase or already converged).
    Done,
}

/// Derives the next step for `plan` from `observed`.
///
/// Pure function of (persisted phase, observed membership): the
/// reconciler calls it every pass and executes the returned step, so a
/// retried or reordered pass converges instead of duplicating harmful
/// work. Leadership transfer is only requested when the departing
/// source is the observed leader.
///
/// The length is the phase dispatch table itself (one arm per phase);
/// splitting it would scatter the state machine across helpers for no
/// readability gain.
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn next_step(
    plan: &kivi_control::MigrationPlan,
    observed: &ObservedMembership,
) -> ReconcileStep {
    use kivi_control::MigrationPhase as Phase;
    // Terminal phases never act.
    if plan.phase.is_terminal() {
        return ReconcileStep::Done;
    }
    let source = plan.from;
    let target = plan.to;
    let desired: Vec<NodeId> = plan.desired_voters.clone();
    // Joint configs in flight always win: wait for OpenRaft's internal
    // flatten before proposing anything else (propose-after-commit).
    if observed.is_joint {
        return ReconcileStep::WaitJoint {
            tablet: plan.tablet,
        };
    }
    let target_voter = observed.is_voter(target);
    let target_learner = observed.is_learner(target);
    let source_voter = observed.is_voter(source);
    let converged = !source_voter && target_voter;
    match plan.phase {
        Phase::Planned => ReconcileStep::EnsureTargetReplica {
            tablet: plan.tablet,
            target,
        },
        Phase::TargetStarting => {
            if target_voter || target_learner {
                ReconcileStep::AdvancePlan {
                    plan: plan.id,
                    phase: Phase::LearnerAdded,
                }
            } else {
                ReconcileStep::AddLearner {
                    tablet: plan.tablet,
                    target,
                }
            }
        }
        Phase::LearnerAdded => {
            if target_voter {
                ReconcileStep::AdvancePlan {
                    plan: plan.id,
                    phase: Phase::Committed,
                }
            } else if target_learner {
                // The blocking `add_learner` returned once line-rate;
                // the reconciler treats presence-as-learner after a
                // successful blocking add as `Ready`. When catch-up is
                // still in flight the metrics show a large lag and the
                // pass waits instead of promoting early.
                ReconcileStep::WaitCatchUp {
                    tablet: plan.tablet,
                    target,
                }
            } else {
                ReconcileStep::AddLearner {
                    tablet: plan.tablet,
                    target,
                }
            }
        }
        Phase::Ready => {
            if target_voter {
                ReconcileStep::AdvancePlan {
                    plan: plan.id,
                    phase: Phase::Committed,
                }
            } else {
                ReconcileStep::ChangeMembership {
                    tablet: plan.tablet,
                    desired,
                }
            }
        }
        Phase::MembershipChanging => {
            // Strict convergence only: the source must be gone AND the
            // target a voter. A uniform set that still names the source
            // (or names extra voters) is NOT done — declaring Committed
            // early strands the plan in SourceRetiring forever, because
            // retiring the local replica never changes membership.
            if converged {
                ReconcileStep::AdvancePlan {
                    plan: plan.id,
                    phase: Phase::Committed,
                }
            } else {
                ReconcileStep::ChangeMembership {
                    tablet: plan.tablet,
                    desired,
                }
            }
        }
        Phase::Committed => {
            let leader_is_source = observed
                .leader
                .is_some_and(|leader| leader.node() == source);
            if source_voter && leader_is_source {
                // Graceful handoff first (deterministic successor: the
                // first retained desired voter).
                let successor = desired.into_iter().find(|node| *node != source);
                match successor {
                    Some(to) => ReconcileStep::TransferLeadership {
                        tablet: plan.tablet,
                        to,
                    },
                    None => ReconcileStep::RetireSource {
                        tablet: plan.tablet,
                        source,
                    },
                }
            } else if source_voter {
                // Membership regressed or never converged (residue set
                // still naming the source): re-drive convergence. The
                // re-proposal is safe (propose-after-commit holds on a
                // committed joint; a committed uniform makes it a
                // harmless redundant entry) and lands on the desired
                // uniform set.
                ReconcileStep::ChangeMembership {
                    tablet: plan.tablet,
                    desired,
                }
            } else {
                ReconcileStep::RetireSource {
                    tablet: plan.tablet,
                    source,
                }
            }
        }
        Phase::SourceRetiring => {
            if source_voter {
                // Same repair as above: the source must leave the voter
                // set before retirement completes.
                ReconcileStep::ChangeMembership {
                    tablet: plan.tablet,
                    desired: plan.desired_voters.clone(),
                }
            } else {
                // Membership converged; ensure the local replica is
                // retired (idempotent) before completing.
                ReconcileStep::RetireSource {
                    tablet: plan.tablet,
                    source,
                }
            }
        }
        Phase::Completed | Phase::Failed => ReconcileStep::Done,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use kivi_control::{MigrationPhase, MigrationPlan, MigrationPlanId, PlacementVersion};
    use kivi_types::{NodeId, TabletId};

    use super::*;
    use crate::types::{ConsensusTerm, ReplicaId};

    fn plan_at(phase: MigrationPhase) -> MigrationPlan {
        let mut plan = MigrationPlan::new(
            MigrationPlanId::from_u64(1),
            TabletId::from_u64(17),
            PlacementVersion::from_u64(2),
            NodeId::from_u64(1),
            NodeId::from_u64(4),
            vec![
                NodeId::from_u64(2),
                NodeId::from_u64(3),
                NodeId::from_u64(4),
            ],
        )
        .expect("plan");
        plan.phase = phase;
        plan
    }

    fn observed(voters: &[u64], learners: &[u64], leader: Option<u64>) -> ObservedMembership {
        ObservedMembership::new(
            ConsensusGroupId::of_tablet(TabletId::from_u64(17)),
            voters.iter().copied().collect::<BTreeSet<_>>(),
            learners.iter().copied().collect::<BTreeSet<_>>(),
            leader.map(|node| ReplicaId::of_node(NodeId::from_u64(node))),
            ConsensusTerm::new(3),
            false,
        )
    }

    #[test]
    fn planned_creates_target_first() {
        let step = next_step(
            &plan_at(MigrationPhase::Planned),
            &observed(&[1, 2, 3], &[], Some(1)),
        );
        assert!(matches!(step, ReconcileStep::EnsureTargetReplica { .. }));
    }

    #[test]
    fn learner_flow_waits_then_promotes() {
        // Learner present but not voter: wait, never promote early.
        let step = next_step(
            &plan_at(MigrationPhase::LearnerAdded),
            &observed(&[1, 2, 3], &[4], Some(1)),
        );
        assert!(matches!(step, ReconcileStep::WaitCatchUp { .. }));
        // Ready with a caught-up learner: change membership.
        let step = next_step(
            &plan_at(MigrationPhase::Ready),
            &observed(&[1, 2, 3], &[4], Some(2)),
        );
        assert!(matches!(step, ReconcileStep::ChangeMembership { .. }));
    }

    #[test]
    fn joint_membership_waits() {
        let mut obs = observed(&[1, 2, 3], &[4], Some(2));
        obs.is_joint = true;
        let step = next_step(&plan_at(MigrationPhase::MembershipChanging), &obs);
        assert!(matches!(step, ReconcileStep::WaitJoint { .. }));
    }

    #[test]
    fn leader_source_transfers_before_retire() {
        // Source still votes and leads: graceful handoff first.
        let step = next_step(
            &plan_at(MigrationPhase::Committed),
            &observed(&[1, 2, 3, 4], &[], Some(1)),
        );
        assert!(matches!(step, ReconcileStep::TransferLeadership { .. }));
        // Removed source that somehow still leads (stale): retire, not
        // transfer — it is already out of the voter set.
        let step = next_step(
            &plan_at(MigrationPhase::Committed),
            &observed(&[2, 3, 4], &[], Some(1)),
        );
        assert!(matches!(step, ReconcileStep::RetireSource { .. }));
        // Source gone from voters: straight to retire.
        let step = next_step(
            &plan_at(MigrationPhase::Committed),
            &observed(&[2, 3, 4], &[], Some(2)),
        );
        assert!(matches!(step, ReconcileStep::RetireSource { .. }));
    }

    #[test]
    fn residue_membership_repairs_instead_of_stalling() {
        // Uniform residue still naming the source is NOT committed:
        // re-drive convergence (both phases heal the same way).
        let step = next_step(
            &plan_at(MigrationPhase::Committed),
            &observed(&[1, 2, 3, 4], &[], Some(2)),
        );
        assert!(matches!(step, ReconcileStep::ChangeMembership { .. }));
        let step = next_step(
            &plan_at(MigrationPhase::SourceRetiring),
            &observed(&[1, 2, 3, 4], &[], Some(2)),
        );
        assert!(matches!(step, ReconcileStep::ChangeMembership { .. }));
        // Converged source-retiring retires locally.
        let step = next_step(
            &plan_at(MigrationPhase::SourceRetiring),
            &observed(&[2, 3, 4], &[], Some(2)),
        );
        assert!(matches!(step, ReconcileStep::RetireSource { .. }));
    }

    #[test]
    fn terminal_plans_rest() {
        let step = next_step(
            &plan_at(MigrationPhase::Completed),
            &observed(&[1, 2, 3], &[], Some(1)),
        );
        assert_eq!(step, ReconcileStep::Done);
    }
}
