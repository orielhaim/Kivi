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

/// Authority of one tablet-group observation: an actual member's view
/// always outranks a non-member local placeholder.
///
/// An embryonic local Raft group (created by `ensure_group` before
/// `add_learner`, with empty membership and no leader, where the local
/// node is neither voter nor learner) must NEVER shadow an authoritative
/// observation from actual members. A local process existing is not
/// evidence that its group view is authoritative. Centralize that rule
/// here instead of scattering `membership.is_empty()` checks through the
/// reconciler; migration, repair, split-child, and merge-target creation
/// all share this path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationAuthority {
    /// The observer is a voter or learner: its view is authoritative.
    Member,
    /// The observer is not a member: usable only when no member answers.
    NonMemberFallback,
}

/// Classifies a local observation's authority for `local` node.
///
/// A missing local group has no authority at all (`None`); a present
/// group where `local` votes or learns is authoritative; any other
/// present group (including embryonic empty-membership leaderless
/// placeholders) is a last-resort fallback only.
#[must_use]
pub fn observation_authority(
    observed: Option<&ObservedMembership>,
    local: NodeId,
) -> Option<ObservationAuthority> {
    let observed = observed?;
    if observed.is_voter(local) || observed.is_learner(local) {
        Some(ObservationAuthority::Member)
    } else {
        Some(ObservationAuthority::NonMemberFallback)
    }
}

/// Selects the authoritative observation: a member-local view wins
/// immediately; otherwise the caller-provided remote view wins when
/// present; otherwise the non-member local fallback is returned (which
/// may still be `None` when nothing hosts the group).
#[must_use]
pub fn select_authoritative(
    local: Option<ObservedMembership>,
    local_node: NodeId,
    remote: Option<ObservedMembership>,
) -> Option<ObservedMembership> {
    if matches!(
        observation_authority(local.as_ref(), local_node),
        Some(ObservationAuthority::Member)
    ) {
        return local;
    }
    remote.or(local)
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
/// work. Leadership transfer is requested both after convergence (the
/// departing source still leads) and before reconfiguring out from
/// under a leading source (graceful handoff via [`handoff_successor`]).
///
/// The length is the phase dispatch table itself (one arm per phase);
/// splitting it would scatter the state machine across helpers for no
/// readability gain.
#[allow(clippy::too_many_lines)]
#[must_use]
/// Graceful-handoff target when the migration source still leads: the
/// most caught-up retained desired voter that already votes (desired
/// order breaks lag ties).
///
/// Rationale: changing membership out from under a leading source
/// removes the only leader — the uniform commit steps the source down
/// and the group goes leaderless until a fresh election completes,
/// burning client redirect budgets with `Overloaded` the whole window.
/// Handing off first keeps a live leader across the reconfiguration, so
/// migration is hitless. Returns `None` when no established voter can
/// take over yet (all lagged/unknown): the caller then proceeds with the
/// membership change rather than stalling migration behind a transfer —
/// a brief election window beats a stuck plan. The newcomer target is
/// never chosen: it only becomes a voter through the change itself.
/// Single-voter moves have no candidate by construction (the outage is
/// unavoidable there, not a handoff bug).
fn handoff_successor(
    observed: &ObservedMembership,
    desired: &[NodeId],
    source: NodeId,
) -> Option<NodeId> {
    desired
        .iter()
        .filter(|node| **node != source && observed.is_voter(**node))
        .filter(|node| observed.lag_of(**node) <= CATCH_UP_LAG_THRESHOLD)
        .min_by_key(|node| (observed.lag_of(**node), desired_position(desired, **node)))
        .copied()
}

/// Position of `node` in `desired` (desired order breaks lag ties so the
/// choice is deterministic for a given plan).
fn desired_position(desired: &[NodeId], node: NodeId) -> usize {
    desired
        .iter()
        .position(|candidate| *candidate == node)
        .unwrap_or(usize::MAX)
}

/// Derives the next step for `plan` from `observed` (phase dispatch
/// table; see the function body for the per-phase rules, including the
/// graceful leadership handoff before reconfiguring out from under a
/// leading migration source).
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
            } else if observed
                .leader
                .as_ref()
                .is_some_and(|leader| leader.node() == source)
                && let Some(to) = handoff_successor(observed, &desired, source)
            {
                // The source still leads: hand off to an established
                // retained voter BEFORE changing membership (see
                // [`handoff_successor`]), so the reconfiguration never
                // removes a live leader.
                ReconcileStep::TransferLeadership {
                    tablet: plan.tablet,
                    to,
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
            } else if observed
                .leader
                .as_ref()
                .is_some_and(|leader| leader.node() == source)
                && let Some(to) = handoff_successor(observed, &desired, source)
            {
                // Same graceful handoff on re-drives: never reconfigure
                // out from under a leading source.
                ReconcileStep::TransferLeadership {
                    tablet: plan.tablet,
                    to,
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
        // Ready with a caught-up learner and a non-source leader: change
        // membership (no handoff needed).
        let step = next_step(
            &plan_at(MigrationPhase::Ready),
            &observed(&[1, 2, 3], &[4], Some(2)),
        );
        assert!(matches!(step, ReconcileStep::ChangeMembership { .. }));
    }

    fn observed_with_lag(
        voters: &[u64],
        learners: &[u64],
        leader: Option<u64>,
        lag: &[(u64, u64)],
    ) -> ObservedMembership {
        ObservedMembership::with_lag(
            ConsensusGroupId::of_tablet(TabletId::from_u64(17)),
            voters.iter().copied().collect::<BTreeSet<_>>(),
            learners.iter().copied().collect::<BTreeSet<_>>(),
            leader.map(|node| ReplicaId::of_node(NodeId::from_u64(node))),
            ConsensusTerm::new(3),
            false,
            lag.iter().copied().collect(),
        )
    }

    #[test]
    fn ready_hands_off_before_removing_a_leading_source() {
        // Source still leads with caught-up retained voters: hand off to
        // the first retained desired voter (2) instead of reconfiguring
        // out from under the only leader.
        let step = next_step(
            &plan_at(MigrationPhase::Ready),
            &observed_with_lag(&[1, 2, 3], &[4], Some(1), &[(1, 0), (2, 0), (3, 0)]),
        );
        assert!(matches!(
            step,
            ReconcileStep::TransferLeadership { to, .. } if to == NodeId::from_u64(2)
        ));
        // Same handoff on membership re-drives while the source leads.
        let step = next_step(
            &plan_at(MigrationPhase::MembershipChanging),
            &observed_with_lag(&[1, 2, 3], &[4], Some(1), &[(1, 0), (2, 5), (3, 0)]),
        );
        assert!(matches!(
            step,
            ReconcileStep::TransferLeadership { to, .. } if to == NodeId::from_u64(3)
        ));
        // No caught-up established voter (lags unknown): proceed with the
        // membership change rather than stalling behind a transfer.
        let step = next_step(
            &plan_at(MigrationPhase::Ready),
            &observed(&[1, 2, 3], &[4], Some(1)),
        );
        assert!(matches!(step, ReconcileStep::ChangeMembership { .. }));
        // Non-source leader: no handoff, straight to membership change.
        let step = next_step(
            &plan_at(MigrationPhase::Ready),
            &observed_with_lag(&[1, 2, 3], &[4], Some(2), &[(1, 0), (2, 0), (3, 0)]),
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

    #[test]
    fn embryonic_local_never_shadows_member() {
        // Regression for the drain stall: the control leader is also the
        // migration target, `ensure_group` created an embryonic local
        // group (empty membership, no leader, not voter/learner), while
        // the real group still lives on members. The embryonic view must
        // lose to the member view.
        let embryonic = ObservedMembership::new(
            ConsensusGroupId::of_tablet(TabletId::from_u64(17)),
            BTreeSet::new(),
            BTreeSet::new(),
            None,
            ConsensusTerm::new(0),
            false,
        );
        let member = observed(&[1, 2, 3], &[], Some(1));
        // Embryonic local is a non-member fallback, not authoritative.
        assert_eq!(
            observation_authority(Some(&embryonic), NodeId::from_u64(4)),
            Some(ObservationAuthority::NonMemberFallback)
        );
        // Member local is authoritative.
        assert_eq!(
            observation_authority(Some(&member), NodeId::from_u64(1)),
            Some(ObservationAuthority::Member)
        );
        // Selection prefers the remote member over the embryonic local.
        let picked = select_authoritative(
            Some(embryonic.clone()),
            NodeId::from_u64(4),
            Some(member.clone()),
        );
        assert_eq!(picked, Some(member.clone()));
        // Without any remote, the embryonic fallback is returned (caller
        // defers on leaderless), never mistaken for authority.
        let fallback = select_authoritative(Some(embryonic), NodeId::from_u64(4), None);
        assert!(fallback.is_some());
        assert_eq!(
            observation_authority(fallback.as_ref(), NodeId::from_u64(4)),
            Some(ObservationAuthority::NonMemberFallback)
        );
        // Missing local yields the remote member directly.
        let picked = select_authoritative(None, NodeId::from_u64(4), Some(member.clone()));
        assert_eq!(picked, Some(member));
    }

    #[test]
    fn learner_local_is_authoritative() {
        // A learner (catching-up target) observes authoritatively: it is a
        // real group participant, unlike an embryonic placeholder.
        let learner = observed(&[1, 2, 3], &[4], Some(1));
        assert_eq!(
            observation_authority(Some(&learner), NodeId::from_u64(4)),
            Some(ObservationAuthority::Member)
        );
        let picked = select_authoritative(
            Some(learner.clone()),
            NodeId::from_u64(4),
            Some(observed(&[1, 2, 3], &[], Some(2))),
        );
        assert_eq!(picked, Some(learner));
    }
}
