//! Conservative failure detector: ephemeral observations feeding the
//! replicated control plane, never consensus itself.
//!
//! ```text
//! node heartbeats / probes
//!         ↓
//! ephemeral observation at active control leader
//!         ↓
//! stable threshold crossed
//!         ↓
//! replicated node-state transition / repair intent
//! ```
//!
//! A few seconds of packet loss must not trigger hundreds of migrations:
//! `Active -> Suspect` fires after `suspicion_threshold` consecutive
//! missed polls, and `Suspect -> Unavailable` (the only repair-worthy
//! transition) fires after `repair_grace` further misses. A single
//! successful probe recovers `Suspect -> Active` immediately; an
//! `Unavailable` node recovers to `Active` on its first successful probe
//! (fenced: obsolete memberships never resurrect, see the open-time
//! desired oracle). `repair_cooldown` bounds how often one node can flip
//! back into repair after recovery.
//!
//! The detector holds no replicated state: the control leader owns it in
//! memory, and a new leader starts empty (all nodes unknown, no suspicion).
//! Only replicated `SetNodeState` commits change liveness; a false
//! suspicion may cause unnecessary movement through the standard
//! learner/catch-up/`change_membership` path, but never violates Raft
//! safety.

use std::collections::BTreeMap;
use std::time::Duration;

use kivi_types::NodeId;

use crate::node::NodeState;

/// Failure-detector tuning: all bounds explicit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailureDetectorConfig {
    /// Delay between liveness polls (reconciler pass interval).
    pub poll_interval: Duration,
    /// Consecutive missed polls before `Active -> Suspect`.
    pub suspicion_threshold: u32,
    /// Further consecutive misses (after suspicion) before
    /// `Suspect -> Unavailable` (repair-worthy).
    pub repair_grace_misses: u32,
    /// Minimum time between an `Unavailable -> Active` recovery and the
    /// next suspicion of the same node (anti-flap).
    pub repair_cooldown: Duration,
}

impl FailureDetectorConfig {
    /// Conservative production defaults: 500 ms polls, suspect after 4
    /// misses (~2 s), unavailable after 10 more misses (~5 s further, ~7 s
    /// total), 60 s cooldown after recovery.
    #[must_use]
    pub const fn conservative() -> Self {
        Self {
            poll_interval: Duration::from_millis(500),
            suspicion_threshold: 4,
            repair_grace_misses: 10,
            repair_cooldown: Duration::from_secs(60),
        }
    }

    /// Fast lab defaults for product tests (sub-second detection without
    /// flapping on a single missed poll).
    #[must_use]
    pub const fn lab() -> Self {
        Self {
            poll_interval: Duration::from_millis(200),
            suspicion_threshold: 3,
            repair_grace_misses: 5,
            repair_cooldown: Duration::from_secs(5),
        }
    }

    /// Total misses from healthy to repair-worthy.
    #[must_use]
    pub const fn unavailable_after_misses(self) -> u32 {
        self.suspicion_threshold
            .saturating_add(self.repair_grace_misses)
    }
}

/// Per-node ephemeral liveness counters (never persisted).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NodeLiveness {
    /// Consecutive missed polls.
    misses: u32,
    /// Polls since last `Unavailable -> Active` recovery (cooldown).
    since_recovery: u32,
}

impl NodeLiveness {
    const fn fresh() -> Self {
        Self {
            misses: 0,
            since_recovery: u32::MAX,
        }
    }
}

/// Ephemeral failure detector owned by the active control leader.
///
/// Tracks one counter per node; `observe` feeds probe results (any
/// successful admin/Raft contact counts as alive), and `pending` returns
/// the replicated `SetNodeState` transitions whose thresholds crossed.
/// All timing is in polls (the reconciler calls `observe` once per pass
/// per node), so no wall clock is needed and unit tests are deterministic.
#[derive(Debug, Clone)]
pub struct FailureDetector {
    config: FailureDetectorConfig,
    liveness: BTreeMap<u64, NodeLiveness>,
}

impl FailureDetector {
    /// Creates an empty detector (a new control leader starts here).
    #[must_use]
    pub const fn new(config: FailureDetectorConfig) -> Self {
        Self {
            config,
            liveness: BTreeMap::new(),
        }
    }

    /// Returns the detector configuration.
    #[must_use]
    pub const fn config(&self) -> &FailureDetectorConfig {
        &self.config
    }

    /// Retunes thresholds live, preserving per-node counters (a policy
    /// update never wipes suspicion history).
    pub fn set_config(&mut self, config: FailureDetectorConfig) {
        self.config = config;
    }

    /// Feeds one poll result for `node` given its current replicated
    /// state. Returns the desired replicated state when a threshold
    /// crossed (`Some`), or `None` when the current state stands.
    pub fn observe(&mut self, node: NodeId, alive: bool, current: NodeState) -> Option<NodeState> {
        let entry = self
            .liveness
            .entry(node.as_u64())
            .or_insert_with(NodeLiveness::fresh);
        if alive {
            let was_unavailable = current == NodeState::Unavailable;
            entry.misses = 0;
            if was_unavailable {
                entry.since_recovery = 0;
                return Some(NodeState::Active);
            }
            if current == NodeState::Suspect {
                return Some(NodeState::Active);
            }
            entry.since_recovery = entry.since_recovery.saturating_add(1);
            return None;
        }
        // Missed poll.
        entry.misses = entry.misses.saturating_add(1);
        entry.since_recovery = entry.since_recovery.saturating_add(1);
        // Cooldown after recovery: suppress immediate re-suspicion while
        // the node stabilizes (bounded: at most `cooldown / poll` polls).
        let cooldown_polls = u32::try_from(
            self.config
                .repair_cooldown
                .as_millis()
                .saturating_div(self.config.poll_interval.as_millis().max(1)),
        )
        .unwrap_or(u32::MAX);
        if entry.since_recovery < cooldown_polls && current == NodeState::Active {
            return None;
        }
        match current {
            NodeState::Active if entry.misses >= self.config.suspicion_threshold => {
                Some(NodeState::Suspect)
            }
            NodeState::Suspect
                if entry.misses
                    >= self
                        .config
                        .suspicion_threshold
                        .saturating_add(self.config.repair_grace_misses) =>
            {
                Some(NodeState::Unavailable)
            }
            // Direct Active -> Unavailable when grace already effectively
            // expired (e.g., a node missed many polls while the leader
            // was down and the new leader's first observations arrive in
            // bulk): never flap, always pass through Suspect first in the
            // replicated log — the caller issues one transition per pass,
            // so this arm is unreachable in practice but documents intent.
            _ => None,
        }
    }

    /// Forgets tracking for removed nodes (bounded memory on long runs).
    pub fn forget(&mut self, node: NodeId) {
        self.liveness.remove(&node.as_u64());
    }
}

/// Placement health of one tablet for repair prioritization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TabletHealth {
    /// Desired == actual usable voters, no live plan.
    Healthy,
    /// A live plan is converging toward desired.
    Converging,
    /// Usable replicas above zero but below policy (needs repair soon).
    UnderReplicated,
    /// No usable replica can serve (quorum danger first).
    Unavailable,
    /// More usable replicas than policy (cosmetic shrink, lowest priority).
    OverReplicated,
}

impl TabletHealth {
    /// Whether this health needs repair planning.
    #[must_use]
    pub const fn needs_repair(self) -> bool {
        matches!(self, Self::UnderReplicated | Self::Unavailable)
    }
}

/// Classifies one tablet's health from desired voters, usable voters
/// (desired voters in `Active`/`Suspect`/`Draining` states — i.e., not
/// `Unavailable`/`Removed`), and whether a live plan already covers it.
#[must_use]
pub fn classify_tablet(
    desired_len: usize,
    usable_len: usize,
    live_plan: bool,
    replication_factor: usize,
) -> TabletHealth {
    if live_plan {
        return TabletHealth::Converging;
    }
    if usable_len == 0 {
        return TabletHealth::Unavailable;
    }
    if usable_len < replication_factor.min(desired_len.max(1)) {
        return TabletHealth::UnderReplicated;
    }
    if usable_len < desired_len {
        return TabletHealth::UnderReplicated;
    }
    if usable_len > replication_factor {
        return TabletHealth::OverReplicated;
    }
    TabletHealth::Healthy
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u64) -> NodeId {
        NodeId::from_u64(id)
    }

    #[test]
    fn single_miss_never_suspects() {
        let mut detector = FailureDetector::new(FailureDetectorConfig::lab());
        assert_eq!(detector.observe(node(3), false, NodeState::Active), None);
        assert_eq!(detector.observe(node(3), true, NodeState::Active), None);
        // Back to zero misses: needs 3 consecutive misses to suspect.
        assert_eq!(detector.observe(node(3), false, NodeState::Active), None);
        assert_eq!(detector.observe(node(3), false, NodeState::Active), None);
        assert_eq!(
            detector.observe(node(3), false, NodeState::Active),
            Some(NodeState::Suspect)
        );
    }

    #[test]
    fn suspect_recovers_on_single_success() {
        let mut detector = FailureDetector::new(FailureDetectorConfig::lab());
        for _ in 0..3 {
            let _ = detector.observe(node(3), false, NodeState::Active);
        }
        assert_eq!(
            detector.observe(node(3), true, NodeState::Suspect),
            Some(NodeState::Active)
        );
    }

    #[test]
    fn grace_expires_to_unavailable() {
        let mut detector = FailureDetector::new(FailureDetectorConfig::lab());
        // 3 (suspicion) + 5 (grace) = 8 misses to unavailable.
        let mut transition = None;
        for index in 0..8 {
            let current = transition.unwrap_or(if index < 3 {
                NodeState::Active
            } else {
                NodeState::Suspect
            });
            // After the 3rd miss the replicated state is Suspect.
            let current = if index >= 3 {
                NodeState::Suspect
            } else {
                current
            };
            transition = detector.observe(node(3), false, current);
        }
        assert_eq!(transition, Some(NodeState::Unavailable));
    }

    #[test]
    fn unavailable_recovers_fenced_to_active() {
        let mut detector = FailureDetector::new(FailureDetectorConfig::lab());
        assert_eq!(
            detector.observe(node(3), true, NodeState::Unavailable),
            Some(NodeState::Active)
        );
    }

    #[test]
    fn short_outage_never_repairs() {
        // Two misses then recovery: stays Active the whole time.
        let mut detector = FailureDetector::new(FailureDetectorConfig::lab());
        assert_eq!(detector.observe(node(3), false, NodeState::Active), None);
        assert_eq!(detector.observe(node(3), false, NodeState::Active), None);
        assert_eq!(detector.observe(node(3), true, NodeState::Active), None);
    }

    #[test]
    fn health_classification_orders_safety_first() {
        assert_eq!(classify_tablet(3, 0, false, 3), TabletHealth::Unavailable);
        assert_eq!(
            classify_tablet(3, 2, false, 3),
            TabletHealth::UnderReplicated
        );
        assert_eq!(classify_tablet(3, 3, false, 3), TabletHealth::Healthy);
        assert_eq!(classify_tablet(3, 2, true, 3), TabletHealth::Converging);
        assert_eq!(
            classify_tablet(3, 4, false, 3),
            TabletHealth::OverReplicated
        );
    }
}
