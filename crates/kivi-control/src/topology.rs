//! Unified control-plan model: one id/ownership/conflict vocabulary over
//! migration, split, and merge plans.
//!
//! Variant-specific payloads and phases stay in their own modules
//! ([`MigrationPlan`](crate::migration::MigrationPlan),
//! [`SplitPlan`](crate::split::SplitPlan),
//! [`MergePlan`](crate::merge::MergePlan)); forcing all phases into one
//! giant enum would make the code worse. What is shared:
//!
//! ```text
//! idempotency (generation-fenced advances)
//! ownership (one live plan per tablet at a time)
//! conflict detection (overlapping tablet sets never run together)
//! progress reporting (live vs terminal per kind)
//! retry/backoff (reconciler re-derives steps from persisted phase)
//! ```
//!
//! Plan ids share one sequence (`next_plan_id`): an id is unique across
//! kinds, so logs and admin surfaces never confuse a migration with a
//! split. Tablet ids for dynamically allocated children/merged tablets
//! share a second sequence (`next_tablet_id`, see
//! [`ControlState`](crate::state::ControlState)).

use kivi_types::TabletId;

/// Identity shared across plan kinds (migrations, splits, merges).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlanId(pub u64);

impl PlanId {
    /// Wraps a raw plan id.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// Which workflow a plan belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanKind {
    /// Physical move (`MigrationPlan`).
    Migration,
    /// Logical split (`SplitPlan`).
    Split,
    /// Logical merge (`MergePlan`).
    Merge,
}

/// Scheduler priority: safety wins over optimization.
///
/// 1. unavailable / quorum-danger tablets (repair)
///
/// 2. under-replicated tablets (repair)
///
/// 3. node drain
///
/// 4. normal rebalance
///
/// 5. split/merge optimization
///
/// 6. leadership cosmetic balance
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PlanPriority {
    /// Quorum-danger / unavailable-tablet repair.
    RepairCritical = 0,
    /// Under-replicated repair.
    Repair = 1,
    /// Operator drain.
    Drain = 2,
    /// Manual move / rebalance.
    Rebalance = 3,
    /// Split/merge optimization.
    Topology = 4,
    /// Leadership balance (cosmetic, never automatic churn).
    Leadership = 5,
}

/// Scheduler bounds across plan kinds (one source of active-plan limits).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanSchedulerConfig {
    /// Maximum live migrations (manual + rebalance + repair) cluster-wide.
    pub max_live_migrations: usize,
    /// Maximum live splits cluster-wide.
    pub max_live_splits: usize,
    /// Maximum live merges cluster-wide.
    pub max_live_merges: usize,
}

impl PlanSchedulerConfig {
    /// Conservative production defaults: migrations bounded by the
    /// reconciler cap, one split and one merge at a time (topology
    /// changes serialize globally until proven safe concurrently).
    #[must_use]
    pub const fn conservative() -> Self {
        Self {
            max_live_migrations: 32,
            max_live_splits: 1,
            max_live_merges: 1,
        }
    }
}

/// Whether two tablet sets conflict (share any tablet): a tablet never
/// executes two incompatible topology plans at once (e.g., migrating
/// `A→D` while splitting the same tablet serializes; repair wins over
/// split — see planner eligibility).
#[must_use]
pub fn tablet_sets_conflict(first: &[TabletId], second: &[TabletId]) -> bool {
    first.iter().any(|tablet| second.contains(tablet))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conflict_detects_overlap() {
        let first = [TabletId::from_u64(1), TabletId::from_u64(2)];
        let second = [TabletId::from_u64(2), TabletId::from_u64(3)];
        let third = [TabletId::from_u64(4)];
        assert!(tablet_sets_conflict(&first, &second));
        assert!(!tablet_sets_conflict(&first, &third));
    }

    #[test]
    fn safety_orders_before_optimization() {
        assert!(PlanPriority::RepairCritical < PlanPriority::Repair);
        assert!(PlanPriority::Repair < PlanPriority::Drain);
        assert!(PlanPriority::Drain < PlanPriority::Rebalance);
        assert!(PlanPriority::Rebalance < PlanPriority::Topology);
        assert!(PlanPriority::Topology < PlanPriority::Leadership);
    }
}
