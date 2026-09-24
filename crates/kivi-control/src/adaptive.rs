//! Phase 12 adaptive control decisions over typed observations.

use core::time::Duration;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use kivi_observation::{
    Confidence, FreshnessStatus, HotnessClass, ObservationScope, ObservationSequence,
    ObservationSnapshot, ObservationWindowId, ScopeSnapshot,
};
use kivi_types::{ClusterId, NamespaceId, NodeId, TabletId, Ticks};

const PPM_SCALE: u64 = 1_000_000;
const MODEL_FEATURE_COUNT: usize = 11;
const MAX_MODEL_BYTES: usize = 64 * 1024;

/// A policy fraction expressed in parts per million.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Ppm(u32);

impl Ppm {
    /// Zero policy units.
    pub const ZERO: Self = Self(0);
    /// One whole policy unit.
    pub const ONE: Self = Self(1_000_000);

    /// Creates a value when it is inside the closed unit interval.
    #[must_use]
    pub const fn new(value: u32) -> Option<Self> {
        if value <= 1_000_000 {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Returns the parts-per-million representation.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Returns the value as a bounded `u64`.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0 as u64
    }

    /// Adds without wrapping and caps at one whole unit.
    #[must_use]
    pub const fn saturating_add(self, other: Self) -> Self {
        let value = self.0.saturating_add(other.0);
        if value > 1_000_000 {
            Self(1_000_000)
        } else {
            Self(value)
        }
    }

    /// Subtracts without wrapping and floors at zero.
    #[must_use]
    pub const fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }
}

/// A bounded memory budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryBudget(Ppm);

impl MemoryBudget {
    /// Creates a valid memory budget.
    #[must_use]
    pub const fn new(value: u32) -> Option<Self> {
        match Ppm::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
    /// Returns the underlying policy fraction.
    #[must_use]
    pub const fn as_ppm(self) -> Ppm {
        self.0
    }
    /// Adds a bounded step.
    #[must_use]
    pub const fn saturating_add(self, step: Ppm) -> Self {
        Self(self.0.saturating_add(step))
    }
    /// Subtracts a bounded step.
    #[must_use]
    pub const fn saturating_sub(self, step: Ppm) -> Self {
        Self(self.0.saturating_sub(step))
    }
}

/// A bounded repair budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RepairBudget(Ppm);

impl RepairBudget {
    /// Creates a valid repair budget.
    #[must_use]
    pub const fn new(value: u32) -> Option<Self> {
        match Ppm::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
    /// Returns the underlying policy fraction.
    #[must_use]
    pub const fn as_ppm(self) -> Ppm {
        self.0
    }
    /// Adds a bounded step.
    #[must_use]
    pub const fn saturating_add(self, step: Ppm) -> Self {
        Self(self.0.saturating_add(step))
    }
    /// Subtracts a bounded step.
    #[must_use]
    pub const fn saturating_sub(self, step: Ppm) -> Self {
        Self(self.0.saturating_sub(step))
    }
}

/// A bounded scrub budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScrubBudget(Ppm);

impl ScrubBudget {
    /// Creates a valid scrub budget.
    #[must_use]
    pub const fn new(value: u32) -> Option<Self> {
        match Ppm::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
    /// Returns the underlying policy fraction.
    #[must_use]
    pub const fn as_ppm(self) -> Ppm {
        self.0
    }
    /// Adds a bounded step.
    #[must_use]
    pub const fn saturating_add(self, step: Ppm) -> Self {
        Self(self.0.saturating_add(step))
    }
    /// Subtracts a bounded step.
    #[must_use]
    pub const fn saturating_sub(self, step: Ppm) -> Self {
        Self(self.0.saturating_sub(step))
    }
}

/// Expected benefit attached to an action proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpectedBenefit {
    /// Expected latency reduction.
    pub latency: Ppm,
    /// Expected throughput gain.
    pub throughput: Ppm,
    /// Expected general resource saving.
    pub resource: Ppm,
    /// Expected memory saving.
    pub memory: Ppm,
    /// Expected storage saving.
    pub storage: Ppm,
    /// Expected network saving.
    pub network: Ppm,
    /// Expected repair-debt reduction.
    pub repair_debt: Ppm,
    /// Expected durability-risk reduction.
    pub durability_risk: Ppm,
    /// Expected movement cost.
    pub movement_cost: Ppm,
    /// Expected churn cost.
    pub churn: Ppm,
}

impl ExpectedBenefit {
    /// Creates a bounded expected-benefit vector.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        latency: u32,
        throughput: u32,
        resource: u32,
        memory: u32,
        storage: u32,
        network: u32,
        repair_debt: u32,
        durability_risk: u32,
        movement_cost: u32,
        churn: u32,
    ) -> Option<Self> {
        match (
            Ppm::new(latency),
            Ppm::new(throughput),
            Ppm::new(resource),
            Ppm::new(memory),
            Ppm::new(storage),
            Ppm::new(network),
            Ppm::new(repair_debt),
            Ppm::new(durability_risk),
            Ppm::new(movement_cost),
            Ppm::new(churn),
        ) {
            (
                Some(latency),
                Some(throughput),
                Some(resource),
                Some(memory),
                Some(storage),
                Some(network),
                Some(repair_debt),
                Some(durability_risk),
                Some(movement_cost),
                Some(churn),
            ) => Some(Self {
                latency,
                throughput,
                resource,
                memory,
                storage,
                network,
                repair_debt,
                durability_risk,
                movement_cost,
                churn,
            }),
            _ => None,
        }
    }

    /// Returns a zero expected-benefit vector.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            latency: Ppm::ZERO,
            throughput: Ppm::ZERO,
            resource: Ppm::ZERO,
            memory: Ppm::ZERO,
            storage: Ppm::ZERO,
            network: Ppm::ZERO,
            repair_debt: Ppm::ZERO,
            durability_risk: Ppm::ZERO,
            movement_cost: Ppm::ZERO,
            churn: Ppm::ZERO,
        }
    }
}

/// Resource and movement cost estimate for one action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct ActionCost {
    /// Expected CPU time.
    pub cpu_ns: u64,
    /// Expected storage I/O time.
    pub io_ns: u64,
    /// Expected network bytes.
    pub network_bytes: u64,
    /// Expected storage bytes.
    pub storage_bytes: u64,
    /// Expected temporary memory bytes.
    pub memory_bytes: u64,
}

impl ActionCost {
    /// Returns a zero cost estimate.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            cpu_ns: 0,
            io_ns: 0,
            network_bytes: 0,
            storage_bytes: 0,
            memory_bytes: 0,
        }
    }

    /// Adds two estimates without wrapping.
    #[must_use]
    pub const fn saturating_add(self, other: Self) -> Self {
        Self {
            cpu_ns: self.cpu_ns.saturating_add(other.cpu_ns),
            io_ns: self.io_ns.saturating_add(other.io_ns),
            network_bytes: self.network_bytes.saturating_add(other.network_bytes),
            storage_bytes: self.storage_bytes.saturating_add(other.storage_bytes),
            memory_bytes: self.memory_bytes.saturating_add(other.memory_bytes),
        }
    }
}

/// Result class returned by an actuator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionResult {
    /// The requested policy change was applied.
    Succeeded,
    /// The policy was already in the requested state.
    NoChange,
    /// The requested policy change was rolled back.
    RolledBack,
    /// The actuator rejected the request.
    Rejected,
    /// Execution failed.
    Failed,
}

impl ActionResult {
    /// Whether the result is a successful terminal outcome.
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Succeeded | Self::NoChange)
    }
}

/// Actual outcome and cost of executing an action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionOutcome {
    /// Terminal result class.
    pub result: ActionResult,
    /// Benefit observed after execution.
    pub observed_benefit: ExpectedBenefit,
    /// Execution duration supplied by the caller.
    pub duration: Duration,
    /// Resource cost paid by execution.
    pub cost: ActionCost,
}

impl ActionOutcome {
    /// Creates a successful outcome.
    #[must_use]
    pub const fn success(
        observed_benefit: ExpectedBenefit,
        duration: Duration,
        cost: ActionCost,
    ) -> Self {
        Self {
            result: ActionResult::Succeeded,
            observed_benefit,
            duration,
            cost,
        }
    }
    /// Creates a no-change outcome.
    #[must_use]
    pub const fn no_change(duration: Duration, cost: ActionCost) -> Self {
        Self {
            result: ActionResult::NoChange,
            observed_benefit: ExpectedBenefit::zero(),
            duration,
            cost,
        }
    }
    /// Creates a failed outcome.
    #[must_use]
    pub const fn failed(result: ActionResult, duration: Duration, cost: ActionCost) -> Self {
        Self {
            result,
            observed_benefit: ExpectedBenefit::zero(),
            duration,
            cost,
        }
    }
}

/// Cluster and namespace boundary for an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ActionScope {
    /// Cluster owning the action.
    pub cluster: ClusterId,
    /// Namespace owning the action.
    pub namespace: NamespaceId,
}

impl ActionScope {
    /// Creates an action scope.
    #[must_use]
    pub const fn new(cluster: ClusterId, namespace: NamespaceId) -> Self {
        Self { cluster, namespace }
    }
}

/// Failure while constructing a canonical tablet set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TargetSetError {
    /// A target set must not be empty.
    #[error("target set must not be empty")]
    Empty,
    /// A target set must not contain a duplicate tablet.
    #[error("target set contains a duplicate tablet")]
    Duplicate,
}

/// A canonical, bounded set of tablet targets.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TabletSet(Vec<TabletId>);

impl TabletSet {
    /// Creates a sorted, duplicate-free set.
    ///
    /// # Errors
    ///
    /// Returns [`TargetSetError`] for empty or duplicate input.
    pub fn new<I>(tablets: I) -> Result<Self, TargetSetError>
    where
        I: IntoIterator<Item = TabletId>,
    {
        let mut values: Vec<TabletId> = tablets.into_iter().collect();
        values.sort_by_key(|tablet| tablet.as_u64());
        if values.is_empty() {
            return Err(TargetSetError::Empty);
        }
        if values.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(TargetSetError::Duplicate);
        }
        Ok(Self(values))
    }
    /// Returns the canonical target slice.
    #[must_use]
    pub fn as_slice(&self) -> &[TabletId] {
        &self.0
    }
    /// Returns the target count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }
    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    /// Whether the set contains a tablet.
    #[must_use]
    pub fn contains(&self, tablet: TabletId) -> bool {
        self.0.contains(&tablet)
    }
    /// Iterates targets in canonical order.
    pub fn iter(&self) -> impl Iterator<Item = TabletId> + '_ {
        self.0.iter().copied()
    }
}

/// Typed target identity used in action keys and conflict checks.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActionTarget {
    /// One tablet.
    Tablet(TabletId),
    /// A canonical set of tablets.
    Tablets(TabletSet),
    /// A tablet move with explicit endpoints.
    Migration {
        /// Tablet being moved.
        tablet: TabletId,
        /// Replica source.
        from: NodeId,
        /// Replica destination.
        to: NodeId,
    },
}

/// Closed action kind used for deterministic conflict identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActionKind {
    /// Memory budget adjustment.
    AdjustMemoryBudget,
    /// Materialization intent adjustment.
    ChangeMaterializationIntent,
    /// Redundancy preference adjustment.
    ChangeRedundancyPreference,
    /// Repair budget adjustment.
    AdjustRepairBudget,
    /// Scrub budget adjustment.
    AdjustScrubBudget,
    /// Tablet split recommendation.
    RecommendTabletSplit,
    /// Tablet merge recommendation.
    RecommendTabletMerge,
    /// Migration recommendation.
    RecommendMigration,
}

/// Closed materialization intent controlled by the adaptive loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MaterializationIntent {
    /// Keep data resident and hot.
    Materialize,
    /// Keep data reconstructible but not resident.
    KeepWarm,
    /// Move data to a lower materialization tier.
    Demote,
    /// Evict data while retaining reconstructible representation.
    Evict,
}

/// Bounded replication factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReplicationFactor(u8);

impl ReplicationFactor {
    /// Creates a replication factor in the inclusive range 1..=16.
    #[must_use]
    pub const fn new(value: u8) -> Option<Self> {
        if value == 0 || value > 16 {
            None
        } else {
            Some(Self(value))
        }
    }
    /// Returns the replica count.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self.0
    }
}

/// Bounded erasure-coding layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ErasureCodingLayout {
    /// Number of data fragments.
    pub data_fragments: u8,
    /// Number of parity fragments.
    pub parity_fragments: u8,
}

impl ErasureCodingLayout {
    /// Creates a layout with nonzero fragment counts and at most 64 total.
    #[must_use]
    pub const fn new(data_fragments: u8, parity_fragments: u8) -> Option<Self> {
        if data_fragments == 0
            || parity_fragments == 0
            || data_fragments as u16 + parity_fragments as u16 > 64
        {
            None
        } else {
            Some(Self {
                data_fragments,
                parity_fragments,
            })
        }
    }
    /// Returns the protected fragment count.
    #[must_use]
    pub const fn fragment_count(self) -> u8 {
        self.data_fragments + self.parity_fragments
    }
}

/// Redundancy preference for an adaptive action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RedundancyPreference {
    /// Replicate the asset.
    Replication(ReplicationFactor),
    /// Use an erasure-coded representation.
    ErasureCoding(ErasureCodingLayout),
}

impl RedundancyPreference {
    /// Returns the protected fragment count.
    #[must_use]
    pub const fn fragment_count(self) -> u8 {
        match self {
            Self::Replication(factor) => factor.as_u8(),
            Self::ErasureCoding(layout) => layout.fragment_count(),
        }
    }
}

/// Bounded split recommendation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TabletSplitPolicy {
    /// Minimum logical bytes assigned to either child.
    pub min_child_bytes: u64,
    /// Maximum number of children.
    pub max_children: u8,
}

impl TabletSplitPolicy {
    /// Creates a split policy with 2..=16 children and a nonzero byte floor.
    #[must_use]
    pub const fn new(min_child_bytes: u64, max_children: u8) -> Option<Self> {
        if min_child_bytes == 0 || max_children < 2 || max_children > 16 {
            None
        } else {
            Some(Self {
                min_child_bytes,
                max_children,
            })
        }
    }
}

/// Bounded merge recommendation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TabletMergePolicy {
    /// Minimum combined logical bytes.
    pub min_combined_bytes: u64,
    /// Maximum number of source tablets.
    pub max_sources: u8,
}

impl TabletMergePolicy {
    /// Creates a merge policy with 2..=8 sources and a nonzero byte floor.
    #[must_use]
    pub const fn new(min_combined_bytes: u64, max_sources: u8) -> Option<Self> {
        if min_combined_bytes == 0 || max_sources < 2 || max_sources > 8 {
            None
        } else {
            Some(Self {
                min_combined_bytes,
                max_sources,
            })
        }
    }
}

/// Nonzero placement fence observed for a migration recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MigrationFence(u64);

impl MigrationFence {
    /// First valid placement fence.
    pub const INITIAL: Self = Self(1);
    /// Creates a nonzero migration fence.
    #[must_use]
    pub const fn new(value: u64) -> Option<Self> {
        if value == 0 { None } else { Some(Self(value)) }
    }
    /// Returns the raw fence.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// A bounded, fenced migration recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MigrationRecommendation {
    /// Replica source.
    pub from: NodeId,
    /// Replica destination.
    pub to: NodeId,
    /// Placement fence observed by the controller.
    pub fence: MigrationFence,
    /// Maximum bytes covered by this recommendation.
    pub max_bytes: u64,
}

impl MigrationRecommendation {
    /// Creates a recommendation with distinct endpoints and a bounded size.
    #[must_use]
    pub fn new(from: NodeId, to: NodeId, fence: MigrationFence, max_bytes: u64) -> Option<Self> {
        if from == to || max_bytes == 0 {
            None
        } else {
            Some(Self {
                from,
                to,
                fence,
                max_bytes,
            })
        }
    }
}

/// Policy values included in an action's exact identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PolicyFingerprint {
    /// Memory budget value.
    Memory(Ppm),
    /// Materialization intent.
    Materialization(MaterializationIntent),
    /// Redundancy preference.
    Redundancy(RedundancyPreference),
    /// Repair budget value.
    Repair(Ppm),
    /// Scrub budget value.
    Scrub(Ppm),
    /// Split policy.
    Split(TabletSplitPolicy),
    /// Merge policy.
    Merge(TabletMergePolicy),
    /// Migration recommendation.
    Migration {
        /// Replica source.
        from: NodeId,
        /// Replica destination.
        to: NodeId,
        /// Placement fence observed by the recommendation.
        fence: MigrationFence,
        /// Maximum bytes covered by the recommendation.
        max_bytes: u64,
    },
}

/// Exact deterministic identity of an action.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ActionKey {
    /// Action kind.
    pub kind: ActionKind,
    /// Namespace and cluster scope.
    pub scope: ActionScope,
    /// Canonical target identity.
    pub target: ActionTarget,
    /// Policy value fingerprint.
    pub policy: PolicyFingerprint,
}

/// Identity used to suppress competing policies for one target.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ActionConflictKey {
    /// Action kind.
    pub kind: ActionKind,
    /// Namespace and cluster scope.
    pub scope: ActionScope,
    /// Canonical target identity without policy values.
    pub target: ActionTarget,
}

/// Closed set of policy actions emitted by the controller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Tighten or widen the memory budget.
    AdjustMemoryBudget {
        /// Namespace and cluster scope.
        scope: ActionScope,
        /// Target tablet.
        target: TabletId,
        /// Requested memory budget.
        budget: MemoryBudget,
        /// Expected benefit.
        expected_benefit: ExpectedBenefit,
    },
    /// Change materialization intent.
    ChangeMaterializationIntent {
        /// Namespace and cluster scope.
        scope: ActionScope,
        /// Target tablet.
        target: TabletId,
        /// Requested materialization intent.
        intent: MaterializationIntent,
        /// Expected benefit.
        expected_benefit: ExpectedBenefit,
    },
    /// Change redundancy preference.
    ChangeRedundancyPreference {
        /// Namespace and cluster scope.
        scope: ActionScope,
        /// Target tablet.
        target: TabletId,
        /// Requested redundancy preference.
        preference: RedundancyPreference,
        /// Expected benefit.
        expected_benefit: ExpectedBenefit,
    },
    /// Adjust repair capacity.
    AdjustRepairBudget {
        /// Namespace and cluster scope.
        scope: ActionScope,
        /// Target tablet.
        target: TabletId,
        /// Requested repair budget.
        budget: RepairBudget,
        /// Expected benefit.
        expected_benefit: ExpectedBenefit,
    },
    /// Adjust scrub capacity.
    AdjustScrubBudget {
        /// Namespace and cluster scope.
        scope: ActionScope,
        /// Target tablet.
        target: TabletId,
        /// Requested scrub budget.
        budget: ScrubBudget,
        /// Expected benefit.
        expected_benefit: ExpectedBenefit,
    },
    /// Recommend a hotspot split.
    RecommendTabletSplit {
        /// Namespace and cluster scope.
        scope: ActionScope,
        /// Parent tablet.
        target: TabletId,
        /// Split policy.
        policy: TabletSplitPolicy,
        /// Expected benefit.
        expected_benefit: ExpectedBenefit,
    },
    /// Recommend a topology merge.
    RecommendTabletMerge {
        /// Namespace and cluster scope.
        scope: ActionScope,
        /// Source tablets.
        targets: TabletSet,
        /// Merge policy.
        policy: TabletMergePolicy,
        /// Expected benefit.
        expected_benefit: ExpectedBenefit,
    },
    /// Recommend a fenced migration.
    RecommendMigration {
        /// Namespace and cluster scope.
        scope: ActionScope,
        /// Tablet to move.
        target: TabletId,
        /// Migration recommendation.
        recommendation: MigrationRecommendation,
        /// Expected benefit.
        expected_benefit: ExpectedBenefit,
    },
}

impl Action {
    /// Returns the action kind.
    #[must_use]
    pub const fn kind(&self) -> ActionKind {
        match self {
            Self::AdjustMemoryBudget { .. } => ActionKind::AdjustMemoryBudget,
            Self::ChangeMaterializationIntent { .. } => ActionKind::ChangeMaterializationIntent,
            Self::ChangeRedundancyPreference { .. } => ActionKind::ChangeRedundancyPreference,
            Self::AdjustRepairBudget { .. } => ActionKind::AdjustRepairBudget,
            Self::AdjustScrubBudget { .. } => ActionKind::AdjustScrubBudget,
            Self::RecommendTabletSplit { .. } => ActionKind::RecommendTabletSplit,
            Self::RecommendTabletMerge { .. } => ActionKind::RecommendTabletMerge,
            Self::RecommendMigration { .. } => ActionKind::RecommendMigration,
        }
    }

    /// Returns the action scope.
    #[must_use]
    pub const fn scope(&self) -> ActionScope {
        match self {
            Self::AdjustMemoryBudget { scope, .. }
            | Self::ChangeMaterializationIntent { scope, .. }
            | Self::ChangeRedundancyPreference { scope, .. }
            | Self::AdjustRepairBudget { scope, .. }
            | Self::AdjustScrubBudget { scope, .. }
            | Self::RecommendTabletSplit { scope, .. }
            | Self::RecommendTabletMerge { scope, .. }
            | Self::RecommendMigration { scope, .. } => *scope,
        }
    }

    /// Returns the canonical target identity.
    #[must_use]
    pub fn target(&self) -> ActionTarget {
        match self {
            Self::AdjustMemoryBudget { target, .. }
            | Self::ChangeMaterializationIntent { target, .. }
            | Self::ChangeRedundancyPreference { target, .. }
            | Self::AdjustRepairBudget { target, .. }
            | Self::AdjustScrubBudget { target, .. }
            | Self::RecommendTabletSplit { target, .. } => ActionTarget::Tablet(*target),
            Self::RecommendTabletMerge { targets, .. } => ActionTarget::Tablets(targets.clone()),
            Self::RecommendMigration {
                target,
                recommendation,
                ..
            } => ActionTarget::Migration {
                tablet: *target,
                from: recommendation.from,
                to: recommendation.to,
            },
        }
    }

    /// Returns the exact deterministic action identity.
    #[must_use]
    pub fn key(&self) -> ActionKey {
        let policy = match self {
            Self::AdjustMemoryBudget { budget, .. } => PolicyFingerprint::Memory(budget.as_ppm()),
            Self::ChangeMaterializationIntent { intent, .. } => {
                PolicyFingerprint::Materialization(*intent)
            }
            Self::ChangeRedundancyPreference { preference, .. } => {
                PolicyFingerprint::Redundancy(*preference)
            }
            Self::AdjustRepairBudget { budget, .. } => PolicyFingerprint::Repair(budget.as_ppm()),
            Self::AdjustScrubBudget { budget, .. } => PolicyFingerprint::Scrub(budget.as_ppm()),
            Self::RecommendTabletSplit { policy, .. } => PolicyFingerprint::Split(*policy),
            Self::RecommendTabletMerge { policy, .. } => PolicyFingerprint::Merge(*policy),
            Self::RecommendMigration { recommendation, .. } => PolicyFingerprint::Migration {
                from: recommendation.from,
                to: recommendation.to,
                fence: recommendation.fence,
                max_bytes: recommendation.max_bytes,
            },
        };
        ActionKey {
            kind: self.kind(),
            scope: self.scope(),
            target: self.target(),
            policy,
        }
    }

    /// Returns the conflict identity used to suppress competing policies.
    #[must_use]
    pub fn conflict_key(&self) -> ActionConflictKey {
        ActionConflictKey {
            kind: self.kind(),
            scope: self.scope(),
            target: self.target(),
        }
    }

    /// Returns the expected benefit carried by the action.
    #[must_use]
    pub const fn expected_benefit(&self) -> ExpectedBenefit {
        match self {
            Self::AdjustMemoryBudget {
                expected_benefit, ..
            }
            | Self::ChangeMaterializationIntent {
                expected_benefit, ..
            }
            | Self::ChangeRedundancyPreference {
                expected_benefit, ..
            }
            | Self::AdjustRepairBudget {
                expected_benefit, ..
            }
            | Self::AdjustScrubBudget {
                expected_benefit, ..
            }
            | Self::RecommendTabletSplit {
                expected_benefit, ..
            }
            | Self::RecommendTabletMerge {
                expected_benefit, ..
            }
            | Self::RecommendMigration {
                expected_benefit, ..
            } => *expected_benefit,
        }
    }

    /// Returns the deterministic resource estimate used by safety checks.
    #[must_use]
    pub fn estimated_cost(&self) -> ActionCost {
        match self {
            Self::RecommendMigration { recommendation, .. } => ActionCost {
                cpu_ns: recommendation.max_bytes,
                io_ns: recommendation.max_bytes,
                network_bytes: recommendation.max_bytes,
                storage_bytes: recommendation.max_bytes,
                memory_bytes: 0,
            },
            Self::RecommendTabletSplit { .. } | Self::RecommendTabletMerge { .. } => ActionCost {
                cpu_ns: 1,
                io_ns: 1,
                network_bytes: 1,
                storage_bytes: 1,
                memory_bytes: 1,
            },
            _ => ActionCost::zero(),
        }
    }
}

/// A typed reason for proposing an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActionReason {
    /// Sustained memory pressure.
    SustainedMemoryPressure,
    /// Low pressure with execution-critical work.
    LowPressureCriticalPromotion,
    /// Sustained repair debt.
    SustainedRepairDebt,
    /// Foreground latency protection.
    ForegroundLatencyProtection,
    /// Persistent topology imbalance.
    PersistentTopologyImbalance,
    /// Many-key hotspot with memory or storage pressure.
    ManyKeyHotspot,
    /// Cold large assets.
    ColdLargeAsset,
    /// Hot small assets.
    HotSmallAsset,
    /// Safe learned alternative.
    LearnedAlternative,
}

/// Explicit action lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActionState {
    /// Proposal admitted to the ledger.
    Proposed,
    /// Actuator accepted the proposal.
    Accepted,
    /// Actuator is executing the proposal.
    Executing,
    /// Execution completed successfully or with no change.
    Completed,
    /// Safety or policy rejected the proposal.
    Rejected,
    /// Execution failed.
    Failed,
    /// A newer proposal replaced this one.
    Superseded,
}

impl ActionState {
    /// Whether no further transition is allowed.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Rejected | Self::Failed | Self::Superseded
        )
    }
    /// Whether a transition to `next` is legal.
    #[must_use]
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (
                Self::Proposed,
                Self::Accepted | Self::Rejected | Self::Superseded
            ) | (
                Self::Accepted,
                Self::Executing | Self::Rejected | Self::Superseded
            ) | (
                Self::Executing,
                Self::Completed | Self::Failed | Self::Superseded
            )
        )
    }
}

/// Controller identity attached to a decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ControllerSource {
    /// Deterministic baseline controller.
    Baseline,
    /// Learned controller running without actuation.
    LearnedShadow,
    /// Learned controller receiving outcome feedback.
    Learned,
}

/// Monotonic action identity assigned by the ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ActionId(u64);

impl ActionId {
    /// First action identity.
    pub const FIRST: Self = Self(1);
    /// Creates an action identity.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }
    /// Returns the raw identity.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// Observation identity captured with every action record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActionObservation {
    /// Latest observation sequence.
    pub sequence: ObservationSequence,
    /// Observation window used for the decision.
    pub window: ObservationWindowId,
    /// Observation or controller schema version.
    pub version: u64,
    /// Virtual decision timestamp.
    pub at: Ticks,
}

/// A complete bounded action record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionRecord {
    /// Ledger-assigned action identity.
    pub id: ActionId,
    /// Closed action payload.
    pub action: Action,
    /// Typed reason for the proposal.
    pub reason: ActionReason,
    /// Controller that proposed the action.
    pub source: ControllerSource,
    /// Current lifecycle state.
    pub state: ActionState,
    /// Observation identity used by the decision.
    pub observation: ActionObservation,
    /// Benefit expected before execution.
    pub expected_benefit: ExpectedBenefit,
    /// Benefit and cost observed after execution.
    pub actual_outcome: Option<ActionOutcome>,
    /// Action superseded by a newer action.
    pub superseded_by: Option<ActionId>,
    /// Action that superseded this action.
    pub supersedes: Option<ActionId>,
}

impl ActionRecord {
    /// Returns the exact action key.
    #[must_use]
    pub fn key(&self) -> ActionKey {
        self.action.key()
    }
    /// Returns the action conflict key.
    #[must_use]
    pub fn conflict_key(&self) -> ActionConflictKey {
        self.action.conflict_key()
    }
}

/// Trace status for one decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TraceStatus {
    /// A proposal was generated.
    Proposed,
    /// A proposal entered the active ledger.
    Accepted,
    /// A proposal was rejected by safety or policy.
    Rejected,
    /// A proposal was suppressed as duplicate, conflicting, or rate limited.
    Suppressed,
    /// A learned controller fell back to baseline behavior.
    Fallback,
    /// A learned action was evaluated without actuation.
    Shadow,
    /// A newer action replaced the traced action.
    Superseded,
}

/// One explainable decision trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionTrace {
    /// Observation sequence used by the decision.
    pub sequence: ObservationSequence,
    /// Observation window used by the decision.
    pub window: ObservationWindowId,
    /// Observation version used by the decision.
    pub version: u64,
    /// Controller source.
    pub source: ControllerSource,
    /// Trace status.
    pub status: TraceStatus,
    /// Typed reason for the decision.
    pub reason: ActionReason,
    /// Exact action key, when an action was generated.
    pub action_key: Option<ActionKey>,
    /// Safety violation, when validation rejected an action.
    pub safety_violation: Option<SafetyViolation>,
    /// Ledger suppression reason, when applicable.
    pub ledger_rejection: Option<LedgerRejection>,
    /// Whether baseline fallback was used.
    pub fallback: bool,
}

impl DecisionTrace {
    fn new(
        observation: ActionObservation,
        source: ControllerSource,
        status: TraceStatus,
        reason: ActionReason,
        action_key: Option<ActionKey>,
    ) -> Self {
        Self {
            sequence: observation.sequence,
            window: observation.window,
            version: observation.version,
            source,
            status,
            reason,
            action_key,
            safety_violation: None,
            ledger_rejection: None,
            fallback: false,
        }
    }
}

/// Action ledger retention and suppression configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionLedgerConfig {
    /// Maximum retained action records.
    pub max_history: usize,
    /// Maximum retained traces.
    pub max_traces: usize,
    /// Maximum actions admitted in one observation window.
    pub max_actions_per_window: u32,
    /// Observation windows for which a conflict key is suppressed.
    pub conflict_cooldown_windows: u64,
    /// Observation windows for which an exact key is suppressed.
    pub duplicate_cooldown_windows: u64,
}

impl Default for ActionLedgerConfig {
    fn default() -> Self {
        Self {
            max_history: 256,
            max_traces: 512,
            max_actions_per_window: 8,
            conflict_cooldown_windows: 2,
            duplicate_cooldown_windows: 8,
        }
    }
}

/// Why the action ledger refused a proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LedgerRejection {
    /// The exact action key is still active or within duplicate cooldown.
    #[error("duplicate action key")]
    Duplicate,
    /// Another policy for the same conflict key is active or cooling down.
    #[error("conflicting action key")]
    Conflict,
    /// The observation window action-rate ceiling was reached.
    #[error("action rate ceiling reached")]
    RateLimit,
    /// The action was rejected by the safety validator.
    #[error("action rejected by safety validator")]
    Safety,
    /// The requested action identifier is not retained.
    #[error("action is not retained by the ledger")]
    UnknownAction,
    /// The requested lifecycle transition is illegal.
    #[error("action lifecycle transition is illegal")]
    IllegalTransition,
    /// The ledger reached its action identifier space.
    #[error("action identifier space exhausted")]
    IdExhausted,
}

/// Bounded action history with duplicate, conflict, and rate suppression.
#[derive(Debug, Clone)]
pub struct ActionLedger {
    config: ActionLedgerConfig,
    records: VecDeque<ActionRecord>,
    traces: VecDeque<DecisionTrace>,
    next_id: u64,
}

impl Default for ActionLedger {
    fn default() -> Self {
        Self::new(ActionLedgerConfig::default())
    }
}

impl ActionLedger {
    /// Creates an empty bounded ledger.
    #[must_use]
    pub fn new(config: ActionLedgerConfig) -> Self {
        Self {
            config: ActionLedgerConfig {
                max_history: config.max_history.max(1),
                max_traces: config.max_traces.max(1),
                ..config
            },
            records: VecDeque::new(),
            traces: VecDeque::new(),
            next_id: ActionId::FIRST.as_u64(),
        }
    }
    /// Returns the ledger configuration.
    #[must_use]
    pub const fn config(&self) -> &ActionLedgerConfig {
        &self.config
    }
    /// Returns retained records in proposal order.
    pub fn records(&self) -> impl Iterator<Item = &ActionRecord> {
        self.records.iter()
    }
    /// Returns retained traces in decision order.
    pub fn traces(&self) -> impl Iterator<Item = &DecisionTrace> {
        self.traces.iter()
    }
    /// Finds one retained record.
    #[must_use]
    pub fn record(&self, id: ActionId) -> Option<&ActionRecord> {
        self.records.iter().find(|record| record.id == id)
    }
    /// Returns records that have not reached a terminal state.
    pub fn active_records(&self) -> impl Iterator<Item = &ActionRecord> {
        self.records
            .iter()
            .filter(|record| !record.state.is_terminal())
    }

    /// Admits a validated action to the bounded ledger.
    ///
    /// # Errors
    ///
    /// Returns [`LedgerRejection`] for duplicate, conflict, rate, or identifier failures.
    #[allow(clippy::too_many_lines)]
    pub fn admit(
        &mut self,
        action: Action,
        reason: ActionReason,
        source: ControllerSource,
        observation: ActionObservation,
    ) -> Result<ActionRecord, LedgerRejection> {
        let key = action.key();
        let conflict = action.conflict_key();
        let window = observation.window.as_u64();
        if self.records.iter().any(|record| {
            record.key() == key
                && (!record.state.is_terminal()
                    || Self::within(record, window, self.config.duplicate_cooldown_windows))
        }) {
            self.push_trace(DecisionTrace {
                ledger_rejection: Some(LedgerRejection::Duplicate),
                ..DecisionTrace::new(
                    observation,
                    source,
                    TraceStatus::Suppressed,
                    reason,
                    Some(key),
                )
            });
            return Err(LedgerRejection::Duplicate);
        }
        if self.records.iter().any(|record| {
            record.conflict_key() == conflict
                && (!record.state.is_terminal()
                    || Self::within(record, window, self.config.conflict_cooldown_windows))
        }) {
            self.push_trace(DecisionTrace {
                ledger_rejection: Some(LedgerRejection::Conflict),
                ..DecisionTrace::new(
                    observation,
                    source,
                    TraceStatus::Suppressed,
                    reason,
                    Some(key),
                )
            });
            return Err(LedgerRejection::Conflict);
        }
        let in_window = self
            .records
            .iter()
            .filter(|record| record.observation.window == observation.window)
            .count();
        if u32::try_from(in_window).unwrap_or(u32::MAX) >= self.config.max_actions_per_window {
            self.push_trace(DecisionTrace {
                ledger_rejection: Some(LedgerRejection::RateLimit),
                ..DecisionTrace::new(
                    observation,
                    source,
                    TraceStatus::Suppressed,
                    reason,
                    Some(key),
                )
            });
            return Err(LedgerRejection::RateLimit);
        }
        if self.records.len() >= self.config.max_history {
            let Some(oldest_terminal) = self
                .records
                .iter()
                .position(|record| record.state.is_terminal())
            else {
                self.push_trace(DecisionTrace {
                    ledger_rejection: Some(LedgerRejection::RateLimit),
                    ..DecisionTrace::new(
                        observation,
                        source,
                        TraceStatus::Suppressed,
                        reason,
                        Some(key),
                    )
                });
                return Err(LedgerRejection::RateLimit);
            };
            self.records.remove(oldest_terminal);
        }
        let Some(next_id) = self.next_id.checked_add(1) else {
            self.push_trace(DecisionTrace {
                ledger_rejection: Some(LedgerRejection::IdExhausted),
                ..DecisionTrace::new(
                    observation,
                    source,
                    TraceStatus::Suppressed,
                    reason,
                    Some(key),
                )
            });
            return Err(LedgerRejection::IdExhausted);
        };
        let record = ActionRecord {
            id: ActionId::from_u64(self.next_id),
            expected_benefit: action.expected_benefit(),
            action,
            reason,
            source,
            state: ActionState::Proposed,
            observation,
            actual_outcome: None,
            superseded_by: None,
            supersedes: None,
        };
        self.next_id = next_id;
        self.records.push_back(record.clone());
        while self.records.len() > self.config.max_history {
            self.records.pop_front();
        }
        self.push_trace(DecisionTrace::new(
            observation,
            source,
            TraceStatus::Accepted,
            reason,
            Some(key),
        ));
        Ok(record)
    }

    /// Transitions one retained action.
    ///
    /// # Errors
    ///
    /// Returns [`LedgerRejection`] for an unknown action or illegal transition.
    pub fn transition(
        &mut self,
        id: ActionId,
        next: ActionState,
        observation: ActionObservation,
        outcome: Option<ActionOutcome>,
    ) -> Result<ActionRecord, LedgerRejection> {
        let Some(record) = self.records.iter_mut().find(|record| record.id == id) else {
            return Err(LedgerRejection::UnknownAction);
        };
        if !record.state.can_transition_to(next) {
            return Err(LedgerRejection::IllegalTransition);
        }
        record.state = next;
        if outcome.is_some() {
            record.actual_outcome = outcome;
        }
        let result = record.clone();
        self.push_trace(DecisionTrace::new(
            observation,
            result.source,
            if next.is_terminal() {
                TraceStatus::Accepted
            } else {
                TraceStatus::Proposed
            },
            result.reason,
            Some(result.key()),
        ));
        Ok(result)
    }

    /// Records an execution outcome and moves the action to a terminal state.
    ///
    /// # Errors
    ///
    /// Returns [`LedgerRejection`] for an unknown or non-executing action.
    pub fn record_outcome(
        &mut self,
        id: ActionId,
        observation: ActionObservation,
        outcome: ActionOutcome,
    ) -> Result<ActionRecord, LedgerRejection> {
        let Some(record) = self.records.iter().find(|record| record.id == id) else {
            return Err(LedgerRejection::UnknownAction);
        };
        if record.state != ActionState::Executing {
            return Err(LedgerRejection::IllegalTransition);
        }
        let next = if outcome.result.is_success() {
            ActionState::Completed
        } else {
            ActionState::Failed
        };
        self.transition(id, next, observation, Some(outcome))
    }

    /// Supersedes an active action and links both records.
    ///
    /// # Errors
    ///
    /// Returns [`LedgerRejection`] when either action is missing or old is terminal.
    pub fn supersede(
        &mut self,
        old_id: ActionId,
        new_id: ActionId,
        observation: ActionObservation,
    ) -> Result<(), LedgerRejection> {
        if old_id == new_id {
            return Err(LedgerRejection::UnknownAction);
        }
        let Some(old) = self.records.iter().find(|record| record.id == old_id) else {
            return Err(LedgerRejection::UnknownAction);
        };
        let Some(new) = self.records.iter().find(|record| record.id == new_id) else {
            return Err(LedgerRejection::UnknownAction);
        };
        if old.state.is_terminal() || new.state.is_terminal() {
            return Err(LedgerRejection::IllegalTransition);
        }
        let source = old.source;
        let reason = old.reason;
        let old_index = self
            .records
            .iter()
            .position(|record| record.id == old_id)
            .ok_or(LedgerRejection::UnknownAction)?;
        let new_index = self
            .records
            .iter()
            .position(|record| record.id == new_id)
            .ok_or(LedgerRejection::UnknownAction)?;
        let old = &mut self.records[old_index];
        old.state = ActionState::Superseded;
        old.superseded_by = Some(new_id);
        let new = &mut self.records[new_index];
        new.supersedes = Some(old_id);
        self.push_trace(DecisionTrace::new(
            observation,
            source,
            TraceStatus::Superseded,
            reason,
            None,
        ));
        Ok(())
    }

    fn within(record: &ActionRecord, window: u64, windows: u64) -> bool {
        let recorded = record.observation.window.as_u64();
        window >= recorded && window - recorded <= windows
    }

    fn push_trace(&mut self, trace: DecisionTrace) {
        self.traces.push_back(trace);
        while self.traces.len() > self.config.max_traces {
            self.traces.pop_front();
        }
    }
}

/// Raw and normalized objective dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ObjectiveBreakdown {
    /// Latency improvement or regression.
    pub latency: i64,
    /// Throughput improvement or regression.
    pub throughput: i64,
    /// General resource improvement or regression.
    pub resource: i64,
    /// Memory improvement or regression.
    pub memory: i64,
    /// Storage improvement or regression.
    pub storage: i64,
    /// Network improvement or regression.
    pub network: i64,
    /// Repair-debt improvement or regression.
    pub repair_debt: i64,
    /// Durability-risk improvement or regression.
    pub durability_risk: i64,
    /// Movement cost.
    pub movement_cost: i64,
    /// Churn cost.
    pub churn: i64,
}

impl ObjectiveBreakdown {
    /// Returns all-zero dimensions.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            latency: 0,
            throughput: 0,
            resource: 0,
            memory: 0,
            storage: 0,
            network: 0,
            repair_debt: 0,
            durability_risk: 0,
            movement_cost: 0,
            churn: 0,
        }
    }
    /// Returns dimensions normalized to signed parts per million.
    #[must_use]
    pub fn normalized(self, scales: ObjectiveScales) -> Self {
        Self {
            latency: normalize(self.latency, scales.latency),
            throughput: normalize(self.throughput, scales.throughput),
            resource: normalize(self.resource, scales.resource),
            memory: normalize(self.memory, scales.memory),
            storage: normalize(self.storage, scales.storage),
            network: normalize(self.network, scales.network),
            repair_debt: normalize(self.repair_debt, scales.repair_debt),
            durability_risk: normalize(self.durability_risk, scales.durability_risk),
            movement_cost: normalize(self.movement_cost, scales.movement_cost),
            churn: normalize(self.churn, scales.churn),
        }
    }
}

/// Per-dimension normalization ceilings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectiveScales {
    /// Latency scale.
    pub latency: u64,
    /// Throughput scale.
    pub throughput: u64,
    /// Resource scale.
    pub resource: u64,
    /// Memory scale.
    pub memory: u64,
    /// Storage scale.
    pub storage: u64,
    /// Network scale.
    pub network: u64,
    /// Repair-debt scale.
    pub repair_debt: u64,
    /// Durability-risk scale.
    pub durability_risk: u64,
    /// Movement-cost scale.
    pub movement_cost: u64,
    /// Churn scale.
    pub churn: u64,
}

impl ObjectiveScales {
    /// Returns one scale for every objective dimension.
    #[must_use]
    pub const fn uniform(value: u64) -> Self {
        Self {
            latency: value,
            throughput: value,
            resource: value,
            memory: value,
            storage: value,
            network: value,
            repair_debt: value,
            durability_risk: value,
            movement_cost: value,
            churn: value,
        }
    }
    /// Whether every scale is nonzero.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.latency != 0
            && self.throughput != 0
            && self.resource != 0
            && self.memory != 0
            && self.storage != 0
            && self.network != 0
            && self.repair_debt != 0
            && self.durability_risk != 0
            && self.movement_cost != 0
            && self.churn != 0
    }
}

impl Default for ObjectiveScales {
    fn default() -> Self {
        Self::uniform(1_000_000)
    }
}

/// Explicit objective weights in parts per million.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectiveWeights {
    /// Latency weight.
    pub latency: u32,
    /// Throughput weight.
    pub throughput: u32,
    /// Resource weight.
    pub resource: u32,
    /// Memory weight.
    pub memory: u32,
    /// Storage weight.
    pub storage: u32,
    /// Network weight.
    pub network: u32,
    /// Repair-debt weight.
    pub repair_debt: u32,
    /// Durability-risk weight.
    pub durability_risk: u32,
    /// Movement-cost weight.
    pub movement_cost: u32,
    /// Churn weight.
    pub churn: u32,
}

impl ObjectiveWeights {
    /// Returns equal weights for all dimensions.
    #[must_use]
    pub const fn equal() -> Self {
        Self {
            latency: 100_000,
            throughput: 100_000,
            resource: 100_000,
            memory: 100_000,
            storage: 100_000,
            network: 100_000,
            repair_debt: 100_000,
            durability_risk: 100_000,
            movement_cost: 100_000,
            churn: 100_000,
        }
    }
    /// Sum of all weights.
    #[must_use]
    pub const fn total(self) -> u64 {
        self.latency as u64
            + self.throughput as u64
            + self.resource as u64
            + self.memory as u64
            + self.storage as u64
            + self.network as u64
            + self.repair_debt as u64
            + self.durability_risk as u64
            + self.movement_cost as u64
            + self.churn as u64
    }
    /// Whether the weight vector has at least one nonzero dimension.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.total() != 0
    }
}

impl Default for ObjectiveWeights {
    fn default() -> Self {
        Self::equal()
    }
}

/// A normalized, weighted multi-objective value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectiveVector {
    /// Raw signed dimensions.
    pub raw: ObjectiveBreakdown,
    /// Explicitly normalized signed dimensions.
    pub normalized: ObjectiveBreakdown,
    /// Explicit weights used for scalarization.
    pub weights: ObjectiveWeights,
}

impl ObjectiveVector {
    /// Creates a vector after validating scales and weights.
    #[must_use]
    pub fn new(
        raw: ObjectiveBreakdown,
        scales: ObjectiveScales,
        weights: ObjectiveWeights,
    ) -> Option<Self> {
        if !scales.is_valid() || !weights.is_valid() {
            return None;
        }
        Some(Self {
            raw,
            normalized: raw.normalized(scales),
            weights,
        })
    }
    /// Computes a scalar only after explicit normalization and weighting.
    #[must_use]
    pub fn scalar_score(self) -> i64 {
        let value = i128::from(self.normalized.latency) * i128::from(self.weights.latency)
            + i128::from(self.normalized.throughput) * i128::from(self.weights.throughput)
            + i128::from(self.normalized.resource) * i128::from(self.weights.resource)
            + i128::from(self.normalized.memory) * i128::from(self.weights.memory)
            + i128::from(self.normalized.storage) * i128::from(self.weights.storage)
            + i128::from(self.normalized.network) * i128::from(self.weights.network)
            + i128::from(self.normalized.repair_debt) * i128::from(self.weights.repair_debt)
            + i128::from(self.normalized.durability_risk)
                * i128::from(self.weights.durability_risk)
            + i128::from(self.normalized.movement_cost) * i128::from(self.weights.movement_cost)
            + i128::from(self.normalized.churn) * i128::from(self.weights.churn);
        i64::try_from(value / i128::from(PPM_SCALE)).unwrap_or(if value < 0 {
            i64::MIN
        } else {
            i64::MAX
        })
    }
}

fn normalize(value: i64, scale: u64) -> i64 {
    if scale == 0 {
        return 0;
    }
    let scaled = i128::from(value).saturating_mul(i128::from(PPM_SCALE)) / i128::from(scale);
    let limit = i64::try_from(PPM_SCALE).unwrap_or(i64::MAX);
    let bounded = scaled.clamp(-i128::from(limit), i128::from(limit));
    i64::try_from(bounded).unwrap_or(if bounded < 0 { i64::MIN } else { i64::MAX })
}

/// Typed hard-constraint failure from the safety validator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SafetyViolation {
    /// Action scope did not match the supplied control scope.
    #[error("action scope does not match control scope")]
    ScopeMismatch,
    /// Confidence was below the configured minimum.
    #[error("observation confidence is below the configured minimum")]
    LowConfidence,
    /// Observation was stale or from the future.
    #[error("observation freshness does not satisfy the safety envelope")]
    StaleObservation,
    /// Requested redundancy was below the minimum.
    #[error("requested redundancy is below the minimum")]
    RedundancyBelowMinimum,
    /// Requested placement did not preserve failure-domain independence.
    #[error("failure-domain independence requirement is not satisfied")]
    FailureDomainIndependence,
    /// Migration fencing was not valid in the current context.
    #[error("migration fencing is invalid")]
    MigrationFenceInvalid,
    /// Migration recommendation carried a stale fence.
    #[error("migration recommendation carries a stale fence")]
    MigrationFenceStale,
    /// Demotion or eviction would discard reconstructible representation.
    #[error("demotion or eviction requires reconstructible representation")]
    UnrepresentableDemotion,
    /// Repair reserve was reduced below the mandatory critical reserve.
    #[error("critical repair reserve would be reduced")]
    CriticalRepairReserve,
    /// Critical repair work is starving and the action does not increase repair.
    #[error("critical repair work is starving")]
    RepairStarvation,
    /// A split or merge violated a topology invariant.
    #[error("topology invariant violated: {0:?}")]
    TopologyInvariant(TopologyViolation),
    /// A live topology plan already owns a target.
    #[error("an active topology plan already owns the target")]
    ActiveTopologyPlan,
    /// The action-rate ceiling was reached.
    #[error("action-rate ceiling reached")]
    ActionRateCeiling,
    /// The action resource estimate exceeded the ceiling.
    #[error("action resource estimate exceeds the ceiling")]
    ResourceCeiling,
    /// A migration recommendation had identical endpoints.
    #[error("migration source and destination are identical")]
    NoMigration,
}

/// Typed topology invariant failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TopologyViolation {
    /// Parent was below the split byte floor.
    #[error("split parent is too small: {bytes} < {minimum}")]
    SplitTooSmall {
        /// Observed logical bytes.
        bytes: u64,
        /// Required minimum.
        minimum: u64,
    },
    /// Split would create too many children.
    #[error("split child count exceeds maximum")]
    SplitTooMany,
    /// Merge did not contain enough sources.
    #[error("merge requires at least two distinct tablets")]
    MergeNeedsTwo,
    /// Merge contained too many sources.
    #[error("merge source count exceeds maximum")]
    MergeTooMany,
    /// Merge was below the byte floor.
    #[error("merge sources are too small: {bytes} < {minimum}")]
    MergeTooSmall {
        /// Observed combined logical bytes.
        bytes: u64,
        /// Required minimum.
        minimum: u64,
    },
    /// Merging would violate the minimum tablet count.
    #[error("merge would violate the minimum tablet count")]
    BelowMinimumTablets,
}

const fn default_repair_budget() -> RepairBudget {
    match RepairBudget::new(250_000) {
        Some(value) => value,
        None => RepairBudget(Ppm::ZERO),
    }
}

/// Hard safety limits applied to every controller path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafetyEnvelope {
    /// Minimum protected fragment count.
    pub minimum_redundancy: u8,
    /// Minimum independent failure domains.
    pub minimum_failure_domains: u8,
    /// Mandatory critical repair reserve.
    pub minimum_critical_repair_reserve: RepairBudget,
    /// Maximum active actions in one observation window.
    pub maximum_actions_per_window: u32,
    /// Maximum aggregate action cost in one observation window.
    pub maximum_resource_cost: ActionCost,
    /// Maximum children created by one split recommendation.
    pub maximum_split_children: u8,
    /// Minimum parent bytes for a split recommendation.
    pub minimum_split_bytes: u64,
    /// Minimum combined bytes for a merge recommendation.
    pub minimum_merge_bytes: u64,
    /// Minimum acceptable observation confidence.
    pub minimum_confidence: Confidence,
    /// Maximum observation age.
    pub maximum_freshness_age: Duration,
    /// Age at which critical repair must be prioritized.
    pub repair_starvation_age: Duration,
    /// Minimum tablet count that a merge may not cross.
    pub minimum_tablet_count: u32,
    /// Whether migration recommendations require a valid current fence.
    pub require_migration_fencing: bool,
}

impl Default for SafetyEnvelope {
    fn default() -> Self {
        Self {
            minimum_redundancy: 2,
            minimum_failure_domains: 2,
            minimum_critical_repair_reserve: default_repair_budget(),
            maximum_actions_per_window: 8,
            maximum_resource_cost: ActionCost {
                cpu_ns: 1_000_000_000,
                io_ns: 1_000_000_000,
                network_bytes: 1_000_000_000,
                storage_bytes: 1_000_000_000,
                memory_bytes: 1_000_000_000,
            },
            maximum_split_children: 16,
            minimum_split_bytes: 1,
            minimum_merge_bytes: 1,
            minimum_confidence: Confidence::Medium,
            maximum_freshness_age: Duration::from_secs(5),
            repair_starvation_age: Duration::from_secs(30),
            minimum_tablet_count: 1,
            require_migration_fencing: true,
        }
    }
}

impl SafetyEnvelope {
    /// Whether the envelope has internally valid hard limits.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.minimum_redundancy > 0
            && self.minimum_failure_domains > 0
            && self.maximum_actions_per_window > 0
            && self.maximum_split_children >= 2
            && self.minimum_split_bytes > 0
            && self.minimum_merge_bytes > 0
    }
}

/// Current control state used by the deterministic safety validator.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyContext {
    /// Namespace and cluster boundary.
    pub scope: ActionScope,
    /// Primary tablet under consideration.
    pub primary_target: TabletId,
    /// Current logical bytes for the primary target.
    pub target_logical_bytes: u64,
    /// Current combined bytes for a merge candidate.
    pub target_combined_bytes: u64,
    /// Current protected replica count.
    pub current_replicas: u8,
    /// Current number of independent failure domains.
    pub current_failure_domains: u8,
    /// Current placement fence.
    pub current_fence: MigrationFence,
    /// Whether fencing is currently valid.
    pub migration_fencing_valid: bool,
    /// Whether lower tiers can reconstruct the representation.
    pub representation_reconstructible: bool,
    /// Current critical repair reserve.
    pub critical_repair_reserve: RepairBudget,
    /// Current repair budget.
    pub repair_budget: RepairBudget,
    /// Current scrub budget.
    pub scrub_budget: ScrubBudget,
    /// Current memory budget.
    pub memory_budget: MemoryBudget,
    /// Current materialization intent.
    pub materialization: MaterializationIntent,
    /// Age of the oldest critical repair item.
    pub repair_debt_age: Duration,
    /// Whether critical repair debt exists.
    pub critical_repair_debt: bool,
    /// Tablets already covered by live topology plans.
    pub active_topology_tablets: BTreeSet<TabletId>,
    /// Current number of tablets.
    pub tablet_count: u32,
    /// Candidate migration recommendations.
    pub migration_candidates: Vec<MigrationRecommendation>,
    /// Candidate merge tablet sets and their combined bytes.
    pub merge_candidates: Vec<(TabletSet, u64)>,
    /// Current action count in this observation window.
    pub actions_in_window: u32,
    /// Aggregate resource cost in this observation window.
    pub resource_cost_in_window: ActionCost,
    /// Whether foreground latency is currently unhealthy.
    pub foreground_latency_bad: bool,
    /// Current observation confidence.
    pub confidence: Confidence,
    /// Current observation freshness classification.
    pub freshness: FreshnessStatus,
    /// Age of the newest observation used for this decision.
    pub freshness_age: Duration,
}

impl SafetyContext {
    /// Creates a fail-safe context for one target.
    #[must_use]
    pub fn new(scope: ActionScope, primary_target: TabletId) -> Self {
        Self {
            scope,
            primary_target,
            target_logical_bytes: 1,
            target_combined_bytes: 1,
            current_replicas: 2,
            current_failure_domains: 2,
            current_fence: MigrationFence::INITIAL,
            migration_fencing_valid: true,
            representation_reconstructible: true,
            critical_repair_reserve: default_repair_budget(),
            repair_budget: RepairBudget::new(500_000).unwrap_or(RepairBudget(Ppm::ZERO)),
            scrub_budget: ScrubBudget::new(500_000).unwrap_or(ScrubBudget(Ppm::ZERO)),
            memory_budget: MemoryBudget::new(500_000).unwrap_or(MemoryBudget(Ppm::ZERO)),
            materialization: MaterializationIntent::Materialize,
            repair_debt_age: Duration::ZERO,
            critical_repair_debt: false,
            active_topology_tablets: BTreeSet::new(),
            tablet_count: 2,
            migration_candidates: Vec::new(),
            merge_candidates: Vec::new(),
            actions_in_window: 0,
            resource_cost_in_window: ActionCost::zero(),
            foreground_latency_bad: false,
            confidence: Confidence::High,
            freshness: FreshnessStatus::Current,
            freshness_age: Duration::ZERO,
        }
    }
}

/// Deterministic hard-constraint validator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafetyValidator {
    envelope: SafetyEnvelope,
}

impl Default for SafetyValidator {
    fn default() -> Self {
        Self::new(SafetyEnvelope::default())
    }
}

impl SafetyValidator {
    /// Creates a validator with the supplied hard limits.
    #[must_use]
    pub const fn new(envelope: SafetyEnvelope) -> Self {
        Self { envelope }
    }
    /// Returns the immutable safety envelope.
    #[must_use]
    pub const fn envelope(&self) -> &SafetyEnvelope {
        &self.envelope
    }
    /// Validates one action against the current context.
    ///
    /// # Errors
    ///
    /// Returns a typed [`SafetyViolation`] when any hard constraint fails.
    #[allow(clippy::too_many_lines, clippy::if_not_else)]
    pub fn validate(
        &self,
        action: &Action,
        context: &SafetyContext,
    ) -> Result<(), SafetyViolation> {
        if action.scope() != context.scope {
            return Err(SafetyViolation::ScopeMismatch);
        }
        if context.confidence < self.envelope.minimum_confidence {
            return Err(SafetyViolation::LowConfidence);
        }
        if context.freshness != FreshnessStatus::Current
            || context.freshness_age > self.envelope.maximum_freshness_age
        {
            return Err(SafetyViolation::StaleObservation);
        }
        if context.actions_in_window >= self.envelope.maximum_actions_per_window {
            return Err(SafetyViolation::ActionRateCeiling);
        }
        if exceeds_cost(
            context
                .resource_cost_in_window
                .saturating_add(action.estimated_cost()),
            self.envelope.maximum_resource_cost,
        ) {
            return Err(SafetyViolation::ResourceCeiling);
        }
        match action {
            Action::ChangeMaterializationIntent { intent, .. }
                if matches!(
                    intent,
                    MaterializationIntent::Demote | MaterializationIntent::Evict
                ) && !context.representation_reconstructible =>
            {
                Err(SafetyViolation::UnrepresentableDemotion)
            }
            Action::ChangeRedundancyPreference { preference, .. } => {
                if preference.fragment_count() < self.envelope.minimum_redundancy {
                    Err(SafetyViolation::RedundancyBelowMinimum)
                } else if context.current_failure_domains < self.envelope.minimum_failure_domains {
                    Err(SafetyViolation::FailureDomainIndependence)
                } else {
                    Ok(())
                }
            }
            Action::AdjustRepairBudget { budget, .. } => {
                let reserve_floor = self
                    .envelope
                    .minimum_critical_repair_reserve
                    .as_ppm()
                    .max(context.critical_repair_reserve.as_ppm());
                if budget.as_ppm() < reserve_floor {
                    Err(SafetyViolation::CriticalRepairReserve)
                } else if (context.critical_repair_debt
                    || context.repair_debt_age >= self.envelope.repair_starvation_age)
                    && budget.as_ppm() <= context.repair_budget.as_ppm()
                {
                    Err(SafetyViolation::RepairStarvation)
                } else {
                    Ok(())
                }
            }
            Action::RecommendMigration { recommendation, .. } => {
                if context.current_replicas < self.envelope.minimum_redundancy {
                    Err(SafetyViolation::RedundancyBelowMinimum)
                } else if context.current_failure_domains < self.envelope.minimum_failure_domains {
                    Err(SafetyViolation::FailureDomainIndependence)
                } else if recommendation.from == recommendation.to {
                    Err(SafetyViolation::NoMigration)
                } else if !context
                    .migration_candidates
                    .iter()
                    .any(|candidate| candidate == recommendation)
                {
                    Err(SafetyViolation::MigrationFenceStale)
                } else if self.envelope.require_migration_fencing
                    && (!context.migration_fencing_valid
                        || recommendation.fence != context.current_fence)
                {
                    if !context.migration_fencing_valid {
                        Err(SafetyViolation::MigrationFenceInvalid)
                    } else {
                        Err(SafetyViolation::MigrationFenceStale)
                    }
                } else {
                    Ok(())
                }
            }
            Action::RecommendTabletSplit { target, policy, .. } => {
                if context.current_replicas < self.envelope.minimum_redundancy {
                    Err(SafetyViolation::RedundancyBelowMinimum)
                } else if context.current_failure_domains < self.envelope.minimum_failure_domains {
                    Err(SafetyViolation::FailureDomainIndependence)
                } else if *target != context.primary_target
                    || context.active_topology_tablets.contains(target)
                {
                    Err(SafetyViolation::ActiveTopologyPlan)
                } else if policy.max_children > self.envelope.maximum_split_children {
                    Err(SafetyViolation::TopologyInvariant(
                        TopologyViolation::SplitTooMany,
                    ))
                } else if context.target_logical_bytes < policy.min_child_bytes
                    || context.target_logical_bytes < self.envelope.minimum_split_bytes
                {
                    Err(SafetyViolation::TopologyInvariant(
                        TopologyViolation::SplitTooSmall {
                            bytes: context.target_logical_bytes,
                            minimum: policy
                                .min_child_bytes
                                .max(self.envelope.minimum_split_bytes),
                        },
                    ))
                } else {
                    Ok(())
                }
            }
            Action::RecommendTabletMerge {
                targets, policy, ..
            } => {
                if context.current_replicas < self.envelope.minimum_redundancy {
                    Err(SafetyViolation::RedundancyBelowMinimum)
                } else if context.current_failure_domains < self.envelope.minimum_failure_domains {
                    Err(SafetyViolation::FailureDomainIndependence)
                } else if targets
                    .iter()
                    .any(|tablet| context.active_topology_tablets.contains(&tablet))
                {
                    Err(SafetyViolation::ActiveTopologyPlan)
                } else if targets.len() < 2
                    || !context
                        .merge_candidates
                        .iter()
                        .any(|(candidate, _)| candidate == targets)
                {
                    Err(SafetyViolation::TopologyInvariant(
                        TopologyViolation::MergeNeedsTwo,
                    ))
                } else if targets.len() > usize::from(policy.max_sources) {
                    Err(SafetyViolation::TopologyInvariant(
                        TopologyViolation::MergeTooMany,
                    ))
                } else if context.target_combined_bytes < policy.min_combined_bytes
                    || context.target_combined_bytes < self.envelope.minimum_merge_bytes
                {
                    Err(SafetyViolation::TopologyInvariant(
                        TopologyViolation::MergeTooSmall {
                            bytes: context.target_combined_bytes,
                            minimum: policy
                                .min_combined_bytes
                                .max(self.envelope.minimum_merge_bytes),
                        },
                    ))
                } else if context.tablet_count <= self.envelope.minimum_tablet_count {
                    Err(SafetyViolation::TopologyInvariant(
                        TopologyViolation::BelowMinimumTablets,
                    ))
                } else {
                    Ok(())
                }
            }
            _ => Ok(()),
        }
    }
}

fn exceeds_cost(candidate: ActionCost, limit: ActionCost) -> bool {
    candidate.cpu_ns > limit.cpu_ns
        || candidate.io_ns > limit.io_ns
        || candidate.network_bytes > limit.network_bytes
        || candidate.storage_bytes > limit.storage_bytes
        || candidate.memory_bytes > limit.memory_bytes
}

/// Input frame containing one typed observation snapshot and decision identity.
#[derive(Debug, Clone, PartialEq)]
pub struct ControllerFrame {
    /// Typed observation snapshot.
    pub observation: ObservationSnapshot,
    /// Namespace and cluster scope.
    pub scope: ActionScope,
    /// Sequence used for this decision.
    pub sequence: ObservationSequence,
    /// Window used for this decision.
    pub window: ObservationWindowId,
    /// Observation or controller schema version.
    pub version: u64,
    /// Virtual decision timestamp.
    pub at: Ticks,
}

impl ControllerFrame {
    /// Creates a decision frame from typed observation data.
    #[must_use]
    pub const fn new(
        observation: ObservationSnapshot,
        scope: ActionScope,
        sequence: ObservationSequence,
        window: ObservationWindowId,
        version: u64,
        at: Ticks,
    ) -> Self {
        Self {
            observation,
            scope,
            sequence,
            window,
            version,
            at,
        }
    }
    /// Returns the snapshot for a typed tablet scope when present.
    #[must_use]
    pub fn tablet_snapshot(&self, tablet: TabletId) -> Option<&ScopeSnapshot> {
        self.observation
            .scopes
            .iter()
            .find(|snapshot| snapshot.scope == ObservationScope::Tablet(tablet))
    }
    /// Returns the target snapshot or the first retained scope.
    #[must_use]
    pub fn target_snapshot(&self, target: TabletId) -> Option<&ScopeSnapshot> {
        self.tablet_snapshot(target)
            .or_else(|| self.observation.scopes.first())
    }
    /// Returns the action observation identity.
    #[must_use]
    pub const fn action_observation(&self) -> ActionObservation {
        ActionObservation {
            sequence: self.sequence,
            window: self.window,
            version: self.version,
            at: self.at,
        }
    }
}

/// Integer and bounded features extracted from a typed observation scope.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObservationFeatures {
    /// Mean memory pressure in parts per million.
    pub memory_pressure_ppm: u32,
    /// Mean execution-criticality score.
    pub criticality: u64,
    /// Number of repair assets.
    pub repair_assets: u64,
    /// Repair debt bytes.
    pub repair_bytes: u64,
    /// Age of the oldest repair item.
    pub repair_age: Duration,
    /// Scrub debt bytes.
    pub scrub_bytes: u64,
    /// End-to-end p99 latency.
    pub latency_p99_ns: u64,
    /// Requests per second truncated to an integer.
    pub throughput: u64,
    /// Resident memory bytes.
    pub resident_bytes: u64,
    /// Logical memory bytes.
    pub logical_bytes: u64,
    /// Storage read bytes.
    pub storage_read_bytes: u64,
    /// Storage written bytes.
    pub storage_written_bytes: u64,
    /// Maximum worker imbalance in parts per million.
    pub worker_imbalance_ppm: u32,
    /// Maximum tablet imbalance in parts per million.
    pub tablet_imbalance_ppm: u32,
    /// Number of retained hot keys.
    pub hot_key_count: usize,
    /// Top key share in parts per million.
    pub top_key_share_ppm: u32,
    /// Top ten key share in parts per million.
    pub top_ten_key_share_ppm: u32,
    /// Whether one key dominates the sketch.
    pub strict_single_hot_key: bool,
    /// Whether many keys contribute meaningful traffic.
    pub many_key_hotspot: bool,
    /// Whether the memory footprint is hot.
    pub memory_hotspot: bool,
    /// Whether storage traffic is hot.
    pub storage_hotspot: bool,
    /// Whether retained assets are cold and large.
    pub cold_large_assets: bool,
    /// Whether retained assets are hot and small.
    pub hot_small_assets: bool,
}

impl ObservationFeatures {
    /// Extracts bounded features from a typed observation scope.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn from_snapshot(snapshot: &ScopeSnapshot, config: &BaselineConfig) -> Self {
        let pressure = snapshot
            .memory_pressure
            .map_or(0.0, |summary| summary.mean_pressure.as_f64() * 1_000_000.0)
            .clamp(0.0, 1_000_000.0) as u32;
        let criticality = snapshot
            .execution_criticality
            .map_or(0, |summary| summary.mean_score.as_u64());
        let (repair_assets, repair_bytes, repair_age) =
            snapshot
                .maintenance_debt
                .map_or((0, 0, Duration::ZERO), |summary| {
                    (
                        summary.repair.assets,
                        summary.repair.bytes,
                        summary.repair.oldest_age,
                    )
                });
        let scrub_bytes = snapshot
            .maintenance_debt
            .map_or(0, |summary| summary.scrub.bytes);
        let latency_p99_ns = snapshot.latency.map_or(0, |summary| summary.p99_ns);
        let throughput = snapshot
            .request
            .map_or(0, |summary| summary.requests_per_second.max(0.0) as u64);
        let logical_bytes = snapshot.memory.map_or(0, |summary| summary.logical_bytes);
        let resident_bytes = snapshot.memory.map_or(0, |summary| summary.resident_bytes);
        let storage_read_bytes = snapshot.io.map_or(0, |summary| summary.storage.read);
        let storage_written_bytes = snapshot.io.map_or(0, |summary| summary.storage.written);
        let worker_imbalance_ppm = snapshot
            .topology
            .and_then(|summary| summary.worker_max_to_mean)
            .map_or(0, |value| (value.max(0.0) * 1_000_000.0) as u32);
        let tablet_imbalance_ppm = snapshot
            .topology
            .and_then(|summary| summary.tablet_max_to_mean)
            .map_or(0, |value| (value.max(0.0) * 1_000_000.0) as u32);
        let entries = &snapshot.hot_keys.entries;
        let top_key_share_ppm = entries.first().map_or(0, |entry| {
            (entry.share.clamp(0.0, 1.0) * 1_000_000.0) as u32
        });
        let top_ten_key_share_ppm = (snapshot
            .hot_keys
            .concentration
            .top_ten_share
            .clamp(0.0, 1.0)
            * 1_000_000.0) as u32;
        let strict_single_hot_key =
            entries.len() == 1 && top_key_share_ppm >= config.strict_single_key_share_ppm;
        let many_key_hotspot = entries.len() >= config.many_hot_keys
            && top_ten_key_share_ppm >= config.many_hot_key_share_ppm;
        let memory_pressure_ppm = pressure;
        let memory_hotspot = resident_bytes >= config.memory_hotspot_bytes
            || memory_pressure_ppm >= config.memory_pressure_high_ppm;
        let storage_hotspot = storage_read_bytes.saturating_add(storage_written_bytes)
            >= config.storage_hotspot_bytes;
        let cold_large_assets = logical_bytes >= config.large_asset_bytes
            && entries
                .iter()
                .all(|entry| entry.classification.hotness != HotnessClass::Hot);
        let hot_small_assets = logical_bytes <= config.small_asset_bytes
            && entries
                .iter()
                .any(|entry| entry.classification.hotness == HotnessClass::Hot);
        Self {
            memory_pressure_ppm,
            criticality,
            repair_assets,
            repair_bytes,
            repair_age,
            scrub_bytes,
            latency_p99_ns,
            throughput,
            resident_bytes,
            logical_bytes,
            storage_read_bytes,
            storage_written_bytes,
            worker_imbalance_ppm,
            tablet_imbalance_ppm,
            hot_key_count: entries.len(),
            top_key_share_ppm,
            top_ten_key_share_ppm,
            strict_single_hot_key,
            many_key_hotspot,
            memory_hotspot,
            storage_hotspot,
            cold_large_assets,
            hot_small_assets,
        }
    }
}

/// Baseline controller tuning values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BaselineConfig {
    /// Consecutive windows required for a condition.
    pub sustained_windows: u16,
    /// Maximum retained sustained-window states.
    pub max_tracked_windows: u16,
    /// Minimum retained sample count.
    pub min_samples: usize,
    /// Minimum observation confidence.
    pub minimum_confidence: Confidence,
    /// Memory pressure enter threshold.
    pub memory_pressure_high_ppm: u32,
    /// Memory pressure release threshold.
    pub memory_pressure_low_ppm: u32,
    /// Hysteresis width in parts per million.
    pub hysteresis_ppm: u32,
    /// Criticality threshold used for promotion.
    pub criticality_threshold: u64,
    /// Repair debt byte threshold.
    pub repair_debt_bytes: u64,
    /// Repair debt age threshold.
    pub repair_debt_age: Duration,
    /// Foreground p99 latency threshold.
    pub latency_p99_ns: u64,
    /// Topology imbalance threshold.
    pub topology_imbalance_ppm: u32,
    /// Number of hot keys required for a many-key hotspot.
    pub many_hot_keys: usize,
    /// Many-key share threshold.
    pub many_hot_key_share_ppm: u32,
    /// Dominant single-key share threshold.
    pub strict_single_key_share_ppm: u32,
    /// Memory hotspot byte threshold.
    pub memory_hotspot_bytes: u64,
    /// Storage hotspot byte threshold.
    pub storage_hotspot_bytes: u64,
    /// Large-asset threshold.
    pub large_asset_bytes: u64,
    /// Small-asset threshold.
    pub small_asset_bytes: u64,
    /// Memory budget step.
    pub memory_step: Ppm,
    /// Repair budget step.
    pub repair_step: Ppm,
    /// Scrub budget step.
    pub scrub_step: Ppm,
    /// Initial memory budget.
    pub initial_memory_budget: MemoryBudget,
    /// Initial repair budget.
    pub initial_repair_budget: RepairBudget,
    /// Initial scrub budget.
    pub initial_scrub_budget: ScrubBudget,
    /// Maximum actions admitted in one observation window.
    pub max_actions_per_window: u32,
    /// Maximum aggregate action cost in one observation window.
    pub maximum_resource_cost: ActionCost,
    /// Number of windows between actions for one target.
    pub action_cooldown_windows: u64,
}

impl Default for BaselineConfig {
    fn default() -> Self {
        Self {
            sustained_windows: 3,
            max_tracked_windows: 32,
            min_samples: 3,
            minimum_confidence: Confidence::Medium,
            memory_pressure_high_ppm: 750_000,
            memory_pressure_low_ppm: 250_000,
            hysteresis_ppm: 100_000,
            criticality_threshold: 100,
            repair_debt_bytes: 1_000_000,
            repair_debt_age: Duration::from_secs(5),
            latency_p99_ns: 1_000_000,
            topology_imbalance_ppm: 2_000_000,
            many_hot_keys: 2,
            many_hot_key_share_ppm: 500_000,
            strict_single_key_share_ppm: 900_000,
            memory_hotspot_bytes: 1_000_000,
            storage_hotspot_bytes: 1_000_000,
            large_asset_bytes: 1_000_000,
            small_asset_bytes: 1_000_000,
            memory_step: Ppm::new(100_000).unwrap_or(Ppm::ZERO),
            repair_step: Ppm::new(100_000).unwrap_or(Ppm::ZERO),
            scrub_step: Ppm::new(100_000).unwrap_or(Ppm::ZERO),
            initial_memory_budget: MemoryBudget::new(500_000).unwrap_or(MemoryBudget(Ppm::ZERO)),
            initial_repair_budget: RepairBudget::new(500_000).unwrap_or(RepairBudget(Ppm::ZERO)),
            initial_scrub_budget: ScrubBudget::new(500_000).unwrap_or(ScrubBudget(Ppm::ZERO)),
            max_actions_per_window: 8,
            maximum_resource_cost: ActionCost {
                cpu_ns: 1_000_000_000,
                io_ns: 1_000_000_000,
                network_bytes: 1_000_000_000,
                storage_bytes: 1_000_000_000,
                memory_bytes: 1_000_000_000,
            },
            action_cooldown_windows: 2,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct WindowSignals {
    memory_high: u16,
    memory_low: u16,
    criticality: u16,
    repair_debt: u16,
    latency_bad: u16,
    imbalance: u16,
    many_hotspot: u16,
    cold_large: u16,
    hot_small: u16,
}

#[derive(Debug, Clone)]
struct SustainedWindows {
    states: BTreeMap<ObservationWindowId, WindowSignals>,
    order: VecDeque<ObservationWindowId>,
    last_window: Option<ObservationWindowId>,
    capacity: usize,
}

impl SustainedWindows {
    fn new(capacity: usize) -> Self {
        Self {
            states: BTreeMap::new(),
            order: VecDeque::new(),
            last_window: None,
            capacity: capacity.max(1),
        }
    }

    fn observe(
        &mut self,
        window: ObservationWindowId,
        features: &ObservationFeatures,
        config: &BaselineConfig,
    ) {
        if self.last_window == Some(window) {
            return;
        }
        if self.last_window.is_some_and(|last| window <= last) {
            self.states.clear();
            self.order.clear();
        }
        let mut state = self
            .last_window
            .and_then(|last| self.states.get(&last))
            .copied()
            .unwrap_or_default();
        update_high(
            &mut state.memory_high,
            features.memory_pressure_ppm,
            config.memory_pressure_high_ppm,
            config.hysteresis_ppm,
        );
        update_low(
            &mut state.memory_low,
            features.memory_pressure_ppm,
            config.memory_pressure_low_ppm,
            config.hysteresis_ppm,
        );
        update_count(
            &mut state.criticality,
            features.criticality >= config.criticality_threshold,
        );
        update_count(
            &mut state.repair_debt,
            features.repair_bytes >= config.repair_debt_bytes
                || features.repair_age >= config.repair_debt_age,
        );
        update_count(
            &mut state.latency_bad,
            features.latency_p99_ns >= config.latency_p99_ns,
        );
        update_count(
            &mut state.imbalance,
            features.worker_imbalance_ppm >= config.topology_imbalance_ppm
                || features.tablet_imbalance_ppm >= config.topology_imbalance_ppm,
        );
        update_count(&mut state.many_hotspot, features.many_key_hotspot);
        update_count(&mut state.cold_large, features.cold_large_assets);
        update_count(&mut state.hot_small, features.hot_small_assets);
        self.states.insert(window, state);
        self.order.push_back(window);
        while self.order.len() > self.capacity {
            if let Some(expired) = self.order.pop_front() {
                self.states.remove(&expired);
            }
        }
        self.last_window = Some(window);
    }

    fn sustained(&self, value: fn(&WindowSignals) -> u16, windows: u16) -> bool {
        self.last_window
            .and_then(|window| self.states.get(&window))
            .is_some_and(|state| value(state) >= windows.max(1))
    }
}

fn update_count(value: &mut u16, condition: bool) {
    if condition {
        *value = value.saturating_add(1);
    } else {
        *value = 0;
    }
}

fn update_high(value: &mut u16, observed: u32, enter: u32, hysteresis: u32) {
    let release = enter.saturating_sub(hysteresis);
    update_count(
        value,
        if *value > 0 {
            observed >= release
        } else {
            observed >= enter
        },
    );
}

fn update_low(value: &mut u16, observed: u32, enter: u32, hysteresis: u32) {
    let release = enter.saturating_add(hysteresis).min(1_000_000);
    update_count(
        value,
        if *value > 0 {
            observed <= release
        } else {
            observed <= enter
        },
    );
}

/// A baseline proposal and its explanation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaselineProposal {
    /// Always the deterministic baseline source.
    pub source: ControllerSource,
    /// Validated actions in priority order.
    pub actions: Vec<Action>,
    /// Typed safety rejections encountered while forming the proposal.
    pub violations: Vec<SafetyViolation>,
    /// Whether fallback behavior was used.
    pub fallback: bool,
    /// Explanation for the selected branch.
    pub trace: DecisionTrace,
}

/// Deterministic observation-driven baseline controller.
#[derive(Debug, Clone)]
pub struct BaselineController {
    config: BaselineConfig,
    sustained: SustainedWindows,
    memory_budget: MemoryBudget,
    repair_budget: RepairBudget,
    scrub_budget: ScrubBudget,
    materialization: MaterializationIntent,
    last_action_window: BTreeMap<ActionConflictKey, ObservationWindowId>,
}

impl Default for BaselineController {
    fn default() -> Self {
        Self::new(BaselineConfig::default())
    }
}

impl BaselineController {
    /// Creates a deterministic baseline controller.
    #[must_use]
    pub fn new(config: BaselineConfig) -> Self {
        Self {
            sustained: SustainedWindows::new(config.max_tracked_windows as usize),
            memory_budget: config.initial_memory_budget,
            repair_budget: config.initial_repair_budget,
            scrub_budget: config.initial_scrub_budget,
            materialization: MaterializationIntent::Materialize,
            last_action_window: BTreeMap::new(),
            config,
        }
    }
    /// Returns the controller configuration.
    #[must_use]
    pub const fn config(&self) -> &BaselineConfig {
        &self.config
    }
    /// Returns the current bounded policy state.
    #[must_use]
    pub const fn policy_state(
        &self,
    ) -> (
        MemoryBudget,
        RepairBudget,
        ScrubBudget,
        MaterializationIntent,
    ) {
        (
            self.memory_budget,
            self.repair_budget,
            self.scrub_budget,
            self.materialization,
        )
    }

    /// Proposes validated baseline actions for one observation frame.
    ///
    /// Every candidate branch invokes [`SafetyValidator::validate`].
    #[must_use]
    #[allow(clippy::too_many_lines, clippy::if_not_else, clippy::collapsible_if)]
    pub fn propose(
        &mut self,
        frame: &ControllerFrame,
        context: &SafetyContext,
        validator: &SafetyValidator,
    ) -> BaselineProposal {
        let observation = frame.action_observation();
        let Some(snapshot) = frame.target_snapshot(context.primary_target) else {
            return BaselineProposal {
                source: ControllerSource::Baseline,
                actions: Vec::new(),
                violations: Vec::new(),
                fallback: true,
                trace: DecisionTrace::new(
                    observation,
                    ControllerSource::Baseline,
                    TraceStatus::Fallback,
                    ActionReason::SustainedMemoryPressure,
                    None,
                ),
            };
        };
        let features = ObservationFeatures::from_snapshot(snapshot, &self.config);
        if snapshot.sample_count < self.config.min_samples
            || snapshot.evidence.confidence < self.config.minimum_confidence
        {
            return BaselineProposal {
                source: ControllerSource::Baseline,
                actions: Vec::new(),
                violations: Vec::new(),
                fallback: true,
                trace: DecisionTrace::new(
                    observation,
                    ControllerSource::Baseline,
                    TraceStatus::Fallback,
                    ActionReason::SustainedMemoryPressure,
                    None,
                ),
            };
        }
        self.sustained
            .observe(frame.window, &features, &self.config);
        let mut actions = Vec::new();
        let mut violations = Vec::new();
        let mut reason = ActionReason::SustainedMemoryPressure;
        let memory_high = self
            .sustained
            .sustained(|state| state.memory_high, self.config.sustained_windows);
        let memory_low = self
            .sustained
            .sustained(|state| state.memory_low, self.config.sustained_windows);
        let criticality = self
            .sustained
            .sustained(|state| state.criticality, self.config.sustained_windows);
        let repair_debt = self
            .sustained
            .sustained(|state| state.repair_debt, self.config.sustained_windows);
        let latency_bad = self
            .sustained
            .sustained(|state| state.latency_bad, self.config.sustained_windows);
        let imbalance = self
            .sustained
            .sustained(|state| state.imbalance, self.config.sustained_windows);
        let many_hotspot = self
            .sustained
            .sustained(|state| state.many_hotspot, self.config.sustained_windows);
        let cold_large = self
            .sustained
            .sustained(|state| state.cold_large, self.config.sustained_windows);
        let hot_small = self
            .sustained
            .sustained(|state| state.hot_small, self.config.sustained_windows);

        if memory_high {
            reason = ActionReason::SustainedMemoryPressure;
            if self.materialization != MaterializationIntent::Evict {
                let intent = if features.memory_pressure_ppm >= 900_000 {
                    MaterializationIntent::Evict
                } else {
                    MaterializationIntent::Demote
                };
                let action = Action::ChangeMaterializationIntent {
                    scope: context.scope,
                    target: context.primary_target,
                    intent,
                    expected_benefit: expected(500_000, 0, 0, 300_000, 0, 0, 0, 100_000, 0, 20_000),
                };
                self.push_validated(
                    &mut actions,
                    &mut violations,
                    action,
                    reason,
                    context,
                    validator,
                    observation,
                );
            }
            let action = Action::AdjustMemoryBudget {
                scope: context.scope,
                target: context.primary_target,
                budget: self.memory_budget.saturating_sub(self.config.memory_step),
                expected_benefit: expected(100_000, 0, 0, 500_000, 0, 0, 0, 100_000, 0, 20_000),
            };
            self.push_validated(
                &mut actions,
                &mut violations,
                action,
                reason,
                context,
                validator,
                observation,
            );
        } else if memory_low && criticality {
            reason = ActionReason::LowPressureCriticalPromotion;
            if self.materialization != MaterializationIntent::Materialize {
                let action = Action::ChangeMaterializationIntent {
                    scope: context.scope,
                    target: context.primary_target,
                    intent: MaterializationIntent::Materialize,
                    expected_benefit: expected(500_000, 200_000, 0, 0, 0, 0, 0, 100_000, 0, 20_000),
                };
                self.push_validated(
                    &mut actions,
                    &mut violations,
                    action,
                    reason,
                    context,
                    validator,
                    observation,
                );
            } else {
                let action = Action::AdjustMemoryBudget {
                    scope: context.scope,
                    target: context.primary_target,
                    budget: self.memory_budget.saturating_add(self.config.memory_step),
                    expected_benefit: expected(500_000, 200_000, 0, 0, 0, 0, 0, 100_000, 0, 20_000),
                };
                self.push_validated(
                    &mut actions,
                    &mut violations,
                    action,
                    reason,
                    context,
                    validator,
                    observation,
                );
            }
        } else if latency_bad {
            reason = ActionReason::ForegroundLatencyProtection;
            let action = Action::AdjustScrubBudget {
                scope: context.scope,
                target: context.primary_target,
                budget: self.scrub_budget.saturating_sub(self.config.scrub_step),
                expected_benefit: expected(500_000, 0, 0, 0, 0, 0, 0, 100_000, 0, 50_000),
            };
            self.push_validated(
                &mut actions,
                &mut violations,
                action,
                reason,
                context,
                validator,
                observation,
            );
        } else if repair_debt {
            reason = ActionReason::SustainedRepairDebt;
            let action = if context.foreground_latency_bad {
                Action::AdjustScrubBudget {
                    scope: context.scope,
                    target: context.primary_target,
                    budget: self.scrub_budget.saturating_sub(self.config.scrub_step),
                    expected_benefit: expected(500_000, 0, 0, 0, 0, 0, 0, 100_000, 0, 50_000),
                }
            } else {
                Action::AdjustRepairBudget {
                    scope: context.scope,
                    target: context.primary_target,
                    budget: self.repair_budget.saturating_add(self.config.repair_step),
                    expected_benefit: expected(0, 0, 0, 0, 0, 0, 500_000, 200_000, 0, 20_000),
                }
            };
            self.push_validated(
                &mut actions,
                &mut violations,
                action,
                reason,
                context,
                validator,
                observation,
            );
        } else if imbalance {
            reason = ActionReason::PersistentTopologyImbalance;
            if let Some(recommendation) = context.migration_candidates.first().copied() {
                let action = Action::RecommendMigration {
                    scope: context.scope,
                    target: context.primary_target,
                    recommendation,
                    expected_benefit: expected(
                        200_000, 300_000, 0, 0, 0, 200_000, 0, 100_000, 300_000, 100_000,
                    ),
                };
                self.push_validated(
                    &mut actions,
                    &mut violations,
                    action,
                    reason,
                    context,
                    validator,
                    observation,
                );
            } else if let Some((targets, _)) = context.merge_candidates.first().cloned() {
                if let Some(policy) = TabletMergePolicy::new(self.config.large_asset_bytes, 2) {
                    let action = Action::RecommendTabletMerge {
                        scope: context.scope,
                        targets,
                        policy,
                        expected_benefit: expected(
                            0, 200_000, 100_000, 100_000, 100_000, 100_000, 0, 100_000, 200_000,
                            100_000,
                        ),
                    };
                    self.push_validated(
                        &mut actions,
                        &mut violations,
                        action,
                        reason,
                        context,
                        validator,
                        observation,
                    );
                }
            }
        } else if many_hotspot && (features.memory_hotspot || features.storage_hotspot) {
            reason = ActionReason::ManyKeyHotspot;
            if !features.strict_single_hot_key {
                if let Some(policy) = TabletSplitPolicy::new(self.config.large_asset_bytes / 2, 2) {
                    let action = Action::RecommendTabletSplit {
                        scope: context.scope,
                        target: context.primary_target,
                        policy,
                        expected_benefit: expected(
                            500_000, 300_000, 0, 0, 0, 0, 0, 100_000, 200_000, 100_000,
                        ),
                    };
                    self.push_validated(
                        &mut actions,
                        &mut violations,
                        action,
                        reason,
                        context,
                        validator,
                        observation,
                    );
                }
            }
        } else if cold_large {
            reason = ActionReason::ColdLargeAsset;
            if let Some(layout) = ErasureCodingLayout::new(4, 2) {
                let preference = RedundancyPreference::ErasureCoding(layout);
                let action = Action::ChangeRedundancyPreference {
                    scope: context.scope,
                    target: context.primary_target,
                    preference,
                    expected_benefit: expected(
                        0, 0, 100_000, 0, 500_000, 100_000, 0, 100_000, 0, 20_000,
                    ),
                };
                self.push_validated(
                    &mut actions,
                    &mut violations,
                    action,
                    reason,
                    context,
                    validator,
                    observation,
                );
            }
        } else if hot_small {
            reason = ActionReason::HotSmallAsset;
            if let Some(factor) = ReplicationFactor::new(3) {
                let preference = RedundancyPreference::Replication(factor);
                let action = Action::ChangeRedundancyPreference {
                    scope: context.scope,
                    target: context.primary_target,
                    preference,
                    expected_benefit: expected(200_000, 500_000, 0, 0, 0, 0, 0, 100_000, 0, 20_000),
                };
                self.push_validated(
                    &mut actions,
                    &mut violations,
                    action,
                    reason,
                    context,
                    validator,
                    observation,
                );
            }
        }
        for action in &actions {
            self.last_action_window
                .insert(action.conflict_key(), frame.window);
        }
        while self.last_action_window.len() > self.config.max_tracked_windows as usize {
            let Some(first) = self.last_action_window.keys().next().cloned() else {
                break;
            };
            self.last_action_window.remove(&first);
        }
        let mut trace = DecisionTrace::new(
            observation,
            ControllerSource::Baseline,
            if actions.is_empty() {
                TraceStatus::Fallback
            } else {
                TraceStatus::Proposed
            },
            reason,
            actions.first().map(Action::key),
        );
        trace.safety_violation = violations.first().copied();
        let fallback = actions.is_empty();
        BaselineProposal {
            source: ControllerSource::Baseline,
            actions,
            violations,
            fallback,
            trace,
        }
    }

    /// Updates local policy state after an actuator outcome.
    pub fn on_outcome(&mut self, action: &Action, outcome: &ActionOutcome) {
        if !outcome.result.is_success() {
            self.rollback_step(action);
            return;
        }
        match action {
            Action::AdjustMemoryBudget { budget, .. } => self.memory_budget = *budget,
            Action::ChangeMaterializationIntent { intent, .. } => self.materialization = *intent,
            Action::AdjustRepairBudget { budget, .. } => self.repair_budget = *budget,
            Action::AdjustScrubBudget { budget, .. } => self.scrub_budget = *budget,
            _ => {}
        }
    }

    #[allow(clippy::semicolon_if_nothing_returned)]
    fn rollback_step(&mut self, action: &Action) {
        match action {
            Action::AdjustMemoryBudget { budget, .. } => {
                self.memory_budget = budget.saturating_add(self.config.memory_step)
            }
            Action::AdjustRepairBudget { budget, .. } => {
                self.repair_budget = budget.saturating_sub(self.config.repair_step)
            }
            Action::AdjustScrubBudget { budget, .. } => {
                self.scrub_budget = budget.saturating_add(self.config.scrub_step)
            }
            _ => {}
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push_validated(
        &mut self,
        actions: &mut Vec<Action>,
        violations: &mut Vec<SafetyViolation>,
        action: Action,
        _reason: ActionReason,
        context: &SafetyContext,
        validator: &SafetyValidator,
        observation: ActionObservation,
    ) -> bool {
        let proposed_count = context
            .actions_in_window
            .saturating_add(u32::try_from(actions.len()).unwrap_or(u32::MAX))
            .saturating_add(1);
        if proposed_count > self.config.max_actions_per_window {
            violations.push(SafetyViolation::ActionRateCeiling);
            return false;
        }
        let mut proposed_cost = context.resource_cost_in_window;
        for existing in actions.iter() {
            proposed_cost = proposed_cost.saturating_add(existing.estimated_cost());
        }
        if exceeds_cost(
            proposed_cost.saturating_add(action.estimated_cost()),
            self.config.maximum_resource_cost,
        ) {
            violations.push(SafetyViolation::ResourceCeiling);
            return false;
        }
        if let Err(violation) = validator.validate(&action, context) {
            violations.push(violation);
            return false;
        }
        let conflict = action.conflict_key();
        if self.last_action_window.get(&conflict).is_some_and(|last| {
            observation.window.as_u64().saturating_sub(last.as_u64())
                < self.config.action_cooldown_windows
        }) {
            return false;
        }
        actions.push(action);
        true
    }
}

#[allow(clippy::too_many_arguments)]
fn expected(
    latency: u32,
    throughput: u32,
    resource: u32,
    memory: u32,
    storage: u32,
    network: u32,
    repair_debt: u32,
    durability_risk: u32,
    movement_cost: u32,
    churn: u32,
) -> ExpectedBenefit {
    ExpectedBenefit::new(
        latency,
        throughput,
        resource,
        memory,
        storage,
        network,
        repair_debt,
        durability_risk,
        movement_cost,
        churn,
    )
    .unwrap_or(ExpectedBenefit::zero())
}

/// Learned-controller tuning and persistence limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LearnedConfig {
    /// Minimum samples before a learned choice can leave fallback mode.
    pub minimum_samples: u64,
    /// Maximum normalized uncertainty before falling back.
    pub maximum_uncertainty_ppm: u32,
    /// Bounded exploration step in policy parts per million.
    pub exploration_step: Ppm,
    /// Number of deterministic alternatives considered.
    pub alternative_count: usize,
    /// Online update rate in parts per million.
    pub learning_rate_ppm: u32,
    /// Maximum retained model samples.
    pub maximum_samples: u64,
}

impl Default for LearnedConfig {
    fn default() -> Self {
        Self {
            minimum_samples: 32,
            maximum_uncertainty_ppm: 250_000,
            exploration_step: Ppm::new(50_000).unwrap_or(Ppm::ZERO),
            alternative_count: 3,
            learning_rate_ppm: 10_000,
            maximum_samples: 1_000_000,
        }
    }
}

/// Learned model disable reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelDisableReason {
    /// Too few valid samples.
    InsufficientSamples,
    /// Uncertainty exceeded the configured threshold.
    HighUncertainty,
    /// Persisted data was corrupt.
    CorruptModel,
    /// Persisted data used an incompatible format version.
    IncompatibleModel,
    /// A caller explicitly disabled the model.
    ExplicitlyDisabled,
}

/// Result of loading a versioned learned model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelLoadStatus {
    /// Valid model was loaded.
    Loaded,
    /// Empty input retained the current model.
    Empty,
    /// Unsupported version retained the current model.
    Incompatible,
    /// Corrupt input retained the current model.
    Corrupt,
}

/// Report returned by non-blocking model loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelLoadReport {
    /// Load classification.
    pub status: ModelLoadStatus,
    /// Whether the model is disabled after loading.
    pub disabled: bool,
}

/// Learned model decode failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ModelDecodeError {
    /// Input was shorter than the fixed header or checksum.
    #[error("learned model input is truncated")]
    Truncated,
    /// Input did not start with the model magic.
    #[error("learned model magic is invalid")]
    BadMagic,
    /// Input used an unsupported model version.
    #[error("learned model version {version} is incompatible")]
    IncompatibleVersion {
        /// Unsupported model version.
        version: u16,
    },
    /// Input exceeded the bounded persistence size.
    #[error("learned model input exceeds the size limit")]
    TooLarge,
    /// Input checksum did not match its payload.
    #[error("learned model checksum is invalid")]
    BadChecksum,
    /// Input dimensions or values were invalid.
    #[error("learned model payload is malformed")]
    Malformed,
}

/// A small contextual linear model with explicit uncertainty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LearnedModel {
    weights: Vec<i32>,
    bias: i32,
    samples: u64,
    residual: u64,
    version: u16,
}

impl Default for LearnedModel {
    fn default() -> Self {
        Self {
            weights: vec![0; MODEL_FEATURE_COUNT],
            bias: 0,
            samples: 0,
            residual: 0,
            version: 1,
        }
    }
}

impl LearnedModel {
    /// Creates a deterministic zero model.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Returns the number of observations used by the model.
    #[must_use]
    pub const fn samples(&self) -> u64 {
        self.samples
    }
    /// Returns the model version.
    #[must_use]
    pub const fn version(&self) -> u16 {
        self.version
    }
    /// Returns normalized uncertainty in parts per million.
    #[must_use]
    pub fn uncertainty_ppm(&self) -> u32 {
        if self.samples == 0 {
            return 1_000_000;
        }
        let sample_term = 1_000_000_u64.saturating_div(self.samples.max(1));
        let residual_term = self.residual.saturating_mul(1_000_000).saturating_div(
            self.samples
                .max(1)
                .saturating_mul(self.samples.max(1))
                .max(4),
        );
        u32::try_from(sample_term.saturating_add(residual_term).min(1_000_000)).unwrap_or(1_000_000)
    }
    /// Scores a normalized contextual feature vector.
    #[must_use]
    pub fn score(&self, features: &[i32]) -> i64 {
        self.weights
            .iter()
            .zip(features)
            .fold(0_i64, |score, (weight, feature)| {
                score.saturating_add(i64::from(*weight) * i64::from(*feature))
            })
            .saturating_add(i64::from(self.bias))
    }

    fn update(&mut self, features: &[i32], target: i32, rate_ppm: u32, maximum_samples: u64) {
        if features.len() != self.weights.len() {
            return;
        }
        let mut weights = self.weights.clone();
        for index in 0..features.len() {
            let score = weights
                .iter()
                .zip(features)
                .fold(0_i64, |score, (weight, feature)| {
                    score.saturating_add(i64::from(*weight) * i64::from(*feature))
                })
                .saturating_add(i64::from(self.bias));
            let error = i64::from(target).saturating_sub(score);
            let delta = (i128::from(error)
                .saturating_mul(i128::from(features[index]))
                .saturating_mul(i128::from(rate_ppm)))
                / i128::from(PPM_SCALE);
            let delta = i32::try_from(delta).unwrap_or(if delta < 0 { i32::MIN } else { i32::MAX });
            weights[index] = weights[index].saturating_add(delta);
        }
        self.weights = weights;
        let error = i64::from(target).saturating_sub(self.score(features));
        self.residual = self.residual.saturating_add(error.unsigned_abs());
        self.samples = self.samples.saturating_add(1).min(maximum_samples.max(1));
    }

    /// Encodes a versioned, checksummed model image.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut payload = Vec::with_capacity(4 + 2 + 4 + MODEL_FEATURE_COUNT * 4 + 4 + 8 + 8 + 8);
        payload.extend_from_slice(b"KVLM");
        payload.extend_from_slice(&self.version.to_le_bytes());
        payload.extend_from_slice(
            &u32::try_from(self.weights.len())
                .unwrap_or(u32::MAX)
                .to_le_bytes(),
        );
        for weight in &self.weights {
            payload.extend_from_slice(&weight.to_le_bytes());
        }
        payload.extend_from_slice(&self.bias.to_le_bytes());
        payload.extend_from_slice(&self.samples.to_le_bytes());
        payload.extend_from_slice(&self.residual.to_le_bytes());
        let checksum = model_checksum(&payload);
        payload.extend_from_slice(&checksum.to_le_bytes());
        payload
    }

    /// Decodes a versioned model image without changing global state.
    ///
    /// # Errors
    ///
    /// Returns [`ModelDecodeError`] for incompatible versions, malformed payloads, size violations, or checksum failures.
    pub fn decode(input: &[u8]) -> Result<Self, ModelDecodeError> {
        if input.len() > MAX_MODEL_BYTES {
            return Err(ModelDecodeError::TooLarge);
        }
        if input.len() < 4 + 2 + 4 + MODEL_FEATURE_COUNT * 4 + 4 + 8 + 8 + 8 {
            return Err(ModelDecodeError::Truncated);
        }
        if &input[..4] != b"KVLM" {
            return Err(ModelDecodeError::BadMagic);
        }
        let version = u16::from_le_bytes(input[4..6].try_into().unwrap_or([0; 2]));
        if version != 1 {
            return Err(ModelDecodeError::IncompatibleVersion { version });
        }
        let checksum_offset = input.len() - 8;
        let expected_checksum =
            u64::from_le_bytes(input[checksum_offset..].try_into().unwrap_or([0; 8]));
        if model_checksum(&input[..checksum_offset]) != expected_checksum {
            return Err(ModelDecodeError::BadChecksum);
        }
        let count = u32::from_le_bytes(input[6..10].try_into().unwrap_or([0; 4])) as usize;
        let expected_len = 10usize
            .saturating_add(count.saturating_mul(4))
            .saturating_add(4)
            .saturating_add(8)
            .saturating_add(8)
            .saturating_add(8);
        if count != MODEL_FEATURE_COUNT || input.len() != expected_len {
            return Err(ModelDecodeError::Malformed);
        }
        let mut weights = Vec::with_capacity(count);
        let mut at = 10;
        for _ in 0..count {
            weights.push(i32::from_le_bytes(
                input[at..at + 4].try_into().unwrap_or([0; 4]),
            ));
            at += 4;
        }
        let bias = i32::from_le_bytes(input[at..at + 4].try_into().unwrap_or([0; 4]));
        at += 4;
        let samples = u64::from_le_bytes(input[at..at + 8].try_into().unwrap_or([0; 8]));
        at += 8;
        let residual = u64::from_le_bytes(input[at..at + 8].try_into().unwrap_or([0; 8]));
        if residual > i64::MAX as u64 {
            return Err(ModelDecodeError::Malformed);
        }
        Ok(Self {
            weights,
            bias,
            samples,
            residual,
            version,
        })
    }
}

fn model_checksum(bytes: &[u8]) -> u64 {
    let mut hash = 14_695_981_039_346_656_037_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash
}

/// A learned shadow decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LearnedDecision {
    /// Learned source identity.
    pub source: ControllerSource,
    /// Safe alternative, if one was found.
    pub action: Option<Action>,
    /// Whether baseline behavior was retained.
    pub fallback_to_baseline: bool,
    /// Normalized model uncertainty.
    pub uncertainty_ppm: u32,
    /// Typed safety rejection for a rejected alternative.
    pub violation: Option<SafetyViolation>,
    /// Explanation trace.
    pub trace: DecisionTrace,
}

/// Lightweight optional learned controller.
#[derive(Debug, Clone)]
pub struct LearnedController {
    config: LearnedConfig,
    model: LearnedModel,
    disabled: Option<ModelDisableReason>,
}

impl Default for LearnedController {
    fn default() -> Self {
        Self::new(LearnedConfig::default())
    }
}

impl LearnedController {
    /// Creates a learned controller in deterministic fallback mode.
    #[must_use]
    pub fn new(config: LearnedConfig) -> Self {
        Self {
            config,
            model: LearnedModel::new(),
            disabled: None,
        }
    }
    /// Returns whether the model is allowed to suggest.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.disabled.is_none()
    }
    /// Returns the deterministic disable reason.
    #[must_use]
    pub const fn disable_reason(&self) -> Option<ModelDisableReason> {
        self.disabled
    }
    /// Disables the model with a typed reason.
    pub fn disable(&mut self, reason: ModelDisableReason) {
        self.disabled = Some(reason);
    }
    /// Re-enables the model while retaining learned samples.
    pub fn enable(&mut self) {
        self.disabled = None;
    }
    /// Returns the current model.
    #[must_use]
    pub const fn model(&self) -> &LearnedModel {
        &self.model
    }
    /// Encodes the current model for bounded persistence.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.model.encode()
    }
    /// Loads persisted model bytes without blocking or panicking.
    #[must_use]
    pub fn load(&mut self, input: &[u8]) -> ModelLoadReport {
        if input.is_empty() {
            return ModelLoadReport {
                status: ModelLoadStatus::Empty,
                disabled: self.disabled.is_some(),
            };
        }
        match LearnedModel::decode(input) {
            Ok(model) => {
                self.model = model;
                ModelLoadReport {
                    status: ModelLoadStatus::Loaded,
                    disabled: self.disabled.is_some(),
                }
            }
            Err(ModelDecodeError::IncompatibleVersion { .. }) => {
                self.model = LearnedModel::new();
                self.disable(ModelDisableReason::IncompatibleModel);
                ModelLoadReport {
                    status: ModelLoadStatus::Incompatible,
                    disabled: true,
                }
            }
            Err(_) => {
                self.model = LearnedModel::new();
                self.disable(ModelDisableReason::CorruptModel);
                ModelLoadReport {
                    status: ModelLoadStatus::Corrupt,
                    disabled: true,
                }
            }
        }
    }

    /// Generates a safe learned alternative in shadow mode.
    ///
    /// The baseline action and every bounded alternative are passed through
    /// the same safety validator. Insufficient samples or high uncertainty
    /// retains baseline behavior.
    #[must_use]
    pub fn suggest(
        &mut self,
        frame: &ControllerFrame,
        context: &SafetyContext,
        baseline_action: Option<&Action>,
        validator: &SafetyValidator,
    ) -> LearnedDecision {
        let observation = frame.action_observation();
        let uncertainty = self.model.uncertainty_ppm();
        let baseline_valid =
            baseline_action.is_none_or(|action| validator.validate(action, context).is_ok());
        if self.disabled.is_some()
            || !baseline_valid
            || self.model.samples < self.config.minimum_samples
            || uncertainty > self.config.maximum_uncertainty_ppm
        {
            if uncertainty > self.config.maximum_uncertainty_ppm {
                self.disable(ModelDisableReason::HighUncertainty);
            }
            return LearnedDecision {
                source: ControllerSource::LearnedShadow,
                action: baseline_action.cloned(),
                fallback_to_baseline: true,
                uncertainty_ppm: uncertainty,
                violation: None,
                trace: DecisionTrace {
                    fallback: true,
                    ..DecisionTrace::new(
                        observation,
                        ControllerSource::LearnedShadow,
                        TraceStatus::Fallback,
                        ActionReason::LearnedAlternative,
                        baseline_action.map(Action::key),
                    )
                },
            };
        }
        let Some(baseline) = baseline_action else {
            return LearnedDecision {
                source: ControllerSource::LearnedShadow,
                action: None,
                fallback_to_baseline: true,
                uncertainty_ppm: uncertainty,
                violation: None,
                trace: DecisionTrace::new(
                    observation,
                    ControllerSource::LearnedShadow,
                    TraceStatus::Fallback,
                    ActionReason::LearnedAlternative,
                    None,
                ),
            };
        };
        let mut alternatives = self.alternatives_for(baseline, context);
        alternatives.truncate(self.config.alternative_count.max(1));
        let mut best: Option<(i64, Action)> = None;
        let mut violation = None;
        for action in alternatives {
            if action.key() == baseline.key() {
                continue;
            }
            if let Err(reason) = validator.validate(&action, context) {
                violation = Some(reason);
                continue;
            }
            let features = learned_features(frame, context, &action);
            let score = self.model.score(&features);
            if best
                .as_ref()
                .is_none_or(|(best_score, _)| score > *best_score)
            {
                best = Some((score, action));
            }
        }
        let (action, status) = match best {
            Some((_, action)) => (Some(action), TraceStatus::Shadow),
            None => (Some(baseline.clone()), TraceStatus::Fallback),
        };
        let mut trace = DecisionTrace {
            fallback: status == TraceStatus::Fallback,
            ..DecisionTrace::new(
                observation,
                ControllerSource::LearnedShadow,
                status,
                ActionReason::LearnedAlternative,
                action.as_ref().map(Action::key),
            )
        };
        trace.safety_violation = violation;
        LearnedDecision {
            source: ControllerSource::LearnedShadow,
            action,
            fallback_to_baseline: status == TraceStatus::Fallback,
            uncertainty_ppm: uncertainty,
            violation,
            trace,
        }
    }

    /// Updates the online model from an observed outcome.
    ///
    /// # Errors
    ///
    /// Returns [`SafetyViolation`] when the action is not safe in the supplied context.
    pub fn observe_outcome(
        &mut self,
        frame: &ControllerFrame,
        context: &SafetyContext,
        action: &Action,
        outcome: &ActionOutcome,
        validator: &SafetyValidator,
    ) -> Result<(), SafetyViolation> {
        validator.validate(action, context)?;
        let features = learned_features(frame, context, action);
        let target = if outcome.result.is_success() {
            outcome_score(outcome)
        } else {
            -100_000
        };
        self.model.update(
            &features,
            target,
            self.config.learning_rate_ppm,
            self.config.maximum_samples,
        );
        Ok(())
    }

    fn alternatives_for(&self, baseline: &Action, context: &SafetyContext) -> Vec<Action> {
        let step = self.config.exploration_step;
        match baseline {
            Action::AdjustMemoryBudget { scope, target, .. } => {
                let current = context.memory_budget.as_ppm();
                vec![
                    Action::AdjustMemoryBudget {
                        scope: *scope,
                        target: *target,
                        budget: MemoryBudget::new(current.saturating_add(step).as_u32())
                            .unwrap_or(context.memory_budget),
                        expected_benefit: baseline.expected_benefit(),
                    },
                    Action::AdjustMemoryBudget {
                        scope: *scope,
                        target: *target,
                        budget: MemoryBudget::new(current.saturating_sub(step).as_u32())
                            .unwrap_or(context.memory_budget),
                        expected_benefit: baseline.expected_benefit(),
                    },
                ]
            }
            Action::AdjustRepairBudget { scope, target, .. } => vec![Action::AdjustRepairBudget {
                scope: *scope,
                target: *target,
                budget: RepairBudget::new(
                    context.repair_budget.as_ppm().saturating_add(step).as_u32(),
                )
                .unwrap_or(context.repair_budget),
                expected_benefit: baseline.expected_benefit(),
            }],
            Action::AdjustScrubBudget { scope, target, .. } => vec![Action::AdjustScrubBudget {
                scope: *scope,
                target: *target,
                budget: ScrubBudget::new(
                    context.scrub_budget.as_ppm().saturating_add(step).as_u32(),
                )
                .unwrap_or(context.scrub_budget),
                expected_benefit: baseline.expected_benefit(),
            }],
            _ => Vec::new(),
        }
    }
}

fn learned_features(
    frame: &ControllerFrame,
    context: &SafetyContext,
    action: &Action,
) -> [i32; MODEL_FEATURE_COUNT] {
    let action_feature = learned_action_feature(action);
    let Some(snapshot) = frame.target_snapshot(context.primary_target) else {
        let mut values = [0; MODEL_FEATURE_COUNT];
        values[MODEL_FEATURE_COUNT - 1] = action_feature;
        return values;
    };
    let features = ObservationFeatures::from_snapshot(snapshot, &BaselineConfig::default());
    [
        i32::try_from(features.memory_pressure_ppm / 100_000).unwrap_or(i32::MAX),
        i32::try_from(u32::try_from(features.criticality).unwrap_or(u32::MAX)).unwrap_or(i32::MAX),
        i32::try_from(features.repair_bytes / 1_000_000).unwrap_or(i32::MAX),
        i32::try_from(features.latency_p99_ns / 1_000_000).unwrap_or(i32::MAX),
        i32::try_from(u32::try_from(features.throughput).unwrap_or(u32::MAX)).unwrap_or(i32::MAX),
        i32::try_from(features.resident_bytes / 1_000_000).unwrap_or(i32::MAX),
        i32::try_from(
            (features.worker_imbalance_ppm / 100_000).max(features.tablet_imbalance_ppm / 100_000),
        )
        .unwrap_or(i32::MAX),
        i32::try_from(features.hot_key_count.min(i32::MAX as usize)).unwrap_or(i32::MAX),
        i32::try_from(features.top_key_share_ppm / 100_000).unwrap_or(i32::MAX),
        i32::try_from(
            (features
                .storage_read_bytes
                .saturating_add(features.storage_written_bytes))
                / 1_000_000,
        )
        .unwrap_or(i32::MAX),
        action_feature,
    ]
}

fn learned_action_feature(action: &Action) -> i32 {
    let value = match action {
        Action::AdjustMemoryBudget { budget, .. } => budget.as_ppm().as_u64() / 100_000,
        Action::ChangeMaterializationIntent { intent, .. } => match intent {
            MaterializationIntent::Materialize => 1,
            MaterializationIntent::KeepWarm => 2,
            MaterializationIntent::Demote => 3,
            MaterializationIntent::Evict => 4,
        },
        Action::ChangeRedundancyPreference { preference, .. } => {
            u64::from(preference.fragment_count())
        }
        Action::AdjustRepairBudget { budget, .. } => budget.as_ppm().as_u64() / 100_000,
        Action::AdjustScrubBudget { budget, .. } => budget.as_ppm().as_u64() / 100_000,
        Action::RecommendTabletSplit { policy, .. } => {
            u64::from(policy.max_children).saturating_add(policy.min_child_bytes / 1_000_000)
        }
        Action::RecommendTabletMerge { policy, .. } => {
            u64::from(policy.max_sources).saturating_add(policy.min_combined_bytes / 1_000_000)
        }
        Action::RecommendMigration { recommendation, .. } => recommendation.max_bytes / 1_000_000,
    };
    i32::try_from(value).unwrap_or(i32::MAX)
}

fn outcome_score(outcome: &ActionOutcome) -> i32 {
    let benefit = &outcome.observed_benefit;
    let positive = i128::from(benefit.latency.as_u32())
        + i128::from(benefit.throughput.as_u32())
        + i128::from(benefit.resource.as_u32())
        + i128::from(benefit.memory.as_u32())
        + i128::from(benefit.storage.as_u32())
        + i128::from(benefit.network.as_u32())
        + i128::from(benefit.repair_debt.as_u32())
        + i128::from(benefit.durability_risk.as_u32());
    let negative = i128::from(benefit.movement_cost.as_u32()) + i128::from(benefit.churn.as_u32());
    let score = (positive.saturating_sub(negative) / 8).clamp(-1_000_000, 1_000_000);
    i32::try_from(score).unwrap_or(if score < 0 { i32::MIN } else { i32::MAX })
}

/// Overall adaptive operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdaptiveMode {
    /// Baseline actions may be executed.
    Active,
    /// All generated actions are shadow-only.
    Shadow,
}

/// Per-source bounded execution statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ControllerStats {
    /// Number of generated actions.
    pub proposed: u64,
    /// Number admitted to the ledger.
    pub accepted: u64,
    /// Number rejected by safety.
    pub rejected: u64,
    /// Number suppressed by ledger policy.
    pub suppressed: u64,
    /// Number handed to an actuator.
    pub executed: u64,
    /// Number completed successfully.
    pub completed: u64,
    /// Number failed.
    pub failed: u64,
    /// Number baseline fallbacks.
    pub fallbacks: u64,
}

/// Action returned by an actuator after execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActuatorReceipt {
    /// Action identity acknowledged by the actuator.
    pub action_id: ActionId,
    /// Actual action outcome.
    pub outcome: ActionOutcome,
}

impl ActuatorReceipt {
    /// Creates a receipt for an action.
    #[must_use]
    pub const fn new(action_id: ActionId, outcome: ActionOutcome) -> Self {
        Self { action_id, outcome }
    }
}

/// Failure returned by the adaptive execution boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ActuationError {
    /// Shadow mode never actuates actions.
    #[error("adaptive controller is in shadow mode")]
    ShadowMode,
    /// The action ID is not retained by the ledger.
    #[error("action is not retained by the ledger")]
    UnknownAction,
    /// The action was not in a legal pre-execution state.
    #[error("action is not ready for execution")]
    IllegalState,
    /// The actuator rejected the request.
    #[error("actuator rejected the action")]
    Rejected,
    /// The actuator returned a receipt for a different action.
    #[error("actuator receipt has the wrong action identity")]
    MismatchedReceipt,
}

/// Execution boundary for adaptive actions.
pub trait Actuator {
    /// Executes one action and returns its receipt.
    ///
    /// # Errors
    ///
    /// Returns [`ActuationError`] when the request cannot be executed.
    fn execute(
        &mut self,
        action_id: ActionId,
        action: &Action,
    ) -> Result<ActuatorReceipt, ActuationError>;
}

/// Orchestrator configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveConfig {
    /// Overall execution mode.
    pub mode: AdaptiveMode,
    /// Ledger configuration.
    pub ledger: ActionLedgerConfig,
    /// Maximum retained controller traces.
    pub max_traces: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            mode: AdaptiveMode::Active,
            ledger: ActionLedgerConfig::default(),
            max_traces: 512,
        }
    }
}

/// A bounded decision from the adaptive orchestrator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveDecision {
    /// Active records that may be executed.
    pub active_actions: Vec<ActionRecord>,
    /// Learned or baseline shadow actions that must never be executed.
    pub shadow_actions: Vec<Action>,
    /// Decision traces in deterministic order.
    pub traces: Vec<DecisionTrace>,
    /// Whether any path used baseline fallback.
    pub fallback: bool,
}

/// Orchestrator combining baseline, optional learned shadow, safety, and ledger.
#[derive(Debug, Clone)]
pub struct AdaptiveController {
    baseline: BaselineController,
    learned: Option<LearnedController>,
    validator: SafetyValidator,
    ledger: ActionLedger,
    config: AdaptiveConfig,
    traces: VecDeque<DecisionTrace>,
    stats: BTreeMap<ControllerSource, ControllerStats>,
    feedback: BTreeMap<ActionId, (ControllerFrame, SafetyContext)>,
}

impl Default for AdaptiveController {
    fn default() -> Self {
        Self::new(
            AdaptiveConfig::default(),
            BaselineController::default(),
            None,
        )
    }
}

impl AdaptiveController {
    /// Creates an orchestrator with a baseline and optional learned shadow.
    #[must_use]
    pub fn new(
        config: AdaptiveConfig,
        baseline: BaselineController,
        learned: Option<LearnedController>,
    ) -> Self {
        let mut stats = BTreeMap::new();
        stats.insert(ControllerSource::Baseline, ControllerStats::default());
        stats.insert(ControllerSource::LearnedShadow, ControllerStats::default());
        stats.insert(ControllerSource::Learned, ControllerStats::default());
        Self {
            baseline,
            learned,
            validator: SafetyValidator::default(),
            ledger: ActionLedger::new(config.ledger),
            config: AdaptiveConfig {
                max_traces: config.max_traces.max(1),
                ..config
            },
            traces: VecDeque::new(),
            stats,
            feedback: BTreeMap::new(),
        }
    }
    /// Replaces the safety envelope while preserving controller state.
    pub fn set_safety_validator(&mut self, validator: SafetyValidator) {
        self.validator = validator;
    }
    /// Returns the current safety validator.
    #[must_use]
    pub const fn safety_validator(&self) -> &SafetyValidator {
        &self.validator
    }
    /// Returns the current operating mode.
    #[must_use]
    pub const fn mode(&self) -> AdaptiveMode {
        self.config.mode
    }
    /// Switches between active and shadow modes.
    pub fn set_mode(&mut self, mode: AdaptiveMode) {
        self.config.mode = mode;
    }
    /// Returns the optional learned shadow controller.
    #[must_use]
    pub const fn learned(&self) -> Option<&LearnedController> {
        self.learned.as_ref()
    }
    /// Returns the action ledger.
    #[must_use]
    pub const fn ledger(&self) -> &ActionLedger {
        &self.ledger
    }
    /// Returns bounded controller traces.
    pub fn traces(&self) -> impl Iterator<Item = &DecisionTrace> {
        self.traces.iter()
    }
    /// Returns per-source statistics.
    #[must_use]
    pub const fn stats(&self) -> &BTreeMap<ControllerSource, ControllerStats> {
        &self.stats
    }

    /// Makes a decision without invoking an actuator.
    ///
    /// Baseline actions are admitted only in [`AdaptiveMode::Active`]. Learned
    /// suggestions are always generated, validated, and returned as shadow
    /// actions, including when the overall controller is in shadow mode.
    #[must_use]
    #[allow(clippy::semicolon_if_nothing_returned)]
    pub fn decide(&mut self, frame: &ControllerFrame, context: &SafetyContext) -> AdaptiveDecision {
        let proposal = self.baseline.propose(frame, context, &self.validator);
        let mut traces = vec![proposal.trace.clone()];
        let mut active_actions = Vec::new();
        let mut shadow_actions = Vec::new();
        let mut fallback = proposal.fallback;
        self.update_stats(ControllerSource::Baseline, |stats| {
            stats.proposed = stats.proposed.saturating_add(proposal.actions.len() as u64)
        });
        for action in proposal.actions {
            let observation = frame.action_observation();
            if self.config.mode == AdaptiveMode::Shadow {
                traces.push(DecisionTrace::new(
                    observation,
                    ControllerSource::Baseline,
                    TraceStatus::Shadow,
                    proposal.trace.reason,
                    Some(action.key()),
                ));
                shadow_actions.push(action);
                continue;
            }
            match self.ledger.admit(
                action.clone(),
                proposal.trace.reason,
                ControllerSource::Baseline,
                observation,
            ) {
                Ok(record) => {
                    self.remember_feedback(record.id, frame, context);
                    active_actions.push(record);
                    self.update_stats(ControllerSource::Baseline, |stats| {
                        stats.accepted = stats.accepted.saturating_add(1)
                    });
                }
                Err(rejection) => {
                    self.update_stats(ControllerSource::Baseline, |stats| {
                        stats.suppressed = stats.suppressed.saturating_add(1)
                    });
                    traces.push(DecisionTrace {
                        ledger_rejection: Some(rejection),
                        ..DecisionTrace::new(
                            observation,
                            ControllerSource::Baseline,
                            TraceStatus::Suppressed,
                            proposal.trace.reason,
                            Some(action.key()),
                        )
                    });
                }
            }
        }
        if let Some(learned) = self.learned.as_mut() {
            let baseline_action = active_actions
                .first()
                .map(|record| &record.action)
                .or_else(|| shadow_actions.first());
            let learned_decision =
                learned.suggest(frame, context, baseline_action, &self.validator);
            fallback |= learned_decision.fallback_to_baseline;
            self.update_stats(ControllerSource::LearnedShadow, |stats| {
                stats.proposed = stats.proposed.saturating_add(1)
            });
            if let Some(action) = learned_decision.action {
                shadow_actions.push(action);
            }
            traces.push(learned_decision.trace);
        }
        for trace in traces.iter().cloned() {
            self.push_trace(trace);
        }
        AdaptiveDecision {
            active_actions,
            shadow_actions,
            traces,
            fallback,
        }
    }

    /// Executes an admitted action through an [`Actuator`].
    ///
    /// # Errors
    ///
    /// Returns [`ActuationError`] for shadow mode, unknown IDs, illegal lifecycle state, actuator rejection, or mismatched receipts.
    #[allow(clippy::semicolon_if_nothing_returned)]
    pub fn execute<A: Actuator>(
        &mut self,
        actuator: &mut A,
        action_id: ActionId,
        observation: ActionObservation,
    ) -> Result<ActionRecord, ActuationError> {
        if self.config.mode == AdaptiveMode::Shadow {
            return Err(ActuationError::ShadowMode);
        }
        let record = self
            .ledger
            .record(action_id)
            .ok_or(ActuationError::UnknownAction)?
            .clone();
        if record.state != ActionState::Proposed {
            return Err(ActuationError::IllegalState);
        }
        let action = record.action.clone();
        self.ledger
            .transition(action_id, ActionState::Accepted, observation, None)
            .map_err(|_| ActuationError::IllegalState)?;
        self.ledger
            .transition(action_id, ActionState::Executing, observation, None)
            .map_err(|_| ActuationError::IllegalState)?;
        self.update_stats(record.source, |stats| {
            stats.executed = stats.executed.saturating_add(1)
        });
        match actuator.execute(action_id, &action) {
            Ok(receipt) if receipt.action_id == action_id => {
                let outcome = receipt.outcome;
                let updated = self
                    .ledger
                    .record_outcome(action_id, observation, outcome.clone())
                    .map_err(|_| ActuationError::Rejected)?;
                self.baseline.on_outcome(&action, &outcome);
                self.update_learned_outcome(action_id, &action, &outcome);
                self.update_stats(record.source, |stats| {
                    if outcome.result.is_success() {
                        stats.completed = stats.completed.saturating_add(1);
                    } else {
                        stats.failed = stats.failed.saturating_add(1);
                    }
                });
                Ok(updated)
            }
            Ok(_) => {
                self.fail_action(action_id, observation);
                Err(ActuationError::MismatchedReceipt)
            }
            Err(
                ActuationError::Rejected
                | ActuationError::ShadowMode
                | ActuationError::UnknownAction
                | ActuationError::IllegalState
                | ActuationError::MismatchedReceipt,
            ) => {
                self.fail_action(action_id, observation);
                Err(ActuationError::Rejected)
            }
        }
    }

    /// Records an externally obtained actuator receipt.
    ///
    /// # Errors
    ///
    /// Returns [`ActuationError`] for unknown or non-executing actions.
    #[allow(clippy::needless_pass_by_value, clippy::semicolon_if_nothing_returned)]
    pub fn apply_receipt(
        &mut self,
        receipt: ActuatorReceipt,
        observation: ActionObservation,
    ) -> Result<ActionRecord, ActuationError> {
        let record = self
            .ledger
            .record(receipt.action_id)
            .ok_or(ActuationError::UnknownAction)?
            .clone();
        match record.state {
            ActionState::Proposed => {
                self.ledger
                    .transition(receipt.action_id, ActionState::Accepted, observation, None)
                    .map_err(|_| ActuationError::IllegalState)?;
                self.ledger
                    .transition(receipt.action_id, ActionState::Executing, observation, None)
                    .map_err(|_| ActuationError::IllegalState)?;
            }
            ActionState::Accepted => {
                self.ledger
                    .transition(receipt.action_id, ActionState::Executing, observation, None)
                    .map_err(|_| ActuationError::IllegalState)?;
            }
            ActionState::Executing => {}
            _ => return Err(ActuationError::IllegalState),
        }
        let updated = self
            .ledger
            .record_outcome(receipt.action_id, observation, receipt.outcome.clone())
            .map_err(|_| ActuationError::IllegalState)?;
        self.baseline.on_outcome(&record.action, &receipt.outcome);
        self.update_learned_outcome(receipt.action_id, &record.action, &receipt.outcome);
        self.update_stats(record.source, |stats| {
            if receipt.outcome.result.is_success() {
                stats.completed = stats.completed.saturating_add(1);
            } else {
                stats.failed = stats.failed.saturating_add(1);
            }
        });
        Ok(updated)
    }

    fn remember_feedback(
        &mut self,
        action_id: ActionId,
        frame: &ControllerFrame,
        context: &SafetyContext,
    ) {
        if self.learned.is_none() {
            return;
        }
        self.feedback
            .insert(action_id, (frame.clone(), context.clone()));
        while self.feedback.len() > self.ledger.config().max_history {
            let Some(oldest) = self.feedback.keys().next().copied() else {
                break;
            };
            self.feedback.remove(&oldest);
        }
    }

    fn update_learned_outcome(
        &mut self,
        action_id: ActionId,
        action: &Action,
        outcome: &ActionOutcome,
    ) {
        let Some((frame, context)) = self.feedback.remove(&action_id) else {
            return;
        };
        let Some(learned) = self.learned.as_mut() else {
            return;
        };
        let _ = learned.observe_outcome(&frame, &context, action, outcome, &self.validator);
    }

    #[allow(clippy::collapsible_if)]
    fn fail_action(&mut self, action_id: ActionId, observation: ActionObservation) {
        if let Ok(record) = self.ledger.transition(
            action_id,
            ActionState::Failed,
            observation,
            Some(ActionOutcome::failed(
                ActionResult::Failed,
                Duration::ZERO,
                ActionCost::zero(),
            )),
        ) {
            if let Some(outcome) = record.actual_outcome {
                self.baseline.on_outcome(&record.action, &outcome);
                self.update_learned_outcome(action_id, &record.action, &outcome);
            }
        }
    }

    fn update_stats<F>(&mut self, source: ControllerSource, update: F)
    where
        F: FnOnce(&mut ControllerStats),
    {
        update(self.stats.entry(source).or_default());
    }

    fn push_trace(&mut self, trace: DecisionTrace) {
        self.traces.push_back(trace);
        while self.traces.len() > self.config.max_traces {
            self.traces.pop_front();
        }
    }
}

/// One deterministic simulation input.
#[derive(Debug, Clone, PartialEq)]
pub struct SimulationInput {
    /// Typed observation snapshot.
    pub observation: ObservationSnapshot,
    /// Safety context paired with the snapshot.
    pub context: SafetyContext,
}

impl SimulationInput {
    /// Creates one simulation input.
    #[must_use]
    pub const fn new(observation: ObservationSnapshot, context: SafetyContext) -> Self {
        Self {
            observation,
            context,
        }
    }
}

/// A named, bounded simulation workload.
#[derive(Debug, Clone, PartialEq)]
pub struct SimulationWorkload {
    /// Seed used for deterministic scenario identity.
    pub seed: u64,
    /// Ordered scenario inputs.
    pub inputs: Vec<SimulationInput>,
}

impl SimulationWorkload {
    /// Creates a workload with a fixed seed and ordered inputs.
    #[must_use]
    pub const fn new(seed: u64, inputs: Vec<SimulationInput>) -> Self {
        Self { seed, inputs }
    }
    /// Number of scenario steps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inputs.len()
    }
    /// Whether the workload has no steps.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }
}

/// Public report from a deterministic controller simulation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulationReport {
    /// Workload seed.
    pub seed: u64,
    /// Number of steps processed.
    pub steps: usize,
    /// Number of actions proposed.
    pub proposed: u64,
    /// Number of actions admitted.
    pub admitted: u64,
    /// Number of actions executed.
    pub executed: u64,
    /// Number of successful outcomes.
    pub completed: u64,
    /// Number of failed outcomes.
    pub failed: u64,
    /// Number of shadow decisions generated.
    pub shadow: u64,
    /// Bounded final decision traces.
    pub traces: Vec<DecisionTrace>,
}

/// Deterministic virtual-time simulation runner.
#[derive(Debug, Clone)]
pub struct ControllerSimulation {
    seed: u64,
    sequence: ObservationSequence,
    window: ObservationWindowId,
    ticks: Ticks,
    window_period: Duration,
}

impl ControllerSimulation {
    /// Creates a runner with explicit virtual identity and no wall clock.
    #[must_use]
    pub const fn new(
        seed: u64,
        sequence: ObservationSequence,
        window: ObservationWindowId,
        ticks: Ticks,
        window_period: Duration,
    ) -> Self {
        Self {
            seed,
            sequence,
            window,
            ticks,
            window_period,
        }
    }
    /// Returns the fixed workload seed.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Runs a workload and returns a deterministic report.
    pub fn run<A: Actuator>(
        &mut self,
        controller: &mut AdaptiveController,
        workload: &SimulationWorkload,
        actuator: &mut A,
    ) -> SimulationReport {
        let mut report = SimulationReport {
            seed: self.seed.min(workload.seed),
            steps: 0,
            proposed: 0,
            admitted: 0,
            executed: 0,
            completed: 0,
            failed: 0,
            shadow: 0,
            traces: Vec::new(),
        };
        for input in &workload.inputs {
            let frame = ControllerFrame::new(
                input.observation.clone(),
                input.context.scope,
                self.sequence,
                self.window,
                1,
                self.ticks,
            );
            let decision = controller.decide(&frame, &input.context);
            report.steps = report.steps.saturating_add(1);
            report.proposed = report.proposed.saturating_add(
                (decision.active_actions.len() + decision.shadow_actions.len()) as u64,
            );
            report.admitted = report
                .admitted
                .saturating_add(decision.active_actions.len() as u64);
            report.shadow = report
                .shadow
                .saturating_add(decision.shadow_actions.len() as u64);
            for record in decision.active_actions {
                if controller
                    .execute(actuator, record.id, frame.action_observation())
                    .is_ok()
                {
                    report.executed = report.executed.saturating_add(1);
                    if let Some(updated) = controller.ledger().record(record.id) {
                        if updated.state == ActionState::Completed {
                            report.completed = report.completed.saturating_add(1);
                        } else if updated.state == ActionState::Failed {
                            report.failed = report.failed.saturating_add(1);
                        }
                    }
                }
            }
            for trace in decision.traces {
                report.traces.push(trace);
                if report.traces.len() > controller.config.max_traces {
                    report.traces.remove(0);
                }
            }
            if let Ok(next_sequence) = self.sequence.next() {
                self.sequence = next_sequence;
            }
            if let Ok(next_window) = self.window.next() {
                self.window = next_window;
            }
            self.ticks = self.ticks.advance_by(self.window_period);
        }
        if report.traces.len() > controller.config.max_traces {
            report.traces = report
                .traces
                .split_off(report.traces.len() - controller.config.max_traces);
        }
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_observation::sketch::HotKeyConcentration;
    use kivi_observation::{
        AggregateEvidence, CheckpointDebtSummary, ExecutionCriticalitySummary, Freshness,
        HotKeySummary, LatencySummary, MaintenanceDebtSummary, MemoryPressureSummary,
        ObservationSources, RepairDebtSummary, ScrubDebtSummary, UnitInterval,
    };

    fn scope(
        window: u64,
        sequence: u64,
        pressure: f64,
        repair_bytes: u64,
        latency_ns: u64,
        criticality: u64,
    ) -> ScopeSnapshot {
        let window_id = ObservationWindowId::from_u64(window);
        let sequence_id = ObservationSequence::from_u64(sequence);
        let observed_at = Ticks::from_micros(sequence_id.as_u64());
        ScopeSnapshot {
            scope: ObservationScope::Tablet(TabletId::from_u64(1)),
            first_window: window_id,
            latest_window: window_id,
            latest_sequence: sequence_id,
            sample_count: 8,
            evidence: AggregateEvidence {
                oldest_observed_at: observed_at,
                latest_observed_at: observed_at,
                sources: ObservationSources::from_source(
                    kivi_observation::ObservationSource::Simulator,
                ),
                confidence: Confidence::High,
                freshness: Freshness::classify(observed_at, observed_at, Duration::from_secs(5)),
            },
            request: None,
            latency: Some(LatencySummary {
                sample_count: 8,
                p50_ns: latency_ns,
                p99_ns: latency_ns,
                p999_ns: latency_ns,
                max_ns: latency_ns,
            }),
            queueing: None,
            compute: None,
            memory: None,
            memory_pressure: Some(MemoryPressureSummary {
                mean_pressure: UnitInterval::new(pressure).unwrap_or(UnitInterval::ZERO),
                reclaim_attempts: 0,
                reclaim_failures: 0,
            }),
            compression: None,
            offcore_latency: None,
            io: None,
            consensus_lag: None,
            maintenance_debt: Some(MaintenanceDebtSummary {
                checkpoint: CheckpointDebtSummary {
                    bytes: 0,
                    oldest_age: Duration::ZERO,
                },
                repair: RepairDebtSummary {
                    assets: 1,
                    bytes: repair_bytes,
                    oldest_age: Duration::from_secs(10),
                },
                scrub: ScrubDebtSummary {
                    bytes: 0,
                    oldest_age: Duration::ZERO,
                },
            }),
            redundancy: None,
            degraded_assets: None,
            migration_cost: None,
            topology: None,
            hot_keys: HotKeySummary {
                concentration: HotKeyConcentration::empty(),
                entries: Vec::new(),
            },
            execution_criticality: Some(ExecutionCriticalitySummary {
                mean_score: kivi_observation::CriticalityScore::from_u64(criticality),
                peak_score: kivi_observation::CriticalityScore::from_u64(criticality),
                observation_count: 1,
                access_count: 1,
            }),
        }
    }

    fn frame(
        window: u64,
        sequence: u64,
        pressure: f64,
        repair_bytes: u64,
        latency_ns: u64,
        criticality: u64,
    ) -> ControllerFrame {
        let snapshot = ObservationSnapshot {
            as_of: Ticks::from_micros(sequence),
            last_sequence: Some(ObservationSequence::from_u64(sequence)),
            accepted_samples: sequence,
            scopes: vec![scope(
                window,
                sequence,
                pressure,
                repair_bytes,
                latency_ns,
                criticality,
            )],
        };
        ControllerFrame::new(
            snapshot,
            ActionScope::new(ClusterId::from_u128(1), NamespaceId::from_u64(1)),
            ObservationSequence::from_u64(sequence),
            ObservationWindowId::from_u64(window),
            1,
            Ticks::from_micros(sequence),
        )
    }

    fn context() -> SafetyContext {
        SafetyContext::new(
            ActionScope::new(ClusterId::from_u128(1), NamespaceId::from_u64(1)),
            TabletId::from_u64(1),
        )
    }

    fn memory_action(target: u64, budget: u32) -> Action {
        Action::AdjustMemoryBudget {
            scope: ActionScope::new(ClusterId::from_u128(1), NamespaceId::from_u64(1)),
            target: TabletId::from_u64(target),
            budget: MemoryBudget::new(budget).unwrap_or(MemoryBudget(Ppm::ZERO)),
            expected_benefit: expected(100_000, 0, 0, 100_000, 0, 0, 0, 0, 0, 0),
        }
    }

    #[test]
    fn legal_actions_have_closed_typed_shapes_and_keys() {
        let action = memory_action(1, 400_000);
        assert_eq!(action.kind(), ActionKind::AdjustMemoryBudget);
        assert_eq!(
            action.scope(),
            ActionScope::new(ClusterId::from_u128(1), NamespaceId::from_u64(1))
        );
        assert_eq!(action.target(), ActionTarget::Tablet(TabletId::from_u64(1)));
        assert_eq!(action.key().target, action.conflict_key().target);
        assert!(matches!(action.key().policy, PolicyFingerprint::Memory(_)));
    }

    #[test]
    fn ledger_enforces_action_rate_and_conflict_suppression() {
        let mut ledger = ActionLedger::new(ActionLedgerConfig {
            max_history: 8,
            max_traces: 8,
            max_actions_per_window: 1,
            conflict_cooldown_windows: 10,
            duplicate_cooldown_windows: 10,
        });
        let first = ledger
            .admit(
                memory_action(1, 400_000),
                ActionReason::SustainedMemoryPressure,
                ControllerSource::Baseline,
                ActionObservation {
                    sequence: ObservationSequence::from_u64(1),
                    window: ObservationWindowId::from_u64(1),
                    version: 1,
                    at: Ticks::from_micros(1),
                },
            )
            .expect("first action");
        assert_eq!(first.state, ActionState::Proposed);
        assert!(matches!(
            ledger.admit(
                memory_action(2, 400_000),
                ActionReason::SustainedMemoryPressure,
                ControllerSource::Baseline,
                ActionObservation {
                    sequence: ObservationSequence::from_u64(1),
                    window: ObservationWindowId::from_u64(1),
                    version: 1,
                    at: Ticks::from_micros(1)
                }
            ),
            Err(LedgerRejection::RateLimit)
        ));
        assert!(matches!(
            ledger.admit(
                memory_action(1, 300_000),
                ActionReason::SustainedMemoryPressure,
                ControllerSource::Baseline,
                ActionObservation {
                    sequence: ObservationSequence::from_u64(2),
                    window: ObservationWindowId::from_u64(2),
                    version: 1,
                    at: Ticks::from_micros(2)
                }
            ),
            Err(LedgerRejection::Conflict)
        ));
    }

    #[test]
    fn alternating_workload_does_not_flap_beyond_cooldowns() {
        let config = BaselineConfig {
            sustained_windows: 2,
            min_samples: 1,
            ..BaselineConfig::default()
        };
        let mut controller = BaselineController::new(config);
        let validator = SafetyValidator::default();
        let first = controller.propose(&frame(1, 1, 0.95, 0, 0, 0), &context(), &validator);
        let second = controller.propose(&frame(2, 2, 0.95, 0, 0, 0), &context(), &validator);
        assert!(first.actions.is_empty());
        assert!(!second.actions.is_empty());
        assert!(second.actions.len() <= 2);
        let low = controller.propose(&frame(3, 3, 0.1, 0, 0, 200), &context(), &validator);
        assert!(low.actions.is_empty() || low.actions.len() <= 2);
    }

    #[test]
    fn uncertainty_falls_back_to_baseline() {
        let mut learned = LearnedController::default();
        let action = memory_action(1, 400_000);
        let decision = learned.suggest(
            &frame(1, 1, 0.5, 0, 0, 0),
            &context(),
            Some(&action),
            &SafetyValidator::default(),
        );
        assert!(decision.fallback_to_baseline);
        assert_eq!(decision.action, Some(action));
    }

    #[test]
    fn unsafe_learned_suggestion_is_rejected_by_safety() {
        let mut context = context();
        context.representation_reconstructible = false;
        let action = Action::ChangeMaterializationIntent {
            scope: context.scope,
            target: context.primary_target,
            intent: MaterializationIntent::Evict,
            expected_benefit: ExpectedBenefit::zero(),
        };
        assert_eq!(
            SafetyValidator::default().validate(&action, &context),
            Err(SafetyViolation::UnrepresentableDemotion)
        );
    }

    struct CountingActuator {
        calls: u32,
    }

    impl Actuator for CountingActuator {
        fn execute(
            &mut self,
            action_id: ActionId,
            _action: &Action,
        ) -> Result<ActuatorReceipt, ActuationError> {
            self.calls = self.calls.saturating_add(1);
            Ok(ActuatorReceipt::new(
                action_id,
                ActionOutcome::success(ExpectedBenefit::zero(), Duration::ZERO, ActionCost::zero()),
            ))
        }
    }

    #[test]
    fn shadow_mode_never_actuates() {
        let config = AdaptiveConfig {
            mode: AdaptiveMode::Shadow,
            ..AdaptiveConfig::default()
        };
        let mut controller = AdaptiveController::new(config, BaselineController::default(), None);
        let mut actuator = CountingActuator { calls: 0 };
        let result = controller.execute(
            &mut actuator,
            ActionId::FIRST,
            ActionObservation {
                sequence: ObservationSequence::FIRST,
                window: ObservationWindowId::FIRST,
                version: 1,
                at: Ticks::from_micros(1),
            },
        );
        assert!(matches!(result, Err(ActuationError::ShadowMode)));
        assert_eq!(actuator.calls, 0);
    }

    #[test]
    fn lifecycle_and_conflict_identity_are_explicit() {
        let mut ledger = ActionLedger::default();
        let observation = ActionObservation {
            sequence: ObservationSequence::FIRST,
            window: ObservationWindowId::FIRST,
            version: 1,
            at: Ticks::from_micros(1),
        };
        let first = ledger
            .admit(
                memory_action(1, 400_000),
                ActionReason::SustainedMemoryPressure,
                ControllerSource::Baseline,
                observation,
            )
            .expect("action");
        assert!(ActionState::Proposed.can_transition_to(ActionState::Accepted));
        assert!(!ActionState::Completed.can_transition_to(ActionState::Executing));
        ledger
            .transition(first.id, ActionState::Accepted, observation, None)
            .expect("accepted");
        ledger
            .transition(first.id, ActionState::Executing, observation, None)
            .expect("executing");
        let completed = ledger
            .record_outcome(
                first.id,
                observation,
                ActionOutcome::success(ExpectedBenefit::zero(), Duration::ZERO, ActionCost::zero()),
            )
            .expect("completed");
        assert_eq!(completed.state, ActionState::Completed);
    }

    #[test]
    fn model_corruption_is_rejected_and_falls_back() {
        let mut learned = LearnedController::default();
        let mut bytes = learned.encode();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        let report = learned.load(&bytes);
        assert_eq!(report.status, ModelLoadStatus::Corrupt);
        assert!(!learned.is_enabled());
        assert_eq!(learned.model().samples(), 0);
    }

    #[test]
    fn objective_dimensions_are_explicit_before_scalarization() {
        let raw = ObjectiveBreakdown {
            latency: 500_000,
            throughput: 250_000,
            resource: 0,
            memory: 100_000,
            storage: 0,
            network: 0,
            repair_debt: 100_000,
            durability_risk: 0,
            movement_cost: -50_000,
            churn: -10_000,
        };
        let vector = ObjectiveVector::new(
            raw,
            ObjectiveScales::uniform(1_000_000),
            ObjectiveWeights::equal(),
        )
        .expect("valid objective");
        assert_eq!(vector.normalized.latency, 500_000);
        assert_eq!(vector.normalized.movement_cost, -50_000);
        assert!(vector.scalar_score() > 0);
    }

    #[test]
    fn criticality_is_not_hotness() {
        use kivi_observation::{ExecutionCriticalityApproximation, ExecutionCriticalitySignal};
        let rare = ExecutionCriticalityApproximation::from_signal(&ExecutionCriticalitySignal {
            accesses: 1,
            serial_accesses: 1,
            critical_path_accesses: 1,
            tail_accesses: 1,
            stall_time_ns: 900_000,
            access_latency_ns: 1_000_000,
            logical_bytes: 64,
        });
        let frequent =
            ExecutionCriticalityApproximation::from_signal(&ExecutionCriticalitySignal {
                accesses: 10_000,
                serial_accesses: 10,
                critical_path_accesses: 0,
                tail_accesses: 0,
                stall_time_ns: 0,
                access_latency_ns: 100,
                logical_bytes: 64,
            });
        assert!(rare.score.as_u64() > frequent.score.as_u64());
    }

    #[test]
    fn ledger_never_evicts_an_active_action_to_make_history_room() {
        let mut ledger = ActionLedger::new(ActionLedgerConfig {
            max_history: 1,
            max_traces: 4,
            max_actions_per_window: 8,
            conflict_cooldown_windows: 0,
            duplicate_cooldown_windows: 0,
        });
        let observation = ActionObservation {
            sequence: ObservationSequence::FIRST,
            window: ObservationWindowId::FIRST,
            version: 1,
            at: Ticks::from_micros(1),
        };
        let first = ledger
            .admit(
                memory_action(1, 400_000),
                ActionReason::SustainedMemoryPressure,
                ControllerSource::Baseline,
                observation,
            )
            .expect("first action");
        assert!(matches!(
            ledger.admit(
                memory_action(2, 300_000),
                ActionReason::SustainedMemoryPressure,
                ControllerSource::Baseline,
                observation,
            ),
            Err(LedgerRejection::RateLimit)
        ));
        assert!(ledger.record(first.id).is_some());
        ledger
            .transition(first.id, ActionState::Accepted, observation, None)
            .expect("accepted");
        ledger
            .transition(first.id, ActionState::Executing, observation, None)
            .expect("executing");
        ledger
            .record_outcome(
                first.id,
                observation,
                ActionOutcome::success(ExpectedBenefit::zero(), Duration::ZERO, ActionCost::zero()),
            )
            .expect("completed");
        assert!(
            ledger
                .admit(
                    memory_action(2, 300_000),
                    ActionReason::SustainedMemoryPressure,
                    ControllerSource::Baseline,
                    observation,
                )
                .is_ok()
        );
    }

    #[test]
    fn baseline_policy_state_changes_only_after_an_actuator_outcome() {
        let mut baseline = BaselineController::new(BaselineConfig {
            sustained_windows: 1,
            min_samples: 1,
            ..BaselineConfig::default()
        });
        let proposal = baseline.propose(
            &frame(1, 1, 0.95, 0, 0, 0),
            &context(),
            &SafetyValidator::default(),
        );
        let action = proposal
            .actions
            .iter()
            .find(|action| matches!(action, Action::ChangeMaterializationIntent { .. }))
            .expect("materialization action")
            .clone();
        assert_eq!(
            baseline.policy_state().3,
            MaterializationIntent::Materialize
        );
        baseline.on_outcome(
            &action,
            &ActionOutcome::failed(ActionResult::Failed, Duration::ZERO, ActionCost::zero()),
        );
        assert_eq!(
            baseline.policy_state().3,
            MaterializationIntent::Materialize
        );
        baseline.on_outcome(
            &action,
            &ActionOutcome::success(
                action.expected_benefit(),
                Duration::ZERO,
                ActionCost::zero(),
            ),
        );
        assert_ne!(
            baseline.policy_state().3,
            MaterializationIntent::Materialize
        );
    }
}
