//! Replicated cluster control plane: typed desired topology.
//!
//! The control plane owns **desired** cluster topology; each tablet's own
//! Raft membership remains authoritative for its actual voter state.
//! Migration reconciles actual toward desired through persisted plans.
//!
//! ```text
//! Cluster Control Plane (one system Raft group)
//!         │
//!         ├── NodeRegistry (admitted nodes + lifecycle)
//!         ├── DesiredReplicaSets (per-tablet desired voters)
//!         ├── MigrationPlans (persistent multi-step moves)
//!         └── PlacementVersion / ClusterGeneration (fencing)
//! ```
//!
//! Control state is project-owned typed mutations with canonical
//! little-endian encoding. `OpenRaft` Rust structs are never persisted
//! as canonical control state.

pub mod adaptive;
pub mod catalog;
pub mod failure;
pub mod layouts;
pub mod merge;
pub mod migration;
pub mod mutation;
pub mod node;
pub mod placement;
pub mod planner;
pub mod redundancy;
pub mod split;
pub mod state;
pub mod topology;

pub use adaptive::{
    Action, ActionCost, ActionId, ActionKind, ActionObservation, ActionOutcome, ActionRecord,
    ActionResult, ActionScope, ActionState, ActionTarget, ActuationError, Actuator,
    ActuatorReceipt, AdaptiveConfig, AdaptiveController, AdaptiveDecision, AdaptiveMode,
    BaselineConfig, BaselineController, BaselineProposal, ControllerFrame, ControllerSimulation,
    ControllerSource, ControllerStats, DecisionTrace, ErasureCodingLayout, ExpectedBenefit,
    LearnedConfig, LearnedController, LearnedDecision, LearnedModel, MaterializationIntent,
    MemoryBudget, MigrationFence, MigrationRecommendation, ModelDecodeError, ModelDisableReason,
    ModelLoadReport, ModelLoadStatus, ObjectiveBreakdown, ObjectiveScales, ObjectiveVector,
    ObjectiveWeights, Ppm, RedundancyPreference, RepairBudget, SafetyContext, SafetyEnvelope,
    SafetyValidator, SafetyViolation, ScrubBudget, SimulationInput, SimulationReport,
    SimulationWorkload, TabletMergePolicy, TabletSet, TabletSplitPolicy, TopologyViolation,
    TraceStatus,
};

pub use catalog::{
    CatalogError, CatalogIndexKind, CatalogIndexState, CatalogLayout, IndexRecord, NamespaceRecord,
};
pub use failure::{FailureDetector, FailureDetectorConfig, TabletHealth, classify_tablet};
pub use layouts::{LayoutError, LayoutKey, LayoutRecord};
pub use merge::{MergeError, MergePhase, MergePlan};
pub use migration::{MigrationError, MigrationPhase, MigrationPlan};
pub use mutation::{CONTROL_MUTATION_VERSION, ControlMutation, ControlMutationError};
pub use node::{NodeError, NodeRecord, NodeState};
pub use placement::{DesiredReplicaSet, PlacementError, PlacementVersion};
pub use planner::{
    MigrationIntent, PlannerConfig, PlannerError, plan_drain, plan_rebalance, plan_repair,
};
pub use redundancy::{
    control_generation_of, descriptors_from_registry, drain_plan, health_of, layout_key,
    published_layout,
};
pub use split::{SplitError, SplitPhase, SplitPlan};
pub use state::{ClusterGeneration, ControlApplyError, ControlSnapshotError, ControlState};
pub use topology::{PlanId, PlanKind, PlanPriority, PlanSchedulerConfig, tablet_sets_conflict};
