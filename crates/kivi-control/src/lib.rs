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
    Action, ActionCost, ActionId, ActionOutcome, ActionRecord, ActionResult, ActionScope,
    ActionState, ActionTarget, ActuationError, Actuator, ActuatorReceipt, AdaptiveConfig,
    AdaptiveController, AdaptiveDecision, AdaptiveMode, BaselineConfig, BaselineController,
    ControllerFrame, ControllerSimulation, ControllerSource, ControllerStats, DecisionTrace,
    ExpectedBenefit, LearnedController, MaterializationIntent, MemoryBudget, MigrationFence,
    MigrationRecommendation, Ppm, RedundancyPreference, RepairBudget, SafetyContext,
    SafetyValidator, ScrubBudget, SimulationInput, SimulationReport, SimulationWorkload,
    TabletMergePolicy, TabletSet, TabletSplitPolicy,
};
pub use catalog::{
    CatalogIndexKind, CatalogIndexState, CatalogLayout, IndexRecord, NamespaceRecord,
};
pub use failure::{FailureDetector, FailureDetectorConfig, TabletHealth, classify_tablet};
pub use layouts::{LayoutKey, LayoutRecord};
pub use merge::{MergePhase, MergePlan};
pub use migration::{MigrationPhase, MigrationPlan};
pub use mutation::{ControlMutation, ControlMutationError};
pub use node::{NodeRecord, NodeState};
pub use placement::{DesiredReplicaSet, PlacementVersion};
pub use planner::{MigrationIntent, PlannerConfig, plan_drain, plan_rebalance, plan_repair};
pub use redundancy::{descriptors_from_registry, layout_key, published_layout};
pub use split::{SplitPhase, SplitPlan};
pub use state::{ControlApplyError, ControlState};
pub use topology::{PlanId, PlanKind, PlanPriority, PlanSchedulerConfig, tablet_sets_conflict};
