//! Redundancy Fabric: explicit physical protection for immutable information.
//!
//! Raft decides ordered replicated logical state; durability decides what
//! must survive; redundancy decides how immutable information is physically
//! protected; the memory fabric decides runtime materialization; chunk and
//! checkpoint storage provide immutable physical data. This crate is the
//! third decision: one [`RedundancyFabric`] protects [`InformationAsset`]s
//! (chunks, manifests, checkpoint bands/segments, and other immutable
//! durability artifacts) behind declarative [`RedundancyIntent`]s, using
//! interchangeable replication and Reed-Solomon providers, deterministic
//! failure-domain-aware placement, generation-fenced `build → verify →
//! publish → retire` transitions, bounded repair, integrity verification,
//! observability, and deterministic simulation hooks.
//!
//! ```text
//! intent (tolerance, locality, cost — never "RS(8,3)")
//!   │
//!   ▼
//! planner (deterministic baseline; adaptive control fits later)
//!   │
//!   ▼
//! layout (scheme + bounded params + placement map)
//!   │
//!   ├── build fragments (replication or RS kernels)
//!   ├── verify (decode + content-hash vs asset id)
//!   ├── publish (atomic CURRENT; readers see only published)
//!   └── retire (old generation after the replacement proves usable)
//! ```
//!
//! The fabric never becomes another consensus protocol: it protects
//! immutable assets only, never orders mutable state, never erasure-codes
//! Raft state, and never weakens the Raft durability gate.

pub mod asset;
pub mod catalog;
pub mod dist;
pub mod domain;
pub mod error;
pub mod fabric;
pub mod fragment;
pub mod healing;
pub mod integration;
pub mod intent;
pub mod lane;
pub mod metrics;
pub mod placement;
pub mod proto;
pub mod repair;
pub mod replication;
pub mod rs;
pub mod scheme;
pub mod sim;
pub mod store;
pub mod transport;

pub use asset::{AssetId, AssetKind, InformationAsset};
pub use catalog::{
    CatalogPublish, Discovery, LayoutCatalog, MemCatalog, PublishedLayout, merge_discovery,
};
pub use dist::{DistConfig, DistributedAssessment, DistributedFabric, RepairReport, SweepReport};
pub use domain::{FailureDomain, FailureScope, NodeDescriptor, NodeHealth};
pub use error::RedundancyError;
pub use fabric::{
    FabricConfig, ReconstructionBudgets, RecoveryReport, RedundancyFabric, TransitionState,
};
pub use fragment::{FragmentId, FragmentRecord, FragmentRole, RedundancyLayout};
pub use healing::{
    ControllerSnapshot, DebtCounters, MaintenanceBudgets, MaintenanceReport, RepairDebt,
    RepairDebtEntry, RepairRisk, SchemeRequirements, ScrubBatch, ScrubCursor, ScrubMode,
    ScrubRange, ScrubSchedule, ScrubTarget,
};
pub use intent::{LocalityRequirement, RedundancyIntent, RepairConstraints, StorageCost};
pub use lane::{
    FenceHook, LaneConfig, LaneJoin, LanePriority, LaneStats, LaneValue, RedundancyLane,
};
pub use metrics::{AssetHealth, FabricMetrics, FabricMetricsSnapshot, FragmentCensus};
pub use placement::{
    DegradedPlacement, ReconstructionTargets, independent_survivable, plan_placement,
};
pub use placement::{
    DesiredPlacement, FragmentPlacement, HealthyPlacement, plan_placement_with_locality,
    reconstruction_targets,
};
pub use proto::{
    FragmentInventoryEntry, FragmentKey, FragmentReply, FragmentRpc, MAX_FRAGMENT_WIRE_BYTES,
    MAX_INVENTORY_ENTRIES, MAX_LAYOUT_WIRE_BYTES, PROTO_MAJOR, PROTO_MINOR, RefuseReason,
};
pub use repair::{
    AssetAssessment, FragmentVerdict, RepairBudgets, RepairEngine, RepairReason, RepairTask,
    ScrubReport, assess, assess_with_min_available, scrub_report,
};
pub use scheme::{
    ReplicationParams, RsParams, SchemeCapabilities, SchemeId, SchemeParams, plan_baseline,
};
pub use sim::{FaultKind, SimFragments};
pub use store::{AcceptedLayout, FragmentStore, LocalFragmentStore, StagedFragment, StoreRecovery};
pub use transport::{FragmentTransport, LoopbackTransport, PeerTarget, TransportFailure};
