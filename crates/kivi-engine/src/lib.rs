//! Production single-node Kivi engine: live tablets, worker threads, local
//! routing, and the embedded client API.
//!
//! Ownership model (no locks on the data path): each worker owns its OS
//! thread, its [`LiveTablet`]s, and their object stores. Requests travel
//! over bounded channels; routing snapshots publish through RCU-style atomic
//! swaps, never a global mutex.
//!
//! The embedded [`LocalEngine`]/[`LocalClient`] path is the local API —
//! future network connections will live directly on workers and call tablet
//! execution without this cross-thread hop. The two paths share tablet logic
//! but are not optimized as one.

pub mod checkpoint;
pub mod chunk_lane;
pub mod commit;
pub mod compound;
pub mod engine;
pub mod fabric;
pub mod net;
pub mod placement;
pub mod redundancy;
pub mod routing;
pub mod tablet;
pub mod worker;

pub use checkpoint::{CheckpointAdminState, CheckpointCommand, CheckpointConfig, CheckpointInfo};
pub use chunk_lane::{
    CHUNK_JOB_DEPTH, ChunkJob, ChunkLaneGuard, ChunkLaneHandle, ChunkLaneStats, ChunkReply,
    DEFAULT_CHUNK_CACHE_BYTES, LargeSetSplit, StagedValue, StagingPins, spawn_lane,
    split_large_set,
};
pub use commit::{BatchPolicy, CommitCoordinator, CommitMetricsSnapshot, LaneMaintenance};
pub use engine::{
    AdminHandle, ChunkFabricConfig, DurabilityMode, DurableConfig, EngineConfig, EngineDurability,
    EngineError, LocalClient, LocalEngine, ShutdownReport, TxnStatusReport,
};
pub use fabric::{
    FABRIC_INLINE_MAX, FabricConfig, FabricError, FabricPaths, FabricStatsSnapshot, StagedSeal,
    TabletFabric,
};
pub use kivi_core::SystemClock;
pub use kivi_memory::offcore_lane::{
    DemotedRecord, OFFCORE_JOB_DEPTH, OffcoreLaneGuard, OffcoreLaneHandle, OffcoreLaneStats,
    OffcoreReply, PromotedBytes, spawn_offcore_lane,
};
pub use kivi_memory::{Footprint, MemoryControlPolicy, MemoryControlPolicyError};
pub use net::{ConnLimits, EngineNetwork, NetConfig, NetStartError, TurnBudget};
pub use placement::{DEFAULT_ROLES, HardwareConfig, PlacementError, ThreadPlacement};
pub use redundancy::{
    CheckpointProtection, EngineRedundancy, ProtectedArtifact, RedundancyAdminSnapshot,
    protect_staged_value,
};
pub use routing::{Placement, RoutingSnapshot};
pub use tablet::{
    DedupEntry, DurablePrepared, LiveTablet, SessionDedup, TabletError, TabletMetrics,
};
pub use worker::{
    FabricWorkerReport, LaneAccess, TabletCheckpointStatus, WorkerControl, WorkerDurability,
    WorkerMetrics,
};
