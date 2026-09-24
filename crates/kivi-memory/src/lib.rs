//! Kivi Memory Fabric (Phase 9): logical state independent from physical
//! materialization.
//!
//! One logical object (small authoritative root plus version) may
//! simultaneously exist as decoded DRAM, compressed DRAM, `NVMe`-resident
//! bytes, or (for large values) immutable content-addressed chunks. The
//! fabric chooses and continuously reevaluates placement from declarative
//! [`MaterializationIntent`] against calibrated
//! [`ProviderCaps`](crate::provider::ProviderCaps); any placement decision
//! may affect latency, memory, or cost but never alters logical state.
//!
//! # Safety invariant
//!
//! Before reclaiming a representation, the fabric proves another valid
//! representation or reconstruction source exists. Promotion builds,
//! verifies, and publishes the new representation before retiring the old
//! one (Nomad-style non-exclusive tiering with shadow copies). A
//! materialization optimization can therefore make the system slower or
//! more expensive, never incorrect.
//!
//! # Size classes
//!
//! - Tiny (`<= TINY_INLINE_MAX`): inline bytes in the object root. No
//!   arena, compression, or storage machinery (RFC §25 small-value fast
//!   path preserved).
//! - Medium (`<= MEDIUM_MAX`): worker-local behavior-aware arenas with
//!   generational handles ([`handle`], [`arena`]), compressible,
//!   demotable to `NVMe`.
//! - Large (`> MEDIUM_MAX`): immutable content-addressed chunks owned by
//!   `kivi-chunk`. This crate never creates a competing large-object
//!   system; it only names chunk references as reconstruction sources.
//!
//! # Determinism
//!
//! Placement, transition, and movement decisions take explicit timestamps
//! and seeded randomness from the caller. The fabric never reads a
//! hardware clock or thread-local RNG internally, so deterministic
//! simulation ([`sim`]) replays executions exactly.

pub mod arena;
pub mod calibrate;
pub mod compressed;
pub mod criticality;
pub mod dram;
pub mod error;
pub mod fabric;
pub mod handle;
pub mod intent;
pub mod journal;
pub mod movement;
pub mod numa;
pub mod nvme;
pub mod offcore;
pub mod offcore_lane;
pub mod placement;
pub mod policy;
pub mod provider;
pub mod repr;
pub mod sim;
pub mod tablet;
pub mod telemetry;
pub mod transition;

pub use kivi_observation::UnitInterval;

pub use arena::{ArenaStats, WorkerArenas};
pub use calibrate::{Calibrator, Ewma, ProviderCalibration};
pub use criticality::{
    AccessSignals, ExecutionCriticality, Mutability, TelemetrySource, criticality_of,
};
pub use error::MemoryError;
pub use fabric::{
    ExternalOp, FabricStats, Footprint, GetOutcome, MemoryFabric, MemoryFabricConfig, OffcoreImport,
};
pub use handle::{BehaviorClass, ObjectHandle};
pub use intent::{
    CoherenceRequirement, LocalityRequirement, MaterializationIntent, SharingScope, TenantTag,
    VolatilityAllowed,
};
pub use journal::{
    JournalEntry, JournalFile, import_from_entry, journal_append_batch, journal_compact,
    journal_load, open_material_provider,
};
pub use movement::{BoundedMoveQueue, MovementBudget, MovementScheduler};
pub use numa::{NumaTopology, OsNuma};
pub use nvme::{
    Admission, AdmissionController, NvmeOptions, NvmeProvider, demote_async, promote_async,
};
pub use offcore::{OffcoreCompletion, OffcoreKind, OffcoreOp, OffcoreOpId, OffcoreQueue};
pub use offcore_lane::{
    DemotedRecord, OFFCORE_JOB_DEPTH, OffcoreLaneGuard, OffcoreLaneHandle, OffcoreLaneStats,
    OffcoreReply, PromotedBytes, spawn_offcore_lane,
};
pub use placement::{
    CriticalityPlanner, PlacementDecision, Planner, S3FifoBaseline, SieveBaseline,
};
pub use policy::{
    DEFAULT_COMPRESSION_AGGRESSIVENESS, DEFAULT_CRITICALITY_WEIGHTING,
    DEFAULT_DRAM_RESIDENCY_TARGET, DEFAULT_NVME_DEMOTION_PRESSURE, DEFAULT_PROMOTION_THRESHOLD,
    MemoryControlPolicy, MemoryControlPolicyError,
};
pub use provider::{
    LocalityCaps, ProviderCaps, ProviderDescriptor, ProviderId, ProviderKind, ProviderRegistry,
};
pub use repr::{ObjectMaterialization, PhysicalRepr, ReconstructionSource, Residence};
pub use sim::{SimFault, SimProvider};
pub use tablet::{TabletEnvelope, TabletMaterializations};
pub use telemetry::{AccessCounters, DEFAULT_OBJECT_CAPACITY, SoftwareTelemetry, TelemetryLimits};

/// Tiny-value ceiling in bytes: values at or below this stay inline in
/// the object root with no arena/compression/storage machinery (RFC §25).
///
/// 256 bytes keeps the common session/counter/pointer cases on the fast
/// path while pushing anything with real footprint into arenas where
/// behavior grouping pays off.
pub const TINY_INLINE_MAX: usize = 256;

/// Medium-object ceiling in bytes. Matches the chunked-representation
/// threshold (`DEFAULT_INLINE_THRESHOLD` = 256 KiB): larger values become
/// immutable content-addressed chunks owned by `kivi-chunk`, never arena
/// residents.
pub const MEDIUM_MAX: usize = 262_144;

/// Current physical-format version for `NVMe` materialization records.
/// Bumped on any incompatible change; old versions fail loudly on load.
pub const MATERIALIZATION_FORMAT_VERSION: u16 = 1;
