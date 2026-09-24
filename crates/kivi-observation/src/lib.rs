//! Typed, bounded observation collection for Kivi's Phase 12 control loops.
//!
//! The fabric accepts a closed set of domain signals, combines them by
//! strongly typed scope, and emits deterministic snapshots. It retains only
//! bounded latency windows, aggregate counters, and a fixed-capacity hot-key
//! sketch. It never stores an unrestricted time series or an untyped metrics
//! map.

pub mod fabric;
pub mod latency;
pub mod metadata;
pub mod scope;
pub mod signals;
pub mod sketch;
pub mod window;

pub use fabric::{
    CheckpointDebtSummary, CollectError, ComputeSummary, ConsensusLagSummary,
    DegradedAssetsSummary, ExecutionCriticalitySummary, HotKeySummary, IoSummary,
    MaintenanceDebtSummary, MemoryPressureSummary, MemorySummary, MigrationCostSummary,
    ObservationFabric, ObservationLimits, ObservationSnapshot, OffcoreLatencySummary,
    QueueingSummary, ReadWriteRatio, RedundancySummary, RepairDebtSummary, RequestSummary,
    ScopeSnapshot, ScrubDebtSummary, TopologySummary,
};
pub use latency::LatencySummary;
pub use metadata::{
    AggregateEvidence, Confidence, Freshness, FreshnessStatus, ObservationMetadata,
    ObservationSource, ObservationSources,
};
pub use scope::ObservationScope;
pub use signals::{
    CheckpointDebt, CompressionSignal, ComputeSignal, ConsensusLagSignal, CriticalityScore,
    DegradedAssetsSignal, ExecutionCriticalityApproximation, ExecutionCriticalitySignal,
    HotKeySignal, IoSignal, LatencySignal, MaintenanceDebtSignal, MemoryPressureSignal,
    MemorySignal, MigrationCostSignal, NetworkBytes, ObservationSample, OffcoreLatencySignal,
    QueueingSignal, RedundancySignal, RepairDebt, RequestSignal, ScrubDebt, StorageBytes,
    TopologyImbalanceSignal, TypedSignal, UnitInterval,
};
pub use sketch::{
    ExecutionClass, HotKeyClassification, HotKeyConcentration, HotKeyEntry, HotKeyError, HotKeyId,
    HotKeySketch, HotnessClass,
};
pub use window::{
    ObservationPeriod, ObservationSequence, ObservationStamp, ObservationWindowId,
    SequenceExhausted, WindowIdExhausted,
};
