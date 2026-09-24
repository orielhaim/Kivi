//! Bounded typed collector and deterministic snapshot fabric.

use core::time::Duration;
use std::collections::BTreeMap;

use kivi_types::Ticks;

use crate::latency::{LatencySummary, LatencyWindow};
use crate::metadata::{
    AggregateEvidence, Confidence, Freshness, ObservationMetadata, ObservationSources,
};
use crate::scope::ObservationScope;
use crate::signals::{
    CompressionSignal, ComputeSignal, ConsensusLagSignal, CriticalityScore, DegradedAssetsSignal,
    ExecutionCriticalityApproximation, ExecutionCriticalitySignal, IoSignal, MaintenanceDebtSignal,
    MemoryPressureSignal, MemorySignal, MigrationCostSignal, ObservationSample, QueueingSignal,
    RedundancySignal, RequestSignal, TopologyImbalanceSignal, TypedSignal, UnitInterval,
};
use crate::sketch::{HotKeyConcentration, HotKeyEntry, HotKeySketch};
use crate::window::{ObservationSequence, ObservationWindowId};

/// Hard retention and freshness limits for one [`ObservationFabric`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObservationLimits {
    max_scopes: usize,
    max_samples_per_scope: usize,
    max_latency_samples: usize,
    hot_key_capacity: usize,
    freshness_budget: Duration,
}

impl ObservationLimits {
    /// Creates limits when every capacity is nonzero and latency retention
    /// does not exceed per-scope sample retention.
    #[must_use]
    pub fn new(
        max_scopes: usize,
        max_samples_per_scope: usize,
        max_latency_samples: usize,
        hot_key_capacity: usize,
        freshness_budget: Duration,
    ) -> Option<Self> {
        if max_scopes == 0
            || max_samples_per_scope == 0
            || max_latency_samples == 0
            || max_latency_samples > max_samples_per_scope
            || hot_key_capacity == 0
        {
            return None;
        }
        Some(Self {
            max_scopes,
            max_samples_per_scope,
            max_latency_samples,
            hot_key_capacity,
            freshness_budget,
        })
    }

    /// Maximum number of retained scopes.
    #[must_use]
    pub const fn max_scopes(&self) -> usize {
        self.max_scopes
    }

    /// Maximum accepted samples retained for one scope.
    #[must_use]
    pub const fn max_samples_per_scope(&self) -> usize {
        self.max_samples_per_scope
    }

    /// Maximum values retained in each latency window.
    #[must_use]
    pub const fn max_latency_samples(&self) -> usize {
        self.max_latency_samples
    }

    /// Maximum keys retained in each scope's hot-key sketch.
    #[must_use]
    pub const fn hot_key_capacity(&self) -> usize {
        self.hot_key_capacity
    }

    /// Maximum age treated as current in emitted snapshots.
    #[must_use]
    pub const fn freshness_budget(&self) -> Duration {
        self.freshness_budget
    }
}

/// Reason a typed sample was not accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectError {
    /// Sequence was not strictly greater than the last accepted sequence.
    NonMonotonicSequence {
        /// Last accepted sequence, or `None` before the first sample.
        last: Option<ObservationSequence>,
        /// Rejected sequence.
        received: ObservationSequence,
    },
    /// A new scope could not be retained.
    ScopeCapacity {
        /// Configured scope bound.
        limit: usize,
    },
    /// The scope already retained its configured sample bound.
    SampleCapacity {
        /// Scope at capacity.
        scope: ObservationScope,
        /// Configured per-scope sample bound.
        limit: usize,
    },
    /// A bounded hot-key counter reached `u64::MAX`.
    HotKeyCounterExhausted,
}

impl core::fmt::Display for CollectError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonMonotonicSequence { last, received } => write!(
                formatter,
                "observation sequence {} is not greater than {:?}",
                received.as_u64(),
                last.map(ObservationSequence::as_u64)
            ),
            Self::ScopeCapacity { limit } => {
                write!(formatter, "observation scope capacity {limit} reached")
            }
            Self::SampleCapacity { scope, limit } => {
                write!(formatter, "sample capacity {limit} reached for {scope:?}")
            }
            Self::HotKeyCounterExhausted => {
                formatter.write_str("hot-key observation counter exhausted")
            }
        }
    }
}

impl std::error::Error for CollectError {}

/// Read/write mix for a retained request window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReadWriteRatio(Option<f64>);

impl ReadWriteRatio {
    /// No request was observed.
    pub const NO_REQUESTS: Self = Self(None);

    /// Computes reads divided by reads plus writes.
    #[must_use]
    pub fn from_counts(reads: u64, writes: u64) -> Self {
        Self::from_counts_u128(u128::from(reads), u128::from(writes))
    }

    fn from_counts_u128(reads: u128, writes: u128) -> Self {
        let requests = reads.saturating_add(writes);
        if requests == 0 {
            Self::NO_REQUESTS
        } else {
            #[allow(clippy::cast_precision_loss)]
            let value = reads as f64 / requests as f64;
            Self(Some(value))
        }
    }

    /// Returns the numeric ratio, or `None` without requests.
    #[must_use]
    pub const fn as_option(self) -> Option<f64> {
        self.0
    }
}

/// Request, read, and write rates across retained request windows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RequestSummary {
    /// Combined request rate per second.
    pub requests_per_second: f64,
    /// Read rate per second.
    pub reads_per_second: f64,
    /// Write rate per second.
    pub writes_per_second: f64,
    /// Read share of combined requests.
    pub read_ratio: ReadWriteRatio,
}

/// Mean CPU and service work across retained compute windows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComputeSummary {
    /// Mean CPU nanoseconds per compute observation.
    pub mean_cpu_time_ns: f64,
    /// Mean service nanoseconds per compute observation.
    pub mean_service_time_ns: f64,
    /// CPU time divided by service time, or `None` without service time.
    pub cpu_to_service_ratio: Option<f64>,
}

/// Queue behavior across retained windows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueueingSummary {
    /// Maximum observed queue depth.
    pub max_depth: u64,
    /// Mean reported median queue wait in nanoseconds.
    pub mean_wait_p50_ns: f64,
    /// Mean reported 99th percentile queue wait in nanoseconds.
    pub mean_wait_p99_ns: f64,
    /// Mean reported 99.9th percentile queue wait in nanoseconds.
    pub mean_wait_p999_ns: f64,
    /// Sum of admission rejections.
    pub admission_rejections: u64,
}

/// Latest physical and logical memory footprint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemorySummary {
    /// Latest resident bytes.
    pub resident_bytes: u64,
    /// Latest logical bytes.
    pub logical_bytes: u64,
    /// Resident divided by logical bytes, or `None` without logical data.
    pub residency_ratio: Option<f64>,
}

/// Memory pressure and reclaim totals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryPressureSummary {
    /// Mean reported pressure in `[0, 1]`.
    pub mean_pressure: UnitInterval,
    /// Sum of reclaim attempts.
    pub reclaim_attempts: u64,
    /// Sum of reclaim failures.
    pub reclaim_failures: u64,
}

/// Compression effectiveness across retained windows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressionSummary {
    /// Sum of logical bytes presented to compression.
    pub logical_bytes: u64,
    /// Sum of compressed bytes produced.
    pub compressed_bytes: u64,
    /// Sum of compression CPU time.
    pub cpu_time_ns: u64,
    /// Compressed divided by logical bytes, or `None` without input.
    pub compression_ratio: Option<f64>,
    /// One minus compression ratio; negative when compression expands input.
    pub space_savings: Option<f64>,
}

/// Bounded offcore latency window and promoted byte total.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OffcoreLatencySummary {
    /// Exact quantiles over retained offcore observations.
    pub latency: LatencySummary,
    /// Sum of promoted or reconstructed bytes from all accepted offcore observations.
    pub promoted_bytes: u64,
}

/// Network and storage byte totals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct IoSummary {
    /// Sum of network counters.
    pub network: crate::signals::NetworkBytes,
    /// Sum of storage counters.
    pub storage: crate::signals::StorageBytes,
}

/// Mean and peak consensus lag.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConsensusLagSummary {
    /// Mean commit-path lag in nanoseconds.
    pub mean_commit_lag_ns: f64,
    /// Maximum commit-path lag in nanoseconds.
    pub max_commit_lag_ns: u64,
    /// Mean slowest-replica lag in nanoseconds.
    pub mean_replica_lag_ns: f64,
    /// Maximum slowest-replica lag in nanoseconds.
    pub max_replica_lag_ns: u64,
}

/// Peak checkpoint debt retained in the observation window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CheckpointDebtSummary {
    /// Maximum bytes awaiting checkpoint publication.
    pub bytes: u64,
    /// Maximum age of checkpoint debt.
    pub oldest_age: Duration,
}

/// Peak repair debt retained in the observation window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RepairDebtSummary {
    /// Maximum assets awaiting repair.
    pub assets: u64,
    /// Maximum bytes awaiting repair.
    pub bytes: u64,
    /// Maximum age of repair debt.
    pub oldest_age: Duration,
}

/// Peak scrub debt retained in the observation window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScrubDebtSummary {
    /// Maximum bytes awaiting scrub.
    pub bytes: u64,
    /// Maximum age of scrub debt.
    pub oldest_age: Duration,
}

/// Peak checkpoint, repair, and scrub debt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaintenanceDebtSummary {
    /// Peak checkpoint debt.
    pub checkpoint: CheckpointDebtSummary,
    /// Peak repair debt.
    pub repair: RepairDebtSummary,
    /// Peak scrub debt.
    pub scrub: ScrubDebtSummary,
}

/// Logical-to-physical redundancy volume and amplification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RedundancySummary {
    /// Maximum logical bytes observed.
    pub logical_bytes: u64,
    /// Maximum physical bytes observed.
    pub physical_bytes: u64,
    /// Maximum fragment count observed.
    pub fragments: u64,
    /// Physical divided by logical bytes, or `None` without logical data.
    pub amplification_ratio: Option<f64>,
}

/// Peak degraded asset count and age.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DegradedAssetsSummary {
    /// Maximum degraded asset count.
    pub assets: u64,
    /// Maximum degraded asset age.
    pub oldest_age: Duration,
}

/// Migration work totals and normalized estimated cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MigrationCostSummary {
    /// Sum of migration bytes.
    pub bytes: u64,
    /// Sum of migration operations.
    pub operations: u64,
    /// Sum of estimated CPU time.
    pub cpu_time_ns: u64,
    /// Sum of estimated I/O time.
    pub io_time_ns: u64,
    /// Estimated CPU nanoseconds per byte, or `None` without bytes.
    pub cpu_ns_per_byte: Option<f64>,
    /// Estimated I/O nanoseconds per byte, or `None` without bytes.
    pub io_ns_per_byte: Option<f64>,
}

/// Latest worker and tablet max-to-mean load ratios.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopologySummary {
    /// Busiest worker load divided by mean worker load.
    pub worker_max_to_mean: Option<f64>,
    /// Busiest tablet load divided by mean tablet load.
    pub tablet_max_to_mean: Option<f64>,
}

/// Bounded hot-key entries and concentration.
#[derive(Debug, Clone, PartialEq)]
pub struct HotKeySummary {
    /// Top-share and Herfindahl concentration.
    pub concentration: HotKeyConcentration,
    /// Entries sorted by descending estimate and ascending key.
    pub entries: Vec<HotKeyEntry>,
}

/// Access-weighted execution criticality and peak observed score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExecutionCriticalitySummary {
    /// Access-weighted mean criticality score.
    pub mean_score: CriticalityScore,
    /// Maximum individual signal score.
    pub peak_score: CriticalityScore,
    /// Number of criticality observations.
    pub observation_count: u64,
    /// Sum of accesses represented by those observations.
    pub access_count: u64,
}

/// Complete typed state for one observation scope.
///
/// Every optional field corresponds to a named domain signal; there is no
/// fallback string, arbitrary metric map, or weakly typed value bag.
#[derive(Debug, Clone, PartialEq)]
pub struct ScopeSnapshot {
    /// Scope represented by this snapshot.
    pub scope: ObservationScope,
    /// Lowest producer window ID observed in the retained window.
    pub first_window: ObservationWindowId,
    /// Window ID of the latest accepted sample.
    pub latest_window: ObservationWindowId,
    /// Sequence of the latest accepted sample.
    pub latest_sequence: ObservationSequence,
    /// Number of accepted samples represented by this scope.
    pub sample_count: usize,
    /// Aggregated provenance and freshness.
    pub evidence: AggregateEvidence,
    /// Request throughput summary.
    pub request: Option<RequestSummary>,
    /// Exact retained end-to-end latency quantiles.
    pub latency: Option<LatencySummary>,
    /// Queue behavior summary.
    pub queueing: Option<QueueingSummary>,
    /// CPU and service time summary.
    pub compute: Option<ComputeSummary>,
    /// Memory footprint summary.
    pub memory: Option<MemorySummary>,
    /// Memory pressure summary.
    pub memory_pressure: Option<MemoryPressureSummary>,
    /// Compression summary.
    pub compression: Option<CompressionSummary>,
    /// Offcore latency summary.
    pub offcore_latency: Option<OffcoreLatencySummary>,
    /// I/O byte summary.
    pub io: Option<IoSummary>,
    /// Consensus lag summary.
    pub consensus_lag: Option<ConsensusLagSummary>,
    /// Maintenance debt summary.
    pub maintenance_debt: Option<MaintenanceDebtSummary>,
    /// Redundancy amplification summary.
    pub redundancy: Option<RedundancySummary>,
    /// Degraded asset summary.
    pub degraded_assets: Option<DegradedAssetsSummary>,
    /// Migration cost summary.
    pub migration_cost: Option<MigrationCostSummary>,
    /// Topology imbalance summary.
    pub topology: Option<TopologySummary>,
    /// Hot-key sketch output.
    pub hot_keys: HotKeySummary,
    /// Execution criticality summary.
    pub execution_criticality: Option<ExecutionCriticalitySummary>,
}

/// Deterministic point-in-time observation snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationSnapshot {
    /// Caller-supplied monotonic time used for freshness.
    pub as_of: Ticks,
    /// Last accepted monotonic sequence.
    pub last_sequence: Option<ObservationSequence>,
    /// Lifetime number of accepted samples.
    pub accepted_samples: u64,
    /// Scope snapshots sorted by [`ObservationScope`] ordering.
    pub scopes: Vec<ScopeSnapshot>,
}

/// Bounded collector that combines typed signals by scope.
#[derive(Debug)]
pub struct ObservationFabric {
    limits: ObservationLimits,
    scopes: BTreeMap<ObservationScope, ScopeState>,
    last_sequence: Option<ObservationSequence>,
    accepted_samples: u64,
}

impl ObservationFabric {
    /// Creates an empty fabric with fixed retention limits.
    #[must_use]
    pub fn new(limits: ObservationLimits) -> Self {
        Self {
            limits,
            scopes: BTreeMap::new(),
            last_sequence: None,
            accepted_samples: 0,
        }
    }

    /// Returns this fabric's immutable limits.
    #[must_use]
    pub const fn limits(&self) -> &ObservationLimits {
        &self.limits
    }

    /// Returns the number of retained scopes.
    #[must_use]
    pub fn scope_count(&self) -> usize {
        self.scopes.len()
    }

    /// Returns the number of samples currently retained across scopes.
    #[must_use]
    pub fn retained_samples(&self) -> usize {
        self.scopes.values().fold(0_usize, |total, state| {
            total.saturating_add(state.sample_count())
        })
    }

    /// Returns the last accepted sequence.
    #[must_use]
    pub const fn last_sequence(&self) -> Option<ObservationSequence> {
        self.last_sequence
    }

    /// Accepts one strictly sequenced typed sample.
    ///
    /// Rejected samples do not consume sequence numbers or mutate retained
    /// state. Once a scope reaches its sample limit, later samples for that
    /// scope are rejected until the scope is removed.
    ///
    /// # Errors
    ///
    /// Returns [`CollectError`] for non-monotonic sequences, scope or sample
    /// capacity exhaustion, or an exhausted hot-key counter.
    pub fn collect(&mut self, sample: &ObservationSample) -> Result<(), CollectError> {
        if self
            .last_sequence
            .is_some_and(|last| sample.stamp.sequence <= last)
        {
            return Err(CollectError::NonMonotonicSequence {
                last: self.last_sequence,
                received: sample.stamp.sequence,
            });
        }
        let is_new_scope = !self.scopes.contains_key(&sample.scope);
        if is_new_scope && self.scopes.len() == self.limits.max_scopes {
            return Err(CollectError::ScopeCapacity {
                limit: self.limits.max_scopes,
            });
        }
        if is_new_scope {
            self.scopes.insert(
                sample.scope,
                ScopeState::new(&sample.stamp, &sample.metadata, self.limits),
            );
        }
        let Some(state) = self.scopes.get_mut(&sample.scope) else {
            self.scopes.remove(&sample.scope);
            return Err(CollectError::ScopeCapacity {
                limit: self.limits.max_scopes,
            });
        };
        if state.sample_count() == self.limits.max_samples_per_scope {
            if is_new_scope {
                self.scopes.remove(&sample.scope);
            }
            return Err(CollectError::SampleCapacity {
                scope: sample.scope,
                limit: self.limits.max_samples_per_scope,
            });
        }
        if let Err(error) = state.apply(sample) {
            if is_new_scope {
                self.scopes.remove(&sample.scope);
            }
            return Err(match error {
                ApplyError::HotKey => CollectError::HotKeyCounterExhausted,
            });
        }
        self.last_sequence = Some(sample.stamp.sequence);
        self.accepted_samples = self.accepted_samples.saturating_add(1);
        Ok(())
    }

    /// Begins a new rolling window by dropping retained per-scope aggregates.
    ///
    /// The caller then uses strict [`collect`](Self::collect) for every
    /// sample in that window. This keeps multi-signal scopes intact while
    /// making long-running producers bounded.
    pub fn begin_window(&mut self) {
        self.scopes.clear();
    }

    /// Accepts one sample after explicitly beginning a rolling window.
    ///
    /// This convenience form is intended for producers that emit one sample
    /// per scope per window. It clears that scope and delegates to strict
    /// collection.
    ///
    /// # Errors
    ///
    /// Returns [`CollectError`] for non-monotonic sequences or a new-scope
    /// capacity failure.
    pub fn collect_rolling(&mut self, sample: &ObservationSample) -> Result<(), CollectError> {
        self.scopes.remove(&sample.scope);
        self.collect(sample)
    }

    /// Removes one scope and returns whether it existed.
    pub fn remove_scope(&mut self, scope: &ObservationScope) -> bool {
        self.scopes.remove(scope).is_some()
    }

    /// Emits deterministic snapshots sorted by scope and hot-key ordering.
    #[must_use]
    pub fn snapshot(&self, as_of: Ticks) -> ObservationSnapshot {
        ObservationSnapshot {
            as_of,
            last_sequence: self.last_sequence,
            accepted_samples: self.accepted_samples,
            scopes: self
                .scopes
                .iter()
                .map(|(scope, state)| state.snapshot(*scope, as_of, self.limits.freshness_budget))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ApplyError {
    HotKey,
}

#[derive(Debug)]
struct ScopeState {
    first_window: ObservationWindowId,
    latest_window: ObservationWindowId,
    latest_sequence: ObservationSequence,
    sample_count: usize,
    oldest_observed_at: Ticks,
    latest_observed_at: Ticks,
    sources: ObservationSources,
    confidence: Confidence,
    request: RequestAccumulator,
    latency: LatencyWindow,
    queueing: QueueingAccumulator,
    compute: ComputeAccumulator,
    memory: Option<MemorySignal>,
    memory_pressure: PressureAccumulator,
    compression: CompressionAccumulator,
    offcore_bytes: u64,
    offcore_window: LatencyWindow,
    io: IoAccumulator,
    consensus_lag: ConsensusAccumulator,
    maintenance_debt: Option<crate::signals::MaintenanceDebtSignal>,
    redundancy: Option<RedundancySignal>,
    degraded_assets: Option<DegradedAssetsSignal>,
    migration_cost: MigrationAccumulator,
    topology: Option<TopologySummary>,
    hot_keys: HotKeySketch,
    criticality: CriticalityAccumulator,
}

impl ScopeState {
    fn new(
        stamp: &crate::window::ObservationStamp,
        metadata: &ObservationMetadata,
        limits: ObservationLimits,
    ) -> Self {
        Self {
            first_window: stamp.window,
            latest_window: stamp.window,
            latest_sequence: stamp.sequence,
            sample_count: 0,
            oldest_observed_at: metadata.observed_at,
            latest_observed_at: metadata.observed_at,
            sources: ObservationSources::from_source(metadata.source),
            confidence: metadata.confidence,
            request: RequestAccumulator::default(),
            latency: LatencyWindow::new(limits.max_latency_samples),
            queueing: QueueingAccumulator::default(),
            compute: ComputeAccumulator::default(),
            memory: None,
            memory_pressure: PressureAccumulator::default(),
            compression: CompressionAccumulator::default(),
            offcore_bytes: 0,
            offcore_window: LatencyWindow::new(limits.max_latency_samples),
            io: IoAccumulator::default(),
            consensus_lag: ConsensusAccumulator::default(),
            maintenance_debt: None,
            redundancy: None,
            degraded_assets: None,
            migration_cost: MigrationAccumulator::default(),
            topology: None,
            hot_keys: HotKeySketch::new(limits.hot_key_capacity)
                .expect("validated hot-key capacity is nonzero"),
            criticality: CriticalityAccumulator::default(),
        }
    }

    const fn sample_count(&self) -> usize {
        self.sample_count
    }

    fn apply(&mut self, sample: &ObservationSample) -> Result<(), ApplyError> {
        match sample.signal {
            TypedSignal::Request(signal) => self.request.observe(&signal, sample.period),
            TypedSignal::Latency(signal) => self.latency.observe(signal),
            TypedSignal::Queueing(signal) => self.queueing.observe(&signal),
            TypedSignal::Compute(signal) => self.compute.observe(&signal),
            TypedSignal::Memory(signal) => self.memory = Some(signal),
            TypedSignal::MemoryPressure(signal) => self.memory_pressure.observe(&signal),
            TypedSignal::Compression(signal) => self.compression.observe(&signal),
            TypedSignal::OffcoreLatency(signal) => {
                self.offcore_window.observe_value(signal.duration_ns);
                self.offcore_bytes = self.offcore_bytes.saturating_add(signal.promoted_bytes);
            }
            TypedSignal::Io(signal) => self.io.observe(&signal),
            TypedSignal::ConsensusLag(signal) => self.consensus_lag.observe(&signal),
            TypedSignal::MaintenanceDebt(signal) => {
                self.maintenance_debt = Some(merge_maintenance_debt(self.maintenance_debt, signal));
            }
            TypedSignal::Redundancy(signal) => {
                self.redundancy = Some(merge_redundancy(self.redundancy, signal));
            }
            TypedSignal::DegradedAssets(signal) => {
                self.degraded_assets = Some(merge_degraded_assets(self.degraded_assets, signal));
            }
            TypedSignal::MigrationCost(signal) => self.migration_cost.observe(&signal),
            TypedSignal::TopologyImbalance(signal) => self.topology = Some(topology(&signal)),
            TypedSignal::HotKey(signal) => self
                .hot_keys
                .observe(&signal)
                .map_err(|_| ApplyError::HotKey)?,
            TypedSignal::ExecutionCriticality(signal) => self.criticality.observe(&signal),
        }
        self.oldest_observed_at = self.oldest_observed_at.min(sample.metadata.observed_at);
        self.latest_observed_at = self.latest_observed_at.max(sample.metadata.observed_at);
        self.sources.insert(sample.metadata.source);
        self.confidence = self.confidence.min(sample.metadata.confidence);
        self.first_window = self.first_window.min(sample.stamp.window);
        self.latest_window = sample.stamp.window;
        self.latest_sequence = sample.stamp.sequence;
        self.sample_count = self.sample_count.saturating_add(1);
        Ok(())
    }

    fn snapshot(
        &self,
        scope: ObservationScope,
        as_of: Ticks,
        freshness_budget: Duration,
    ) -> ScopeSnapshot {
        let evidence = AggregateEvidence {
            oldest_observed_at: self.oldest_observed_at,
            latest_observed_at: self.latest_observed_at,
            sources: self.sources,
            confidence: self.confidence,
            freshness: Freshness::classify(self.latest_observed_at, as_of, freshness_budget),
        };
        ScopeSnapshot {
            scope,
            first_window: self.first_window,
            latest_window: self.latest_window,
            latest_sequence: self.latest_sequence,
            sample_count: self.sample_count,
            evidence,
            request: self.request.summary(),
            latency: self.latency.summary(),
            queueing: self.queueing.summary(),
            compute: self.compute.summary(),
            memory: self.memory.as_ref().map(|signal| MemorySummary {
                resident_bytes: signal.resident_bytes,
                logical_bytes: signal.logical_bytes,
                residency_ratio: nonunit_ratio(signal.resident_bytes, signal.logical_bytes),
            }),
            memory_pressure: self.memory_pressure.summary(),
            compression: self.compression.summary(),
            offcore_latency: self
                .offcore_window
                .summary()
                .map(|latency| OffcoreLatencySummary {
                    latency,
                    promoted_bytes: self.offcore_bytes,
                }),
            io: self.io.summary(),
            consensus_lag: self.consensus_lag.summary(),
            maintenance_debt: self.maintenance_debt.map(|signal| MaintenanceDebtSummary {
                checkpoint: CheckpointDebtSummary {
                    bytes: signal.checkpoint.bytes,
                    oldest_age: signal.checkpoint.oldest_age,
                },
                repair: RepairDebtSummary {
                    assets: signal.repair.assets,
                    bytes: signal.repair.bytes,
                    oldest_age: signal.repair.oldest_age,
                },
                scrub: ScrubDebtSummary {
                    bytes: signal.scrub.bytes,
                    oldest_age: signal.scrub.oldest_age,
                },
            }),
            redundancy: self.redundancy.as_ref().map(|signal| RedundancySummary {
                logical_bytes: signal.logical_bytes,
                physical_bytes: signal.physical_bytes,
                fragments: signal.fragments,
                amplification_ratio: nonunit_ratio(signal.physical_bytes, signal.logical_bytes),
            }),
            degraded_assets: self
                .degraded_assets
                .as_ref()
                .map(|signal| DegradedAssetsSummary {
                    assets: signal.assets,
                    oldest_age: signal.oldest_age,
                }),
            migration_cost: self.migration_cost.summary(),
            topology: self.topology,
            hot_keys: HotKeySummary {
                concentration: self.hot_keys.concentration(),
                entries: self.hot_keys.entries(),
            },
            execution_criticality: self.criticality.summary(),
        }
    }
}

#[derive(Debug, Default)]
struct RequestAccumulator {
    reads: u128,
    writes: u128,
    period_micros: u128,
    samples: u64,
}

impl RequestAccumulator {
    fn observe(&mut self, signal: &RequestSignal, period: crate::window::ObservationPeriod) {
        self.reads = self.reads.saturating_add(u128::from(signal.reads));
        self.writes = self.writes.saturating_add(u128::from(signal.writes));
        self.period_micros = self.period_micros.saturating_add(period.micros());
        self.samples = self.samples.saturating_add(1);
    }

    #[allow(clippy::cast_precision_loss)]
    fn summary(&self) -> Option<RequestSummary> {
        if self.samples == 0 {
            return None;
        }
        let seconds = self.period_micros as f64 / 1_000_000.0;
        let requests = self.reads.saturating_add(self.writes);
        let ratio = ReadWriteRatio::from_counts_u128(self.reads, self.writes);
        Some(RequestSummary {
            requests_per_second: requests as f64 / seconds,
            reads_per_second: self.reads as f64 / seconds,
            writes_per_second: self.writes as f64 / seconds,
            read_ratio: ratio,
        })
    }
}

#[derive(Debug, Default)]
struct Mean {
    total: f64,
    count: u64,
}

impl Mean {
    fn observe(&mut self, value: f64) {
        self.total = if self.total.is_finite() {
            (self.total + value).min(f64::MAX)
        } else {
            f64::MAX
        };
        self.count = self.count.saturating_add(1);
    }

    #[allow(clippy::cast_precision_loss)]
    fn value(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total / self.count as f64
        }
    }
}

#[derive(Debug, Default)]
struct ComputeAccumulator {
    cpu: Mean,
    service: Mean,
    total_cpu: u128,
    total_service: u128,
}

impl ComputeAccumulator {
    fn observe(&mut self, signal: &ComputeSignal) {
        self.cpu.observe(u64_f64(signal.cpu_time_ns));
        self.service.observe(u64_f64(signal.service_time_ns));
        self.total_cpu = self
            .total_cpu
            .saturating_add(u128::from(signal.cpu_time_ns));
        self.total_service = self
            .total_service
            .saturating_add(u128::from(signal.service_time_ns));
    }

    fn summary(&self) -> Option<ComputeSummary> {
        (self.cpu.count > 0).then(|| ComputeSummary {
            mean_cpu_time_ns: self.cpu.value(),
            mean_service_time_ns: self.service.value(),
            cpu_to_service_ratio: nonunit_ratio_u128(self.total_cpu, self.total_service),
        })
    }
}

#[derive(Debug, Default)]
struct QueueingAccumulator {
    count: u64,
    max_depth: u64,
    p50: Mean,
    p99: Mean,
    p999: Mean,
    rejections: u64,
}

impl QueueingAccumulator {
    fn observe(&mut self, signal: &QueueingSignal) {
        self.count = self.count.saturating_add(1);
        self.max_depth = self.max_depth.max(signal.depth);
        self.p50.observe(u64_f64(signal.wait_p50_ns));
        self.p99.observe(u64_f64(signal.wait_p99_ns));
        self.p999.observe(u64_f64(signal.wait_p999_ns));
        self.rejections = self.rejections.saturating_add(signal.admission_rejections);
    }

    fn summary(&self) -> Option<QueueingSummary> {
        (self.count > 0).then(|| QueueingSummary {
            max_depth: self.max_depth,
            mean_wait_p50_ns: self.p50.value(),
            mean_wait_p99_ns: self.p99.value(),
            mean_wait_p999_ns: self.p999.value(),
            admission_rejections: self.rejections,
        })
    }
}

#[derive(Debug, Default)]
struct PressureAccumulator {
    mean: Mean,
    attempts: u64,
    failures: u64,
}

impl PressureAccumulator {
    fn observe(&mut self, signal: &MemoryPressureSignal) {
        self.mean.observe(signal.pressure.as_f64());
        self.attempts = self.attempts.saturating_add(signal.reclaim_attempts);
        self.failures = self.failures.saturating_add(signal.reclaim_failures);
    }

    fn summary(&self) -> Option<MemoryPressureSummary> {
        (self.mean.count > 0).then(|| MemoryPressureSummary {
            mean_pressure: UnitInterval::new(self.mean.value()).unwrap_or(UnitInterval::ZERO),
            reclaim_attempts: self.attempts,
            reclaim_failures: self.failures,
        })
    }
}

#[derive(Debug, Default)]
struct CompressionAccumulator {
    logical: u128,
    compressed: u128,
    cpu: u128,
    present: bool,
}

impl CompressionAccumulator {
    fn observe(&mut self, signal: &CompressionSignal) {
        self.logical = self
            .logical
            .saturating_add(u128::from(signal.logical_bytes));
        self.compressed = self
            .compressed
            .saturating_add(u128::from(signal.compressed_bytes));
        self.cpu = self.cpu.saturating_add(u128::from(signal.cpu_time_ns));
        self.present = true;
    }

    fn summary(&self) -> Option<CompressionSummary> {
        if !self.present {
            return None;
        }
        let ratio = nonunit_ratio_u128(self.compressed, self.logical);
        Some(CompressionSummary {
            logical_bytes: u64_saturating(self.logical),
            compressed_bytes: u64_saturating(self.compressed),
            cpu_time_ns: u64_saturating(self.cpu),
            compression_ratio: ratio,
            space_savings: ratio.map(|value| 1.0 - value),
        })
    }
}

#[derive(Debug, Default)]
struct MigrationAccumulator {
    bytes: u128,
    operations: u128,
    cpu: u128,
    io: u128,
    present: bool,
}

impl MigrationAccumulator {
    fn observe(&mut self, signal: &MigrationCostSignal) {
        self.bytes = self.bytes.saturating_add(u128::from(signal.bytes));
        self.operations = self
            .operations
            .saturating_add(u128::from(signal.operations));
        self.cpu = self.cpu.saturating_add(u128::from(signal.cpu_time_ns));
        self.io = self.io.saturating_add(u128::from(signal.io_time_ns));
        self.present = true;
    }

    fn summary(&self) -> Option<MigrationCostSummary> {
        if !self.present {
            return None;
        }
        Some(MigrationCostSummary {
            bytes: u64_saturating(self.bytes),
            operations: u64_saturating(self.operations),
            cpu_time_ns: u64_saturating(self.cpu),
            io_time_ns: u64_saturating(self.io),
            cpu_ns_per_byte: nonunit_ratio_u128(self.cpu, self.bytes),
            io_ns_per_byte: nonunit_ratio_u128(self.io, self.bytes),
        })
    }
}

#[derive(Debug, Default)]
struct IoAccumulator {
    count: u64,
    network_received: u128,
    network_transmitted: u128,
    storage_read: u128,
    storage_written: u128,
    storage_synced: u128,
}

impl IoAccumulator {
    fn observe(&mut self, signal: &IoSignal) {
        self.count = self.count.saturating_add(1);
        self.network_received = self
            .network_received
            .saturating_add(u128::from(signal.network.received));
        self.network_transmitted = self
            .network_transmitted
            .saturating_add(u128::from(signal.network.transmitted));
        self.storage_read = self
            .storage_read
            .saturating_add(u128::from(signal.storage.read));
        self.storage_written = self
            .storage_written
            .saturating_add(u128::from(signal.storage.written));
        self.storage_synced = self
            .storage_synced
            .saturating_add(u128::from(signal.storage.synced));
    }

    fn summary(&self) -> Option<IoSummary> {
        if self.count == 0 {
            return None;
        }
        Some(IoSummary {
            network: crate::signals::NetworkBytes {
                received: u64_saturating(self.network_received),
                transmitted: u64_saturating(self.network_transmitted),
            },
            storage: crate::signals::StorageBytes {
                read: u64_saturating(self.storage_read),
                written: u64_saturating(self.storage_written),
                synced: u64_saturating(self.storage_synced),
            },
        })
    }
}

#[derive(Debug, Default)]
struct ConsensusAccumulator {
    count: u64,
    mean_commit: Mean,
    mean_replica: Mean,
    max_commit: u64,
    max_replica: u64,
}

impl ConsensusAccumulator {
    fn observe(&mut self, signal: &ConsensusLagSignal) {
        self.count = self.count.saturating_add(1);
        self.mean_commit.observe(u64_f64(signal.commit_lag_ns));
        self.mean_replica.observe(u64_f64(signal.replica_lag_ns));
        self.max_commit = self.max_commit.max(signal.commit_lag_ns);
        self.max_replica = self.max_replica.max(signal.replica_lag_ns);
    }

    fn summary(&self) -> Option<ConsensusLagSummary> {
        (self.count > 0).then(|| ConsensusLagSummary {
            mean_commit_lag_ns: self.mean_commit.value(),
            max_commit_lag_ns: self.max_commit,
            mean_replica_lag_ns: self.mean_replica.value(),
            max_replica_lag_ns: self.max_replica,
        })
    }
}

#[derive(Debug, Default)]
struct CriticalityAccumulator {
    observations: u64,
    accesses: u64,
    weighted_score: u128,
    peak: CriticalityScore,
    present: bool,
}

impl CriticalityAccumulator {
    fn observe(&mut self, signal: &ExecutionCriticalitySignal) {
        let approximation = ExecutionCriticalityApproximation::from_signal(signal);
        self.present = true;
        self.observations = self.observations.saturating_add(1);
        self.accesses = self.accesses.saturating_add(signal.accesses);
        self.weighted_score = self.weighted_score.saturating_add(
            u128::from(approximation.score.as_u64()).saturating_mul(u128::from(signal.accesses)),
        );
        self.peak = self.peak.max(approximation.score);
    }

    fn summary(&self) -> Option<ExecutionCriticalitySummary> {
        if !self.present {
            return None;
        }
        let mean = if self.accesses == 0 {
            0
        } else {
            self.weighted_score / u128::from(self.accesses)
        };
        Some(ExecutionCriticalitySummary {
            mean_score: CriticalityScore::from(u64_saturating(mean)),
            peak_score: self.peak,
            observation_count: self.observations,
            access_count: self.accesses,
        })
    }
}

fn merge_maintenance_debt(
    existing: Option<MaintenanceDebtSignal>,
    incoming: MaintenanceDebtSignal,
) -> MaintenanceDebtSignal {
    existing.map_or(incoming, |current| MaintenanceDebtSignal {
        checkpoint: crate::signals::CheckpointDebt {
            bytes: current.checkpoint.bytes.max(incoming.checkpoint.bytes),
            oldest_age: current
                .checkpoint
                .oldest_age
                .max(incoming.checkpoint.oldest_age),
        },
        repair: crate::signals::RepairDebt {
            assets: current.repair.assets.max(incoming.repair.assets),
            bytes: current.repair.bytes.max(incoming.repair.bytes),
            oldest_age: current.repair.oldest_age.max(incoming.repair.oldest_age),
        },
        scrub: crate::signals::ScrubDebt {
            bytes: current.scrub.bytes.max(incoming.scrub.bytes),
            oldest_age: current.scrub.oldest_age.max(incoming.scrub.oldest_age),
        },
    })
}

fn merge_redundancy(
    existing: Option<RedundancySignal>,
    incoming: RedundancySignal,
) -> RedundancySignal {
    existing.map_or(incoming, |current| RedundancySignal {
        logical_bytes: current.logical_bytes.max(incoming.logical_bytes),
        physical_bytes: current.physical_bytes.max(incoming.physical_bytes),
        fragments: current.fragments.max(incoming.fragments),
    })
}

fn merge_degraded_assets(
    existing: Option<DegradedAssetsSignal>,
    incoming: DegradedAssetsSignal,
) -> DegradedAssetsSignal {
    existing.map_or(incoming, |current| DegradedAssetsSignal {
        assets: current.assets.max(incoming.assets),
        oldest_age: current.oldest_age.max(incoming.oldest_age),
    })
}

fn topology(signal: &TopologyImbalanceSignal) -> TopologySummary {
    TopologySummary {
        worker_max_to_mean: nonunit_ratio(signal.busiest_worker_load, signal.mean_worker_load),
        tablet_max_to_mean: nonunit_ratio(signal.busiest_tablet_load, signal.mean_tablet_load),
    }
}

#[allow(clippy::cast_precision_loss)]
fn u64_f64(value: u64) -> f64 {
    value as f64
}

fn u64_saturating(value: u128) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[allow(clippy::cast_precision_loss)]
fn nonunit_ratio(numerator: u64, denominator: u64) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

#[allow(clippy::cast_precision_loss)]
fn nonunit_ratio_u128(numerator: u128, denominator: u128) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

#[cfg(test)]
mod tests {
    use core::time::Duration;

    use kivi_types::{ClusterId, WorkerId};

    use super::*;
    use crate::metadata::{FreshnessStatus, ObservationSource};
    use crate::signals::{
        CheckpointDebt, HotKeySignal, MaintenanceDebtSignal, NetworkBytes, RepairDebt, ScrubDebt,
        StorageBytes,
    };
    use crate::sketch::HotKeyId;
    use crate::window::{ObservationPeriod, ObservationStamp};

    fn limits(max_scopes: usize, max_samples: usize, max_latency: usize) -> ObservationLimits {
        ObservationLimits::new(
            max_scopes,
            max_samples,
            max_latency,
            4,
            Duration::from_secs(1),
        )
        .expect("nonzero limits")
    }

    fn sample(scope: ObservationScope, sequence: u64, signal: TypedSignal) -> ObservationSample {
        ObservationSample::new(
            scope,
            ObservationStamp {
                window: ObservationWindowId::from_u64(9),
                sequence: ObservationSequence::from_u64(sequence),
            },
            ObservationPeriod::new(Duration::from_secs(1)).expect("period"),
            ObservationMetadata::new(
                Ticks::from_micros(100),
                ObservationSource::RuntimeCounter,
                Confidence::High,
            ),
            signal,
        )
    }

    fn worker(id: u64) -> ObservationScope {
        ObservationScope::Worker(WorkerId::from_u64(id))
    }

    #[test]
    fn limits_reject_unbounded_configurations() {
        assert!(ObservationLimits::new(0, 1, 1, 1, Duration::ZERO).is_none());
        assert!(ObservationLimits::new(1, 0, 1, 1, Duration::ZERO).is_none());
        assert!(ObservationLimits::new(1, 1, 0, 1, Duration::ZERO).is_none());
        assert!(ObservationLimits::new(1, 1, 2, 1, Duration::ZERO).is_none());
        assert!(ObservationLimits::new(1, 1, 1, 0, Duration::ZERO).is_none());
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn all_typed_signals_aggregate_without_a_metric_bag() {
        let pressure = UnitInterval::new(0.75).expect("pressure");
        let signals = [
            TypedSignal::Request(RequestSignal {
                reads: 10,
                writes: 5,
            }),
            TypedSignal::Latency(crate::signals::LatencySignal { duration_ns: 100 }),
            TypedSignal::Queueing(QueueingSignal {
                depth: 7,
                wait_p50_ns: 10,
                wait_p99_ns: 20,
                wait_p999_ns: 30,
                admission_rejections: 2,
            }),
            TypedSignal::Compute(ComputeSignal {
                cpu_time_ns: 40,
                service_time_ns: 80,
            }),
            TypedSignal::Memory(MemorySignal {
                resident_bytes: 1_250,
                logical_bytes: 1_000,
            }),
            TypedSignal::MemoryPressure(MemoryPressureSignal {
                pressure,
                reclaim_attempts: 3,
                reclaim_failures: 1,
            }),
            TypedSignal::Compression(CompressionSignal {
                logical_bytes: 1_000,
                compressed_bytes: 400,
                cpu_time_ns: 50,
            }),
            TypedSignal::OffcoreLatency(crate::signals::OffcoreLatencySignal {
                duration_ns: 900,
                promoted_bytes: 512,
            }),
            TypedSignal::Io(IoSignal {
                network: NetworkBytes {
                    received: 10,
                    transmitted: 20,
                },
                storage: StorageBytes {
                    read: 30,
                    written: 40,
                    synced: 50,
                },
            }),
            TypedSignal::ConsensusLag(ConsensusLagSignal {
                commit_lag_ns: 60,
                replica_lag_ns: 70,
            }),
            TypedSignal::MaintenanceDebt(MaintenanceDebtSignal {
                checkpoint: CheckpointDebt {
                    bytes: 80,
                    oldest_age: Duration::from_secs(1),
                },
                repair: RepairDebt {
                    assets: 2,
                    bytes: 90,
                    oldest_age: Duration::from_secs(2),
                },
                scrub: ScrubDebt {
                    bytes: 100,
                    oldest_age: Duration::from_secs(3),
                },
            }),
            TypedSignal::Redundancy(RedundancySignal {
                logical_bytes: 1_000,
                physical_bytes: 1_500,
                fragments: 3,
            }),
            TypedSignal::DegradedAssets(DegradedAssetsSignal {
                assets: 2,
                oldest_age: Duration::from_secs(4),
            }),
            TypedSignal::MigrationCost(MigrationCostSignal {
                bytes: 256,
                operations: 2,
                cpu_time_ns: 512,
                io_time_ns: 1_024,
            }),
            TypedSignal::TopologyImbalance(TopologyImbalanceSignal {
                busiest_worker_load: 200,
                mean_worker_load: 100,
                busiest_tablet_load: 150,
                mean_tablet_load: 75,
            }),
            TypedSignal::HotKey(HotKeySignal {
                key: HotKeyId::from_u128(42),
                on_critical_path: true,
                affects_tail: false,
                stall_time_ns: 80,
            }),
            TypedSignal::ExecutionCriticality(ExecutionCriticalitySignal {
                accesses: 4,
                serial_accesses: 2,
                critical_path_accesses: 2,
                tail_accesses: 1,
                stall_time_ns: 100,
                access_latency_ns: 200,
                logical_bytes: 64,
            }),
        ];
        let mut fabric = ObservationFabric::new(limits(1, 32, 4));
        for (index, signal) in signals.into_iter().enumerate() {
            let sequence = u64::try_from(index + 1).expect("sequence");
            fabric
                .collect(&sample(worker(1), sequence, signal))
                .expect("collect");
        }
        let snapshot = fabric.snapshot(Ticks::from_micros(1_000));
        let state = snapshot.scopes.first().expect("worker state");
        assert_eq!(state.sample_count, 17);
        assert_eq!(state.evidence.confidence, Confidence::High);
        assert_eq!(state.evidence.freshness.status, FreshnessStatus::Current);
        assert!(
            state
                .evidence
                .sources
                .contains(ObservationSource::RuntimeCounter)
        );
        assert_eq!(
            state.request.expect("request").read_ratio,
            ReadWriteRatio::from_counts(10, 5)
        );
        assert_eq!(state.latency.expect("latency").p999_ns, 100);
        assert!(state.queueing.is_some());
        assert!(state.compute.is_some());
        assert_eq!(state.memory.expect("memory").residency_ratio, Some(1.25));
        assert!(state.memory_pressure.is_some());
        assert_eq!(
            state.compression.expect("compression").space_savings,
            Some(0.6)
        );
        assert_eq!(state.offcore_latency.expect("offcore").promoted_bytes, 512);
        assert_eq!(state.io.expect("io").network.received, 10);
        assert!(state.consensus_lag.is_some());
        assert_eq!(state.maintenance_debt.expect("debt").repair.assets, 2);
        assert_eq!(
            state.redundancy.expect("redundancy").amplification_ratio,
            Some(1.5)
        );
        assert_eq!(state.degraded_assets.expect("degraded").assets, 2);
        assert_eq!(state.migration_cost.expect("migration").bytes, 256);
        assert_eq!(
            state.topology.expect("topology").worker_max_to_mean,
            Some(2.0)
        );
        assert_eq!(state.hot_keys.entries.len(), 1);
        assert!(state.execution_criticality.is_some());
    }

    #[test]
    fn scope_and_sample_capacity_are_hard_bounds() {
        let mut fabric = ObservationFabric::new(limits(1, 2, 2));
        let signal = TypedSignal::Memory(MemorySignal {
            resident_bytes: 1,
            logical_bytes: 1,
        });
        fabric
            .collect(&sample(worker(1), 1, signal))
            .expect("first");
        fabric
            .collect(&sample(worker(1), 2, signal))
            .expect("second");
        assert_eq!(
            fabric.collect(&sample(worker(1), 3, signal)),
            Err(CollectError::SampleCapacity {
                scope: worker(1),
                limit: 2,
            })
        );
        assert_eq!(
            fabric.last_sequence(),
            Some(ObservationSequence::from_u64(2))
        );
        assert_eq!(fabric.retained_samples(), 2);
        assert!(fabric.remove_scope(&worker(1)));
        fabric
            .collect(&sample(worker(2), 3, signal))
            .expect("replacement scope");
        assert_eq!(
            fabric.collect(&sample(worker(3), 4, signal)),
            Err(CollectError::ScopeCapacity { limit: 1 })
        );
        assert_eq!(fabric.scope_count(), 1);
        assert_eq!(fabric.retained_samples(), 1);
    }

    #[test]
    fn latency_retention_is_a_bounded_rolling_window() {
        let mut fabric = ObservationFabric::new(limits(1, 8, 3));
        for value in 1..=8 {
            fabric
                .collect(&sample(
                    worker(1),
                    value,
                    TypedSignal::Latency(crate::signals::LatencySignal { duration_ns: value }),
                ))
                .expect("latency");
        }
        let snapshot = fabric.snapshot(Ticks::from_micros(1_000));
        let latency = snapshot
            .scopes
            .first()
            .expect("scope")
            .latency
            .expect("latency");
        assert_eq!(latency.sample_count, 3);
        assert_eq!(latency.p50_ns, 7);
        assert_eq!(latency.max_ns, 8);
    }

    #[test]
    fn replay_is_deterministic_and_scopes_are_sorted() {
        let first_order = [
            (
                worker(1),
                TypedSignal::HotKey(HotKeySignal {
                    key: HotKeyId::from_u128(1),
                    on_critical_path: false,
                    affects_tail: false,
                    stall_time_ns: 0,
                }),
            ),
            (
                ObservationScope::Cluster(ClusterId::from_u128(3)),
                TypedSignal::Request(RequestSignal {
                    reads: 1,
                    writes: 1,
                }),
            ),
            (
                worker(2),
                TypedSignal::HotKey(HotKeySignal {
                    key: HotKeyId::from_u128(2),
                    on_critical_path: true,
                    affects_tail: false,
                    stall_time_ns: 10,
                }),
            ),
        ];
        let second_order = first_order;
        let mut first = ObservationFabric::new(limits(4, 8, 4));
        let mut second = ObservationFabric::new(limits(4, 8, 4));
        for (index, (scope, signal)) in first_order.into_iter().enumerate() {
            first
                .collect(&sample(
                    scope,
                    u64::try_from(index + 1).expect("sequence"),
                    signal,
                ))
                .expect("first replay");
        }
        for (index, (scope, signal)) in second_order.into_iter().enumerate() {
            second
                .collect(&sample(
                    scope,
                    u64::try_from(index + 1).expect("sequence"),
                    signal,
                ))
                .expect("second replay");
        }
        let first_snapshot = first.snapshot(Ticks::from_micros(1_000));
        let second_snapshot = second.snapshot(Ticks::from_micros(1_000));
        assert_eq!(first_snapshot, second_snapshot);
        assert_eq!(first_snapshot.scopes[0].scope, worker(1));
        assert_eq!(first_snapshot.scopes[1].scope, worker(2));
        assert_eq!(
            first_snapshot.scopes[2].scope,
            ObservationScope::Cluster(ClusterId::from_u128(3))
        );
    }

    #[test]
    fn rejected_sequences_do_not_mutate_state() {
        let mut fabric = ObservationFabric::new(limits(1, 4, 2));
        let signal = TypedSignal::Memory(MemorySignal {
            resident_bytes: 7,
            logical_bytes: 7,
        });
        fabric
            .collect(&sample(worker(1), 2, signal))
            .expect("first");
        assert!(matches!(
            fabric.collect(&sample(worker(1), 1, signal)),
            Err(CollectError::NonMonotonicSequence { .. })
        ));
        assert!(matches!(
            fabric.collect(&sample(worker(1), 2, signal)),
            Err(CollectError::NonMonotonicSequence { .. })
        ));
        assert_eq!(
            fabric.last_sequence(),
            Some(ObservationSequence::from_u64(2))
        );
        assert_eq!(fabric.retained_samples(), 1);
        assert_eq!(
            fabric.snapshot(Ticks::from_micros(1_000)).scopes[0]
                .memory
                .expect("memory")
                .resident_bytes,
            7
        );
    }
}
