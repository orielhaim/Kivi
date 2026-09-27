//! Closed, strongly typed observation signal schema.

use kivi_types::Ticks;

use crate::metadata::ObservationMetadata;
use crate::scope::ObservationScope;
use crate::sketch::HotKeyId;
use crate::window::{ObservationPeriod, ObservationStamp};

/// A finite value constrained to the closed interval `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct UnitInterval(f64);

impl UnitInterval {
    /// Zero.
    pub const ZERO: Self = Self(0.0);
    /// One.
    pub const ONE: Self = Self(1.0);

    /// Creates a value when `value` is finite and in `[0, 1]`.
    #[must_use]
    pub fn new(value: f64) -> Option<Self> {
        if value.is_finite() && (0.0..=1.0).contains(&value) {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Returns the represented value.
    #[must_use]
    pub const fn as_f64(self) -> f64 {
        self.0
    }
}

/// Read and write counts observed during one window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct RequestSignal {
    /// Completed read operations in the window.
    pub reads: u64,
    /// Completed write operations in the window.
    pub writes: u64,
}

/// One end-to-end request latency observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LatencySignal {
    /// End-to-end latency in nanoseconds.
    pub duration_ns: u64,
}

/// Queue occupancy, wait quantiles, and admission rejection counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueueingSignal {
    /// Maximum queued operation count in the window.
    pub depth: u64,
    /// Median queue wait in nanoseconds.
    pub wait_p50_ns: u64,
    /// 99th percentile queue wait in nanoseconds.
    pub wait_p99_ns: u64,
    /// 99.9th percentile queue wait in nanoseconds.
    pub wait_p999_ns: u64,
    /// Requests rejected by admission control.
    pub admission_rejections: u64,
}

/// CPU and service work attributed to requests in one window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeSignal {
    /// Total CPU time in nanoseconds.
    pub cpu_time_ns: u64,
    /// Total request service time in nanoseconds.
    pub service_time_ns: u64,
}

/// Physical and logical byte footprint for one scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemorySignal {
    /// Resident physical bytes.
    pub resident_bytes: u64,
    /// Logical bytes represented by resident data.
    pub logical_bytes: u64,
}

/// Memory pressure and reclaim outcomes for one window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryPressureSignal {
    /// Observed pressure in `[0, 1]`.
    pub pressure: UnitInterval,
    /// Reclaim attempts started.
    pub reclaim_attempts: u64,
    /// Reclaim attempts that failed.
    pub reclaim_failures: u64,
}

/// Compression input, output, and CPU cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompressionSignal {
    /// Uncompressed logical bytes processed.
    pub logical_bytes: u64,
    /// Compressed stored bytes produced.
    pub compressed_bytes: u64,
    /// CPU time spent compressing in nanoseconds.
    pub cpu_time_ns: u64,
}

/// One offcore promotion latency and its byte count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OffcoreLatencySignal {
    /// Offcore execution latency in nanoseconds.
    pub duration_ns: u64,
    /// Bytes promoted or reconstructed.
    pub promoted_bytes: u64,
}

/// Network byte counters for one window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct NetworkBytes {
    /// Bytes received from peers or clients.
    pub received: u64,
    /// Bytes sent to peers or clients.
    pub transmitted: u64,
}

/// Storage byte counters for one window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct StorageBytes {
    /// Bytes read from storage.
    pub read: u64,
    /// Bytes written to storage.
    pub written: u64,
    /// Bytes durably synchronized.
    pub synced: u64,
}

/// Network and storage traffic for one window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct IoSignal {
    /// Network traffic.
    pub network: NetworkBytes,
    /// Storage traffic.
    pub storage: StorageBytes,
}

/// Consensus application and replica lag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConsensusLagSignal {
    /// Commit-path lag in nanoseconds.
    pub commit_lag_ns: u64,
    /// Slowest replica lag in nanoseconds.
    pub replica_lag_ns: u64,
}

/// Checkpoint debt carried into a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CheckpointDebt {
    /// Logical bytes awaiting checkpoint publication.
    pub bytes: u64,
    /// Age of the oldest debt item.
    pub oldest_age: core::time::Duration,
}

/// Repair debt carried into a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RepairDebt {
    /// Assets awaiting repair.
    pub assets: u64,
    /// Physical bytes awaiting repair.
    pub bytes: u64,
    /// Age of the oldest debt item.
    pub oldest_age: core::time::Duration,
}

/// Scrub debt carried into a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScrubDebt {
    /// Physical bytes awaiting integrity verification.
    pub bytes: u64,
    /// Age of the oldest debt item.
    pub oldest_age: core::time::Duration,
}

/// Checkpoint, repair, and scrub debt for one window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaintenanceDebtSignal {
    /// Checkpoint debt.
    pub checkpoint: CheckpointDebt,
    /// Repair debt.
    pub repair: RepairDebt,
    /// Scrub debt.
    pub scrub: ScrubDebt,
}

/// Logical and physical redundancy volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RedundancySignal {
    /// Logical information bytes.
    pub logical_bytes: u64,
    /// Physical protected bytes across fragments or replicas.
    pub physical_bytes: u64,
    /// Number of physical fragments represented.
    pub fragments: u64,
}

/// Degraded asset count and age.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DegradedAssetsSignal {
    /// Assets currently degraded.
    pub assets: u64,
    /// Age of the oldest degraded asset.
    pub oldest_age: core::time::Duration,
}

/// Migration work and estimated execution cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MigrationCostSignal {
    /// Logical bytes in the migration set.
    pub bytes: u64,
    /// Number of migration operations.
    pub operations: u64,
    /// Estimated migration CPU time in nanoseconds.
    pub cpu_time_ns: u64,
    /// Estimated migration I/O time in nanoseconds.
    pub io_time_ns: u64,
}

/// Worker and tablet load imbalance in one topology sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TopologyImbalanceSignal {
    /// Highest load assigned to one worker.
    pub busiest_worker_load: u64,
    /// Mean load across workers.
    pub mean_worker_load: u64,
    /// Highest load assigned to one tablet.
    pub busiest_tablet_load: u64,
    /// Mean load across tablets.
    pub mean_tablet_load: u64,
}

/// One observed hot-key access with execution-path evidence.
///
/// The key is represented by a fixed-width producer identity so the sketch's
/// retained state is always bounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HotKeySignal {
    /// Stable producer identity for the key.
    pub key: HotKeyId,
    /// Whether this access was on a request critical path.
    pub on_critical_path: bool,
    /// Whether this access contributed to a tail-latency sample.
    pub affects_tail: bool,
    /// Stall time attributed to this access in nanoseconds.
    pub stall_time_ns: u64,
}

/// Access evidence used to approximate execution criticality.
///
/// This model intentionally separates serial execution and critical-path
/// weight from raw access frequency. Component counts above `accesses` are
/// clamped to `accesses` when computing the approximation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionCriticalitySignal {
    /// Total observed accesses.
    pub accesses: u64,
    /// Accesses lacking effective memory-level parallelism.
    pub serial_accesses: u64,
    /// Accesses on a request critical path.
    pub critical_path_accesses: u64,
    /// Accesses contributing to tail latency.
    pub tail_accesses: u64,
    /// Stall time attributed to the key in nanoseconds.
    pub stall_time_ns: u64,
    /// Typical access latency in nanoseconds.
    pub access_latency_ns: u64,
    /// Logical bytes represented by the key.
    pub logical_bytes: u64,
}

/// Aggregated hardware performance evidence for one scope.
///
/// The aggregation is deliberately a *minimum* for trust and a *maximum* for
/// distress, which is the same rule the fabric uses everywhere else: a window in
/// which the counters were multiplexed lowers confidence for the whole scope
/// and cannot be averaged away, and a window in which the core stalled badly is
/// the worst window, not the mean one. Averaging a stall fraction would hide
/// exactly the event a controller needs to see.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HardwareSummary {
    /// Worst stall fraction observed, in `[0, 1]`.
    pub worst_stall_fraction: f64,
    /// Worst last-level cache miss ratio observed, in `[0, 1]`.
    pub worst_cache_miss_ratio: f64,
    /// Worst branch misprediction ratio observed, in `[0, 1]`.
    pub worst_branch_miss_ratio: f64,
    /// Best instructions-per-cycle observed, in `[0, 8]`. The *best*, because a
    /// controller deciding whether a worker is CPU bound needs to know whether
    /// it ever reached full issue, not whether it usually did.
    pub best_ipc: f64,
    /// Total minor page faults across the window.
    pub page_faults: u64,
    /// Total involuntary context switches across the window.
    pub context_switches: u64,
    /// Total CPU migrations across the window.
    pub migrations: u64,
    /// Whether any sample in the window carried hardware evidence.
    pub measured: bool,
    /// Lowest source trust across the window, in `(0, 1]`. Zero when nothing
    /// was measured.
    pub trust: f64,
    /// Worst cost multiplier observed, in `[1, 4]`.
    pub worst_cost_multiplier: f64,
    /// Cache sensitivity of the worst window.
    pub cache_sensitivity: HardwareCacheSensitivity,
}

impl HardwareSummary {
    /// The summary of a window with no hardware evidence: every field neutral
    /// and `measured` false, so a consumer's "was this measured" question has
    /// one answer.
    #[must_use]
    pub const fn absent() -> Self {
        Self {
            worst_stall_fraction: 0.0,
            worst_cache_miss_ratio: 0.0,
            worst_branch_miss_ratio: 0.0,
            best_ipc: 0.0,
            page_faults: 0,
            context_switches: 0,
            migrations: 0,
            measured: false,
            trust: 0.0,
            worst_cost_multiplier: 1.0,
            cache_sensitivity: HardwareCacheSensitivity::Resident,
        }
    }
}

/// Nonnegative integer execution-criticality score.
///
/// The unit is an approximate weighted nanosecond of pressure per logical
/// byte, saturated rather than wrapped at `u64::MAX`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CriticalityScore(u64);

impl CriticalityScore {
    /// Wraps a raw score.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw score.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for CriticalityScore {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

/// Deterministic approximation of performance impact beyond hotness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExecutionCriticalityApproximation {
    /// Weighted pressure per logical byte.
    pub score: CriticalityScore,
    /// Serial accesses divided by all accesses.
    pub serial_fraction: UnitInterval,
    /// Critical-path accesses divided by all accesses.
    pub critical_path_fraction: UnitInterval,
    /// Tail-contributing accesses divided by all accesses.
    pub tail_fraction: UnitInterval,
    /// Accesses contributing to the approximation.
    pub access_count: u64,
}

impl ExecutionCriticalityApproximation {
    /// Computes the approximation from typed access evidence.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn from_signal(signal: &ExecutionCriticalitySignal) -> Self {
        let total = signal.accesses;
        let serial = signal.serial_accesses.min(total);
        let critical = signal.critical_path_accesses.min(total);
        let tail = signal.tail_accesses.min(total);
        let serial_fraction =
            UnitInterval::new(fraction(serial, total)).unwrap_or(UnitInterval::ZERO);
        let critical_fraction =
            UnitInterval::new(fraction(critical, total)).unwrap_or(UnitInterval::ZERO);
        let tail_fraction = UnitInterval::new(fraction(tail, total)).unwrap_or(UnitInterval::ZERO);
        let critical_ppm = (critical_fraction.as_f64() * 1_000_000.0) as u64;
        let tail_ppm = (tail_fraction.as_f64() * 1_000_000.0) as u64;
        let weight_ppm = 100_000_u64
            .saturating_add(critical_ppm / 2)
            .saturating_add(tail_ppm / 2);
        let pressure_ns = u128::from(serial)
            .saturating_mul(u128::from(signal.access_latency_ns))
            .saturating_add(u128::from(signal.stall_time_ns));
        let weighted = pressure_ns
            .saturating_mul(u128::from(weight_ppm))
            .saturating_div(1_000_000);
        let bytes = u128::from(signal.logical_bytes.max(1));
        let score = weighted.saturating_div(bytes);
        Self {
            score: CriticalityScore::from(u64::try_from(score).unwrap_or(u64::MAX)),
            serial_fraction,
            critical_path_fraction: critical_fraction,
            tail_fraction,
            access_count: total,
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn fraction(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

/// How sensitive a serving thread's working set is to cache capacity.
///
/// Ordered by severity, so a fold that keeps the worst window is a comparison
/// rather than a hand-written match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HardwareCacheSensitivity {
    /// The working set fits wherever it is put.
    Resident,
    /// Larger than the last-level cache, so tier and core both matter.
    Sensitive,
    /// Memory bound wherever it lives; only *where* it lives is a lever.
    MemoryBound,
}

/// Hardware performance observations for one worker over one window.
///
/// This is the *only* shape in which hardware evidence enters the fabric. It
/// is dimensionless and bounded, and it carries its own provenance
/// (`hardware_measured`, `hardware_confidence_ppm`) rather than relying on the
/// sample's metadata, because a consumer reads the signal and the metadata
/// separately and must not have to guess which of the two describes the
/// evidence.
///
/// Every field is a zero when no hardware source was available, which is
/// deliberately the same value a source that measured nothing would produce:
/// a machine without a PMU and a PMU that reported nothing are the same case,
/// and nothing downstream can tell them apart or needs to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HardwareSignal {
    /// Instructions retired per hardware cycle, in `[0, 8]`. Zero when the
    /// denominator was not hardware cycles.
    pub ipc: f64,
    /// Fraction of the interval the core could not issue.
    pub stall_fraction: UnitInterval,
    /// Last-level cache misses per cache reference.
    pub cache_miss_ratio: UnitInterval,
    /// Branch mispredictions per branch.
    pub branch_miss_ratio: UnitInterval,
    /// Minor page faults in the interval.
    pub page_faults: u64,
    /// Involuntary context switches in the interval.
    pub context_switches: u64,
    /// CPU migrations of the serving thread in the interval.
    pub migrations: u64,
    /// Cache sensitivity implied by the traffic and the machine's behaviour.
    pub cache_sensitivity: HardwareCacheSensitivity,
    /// Relative cost of the interval's work in parts per million, where
    /// `1_000_000` is unchanged and the cap is `4_000_000`.
    pub cost_multiplier_ppm: u32,
    /// Whether a hardware source actually contributed. `false` means every
    /// numeric field above is neutral and nothing may be concluded from it.
    pub hardware_measured: bool,
    /// How much the source can be trusted, in parts per million where
    /// `1_000_000` is exact. A multiplexed counter set reports less.
    pub hardware_confidence_ppm: u32,
}

impl HardwareSignal {
    /// The software-only reading: every hardware field at its neutral value.
    #[must_use]
    pub const fn software_only() -> Self {
        Self {
            ipc: 0.0,
            stall_fraction: UnitInterval::ZERO,
            cache_miss_ratio: UnitInterval::ZERO,
            branch_miss_ratio: UnitInterval::ZERO,
            page_faults: 0,
            context_switches: 0,
            migrations: 0,
            cache_sensitivity: HardwareCacheSensitivity::Resident,
            cost_multiplier_ppm: 1_000_000,
            hardware_measured: false,
            hardware_confidence_ppm: 0,
        }
    }

    /// Whether the serving thread was moved between cores, which means the
    /// placement plan's core is not the core the work happened on.
    #[must_use]
    pub const fn placement_unstable(&self) -> bool {
        self.migrations > 0
    }

    /// Whether the serving thread looks memory bound: heavy stalls *and* heavy
    /// last-level misses.
    #[must_use]
    pub const fn memory_bound(&self) -> bool {
        self.stall_fraction.as_f64() > 0.35 && self.cache_miss_ratio.as_f64() > 0.2
    }

    /// Whether the serving thread looks CPU bound.
    #[must_use]
    pub const fn cpu_bound(&self) -> bool {
        self.ipc >= 1.5 && self.stall_fraction.as_f64() < 0.15
    }

    /// How much the hardware raised the estimated cost of the interval, as a
    /// multiple. Exactly `1.0` when nothing was measured.
    #[must_use]
    pub fn cost_multiplier(&self) -> f64 {
        f64::from(self.cost_multiplier_ppm) / 1_000_000.0
    }

    /// How much the hardware source can be trusted, in `[0, 1]`.
    #[must_use]
    pub fn trust(&self) -> f64 {
        f64::from(self.hardware_confidence_ppm) / 1_000_000.0
    }

    /// Whether a controller may act on this reading without corroboration.
    ///
    /// A hardware sample alone is never enough. The threshold is deliberately
    /// high: a counter group the kernel had to multiplex is a real measurement
    /// but a coarse one, and a decision that moves data between tiers should be
    /// corroborated by a software counter that saw the same work.
    #[must_use]
    pub const fn is_actionable(&self, corroborating_accesses: u64) -> bool {
        self.hardware_measured
            && self.hardware_confidence_ppm >= 900_000
            && corroborating_accesses > 0
    }
}

/// Closed set of signals accepted by the observation fabric.
///
/// Adding a signal requires adding a named variant and an explicit
/// accumulator; callers cannot inject arbitrary metric names or values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TypedSignal {
    /// Read and write throughput evidence.
    Request(RequestSignal),
    /// One end-to-end latency observation.
    Latency(LatencySignal),
    /// Queue occupancy and wait evidence.
    Queueing(QueueingSignal),
    /// CPU and service work.
    Compute(ComputeSignal),
    /// Physical and logical memory footprint.
    Memory(MemorySignal),
    /// Memory pressure and reclaim behavior.
    MemoryPressure(MemoryPressureSignal),
    /// Compression ratio and CPU cost.
    Compression(CompressionSignal),
    /// One offcore execution observation.
    OffcoreLatency(OffcoreLatencySignal),
    /// Network and storage traffic.
    Io(IoSignal),
    /// Consensus lag.
    ConsensusLag(ConsensusLagSignal),
    /// Checkpoint, repair, and scrub debt.
    MaintenanceDebt(MaintenanceDebtSignal),
    /// Redundancy volume amplification.
    Redundancy(RedundancySignal),
    /// Degraded assets and age.
    DegradedAssets(DegradedAssetsSignal),
    /// Migration work and cost.
    MigrationCost(MigrationCostSignal),
    /// Worker and tablet load imbalance.
    TopologyImbalance(TopologyImbalanceSignal),
    /// One fixed hot-key access.
    HotKey(HotKeySignal),
    /// Execution-criticality evidence.
    ExecutionCriticality(ExecutionCriticalitySignal),
    /// Hardware performance evidence for the serving thread.
    Hardware(HardwareSignal),
}

/// One complete typed observation ready for collection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObservationSample {
    /// Scope to which the signal belongs.
    pub scope: ObservationScope,
    /// Producer window and monotonic replay position.
    pub stamp: ObservationStamp,
    /// Nonzero duration represented by the signal.
    pub period: ObservationPeriod,
    /// Source, confidence, and producer timestamp.
    pub metadata: ObservationMetadata,
    /// Closed typed signal payload.
    pub signal: TypedSignal,
}

impl ObservationSample {
    /// Creates a typed sample from already validated identity and period.
    #[must_use]
    pub const fn new(
        scope: ObservationScope,
        stamp: ObservationStamp,
        period: ObservationPeriod,
        metadata: ObservationMetadata,
        signal: TypedSignal,
    ) -> Self {
        Self {
            scope,
            stamp,
            period,
            metadata,
            signal,
        }
    }

    /// Returns whether this sample was produced at `observed_at`.
    #[must_use]
    pub const fn observed_at(&self) -> Ticks {
        self.metadata.observed_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_interval_rejects_nonfinite_and_out_of_range_values() {
        assert_eq!(UnitInterval::new(f64::NAN), None);
        assert_eq!(UnitInterval::new(f64::INFINITY), None);
        assert_eq!(UnitInterval::new(-0.01), None);
        assert_eq!(UnitInterval::new(1.01), None);
        assert_eq!(UnitInterval::new(0.25), Some(UnitInterval(0.25)));
    }

    /// A measured reading, for the classification tests below.
    fn measured(
        ipc: f64,
        stall: f64,
        cache_miss: f64,
        migrations: u64,
        confidence_ppm: u32,
    ) -> HardwareSignal {
        HardwareSignal {
            ipc,
            stall_fraction: UnitInterval::new(stall).unwrap_or(UnitInterval::ZERO),
            cache_miss_ratio: UnitInterval::new(cache_miss).unwrap_or(UnitInterval::ZERO),
            branch_miss_ratio: UnitInterval::new(0.05).unwrap_or(UnitInterval::ZERO),
            page_faults: 0,
            context_switches: 0,
            migrations,
            cache_sensitivity: HardwareCacheSensitivity::Resident,
            cost_multiplier_ppm: 1_000_000,
            hardware_measured: true,
            hardware_confidence_ppm: confidence_ppm,
        }
    }

    #[test]
    fn a_software_only_hardware_signal_concludes_nothing() {
        // The whole design turns on this: "no counter" and "a counter that
        // reported nothing" must be the same case, and neither is evidence.
        let signal = HardwareSignal::software_only();
        assert!(!signal.hardware_measured);
        assert!(!signal.placement_unstable());
        assert!(!signal.memory_bound());
        assert!(!signal.cpu_bound());
        assert!(signal.trust() <= 0.0);
        assert!(
            (signal.cost_multiplier() - 1.0).abs() <= f64::EPSILON,
            "an unmeasured interval costs exactly what it would have cost anyway"
        );
    }

    #[test]
    fn a_migration_is_distinguishable_from_a_context_switch() {
        // A context switch is the scheduler's business; a migration means the
        // work happened on a core the placement plan did not choose, which is a
        // different fact and the only one a controller can act on.
        let switched = measured(2.0, 0.1, 0.1, 0, 1_000_000);
        assert!(!switched.placement_unstable());
        let migrated = measured(2.0, 0.1, 0.1, 1, 1_000_000);
        assert!(migrated.placement_unstable());
    }

    #[test]
    fn memory_bound_needs_both_stalls_and_misses() {
        // Stalls alone could be a dependency chain; misses alone could be a cold
        // start. Only the pair is a working set that does not fit.
        let stalls_only = measured(0.4, 0.9, 0.05, 0, 1_000_000);
        assert!(!stalls_only.memory_bound());
        let misses_only = measured(0.4, 0.05, 0.9, 0, 1_000_000);
        assert!(!misses_only.memory_bound());
        let both = measured(0.4, 0.9, 0.9, 0, 1_000_000);
        assert!(both.memory_bound());
    }

    #[test]
    fn cpu_bound_needs_full_issue_and_few_stalls() {
        let hot = measured(3.0, 0.01, 0.5, 0, 1_000_000);
        assert!(hot.cpu_bound());
        let stalled = measured(3.0, 0.9, 0.1, 0, 1_000_000);
        assert!(
            !stalled.cpu_bound(),
            "a core that cannot issue is not compute bound"
        );
        let slow = measured(0.4, 0.01, 0.1, 0, 1_000_000);
        assert!(
            !slow.cpu_bound(),
            "more cores would not help a serial chain"
        );
    }

    #[test]
    fn a_multiplexed_source_reports_partial_trust() {
        // The value a controller gates on, so it is asserted directly rather than
        // only through the fabric fold.
        let exact = measured(2.0, 0.1, 0.1, 0, 1_000_000);
        let scaled = measured(2.0, 0.1, 0.1, 0, 400_000);
        assert!((exact.trust() - 1.0).abs() < 1e-9);
        assert!((scaled.trust() - 0.4).abs() < 1e-9);
        assert!(scaled.trust() < exact.trust());
    }

    #[test]
    fn cache_sensitivity_is_ordered_so_a_worst_window_is_a_comparison() {
        // The fold in the fabric keeps the maximum, so the ordering is what makes
        // "keep the worst" one operation rather than a hand-written match.
        assert!(HardwareCacheSensitivity::Resident < HardwareCacheSensitivity::Sensitive);
        assert!(HardwareCacheSensitivity::Sensitive < HardwareCacheSensitivity::MemoryBound);
        let worst = [
            HardwareCacheSensitivity::Resident,
            HardwareCacheSensitivity::MemoryBound,
        ];
        assert_eq!(
            worst.into_iter().max(),
            Some(HardwareCacheSensitivity::MemoryBound)
        );
    }

    #[test]
    fn criticality_outranks_frequency_beyond_hotness() {
        let rare_serial = ExecutionCriticalitySignal {
            accesses: 1,
            serial_accesses: 1,
            critical_path_accesses: 1,
            tail_accesses: 1,
            stall_time_ns: 900_000,
            access_latency_ns: 1_000_000,
            logical_bytes: 64,
        };
        let frequent_parallel = ExecutionCriticalitySignal {
            accesses: 10_000,
            serial_accesses: 10,
            critical_path_accesses: 0,
            tail_accesses: 0,
            stall_time_ns: 0,
            access_latency_ns: 100,
            logical_bytes: 64,
        };
        let rare = ExecutionCriticalityApproximation::from_signal(&rare_serial);
        let frequent = ExecutionCriticalityApproximation::from_signal(&frequent_parallel);
        assert!(
            rare.score.as_u64() > frequent.score.as_u64(),
            "rare serial score {} should exceed parallel-hot score {}",
            rare.score.as_u64(),
            frequent.score.as_u64()
        );
        assert_eq!(rare.serial_fraction, UnitInterval::ONE);
        assert_eq!(rare.critical_path_fraction, UnitInterval::ONE);
        assert!((frequent.serial_fraction.as_f64() - 0.001).abs() < f64::EPSILON);
    }

    #[test]
    fn criticality_is_deterministic_for_equal_input() {
        let signal = ExecutionCriticalitySignal {
            accesses: 123,
            serial_accesses: 17,
            critical_path_accesses: 11,
            tail_accesses: 5,
            stall_time_ns: 9_001,
            access_latency_ns: 333,
            logical_bytes: 257,
        };
        assert_eq!(
            ExecutionCriticalityApproximation::from_signal(&signal),
            ExecutionCriticalityApproximation::from_signal(&signal)
        );
    }
}
