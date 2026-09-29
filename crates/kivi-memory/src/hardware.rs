//! Hardware observations, translated into the typed observation model.
//!
//! ## The gap this closes
//!
//! `kivi_hardware` produces hardware samples and `kivi-observation` consumes
//! typed signals. Between them was nothing: a `PmuSample` stopped at
//! `kivi-hardware::telemetry`, so no controller could distinguish a software
//! estimate from a hardware measurement, and no signal ever told the control
//! loop that a worker had been migrated off the core it was placed on.
//!
//! This module is the seam. It does one thing: turn a hardware sample into a
//! typed observation, carrying the provenance that makes the number
//! interpretable, and it does it without inventing a second metrics system.
//!
//! ## Why the provenance is not optional
//!
//! A controller that cannot tell "this worker stalled 40% of its cycles" from
//! "no counter was readable, so the field is zero" will act on the second as if
//! it were the first. So every sample carries:
//!
//! * whether a hardware source contributed at all,
//! * how much the source can be trusted, which a multiplexed counter group
//!   lowers and DAMON's coarse aggregation lowers,
//! * and which named source it came from.
//!
//! A machine with no PMU produces `Confidence::Low` samples that say exactly
//! that, and a controller's minimum-confidence gate keeps them out of the same
//! decision path a real measurement reaches.
//!
//! ## What is deliberately absent
//!
//! There is no background thread here. A counter is opened by the thread whose
//! work it describes, sampled once per window, and handed over. A collector on
//! another thread would either need to signal the owner (a wakeup on the hot
//! path) or sample the wrong thread's counters (a measurement of the wrong
//! thing). Kivi's owner threads already drain telemetry once per window; that
//! is where the sample belongs.

use kivi_hardware::telemetry::pmu::PmuCounters;
use kivi_hardware::telemetry::signals::{CacheSensitivity, HardwareSignals};
use kivi_observation::{
    Confidence, HardwareSignal as ObservedHardware, ObservationMetadata, ObservationSource,
    TypedSignal, UnitInterval,
};
use kivi_types::{Ticks, WorkerId};

/// How much of a hardware sample is evidence and how much is provenance.
///
/// The two numbers a controller needs and the two it must never be given: it
/// must know the reading is measured rather than assumed, and it must know how
/// much the reading is worth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HardwareReading {
    /// Whether any hardware source contributed.
    pub measured: bool,
    /// How much the source can be trusted, in `(0, 1]`.
    pub confidence: f64,
    /// Which source produced it.
    pub source: ReadingSource,
}

impl HardwareReading {
    /// No hardware source contributed. Every field is at its neutral value, so
    /// a consumer that ignores this struct and uses the signals alone still
    /// produces the software-only answer.
    #[must_use]
    pub const fn absent() -> Self {
        Self {
            measured: false,
            confidence: 0.0,
            source: ReadingSource::None,
        }
    }

    /// The observation-fabric confidence for this reading.
    ///
    /// A reading that came from hardware and was scaled by the kernel is
    /// [`Confidence::Medium`]; one that was exact is [`Confidence::High`]. An
    /// absent reading is [`Confidence::Low`], which is below the controller's
    /// default minimum and therefore never reaches a decision on its own.
    #[must_use]
    pub const fn observation_confidence(&self) -> Confidence {
        if !self.measured {
            Confidence::Low
        } else if self.confidence < 1.0 {
            Confidence::Medium
        } else {
            Confidence::High
        }
    }
}

/// Which hardware source produced a reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadingSource {
    /// Nothing hardware was available.
    None,
    /// A performance monitoring unit counter group.
    PerformanceCounters,
    /// The kernel's memory access monitor.
    MemoryAccessMonitor,
}

impl ReadingSource {
    /// The observation-fabric source this maps to.
    #[must_use]
    pub const fn observation_source(self) -> ObservationSource {
        match self {
            // A missing source is still a sample; calling it a runtime counter
            // is the honest description of what produced the (zero) numbers,
            // and it keeps a sample from a machine with no PMU in the same
            // stream as one with.
            Self::None => ObservationSource::RuntimeCounter,
            Self::PerformanceCounters | Self::MemoryAccessMonitor => {
                // PMU counters and DAMON reports are both sampled, not counted
                // on the request path, which is exactly what
                // `SampledTelemetry` names.
                ObservationSource::SampledTelemetry
            }
        }
    }

    /// A short name for diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PerformanceCounters => "pmu",
            Self::MemoryAccessMonitor => "damon",
        }
    }
}

impl std::fmt::Display for ReadingSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The typed signal a hardware sample becomes.
///
/// Every field is dimensionless and bounded, which is what makes it safe to
/// hand to a controller that will eventually act on it. A raw cycle count is
/// not in this struct: it means nothing without the interval it came from, and
/// the interval is the caller's business, not the controller's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HardwareSample {
    /// Instructions retired per hardware cycle, in `[0, 8]`. Zero when the
    /// denominator was not hardware cycles, which is a different fact from "the
    /// core issued at 0 IPC".
    pub ipc: f64,
    /// Fraction of the interval the core could not issue, in `[0, 1]`.
    pub stall_fraction: f64,
    /// Last-level cache misses per cache reference, in `[0, 1]`.
    pub cache_miss_ratio: f64,
    /// Branch mispredictions per branch, in `[0, 1]`.
    pub branch_miss_ratio: f64,
    /// Minor page faults in the interval.
    pub page_faults: u64,
    /// Involuntary context switches in the interval.
    pub context_switches: u64,
    /// CPU migrations of the serving thread in the interval. Unlike a context
    /// switch this is unambiguous: the thread moved to a different core.
    pub migrations: u64,
    /// How sensitive the serving thread's working set is to cache capacity.
    pub cache_sensitivity: CacheSensitivity,
    /// Relative cost of the interval's work, in `[1, 4]`. Exactly `1.0` when
    /// no hardware source contributed, so a consumer's cost model is unchanged
    /// rather than merely similar on a machine without counters.
    pub cost_multiplier: f64,
    /// Provenance.
    pub reading: HardwareReading,
}

impl HardwareSample {
    /// The software-only reading: every hardware field neutral, cost unchanged.
    ///
    /// This is what a machine with no PMU produces, and it is deliberately
    /// *identical* to what a PMU that reported nothing would produce, so the
    /// two cases cannot diverge downstream.
    #[must_use]
    pub const fn software_only() -> Self {
        Self {
            ipc: 0.0,
            stall_fraction: 0.0,
            cache_miss_ratio: 0.0,
            branch_miss_ratio: 0.0,
            page_faults: 0,
            context_switches: 0,
            migrations: 0,
            cache_sensitivity: CacheSensitivity::Resident,
            cost_multiplier: 1.0,
            reading: HardwareReading::absent(),
        }
    }

    /// Whether the serving thread was moved between cores in the interval.
    #[must_use]
    pub const fn placement_unstable(&self) -> bool {
        self.migrations > 0
    }

    /// Whether the serving thread looks memory bound: heavy stalls *and* heavy
    /// last-level cache misses, which is the signature of a working set that
    /// does not fit anywhere it currently is.
    #[must_use]
    pub fn memory_bound(&self) -> bool {
        self.stall_fraction > 0.35 && self.cache_miss_ratio > 0.2
    }

    /// Whether the serving thread looks CPU bound: high issue rate and few
    /// stalls, which is the signature of work that more cores would speed up.
    #[must_use]
    pub fn cpu_bound(&self) -> bool {
        self.ipc >= 1.5 && self.stall_fraction < 0.15
    }

    /// Whether this sample carries enough evidence for a controller to act on
    /// without further corroboration.
    ///
    /// A hardware sample alone is never enough. The threshold is deliberately
    /// high: a counter group that the kernel had to multiplex is a real
    /// measurement but a coarse one, and a decision that moves data between
    /// tiers should be corroborated by a software counter that saw the same
    /// object twice.
    #[must_use]
    pub fn is_actionable(&self, corroborating_accesses: u64) -> bool {
        self.reading.measured && self.reading.confidence >= 0.9 && corroborating_accesses > 0
    }

    /// The sample as the observation fabric's closed signal type.
    ///
    /// `confidence` is clamped into the unit interval here rather than being
    /// trusted, because a controller's safety envelope is written in terms of
    /// that interval and a value outside it would be a different number.
    #[must_use]
    pub fn to_signal(&self) -> TypedSignal {
        TypedSignal::Hardware(self.to_hardware_signal())
    }

    /// The closed hardware signal, without the enum wrapper.
    ///
    /// Same clamping as [`HardwareSample::to_signal`], for a caller that
    /// already knows the variant.
    #[must_use]
    pub fn to_hardware_signal(&self) -> ObservedHardware {
        ObservedHardware {
            ipc: self.ipc,
            stall_fraction: UnitInterval::new(self.stall_fraction.clamp(0.0, 1.0))
                .unwrap_or(UnitInterval::ZERO),
            cache_miss_ratio: UnitInterval::new(self.cache_miss_ratio.clamp(0.0, 1.0))
                .unwrap_or(UnitInterval::ZERO),
            branch_miss_ratio: UnitInterval::new(self.branch_miss_ratio.clamp(0.0, 1.0))
                .unwrap_or(UnitInterval::ZERO),
            page_faults: self.page_faults,
            context_switches: self.context_switches,
            migrations: self.migrations,
            cache_sensitivity: match self.cache_sensitivity {
                CacheSensitivity::Resident => kivi_observation::HardwareCacheSensitivity::Resident,
                CacheSensitivity::Sensitive => {
                    kivi_observation::HardwareCacheSensitivity::Sensitive
                }
                CacheSensitivity::MemoryBound => {
                    kivi_observation::HardwareCacheSensitivity::MemoryBound
                }
            },
            cost_multiplier_ppm: cost_multiplier_ppm(self.cost_multiplier),
            hardware_measured: self.reading.measured,
            hardware_confidence_ppm: confidence_ppm(self.reading.confidence),
        }
    }

    /// The metadata a sample carries into the fabric.
    ///
    /// `observed_at` and the stamp come from the caller because the caller owns
    /// the window; a hardware source does not know which observation window it
    /// belongs to and must not invent one.
    #[must_use]
    pub fn metadata(&self, observed_at: Ticks) -> ObservationMetadata {
        ObservationMetadata::new(
            observed_at,
            self.reading.source.observation_source(),
            self.reading.observation_confidence(),
        )
    }
}

/// The cost multiplier in parts per million, saturated at the cap.
///
/// Parts per million rather than a float because a controller's feature vector
/// is integer and a float would be lossy in a place that is already lossy.
///
/// The casts are range-checked on the line above each one: the first branch has
// returned for every value `<= 1.0` and the second for every value `>= 4.0`
// or NaN, so what reaches the conversion lies in `(1.0, 4.0)`.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub const fn cost_multiplier_ppm(multiplier: f64) -> u32 {
    if multiplier <= 1.0 {
        return 1_000_000;
    }
    // A NaN compares false against everything, so the first branch catches it
    // only when it is not greater than 1.0; a NaN that slipped through lands in
    // the saturated branch below and becomes the neutral value.
    let ppm = multiplier.mul_add(1_000_000.0, 0.0);
    if ppm >= 4_000_000.0 || ppm.is_nan() {
        return 4_000_000;
    }
    ppm as u32
}

/// A confidence in parts per million, saturated into `[0, 1]`.
///
/// A value outside the range, and a `NaN`, become zero. The bounds are written out
/// rather than delegated to a range check because that check is not available
/// in a `const fn`, and a non-`const` helper here would be the only way to call
/// this from another `const` context.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub const fn confidence_ppm(confidence: f64) -> u32 {
    let ppm = confidence * 1_000_000.0;
    if ppm <= 0.0 {
        return 0;
    }
    if ppm >= 1_000_000.0 {
        return 1_000_000;
    }
    // A NaN compares false against both bounds above and lands here, where the
    // conversion is defined and yields zero.
    ppm as u32
}

/// Per-worker hardware sampling.
///
/// One instance per owner thread, because a counter group is `!Send` and
/// because a counter opened on one thread counts only that thread's
/// instructions. The instance is the thread's own view of what the machine did
/// while it was working, which is the only view that means anything.
#[derive(Debug)]
pub struct HardwareSampler {
    counters: PmuCounters,
    previous: Option<kivi_hardware::telemetry::pmu::PmuSample>,
    /// Software requests served in the window being sampled, used to decide
    /// whether a hardware reading is corroborated by anything.
    window_accesses: u64,
    /// The last cumulative work count [`HardwareSampler::observe_work`] saw.
    previous_work: Option<u64>,
}

impl Default for HardwareSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareSampler {
    /// Opens counters for the calling thread.
    ///
    /// On a machine or a privilege level that forbids counters this is an
    /// unattached sampler that reports the software-only reading, and the
    /// process starts normally. Hardware telemetry is never a startup gate.
    #[must_use]
    pub fn new() -> Self {
        Self {
            counters: PmuCounters::attach(),
            previous: None,
            window_accesses: 0,
            previous_work: None,
        }
    }

    /// Whether hardware counters are open on this thread.
    #[must_use]
    pub fn is_attached(&self) -> bool {
        self.counters.is_attached()
    }

    /// Which counters are open.
    #[must_use]
    pub fn support(&self) -> &kivi_hardware::PmuSupport {
        self.counters.support()
    }

    /// Records the serving thread's cumulative work count.
    ///
    /// Takes a *monotonic total*, not an increment, and derives the window's
    /// delta itself. A caller that passed an increment could pass a number
    /// supporting whatever it had already decided; a total cannot be invented,
    /// and a total that goes backwards (a restarted counter) contributes
    /// nothing rather than a fabricated amount of work.
    pub fn observe_work(&mut self, total_work: u64) {
        if let Some(previous) = self.previous_work.replace(total_work) {
            self.window_accesses = self
                .window_accesses
                .saturating_add(total_work.saturating_sub(previous));
        }
    }

    /// Ends the window: folds in the serving thread's work count, reads the
    /// counters, and clears the corroboration.
    ///
    /// One call rather than three because the three are one operation, and
    /// splitting them invites a caller that reads the counters and forgets to
    /// close the window - which silently carries one window's corroboration
    /// into the next.
    pub fn end_of_window(&mut self, total_work: u64) -> HardwareSample {
        self.observe_work(total_work);
        let sample = self.sample();
        self.end_window();
        sample
    }

    /// Reads the counters and returns the interval's reading.
    ///
    /// The first call establishes the baseline and returns the software-only
    /// reading, because a delta against an unopened counter is not a
    /// measurement. Every later call is a real interval.
    pub fn sample(&mut self) -> HardwareSample {
        let current = self.counters.sample();
        let Some(previous) = self.previous.replace(current) else {
            return HardwareSample::software_only();
        };
        // A counter that went backwards means the thread restarted or the
        // counter was reset. The delta is a lower bound, and a sample that
        // cannot be compared with the last one is not actionable.
        let delta = current.since(&previous);
        if delta.ordering != kivi_hardware::telemetry::pmu::Ordering::Monotonic
            || !current.comparable()
        {
            return HardwareSample::software_only();
        }
        let signals: HardwareSignals = current.signals();
        let measured = signals.trust() > 0.0;
        let sensitivity = if signals.memory_bound() {
            CacheSensitivity::MemoryBound
        } else if signals.cache_miss_ratio > 0.2 {
            CacheSensitivity::Sensitive
        } else {
            CacheSensitivity::Resident
        };
        HardwareSample {
            ipc: signals.ipc,
            stall_fraction: signals.stall_fraction,
            cache_miss_ratio: signals.cache_miss_ratio,
            branch_miss_ratio: signals.branch_miss_ratio,
            page_faults: signals.page_faults,
            context_switches: signals.context_switches,
            migrations: signals.migrations,
            cache_sensitivity: sensitivity,
            cost_multiplier: 1.0 + signals.cache_miss_ratio.clamp(0.0, 1.0),
            reading: HardwareReading {
                measured,
                confidence: signals.trust(),
                source: if measured {
                    ReadingSource::PerformanceCounters
                } else {
                    ReadingSource::None
                },
            },
        }
    }

    /// Ends the window and clears the corroboration count.
    pub fn end_window(&mut self) {
        self.window_accesses = 0;
    }

    /// Whether the last window had enough software corroboration for a
    /// hardware-only reading to be acted on.
    #[must_use]
    pub const fn is_corroborated(&self) -> bool {
        self.window_accesses > 0
    }

    /// The typed sample for `worker` in `window`, ready for the fabric.
    ///
    /// The caller owns the stamp and the period, because a hardware source does
    /// not know which observation window it belongs to and must not invent one:
    /// a period guessed here would make every rate the controller computes from
    /// the sample wrong by a factor of the guess.
    #[must_use]
    pub fn observation(
        &mut self,
        worker: WorkerId,
        stamp: kivi_observation::ObservationStamp,
        period: kivi_observation::ObservationPeriod,
        observed_at: Ticks,
    ) -> (kivi_observation::ObservationSample, HardwareSample) {
        let sample = self.sample();
        let typed = kivi_observation::ObservationSample::new(
            kivi_observation::ObservationScope::Worker(worker),
            stamp,
            period,
            sample.metadata(observed_at),
            sample.to_signal(),
        );
        (typed, sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::time::Duration;

    #[test]
    fn an_absent_reading_is_low_confidence_and_neutral() {
        let sample = HardwareSample::software_only();
        assert!(!sample.reading.measured);
        assert!(sample.reading.confidence <= 0.0);
        assert_eq!(sample.reading.observation_confidence(), Confidence::Low);
        assert!((sample.cost_multiplier - 1.0).abs() <= f64::EPSILON);
        assert_eq!(confidence_ppm(0.0), 0);
        assert_eq!(cost_multiplier_ppm(1.0), 1_000_000);
        // Not actionable, at any corroboration level, because there is nothing
        // measured behind it.
        assert!(!sample.is_actionable(u64::MAX));
    }

    #[test]
    fn a_scaled_reading_is_usable_but_not_actionable_alone() {
        let sample = HardwareSample {
            reading: HardwareReading {
                measured: true,
                confidence: 0.5,
                source: ReadingSource::PerformanceCounters,
            },
            ..HardwareSample::software_only()
        };
        assert_eq!(sample.reading.observation_confidence(), Confidence::Medium);
        assert!(
            !sample.is_actionable(1000),
            "a multiplexed counter set is a measurement but a coarse one"
        );
    }

    #[test]
    fn an_exact_reading_needs_corroboration_to_act() {
        let sample = HardwareSample {
            reading: HardwareReading {
                measured: true,
                confidence: 1.0,
                source: ReadingSource::PerformanceCounters,
            },
            ..HardwareSample::software_only()
        };
        assert_eq!(sample.reading.observation_confidence(), Confidence::High);
        assert!(!sample.is_actionable(0), "no software access saw it");
        assert!(sample.is_actionable(1));
    }

    #[test]
    fn the_typed_signal_is_closed_and_bounded() {
        let hostile = HardwareSample {
            ipc: 99.0,
            stall_fraction: 5.0,
            cache_miss_ratio: -1.0,
            branch_miss_ratio: f64::NAN,
            page_faults: 7,
            context_switches: 3,
            migrations: 1,
            cache_sensitivity: CacheSensitivity::MemoryBound,
            cost_multiplier: 1.0e9,
            reading: HardwareReading {
                measured: true,
                confidence: 4.0,
                source: ReadingSource::PerformanceCounters,
            },
        };
        let TypedSignal::Hardware(signal) = hostile.to_signal() else {
            panic!("a hardware sample becomes a hardware signal");
        };
        assert!(signal.stall_fraction.as_f64() <= 1.0);
        assert!(signal.cache_miss_ratio.as_f64() >= 0.0);
        // A NaN cannot satisfy the unit interval, so it becomes zero rather
        // than propagating into a controller's arithmetic.
        assert!(signal.branch_miss_ratio.as_f64() >= 0.0);
        assert_eq!(signal.cost_multiplier_ppm, 4_000_000, "the cap holds");
        // A confidence above one saturates rather than being discarded: the
        // sample said "as sure as this interface can be sure", and rounding that
        // to zero would claim the opposite of what it reported.
        assert_eq!(signal.hardware_confidence_ppm, 1_000_000);
        assert!(signal.hardware_measured);
    }

    #[test]
    fn a_migration_is_distinguishable_from_a_context_switch() {
        let switched = HardwareSample {
            context_switches: 500,
            ..HardwareSample::software_only()
        };
        assert!(!switched.placement_unstable());
        let migrated = HardwareSample {
            context_switches: 500,
            migrations: 1,
            ..HardwareSample::software_only()
        };
        assert!(migrated.placement_unstable());
    }

    #[test]
    fn the_classifications_need_both_conditions() {
        let stalled_only = HardwareSample {
            stall_fraction: 0.9,
            ..HardwareSample::software_only()
        };
        assert!(!stalled_only.memory_bound());
        let both = HardwareSample {
            stall_fraction: 0.9,
            cache_miss_ratio: 0.5,
            ..HardwareSample::software_only()
        };
        assert!(both.memory_bound());
        let cpu = HardwareSample {
            ipc: 3.0,
            stall_fraction: 0.01,
            ..HardwareSample::software_only()
        };
        assert!(cpu.cpu_bound());
    }

    #[test]
    fn a_sampler_never_claims_more_than_the_counters_deliver() {
        // What this test protects is the invariant, not a particular machine's
        // answer: a sample claims evidence only when the counters behind it were
        // both *scheduled* and *scheduled together*. A machine whose PMU is
        // absent, whose privileges forbid counters, or whose hypervisor refuses
        // to form a group all produce the software-only reading, and none of
        // them may produce a fabricated ratio.
        let mut sampler = HardwareSampler::new();
        if sampler.is_attached() {
            let mut acc = 0_u64;
            for i in 0..20_000_u64 {
                acc = acc.wrapping_add(i);
            }
            std::hint::black_box(acc);
            let sample = sampler.sample();
            // Whether this particular kernel granted a commensurable group is a
            // machine fact, not a test premise. What must hold either way is
            // that an unmeasured sample carries no ratio and a measured one
            // carries a confidence in range.
            if sample.reading.measured {
                assert!(sample.reading.confidence > 0.0 && sample.reading.confidence <= 1.0);
                assert!(sample.ipc <= 8.0);
                assert!(sample.stall_fraction <= 1.0);
            } else {
                assert_eq!(sample, HardwareSample::software_only());
            }
            return;
        }
        for _ in 0..3 {
            let sample = sampler.sample();
            assert_eq!(sample, HardwareSample::software_only());
            assert!(!sample.reading.measured);
            assert!((sample.cost_multiplier - 1.0).abs() <= f64::EPSILON);
        }
    }

    #[test]
    fn the_first_sample_is_a_baseline_not_evidence() {
        // The delta of a counter against its own opening value measures the
        // interval since the process started, not the window a controller is
        // reasoning about. Reporting it would let a controller act on startup
        // cost.
        let mut sampler = HardwareSampler::new();
        let first = sampler.sample();
        assert!(
            !first.reading.measured,
            "the opening sample establishes a baseline and claims nothing"
        );
    }

    #[test]
    fn corroboration_accumulates_within_a_window_and_resets() {
        let mut sampler = HardwareSampler::new();
        assert!(!sampler.is_corroborated());
        // The first total establishes the baseline; the next one is the
        // window's work.
        sampler.observe_work(1_000);
        assert!(!sampler.is_corroborated());
        sampler.observe_work(1_003);
        sampler.observe_work(1_004);
        assert!(sampler.is_corroborated());
        sampler.end_window();
        assert!(!sampler.is_corroborated());
    }

    #[test]
    fn a_work_counter_that_went_backwards_corrupts_nothing() {
        // A restarted or reset counter is a fact about the caller, not evidence
        // that the window served `u64::MAX` requests. The delta contributes
        // nothing rather than saturating into corroboration.
        let mut sampler = HardwareSampler::new();
        sampler.observe_work(5_000);
        sampler.observe_work(0);
        assert!(!sampler.is_corroborated());
        sampler.observe_work(7);
        assert!(sampler.is_corroborated());
    }

    #[test]
    fn every_sample_becomes_a_worker_scoped_observation() {
        let mut sampler = HardwareSampler::new();
        let stamp = kivi_observation::ObservationStamp {
            window: kivi_observation::ObservationWindowId::from_u64(1),
            sequence: kivi_observation::ObservationSequence::from_u64(1),
        };
        let period = kivi_observation::ObservationPeriod::new(Duration::from_secs(1))
            .expect("a nonzero period");
        let (observation, sample) = sampler.observation(
            WorkerId::from_u64(7),
            stamp,
            period,
            Ticks::from_micros(1_000),
        );
        assert_eq!(
            observation.scope,
            kivi_observation::ObservationScope::Worker(WorkerId::from_u64(7))
        );
        assert!(matches!(observation.signal, TypedSignal::Hardware(_)));
        assert_eq!(
            observation.metadata.source,
            ReadingSource::None.observation_source()
        );
        // The sample and the observation must agree, or a controller reading
        // one and a human reading the other would draw opposite conclusions.
        assert_eq!(
            observation.metadata.confidence,
            sample.reading.observation_confidence()
        );
    }

    #[test]
    fn sources_map_to_observation_provenance() {
        assert_eq!(
            ReadingSource::PerformanceCounters.observation_source(),
            ObservationSource::SampledTelemetry
        );
        assert_eq!(
            ReadingSource::MemoryAccessMonitor.observation_source(),
            ObservationSource::SampledTelemetry
        );
        assert_eq!(ReadingSource::None.as_str(), "none");
        assert_eq!(format!("{}", ReadingSource::PerformanceCounters), "pmu");
    }
}
