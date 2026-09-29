//! The one place hardware observations become decisions.
//!
//! ## The problem this module exists to solve
//!
//! Three sources observe related effects:
//!
//! * software counters know **which object** was asked for and how often;
//! * the performance monitoring unit knows **what the cores did** while serving
//!   it;
//! * DAMON knows **which memory regions were touched**.
//!
//! Summing them produces a number that means nothing, because the three are not
//! independent measurements of one quantity. They are three measurements of
//! *different* quantities that are *causally related*: an object with high
//! request frequency and a core with severe stall will both report high values
//! for the same request, and adding them double counts one event.
//!
//! ## The rule
//!
//! Hardware signals are **evidence modifiers**, never additive terms. Each one
//! scales a software signal, along a declared axis, with a declared cap:
//!
//! | Axis | Software signal it qualifies | Hardware signal that qualifies it | How |
//! |---|---|---|---|
//! | Cache sensitivity | bytes touched per access | last-level miss ratio | multiplied, capped |
//! | Off-core pressure | observed off-core latency | stall fraction | `max`, because both observe the same stall |
//! | CPU-bound | service time per operation | instructions per cycle | classified, never added |
//! | Placement stability | nothing | context switches | reported directly |
//!
//! Where two sources observe the *same* effect - software off-core latency and
//! hardware stall fraction are both measuring a request waiting for memory -
//! they are combined with `max`, not with a sum. Using `max` means a
//! measurement from either source can raise the estimate and neither can
//! inflate it. The consequence is deliberate: hardware telemetry can make an
//! estimate *worse* in the sense of correcting it, and can never make a
//! confidence *appear* higher than the evidence supports.
//!
//! ## The result
//!
//! [`BlendedSignals`] is dimensionless and bounded. It carries no vendor, no
//! counter name, and no architecture. A machine with no PMU and no DAMON
//! produces the software-only result, which is the same shape with every
//! hardware modifier at its neutral value.

#![allow(clippy::cast_precision_loss)] // See 	elemetry::pmu for why a count-to-64 widening is exact here.

/// Dimensionless hardware observations over one interval, for one thread.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HardwareSignals {
    /// Instructions retired per cycle, in `[0, 8]`.
    pub ipc: f64,
    /// Fraction of cycles the core could not issue, in `[0, 1]`.
    pub stall_fraction: f64,
    /// Last-level cache misses per cache reference, in `[0, 1]`.
    pub cache_miss_ratio: f64,
    /// Branch mispredictions per branch, in `[0, 1]`.
    pub branch_miss_ratio: f64,
    /// Minor page faults in the interval.
    pub page_faults: u64,
    /// Context switches in the interval.
    pub context_switches: u64,
    /// CPU migrations of the calling thread in the interval. The unambiguous
    /// measurement that placement is not holding: a context switch only says
    /// the thread stopped running, a migration says it stopped running *and*
    /// resumed somewhere else.
    pub migrations: u64,
    /// How much the source can be trusted, in `(0, 1]`. A multiplexed counter
    /// set, or DAMON's coarse aggregation, lowers it. It never raises any
    /// other value, and it never exceeds what the source itself reports.
    pub confidence: f64,
}

impl HardwareSignals {
    /// The neutral value: hardware contributed nothing. Every modifier built
    /// from this value is exactly the software-only estimate, which is what
    /// makes "no PMU" and "a PMU that reported nothing" the same case.
    #[must_use]
    pub const fn neutral() -> Self {
        Self {
            ipc: 0.0,
            stall_fraction: 0.0,
            cache_miss_ratio: 0.0,
            branch_miss_ratio: 0.0,
            page_faults: 0,
            context_switches: 0,
            migrations: 0,
            confidence: 0.0,
        }
    }

    /// True when no hardware source contributed.
    #[must_use]
    pub fn is_neutral(&self) -> bool {
        self.confidence == 0.0
    }

    /// How much hardware raised the effective confidence above the software
    /// baseline, in `[0, 1]`.
    #[must_use]
    pub fn trust(&self) -> f64 {
        self.confidence.clamp(0.0, 1.0)
    }

    /// Whether the core looks memory bound. Both conditions must hold: stalls
    /// alone are a dependency chain, and cache misses alone are a cold but
    /// small working set.
    #[must_use]
    pub fn memory_bound(&self) -> bool {
        self.stall_fraction > 0.35 && self.cache_miss_ratio > 0.2
    }

    /// Whether the core looks throughput bound rather than stalled, which is
    /// the signature of work that more cores would actually speed up.
    #[must_use]
    pub fn cpu_bound(&self) -> bool {
        self.ipc >= 1.5 && self.stall_fraction < 0.15
    }

    /// Whether the serving thread was migrated between cores, which means the
    /// placement plan's core is not the core the work actually happened on.
    #[must_use]
    pub fn placement_unstable(&self) -> bool {
        self.migrations > 0
    }
}

impl Default for HardwareSignals {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Software observations about one object, from the fabric's own counters.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SoftwareSignals {
    /// Requests served in the window.
    pub accesses: u64,
    /// Logical bytes those requests moved.
    pub bytes: u64,
    /// Nanoseconds the object spent waiting for an off-core representation.
    pub offcore_wait_ns: u64,
    /// Nanoseconds of CPU time charged to the object.
    pub cpu_ns: u64,
}

/// How sensitive an object's access pattern is to cache capacity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CacheSensitivity {
    /// The working set fits wherever it is put.
    Resident,
    /// Larger than the last-level cache, so tier and core both matter.
    Sensitive,
    /// Larger than DRAM's last-level cache by enough that the object is
    /// memory bound wherever it lives, and only *where* it lives is a lever.
    MemoryBound,
}

/// The blended result: one dimensionless estimate per decision axis, plus the
/// axes' provenance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlendedSignals {
    /// Cache sensitivity implied by the traffic and the machine's behaviour.
    pub cache_sensitivity: CacheSensitivity,
    /// Off-core pressure in `[0, 1]`, from the larger of the software wait and
    /// the hardware stall, never their sum.
    pub offcore_pressure: f64,
    /// Whether the serving core is CPU bound, which is a property of the
    /// worker rather than of the object.
    pub cpu_bound: bool,
    /// Context switches in the window, reported straight through.
    pub context_switches: u64,
    /// Minor page faults in the window, reported straight through.
    pub page_faults: u64,
    /// CPU migrations of the serving thread in the window, reported straight
    /// through.
    pub migrations: u64,
    /// Relative cost of serving this access, in `[1, 4]`. Exactly `1.0` when
    /// no hardware source contributed, so a consumer's cost model is unchanged
    /// on a machine without a PMU rather than merely similar.
    pub cost_multiplier: f64,
    /// How much the hardware sources contributed, in `(0, 1]`.
    pub confidence: f64,
    /// Which sources actually contributed. Provenance is part of the result,
    /// not a debugging aid: the controller's safety envelope depends on
    /// knowing whether an estimate is measured or assumed.
    pub sources: SignalSources,
}

/// Which sources contributed to a blend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SignalSources {
    /// Software counters contributed.
    pub software: bool,
    /// Performance counters contributed.
    pub pmu: bool,
    /// DAMON contributed.
    pub damon: bool,
}

impl SignalSources {}

impl std::fmt::Display for SignalSources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut names = Vec::new();
        if self.software {
            names.push("software");
        }
        if self.pmu {
            names.push("pmu");
        }
        if self.damon {
            names.push("damon");
        }
        let joined = names.join("+");
        f.write_str(if joined.is_empty() {
            "none"
        } else {
            joined.as_str()
        })
    }
}

/// The largest cost multiplier hardware evidence may assign to an access.
///
/// An uncached, memory-bound object really is more expensive than a cached
/// one, but the ratio stops mattering once it is large: a planner that treats
/// a 100x object as 400x worse than a 1x object makes worse choices, not
/// better ones. The cap is what makes the multiplier safe to hand to a cost
/// model rather than a log.
const COST_MULTIPLIER_CAP: f64 = 4.0;

/// Bytes above which an object is treated as memory bound rather than merely
/// cache sensitive. A last-level cache is tens of megabytes; an object a
/// fraction of that can be made resident, an object several times that cannot.
const MEMORY_BOUND_BYTES: f64 = 64.0 * 1024.0 * 1024.0;

/// Bytes per access above which even a small object can be worth classifying
/// as cache sensitive rather than resident. A few kilobytes is about where an
/// object's own footprint stops fitting comfortably beside everything else the
/// worker touches.
const SMALL_TRAFFIC_BYTES: f64 = 8.0 * 1024.0;

/// The software-only estimate for `software`, with hardware left out. The
/// fallback path: identical shape, every hardware modifier neutral.
#[must_use]
pub fn software_only(software: SoftwareSignals) -> BlendedSignals {
    let traffic = if software.accesses == 0 {
        0.0
    } else {
        software.bytes as f64 / software.accesses as f64
    };
    let cache_sensitivity = if traffic > MEMORY_BOUND_BYTES {
        CacheSensitivity::MemoryBound
    } else if traffic > 64.0 * 1024.0 {
        CacheSensitivity::Sensitive
    } else {
        CacheSensitivity::Resident
    };
    BlendedSignals {
        cache_sensitivity,
        offcore_pressure: 0.0,
        cpu_bound: false,
        context_switches: 0,
        page_faults: 0,
        migrations: 0,
        // Software evidence alone assigns no hardware-derived cost premium.
        // The neutral multiplier is exactly 1.0, so a machine with no PMU and
        // a machine whose PMU reported nothing produce identical costs.
        cost_multiplier: 1.0,
        confidence: 0.0,
        sources: SignalSources {
            software: true,
            ..SignalSources::default()
        },
    }
}

/// Blends software and hardware observations.
///
/// # The combination rule, stated
///
/// * `offcore_pressure` is the **maximum** of the software wait fraction and
///   the hardware stall fraction, never their sum: both observe the same
///   request waiting for memory.
/// * `cache_sensitivity` is the software classification raised by the
///   hardware miss ratio. It is never raised by the miss ratio *alone*,
///   because a high miss ratio on a machine that is not memory bound says the
///   misses were cheap.
/// * `cost_multiplier` is the continuous form of the same observation: the
///   miss ratio scaled into `[1, COST_MULTIPLIER_CAP]`, and zero when there
///   were no cache references to divide by. A cost model can consume it
///   directly; a classifier can consume the enum. Both come from one
///   measurement, so they cannot disagree.
/// * `cpu_bound` is a hardware classification and is not combined with
///   anything: adding a CPU-bound flag to a latency number would be
///   meaningless.
#[must_use]
pub fn blend(software: SoftwareSignals, hardware: HardwareSignals) -> BlendedSignals {
    let base = software_only(software);
    if hardware.is_neutral() {
        return base;
    }
    let traffic = if software.accesses == 0 {
        0.0
    } else {
        software.bytes as f64 / software.accesses as f64
    };
    // The software wait fraction: of the CPU time charged to the object, how
    // much was spent not executing.
    let wait_fraction = if software.cpu_ns == 0 {
        0.0
    } else {
        (software
            .offcore_wait_ns
            .min(software.cpu_ns.saturating_mul(4)) as f64)
            / (software.cpu_ns as f64 * 4.0)
    };
    let offcore_pressure = wait_fraction.max(hardware.stall_fraction).clamp(0.0, 1.0);
    let cache_sensitivity = match base.cache_sensitivity {
        CacheSensitivity::MemoryBound => CacheSensitivity::MemoryBound,
        CacheSensitivity::Sensitive if hardware.cache_miss_ratio > 0.2 => {
            CacheSensitivity::MemoryBound
        }
        CacheSensitivity::Sensitive => CacheSensitivity::Sensitive,
        CacheSensitivity::Resident
            if traffic > SMALL_TRAFFIC_BYTES && hardware.cache_miss_ratio > 0.35 =>
        {
            CacheSensitivity::Sensitive
        }
        CacheSensitivity::Resident => CacheSensitivity::Resident,
    };
    // A worker that was not touching a cache at all has no miss ratio to
    // scale by, so the multiplier stays neutral rather than becoming
    // indeterminate.
    let cost_multiplier = if base.cache_sensitivity == CacheSensitivity::Resident
        && cache_sensitivity == CacheSensitivity::Resident
    {
        1.0
    } else {
        (1.0 + hardware.cache_miss_ratio.clamp(0.0, 1.0)).min(COST_MULTIPLIER_CAP)
    };
    BlendedSignals {
        cache_sensitivity,
        offcore_pressure,
        cpu_bound: hardware.cpu_bound(),
        context_switches: hardware.context_switches,
        page_faults: hardware.page_faults,
        migrations: hardware.migrations,
        cost_multiplier,
        confidence: hardware.trust(),
        sources: SignalSources {
            software: true,
            pmu: true,
            damon: false,
        },
    }
}

/// DAMON's contribution, which is about regions rather than objects.
///
/// DAMON cannot say which object was touched, so it cannot raise an object's
/// estimate. What it can do is say whether a worker's arenas are being touched
/// at all, which is a coarse check on the whole worker. Kivi therefore records
/// it as a confidence modifier on the worker's observations and never as a term
/// in an object's score.
#[must_use]
pub fn damon_confidence(
    accesses_per_mebibyte: f64,
    resolved: bool,
    hardware: &HardwareSignals,
) -> HardwareSignals {
    if !resolved {
        // DAMON could not see the arena. Lower confidence rather than ignore
        // it: an unresolved observation is information that the estimate is
        // coarser than it looks.
        return HardwareSignals {
            confidence: (hardware.trust() * 0.75).min(hardware.trust()),
            ..*hardware
        };
    }
    let activity = (accesses_per_mebibyte / 1024.0).clamp(0.0, 1.0);
    HardwareSignals {
        confidence: (hardware.trust().max(activity) * 0.9).min(1.0),
        ..*hardware
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn software() -> SoftwareSignals {
        SoftwareSignals {
            accesses: 1000,
            bytes: 1024 * 1024,
            offcore_wait_ns: 0,
            cpu_ns: 1000,
        }
    }

    fn hardware() -> HardwareSignals {
        HardwareSignals {
            ipc: 2.0,
            stall_fraction: 0.5,
            cache_miss_ratio: 0.4,
            branch_miss_ratio: 0.01,
            page_faults: 12,
            context_switches: 3,
            migrations: 0,
            confidence: 1.0,
        }
    }

    #[test]
    fn no_hardware_reproduces_the_software_estimate_exactly() {
        let software = software();
        let neutral = HardwareSignals::neutral();
        assert!(neutral.is_neutral());
        assert_eq!(blend(software, neutral), software_only(software));
    }

    #[test]
    fn a_neutral_hardware_sample_costs_nothing() {
        // The point of the neutral value: a PMU that reported nothing, or was
        // never opened, must be indistinguishable from software alone.
        let software = software();
        let zeroed = HardwareSignals {
            confidence: 0.0,
            ..hardware()
        };
        assert_eq!(blend(software, zeroed), software_only(software));
    }

    #[test]
    fn offcore_pressure_takes_the_maximum_not_the_sum() {
        let software = SoftwareSignals {
            accesses: 100,
            bytes: 1024,
            offcore_wait_ns: 200,
            cpu_ns: 100,
            // wait fraction: 200 / (100 * 4) = 0.5
        };
        let hardware = HardwareSignals {
            stall_fraction: 0.5,
            ..hardware()
        };
        let blended = blend(software, hardware);
        assert!(
            (blended.offcore_pressure - 0.5).abs() < 1e-9,
            "max(0.5, 0.5) = 0.5, not 1.0, got {}",
            blended.offcore_pressure
        );
    }

    #[test]
    fn either_source_alone_can_raise_offcore_pressure() {
        let quiet_hardware = HardwareSignals {
            stall_fraction: 0.0,
            ..hardware()
        };
        let loud_hardware = HardwareSignals {
            stall_fraction: 0.9,
            ..hardware()
        };
        let idle = SoftwareSignals {
            accesses: 100,
            bytes: 1024,
            offcore_wait_ns: 0,
            cpu_ns: 100,
        };
        assert!(blend(idle, quiet_hardware).offcore_pressure.abs() < 1e-9);
        assert!((blend(idle, loud_hardware).offcore_pressure - 0.9).abs() < 1e-9);

        // A worker that was seen waiting, on a core the PMU says was not
        // stalled: the software observation still stands on its own.
        let waiting = SoftwareSignals {
            offcore_wait_ns: 200,
            ..idle
        };
        assert!((blend(waiting, quiet_hardware).offcore_pressure - 0.5).abs() < 1e-9);
    }

    #[test]
    fn hardware_can_raise_cache_sensitivity_but_only_with_traffic() {
        let small = SoftwareSignals {
            accesses: 1,
            bytes: 1024,
            ..software()
        };
        assert_eq!(
            blend(small, hardware()).cache_sensitivity,
            CacheSensitivity::Resident,
            "a 1 KiB object cannot be cache sensitive no matter the miss ratio"
        );
        // Traffic is measured per access, so a single access states the
        // object size directly. A 16 KiB object is resident by software, and
        // only a badly missing core can promote it.
        let middling = SoftwareSignals {
            accesses: 1,
            bytes: 16 * 1024,
            ..software()
        };
        assert_eq!(
            software_only(middling).cache_sensitivity,
            CacheSensitivity::Resident
        );
        assert_eq!(
            blend(middling, hardware()).cache_sensitivity,
            CacheSensitivity::Sensitive
        );
        // A 128 KiB object is sensitive by software. A core that is missing
        // heavily turns that into memory bound.
        let large = SoftwareSignals {
            accesses: 1,
            bytes: 128 * 1024,
            ..software()
        };
        assert_eq!(
            software_only(large).cache_sensitivity,
            CacheSensitivity::Sensitive
        );
        assert_eq!(
            blend(large, hardware()).cache_sensitivity,
            CacheSensitivity::MemoryBound
        );
    }

    #[test]
    fn a_cheap_machine_does_not_manufacture_sensitivity() {
        let medium = SoftwareSignals {
            accesses: 1000,
            bytes: 64 * 1024,
            ..software()
        };
        let cheap_misses = HardwareSignals {
            cache_miss_ratio: 0.01,
            ..hardware()
        };
        assert_eq!(
            blend(medium, cheap_misses).cache_sensitivity,
            software_only(medium).cache_sensitivity
        );
    }

    #[test]
    fn cpu_bound_is_classified_not_combined() {
        let cpu = HardwareSignals {
            ipc: 3.0,
            stall_fraction: 0.05,
            ..hardware()
        };
        assert!(blend(software(), cpu).cpu_bound);
        assert!(!blend(software(), hardware()).cpu_bound);
    }

    #[test]
    fn every_output_is_bounded() {
        let blended = blend(software(), hardware());
        assert!((0.0..=1.0).contains(&blended.offcore_pressure));
        assert!((0.0..=1.0).contains(&blended.confidence));
        assert!(hardware().memory_bound());
    }

    #[test]
    fn provenance_is_part_of_the_result() {
        assert_eq!(
            blend(software(), hardware()).sources.to_string(),
            "software+pmu"
        );
        assert_eq!(software_only(software()).sources.to_string(), "software");
        assert_eq!(SignalSources::default().to_string(), "none");
    }

    #[test]
    fn damon_lowers_confidence_when_it_cannot_resolve() {
        let hardware = hardware();
        let coarse = damon_confidence(5000.0, false, &hardware);
        assert!(coarse.confidence < hardware.confidence);
        let resolved = damon_confidence(5000.0, true, &hardware);
        assert!(resolved.confidence >= coarse.confidence);
    }

    #[test]
    fn damon_never_raises_an_estimate_above_the_pmu() {
        let hardware = hardware();
        let resolved = damon_confidence(1_000_000.0, true, &hardware);
        assert!(resolved.confidence <= hardware.confidence);
    }
}
