//! Hardware execution observations, on their own axis.
//!
//! ## What a counter can and cannot tell Kivi
//!
//! The valuable quantities are all *ratios between counters over the same
//! interval*, because a ratio is insensitive to clock speed, to how busy the
//! core was, and to how long the interval was. Absolute counts over an
//! interval are close to useless on their own.
//!
//! | Signal | Reading | Kivi decision it improves |
//! |---|---|---|
//! | `ipc` | Instructions per cycle. Near 1 is a dependency chain; near 4 is throughput bound. | Separates a CPU-bound worker from a stalled one. |
//! | `stall_fraction` | Fraction of cycles the core could not issue. | Distinguishes high-frequency-but-cheap from moderate-frequency-and-terrible. |
//! | `cache_miss_ratio` | Last-level cache misses per cache reference. | Whether a working set deserves a larger tier or a different core. |
//! | `branch_miss_ratio` | Branch mispredictions per branch. | Whether a branchy kernel is paying more in mispredicts than in work. |
//! | `page_faults`, `context_switches` | Per interval, not per request. | Whether placement is holding and whether an arena is being touched cold. |
//!
//! Deliberately not collected: raw cycle counts and raw instruction counts as
//! ends in themselves, and anything that would need a per-request syscall. A
//! counter Kivi cannot act on is overhead with no return.
//!
//! ## Why a group, and not one counter per event
//!
//! The kernel schedules each counter independently and multiplexes them when
//! there are not enough hardware registers. Two independently-scheduled
//! counters therefore observe *different* sets of instructions, and the ratio
//! of their counts is not a ratio of anything: it moves with scheduling
//! pressure rather than with the workload. Every ratio in this module is
//! meaningless unless its numerator and denominator counted the same
//! instructions.
//!
//! The whole set is opened as one group, which the kernel only schedules when
//! every member fits, so members are always commensurable. When a group still
//! cannot be scheduled for the full interval - more events than the PMU has
//! counters, and other software competing for them - the kernel reports
//! `time_running < time_enabled`, the sample is marked [`PmuSample::scaled`],
//! and every consumer lowers its confidence.
//!
//! ## Capability is proportional, not binary
//!
//! Machines disagree about which events they have, and the disagreements are
//! not hypothetical. A virtual machine with no virtualised PMU refuses every
//! `PERF_TYPE_HARDWARE` event with `ENOENT` while accepting every
//! `PERF_TYPE_SOFTWARE` one. A distribution that sets
//! `perf_event_paranoid = 4` refuses hardware events for unprivileged
//! processes. An older CPU has cycles and instructions but no stall counter.
//!
//! The first version of this module opened a group led by hardware cycles and
//! reported "unavailable" whenever that failed. That threw away page faults and
//! context switches on exactly the machines where they are the only hardware
//! evidence available, and where the placement-stability question is most worth
//! asking, so the design is proportional:
//! * the **denominator** is the best event the kernel will give: hardware
//!   cycles if available, otherwise the per-thread task clock
//!   (`PERF_COUNT_SW_TASK_CLOCK`, nanoseconds, never refused), and
//!   [`PmuSample::cycle_source`] records which one it is;
//! * **numerators** are added one at a time, each only if the kernel grants it;
//! * what a sample means depends on the denominator, so the two rate readings
//!   are separate methods rather than one field that is sometimes IPC and
//!   sometimes instructions per nanosecond.
//!
//! ## Sampling
//!
//! One `read` syscall per interval covers every counter, because
//! `PERF_FORMAT_GROUP` returns the whole group from the leader's descriptor.
//! Nine separate descriptors would mean nine syscalls for the same answer.

#![allow(clippy::cast_precision_loss)]
// Counter ratios are exactly the case where widening to `f64` is harmless: the
// numerator and denominator are both whole counts of events in one interval, the
// quotient is a ratio in a small fixed range, and an `f64`'s 52-bit mantissa
// represents every count below 2^52 exactly.

use std::fmt;

use crate::capability::{PmuCounter, PmuSupport, Support};

/// What the sample's denominator counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleSource {
    /// Hardware cycles. This is the only source for which instructions per
    /// cycle is a meaningful quantity.
    Hardware,
    /// `PERF_COUNT_SW_TASK_CLOCK`: nanoseconds this thread was scheduled. A
    /// real, comparable denominator that every kernel provides, but it moves
    /// with frequency, so a ratio against it is not IPC.
    TaskClockNanoseconds,
    /// The group was never opened.
    None,
}

impl CycleSource {
    /// Whether this source supports an instructions-per-cycle reading.
    #[must_use]
    pub const fn is_cycles(self) -> bool {
        matches!(self, Self::Hardware)
    }
}

impl fmt::Display for CycleSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hardware => f.write_str("hardware-cycles"),
            Self::TaskClockNanoseconds => f.write_str("task-clock-ns"),
            Self::None => f.write_str("none"),
        }
    }
}

/// One interval of execution observations for one thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PmuSample {
    /// The denominator: hardware cycles, or scheduled nanoseconds.
    pub cycles: u64,
    /// What [`PmuSample::cycles`] counts.
    pub cycle_source: CycleSource,
    /// Instructions retired.
    pub instructions: u64,
    /// Last-level cache references.
    pub cache_references: u64,
    /// Last-level cache misses.
    pub cache_misses: u64,
    /// Cycles the core could not issue an instruction.
    pub stalled_cycles: u64,
    /// Branches retired.
    pub branches: u64,
    /// Branch mispredictions.
    pub branch_misses: u64,
    /// Minor page faults.
    pub page_faults: u64,
    /// Involuntary context switches.
    pub context_switches: u64,
    /// CPU migrations of this thread, the direct measurement of a placement
    /// that is not holding.
    pub migrations: u64,
    /// Whether the kernel could not schedule the whole group for the full
    /// interval, in which case every count is a scaled estimate.
    pub scaled: bool,
    /// Whether the counters in this sample were scheduled together, which is
    /// the precondition for every ratio over them.
    ///
    /// `false` means the kernel would not form a group and each counter ran
    /// independently. The absolute counts are then still exact - a
    /// per-thread page-fault counter does not need a second counter to be
    /// true - but nothing may be divided by anything else in the sample, so
    /// every ratio method returns `None` rather than a plausible wrong number.
    pub comparable: bool,
}

impl PmuSample {
    /// A zero sample, used before the first interval completes and when no
    /// counter is open. A missing counter is a zero measurement, not an
    /// infinite one: the ratios below must stay defined.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            cycles: 0,
            cycle_source: CycleSource::None,
            instructions: 0,
            cache_references: 0,
            cache_misses: 0,
            stalled_cycles: 0,
            branches: 0,
            branch_misses: 0,
            page_faults: 0,
            context_switches: 0,
            migrations: 0,
            scaled: false,
            comparable: true,
        }
    }

    /// Writes one counter's count into its field, by tag rather than by
    /// position, so a kernel that refuses one event cannot shift every
    /// subsequent one by one slot.
    #[cfg_attr(
        not(all(feature = "pmu", any(target_os = "linux", target_os = "android"))),
        allow(
            dead_code,
            reason = "only a backend that can name a counter needs to write one"
        )
    )]
    fn set(&mut self, counter: PmuCounter, count: u64) {
        match counter {
            // The task clock is a denominator, and `cycles` is where a sample
            // keeps whichever denominator it got - hardware cycles when the
            // kernel had them, nanoseconds of scheduled time when it did not.
            // `cycle_source` records which one it was, so the two never have to
            // be told apart here.
            PmuCounter::Cycles | PmuCounter::TaskClock => self.cycles = count,
            PmuCounter::Instructions => self.instructions = count,
            PmuCounter::CacheReferences => self.cache_references = count,
            PmuCounter::CacheMisses => self.cache_misses = count,
            PmuCounter::StalledCycles => self.stalled_cycles = count,
            PmuCounter::Branches => self.branches = count,
            PmuCounter::BranchMisses => self.branch_misses = count,
            PmuCounter::PageFaults => self.page_faults = count,
            PmuCounter::ContextSwitches => self.context_switches = count,
            PmuCounter::Migrations => self.migrations = count,
        }
    }

    /// Whether any ratio may be computed over this sample.
    #[must_use]
    pub const fn comparable(&self) -> bool {
        self.comparable
    }

    /// Instructions retired per hardware cycle, or `None` when there are no
    /// hardware cycles or no instructions.
    ///
    /// `None` on a task-clock denominator is deliberate. "Instructions per
    /// nanosecond" moves with the core's frequency, so reporting it here would
    /// make a frequency change look like a change in the workload.
    #[must_use]
    pub fn ipc(&self) -> Option<f64> {
        if !self.comparable || !self.cycle_source.is_cycles() || self.cycles == 0 {
            return None;
        }
        Some((self.instructions as f64 / self.cycles as f64).clamp(0.0, 8.0))
    }

    /// Instructions retired per nanosecond of scheduled time, when that is what
    /// the denominator is.
    ///
    /// Comparable between intervals on the same machine, and a statement about
    /// how much work a worker retired rather than how efficiently it issued.
    #[must_use]
    pub fn instructions_per_ns(&self) -> Option<f64> {
        if !self.comparable
            || self.cycle_source != CycleSource::TaskClockNanoseconds
            || self.cycles == 0
        {
            return None;
        }
        Some((self.instructions as f64 / self.cycles as f64).clamp(0.0, 8.0))
    }

    /// Fraction of the interval the core could not issue an instruction, in
    /// `[0, 1]`.
    ///
    /// Zero when there is no hardware cycle count, because a stall counter
    /// without a cycle count has no denominator and inventing one would produce
    /// a plausible-looking number with no meaning.
    #[must_use]
    pub fn stall_fraction(&self) -> f64 {
        if !self.comparable || !self.cycle_source.is_cycles() || self.cycles == 0 {
            return 0.0;
        }
        (self.stalled_cycles.min(self.cycles) as f64) / self.cycles as f64
    }

    /// Last-level cache misses per cache reference, in `[0, 1]`. A high value
    /// means the working set does not fit the shared cache, which is the
    /// cheapest early warning there is before a memory-bound tail appears.
    #[must_use]
    pub fn cache_miss_ratio(&self) -> f64 {
        if !self.comparable || self.cache_references == 0 {
            return 0.0;
        }
        (self.cache_misses.min(self.cache_references) as f64) / self.cache_references as f64
    }

    /// Branch mispredictions per branch, in `[0, 1]`.
    #[must_use]
    pub fn branch_miss_ratio(&self) -> f64 {
        if !self.comparable || self.branches == 0 {
            return 0.0;
        }
        (self.branch_misses.min(self.branches) as f64) / self.branches as f64
    }

    /// Whether the core looks memory bound: most cycles stalled *and* the last
    /// level cache was missing. Either alone is ambiguous - stalls also come
    /// from a dependency chain, misses also come from a cold but small working
    /// set - but together they are a statement about memory.
    #[must_use]
    pub fn memory_bound(&self) -> bool {
        self.stall_fraction() > 0.35 && self.cache_miss_ratio() > 0.2
    }

    /// Whether placement is not holding for this thread.
    ///
    /// A migration is unambiguous: the thread was moved between cores, which
    /// means the cache the plan assumed is not the cache the worker has.
    /// Context switches alone are not evidence, because a thread that simply
    /// was not the one running is not a misplaced thread.
    #[must_use]
    pub const fn placement_unstable(&self) -> bool {
        self.migrations > 0
    }

    /// The interval as generic signals, which is the only form a consumer
    /// above this crate may use.
    #[must_use]
    pub fn signals(&self) -> crate::telemetry::signals::HardwareSignals {
        crate::telemetry::signals::HardwareSignals {
            ipc: self.ipc().unwrap_or(0.0),
            stall_fraction: self.stall_fraction(),
            cache_miss_ratio: self.cache_miss_ratio(),
            branch_miss_ratio: self.branch_miss_ratio(),
            page_faults: self.page_faults,
            context_switches: self.context_switches,
            migrations: self.migrations,
            confidence: self.trust(),
        }
    }

    /// How much this sample can be trusted, in `(0, 1]`.
    ///
    /// Three things can lower it, and they are different:
    ///
    /// * a group the kernel could not schedule for the full interval, so every
    ///   count is a scaled estimate;
    /// * counters that were not scheduled together, so nothing in the sample
    ///   can be divided by anything else in it;
    /// * a sample with no counted events at all, which is the *baseline* a
    ///   counter set establishes when it is opened, not a measurement of an
    ///   interval. Reporting full confidence for it would let a controller act
    ///   on the absence of evidence.
    #[must_use]
    pub const fn trust(&self) -> f64 {
        if self.scaled {
            0.5
        } else if !self.comparable || self.cycles == 0 {
            0.0
        } else {
            1.0
        }
    }

    /// The difference between two samples taken on adjacent intervals.
    ///
    /// Counters are monotonic within a thread's lifetime, so a restart or a
    /// counter reset produces a negative delta. That is reported as an
    /// [`Ordering::Relaxed`] [`Delta`] rather than saturated to zero, because a
    /// silent zero would look like an idle interval.
    #[must_use]
    pub fn since(&self, earlier: &Self) -> Delta {
        Delta {
            cycles: self.cycles.saturating_sub(earlier.cycles),
            instructions: self.instructions.saturating_sub(earlier.instructions),
            cache_references: self
                .cache_references
                .saturating_sub(earlier.cache_references),
            cache_misses: self.cache_misses.saturating_sub(earlier.cache_misses),
            stalled_cycles: self.stalled_cycles.saturating_sub(earlier.stalled_cycles),
            branches: self.branches.saturating_sub(earlier.branches),
            branch_misses: self.branch_misses.saturating_sub(earlier.branch_misses),
            page_faults: self.page_faults.saturating_sub(earlier.page_faults),
            context_switches: self
                .context_switches
                .saturating_sub(earlier.context_switches),
            migrations: self.migrations.saturating_sub(earlier.migrations),
            ordering: if self.cycles >= earlier.cycles {
                Ordering::Monotonic
            } else {
                Ordering::Relaxed
            },
        }
    }
}

impl fmt::Display for PmuSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "pmu(denom={} {} insns={} llc_ref={} llc_miss={} stall={:.2} ipc={:.2} \
             br_miss={:.3} pf={} cs={} mig={} scaled={})",
            self.cycle_source,
            self.cycles,
            self.instructions,
            self.cache_references,
            self.cache_misses,
            self.stall_fraction(),
            self.ipc().unwrap_or(0.0),
            self.branch_miss_ratio(),
            self.page_faults,
            self.context_switches,
            self.migrations,
            self.scaled,
        )
    }
}

/// Whether two samples can be subtracted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ordering {
    /// The later sample is not below the earlier one, so the delta is exact.
    Monotonic,
    /// The later sample is below the earlier one, so the thread restarted or a
    /// counter was reset. The delta is a lower bound and the sample should not
    /// be compared with earlier ones.
    Relaxed,
}

/// The change in a counter set across one interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Delta {
    /// The denominator advanced.
    pub cycles: u64,
    /// Instructions advanced.
    pub instructions: u64,
    /// Cache references advanced.
    pub cache_references: u64,
    /// Cache misses advanced.
    pub cache_misses: u64,
    /// Stalled cycles advanced.
    pub stalled_cycles: u64,
    /// Branches advanced.
    pub branches: u64,
    /// Branch mispredictions advanced.
    pub branch_misses: u64,
    /// Page faults advanced.
    pub page_faults: u64,
    /// Context switches advanced.
    pub context_switches: u64,
    /// CPU migrations advanced.
    pub migrations: u64,
    /// Whether the subtraction is exact.
    pub ordering: Ordering,
}

/// A live counter group attached to the calling thread.
///
/// Construction is the only place a syscall happens. [`PmuCounters::sample`] is
/// one `read` per interval over the whole group, which is the cheapest form
/// the interface allows and the only one that keeps the counters commensurable.
///
/// A group is bound to the thread that opened it, which is the correct
/// lifetime for a per-thread counter: a counter opened on one thread counts
/// only that thread's instructions, so sampling it from another would report
/// the wrong thread's work.
pub struct PmuCounters {
    inner: Option<backend::Counters>,
    support: PmuSupport,
    source: CycleSource,
}

impl fmt::Debug for PmuCounters {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PmuCounters")
            .field("attached", &self.inner.is_some())
            .field("cycle_source", &self.source)
            .field("support", &self.support)
            .finish()
    }
}

impl PmuCounters {
    /// Opens a counter group for the calling thread, with whatever the kernel
    /// will grant.
    ///
    /// An unattached set is returned only when not even the per-thread task
    /// clock can be opened, which no Linux kernel in service refuses. Reading
    /// an unattached set yields a zero sample, which keeps every ratio defined
    /// and every decision unchanged.
    #[must_use]
    pub fn attach() -> Self {
        match backend::Counters::open() {
            Some((counters, source)) => {
                let support = counters.support();
                Self {
                    inner: Some(counters),
                    support,
                    source,
                }
            }
            None => Self {
                inner: None,
                support: PmuSupport::default(),
                source: CycleSource::None,
            },
        }
    }

    /// Whether counters are actually open.
    #[must_use]
    pub fn is_attached(&self) -> bool {
        self.inner.is_some()
    }

    /// What the open counters can measure.
    #[must_use]
    pub const fn support(&self) -> &PmuSupport {
        &self.support
    }

    /// What the sample's denominator counts. A consumer that wants an
    /// instructions-per-cycle reading must check this rather than assume one.
    #[must_use]
    pub const fn cycle_source(&self) -> CycleSource {
        self.source
    }

    /// Reads the whole group once.
    #[must_use]
    pub fn sample(&mut self) -> PmuSample {
        self.inner
            .as_mut()
            .map_or_else(PmuSample::zero, backend::Counters::read)
    }
}

/// Detects the platform's performance counters.
///
/// The distinction that matters is between "this machine has no such counter"
/// and "this process is not allowed to read it", because an operator fixes them
/// differently: the first is a VM or a virtualised PMU, the second is a
/// `perf_event_paranoid` setting. Kivi reports them as different values and
/// never requires either.
#[must_use]
pub fn detect() -> Support<PmuSupport> {
    backend::detect()
}

/// Why each individual event is or is not openable, one line each.
///
/// [`detect`] collapses every refusal into a single answer, which is the right
/// shape for a capability snapshot and the wrong shape for a machine an
/// operator is trying to configure. This returns the per-event detail: a
/// hypervisor that refuses `PERF_TYPE_HARDWARE` outright, a distribution that
/// set `perf_event_paranoid` too high, and a CPU that simply does not implement
/// a stall counter produce three different lines here and the same single line
/// there.
///
/// The probe opens and closes one counter per event, so the cost is a handful
/// of syscalls and it is never called from a request path.
#[must_use]
pub fn diagnose() -> Vec<(String, String)> {
    backend::diagnose()
}

#[cfg(all(feature = "pmu", any(target_os = "linux", target_os = "android")))]
mod backend {
    use perf_event_open::config::sibling::Opts as SiblingOpts;
    use perf_event_open::config::{Cpu, Opts, Priv, Proc, StatFormat};
    use perf_event_open::count::Counter;
    use perf_event_open::count::group::CounterGroup;
    use perf_event_open::event::hw::Hardware;
    use perf_event_open::event::sw::Software;

    use super::{CycleSource, PmuSample, PmuSupport};
    use crate::capability::{PmuCounter, Support, Unavailable};

    /// One group member and what it means, so a sample is built by tag rather
    /// than by position. The leader's count arrives first and every sibling
    /// after it, in the order the siblings were added; position would silently
    /// misattribute one event to another the moment the kernel refused one of
    /// them, which is the normal case on a partial PMU.
    ///
    /// The leader is not in this list because it is whichever denominator was
    /// opened first, and [`super::CycleSource`] records which.
    #[derive(Debug, Clone, Copy)]
    pub(super) struct Field(pub(super) PmuCounter);

    /// Whether the kernel ran a counter for less than the whole interval.
    ///
    /// `time_running < time_enabled` is the kernel's own statement that it
    /// stopped the counter to run something else, so the count it reports is a
    /// scaled estimate rather than the real number of events. A kernel that
    /// reports neither value is taken at its word: it declined to answer, and
    /// inventing "not multiplexed" from an absent answer would be the one
    /// reading that cannot be corrected later.
    const fn multiplexed(enabled: Option<u64>, running: Option<u64>) -> bool {
        match (enabled, running) {
            (Some(enabled), Some(running)) => enabled > 0 && running < enabled,
            _ => false,
        }
    }

    /// Hardware members requested of the kernel, in the order they are tried.
    /// A CPU that does not implement one simply refuses it and the group keeps
    /// going, so a machine with four programmable counters keeps four of these.
    const HW_MEMBERS: [(Hardware, PmuCounter); 6] = [
        (Hardware::Instr, PmuCounter::Instructions),
        (Hardware::CacheAccess, PmuCounter::CacheReferences),
        (Hardware::CacheMiss, PmuCounter::CacheMisses),
        (Hardware::BranchInstr, PmuCounter::Branches),
        (Hardware::BranchMiss, PmuCounter::BranchMisses),
        (Hardware::FrontendStalledCycle, PmuCounter::StalledCycles),
    ];

    /// Software members. These need no PMU and no privilege, which is why they
    /// are the part of this module that works on a machine with no virtualised
    /// performance counters at all.
    const SW_MEMBERS: [(Software, PmuCounter); 3] = [
        (Software::MinorPageFault, PmuCounter::PageFaults),
        (Software::CtxSwitch, PmuCounter::ContextSwitches),
        (Software::CpuMigration, PmuCounter::Migrations),
    ];

    /// How the open counters relate to one another, which decides whether any
    /// ratio over them is a statement about the workload.
    pub(super) enum Counters {
        /// Every counter is in one group, so all of them counted the same
        /// instructions and every ratio in [`PmuSample`] is meaningful.
        Grouped {
            group: CounterGroup,
            /// Field tag per sibling, in the order the siblings were added.
            order: Vec<Field>,
            source: CycleSource,
        },
        /// The kernel refused to form a group, so each counter was opened on its
        /// own.
        ///
        /// A ratio over independently opened counters is only meaningful when the
        /// kernel scheduled every one of them for the whole interval, which
        /// [`StatFormat::time_running`] against `time_enabled` is the kernel's own
        /// statement about. When that holds the sample stays commensurable even
        /// though the counters are independent; when it does not, every ratio is
        /// withheld. The absolute counts are exact either way: a per-thread
        /// page-fault counter does not need a second counter to be true.
        Independent {
            /// The interval denominator, when one could be opened. Kept apart
            /// from `counters` because it is a scale for `cycles`, not a value a
            /// consumer reads.
            denominator: Option<Counter>,
            /// The counted events, in open order.
            counters: Vec<(Field, Counter)>,
        },
    }

    impl Counters {
        pub(super) fn open() -> Option<(Self, CycleSource)> {
            // Count only user-mode instructions. An unprivileged process cannot
            // read kernel-mode events at `perf_event_paranoid >= 2`, and asking
            // for them anyway means the open fails rather than counting less.
            let exclude = Priv {
                kernel: true,
                hv: true,
                ..Priv::default()
            };
            let leader_opts = &Opts {
                // The whole point of the group: one read returns every member,
                // and `time_running < time_enabled` is how a multiplexed group
                // is detected.
                stat_format: StatFormat {
                    siblings: true,
                    time_enabled: true,
                    time_running: true,
                    ..StatFormat::default()
                },
                exclude,
                enable: true,
                ..Opts::default()
            };

            // Hardware cycles first, because a cycle denominator is what makes
            // every ratio in this module mean what it says. A machine without
            // them gets the per-thread task clock instead, which is a weaker but
            // real denominator, and the sample says which it got.
            let (leader, source) =
                match Counter::new(Hardware::CpuCycle, (Proc::CURRENT, Cpu::ALL), leader_opts) {
                    Ok(leader) => (leader, CycleSource::Hardware),
                    Err(_) => match Counter::new(
                        Software::TaskClock,
                        (Proc::CURRENT, Cpu::ALL),
                        leader_opts,
                    ) {
                        Ok(leader) => (leader, CycleSource::TaskClockNanoseconds),
                        Err(_) => return None,
                    },
                };
            let mut group = CounterGroup::from(leader);
            // Siblings take the narrower `sibling::Opts`, which has no
            // `exclude` or `enable` field: the kernel inherits those from the
            // group leader, which is exactly what a grouped event must do.
            let sibling = SiblingOpts::default();
            let mut order: Vec<Field> = Vec::new();

            for (event, counter) in HW_MEMBERS {
                if group.add(event, &sibling).is_ok() {
                    order.push(Field(counter));
                }
            }
            for (event, counter) in SW_MEMBERS {
                if group.add(event, &sibling).is_ok() {
                    order.push(Field(counter));
                }
            }
            if order.is_empty() {
                // The leader opened but no member would join the group. A
                // hypervisor that filters grouped events produces exactly this:
                // the leader succeeds, every `PERF_IOC_ADD_EVENT` is refused, and
                // the only usable shape is one counter per descriptor. WSL2 is
                // one such machine - `PERF_IOC_ADD_EVENT` on a software-event
                // group returns EACCES while standalone software counters open
                // cleanly.
                return Self::open_independent().map(|counters| {
                    let source = counters.denominator_source();
                    (counters, source)
                });
            }
            Some((
                Self::Grouped {
                    group,
                    order,
                    source,
                },
                source,
            ))
        }

        /// Opens each counter on its own.
        ///
        /// This is not a degraded mode bolted on; it is the only shape that
        /// works on a hypervisor that permits standalone software events but
        /// refuses to schedule a group. What it costs is decided *per sample*,
        /// not assumed: the counters are independently opened, so they are only
        /// commensurable if the kernel scheduled every one of them for the whole
        /// interval, and `read` checks exactly that.
        ///
        /// The interval denominator is opened here too. Discarding it because
        /// the group failed would throw away the one number that is still exact:
        /// the kernel's own record of how long this thread ran.
        fn open_independent() -> Option<Self> {
            // User-mode only, for the same reason the group leader is: an
            // unprivileged process cannot count kernel events at
            // `perf_event_paranoid >= 2`, and asking anyway turns a usable
            // counter into a refused one.
            let opts = Opts {
                exclude: Priv {
                    kernel: true,
                    hv: true,
                    ..Priv::default()
                },
                enable: true,
                stat_format: StatFormat {
                    time_enabled: true,
                    time_running: true,
                    ..StatFormat::default()
                },
                ..Opts::default()
            };
            let denominator = Counter::new(Software::TaskClock, (Proc::CURRENT, Cpu::ALL), &opts)
                .ok()
                .and_then(|handle| handle.enable().ok().map(|()| handle));
            let mut counters: Vec<(Field, Counter)> = Vec::new();
            for (event, counter) in SW_MEMBERS {
                let Ok(handle) = Counter::new(event, (Proc::CURRENT, Cpu::ALL), &opts) else {
                    continue;
                };
                if handle.enable().is_ok() {
                    counters.push((Field(counter), handle));
                }
            }
            // A denominator with nothing to count, or counts with nothing to
            // scale them by, is not a counter set. Returning `None` here is what
            // makes `PmuCounters` unattached rather than attached and empty.
            (denominator.is_some() || !counters.is_empty()).then_some(Self::Independent {
                denominator,
                counters,
            })
        }

        /// Exactly the counters this set actually opened, derived from the live
        /// handles. A support set that is not derived from them is a support set
        /// that can disagree with them, and every consumer here treats it as a
        /// promise.
        pub(super) fn support(&self) -> PmuSupport {
            let mut support = PmuSupport::default();
            match self {
                Self::Grouped { order, source, .. } => {
                    if source.is_cycles() {
                        support.insert(PmuCounter::Cycles);
                    }
                    for field in order {
                        support.insert(field.0);
                    }
                }
                Self::Independent {
                    counters,
                    denominator,
                } => {
                    if denominator.is_some() {
                        // The denominator is a real counter the kernel opened, so
                        // it belongs in the support set a caller may plan against.
                        support.insert(PmuCounter::TaskClock);
                    }
                    for (field, _) in counters {
                        support.insert(field.0);
                    }
                }
            }
            support
        }

        pub(super) fn read(&mut self) -> PmuSample {
            match self {
                Self::Grouped {
                    group,
                    order,
                    source,
                } => {
                    let mut sample = PmuSample {
                        cycle_source: *source,
                        ..PmuSample::zero()
                    };
                    let Ok(stat) = group.leader().stat() else {
                        return sample;
                    };
                    sample.cycles = stat.count;
                    for (sibling, field) in stat.siblings.iter().zip(order.iter()) {
                        sample.set(field.0, sibling.count);
                    }
                    sample.scaled = multiplexed(stat.time_enabled, stat.time_running);
                    sample
                }
                Self::Independent {
                    denominator,
                    counters,
                } => {
                    let mut sample = PmuSample {
                        cycle_source: CycleSource::TaskClockNanoseconds,
                        ..PmuSample::zero()
                    };
                    if let Some(handle) = denominator.as_mut()
                        && let Ok(stat) = handle.stat()
                    {
                        sample.cycles = stat.count;
                        // The denominator multiplexed, so every count beside it
                        // is measured over a different fraction of the interval.
                        sample.scaled = multiplexed(stat.time_enabled, stat.time_running);
                    }
                    // `multiplexed` is OR-ed across every counter rather than
                    // set from one, because a single member that was scaled is
                    // enough to make a ratio over the set meaningless.
                    for (field, counter) in counters.iter_mut() {
                        if let Ok(stat) = counter.stat() {
                            sample.set(field.0, stat.count);
                            sample.scaled |= multiplexed(stat.time_enabled, stat.time_running);
                        }
                    }
                    // Independently opened does not by itself mean
                    // non-commensurable: if the kernel scheduled every counter
                    // for the whole interval, they all counted the same
                    // instructions and the ratios mean what they say. Only
                    // multiplexing breaks that, and the kernel's own
                    // `time_running` against `time_enabled` is what says so.
                    sample.comparable = !sample.scaled;
                    sample
                }
            }
        }

        /// What denominator an independent counter set ended up with.
        fn denominator_source(&self) -> CycleSource {
            match self {
                Self::Grouped { source, .. } => *source,
                Self::Independent { denominator, .. } => match denominator {
                    Some(_) => CycleSource::TaskClockNanoseconds,
                    None => CycleSource::None,
                },
            }
        }
    }

    pub(super) fn detect() -> Support<PmuSupport> {
        match Counters::open() {
            Some((counters, _)) => Support::Available(counters.support()),
            None => Support::Unavailable(reason()),
        }
    }

    /// Why the *hardware* counters are missing, which is a different question
    /// from why no counter at all could be opened.
    ///
    /// The probe asks for exactly what the real attempt asks for - user-mode
    /// only. Asking for kernel-mode too, as an earlier version did, returns
    /// `EACCES` on any machine with `perf_event_paranoid >= 2` and would have
    /// reported a permissions problem on a machine whose real problem is that
    /// it has no PMU at all.
    fn reason() -> Unavailable {
        let exclude = Priv {
            kernel: true,
            hv: true,
            ..Priv::default()
        };
        match Counter::new(
            Hardware::CpuCycle,
            (Proc::CURRENT, Cpu::ALL),
            Opts {
                exclude,
                ..Opts::default()
            },
        ) {
            // `ENOENT` means the kernel does not implement the event, which is
            // what a machine with no virtualised PMU does. `EACCES`/`EPERM` is
            // a permissions fact and an operator can change it.
            Err(error) => match error.raw_os_error() {
                Some(libc::EACCES | libc::EPERM) => Unavailable::PermissionDenied,
                _ => Unavailable::UnsupportedHardware,
            },
            Ok(_) => {
                // Cycles opened, so the refusal was about the group, which
                // means the PMU is already fully committed to something else.
                Unavailable::UnsupportedHardware
            }
        }
    }

    /// The per-event detail behind [`super::diagnose`].
    pub(super) fn diagnose() -> Vec<(String, String)> {
        let exclude = Priv {
            kernel: true,
            hv: true,
            ..Priv::default()
        };
        let opts = Opts {
            exclude,
            ..Opts::default()
        };
        let mut out = Vec::new();
        let mut record = |name: &str, result: Result<Counter, std::io::Error>| {
            out.push((
                name.to_string(),
                match result {
                    Ok(_) => "open".to_string(),
                    Err(error) => match error.raw_os_error() {
                        // `ENOENT` from `perf_event_open` means the kernel does
                        // not implement the event at all.
                        Some(libc::ENOENT) => "not-implemented-by-this-kernel-or-cpu".to_string(),
                        Some(libc::EACCES | libc::EPERM) => "permission-denied".to_string(),
                        Some(code) => format!("errno {code}"),
                        None => error.to_string(),
                    },
                },
            ));
        };
        for (name, event) in [
            ("cycles", Hardware::CpuCycle),
            ("instructions", Hardware::Instr),
            ("cache-references", Hardware::CacheAccess),
            ("cache-misses", Hardware::CacheMiss),
            ("branches", Hardware::BranchInstr),
            ("branch-misses", Hardware::BranchMiss),
            ("frontend-stalled-cycles", Hardware::FrontendStalledCycle),
        ] {
            record(
                name,
                Counter::new(event, (Proc::CURRENT, Cpu::ALL), &opts).inspect(|counter| {
                    let _ = counter.enable();
                }),
            );
        }
        for (name, event) in [
            ("task-clock", Software::TaskClock),
            ("minor-page-faults", Software::MinorPageFault),
            ("context-switches", Software::CtxSwitch),
            ("migrations", Software::CpuMigration),
        ] {
            record(
                name,
                Counter::new(event, (Proc::CURRENT, Cpu::ALL), &opts).inspect(|counter| {
                    let _ = counter.enable();
                }),
            );
        }
        out
    }
}

#[cfg(not(all(feature = "pmu", any(target_os = "linux", target_os = "android"))))]
mod backend {
    use crate::capability::{PmuSupport, Support, Unavailable};

    use super::{CycleSource, PmuSample};

    /// Never constructed: with no backend there is no counter set to hold.
    ///
    /// `support` and `read` exist only so the call sites in [`PmuCounters`] type
    /// check on this branch. They are unreachable: `open` returns `None`, so
    /// `attach` takes the empty arm and no `Counters` value is ever created. Their
    /// bodies are therefore the conservative answer - an empty support set and a
    /// zeroed sample - rather than an invented one, because a support set is a
    /// promise about what can be measured and nothing here can measure anything.
    pub(super) struct Counters;

    impl Counters {
        pub(super) fn open() -> Option<(Self, CycleSource)> {
            None
        }

        pub(super) fn support(&self) -> PmuSupport {
            let _ = self;
            PmuSupport::default()
        }

        pub(super) fn read(&mut self) -> PmuSample {
            let _ = self;
            PmuSample::zero()
        }
    }

    pub(super) fn detect() -> Support<PmuSupport> {
        if cfg!(not(any(target_os = "linux", target_os = "android"))) {
            // No `perf_event_open` exists here at all. Windows exposes
            // performance counters through ETW and the performance data helper
            // library, which are process-wide, sampling-based, and not
            // per-thread hardware counters; there is nothing to map onto this
            // interface, and pretending otherwise would produce numbers that do
            // not mean what the ratios assume.
            Support::Unavailable(Unavailable::UnsupportedPlatform)
        } else {
            Support::Unavailable(Unavailable::NotCompiledIn)
        }
    }

    pub(super) fn diagnose() -> Vec<(String, String)> {
        // One line saying the platform has no such interface, rather than an
        // empty report: a caller that prints the diagnostic should not have to
        // special-case an empty vector to explain itself.
        vec![("all".to_string(), "unsupported-platform".to_string())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample(cycles: u64, instructions: u64, misses: u64, references: u64) -> PmuSample {
        PmuSample {
            cycles,
            cycle_source: CycleSource::Hardware,
            instructions,
            cache_references: references,
            cache_misses: misses,
            stalled_cycles: cycles / 2,
            branches: 1000,
            branch_misses: 10,
            page_faults: 4,
            context_switches: 2,
            migrations: 0,
            scaled: false,
            comparable: true,
        }
    }

    /// A sample is only *evidence* when the kernel scheduled its counters
    /// together and something was counted. A multiplexed group is still worth
    /// something - it is a scaled estimate, not nothing - but it is never full
    /// trust, and a non-comparable set or an idle counter set must read as
    /// carrying nothing at all, or a controller would act on the absence of a
    /// measurement.
    #[test]
    fn only_a_measured_commensurable_sample_carries_full_trust() {
        let measured = sample(1000, 500, 10, 100);
        assert!(
            measured.trust() > 0.99,
            "a measured sample is fully trusted"
        );
        assert!(!measured.signals().is_neutral());

        // Multiplexed: a real but scaled estimate.
        let scaled = PmuSample {
            scaled: true,
            ..measured
        };
        assert!(scaled.trust() > 0.0, "a scaled count is still evidence");
        assert!(scaled.trust() < measured.trust(), "but not full trust");

        for no_evidence in [
            // Independently scheduled, so nothing may be divided by anything.
            PmuSample {
                comparable: false,
                ..measured
            },
            // Scheduled but idle: the state every counter set is in at open.
            PmuSample::zero(),
        ] {
            assert!(
                no_evidence.trust().abs() < f64::EPSILON,
                "{no_evidence} must not trust"
            );
            assert!(
                no_evidence.signals().is_neutral(),
                "{no_evidence} must read as neutral"
            );
        }
    }

    /// A non-comparable sample withholds every ratio rather than reporting a
    /// number computed from counters that never counted the same
    /// instructions. The absolute counts survive, because a per-thread
    /// page-fault counter does not need a second counter to be true.
    #[test]
    fn a_non_comparable_sample_withholds_every_ratio() {
        let mut s = PmuSample {
            page_faults: 1000,
            context_switches: 5,
            migrations: 2,
            instructions: 4_000_000,
            cache_references: 900,
            cache_misses: 800,
            branches: 700,
            branch_misses: 600,
            stalled_cycles: 900_000,
            cycles: 1_000_000,
            cycle_source: CycleSource::Hardware,
            comparable: false,
            scaled: false,
        };
        assert!(s.ipc().is_none());
        assert!(s.instructions_per_ns().is_none());
        assert!(s.stall_fraction().abs() < f64::EPSILON);
        assert!(s.cache_miss_ratio().abs() < f64::EPSILON);
        assert!(s.branch_miss_ratio().abs() < f64::EPSILON);
        assert!(!s.memory_bound());
        assert_eq!(s.page_faults, 1000);
        assert!(s.placement_unstable());

        // Forming a group restores the ratios.
        s.comparable = true;
        assert!(s.ipc().is_some());
    }

    /// A migration is unambiguous evidence that placement is not holding: the
    /// thread was moved between cores, so the cache the plan chose is not the
    /// one the worker has. A context switch is not evidence - a thread that
    /// simply was not the one running has not been misplaced.
    #[test]
    fn a_migration_is_unambiguous_and_a_context_switch_is_not() {
        let mut switched = sample(1000, 500, 0, 0);
        switched.context_switches = 50;
        assert!(!switched.placement_unstable());
        switched.migrations = 1;
        assert!(switched.placement_unstable());
    }

    /// A denominator that is not hardware cycles cannot support a cycle ratio.
    /// Instructions per nanosecond moves with the core's frequency, so
    /// reporting it as IPC would make a frequency change look like a change in
    /// the workload - and a stall count with no cycle denominator has no
    /// fraction to be a fraction of.
    #[test]
    fn a_task_clock_denominator_yields_no_cycle_ratio() {
        let software = PmuSample {
            cycles: 1_000_000,
            cycle_source: CycleSource::TaskClockNanoseconds,
            instructions: 2_000_000,
            stalled_cycles: 500_000,
            comparable: true,
            ..PmuSample::zero()
        };
        assert_eq!(software.ipc(), None, "no hardware cycles, no IPC");
        assert!((software.stall_fraction()).abs() <= f64::EPSILON);
        assert!(!software.memory_bound());
        // The absolute counts and the real denominator are unaffected.
        assert_eq!(software.instructions, 2_000_000);
        assert_eq!(software.cycles, 1_000_000);
        assert!((software.instructions_per_ns().expect("per-ns is defined") - 2.0).abs() < 1e-9);
    }

    /// A multiplexed or reset counter set must be visible, not smoothed over.
    /// A counter that stopped incrementing makes an estimate worse; hiding it
    /// makes the estimate a lie.
    #[test]
    fn a_scaled_or_reset_counter_is_reported_not_hidden() {
        let exact = sample(100, 100, 10, 100);
        let scaled = PmuSample {
            scaled: true,
            ..exact
        };
        assert!(scaled.signals().confidence < exact.signals().confidence);

        let earlier = sample(1000, 500, 10, 100);
        let restarted = sample(10, 5, 0, 10);
        let delta = restarted.since(&earlier);
        assert_eq!(delta.ordering, Ordering::Relaxed);
        assert_eq!(delta.cycles, 0, "a lower bound, not a negative");
        assert_eq!(
            sample(2000, 900, 20, 200).since(&earlier).ordering,
            Ordering::Monotonic
        );
    }

    /// An unattached counter set must still answer: `None` from every ratio,
    /// not a division by zero and not a panic. This is the state a machine
    /// with `perf_event_paranoid = 4` is permanently in.
    #[test]
    fn an_unattached_counter_set_reads_as_zero_rather_than_failing() {
        let mut counters = PmuCounters::attach();
        if counters.is_attached() {
            return;
        }
        assert_eq!(counters.support().iter().count(), 0);
        assert_eq!(counters.cycle_source(), CycleSource::None);
        let sample = counters.sample();
        assert_eq!(sample, PmuSample::zero());
        assert!(!sample.memory_bound());
    }

    /// Whatever the kernel grants, the set must report a denominator and the
    /// support set must name the one it actually got. Discarding the task
    /// clock the independent fallback had just opened made a working machine
    /// read as "no PMU", which is a different fault with a different fix.
    #[test]
    fn an_attached_set_keeps_and_names_its_denominator() {
        let mut counters = PmuCounters::attach();
        if !counters.is_attached() {
            return;
        }
        match counters.cycle_source() {
            CycleSource::Hardware => assert!(
                counters.support().has(PmuCounter::Cycles),
                "hardware cycles are named"
            ),
            CycleSource::TaskClockNanoseconds => assert!(
                counters.support().has(PmuCounter::TaskClock),
                "the task clock is named, not passed off as cycles"
            ),
            CycleSource::None => panic!("an attached set never has no denominator"),
        }
        // The source is a property of the set, not of the interval, and the
        // denominator only moves forward: a source that never advances is a
        // counter the kernel opened and then never scheduled.
        let first = counters.sample();
        let second = counters.sample();
        assert_eq!(first.cycle_source, counters.cycle_source());
        assert_eq!(first.cycle_source, second.cycle_source);
        assert!(second.cycles >= first.cycles);
        // Comparability and multiplexing are the same fact: a sample claims
        // commensurability exactly when the kernel did not scale it.
        assert_eq!(first.comparable(), !first.scaled);
    }

    /// Ratios are clamped to what the hardware can physically report, so a
    /// counter reading above its own denominator produces a bounded number
    /// rather than an impossible one.
    #[test]
    fn ratios_are_clamped_to_what_the_hardware_can_see() {
        let impossible = PmuSample {
            cycles: 1,
            cycle_source: CycleSource::Hardware,
            instructions: 10_000,
            cache_references: 1,
            cache_misses: 10,
            stalled_cycles: 1_000,
            branches: 1,
            branch_misses: 1_000,
            ..PmuSample::zero()
        };
        assert!(impossible.ipc().expect("cycles present") <= 8.0);
        assert!(impossible.cache_miss_ratio() <= 1.0);
        assert!(impossible.stall_fraction() <= 1.0);
        assert!(impossible.branch_miss_ratio() <= 1.0);
    }

    /// A memory-bound reading needs both signals: stalls alone are a
    /// dependency chain, and misses alone are a cold but small working set.
    #[test]
    fn memory_bound_needs_stalls_and_misses_together() {
        let mut bound = sample(1000, 500, 800, 1000);
        bound.stalled_cycles = 800;
        assert!(bound.memory_bound());
        bound.cache_misses = 10;
        assert!(!bound.memory_bound(), "stalls alone are a dependency chain");
    }
}
