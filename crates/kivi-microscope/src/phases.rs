//! Request-path phase decomposition.
//!
//! The program asks where a request's time goes: socket receive, parse,
//! classification, hash, route, probe, value access, execution, reply, submit,
//! completion. This module measures that decomposition directly, per phase, on a
//! real request path, and reports it as a table.
//!
//! # Cost, and the three rules that follow from it
//!
//! A [`read`](tsc::read) costs about 23 TSC ticks on this host - roughly 6 ns, or
//! about 3% of a 180 ns `GET`. Timing each phase with two reads would therefore
//! add tens of percent to the path being measured. So:
//!
//! 1. **Boundaries are adjacent, not overlapping.** A phase is the delta from the
//!    previous boundary, so a span is timed once, not twice, and each phase
//!    carries exactly one read's cost.
//! 2. **That one read's cost is subtracted, not ignored.**
//!    [`Distribution::corrected_median`] removes it, because a phase measured at
//!    16 ns when the clock costs 6 ns is a 10 ns phase with 6 ns of instrument
//!    in it, and reporting the 16 is overstating the work by 60%. A phase whose
//!    corrected median is zero is below this program's resolution, and
//!    `corrected_median` says so rather than returning a small positive number
//!    that reads like a measurement.
//! 3. **Instrumented and uninstrumented totals are both reported.**
//!    [`Report::overhead`] is the difference between them, and a decomposition
//!    whose probes added a large fraction of the real cost is one whose absolute
//!    numbers should not be quoted even after correction.
//!
//! # Reading a decomposition
//!
//! Phase times are medians, not means: one page fault in a thousand samples moves
//! a mean and leaves a median alone. p90 is reported because a phase whose p90 is
//! 3× its median is a jitter phase, and averaging it with the rest hides that.
//! A phase's median and the marginal cost of *adding* that stage to a prefix
//! should agree; where they do not, the boundaries are misplaced and the table is
//! describing something other than what it claims.

use std::cell::RefCell;
use std::time::Duration;

use crate::tsc;

/// One stage of a request path.
///
/// A closed set, in the order a request passes through it, so a table built from
/// it reads top to bottom the way the request did. Adding a stage means adding a
/// variant here, where a reader will see it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(usize)]
pub enum Phase {
    /// Waiting for, and completing, the socket read that delivered the request.
    SocketReceive,
    /// Turning received bytes into a command frame.
    Parse,
    /// Deciding what the command is, and turning it into a semantic request.
    ///
    /// One phase, not two: classification and the construction of the request
    /// that follows it are not separable on a path that owns the request's
    /// representation, and a boundary between them would report the same work
    /// under two names.
    Classify,
    /// Hashing the key.
    Hash,
    /// Resolving the hash to an owner and a local slot.
    Route,
    /// Probing the physical table.
    Probe,
    /// Reading the value out of its entry.
    ValueAccess,
    /// Executing the operation's semantics.
    Execute,
    /// Encoding the reply.
    ReplyFormat,
    /// Handing the reply to the kernel.
    Submit,
    /// Deciding whether the request can run inline, and preparing durable state.
    Prepare,
    /// Building the canonical mutation.
    Mutation,
    /// Framing the record for the log.
    WalFrame,
    /// Queueing the record for the device.
    DurabilityQueue,
    /// The physical write.
    PhysicalWrite,
    /// The durability barrier.
    Barrier,
    /// Applying the prepared mutation to state.
    Apply,
    /// Cost not attributed to any phase above.
    Unattributed,
}

impl Phase {
    /// Every phase, in path order.
    pub const ALL: [Phase; Phase::COUNT] = [
        Phase::SocketReceive,
        Phase::Parse,
        Phase::Classify,
        Phase::Hash,
        Phase::Route,
        Phase::Probe,
        Phase::ValueAccess,
        Phase::Execute,
        Phase::ReplyFormat,
        Phase::Submit,
        Phase::Prepare,
        Phase::Mutation,
        Phase::WalFrame,
        Phase::DurabilityQueue,
        Phase::PhysicalWrite,
        Phase::Barrier,
        Phase::Apply,
        Phase::Unattributed,
    ];

    /// The phases that only a mutating request can cross.
    ///
    /// A named list rather than an inline `matches!`, because it has to be used in
    /// two places that must agree: [`Phase::belongs_to`], which decides whether a
    /// row appears in a read's table, and the tests that check it. An earlier
    /// version inlined the pattern and restated it in a test, and the two drifted -
    /// `Mutation` was excluded from a read's report by the first and asserted to
    /// belong there by the second.
    pub const WRITE_ONLY: [Phase; 7] = [
        Phase::Prepare,
        Phase::Mutation,
        Phase::WalFrame,
        Phase::DurabilityQueue,
        Phase::PhysicalWrite,
        Phase::Barrier,
        Phase::Apply,
    ];

    /// Number of phases.
    pub const COUNT: usize = Phase::Unattributed as usize + 1;

    /// Index into per-phase arrays.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Short name for a report row.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SocketReceive => "socket-receive",
            Self::Parse => "parse",
            Self::Classify => "classify+build",
            Self::Hash => "hash",
            Self::Route => "route",
            Self::Probe => "probe",
            Self::ValueAccess => "value-access",
            Self::Execute => "execute",
            Self::ReplyFormat => "reply-format",
            Self::Submit => "submit",
            Self::Prepare => "prepare",
            Self::Mutation => "mutation",
            Self::WalFrame => "wal-frame",
            Self::DurabilityQueue => "durability-queue",
            Self::PhysicalWrite => "physical-write",
            Self::Barrier => "barrier",
            Self::Apply => "apply",
            Self::Unattributed => "unattributed",
        }
    }

    /// Whether this phase belongs to `path`.
    ///
    /// A read crosses no write-only phase. A report that shows one is describing
    /// two different requests averaged together, which is worse than describing
    /// neither.
    #[must_use]
    pub const fn belongs_to(self, path: Path) -> bool {
        match path {
            Path::Read => !Self::is_write_only(self),
            Path::Write => true,
        }
    }

    /// Whether this phase can only appear on a mutating request.
    #[must_use]
    pub const fn is_write_only(self) -> bool {
        let mut index = 0;
        while index < Phase::WRITE_ONLY.len() {
            if Phase::WRITE_ONLY[index] as u8 == self as u8 {
                return true;
            }
            index += 1;
        }
        false
    }
}

/// Which request path a decomposition describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Path {
    /// A read: nothing before execution is a mutation.
    Read,
    /// A write, including whatever durability it selected.
    Write,
}

/// Per-phase sample statistics, in ticks.
#[derive(Debug, Clone, Default)]
struct Stats {
    /// Every sample, unsorted.
    samples: Vec<u64>,
    /// Samples dropped because the previous boundary read after this one, which
    /// would make the delta meaningless.
    inversions: u64,
}

/// The distribution of one phase's cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Distribution {
    /// Median ticks, including the clock read that closed the phase.
    pub median: f64,
    /// 90th-percentile ticks, including that read.
    pub p90: f64,
    /// Largest sample in ticks.
    pub max: f64,
    /// Samples that contributed.
    pub count: u64,
    /// Samples discarded as out-of-order timestamp pairs.
    pub inversions: u64,
}

impl Distribution {
    /// A distribution with no samples.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            median: 0.0,
            p90: 0.0,
            max: 0.0,
            count: 0,
            inversions: 0,
        }
    }

    /// Median ticks with the closing clock read removed, never below zero.
    ///
    /// Each phase is closed by exactly one `rdtsc`, and that read is inside the
    /// measured interval. Subtracting it is the difference between reporting a
    /// phase's cost and reporting the cost of measuring it.
    #[must_use]
    pub fn corrected_median(&self) -> f64 {
        (self.median - clock_cost() as f64).max(0.0)
    }

    /// Whether the corrected median is indistinguishable from zero at this
    /// program's resolution.
    ///
    /// True for a phase that costs less than the clock read that closed it. Such
    /// a phase is not measured, and a caller that treats its corrected median as a
    /// small positive number is reporting the instrument.
    #[must_use]
    pub fn below_resolution(&self) -> bool {
        self.count > 0 && self.corrected_median() == 0.0
    }
}

/// Ticks one [`tsc::read`] costs, measured once and cached.
///
/// Measured rather than assumed, because the correction above is only valid if the
/// instrument's cost is known, and the instrument's cost is a property of the CPU
/// rather than of this code.
fn clock_cost() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COST: AtomicU64 = AtomicU64::new(0);
    let cached = COST.load(Ordering::Relaxed);
    if cached != 0 {
        return cached;
    }
    let measured = tsc::rdtsc_cost();
    COST.store(measured, Ordering::Relaxed);
    measured
}

/// A request path's time decomposition.
#[derive(Debug, Clone)]
pub struct Report {
    /// Which path this describes.
    pub path: Path,
    /// Per-phase distribution, indexed by [`Phase::index`].
    pub phases: [Distribution; Phase::COUNT],
    /// Requests sampled.
    pub requests: u64,
    /// Ticks between the first and last boundary, summed over requests, with no
    /// probe attached.
    pub instrumented_ticks: u64,
    /// Ticks for the same work with no probes attached.
    pub baseline_ticks: u64,
    /// The cycle ratio these numbers convert with, if one was measured.
    pub ratio: Option<tsc::Ratio>,
}

impl Report {
    /// An empty report for `path`.
    #[must_use]
    pub fn empty(path: Path) -> Self {
        Self {
            path,
            phases: [Distribution::empty(); Phase::COUNT],
            requests: 0,
            instrumented_ticks: 0,
            baseline_ticks: 0,
            ratio: None,
        }
    }

    /// Percentage the probes added, or `None` without a baseline.
    ///
    /// A decomposition is worth believing when this is small. Above about 25% the
    /// absolute numbers are the probes as much as the work, and
    /// [`Distribution::corrected_median`] becomes the only figure to quote.
    #[must_use]
    pub fn overhead(&self) -> Option<f64> {
        if self.baseline_ticks == 0 {
            return None;
        }
        Some(
            (self.instrumented_ticks as f64 - self.baseline_ticks as f64)
                / self.baseline_ticks as f64
                * 100.0,
        )
    }

    /// Instrumented ticks per request.
    #[must_use]
    pub fn instrumented_ticks_per_request(&self) -> f64 {
        self.instrumented_ticks as f64 / self.requests.max(1) as f64
    }

    /// Uninstrumented baseline ticks per request.
    #[must_use]
    pub fn baseline_ticks_per_request(&self) -> f64 {
        self.baseline_ticks as f64 / self.requests.max(1) as f64
    }

    /// `phase`'s share of the instrumented per-request total, or `None` when
    /// either is zero.
    ///
    /// Divided by the *per-request* total, not the whole-run total. A share of a
    /// cumulative figure is not a share of a request, and reporting 0.0% for
    /// every phase because the denominator was 200,000 requests too large is the
    /// kind of plausible-looking zero that hides an instrument bug.
    #[must_use]
    pub fn share(&self, phase: Phase) -> Option<f64> {
        let per_request = self.instrumented_ticks_per_request();
        if per_request == 0.0 {
            return None;
        }
        Some(self.phases[phase.index()].median / per_request)
    }

    /// Renders the report as a markdown table, in nanoseconds.
    ///
    /// Reports both the raw median and the instrument-corrected one, because they
    /// answer different questions: the raw median is what the instrumented build
    /// actually did, and the corrected one is what the phase cost.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        use std::fmt::Write as _;

        let per_tick = nanos_per_tick();
        let clock_ns = clock_cost() as f64 * per_tick;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "  one clock read: {clock_ns:.1} ns, subtracted from every phase\n"
        );
        let _ = writeln!(
            out,
            "| phase | ns measured | ns corrected | share | p90 ns | samples |"
        );
        let _ = writeln!(out, "| --- | ---: | ---: | ---: | ---: | ---: |");
        for phase in Phase::ALL {
            if !phase.belongs_to(self.path) {
                continue;
            }
            let distribution = self.phases[phase.index()];
            if distribution.count == 0 {
                continue;
            }
            let share = self.share(phase).unwrap_or(0.0);
            let corrected = distribution.corrected_median();
            let corrected_cell = if distribution.below_resolution() {
                "< floor".to_owned()
            } else {
                format!("{:.1}", corrected * per_tick)
            };
            let _ = writeln!(
                out,
                "| {} | {:.1} | {corrected_cell} | {:.1}% | {:.1} | {} |",
                phase.label(),
                distribution.median * per_tick,
                share * 100.0,
                distribution.p90 * per_tick,
                distribution.count,
            );
        }
        let _ = write!(
            out,
            "\n  instrumented {:.1} ns/request, baseline {:.1} ns/request",
            self.instrumented_ticks_per_request() * per_tick,
            self.baseline_ticks_per_request() * per_tick,
        );
        match self.overhead() {
            Some(overhead) => {
                let _ = write!(out, ", probe overhead {overhead:+.1}%");
            }
            None => out.push_str(", no baseline captured"),
        }
        out
    }
}

/// Collects phase timings for the current thread.
///
/// Per-thread because the request path being instrumented is per-thread by
/// construction - one worker, one connection, one request at a time. A shared
/// probe would add the synchronisation it exists to measure.
#[derive(Debug)]
pub struct Probe {
    path: Path,
    stats: RefCell<[Stats; Phase::COUNT]>,
    requests: std::cell::Cell<u64>,
    instrumented: std::cell::Cell<u64>,
    baseline: std::cell::Cell<u64>,
    previous: std::cell::Cell<u64>,
}

impl Probe {
    /// A probe for `path` on this thread.
    #[must_use]
    pub fn new(path: Path) -> Self {
        Self {
            path,
            stats: RefCell::new(std::array::from_fn(|_| Stats::default())),
            requests: std::cell::Cell::new(0),
            instrumented: std::cell::Cell::new(0),
            baseline: std::cell::Cell::new(0),
            previous: std::cell::Cell::new(0),
        }
    }

    /// Closes the previous boundary and attributes the interval to `phase`.
    ///
    /// The first call after [`Probe::begin`] attributes nothing, because there is
    /// no previous boundary yet. A timestamp that is not greater than its
    /// predecessor is discarded rather than wrapped: a negative delta means the two
    /// reads raced, and averaging in a fabricated value would understate every
    /// phase.
    #[inline]
    pub fn mark(&self, phase: Phase) {
        let now = tsc::read();
        let previous = self.previous.replace(now);
        if now < previous {
            let mut stats = self.stats.borrow_mut();
            stats[phase.index()].inversions += 1;
            return;
        }
        self.instrumented
            .set(self.instrumented.get() + now - previous);
        let mut stats = self.stats.borrow_mut();
        let entry = &mut stats[phase.index()];
        if entry.samples.len() < MAX_SAMPLES {
            entry.samples.push(now - previous);
        }
    }

    /// Starts a request. The next [`Probe::mark`] closes it.
    #[inline]
    pub fn begin(&self) {
        self.requests.set(self.requests.get() + 1);
        self.previous.set(tsc::read());
    }

    /// Records uninstrumented ticks for the same work, measured with no probes
    /// attached.
    ///
    /// This is what makes a decomposition credible: the instrumented total and
    /// this baseline measure the same work, so their difference is the probes. A
    /// caller that never records one gets a decomposition it cannot check, which
    /// is why [`Report::overhead`] returns `None` rather than a flattering zero.
    #[inline]
    pub fn baseline(&self, ticks: u64) {
        self.baseline.set(self.baseline.get().saturating_add(ticks));
    }

    /// Reads the accumulated decomposition, clearing it.
    #[must_use]
    pub fn take(&self) -> Report {
        let mut stats = self.stats.borrow_mut();
        let mut phases = [Distribution::empty(); Phase::COUNT];
        for (index, entry) in stats.iter_mut().enumerate() {
            if entry.samples.is_empty() {
                phases[index].inversions = entry.inversions;
                continue;
            }
            entry.samples.sort_unstable();
            phases[index] = Distribution {
                median: percentile(&entry.samples, 50),
                p90: percentile(&entry.samples, 90),
                max: *entry.samples.last().unwrap_or(&0) as f64,
                count: entry.samples.len() as u64,
                inversions: entry.inversions,
            };
            entry.samples.clear();
            entry.inversions = 0;
        }
        Report {
            path: self.path,
            phases,
            requests: self.requests.replace(0),
            // The sum of boundary deltas: what the boundaries actually observed.
            // The per-phase medians are reported separately and deliberately not
            // summed into it, because medians of different phases do not add to
            // the median of their sum and pretending otherwise produces a table
            // whose rows do not reconcile with its total.
            instrumented_ticks: self.instrumented.replace(0),
            baseline_ticks: self.baseline.replace(0),
            ratio: CACHED_RATIO.get(),
        }
    }
}

/// Upper bound on retained samples per phase, so a long run cannot grow without
/// limit. Past it, later samples are dropped; the reported count then
/// under-describes the run and the medians describe the first `MAX_SAMPLES`.
const MAX_SAMPLES: usize = 100_000;

/// Nearest-rank percentile of a sorted slice.
fn percentile(sorted: &[u64], percentile: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (percentile * sorted.len()).div_ceil(100);
    let index = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[index] as f64
}

/// A zero-cost phase marker.
///
/// With no probe attached, [`Span::mark`] is a null-pointer test and nothing
/// else, which is what lets instrumented code live on the hot path permanently
/// while an authoritative run pays nothing.
#[derive(Debug, Clone, Copy)]
pub struct Span {
    probe: Option<core::ptr::NonNull<Probe>>,
}

impl Span {
    /// Starts a span on `probe`.
    #[inline]
    #[must_use]
    pub fn new(probe: &Probe) -> Self {
        Self {
            probe: Some(core::ptr::NonNull::from(probe)),
        }
    }

    /// A span that records nothing.
    #[inline]
    #[must_use]
    pub const fn disabled() -> Self {
        Self { probe: None }
    }

    /// Closes the previous boundary and attributes the interval to `phase`.
    #[inline]
    pub fn mark(&self, phase: Phase) {
        if let Some(probe) = self.probe {
            // SAFETY: the `Probe` outlives the `Span`. A `Span` is a stack value
            // built from a `&Probe` in the same frame, and a `Probe` is never
            // moved while a span into it exists - `Span` is `Copy` and holds a
            // pointer rather than a borrow only so instrumented code can hold one
            // across a call.
            unsafe { probe.as_ref() }.mark(phase);
        }
    }
}

impl Default for Span {
    fn default() -> Self {
        Self::disabled()
    }
}

thread_local! {
    /// The last calibration's ratio, so a report can be converted to cycles
    /// without re-running calibration. Absent means the report stays in ticks,
    /// which is the honest default.
    static CACHED_RATIO: std::cell::Cell<Option<tsc::Ratio>> =
        const { std::cell::Cell::new(None) };
}

/// Records the cycle ratio for reports on this thread.
///
/// Called by [`tsc::calibrate`], the only thing that measures a ratio and
/// therefore the only thing allowed to supply one.
pub(crate) fn remember_ratio(ratio: tsc::Ratio) {
    let _ = CACHED_RATIO.try_with(|cached| cached.set(Some(ratio)));
}

/// Nanoseconds per TSC tick, or `0.0` before calibration.
///
/// The conversion every report and every marginal cost is scaled by, exposed so a
/// caller computing its own figures uses this one rather than dividing by a
/// nominal frequency it typed itself. Zero before calibration, so a figure
/// rendered early is obviously unusable rather than plausibly wrong.
#[must_use]
pub fn nanos_per_tick() -> f64 {
    match tsc::hz() {
        Some(hz) => 1e9 / hz as f64,
        None => 0.0,
    }
}

/// Converts a tick count to a [`Duration`], using the calibrated TSC rate.
#[must_use]
pub fn to_duration(ticks: u64) -> Duration {
    match tsc::hz() {
        Some(hz) => Duration::from_nanos(ticks.saturating_mul(1_000_000_000) / hz),
        None => Duration::ZERO,
    }
}

/// Times `body` and returns its ticks, attributing nothing to a phase.
///
/// The uninstrumented measurement: a caller runs the same work with and without
/// spans to learn what the spans cost.
#[must_use]
pub fn measure(body: impl FnOnce()) -> u64 {
    let start = tsc::read();
    body();
    tsc::read() - start
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_disabled_span_records_nothing() {
        let probe = Probe::new(Path::Read);
        let before = probe.requests.get();
        Span::disabled().mark(Phase::Parse);
        assert_eq!(probe.requests.get(), before);
    }

    #[test]
    fn marks_accumulate_per_phase() {
        let _ = tsc::calibrate();
        let probe = Probe::new(Path::Read);
        for _ in 0..100 {
            probe.begin();
            // A measurable interval, so the sample is not zero-length noise.
            let until = tsc::read() + tsc::ticks_for_nanos(1_000);
            while tsc::read() < until {}
            probe.mark(Phase::Parse);
        }
        let report = probe.take();
        assert_eq!(report.requests, 100);
        let parse = report.phases[Phase::Parse.index()];
        assert_eq!(parse.count, 100);
        assert!(
            parse.median > clock_cost() as f64,
            "a 1 us interval must survive the clock correction, got {}",
            parse.corrected_median()
        );
        assert!(report.instrumented_ticks > 0);
    }

    #[test]
    fn take_clears_so_a_second_report_is_independent() {
        let probe = Probe::new(Path::Read);
        probe.begin();
        probe.mark(Phase::Hash);
        let first = probe.take();
        probe.begin();
        probe.mark(Phase::Hash);
        let second = probe.take();
        assert_eq!(first.requests, 1);
        assert_eq!(second.requests, 1);
        assert_ne!(first.instrumented_ticks, 0);
    }
    /// A phase at or below the clock read's own cost is not a measurement. The
    /// correction clamps to zero rather than reporting the instrument's cost as
    /// the phase's, and a negative would be worse than either.
    #[test]
    fn a_phase_below_the_clock_cost_correction_clamps_to_zero() {
        for median in [1.0, clock_cost() as f64] {
            let distribution = Distribution {
                median,
                p90: median,
                max: median,
                count: 10,
                inversions: 0,
            };
            assert!(
                distribution.corrected_median().abs() < f64::EPSILON,
                "median {median} corrected to {}",
                distribution.corrected_median()
            );
        }
        let at_clock_cost = Distribution {
            median: clock_cost() as f64,
            p90: clock_cost() as f64,
            max: clock_cost() as f64,
            count: 10,
            inversions: 0,
        };
        assert!(at_clock_cost.below_resolution());
    }

    /// Shares are per request, not per run. A report of 100,000 requests whose
    /// phases each cost 100 ticks must show shares summing to about 1.0, and a
    /// denominator of the whole-run total would show 0.0% for every row.
    #[test]
    fn shares_are_a_fraction_of_one_request() {
        let mut report = Report::empty(Path::Read);
        report.requests = 100_000;
        report.instrumented_ticks = 100_000 * 1_000;
        report.phases[Phase::Parse.index()] = Distribution {
            median: 500.0,
            p90: 500.0,
            max: 500.0,
            count: 100_000,
            inversions: 0,
        };
        report.phases[Phase::Hash.index()] = Distribution {
            median: 250.0,
            p90: 250.0,
            max: 250.0,
            count: 100_000,
            inversions: 0,
        };
        let parse = report.share(Phase::Parse).expect("share");
        let hash = report.share(Phase::Hash).expect("share");
        assert!(
            (parse + hash - 0.75).abs() < 1e-9,
            "shares sum to {}",
            parse + hash
        );
    }

    #[test]
    fn overhead_is_none_without_a_baseline_and_a_percentage_with_one() {
        let mut report = Report::empty(Path::Read);
        assert!(report.overhead().is_none());
        report.requests = 100;
        report.instrumented_ticks = 1_200;
        report.baseline_ticks = 1_000;
        let overhead = report.overhead().expect("overhead");
        assert!((overhead - 20.0).abs() < 1e-9, "got {overhead}");
    }

    /// A read crosses no write-only phase, and the list that decides it is the same
    /// list this test reads - so a phase added to one and not the other fails here
    /// rather than producing a read report with a barrier row in it.
    #[test]
    fn a_read_crosses_no_write_only_phase() {
        for phase in Phase::ALL {
            assert_eq!(
                phase.belongs_to(Path::Read),
                !phase.is_write_only(),
                "{phase:?}: belongs_to and is_write_only disagree"
            );
        }
        for phase in Phase::WRITE_ONLY {
            assert!(
                !phase.belongs_to(Path::Read),
                "{phase:?} must not appear in a read"
            );
        }
        assert!(Report::empty(Path::Read).share(Phase::Barrier).is_none());
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let sorted = [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert!((percentile(&sorted, 50) - 5.0).abs() < 1e-9);
        assert!((percentile(&sorted, 90) - 9.0).abs() < 1e-9);
        assert!((percentile(&sorted, 100) - 10.0).abs() < 1e-9);
        assert!(percentile(&[], 50).abs() < 1e-9);
    }
}
