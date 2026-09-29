//! Timekeeping for measurement.
//!
//! # What this host can and cannot count
//!
//! WSL2 exposes no hardware PMU: `perf list hw` is empty and every
//! `cycles`/`instructions`/`cache-misses` event fails with "No supported events
//! found". Instruction, cache and branch counts therefore come from
//! [callgrind], not from a sampling counter, and this module supplies only the
//! half callgrind cannot: wall time.
//!
//! # The TSC is a clock, not a cycle counter
//!
//! `constant_tsc`, `nonstop_tsc` and `tsc_reliable` are all set, so
//! [`read`] is monotonic and comparable across threads and cores. It is *not*
//! the core clock. On this host a dependent `add` chain - exactly one core
//! cycle per iteration - measures 0.6993 TSC ticks per iteration, so the core
//! runs at about 1.43× the TSC frequency and the two move apart whenever
//! turbo or thermal state changes.
//!
//! Consequently:
//!
//! * [`read`] is the authoritative duration measurement. Report nanoseconds.
//! * [`Ratio`] converts TSC ticks to core cycles using a ratio measured on the
//!   same core, in the same run, bracketing the measurement. A cycle number
//!   without its ratio and its spread is not a measurement.
//! * Nothing in Kivi may divide wall time by the nominal TSC frequency and
//!   call the result cycles. [`calibrate`] exists to make that mistake hard to
//!   write by accident.
//!
//! [callgrind]: https://valgrind.org/docs/manual/cg-manual.html

use core::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Reads the time stamp counter.
///
/// `rdtsc` is not a serialising instruction: on the out-of-order cores here the
/// instructions after it can retire before it, and a following load can be
/// hoisted above it. That is fine for measuring a region that ends in
/// [`read`] (nothing needs to observe the region's memory afterwards) and
/// wrong for a region whose point is a side effect, which is what
/// [`read_and_fence`] is for.
///
/// # Safety
///
/// Caller must be on x86-64. On any other architecture this is a compile
/// error rather than a silent zero: a wrong clock is worse than no clock.
#[inline(always)]
// `inline(always)` is deliberate and is the one place in the workspace it is
// correct. A `rdtsc` that survives a call boundary measures the call: the
// measurement then contains the cost of the thing doing the measuring, which is
// the failure mode this whole crate exists to avoid.
#[allow(
    clippy::inline_always,
    reason = "an out-of-line clock read measures the call, not the region"
)]
#[must_use]
pub fn read() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `rdtsc` is unconditionally available on x86-64. It has no
        // memory operands, cannot fault, and touches no architectural state.
        unsafe { core::arch::x86_64::_rdtsc() }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        compile_error!(
            "kivi-microscope requires x86-64 rdtsc; no portable fallback is correct enough to measure with"
        )
    }
}

/// Reads the TSC and orders every earlier load and store before it.
///
/// `rdtscp` is partially serialising: loads issued after it do not pass it, so
/// the counter is read after the memory operations that preceded it are
/// complete. Stores are still only ordered by the implicit `lfence` the
/// instruction performs, which is what a store-observable boundary needs.
///
/// This costs roughly 2.5× [`read`] (measured: about 64 ticks against about
/// 24). It is for a handful of region boundaries, not a hot loop.
///
/// # Safety
///
/// Caller must be on x86-64.
#[allow(
    clippy::inline_always,
    reason = "an out-of-line clock read measures the call, not the region"
)]
#[must_use]
pub fn read_and_fence() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        let low: u32;
        let high: u32;
        let mut aux = 0u32;
        // SAFETY: `rdtscp` is unconditionally available on x86-64, has no
        // memory operands, and cannot fault. `aux` is a write-only output.
        unsafe {
            core::arch::asm!(
                "rdtscp",
                out("eax") low,
                out("edx") high,
                out("ecx") aux,
                options(nomem, nostack, preserves_flags),
            );
        }
        // `aux` names the CPU this read happened on. Deliberately dropped: a
        // measurement spans migrations, and reporting the last CPU would
        // misrepresent a whole-region number.
        core::hint::black_box(aux);
        (u64::from(high) << 32) | u64::from(low)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        compile_error!("kivi-microscope requires x86-64 rdtscp")
    }
}

/// Cached TSC frequency in ticks per second, or zero before [`calibrate`].
///
/// Set once by [`calibrate`]. Read through [`hz`] rather than touched
/// directly.
static HZ: AtomicU64 = AtomicU64::new(0);

/// Cached core-cycles-per-TSC-tick ratio, scaled by [`RATIO_SCALE`].
///
/// Set once by [`calibrate`].
static CYCLES_PER_TICK_Q16: AtomicU64 = AtomicU64::new(0);

/// Fixed-point scale for the cycle ratio. 2^16 keeps ~1.5e-5 of precision,
/// far below the run-to-run spread of a turbo-governed core.
const RATIO_SCALE: u64 = 1 << 16;

/// Ratio samples taken per calibration.
///
/// Four is a deliberate minimum, not a preference: fewer cannot separate a
/// stable ratio from a drifting one, and a drift larger than the spread is
/// exactly the error a cycle figure would hide.
const RATIO_SAMPLES: usize = 4;

/// Iterations per ratio sample.
///
/// Large enough that loop overhead cannot matter - the loop body is four
/// instructions, so its share is under 0.01% - and small enough that four
/// samples complete before the core's thermal state has moved appreciably.
const RATIO_ITERATIONS: u64 = 20_000_000;

/// Converts TSC ticks to nanoseconds without a floating-point multiply.
const NS_PER_SEC: u64 = 1_000_000_000;

/// TSC frequency in ticks per second, or `None` before [`calibrate`].
///
/// [`calibrate`]: fn@calibrate
#[must_use]
pub fn hz() -> Option<u64> {
    match HZ.load(Ordering::Relaxed) {
        0 => None,
        hz => Some(hz),
    }
}

/// Number of `nanos` nanoseconds, as TSC ticks.
///
/// # Panics
///
/// If [`calibrate`] has not run, because guessing the frequency is the exact
/// mistake this module exists to prevent.
#[must_use]
pub fn ticks_for_nanos(nanos: u64) -> u64 {
    let hz = hz().expect("kivi_microscope::tsc::hz: calibrate() must run first");
    (nanos.saturating_mul(hz)) / NS_PER_SEC
}

/// A measured core-cycles-per-TSC-tick ratio and how well it held.
///
/// Reporting a cycle count without the spread would hide exactly the
/// uncertainty that makes cycles-per-op untrustworthy on a turbo part: the
/// same binary can differ by tens of percent between a cold and a warmed core.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ratio {
    /// Core cycles per TSC tick, as f64.
    pub cycles_per_tick: f64,
    /// Half the spread across calibration samples, as a fraction of
    /// `cycles_per_tick`.
    pub half_spread: f64,
    /// Samples that produced this estimate.
    pub samples: u32,
}

impl Ratio {
    /// Converts TSC ticks to core cycles.
    ///
    /// The result is an estimate whose accuracy is bounded by `half_spread`,
    /// which [`Calibration::report`] prints next to it. Prefer
    /// [`Ratio::spread_cycles`] when the caller needs a bound rather than a
    /// point estimate.
    #[must_use]
    pub fn to_cycles(self, ticks: u64) -> f64 {
        ticks as f64 * self.cycles_per_tick
    }

    /// The cycle count's uncertainty, for `ticks` ticks.
    #[must_use]
    pub fn spread_cycles(self, ticks: u64) -> f64 {
        ticks as f64 * self.cycles_per_tick * self.half_spread
    }
}

/// A calibration result: the clock's frequency and the core's ratio to it.
#[derive(Debug, Clone, Copy)]
pub struct Calibration {
    /// TSC ticks per second, median of several samples.
    pub hz: u64,
    /// Core cycles per TSC tick.
    pub ratio: Ratio,
    /// Spread of the frequency samples, as a fraction of the median.
    ///
    /// Not zero, and not noise to be hidden: it is `thread::sleep` overshoot,
    /// which is the only imprecise input to the frequency. A nanosecond figure
    /// derived from `hz` carries this much uncertainty, so a duration quoted to
    /// better than `hz_spread_ppm` is quoting more precision than the clock's
    /// rate was measured to.
    pub hz_spread_ppm: f64,
}

impl Calibration {
    /// The frequency's uncertainty in parts per million.
    #[must_use]
    pub fn hz_spread_ppm(&self) -> f64 {
        self.hz_spread_ppm
    }
}

impl core::fmt::Display for Calibration {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.report())
    }
}

impl Calibration {
    /// One line describing the clock and the core's relationship to it.
    ///
    /// The ratio's spread travels with it because a cycle number without it is
    /// not reproducible.
    #[must_use]
    pub fn report(&self) -> String {
        format!(
            "tsc {:.6} GHz (±{:.0} ppm) | {:.4} core cycles/tsc tick (±{:.2}%, {} samples)",
            self.hz as f64 / 1e9,
            self.hz_spread_ppm,
            self.ratio.cycles_per_tick,
            self.ratio.half_spread * 100.0,
            self.ratio.samples,
        )
    }
}

/// Measures the TSC frequency against the monotonic clock and the core's cycle
/// rate against the TSC, then caches both.
///
/// The cycle rate comes from a dependent `add` chain in inline assembly: the
/// next add cannot issue until the previous one writes its result, so the loop
/// retires at exactly one core cycle per iteration and the instruction count is
/// known without a disassembler. A compiler-generated loop would not do - LLVM
/// folds `x += 1` in a counted loop to a constant, and an earlier version of
/// this probe measured exactly 0.0000 ticks per iteration for that reason.
///
/// # Cost
///
/// A few tens of milliseconds. Run it once per measurement, not per iteration.
#[must_use]
pub fn calibrate() -> Calibration {
    let (hz, hz_spread) = measure_hz();
    HZ.store(hz, Ordering::Relaxed);

    let mut samples = [0.0f64; RATIO_SAMPLES];
    for sample in &mut samples {
        *sample = measure_ratio(RATIO_ITERATIONS);
    }

    let cycles_per_tick = samples.iter().sum::<f64>() / samples.len() as f64;
    let max_dev = samples
        .iter()
        .map(|s| (s - cycles_per_tick).abs())
        .fold(0.0f64, f64::max);
    let ratio = Ratio {
        cycles_per_tick,
        half_spread: if cycles_per_tick > 0.0 {
            max_dev / cycles_per_tick
        } else {
            0.0
        },
        samples: u32::try_from(samples.len()).unwrap_or(u32::MAX),
    };
    // Fixed-point cache of the ratio. Truncation here is deliberate and
    // bounded: the cached value is a fast approximation for callers that only
    // need a rough cycle count, while any figure that gets reported goes
    // through `Calibration::ratio` and keeps full precision.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a Q16 cache of a ratio near 1.4 truncates below the reported precision"
    )]
    let cached = (ratio.cycles_per_tick * RATIO_SCALE as f64) as u64;
    CYCLES_PER_TICK_Q16.store(cached, Ordering::Relaxed);
    crate::phases::remember_ratio(ratio);
    Calibration {
        hz,
        ratio,
        hz_spread_ppm: hz_spread * 1e6,
    }
}

/// Ticks elapsed by a dependent-add chain of `iterations`, converted to core
/// cycles per tick.
fn measure_ratio(iterations: u64) -> f64 {
    // Warm the code path and the branch predictor so the sample measures the
    // loop and not its first execution.
    core::hint::black_box(dependent_adds(1_000_000));
    let start = read();
    core::hint::black_box(dependent_adds(iterations));
    let elapsed = read() - start;
    elapsed as f64 / iterations as f64
}

/// `iterations` dependent 64-bit adds, exactly one core cycle each.
#[inline(never)]
fn dependent_adds(iterations: u64) -> u64 {
    let mut accumulator: u64 = 0;
    let mut remaining = iterations;
    // SAFETY: the block contains only register operations on operands Rust
    // allocated, no memory access, and no branch that could leave `remaining`
    // unzeroed: the loop decrements before testing, so it exits only at zero.
    //
    // `inout` on `remaining` reads a value the loop overwrites, hence the
    // `unused_assignments` allowance. `inout` is what makes the register
    // value well-defined on entry; the loop leaves it at zero either way.
    #[allow(unused_assignments)]
    unsafe {
        core::arch::asm!(
            "2:",
            "add {accumulator}, 1",
            "sub {remaining}, 1",
            "jnz 2b",
            accumulator = inout(reg) accumulator,
            remaining = inout(reg) remaining,
            options(nomem, nostack),
        );
    }
    accumulator
}

/// Measures TSC ticks per second against [`Instant`].
///
/// The noise here is entirely `thread::sleep`, which overshoots by a
/// scheduler-dependent amount that varies between calls, and the TSC is
/// invariant. So one sample is not a frequency measurement - it is a frequency
/// measurement with a few parts per million of scheduler noise on it. Several
/// samples are taken and the median is returned, which rejects the occasional
/// large overshoot that a mean would drag the estimate toward.
///
/// A tighter bound is available for free: pairing the *largest* tick delta with
/// the *smallest* elapsed time gives the fastest consistent rate, and the
/// *smallest* tick delta with the *largest* elapsed time the slowest. The gap
/// between them is the uncertainty, and [`Calibration::hz_spread_ppm`] reports
/// it so a caller can see how much to trust a nanosecond figure derived from it.
fn measure_hz() -> (u64, f64) {
    const SLEEP: core::time::Duration = core::time::Duration::from_millis(100);
    const SAMPLES: usize = 5;

    let mut rates = [0.0f64; SAMPLES];
    for (index, rate) in rates.iter_mut().enumerate() {
        // Vary the window so a systematic effect in one length cannot pass for
        // the clock's rate.
        let window = SLEEP + core::time::Duration::from_millis(20 * index as u64);
        let start_instant = Instant::now();
        let start = read();
        std::thread::sleep(window);
        let end = read();
        let nanos = u64::try_from(start_instant.elapsed().as_nanos()).unwrap_or(u64::MAX);
        *rate = if nanos == 0 {
            0.0
        } else {
            (end - start) as f64 * NS_PER_SEC as f64 / nanos as f64
        };
    }

    let mut sorted = rates;
    sorted.sort_by(f64::total_cmp);
    let median = sorted[SAMPLES / 2];
    let spread = if median > 0.0 {
        (sorted[SAMPLES - 1] - sorted[0]) / median
    } else {
        0.0
    };
    // A clock rate is billions, far inside both `u64` and `f64`'s exact-integer
    // range, so the conversion is total. The bounds are checked rather than
    // assumed, and saturate rather than wrap if a measurement ever returned
    // something absurd.
    let rounded = median.round();
    let ceiling = u64::MAX as f64;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "rounded is provably within (0, u64::MAX] by the guards on either side"
    )]
    let hz = if rounded <= 0.0 {
        0
    } else if rounded >= ceiling {
        u64::MAX
    } else {
        rounded as u64
    };
    (hz, spread)
}

/// What this host's performance counters can do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareCounters {
    /// A PMU is present and usable.
    Available,
    /// `perf` runs but no hardware events are supported, which is what WSL2 and
    /// most virtual machines report. Instruction, cache and branch counts must
    /// come from callgrind.
    Unavailable,
    /// `perf` is not installed, so the question is unanswered.
    NoPerf,
}

impl core::fmt::Display for HardwareCounters {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Available => f.write_str("available"),
            Self::Unavailable => f.write_str("unavailable (use callgrind)"),
            Self::NoPerf => f.write_str("unknown (perf not installed)"),
        }
    }
}

/// Whether this host can count instructions, cache misses and branch
/// mispredicts in hardware.
///
/// Determined by trying a hardware event through `perf`, which is the only
/// portable way to ask: `perf_event_open` succeeding is not enough, because WSL2
/// and most virtual machines accept the syscall and then fail to program a
/// counter. A benchmark that assumed a PMU and did not check would report
/// nothing while looking broken.
#[must_use]
pub fn hardware_counters() -> HardwareCounters {
    match std::process::Command::new("perf")
        .args(["stat", "-e", "cycles", "--", "true"])
        .output()
    {
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("No supported events") {
                HardwareCounters::Unavailable
            } else {
                HardwareCounters::Available
            }
        }
        Err(_) => HardwareCounters::NoPerf,
    }
}

/// Ticks one [`read`] costs, measured as instruction throughput.
///
/// Reported because it bounds phase resolution: every
/// [`phases::Span::mark`](crate::phases::Span::mark) pays this, so a phase
/// shorter than a few reads cannot be measured. That number is what tells a
/// reader whether a reported sub-microsecond phase is a measurement or an
/// artifact of the clock reading itself.
#[must_use]
pub fn rdtsc_cost() -> u64 {
    const SAMPLES: u64 = 2_000_000;
    let start = read();
    let mut accumulator = 0u64;
    for _ in 0..SAMPLES {
        // Reads are independent, so this measures the instruction's throughput,
        // not its latency.
        accumulator = accumulator.wrapping_add(read());
    }
    let elapsed = read() - start;
    core::hint::black_box(accumulator);
    elapsed / SAMPLES
}

/// The machine's measurement capabilities, as JSON.
///
/// The first thing to run on a new host, before any benchmark, because every
/// later number's meaning depends on it.
#[must_use]
pub fn capabilities() -> String {
    let calibration = calibrate();
    format!(
        "{{\n  \"tsc_hz\": {},\n  \"hz_spread_ppm\": {:.1},\n  \
         \"core_cycles_per_tick\": {:.4},\n  \"cycle_spread_pct\": {:.2},\n  \
         \"rdtsc_ticks\": {},\n  \"hardware_counters\": \"{}\"\n}}",
        calibration.hz,
        calibration.hz_spread_ppm(),
        calibration.ratio.cycles_per_tick,
        calibration.ratio.half_spread * 100.0,
        rdtsc_cost(),
        hardware_counters(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The clock's own cost bounds what any phase measurement can resolve, and
    /// a value that is too large would make every sub-microsecond phase in the
    /// program an artifact.
    #[test]
    fn the_clock_read_is_cheap_enough_to_instrument_with() {
        let cost = rdtsc_cost();
        assert!(
            (5..=200).contains(&cost),
            "one TSC read costs {cost} ticks; a phase decomposition finer than \
             that is measuring the clock"
        );
    }

    /// The TSC must be usable before anything else is measured, so
    /// calibration has to work and has to be repeatable.
    #[test]
    fn calibration_is_stable_and_plausible() {
        let first = calibrate();
        assert!(
            (2_000_000_000..=5_000_000_000).contains(&first.hz),
            "implausible TSC frequency: {} Hz",
            first.hz
        );
        // A TSC slower than the core's cycle rate would make the ratio below 1,
        // which is impossible for a core that is not sleeping.
        assert!(
            (0.5..=4.0).contains(&first.ratio.cycles_per_tick),
            "implausible cycles/tick: {}",
            first.ratio.cycles_per_tick
        );
        // The TSC is invariant, so two calibrations must agree closely. They do
        // not agree *exactly*, and asserting they do would assert that
        // `thread::sleep` is exact. Each calibration measures the rate against a
        // sleep the OS overshoots by a varying amount, so the achievable bound
        // is the reported spread - a few tens of ppm. A test that asserted
        // bit-equality would fail on a loaded box and teach its reader to
        // re-run the benchmark instead of reading it, which is the opposite of
        // what a measurement crate is for.
        let second = calibrate();
        let drift_ppm = second.hz.abs_diff(first.hz) as f64 * 1e6 / first.hz as f64;
        assert!(
            drift_ppm < 1_000.0,
            "TSC frequency drifted {drift_ppm:.0} ppm between calibrations \
             ({:.6} then {:.6} GHz); an invariant clock should not",
            first.hz as f64 / 1e9,
            second.hz as f64 / 1e9
        );
    }

    /// The frequency estimate carries a stated uncertainty, and the statement
    /// must be small enough to be useful.
    ///
    /// A `sleep`-derived rate is precise to tens of ppm at best. If the reported
    /// spread were thousands, every nanosecond figure in the program would be
    /// quoting precision the clock was never measured to, and the honest
    /// response would be a different instrument - not a smaller number here.
    #[test]
    fn the_frequency_estimate_knows_its_own_uncertainty() {
        let calibration = calibrate();
        assert!(
            calibration.hz_spread_ppm() < 500.0,
            "frequency spread {} ppm is too wide to quote durations against",
            calibration.hz_spread_ppm()
        );
    }

    /// A dependent add chain is one cycle per iteration. If the probe measured
    /// zero, the compiler deleted the loop and every later number is a lie - and
    /// it did: a compiler-generated version of this loop folds to a constant and
    /// measures exactly 0.0.
    #[test]
    fn the_cycle_probe_is_not_optimized_away() {
        let ticks_per_iteration = measure_ratio(RATIO_ITERATIONS);
        assert!(
            (0.2..2.0).contains(&ticks_per_iteration),
            "a one-cycle chain measured {ticks_per_iteration} TSC ticks per iteration, which is not a clock"
        );
    }

    /// Monotonicity across a sleep is the property every duration measurement
    /// depends on, and the one a wall-clock bug breaks first.
    #[test]
    fn the_clock_is_monotonic() {
        let _ = calibrate();
        let mut previous = read();
        for _ in 0..10_000 {
            let now = read();
            assert!(now >= previous, "TSC went backwards: {previous} then {now}");
            previous = now;
        }
    }

    /// `ticks_for_nanos` is the conversion the whole crate leans on; an error
    /// in it silently rescales every reported duration.
    #[test]
    fn tick_conversion_round_trips() {
        let calibration = calibrate();
        for nanos in [1u64, 1_000, 1_000_000, 100_000_000] {
            let ticks = ticks_for_nanos(nanos);
            let back =
                u64::try_from(u128::from(ticks) * 1_000_000_000 / u128::from(calibration.hz))
                    .unwrap_or(u64::MAX);
            let error = back.abs_diff(nanos);
            assert!(
                error <= 1_000.max(nanos / 1_000),
                "{nanos} ns -> {ticks} ticks -> {back} ns, error {error}"
            );
        }
    }
}
