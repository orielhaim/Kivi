//! Prints what this host can measure.
//!
//! Run it before any benchmark on a new machine. Every number the performance
//! program produces is interpreted against these capabilities, and the two
//! answers that matter are:
//!
//! * **the TSC rate and the core's cycle rate relative to it** - because a TSC
//!   delta is nanoseconds, not cycles, and the two differ on this machine
//! * **whether hardware counters exist** - because on a host without a PMU the
//!   instruction, cache and branch counts the program requires come from
//!   callgrind instead, and a benchmark written as if they came from `perf`
//!   would produce nothing
//!
//! ```text
//! cargo run --release -p kivi-microscope --bin kivi-microscope-caps
//! ```

fn main() {
    println!("{}", kivi_microscope::tsc::capabilities());

    eprintln!();
    eprintln!("Reading this output:");
    eprintln!(
        "  tsc_hz and hz_spread_ppm   durations are quoted as nanoseconds; the\n  \
         spread is thread::sleep overshoot and is the precision a quoted\n  \
         duration can honestly claim."
    );
    eprintln!(
        "  core_cycles_per_tick       1.0 would mean the TSC *is* the core clock.\n  \
         Anything else means a cycle count needs this ratio, measured on the\n  \
         same core in the same run, before it can be reported."
    );
    eprintln!(
        "  hardware_counters          'unavailable' means callgrind is the only\n  \
         source of instructions, cache misses and branch mispredicts. It\n  \
         counts perfectly and distorts time completely, so the two are\n  \
         used together and never substituted for one another."
    );
    eprintln!(
        "  rdtsc_ticks                the floor on phase resolution. A phase\n  \
         shorter than a few reads cannot be measured, and a reported\n  \
         figure near this value is the instrument, not the work."
    );
}
