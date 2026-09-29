//! The architecture microscope: measurement instruments Kivi can trust.
//!
//! This crate exists because the performance program cannot proceed from
//! intuition. Every architectural claim it makes - "this hash is faster", "the
//! route lookup is the bottleneck", "the read path copies three times" - has to
//! be answerable by an instrument that reports what happened rather than what
//! was hoped for.
//!
//! # What this host can measure, and how
//!
//! The host exposes **no hardware PMU**: `perf list hw` is empty and every
//! `cycles`, `instructions` and `cache-misses` event fails. That rules out the
//! obvious instrument and forces a specific combination:
//!
//! | Question | Instrument | Notes |
//! | --- | --- | --- |
//! | how long did this take | [`tsc`] | `rdtsc`, 3.6864 GHz, invariant. Nanoseconds, not cycles - see below |
//! | how many core cycles | [`tsc::Ratio`] | TSC ÷ calibrated ratio. The ratio moves with turbo state and is reported with its spread |
//! | how many instructions | callgrind | `Ir` per function and per call site |
//! | how many cache misses | callgrind | `D1mr`, `DLmr`, `I1mr`, `ILmr`, with a simulated cache geometry |
//! | how many branch mispredicts | callgrind | `Bcm`, `Bim` |
//! | how many allocations | [`alloc`] | exact, unsampled, per thread |
//! | how many bytes copied | [`copies`] | exact, per named hop, per thread |
//! | where a request's cycles go | [`phases`] | per-phase medians with an instrumentation-overhead check |
//!
//! The division is deliberate: callgrind counts perfectly and distorts time
//! completely, the TSC measures time perfectly and counts nothing. Neither
//! alone answers an architecture question.
//!
//! # The TSC is a clock, not a cycle counter
//!
//! This is the single most important thing to know about measuring on this
//! machine, and it is easy to get wrong. `constant_tsc`, `nonstop_tsc` and
//! `tsc_reliable` are all set, so `rdtsc` is stable and comparable - but it runs
//! at 3.6864 GHz while the core runs at roughly 5.27 GHz. A dependent `add`
//! chain, which retires at exactly one core cycle per iteration, measures
//! 0.6993 TSC ticks per iteration.
//!
//! So:
//!
//! * `tsc::read()` deltas are authoritative **as nanoseconds**.
//! * Calling those "cycles" overstates the core's cycle count by about 43%.
//! * Converting needs [`tsc::calibrate`], run on the same core in the same run.
//!   A cycle figure without its ratio and its spread is not a measurement.
//!
//! # Authoritative runs pay nothing
//!
//! Every instrument here is opt-in at the binary level, because
//! `#[global_allocator]` is a static property and a `rdtsc` pair on the hot
//! path costs about 4% of a 160 ns `GET`:
//!
//! * [`alloc::Counting`] is instantiated only where counting is wanted.
//! * [`copies::Meter`] records nothing unless a [`copies::Scope`] is active.
//! * [`phases::Span::disabled`] is what instrumented code compiles to when no
//!   probe is attached, and [`phases::Report::overhead`] fails loudly if the
//!   probes are distorting the result beyond usefulness.
//!
//! # Example
//!
//! ```
//! use kivi_microscope::{alloc, copies, phases, tsc};
//!
//! // Calibrate once per measurement run: nanoseconds need the rate, cycles
//! // need the ratio, and a wrong rate silently rescales everything.
//! let calibration = tsc::calibrate();
//! println!("{calibration}");
//!
//! // Exact allocation accounting for one shape of work. The scope is per thread
//! // and uses its own counters, so it observes the region rather than the process.
//! //
//! // This reports zero, and that is the point: a scope counts what the *process's*
//! // global allocator routes to it, and a binary that installs no
//! // `#[global_allocator]` routes nothing. A measurement binary declares
//! //     #[global_allocator]
//! //     static ALLOCATOR: Counting<System> = Counting::with_static_counters();
//! // and gets real figures; anything else gets a confident zero. The crate's own
//! // tests install one for exactly this reason.
//! let ((), allocations) = alloc::measure(|| {
//!     let data = vec![7u8; 16];
//!     core::hint::black_box(&data);
//! });
//! assert_eq!(allocations.delta.allocations, 0, "no global allocator is installed here");
//!
//! // Exact byte movement per named hop.
//! let meter = copies::Meter::new();
//! let ((), moved) = copies::measure(&meter, || {
//!     copies::record(copies::Site::SocketToBuffer, 16);
//! });
//!
//! // Per-phase decomposition, with the overhead check that makes it credible.
//! let probe = phases::Probe::new(phases::Path::Read);
//! let span = phases::Span::new(&probe);
//! probe.begin();
//! span.mark(phases::Phase::Parse);
//! let decomposition = probe.take();
//! println!("{}", decomposition.to_markdown());
//! # let _ = (calibration, allocations, moved, decomposition);
//! ```

#![forbid(unsafe_op_in_unsafe_fn)]
// A measurement crate converts counters to rates on nearly every line, and every
// one of those conversions is `u64 -> f64`, which `cast_precision_loss` flags by
// construction. The lint is right in general and wrong here: a counter past 2^53
// is not a number a benchmark can produce, and the alternative - saturating
// integer division - would make the reported rates wrong rather than
// approximate. Every `as f64` in this crate is a deliberate, exact-for-realistic-
// ranges conversion.
//
// The remaining pedantic lints stay on: they are how a measurement crate avoids
// measuring the wrong thing.
#![allow(clippy::cast_precision_loss)]

pub mod alloc;
pub mod control;
pub mod copies;
pub mod phases;
pub mod tsc;

pub use alloc::{Counting, PerOperation, ScopeReport, Snapshot as AllocationSnapshot};
pub use control::Control;
pub use copies::{CopyPerOperation, CopyReport, Meter, Site};
pub use phases::{Path, Phase, Probe, Report, Span};
pub use tsc::{Calibration, Ratio};
