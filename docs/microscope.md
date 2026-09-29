# The architecture microscope

Performance work cannot proceed from intuition. Every claim it makes - "this hash is faster", "the route lookup is the bottleneck", "the read
path copies three times" - has to be answerable by an instrument that reports what
happened rather than what was hoped for.

`crates/kivi-microscope` is that instrument. This document records what it can
measure, what it cannot, and the numbers that came out of it.

## 1. What this host can and cannot measure

Measured on the development host: Intel Core Ultra 9 285K, 24 physical cores, no
SMT, one NUMA node, 63.3 GB RAM, WSL2 on kernel `6.18.33.2-microsoft-standard-WSL2`.

```
$ cargo run --release -p kivi-microscope --bin kivi-microscope-caps
{
  "tsc_hz": 3686322143,
  "hz_spread_ppm": 11.9,
  "core_cycles_per_tick": 0.7319,
  "cycle_spread_pct": 5.46,
  "rdtsc_ticks": 24,
  "hardware_counters": "unavailable (use callgrind)"
}
```

| property | value | why it matters |
| --- | --- | --- |
| TSC frequency | 3.686 GHz, ±12 ppm | nanoseconds |
| core cycles per TSC tick | **0.73**, ±5.5% | **the TSC is not a cycle counter** |
| `rdtsc` cost | 24 ticks (6.5 ns) | the floor on phase resolution |
| hardware PMU | **none** | counts come from callgrind |
| L1d / L2 / L3 | 48 KiB / 3 MiB / 36 MiB | cache-miss analysis |
| AES / VAES / GFNI | present | hash construction options |
| AVX-512 | **absent** | group width for a table's control bytes |
| SMT | absent | no sibling contention to discount |

### The three findings that shape everything else

**1. There is no hardware PMU.** `perf list hw` is empty and every `cycles`,
`instructions` and `cache-misses` event fails with *"No supported events found"*.
WSL2 does not pass a PMU through. This is not a gap to work around quietly; it
divides the program's dimensions in two:

- **time** - the TSC, which works
- **counts** - callgrind, which counts perfectly and distorts time completely

The two are used together and never substituted for one another. Callgrind's
`Ir`, `D1mr`, `DLmr`, `Bcm` cover instructions/op, cache-misses/op and
branch-misses/op exactly; the TSC covers ns/op. Neither covers the other.

Cycle accounting is unavailable: there are no hardware performance counters on
this host. That is not the same as cycles being unmeasurable, because the TSC
yields them once its ratio to core clock is calibrated. That is the next finding.

**2. The TSC is a clock, not a cycle counter.** `constant_tsc`, `nonstop_tsc` and
`tsc_reliable` are all set, so `rdtsc` is stable and comparable across threads.
But it runs at 3.686 GHz while the core runs at roughly 5.27 GHz. A dependent
`add` chain - which retires at *exactly* one core cycle per iteration, by
construction - measures **0.6993 TSC ticks per iteration**.

So a TSC delta divided by the nominal frequency is a nanosecond count, and
calling it a cycle count overstates the core's cycles by about 37%. Cycles
require a ratio measured on the same core in the same run, with its spread
reported. `tsc::Ratio` exists for that, and `tsc::calibrate` measures it from an
inline-assembly dependent-add chain - a compiler-generated loop folds to a
constant, and an early version of the probe measured exactly `0.0000` ticks per
iteration for that reason.

**3. `rdtsc` costs 6.5 ns.** A `GET` on this host is about 160 ns, so a single
timestamp is ~4% of a request. Eleven phase boundaries per request would add tens
of percent. Consequences, all enforced:

- boundaries are adjacent, so a span is timed once and not twice
- `phases::Span::disabled()` is what instrumented code compiles to with no probe
  attached, so authoritative throughput runs pay nothing
- `Report::overhead()` compares the instrumented total against an independently
  measured baseline, and a decomposition that distorts its own measurement says
  so

## 2. What the crate provides

| module | instrument | answers |
| --- | --- | --- |
| [`tsc`] | calibrated `rdtsc`, `rdtscp` | how long did this take; how many core cycles |
| [`alloc`] | counting `GlobalAlloc` | exact allocations/op, bytes/op, size histogram |
| [`copies`] | per-site byte meter | exact bytes and copies per named hop |
| [`phases`] | per-phase probe | where a request's cycles go |
| [`control`] | four control servers | what does each stage of the path cost |

`alloc::Counting` is a `#[global_allocator]`, so it is installed per binary and
never in a shipped server. A counting run and an authoritative run are different
binaries by construction, not by a flag that might be left on.

### Allocation accounting is exact, not sampled

A sampling profiler answers "about 1.3% of requests allocate". `alloc::measure`
answers "1,240,000 of 1,000,000 `GET`s allocated exactly 3 times each", with a
33-bucket power-of-two size histogram, because the counters are incremented on
every call with no sampling and no inference.

Scopes are **per thread**. A server measures one connection on one worker, and a
shared counter would add exactly the cross-core traffic a single-core measurement
is trying to detect. The process-wide counters remain the authority on totals; a
scope narrows *when* and *on which thread* an allocation is attributed, never
*whether*.

### Copy accounting counts call sites, not `memcpy`

The obvious implementation - interpose `memcpy` - is the wrong tool in Rust and
would have produced authoritative-looking numbers that are false:

- LLVM inlines small copies as vector moves that never become a `memcpy` call, so
  a `memcpy` counter sees a fraction of the traffic
- catching the standard library's instantiations means interposing per object,
  which does not compose with a Rust workspace
- counting a *call site* is what an architecture decision needs. "The path moves
  2,048 bytes" is not actionable. "socket→buffer 1,024, parser→key 16,
  store→socket 1,024" is.

`copies::Site` is a closed set of the hops a Kivi data path can cross, so adding a
hop means adding a variant where a reader will see it. A path that moves bytes it
cannot attribute records them as `Site::Unattributed`, and
`CopyReport::has_unattributed()` makes an incomplete report announce itself rather
than quietly under-reporting.

### The control servers are the most useful thing here

A performance comparison needs a third point. "Kivi is 0.5× Redis" says nothing
about which of the two is wrong, because the difference could be in either server,
in the client, or in the machine.

| server | socket | RESP parse | classify | hash | table | semantics |
| --- | :-: | :-: | :-: | :-: | :-: | :-: |
| `noop` | ✓ | | | | | |
| `parse-only` | ✓ | ✓ | | | | |
| `hash-only` | ✓ | ✓ | ✓ | ✓ | | |
| `table-only` | ✓ | ✓ | ✓ | ✓ | ✓ | |
| Kivi | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Each row adds exactly one stage, so the difference between adjacent rows is the
marginal cost of that stage. The difference between `table-only` and Kivi is the
semantic layer - typed values, versions, expiry, transactions, durability
selection - which is often the largest single number and is not obtainable any
other way.

**Why this exists, concretely.** A whole per-request
cost is attributable to the wrong stage when the measurement does not separate the
stages: an in-process number can charge "the engine rendezvous" for what is
really half frontend. One extra ~80-line server separates them immediately, and
that is the entire justification.

**The threading model is part of the control.** Every control runs one thread per
connection on blocking sockets, which is what Kivi's RESP edge does. Thread-per-
connection costs 1.38× against a single event loop at identical work - but that is only visible if the control
shares the real server's threading and changes only what it *answers*. A control
with a different threading model measures the wrong difference.

**What a control is not.** Not a competitor, not a KiviTable prototype, not a
Redis-semantics implementation. It implements no semantics and validates no
replies; the question each answers is what a stage costs. A control that grew
into a competing design would make the subtraction meaningless, because the rung
would measure the control's own structure rather than the store's.

## 3. Measurement defects already found and fixed

Recorded because each produced a plausible number first.

**A control server that hashed the command name.** The frame parser's first
element is the command name, so hashing element 0 measures a table with one hot
key and a near-perfect hit rate, and reports it as a hashing cost. Pinned by
`the_frame_key_is_the_key_not_the_command_name`.

**A control table that could not find its own entries.** Fingerprints were
recorded at an entry's *home* bucket while the entry was placed in the *probe*
slot. Every probed entry was invisible to its own lookup: the key was in the table
and the table could not find it. Pinned by
`the_table_does_not_depend_on_insertion_order_for_lookups`.

**A control that reported an empty key as a miss.** A `PING` frame has one
element, and indexing the unused tail of the span array returned `Some(&[])` - a
zero-length key that hashes successfully and misses the table, so the miss count
blamed the table for a frame that never had a key. Pinned by
`a_frame_without_arguments_has_no_key`.

**A parser that invalidated the frame it had just returned.** The consumed prefix
was reclaimed when the last frame was popped, which emptied the buffer a returned
frame still pointed into. Reclaiming on `push` instead fixed it; two frames from
one read can now be held at once, which is also why `Frame` carries offsets
rather than a borrow. Pinned by `pipelined_frames_come_back_in_order`.

**An assertion that `thread::sleep` is exact.** A test asserted two calibrations
return bit-identical frequencies. The TSC *is* invariant, but each calibration
divides a tick count by a sleep the OS overshoots, so the estimate carries tens of
ppm. The assertion now bounds the drift and the calibration *reports* its own
uncertainty, because a duration quoted to better precision than the clock's rate
was measured to is quoting precision that does not exist.

## 4. What comes next

The microscope answers questions about the current code. The next questions are
about replacing it, and the plan and its evidence are in
[`perf-program.md`](perf-program.md).

[`tsc`]: ../crates/kivi-microscope/src/tsc.rs
[`alloc`]: ../crates/kivi-microscope/src/alloc.rs
[`copies`]: ../crates/kivi-microscope/src/copies.rs
[`phases`]: ../crates/kivi-microscope/src/phases.rs
[`control`]: ../crates/kivi-microscope/src/control.rs
