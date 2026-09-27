# Hardware acceleration

Everything in this document was measured. Where a mechanism was not available,
it says so instead of estimating, because "we did not measure direct I/O" and
"direct I/O was not faster" are different claims and only one of them is
supportable from an absent mechanism.

Linux is a first-class target here, not a `cfg` attribute. Every Linux path
below was compiled and run on WSL2 (kernel `6.18.33.2-microsoft-standard-WSL2`)
and every number under "Linux" came out of that machine. Where the two hosts
disagree, both are reported.

The governing rule is unchanged by any of this: **Exact State. Adaptive
Everything Else.** No hardware mechanism below may affect a durability, ordering,
or consensus guarantee. Every one of them is an optimisation with a software
path behind it, and the code says which path it took.

## The two machines

| | Windows | Linux (WSL2) |
| --- | --- | --- |
| CPU | Intel Core Ultra 9 285K | 24 logical, reported as 24 cores |
| Logical / physical | 24 / 24 (SMT off) | 24 / 24 |
| Packages / memory nodes | 1 / 1 | 1 / 1 |
| L1d | 32 KiB x 15, 48 KiB x 7 | 48K x 23 |
| L1i | 64 KiB x 23 | 64K x 23 |
| L2 | 3 MiB x 7, 4 MiB x 3 | 3M x 23 |
| L3 | 36 MiB | 36M |
| ISA in use | `sse4.2`, `avx2` | `sse4.2`, `avx2` |
| Performance-core class | heterogeneous (8 P / 16 E) | uniform |
| `perf_event_paranoid` | n/a | 2 |
| THP policy | n/a | `madvise` |
| Root volume | NTFS | ext4 on `/dev/sdd`, 512-byte sectors |

One node and no SMT siblings limit what could be measured, and both limits are
stated again at the measurement they constrained. Neither host has a hardware
PMU, CXL, pmem, RDMA, or DSA.

## What the machines offer

From `kivi_hardware::snapshot().report()`:

| Dimension | Windows | Linux (WSL2) |
| --- | --- | --- |
| Topology | `windows-topology-api` | `linux-sysfs` |
| Huge pages | reserved, 2 MiB, privilege yes | transparent, `madvise` |
| Direct I/O | mode present, alignment unknown | mode present, **alignment 512** |
| PMU | available (page-faults, context-switches, migrations, task-clock) | **page-faults, context-switches, migrations, task-clock** |
| DAMON | not present | not present (`CONFIG_DAMON` unset) |
| io_uring | unsupported platform | **registered-files, registered-buffers, buffer-ring, completion-events** |
| CXL / pmem regions | none | none |
| RDMA / DSA | none | none |

The capability registry reports the *stage* each mechanism reached, not just
whether a probe returned something:

```text
stages[pmu=backend-initialised damon=hardware-present[not-present]
       huge-pages=backend-initialised direct-io=backend-initialised
       io-uring=backend-initialised cxl-memory=hardware-present[not-present]
       rdma=hardware-present[not-present] dsa=hardware-present[not-present]]
```

`backend-initialised` means a probe actually succeeded, not that a struct
exists. `hardware-present[not-present]` means the hardware is absent and the
code says so.

## Linux: what was proven

### Direct I/O really engages

`cargo run -p kivi-hardware --example dcheck <path>` on the ext4 volume backed
by `/dev/sdd`:

```text
logical_block_size = Some(512)
mode = unbuffered(align=512) unbuffered = true align = 512 len = 65536
offset 0 len 4096 -> 4096 bytes, first=[0] last=Some(79)
offset 1 len 100 -> 100 bytes, first=[1] last=Some(100)
offset 4095 len 2 -> 2 bytes, first=[79] last=Some(80)
offset 8192 len 8192 -> 8192 bytes, first=[160] last=Some(68)
```

`unbuffered = true` is read back from the open handle, and the four windows —
including two deliberately unaligned — return the right bytes. The sector size
comes from `statx(STATX_DIOALIGN)`, with a sysfs
`queue/logical_block_size` fallback (walking partition to parent device), and
`IOCTL_DISK_GET_DRIVE_GEOMETRY` on Windows.

Direct I/O is applied to the bulk `scan_records` path only. A single-record
promotion stays buffered: it reads a 20-byte header and a small payload, and an
unaligned unbuffered read of that size would need a rounding read that is
larger than the record.

### The PMU opens, and a non-commensurable set withholds its ratios

```text
pmu support           : available (page-faults,context-switches,migrations,task-clock)
pmu counters attached : counters=[page-faults,context-switches,migrations,task-clock]
                        source=task-clock-ns comparable=true
                        delta=Delta { cycles: 199987053, instructions: 0, ...,
                                      page_faults: 1, context_switches: 0,
                                      migrations: 0, ordering: Monotonic }
pmu per-event probe:
  cycles                   not-implemented-by-this-kernel-or-cpu
  instructions             not-implemented-by-this-kernel-or-cpu
  cache-references         not-implemented-by-this-kernel-or-cpu
  cache-misses             not-implemented-by-this-kernel-or-cpu
  branches                 not-implemented-by-this-kernel-or-cpu
  branch-misses            not-implemented-by-this-kernel-or-cpu
  frontend-stalled-cycles  not-implemented-by-this-kernel-or-cpu
  task-clock               open
  minor-page-faults        open
  context-switches         open
  migrations               open
```

Three things about this shape are load-bearing.

**The kernel refuses the group.** `PERF_IOC_ADD_EVENT` on a software-event
group returns `EACCES` on WSL2's kernel, while the same events opened
standalone succeed. That is the fallback's reason to exist, and it is a property
of the machine rather than a mode anyone chose.

**The fallback must not discard the denominator.** The first version reported
`CycleSource::None` and dropped the task clock it had just opened, so every
sample read as `comparable: false`, `cycles: 0`, `trust: 0` — identical to "this
machine has no PMU", which is a different fault with a different fix. The
fallback now opens the task clock as its own denominator and reports
`TaskClockNanoseconds`, and `PmuCounter::TaskClock` is a distinct variant so a
consumer can tell "the kernel gave me hardware cycles" from "the kernel gave me
the task clock" — the first supports ratios the second does not.

**Commutability is decided per sample, not from the counter set's shape.**
Independently opened counters are commensurable whenever the kernel scheduled
every one of them for the whole interval, which `time_running == time_enabled`
is the kernel's own statement about. Marking the set non-commensurable the moment
it fell back withheld every ratio on a machine that never multiplexed anything.
`comparable` is now derived per sample from the scaling bits, and the two can
never disagree.

With a task-clock denominator, IPC and stall fraction report *no ratio* rather
than a plausible number, because both need hardware cycles and a frequency change
would move the result. The absolute counts are unaffected. `reason()` still
distinguishes ENOENT (no PMU) from EACCES (privilege), and `diagnose()` probes
each event individually, so `/v1/hardware` names which of the two a machine is
in.

### io_uring is a real syscall, not a shape

```text
io_uring: Available(IoUringSupport { features: {RegisteredFiles, RegisteredBuffers,
                                                 BufferRing, CompletionEvents} })
```

`io_uring_setup` is called and `io_uring_params.features` is read. The features
above are the kernel's own bitmask.

### Huge pages: the advice was taken, and it did not help

Kernel evidence, read back from `/proc/self/smaps` after advising an 8 MiB
region and touching every page:

```text
huge page evidence: advice=true resident=8MiB anon_huge=6MiB mmu_page=4KiB
```

6 of 8 MiB really are anonymous huge pages. `mmu_page=4KiB` because the
remaining 2 MiB is still base-page backed, and the mapping reports the majority
size. `AnonHugePages` is the unambiguous number.

Medians of 15, same binary, same region, same loops:

| Workload | Advised | Unadvised | Ratio |
| --- | --- | --- | --- |
| `memory_huge_page_advice` (faulting) | 230.5 µs | 181.8 µs | **0.79x** |
| `memory_huge_page_advice_chase` (TLB-bound walk) | 247.1 µs | 192.6 µs | **0.78x** |

**Huge pages are a 1.3x loss on this machine.** The promotion is real, the
measurement is real, and the conclusion is that the mechanism does not pay here:
8 MiB is small enough that the TLB never became the bottleneck, while the
slower huge-page fault and the coarser page are pure cost. `HugePagePolicy`
therefore stays `Inherit` by default and `Advice` is opt-in per arena above
`HUGE_PAGE_THRESHOLD_BYTES` (4 MiB).

The benchmark refuses to report a timing at all unless the region was actually
promoted, so the two rows cannot silently become a comparison of one region
with itself.

Two things were wrong in the first version of that benchmark and are fixed:
the unadvised row's stores were dead by the end of the closure and the compiler
elided them (27 ns for 8 MiB of faulting), and the `smaps` parser split blocks
on `\n\n`, which WSL2's kernel does not emit, so it found no mapping at all.
Both are covered by tests now.

### NUMA

`kivi_hardware::numa::Arena` mmaps, binds with `mbind` **before the first
touch**, and advises `MADV_HUGEPAGE`. Placement is then verified by parsing
`/proc/self/numa_maps` for the mapping's per-node page counts — 11 tests on
Linux, including `the_census_agrees_with_where_the_pages_went` and
`a_real_node_binds_or_reports_why_not`. On this single-node host every outcome
resolves to node 0, which the code reports as the node rather than as a
shortfall.

## Windows: what was measured

Medians of 30 from
`cargo bench -p kivi-hardware --bench hardware -- --sample-count 30`; the NVMe
table is from `cargo bench -p kivi-memory --bench nvme_read`.

### CRC32C: already hardware accelerated, do not replace it

This was the open question in RFC §26, and the measurement settles it.

| Workload | Time | Throughput |
| --- | --- | --- |
| `crc32c_1mib_batch_body` | 44.99 µs | **23.3 GB/s** |
| `crc32c_oracle_1mib` (bitwise scalar) | 4.311 ms | 0.24 GB/s |
| `crc32c_4mib_chunk` | 181.7 µs | 23.1 GB/s |
| `crc32c_20_byte_header` | 6.49 ns | — |

**96x over the scalar oracle.** The `crc32c` crate is dispatching the SSE4.2
`CRC32` instruction; 23 GB/s is roughly memory bandwidth on this machine, which
is the ceiling. A different crate advertising SIMD would be advertising the same
instruction. `reed-solomon-simd` and `blake3` are left alone for the same
reason — BLAKE3 measures 7.5 GB/s at 1 MiB, already multi-threaded and
dispatching AVX2.

### SIMD kernels: two kept, one deleted

Kivi's own byte kernels, each dispatched with `multiversion` against a scalar
oracle that is the definition of correctness.

| Kernel | Dispatched | Oracle | Win |
| --- | --- | --- | --- |
| `is_all_zero` | 90.14 µs | 251.7 µs | **2.8x** |
| `position_of` | 36.49 µs | 66.69 µs | **1.8x** |
| ~~`xor_fold`~~ | 95 µs | 89 µs | none — **deleted** |

The XOR fold was implemented, measured, and removed. The compiler already
vectorises that shape, so the dispatch bought nothing and the code was pure
cost. A kernel with no measured win is not a kernel. The two survivors are
verified against the oracle at every length from 0 to 300 plus 1 KiB, 4 KiB and
65537, at five needle values, so a vector path that only handles whole blocks
fails the test rather than shipping.

### CPU affinity: no single-thread win on this machine

| Placement | Median |
| --- | --- |
| `cpu_unpinned` | 67.14 µs |
| `cpu_pinned_to_physical_core` | 67.04 µs |
| `cpu_pinned_to_current_core` | 78.29 µs |

Pinning to a physical core and not pinning are indistinguishable (0.1%), and
both beat "the core the thread happened to start on" — which is a measurement of
which core you landed on, not of affinity. The honest conclusion is that **on
this machine, for this workload, affinity buys nothing on a single thread**. The
case for pinning is contention between threads, and this machine has no SMT
siblings, so that case could not be produced: the `cpu_pinned_to_smt_sibling`
row measures two threads on two separate cores and says so in its output rather
than pretending to be the SMT case.

That is why `kivi-hardware` treats placement as diagnostics-first: the plan is
free (below), so the strategy can be chosen on measured benefit, and on a
machine like this one the answer is "none".

### Planning a placement is free

| Strategy | Median for one worker |
| --- | --- |
| `placement_unbound` | 132 µs |
| `placement_simple` | 136.6 µs |
| `placement_physical_core` | 144 µs |
| `placement_numa_aware` | 144.8 µs |
| `report_once` (a full `Capabilities::detect`) | 291.7 µs |

Every row is dominated by the same `Capabilities::detect()` call, so the
marginal cost of resolving a plan is at or below the measurement floor. The four
strategies differ in *which* processor a thread gets, not in how long deciding
costs — which is what makes it safe to default to a real strategy and to change
that default when a measurement says so.

### The NVMe demotion-store read: the one large win

`NvmeProvider::read_at` used to open the whole demotion store and copy all of
it to answer one record. A promotion was therefore O(store), not O(record). It
now reads the 20-byte record header at the record's own offset, then the
payload, which is O(record).

Both strategies in one binary, same store, 1 KiB records, medians of 15:

| Records | Store | Positioned (all records) | Whole-file (all records) | Speedup |
| --- | --- | --- | --- | --- |
| 512 | 0.51 MB | 11.13 ms | 27.15 ms | 2.4x |
| 2048 | 2.1 MB | 41.97 ms | 1.48 s | **35x** |
| 8192 | 8.6 MB | 170.8 ms | 21.82 s | **128x** |

Per promotion, which is the number that matters:

| Records | Positioned | Whole-file |
| --- | --- | --- |
| 512 | 21.7 µs | 53.0 µs |
| 2048 | 20.5 µs | 722 µs |
| 8192 | 20.8 µs | **2664 µs** |

The positioned cost is flat — 21.7, 20.5, 20.8 µs — because it does not depend
on the store. The old cost grows 50x across the same range. There is no store
size at which the old path is preferable.

The handle is opened per call rather than cached. Caching it needed interior
mutability behind a `&self` method, which cost the provider its `Sync`; the open
costs less than the positioned read it enables, so the trade was refused. The
*append* handle is still cached, because it sits behind `&mut self` on the
single-writer path that owns every device append.

### What the workload matrix does and does not show

Re-running the fabric matrix on this machine produced
`fabric-durable-set-kb1-8t` at 165.9 ops/s in the recorded baseline and
849–1215 ops/s across five runs today. **That is not a 7x hardware win and is not
reported as one.** Two facts rule it out:

* `kivi-durability` is untouched by this phase, and the durable-set benchmark is
  dominated by WAL `fsync`.
* The `fabric-worker-*` directories are **empty** after the run, and
  `GET /v1/fabric` reports `total_objects: 0`, `demotions: 0`, `promotions: 0`.
  A 1 KiB value never reached the NVMe demotion device, so the code this phase
  changed was not on that path at all.

The baseline and today's runs differ by the machine's disk state, not by a code
change. Quoting the ratio would be quoting the weather. The same applies to the
`get` benchmarks, which landed at 41–45k ops/s against a 50.5k baseline in four
tightly-clustered runs; with the fabric provably unused, nothing in this phase is
on that path either, so it is reported as unaccounted-for and environment-bound
rather than as a regression caused by the change or dismissed as noise.

This is why the NVMe table above measures the mechanism directly instead of
inferring it from an end-to-end number.

## Hardware telemetry reaches the controller

A `PmuSample` used to stop at `kivi-hardware::telemetry`, so no controller could
tell a software estimate from a hardware measurement. The chain now runs:

```text
worker thread (owns the counter group)
  -> HardwareSampler::end_of_window(total_work) -> HardwareSample
  -> HardwareSignal (closed, bounded, carries its own provenance)
  -> HardwareSummary (fold by worst-case, trust by minimum, counts additive)
  -> ObservationFeatures -> BaselineController branches
```

The counter group lives on the serving thread because a counter opened on
thread A counts thread A's instructions. The sample is taken by a
`WorkerControl::HardwareSample` rendezvous on that thread, not by a collector
thread that would have to signal the owner on the hot path or measure the wrong
thing. The serving thread's own request count is the corroboration, and a
window that served nothing is reported as unmeasured work rather than as a clean
bill of health.

The summary folds by extremes, not by mean, and that is the whole point:

* worst stall fraction, worst cache-miss ratio, worst branch-miss ratio,
  worst cost multiplier — a controller asking "did this worker ever stall badly"
  needs the worst window;
* best IPC — a controller deciding whether a worker is CPU bound needs to know
  whether it ever reached full issue;
* minimum trust — one multiplexed window in three must not be averaged away by
  two exact ones;
* page faults, context switches, migrations — additive, because a page fault is
  a page fault whether or not a stall counter was readable.

Two new Phase 12 branches act on it, and both *reduce* work rather than adding
it, because a memory-bound or migrating worker is already paying for bandwidth:

| Branch | Condition | Action |
| --- | --- | --- |
| `SustainedMemoryBoundWorker` | measured, trusted, stall > 0.35 and LLC miss > 0.2 for `sustained_windows` | `AdjustScrubBudget` down |
| `SustainedWorkerMigration` | measured, trusted, any migration for `sustained_windows` | `AdjustScrubBudget` down |

`BaselineConfig::hardware_min_trust_ppm` (default 900 000) gates both, so a
multiplexed reading can never reach a decision on its own. A machine with no
counters produces exactly the decision the software-only path produced before
this existed — asserted by `a_machine_without_counters_is_unaffected`, which
compares the two proposals rather than trusting the intent.

### Proven on a live server

`GET /v1/control/adaptive` reports the per-worker `HardwareSummary` for the
latest window, so "the chain is wired" is checkable rather than asserted. A
two-worker `--pmu` server on WSL2, after two control intervals:

```json
{"worker": 0, "measured": true, "trust": 1.0, "page_faults": 2,
 "context_switches": 0, "migrations": 0, "best_ipc": 0.0,
 "worst_stall_fraction": 0.0, "worst_cache_miss_ratio": 0.0,
 "worst_cost_multiplier": 1.0, "cache_sensitivity": "Resident"}
{"worker": 1, "measured": true, "trust": 1.0, "page_faults": 0, ...}
```

Two real page faults counted by worker 0's own counter group, carried through the
sampler, the closed signal, the fabric fold, and out through the admin plane. The
ratios are zero because this machine has no instruction, cycle, or cache
counter — which is the correct answer, not a missing one.

## Classification of every track

| Track | Classification | Decision |
| --- | --- | --- |
| CPU topology discovery | in-tree (`GetLogicalProcessorInformationEx`, Linux sysfs) | **Adopted.** No dependency. |
| Thread placement | in-tree | **Adopted**, diagnostics at `GET /v1/hardware`. A refusal is logged and the thread runs unbound: placement is an optimisation, so it may not gate availability. |
| NUMA-aware fabric | in-tree (`mbind` + `numa_maps` census) | **Adopted.** One provider per memory node. Verified on Linux; a single-node host cannot produce a cross-node case. |
| CRC32C | `crc32c` (upstream) | **Retained.** Measured 96x over scalar; the crate is already using the hardware instruction. |
| BLAKE3 | `blake3` (upstream) | **Retained.** 7.5 GB/s, already dispatched. |
| Reed-Solomon | `reed-solomon-simd` (upstream) | **Retained.** No measured gap. |
| SIMD kernels | in-tree, `multiversion` | **Adopted** for `is_all_zero` and `position_of`; the third was measured and deleted. |
| `hwlocality` | external | **Rejected.** Pre-1.0 (1.0.0-alpha.12), needs C plus pkg-config/cmake/automake, and its Windows story is weak. The topology this phase needs is a few hundred lines of `GetLogicalProcessorInformationEx`, and those lines are now tested. |
| `core_affinity2` | external | **Removed.** Pulls `libafl`, offers no topology, and the placement boundary it served is now `kivi-hardware`. |
| Direct I/O | in-tree, `statx(STATX_DIOALIGN)` + `O_DIRECT` | **Adopted for bulk scans.** Proven engaged on ext4 (`unbuffered = true`, 512-byte alignment). Single-record promotion stays buffered by measurement. |
| Registered buffers | `kivi-hardware::io::BufferPool` | **Model only.** Not wired into the NVMe lane. |
| Huge pages | in-tree policy + Linux `MADV_HUGEPAGE` + `smaps` census | **Adopted as advice, and measured to be a 1.3x loss on this host.** Default stays `Inherit`; `Advice` is opt-in above 4 MiB. |
| Compio advanced I/O | `compio` | **Not adopted.** No measured syscall or latency bottleneck to justify it. |
| PMU | `perf_event_open`, `pmu` feature | **Adopted on Linux.** Counter *group* with independent fallback, software-event fallback, `comparable` flag, per-event `diagnose()`. Wired to the Phase 12 controller. |
| DAMON | sysfs adapter | **Implemented, not present here, parser tested against the documented format.** Reports `NotPresent`; no claim about a machine without it. |
| io_uring | `io_uring_setup` probe | **Implemented.** Reads `io_uring_params.features`. |
| CXL / DAX | in-tree mapping + `CxlProvider` in `kivi-memory` | **Adopted as a provider.** Registers uncalibrated and `healthy: false` until measured; never claims persistence; anonymous emulation is a distinct kind tag. No CXL hardware here. |
| SPDK | — | **Evaluated, not adopted.** No NVMe-oF requirement exists. |
| RDMA | discovery only | **Not adopted.** No fabric requirement; the control and data paths are loopback and TCP. |
| DSA | discovery only | **Not adopted.** No measured compaction bottleneck. |
| DPDK | — | **Not adopted.** No measured packet-processing bottleneck. |

## What is not done

Stated plainly, because a phase that claims completion it cannot evidence is
worse than one that names its gaps:

* No hardware PMU, CXL, pmem, RDMA, or DSA was available to measure on either
  host. The PMU path is proven against software events and a task-clock
  denominator; a hardware-event comparison has not been run, and on this machine
  `ipc` and `stall_fraction` correctly report *no ratio* rather than a value.
* DAMON is absent from both kernels (`CONFIG_DAMON` unset on the WSL2 kernel),
  so the sysfs adapter is exercised by parser tests against the documented
  `damon_stats` layout and reports `NotPresent` on a live machine. It has never
  read a real monitor.
* The host has one NUMA node, so `mbind` always resolves to node 0 and
  `numa-aware` placement reports `single-node` rather than a cross-node plan.
  Nothing here validates the two-node case.
* Huge pages were measured and did not help. `HugePagePolicy::Advice` remains
  available and remains off by default.
* Direct I/O covers the bulk `scan_records` path. Single-record promotion is
  buffered, by choice and by measurement.
* Registered buffers and Compio advanced I/O are not integrated.
* No redundant-coding measurement was taken, because no alternative coding path
  was implemented to compare against.
* Nothing was committed.

## Validation

| | Windows | Linux (WSL2) |
| --- | --- | --- |
| `cargo fmt --all --check` | clean | clean |
| `cargo check --workspace --all-targets` | clean | clean |
| `cargo check --workspace --all-targets --all-features` | clean | clean |
| `cargo clippy --workspace --all-targets --all-features` | zero warnings | zero warnings |
| `cargo nextest run --workspace --all-features` | 1421 run, 1419 pass, 2 fail | 1422 run, 1390 pass, 32 fail |

Every failure is in `kivi-lab`'s multi-process cluster suite, and every one of
them predates this work. That is measured, not asserted: the identical suite was
run against a pristine `git archive HEAD` copy on the same Linux host and fails
the same 31 tests. Diffing the two failure sets by test name gives an empty set
in both directions.

```text
only in the changed tree (potential regressions): (none)
only in the baseline:                            (none)
non-kivi-lab failures:                           0
```

The changed tree's 32nd failure is
`kivi-lab::cluster_elastic control_leader_failover_during_repair`, which is flaky
rather than broken: run alone three times it passes once and fails twice, on the
changed tree *and* on the untouched baseline, with the same pattern. The two
Windows failures are the same family, and `auto_repair_after_hard_kill` passes in
isolation in 78 s.

The root cause of the Linux cluster failures is a harness gap, not a product
defect: the 4th member is spawned into a fresh directory with no
`--control-seeds` and no `add_learner`, so it waits for a control group that a
founding voter already created — `node 4 has to be a member ... membership:
configs [{1,2,3}]`. Fixing it is a change to `kivi-lab`'s cluster fixture, which
is outside this phase.

These tests also leak `kivi-server` processes when they fail, which is what
makes a later `cargo` invocation fail to relink `target/debug/kivi-server.exe`.

## Reproducing

```text
# Windows and Linux
cargo run -p kivi-hardware --example detect
cargo run -p kivi-hardware --features pmu --example detect   # Linux, real PMU probe
cargo run -p kivi-hardware --example dcheck <path>           # prove O_DIRECT engaged
cargo bench -p kivi-hardware --bench hardware -- --sample-count 30
cargo bench -p kivi-memory --bench nvme_read -- --sample-count 15
cargo run -p kivi-server --ephemeral --port 9000 --admin 127.0.0.1:19080 \
       --workers 2 --pmu                                    # then:
curl http://127.0.0.1:19080/v1/hardware
curl http://127.0.0.1:19080/v1/control/adaptive             # per-worker hardware
```

`/v1/hardware` reports the capability snapshot, the topology summary, the ISA
set, every memory node and region, every device with its NUMA locality, the
resolved placement with its shortfalls, where the calling thread actually is,
every planned worker slot, and the classification of every researched track.
`/v1/control/adaptive` adds the per-worker hardware summary the latest window
carried. Both are diagnostics and never gates: a machine with no accelerators
answers with a complete report of absences.
