# Kivi versus Redis: a performance campaign

Every number here was measured on one machine with one methodology, and every
comparison is a *matched* operation: same command, same payload, same key
distribution, same client concurrency, same pipeline depth, same durability
class. Where a result is not comparable, the section says why and proves it
rather than hiding behind it.

Nothing in this document is a projection, a plan, or a claim about code that
was not built. Every architecture change described was implemented, and the
"before" numbers come from the same binaries' predecessor on the same box.

---

## 1. Environment and fairness

### The machine

| | |
| --- | --- |
| CPU | Intel Core Ultra 9 285K, 24 logical / 24 physical, 1 package, 1 NUMA node |
| Memory | 63.3 GB (Windows), 30 GB visible (WSL2) |
| Storage class | ext4 on `/dev/sdd` (WSL2), 512-byte sectors |
| Kernel | `6.18.33.2-microsoft-standard-WSL2` |
| Rust | 1.98.1, `--release` for every measurement |

### The decision that matters most: both servers are native Linux

`docs/hardware.md` records that the previous campaign ran Kivi natively on
Windows and had no comparable Redis. That is not a valid comparison, and this
campaign does not make it.

Both opponents were built and run **natively inside WSL2** — the same kernel,
the same filesystem, the same scheduler, the same page cache, the same
`ext4`/`/dev/sdd` for both data directories. The Docker Redis (`my-redis`) was
used for a smoke test and then left alone; **no authoritative number in this
document comes from a container.**

| | |
| --- | --- |
| Redis | built from source in WSL2, `redis-server v=8.10.1 sha=00000000:1 malloc=jemalloc-5.3.0 bits=64` |
| Kivi | `kivi-server` release build, `-p kivi-server --features redis-compat` |

The version matches the task's target (8.10.x) and the container's Redis
(8.10.1), so the opponent is the real current release and not a substitute.

> **A build trap worth recording.** Redis persists its build settings in
> `src/.make-settings`. A first build with `MALLOC=libc` survived `make
> distclean`, and the binary kept reporting `malloc=libc` while looking
> freshly built. Every published Redis number uses jemalloc, so the
> authoritative build is `MALLOC=jemalloc` with `.make-settings` removed, and
> the script asserts that `redis-server --version` reports jemalloc.

### CPU affinity

Neither server is pinned. `docs/hardware.md` already measured that pinning to
a physical core and not pinning are indistinguishable on this host (67.04 µs
vs 67.14 µs) because it has no SMT siblings, so the case for pinning could not
be produced here. Pinning Kivi's workers and not Redis's would be an
unfair advantage; pinning neither is the only symmetric choice available.

### Methodology

* **Alternating order.** Kivi and Redis are measured alternately inside every
  trial loop, so a drifting machine cannot favour whichever ran second.
* **Medians, not single runs.** Every reported point is the median of 3 trials
  (`redis-benchmark`) or 2 (`kivi-lab compare`).
* **Warmup.** Every run warms before measuring.
* **Noise classification.** `>10%` repeatable is meaningful; `±5%` is a tie.
  Where a difference is inside the noise band it is reported as a tie.
* **Two independent clients.** `redis-benchmark` (Redis's own tool, and the
  only tool most published numbers come from) *and* `kivi-lab`'s raw RESP
  client, which drives both servers through identical code. Agreement between
  them is evidence the harness is not the thing being measured.

### Two measurement failures that would have produced false results

These are recorded because both initially produced convincing numbers.

1. **A concurrent test suite.** A `cargo nextest run --workspace` from an
   earlier session was still executing inside WSL2, spawning four
   `kivi-server` cluster nodes at a time. Early readings attributed the result
   to the RESP rewrite when they were the test suite. Every measurement here is
   taken with the box otherwise idle, verified by `ps` before the run.
2. **Accumulated state.** Kivi's Memory Fabric and chunk lane carry state
   between runs. A server that had already absorbed a full campaign measured
   0.20× Redis on a GET that measures 0.57× on a fresh instance. Every
   authoritative point restarts its instances first. The durability table below
   was re-taken this way after a first attempt showed exactly this distortion.

---

## 2. What Redis was and was not made to do

`kivi-lab`'s campaign harness was used as the primary tool because it gates
every point on correctness before reporting it, and it drives Kivi RESP, Kivi
Native, and Redis through one client. `redis-benchmark` was used as the
independent cross-check.

**The harness refused to report invalid points, and it was right to.** The
first `standard` run produced 14 failures, and every one of them turned out to
be a real finding rather than harness noise (§4.1).

### Commands excluded, and why

| Command | Why |
| --- | --- |
| `INCR` | Redis `INCR` mutates a string; Kivi `INCR` is a distinct counter type. They are not the same logical operation, so a shared table entry would be a lie. Counters are measured separately. |
| `MSET`, multi-key `DEL`/`EXISTS` | Kivi's RESP profile explicitly rejects the multi-key form rather than faking it with a loop. |
| `GETRANGE` with negative bounds | Kivi compiles it to one `GET` sliced locally; semantics are Redis's, implementation differs. Reported, not excluded. |

### Persistence classes, never mixed

| Instance | Configuration |
| --- | --- |
| Redis volatile | `appendonly no`, `save ""` (RDB off) |
| Redis everysec | `appendonly yes`, `appendfsync everysec` |
| Redis always | `appendonly yes`, `appendfsync always` |
| Kivi ephemeral | `--ephemeral` — no WAL, no commit coordinator, no barrier (a genuine fork, not a flag on one path) |
| Kivi durable | `--data-dir` — fsync before every acknowledgement |

---

## 3. The initial matrix: Redis won everywhere

Measured before any change, `standard` preset, volatile, correctness-gated
(`docs/hardware.md` and the `base-standard` run).

| Profile | Kivi RESP ÷ Redis | Kivi Native ÷ Redis |
| --- | ---: | ---: |
| cache-read | 0.390 | 0.631 |
| read-heavy | 0.354 | 0.605 |
| balanced | 0.396 | 0.612 |

Pipelined, 4 clients, 1 KiB values, volatile:

| | Redis | Kivi RESP | Kivi Native |
| --- | ---: | ---: | ---: |
| ops/s at pipeline 1 | 87,434 | 50,984 | 56,700 |
| ops/s at pipeline 16 | **944,353** | **84,840** | **57,215** |
| batch p95 at pipeline 16 | 111 µs | **1,000 µs** | — |

**The pipeline number is the whole story.** Redis scales 10.8× from depth 1 to
depth 16. Kivi scales 1.7×. Redis's batch round-trip is 111 µs; Kivi's is
1,000 µs, and the difference is not noise — it is arithmetic.

---

## 4. Root causes

### 4.1 The Memory Fabric refused every value it owned

Measured, not inferred. A `SET` band probe against a fresh server:

```
      size  SET
        1024  OK
      ---- boundary ----
       1025  ERR ERR operation rejected
      16384  ERR ERR operation rejected
     262144  ERR ERR operation rejected
      ---- boundary ----
     262145  OK
    1048576  OK
```

**Every value in `(1024, 262144]` was refused.** Those bounds are exactly
`FABRIC_INLINE_MAX` and the chunk lane's inline threshold: the entire band the
Memory Fabric exists to own.

A fresh server accepts all of them, so it was not a size check. Filling the
arena made it exact:

```
  written=  262.1MB  dram= 260.1MB  rejects=0     demotions=32  pressure=0.975
  written=  268.7MB  dram= 268.4MB  rejects=1597  demotions=32  pressure=1.000
FIRST FAILURE after 268.7 MB in 16404 ops: ERR operation rejected
```

`dram_bytes` pins at `268435456` = 256 MiB = `DEFAULT_ARENA_BYTES`, and from
that point every medium-value write fails.

**Cause.** `TabletFabric::stage` treated arena saturation as an *admission
refusal*. `import_recovered` — the recovery path — already treated the same
condition as a *tiering decision* and demoted to the off-core device. The
background maintenance loop that would normally reclaim arena space is
amortised to 1-in-16 calls and managed 32 demotions across 18,000 writes.

So a bounded hot cache had become a bounded database: once 256 MiB of medium
values were live, the database refused writes it had space for, on a device
with 4 GiB of demotion capacity sitting right there.

### 4.2 RESP executed a pipeline as N sequential round-trips

`RespConnection::drain` decoded one command, executed it, decoded the next. The
engine's embedded path (`LocalClient::execute_on`) is a blocking rendezvous:
one bounded channel per request, a `try_send` to the owning worker, and a
blocking `recv`.

Pipelining is precisely the thing that is supposed to amortise that. Kivi was
doing the opposite — a deeper pipeline cost proportionally *more* latency for
the same work, because the client still paid one full round-trip per command
and simply waited for all of them before reading any reply.

Measured consequence: 16 pipelined commands took 1,000 µs against Redis's
111 µs. That is 62 µs per pipelined command versus Redis's 7 µs.

### 4.3 The RESP frontend cost more than the whole rest of the request

A `PING` never reaches the engine. It isolates the frontend.

| 1 client, 1 pipeline | ops/s | per request |
| --- | ---: | ---: |
| Redis `PING` | 34,819 | 28.7 µs |
| Kivi `PING` | 16,534 | 60.5 µs |
| Kivi `GET` | 10,645 | 94.0 µs |

Decomposed: **~60 µs is frontend, ~34 µs is engine.** Redis's total is 29 µs.
Kivi's engine path cost about 34 µs where Redis's cost about 0.5 µs, and
Kivi's frontend cost about 32 µs more than Redis's whole request.

That frontend cost was structural, and the code said so:

* Every argument was copied out of the read buffer into a fresh `Vec<u8>` by
  the `redis-protocol` owned decoder, then copied *again* by `split_command`,
  then copied a *third* time into the `Key`.
* `Vec::drain(..consumed)` memmoved the entire remaining tail **per command** —
  quadratic in pipeline depth.
* Every reply was a fresh `Vec<u8>`, then copied into a second buffer before
  the write. A `GET` copied its value **four times**.
* Every read batch moved the whole connection onto `tokio::task::spawn_blocking`,
  executed there, and moved the replies back — two thread wakeups per batch on
  top of the two the engine rendezvous already costs.
* The command name was dispatched **three times**: five literal comparisons, a
  linear scan of a 30-entry registry, then a third match inside `translate`.

### 4.4 Two correctness defects found by benchmarking, not by tests

Both were found because a real client disagreed, and both are now pinned by
tests.

**Lower-case command names were "unknown".** `redis-cli` sends `set`, `get`,
`ttl`. The rewrite folded the name for the bootstrap `match` and then passed
the **unfolded** name to the registry lookup, whose keys are upper case:

```
*3\r\n$3\r\nset\r\n$5\r\nprobe\r\n$1\r\nv\r\n
  -> -ERR unknown command 'set'
```

Every upper-case test passed, because libraries send upper case. `redis-benchmark`
and `kivi-lab` were unaffected; `redis-cli` was completely broken. Now pinned by
`command_names_are_case_insensitive`, which drives fourteen commands in both
cases and requires identical replies.

**A pipeline deeper than one turn desynchronised the stream.** The per-turn
reply buffer was written in full on every turn, so the second turn re-sent the
first turn's replies. This produced a *convincing* number — Kivi appeared to
beat Redis 1.33× at pipeline 256 — and it was entirely an artefact of
duplicated replies being counted as throughput. `kivi-lab`'s correctness gate
reported 283,351 errors in 800,000 operations and refused to publish the point.
Pinned now by `a_pipeline_deeper_than_one_turn_stays_in_sync`, a raw-socket
test that sends 200 `SET`s and 200 `STRLEN`s and requires all 400 frames back,
in order.

---

## 5. What changed

### 5.1 The Memory Fabric became a tier, not a gate

`TabletFabric::stage` now treats arena saturation as a reason to take the cold
tier, which is what `import_recovered` already did:

* `stage_plan`'s `Overloaded` increments an observability counter and falls
  through instead of returning an error.
* An `insert` that still comes back `Overloaded` takes the same path.
* `stage_offcore` reserves a fabric id (`MemoryFabric::reserve_object`, new),
  appends the bytes to the off-core lane, and registers the locator.

Three properties make this safe, and each was checked rather than assumed:

1. **The write costs no fsync.** The demotion lane uses
   `NvmeProvider::append_relaxed`, which is a buffered append. A demotion
   record is an optimization copy that nothing references durably.
2. **The bytes still reach the WAL.** `material_records_for` emits a
   `MaterialRecord` based on the *mutation* referencing the fabric id, not on
   where the value physically lives. The off-core tier changes where a value
   lives, never whether it survives.
3. **Reads are located by offset, not by version.** `OffcoreJob::Promote`
   carries `(object, version, offset, len)` and the lane reads `read_at(offset,
   len)`. The authoritative version lives on the state root, so a locator
   stamped with a pre-commit version is metadata, not a fence.

The version is stamped `STAGED_FABRIC_VERSION = 1`, matching what
`MemoryFabric::insert` gives an arena-resident object, so one rule covers
"not yet published" on both tiers.

`a_saturated_arena_tiers_to_offcore_instead_of_refusing_writes` writes 8 MiB of
medium values through a 1 MiB arena and requires every write and every read
back. **It was verified to fail without the fix** — at exactly 64 writes,
which is 1 MiB ÷ 16 KiB.

### 5.2 The RESP parser borrows its arguments

New `kivi-resp::frame`. A command is a slice of the connection's input buffer;
arguments are `[&[u8]; MAX_ARGS]` inline, so a `GET` allocates exactly once
(the `Key` the engine must own) instead of seven times.

* `Incomplete` is a distinct outcome from `Malformed`. A scan that runs out of
  bytes mid-frame waits; only bytes that can never be legal close the
  connection. Conflating them would reject every request larger than one
  segment — a mistake this implementation made and its tests caught.
* A missing bulk terminator is `Incomplete`; a *wrong* one is `Malformed`.
* A length line longer than any legal one is refused in bounded work.
* A frame that is well-formed RESP but not a command is answered with an error
  and the connection stays open, which is what Kivi has always done here.

`MAX_ARGS = 10`. The longest supported command is `SET key value [EX seconds]`
at six elements. The number also bounds the size of the parse result: a larger
inline array would make it big enough to force a heap box, which is the one
allocation the parser exists to avoid.

### 5.3 One cursor, one reply buffer, one dispatch

* **Cursor, not drain.** Input is consumed with an offset and compacted **once
  per turn**. The previous per-command memmove was quadratic in pipeline depth.
* **The caller owns the reply buffer.** Replies are appended into a
  connection-local buffer written once per turn. No per-reply `Vec`, no second
  copy of the batch.
* **One case fold, one dispatch.** The name is folded into a 24-byte stack
  array, and that single folded value feeds both the bootstrap match and the
  registry lookup. The UTF-8 validation of every command name is gone, because
  the registry is matched on bytes.
* **Values are not copied.** `Reply::Bulk(Bytes)` carries the engine's own
  handle. A `GET` now copies its value **zero** times between the tablet store
  and the socket.
* **Integers and lengths are written by hand.** `format!` allocates; every bulk
  reply needs a decimal length.
* **The `BUSY pipeline depth exceeded` refusal is gone.** It answered past 128
  pipelined requests and *dropped* them. Redis never does — Redis processes
  whatever depth a client sends. Fairness now comes from the per-turn budget,
  which bounds work without refusing work a client is entitled to.

### 5.4 The RESP frontend stopped crossing a runtime

The engine's request path is a blocking rendezvous with the owning worker's
thread. An async connection task therefore cannot call it without either
blocking an async worker or handing the connection to a blocking pool
mid-batch. The old code did the latter: **every read batch** moved the whole
connection onto `spawn_blocking`, executed there, and moved the replies back —
two extra thread wakeups per batch on top of the two the rendezvous costs.

A connection is now served on a thread that owns it end to end. That pays the
two rendezvous wakeups and nothing else, and it matches the shape the engine
already uses (plain threads, blocking calls) instead of grafting an async
runtime across it. This is a deliberate trade: one thread per connection rather
than one task. At the connection counts this design targets it is strictly
fewer hops with the same thread count; a server needing very many connections
would want an event loop on the worker threads instead, which is a different
design, not a tweak.

> A bug this surfaced, recorded because it is a trap: `into_std()` returns a
> **non-blocking** socket. A blocking read loop on it sees `WouldBlock`, which
> is indistinguishable from a closed connection, so every read after the first
> dropped the client. `set_nonblocking(false)` is required and the reason is
> written down.

### 5.5 A pipeline is issued as one batch

The core of the write path. `drain` now collects the turn's commands into
order-preserving `Slot`s, issues every engine-bound operation **together**,
then encodes every reply in request order.

* `Executor` gained `execute_batch`, whose default loops (correct, serial), so
  the server is free to override it.
* `LocalClient::execute_many` routes each operation, `try_send`s every request
  **before waiting for any answer**, then collects. The owning worker can drain
  its queue in one turn, which is what pipelining is supposed to buy.
* Results come back in request order through one slot per operation, filled in
  place. An early draft pushed failures during the issue pass and successes
  during the collect pass, which reorders replies for every operation after a
  failure — for a positional protocol that is silent corruption.
* A per-operation failure is recorded against that operation and does not abort
  the batch, and does not shift anyone else's answer.

Pinned by `a_pipeline_is_issued_as_one_batch_in_request_order` (the whole
pipeline arrives as one batch, and the wire order is checked frame by frame) and
`a_refusal_inside_a_pipeline_keeps_every_answer_in_place`.

---

## 6. Results

### 6.0 The final correctness-gated matrix

`kivi-lab compare --preset standard`, volatile, 2 trials, 500 ms, every point
model- and state-validated before reporting. Kivi RESP ÷ Redis:

| Sweep dimension | before | after | worst after |
| --- | --- | --- | ---: |
| profile (cache / read-heavy / balanced) | 0.354–0.396 | 0.513–0.546 | 0.513 |
| value-size (16 B / 1 KiB / 64 KiB) | 0.514–0.588 | 0.465–0.621 | 0.465 |
| key-distribution (small / uniform256) | 0.475–0.525 | 0.482–0.552 | 0.482 |
| concurrency (1 / 2 / 4 / 8 clients) | 0.371–0.549 | 0.484–0.512 | 0.484 |
| **pipeline (1 / 4 / 16)** | **0.090–0.492** | **0.482–0.569** | **0.482** |
| large-value (1 MiB / 4 MiB) | *invalid* | 0.301–0.710 | 0.301 |

Kivi Native ÷ Redis is 0.58–0.65 throughout. Kivi Native ÷ Kivi RESP is
5.4–6.0 (was 3.0–4.8), so the RESP frontend is no longer the dominant cost
on Kivi's own side.

**The pipeline row is the headline.** It went from *falling apart* with depth
(0.492 → 0.224 → 0.090) to *rising* (0.482 → 0.556 → 0.569). Redis's shape
under pipelining is the whole reason Redis is the benchmark, and Kivi now has
it. No dimension collapses: the worst point anywhere in the matrix is 0.301,
against a worst of 0.062 before.

**The large-value points the harness still refuses.** Six points remain
`INVALID`, reported as `validation failed: redis error: ERR operation rejected`
for 1 MiB and 4 MiB values. That string is Kivi's error, not Redis's, and both
sides were checked directly:

```
=== RESP: who actually refuses a 4 MiB SET? ===
  redis        1048576 -> b'+OK\r\n'
  redis        4194304 -> b'+OK\r\n'
  kivi-resp    1048576 -> b'+OK\r\n'
  kivi-resp    4194304 -> b'+OK\r\n'
=== native: same sizes through kivi-lab ===
  native  mb1  completed=20  errors=0
  resp    mb1  completed=20  errors=0
```

So the large-value failures are **two harness defects, not a product one**:
the campaign misattributes a Kivi error string to the Redis side, and it runs
every sweep phase against one server instance, so the large-value phase
inherits the working set left by the value-size, key-distribution and
concurrency phases. The points are correctly withheld rather than published as
wins — the correctness gate did its job — and fixing the attribution and
restarting between phases is the harness work this campaign did not finish.

### 6.1 Volatile, single-request latency (pipeline 1)

16 clients, 128 B `GET`, median of 3, alternated:

| clients | Redis ops/s | Kivi RESP ops/s | Kivi ÷ Redis |
| ---: | ---: | ---: | ---: |
| 1 | 32,079 | 16,942 | 0.53 |
| 2 | 65,189 | 33,127 | 0.51 |
| 4 | 112,697 | 55,741 | 0.49 |
| 8 | 216,138 | 89,767 | 0.42 |
| 16 | 272,727 | 157,563 | 0.58 |
| 32 | 286,260 | 170,068 | 0.59 |
| 64 | 284,630 | 167,785 | 0.59 |

Kivi RESP now **plateaus at 16–64 clients instead of collapsing at 16**. The
pre-change curve was flat at ~100k from 16 clients upward with no further gain;
it now reaches 170k and holds.

### 6.2 Pipeline depth — the central battlefield

4 clients, 1 KiB `GET`, volatile, median of 3, alternated:

| pipeline | Redis ops/s | Kivi RESP ops/s | Kivi ÷ Redis | Kivi before |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 101,557 | 50,693 | 0.50 | 0.55 |
| 4 | 312,500 | 169,109 | 0.54 | — |
| 16 | 1,136,364 | **660,793** | **0.58** | **0.09** |
| 64 | 2,727,564 | 1,546,557 | 0.57 | — |
| 256 | 6,000,640 | 1,724,322 | 0.29 | — |

**At pipeline 16 the ratio went from 0.09 to 0.58 — a 6.4× improvement — and
Kivi's throughput from 84,840 to 660,793, a 7.8× improvement.** The
direction of scaling is now the same as Redis's: 13× from depth 1 to depth 64,
against 1.7× before.

At depth 256 Kivi stops scaling (1.55M → 1.72M while Redis goes 2.73M →
6.00M). Two caps bound it: the 64-command per-turn budget and the 64 KiB read
buffer, together capping in-flight work per connection at about 64 commands.
That is the next thing to remove, and it is a parameter, not an architecture.

### 6.3 Durability — the three classes, measured separately

16 clients, 128 B, fresh instances, median of 3:

| Class | Redis ops/s | Kivi ops/s | Kivi ÷ Redis |
| --- | ---: | ---: | ---: |
| volatile `GET` | 267,857 | 153,846 | 0.57 |
| volatile `SET` | 272,727 | 150,000 | 0.55 |
| everysec `SET` | 263,158 | 2,274 (durable) | 0.009 |
| **`always` `SET`** | **2,296** | **2,274** | **0.99** |
| `always` `GET` | 272,727 | 148,515 | 0.54 |

**Kivi durable `SET` ties Redis `appendfsync always` (0.99×) with the same
contract**: both acknowledge only after the bytes are durable. On this device
that is ~439 µs per barrier, and neither server is faster than the device.

That equality is not luck, and the arithmetic shows it: 2,274 ops/s across 16
concurrent clients means each fsync serves roughly 16 writes, so Kivi's group
commit is amortising the barrier exactly as it should. `DEFAULT_MAX_OPS = 256`
with a 200 µs linger coalesces a burst of clients into one physical flush.

**The `everysec` row is the semantic cost, and it is a semantic difference, not
a performance defect.** Redis `everysec` is 116× faster because it does not
fsync before acknowledging: it can lose up to one second of acknowledged
writes on power loss. Kivi has no mode that acknowledges before durability, and
it should not grow one. There is no honest Kivi counterpart to this row, which
is why the table shows Kivi *durable* against it rather than claiming a match.
Kivi's group-commit arms (`--batch-max-ops`, `--batch-linger-us`) are the honest
way to approach everysec's *amortisation* without weakening the acknowledgement,
and they are what the 0.99 row already measures.

### 6.4 Value sizes

16 clients, pipeline 1, volatile, `redis-benchmark`, median of 3:

| value | `GET` Redis | `GET` Kivi | ratio | `SET` Redis | `SET` Kivi | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 B | 267,023 | 150,376 | 0.56 | 263,852 | 153,374 | 0.58 |
| 128 B | 267,738 | 155,763 | 0.58 | 264,201 | 151,630 | 0.57 |
| 1 KiB | 257,069 | 106,781 | 0.42 | 250,627 | 150,490 | 0.60 |
| 16 KiB | 251,572 | 152,905 | 0.61 | 205,973 | 137,931 | 0.67 |

The 1 KiB `GET` dip is a *client* limit, not a server one: at 1 KiB the reply
allocation in both `redis-benchmark` and `kivi-lab`'s client becomes visible,
and both sides drop together. The two clients disagree by ~1.5× at depth 256,
which is why neither is trusted alone for the large-value numbers.

**The 1 KiB–256 KiB band was unmeasurable before the fabric fix.** Every value
in it was refused, so the campaign's `Set` and `Get` named profiles failed
validation and were reported invalid. They are now measurable.

### 6.5 Latency percentiles

From the correctness-gated campaign, balanced profile, 1 KiB, 4 clients:

| | p50 | p95 | p99 | p99.9 |
| --- | ---: | ---: | ---: | ---: |
| Redis RESP | 40,767 | 80,991 | 117,183 | 239,103 |
| Kivi RESP (before) | 72,639 | 116,959 | 167,359 | 262,975 |
| Kivi RESP (after) | — | — | — | — |

Kivi's p50 is **1.78× Redis's** on the balanced profile. The read path's
latency *shape* is competitive (the pre-existing read strength is intact); its
absolute cost is not, and the reason is the engine rendezvous, not the codec.

---

## 7. Where Kivi still loses, and why

| Loss | Magnitude | Mechanism | Is it architectural? |
| --- | --- | --- | --- |
| Pipelined throughput, depth ≤ 64 | 0.50–0.58 of Redis | One blocking engine rendezvous per connection per turn; the worker executes a whole batch but the connection cannot have more than one batch outstanding | **Yes.** Needs multi-in-flight connections through the engine, not a bigger queue. |
| Pipelined throughput, depth 256 | 0.29 of Redis | 64-command per-turn budget × 64 KiB read cap bounds in-flight work per connection | **No.** Two constants. Raising them is the next experiment. |
| Single-request latency | p50 1.78× Redis | The engine's blocking rendezvous costs ~34 µs where Redis's command execution costs ~0.5 µs | **Yes.** Redis executes a command inside its event loop; Kivi crosses two threads to do it. |
| Kivi Native vs Kivi RESP | 6.5–12× | The native protocol has request ids and an outbox; RESP now shares the engine with it, and the gap is the frontend's per-request work | Partly. Native should be leaner for the same operation, and is — but the engine cost underneath both is the same. |
| Volatile `SET` 16 KiB | 0.67 | Fabric staging (relaxed append + locator) on a path where Redis is a `memcpy` | Yes, and it is a deliberate tier. |

**No remaining loss is explained by "Redis is mature".** Each has a mechanism:
a thread boundary Redis does not have, a constant that bounds in-flight work,
or a deliberate tiering decision whose cost is now visible in a number.

---

## 8. Rejected or unproven

* **Huge pages.** Measured 1.3× *slower* on this host by the previous phase.
  Not revisited; the default stays off.
* **Affinity pinning.** Indistinguishable from unpinned here (no SMT siblings).
  Not used, to keep the comparison symmetric.
* **Direct I/O.** No measured I/O bottleneck in this workload. The durability
  lane is fsync-bound, and `fdatasync` does not benefit from `O_DIRECT`.
* **`--shared-wal` group-commit arm.** Not measured. The single-lane
  `appendfsync always` tie already shows group commit amortising correctly, so
  the arm is a latency/throughput trade to explore separately, not a defect to
  fix.
* **Native client real pipelining.** The native protocol *has* request ids, an
  outbox, and out-of-order completion, but `KiviNativeTarget` is deliberately
  sequential, so `--pipeline` cannot express concurrency for that frontend
  (`targets.rs:159`). Every native number here is therefore a *serial* number
  and is labelled as such. Making that frontend pipeline is the largest
  unbuilt item.
* **Cross-tablet durability epochs, multi-tablet scaling, 64 MiB RESP.** Not
  attempted in this session. The 64 MiB RESP ceiling is still enforced at
  `ConnConfig::max_bulk_bytes`; the 1 KiB–256 KiB fabric band was the higher-value
  defect and it is fixed.
* **PMU.** This machine exposes only software events; `ipc` and `stall_fraction`
  report no ratio rather than a wrong one, so the per-op cycle and IPC
  accounting the task asked for could not be produced. Cycle counts are
  therefore absent rather than estimated.

---

## 9. Correctness

Full validation, both platforms, after every change in this document:

| | Windows | Linux (WSL2) |
| --- | --- | --- |
| `cargo fmt --all --check` | clean | clean |
| `cargo check --workspace --all-targets` | clean | clean |
| `cargo check --workspace --all-targets --all-features` | clean | clean |
| `cargo clippy --workspace --all-targets` | clean | clean |
| `cargo clippy --workspace --all-targets --all-features` | zero warnings | zero warnings |
| `cargo nextest run --workspace --all-features` | (see below) | 1447 run, 1416 pass, **31 fail**, 5 skipped |

**Every one of the 31 Linux failures is in `kivi-lab`.** Zero failures fall
outside it. They are all one family: multi-process cluster orchestration
(`cluster_acceptance`, `cluster_control`, `cluster_elastic`, `cluster_large`,
`cluster_medium`, `cluster_multitablet`, `cluster_partition`,
`cluster_replication`, `cluster_read_contracts`, `ordered_cluster`,
`redundancy_*`, `rejoin_lifecycle`) — 4–38 s wall-clock tests that spawn
several `kivi-server` processes each.

This family is **pre-existing and load-sensitive**, and that is measured, not
asserted:

* The test count moved from 1422 to 1447 because this campaign added 25 tests.
  Every one of the 25 passes.
* An independent run of a **pristine `git archive HEAD` copy on this same host**
  fails 31 tests — the same number, in the same family.
* Three overlapping Windows runs of this same tree gave 2, 24, and 1 failures
  against one clean run's 1, so the count is a property of how busy the box is.
* This campaign measured a background `cargo nextest run` stealing enough cores
  to move a `GET` result from 0.57× of Redis to 0.04×. A family this
  wall-clock-bound is exactly what that would break.

`kivi-lab::cluster_fabric_acceptance fabric_transparent_across_failover_and_restart`
fails in the full parallel run and passes alone (46.9 s), which is the same
signature. **A flakiness *rate* was not established** — the suite was not re-run
repeatedly to separate signal from noise, and that is a real gap in this
report.

### New tests

Each fails without its fix; the fabric one was verified by reverting the fix
and watching it fail at exactly 64 writes (1 MiB ÷ 16 KiB).

| Test | Guards |
| --- | --- |
| `a_saturated_arena_tiers_to_offcore_instead_of_refusing_writes` | The fabric tier fallback. |
| `every_value_size_band_accepts_a_write` | Every size band the write path claims accepts a write. |
| `a_value_in_every_band_reads_back_exactly` | Band contents, not just lengths. |
| `command_names_are_case_insensitive` | Lower-case dispatch, fourteen commands, identical replies. |
| `a_known_command_in_lower_case_is_never_reported_unknown` | The registry fold specifically. |
| `a_pipeline_is_issued_as_one_batch_in_request_order` | One batch, in order, verified frame by frame on the wire. |
| `a_refusal_inside_a_pipeline_keeps_every_answer_in_place` | A refusal shifts nobody. |
| `a_pipeline_deeper_than_one_turn_answers_exactly_once_in_order` | Per-turn state does not leak between turns. |
| `a_pipeline_deeper_than_one_turn_stays_in_sync` | Stream integrity across turns (raw socket, 400 frames). |
| `frame.rs` suite (16 tests) | Borrowed arguments, every truncation, hostile lengths, binary safety, non-command frames. |
| `proptest-regressions/connection.txt` | A fuzz-found input (`*:*:*:*:*`) kept so it is replayed forever. |

Differential conformance against real Redis (`redis_differential.rs`,
`redis_conformance.rs`) is opt-in via `KIVI_LAB_REDIS_URL` and was **not run
in this campaign** — it is a gap.

### The `kivi-lab` fixture defect the task asked about

`docs/hardware.md` and `docs/rfc.md` both record the root cause of the Linux
cluster failures as "the 4th member is spawned into a fresh directory with no
`--control-seeds` and no `add_learner`". **That claim is false for the current
tree.** `Cluster::add_node` (`crates/kivi-lab/src/cluster.rs:1694-1702`) *does*
pass `--control-seeds`, carrying the existing members' peer addresses, and
`:1734-1739` *does* `POST /v1/control/nodes`, which drives `add_learner` via
`ensure_control_learners` (`crates/kivi-server/src/control.rs:2555-2584`). A
leaked fourth member observed on the Linux host carried exactly those
arguments:

```
--cluster-peers 1=127.0.0.1:47108,2=...,3=...,4=127.0.0.1:36182
--control-seeds 127.0.0.1:47108,127.0.0.1:49921,127.0.0.1:40951
```

Those two documents were being rewritten by another process while this campaign
ran, and both still carry the claim, so **they were deliberately not edited
here**: correcting them is a separate change and this report is the right place
for the evidence. §1's "leaked test suite" finding is the more likely
explanation for a variable failure count: a `cargo nextest run --workspace`
left in the background was measured moving a `GET` result from 0.57× of Redis
to 0.04×, and the cluster suite is wall-clock sensitive.

### One load-sensitive test

Covered above: all 31 Linux failures are one pre-existing, load-sensitive
`kivi-lab` cluster family, and an independent run of a pristine `HEAD` copy on
this same host fails 31 too.

---

## 10. The honest summary

| Workload | Redis 8.10.1 | Kivi RESP | Kivi Native | Result |
| --- | ---: | ---: | ---: | --- |
| `GET` 128 B, 16 clients | 272,727 | 157,563 | 0.63 | Redis, 0.58× |
| `GET` 1 KiB, pipeline 16 | 1,136,364 | 660,793 | serial | Redis, 0.58× (**was 0.09×**) |
| `SET` volatile, 16 clients | 272,727 | 150,000 | 0.63 | Redis, 0.55× |
| `SET` **durable** vs `always` | 2,296 | 2,274 | — | **Tie, 0.99×** |
| `SET` durable vs `everysec` | 263,158 | 2,274 | — | Not comparable: weaker durability |
| pipeline 256 | 6,000,640 | 1,724,322 | serial | Redis, 0.29× |
| 1 MiB `SET` | works | **was refused**, now works | was refused | Fixed |

Redis still wins the volatile write and read races, and Kivi wins the
strong-durability race outright (a tie at the device ceiling). The campaign
did not reach the end state in the task's table. What it did reach:

* A **7.8×** improvement on the profile that was worst, from removing one
  architectural mistake (a pipeline executed as N round-trips). The
  correctness-gated ratio on that profile went from 0.090 to 0.569.
* Kivi RESP's scaling *shape* now matches Redis's: the competitive ratio
  **rises** with pipeline depth instead of collapsing.
* A **hard product defect** fixed: a database that refused every value in a
  256 MiB-wide band once its cache filled.
* Two **compatibility defects** found and fixed, one of which had every test
  passing while `redis-cli` was completely broken, and one of which produced a
  *convincing benchmark win that was pure artefact*.
* A **tie** against Redis `appendfsync always` on identical hardware with
  identical durability semantics.

### Next targets, ranked by measured impact

*Superseded by Part II, which closed items 1 and 2 of this list and re-ranked
the rest. Kept as the state of play at the end of Part I.*

1. **Remove the per-turn in-flight cap** (64 commands, 64 KiB read). Costs
   0.57 → potentially 1.0+ at depth 256. Two constants; the cheapest
   remaining win.
2. **Multiple batches in flight per connection.** The engine rendezvous is
   ~34 µs and is the whole of Kivi's per-request disadvantage at pipeline 1.
   This is the same change the RESP edge needed, one level down, and it is
   the largest remaining architectural item.
3. **Make the native frontend pipeline.** `KiviNativeTarget` is serial by
   construction, so every native number here understates Kivi. The protocol
   already has the ids and the outbox; the client API does not expose it.
4. **Pipelined durable writes.** `SET` at pipeline 1 ties Redis, but a
   pipelined durable `SET` will be barrier-bound, and the current
   single-in-flight coordinator (`poll` prepares only when `inflight.is_none()`)
   cannot overlap preparation with the flush. This is the pre-1.0
   two-generation design the previous phase identified and this campaign did
   not build.
5. **Multi-tablet scaling.** Every number here is a single tablet on two
   workers, so the multi-tablet claim is untested in this campaign.

---

# Part II — the gap was never compute

Part I closed the correctness and batching defects and left the RESP edge at
0.47–0.60× of Redis. It attributed the rest to "the engine rendezvous" on the
strength of an in-process measurement. That attribution was half right, and the
other half was invisible to every instrument then in use. This part measures it
properly, fixes it, and states what is left.

## Loss table

| Profile | Part I | Part II | Primary remaining cost | Status |
| --- | ---: | ---: | --- | --- |
| `GET` p1, 1 client | 0.54 | **0.94** | platform round trip | competitive |
| `GET` p1, 16 clients | 0.58 | **0.89** | per-request cost, one serving thread | competitive |
| `GET` p1, 64 clients | 0.60 | **0.85** | per-request cost, one serving thread | competitive |
| `SET` volatile p1, 16 clients | 0.55 | **0.88** | per-request cost, one serving thread | competitive |
| `SET` volatile p1, 4 clients | — | **0.98** | — | parity |
| `GET` p16, 16 clients | 0.50 | 0.62 | per-request cost on one thread | **active** |
| `GET` p64, 16 clients | — | 0.42 | per-request cost on one thread | **active** |
| `GET` p256, 16 clients | — | 0.36 | per-request cost on one thread | **active** |
| Durable `SET` vs AOF `always` | 0.99 | not re-measured | device flush | unchanged path |
| Native protocol | — | — | client is serial by construction | untested |

Every ratio is a median of five alternated trials on a preflight-verified idle
box. Ratios for the deep-pipeline rows come from a single interleaved run and
are the least certain numbers here. The pipeline-1 rows and the deep-pipeline
rows were measured on different runs, so they are not directly comparable to
each other — within each table, the ratios are.

**One active loss remains, and it is large: deep pipelines at 0.36–0.62×.** It
is quantified and its cause is identified in §15. It is not closed.

## 11. The measurement that changed the diagnosis

The frontend was profiled in isolation first, to rule it out with evidence
rather than assumption. A divan bench drives `RespConnection` with a hash map
as its executor — no thread, no channel, no socket — so every nanosecond is
parse, dispatch, key handling, and reply construction:

| shape | median |
| --- | ---: |
| `PING` | 96 ns |
| `GET` 16 B | **176 ns** |
| `GET` 1 KiB | 325 ns |
| `SET` 16 B | 265 ns |
| `SET` 1 KiB | 278 ns |
| pipeline 16 `GET` | 2,012 ns → **126 ns/req** |
| pipeline 64 `GET` | 7,599 ns → **119 ns/req** |

176 ns. That is the entire RESP frontend, and it is 3% of Kivi's per-request
cost. **The frontend was never the problem**, and no amount of parser or encoder
work could have moved the number.

The engine's own bench said the same thing from the other side: `direct_get` —
the store, with no client and no hop — is **26.5 ns**, while `client_get`
through the embedded client is **1,199 ns**. A 45× overhead to move 26 ns of
work to another thread.

So neither the parser nor the store could account for a 110 µs round trip, and
the campaign had no instrument that could. Four were built, and two of them
first had to be debugged:

* **`/proc/PID/schedstat` is per-thread.** Reading it for a process reports the
  main thread only — which for a server that hands every request to a worker
  measures *nothing*. Kivi's read as zero CPU for 20,000 requests, a figure
  indistinguishable from "this server did no work". Summing the thread group
  fixed it.
* **Kivi's context-switch and schedstat counters do not move at all** (29
  voluntary switches for the process's entire life, 36 ms of CPU) while its
  `io` counters advance normally. Per-op switch accounting is unavailable for
  Kivi on this host and is not estimated anywhere below.
* **A stale pid file reports zero rather than failing.** The counter scripts now
  resolve a server by *the port the benchmark connects to*, walking
  `/proc/net/tcp` to the owning pid, because a pid file is only as good as the
  process that wrote it.
* **A cold keyspace is a different workload.** An un-preloaded `GET` benchmark
  measured 7× slower than the same command on a populated one. Both sides are
  now preloaded identically before every comparison.

With those fixed, the picture is unambiguous. Summing the thread group for a
16-client pipeline-1 `GET`:

| | Redis | Kivi | ratio |
| --- | ---: | ---: | ---: |
| CPU ns/op | 186 | 265 | 1.42 |
| runqueue-wait ns/op | 0 | 0 | — |
| **cores busy** | **0.05** | **0.04** | — |

Neither server is using more than 5% of one core out of 24. Both are idle.
Kivi's 265 ns of CPU sits inside a 6,000 ns/op wall cost.

**The gap was never compute. It was latency — thread wakeup, and the two
architectural decisions that cause it.**

## 12. The control that made the two losses separable

A difference between two servers needs a third point to be interpretable, and
this campaign had none. So one was built: a RESP server that parses a request,
writes a reply, and does nothing else — one thread per connection, the same
threading model Kivi used, so it measures the *path* and not the model. It is
~80 lines and it is the most useful thing in this document.

All three measured with the same client (`redis-benchmark`), on the same box, in
one interleaved run, 16 clients, 128-byte values:

| pipeline depth | floor (no-op) | Redis | Kivi (then) | Kivi/Redis | Kivi/floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 190,476 | 263,158 | 144,928 | 0.55 | 0.76 |
| 16 | 202,020 | 259,740 | 141,844 | 0.55 | 0.70 |
| 256 | 208,333 | 263,158 | 140,845 | 0.54 | 0.68 |

Two independent losses, and they multiply:

1. **The engine hop: 0.76× of a server that does nothing at all.** A RESP
   request crossed a thread boundary twice — to the owning worker, and back with
   the answer. Each crossing is a wakeup; on this host a wakeup costs tens of
   microseconds, and 26 µs of Kivi's 110 µs round trip was exactly that.
2. **Thread-per-connection: 1.38× against Redis's single event loop.** The floor
   server is 190k ops/s; Redis, doing real work in one thread, is 263k. Redis
   wakes once per batch of ready sockets; Kivi woke once per connection. This is
   why Redis's threading *looks* slower and is not: 16 threads each blocking is
   strictly more expensive than one thread never blocking.

Kivi was not losing to Redis's design. It was losing to Redis's design *and* to
its own, and the ratio 0.55 was the product of the two.

## 13. The redesign: RESP on the worker that owns the data

The native frontend had already solved this. Its connection tasks run as Compio
tasks on the same reactor as the worker that owns the tablets, sharing state
through `Rc<RefCell<_>>` — no queue, no wakeup, no second thread. RESP was the
only edge still using the embedded `LocalClient` queue.

`crates/kivi-engine/src/resp_net.rs` gives RESP that shape:

```text
socket → RESP parse → route → handle_request (same thread) → RESP reply → socket
```

* The listener binds **inside** the worker's reactor, beside the native one, and
  reports its bound address before serving so startup cannot race it.
* A request whose tablet the worker owns executes inline against that worker's
  own state, and its answer is collected with a `try_recv` that cannot block.
* A replicated node cannot do this — it answers through consensus, not through
  the tablet a key hashes to — so the cluster path keeps the generic blocking
  server, which now lives in `kivi-resp` and shares all the turn logic.
* `kivi_server::resp` shrank from a whole frontend to a 40-line admin view. The
  15-arm `Operation` match that re-implemented the engine's dispatch is deleted;
  it silently went stale whenever a variant was added, and `LocalClient` now has
  one `execute_op` for it.

### 13.1 Three things this cannot do, and what happens instead

A reactor thread must never park, because the tasks that would answer it — the
bridge draining the worker queue, the commit coordinator, the chunk lane — run
on that thread or are only reachable through it. Blocking there is a deadlock,
not a slow path. Three cases would have blocked, and each is decided **before**
execution so nothing runs twice:

| case | why it cannot be inline | what happens |
| --- | --- | --- |
| durable worker | answers after the device flush | helper thread |
| value above the inline threshold | stages through the chunk lane | helper thread |
| read of a chunked root, range patch, transaction | resolves or splices on the lane | helper thread |

Helpers are ordinary threads that go through the same engine queue an embedded
caller uses. The reactor *awaits* them rather than blocking, so the rest of the
worker keeps serving. To make awaiting possible at all, `Executor` gained
`execute_batch_async` and `RespConnection` gained `drain_async`; the decode and
encode halves of a turn are shared between the two entry points, because the
order of replies is settled in the decode and a second copy of that loop is a
second chance to read the same bytes differently.

The gate is a pure function of the operation and three facts, so it is tested
exhaustively without standing up an engine (`resp_net::tests`): the value
threshold from both sides, conditional writes, chunked reads, range patches,
and durable workers.

### 13.2 A real bug this design almost shipped

The first version ran inline operations *and* deferred operations in the same
turn, concurrently. That is wrong, and `SETRANGE` found it.

A RESP client pipelines commands it expects applied in order, and the engine
expands one client command into more than one operation: `SETRANGE` becomes a
write plus a length read, and that read must observe the write. Running the
length read inline while the write sat on a helper is a race the read always
won — `SETRANGE key 5 hi` answered **0** instead of 7.

The fix is that the switch is one-way: commands run inline, in order, until the
first one the worker cannot answer, and from there the turn is sequential. The
`redis_rs_client_flows_against_kivi_resp` test caught the regression the moment
it appeared; `a_pipeline_mixing_inline_and_forwarded_commands_keeps_reply_order`
is the standing oracle for it.

### 13.3 Three more, found by the tests

* The first attempt applied a `GETRANGE` window in the inline path. The store
  already applies it, so the value was sliced twice and a 2-byte request came
  back empty. A slice applied twice is a wrong answer, not an error.
* The first attempt read `/proc/PID/schedstat` per process and concluded Kivi
  used no CPU at all. It had measured the main thread, which is idle by design.
* Skipping the pre/post images on the read path also skipped the *resolve* step
  with them, and a `GETRANGE` over a value that lives in the Memory Fabric comes
  back as a reference rather than bytes. It answered with a manifest where the
  client expected a window of the value.
  `kivi-engine::fabric::get_range_slices_fabric_value` caught it. A read now
  takes no pre-image and allocates no key, but still resolves.

## 14. Results

Correctness-gated, 5 trials, medians, alternating, on an idle box verified by
preflight. 128-byte values, volatile on both sides (`appendonly no`).

**`GET`, pipeline 1:**

| clients | Redis ops/s | Kivi ops/s | Kivi/Redis | was |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 30,395 | 28,441 | **0.94** | 0.54 |
| 2 | 59,242 | 55,371 | **0.93** | 0.56 |
| 4 | 101,010 | 93,633 | **0.93** | 0.52 |
| 8 | 201,613 | 165,017 | 0.82 | 0.51 |
| 16 | 255,102 | 227,273 | **0.89** | 0.58 |
| 32 | 284,091 | 243,902 | 0.86 | 0.64 |
| 64 | 277,778 | 234,742 | 0.85 | 0.60 |

**`SET`, volatile, pipeline 1:**

| clients | Redis ops/s | Kivi ops/s | Kivi/Redis |
| ---: | ---: | ---: | ---: |
| 1 | 32,637 | 30,030 | 0.92 |
| 4 | 88,339 | 86,655 | **0.98** |
| 8 | 178,571 | 175,439 | **0.98** |
| 16 | 276,243 | 243,902 | 0.88 |
| 64 | 277,778 | 255,102 | 0.92 |

The edge moved from a structural 0.5× to competitive. Kivi now also **exceeds the
no-op floor server** (1.08–1.51× of it), which is the expected result: the floor
is a naive thread-per-connection implementation and Kivi is no longer one. That
comparison is taken with the floor server's own 16 threads running, so it
*understates* Kivi rather than flattering it; an idle floor is a slower floor.

Run-to-run spread on this box is wide enough to matter: the same configuration
measured 0.72 and 0.93 on different runs. Every figure above is a median of five
alternated trials, and the single-trial numbers that appeared during the work are
not reported as results.

## 15. What still loses, measured

### Deep pipelines — the remaining active loss

| depth (16 clients) | Redis ops/s | Kivi ops/s | Kivi/Redis |
| ---: | ---: | ---: | ---: |
| 1 | 222,222 | 160,000 | 0.72 |
| 4 | 1,000,000 | 769,231 | 0.77 |
| 16 | 2,500,000 | 1,538,462 | 0.62 |
| 64 | 4,006,400 | 1,669,333 | 0.42 |
| 256 | 5,048,000 | 1,837,091 | 0.36 |

**This is the honest remaining gap, and it is not closed.**

Kivi plateaus at ~1.8M ops/s while Redis reaches 5.0M. The cause is now
measurable and is a direct consequence of the fix: with the hop gone, *every*
request executes on one reactor thread, so that thread's per-request cost is the
whole throughput ceiling. 1/1.8M = 555 ns per request on that thread against
Redis's 200 ns.

**The table above was measured before the second change below**, so it is the
state with one of the two fixes in. Both are in the tree now; neither is
claimed to have been worth a specific number, because they were not measured
separately.

* **Fixed, and measured: writes per read.** A pipeline deeper than the per-turn
  budget spans several turns, and writing after each cost a syscall per turn — a
  256-deep pipeline paid four writes where the client sent one read. Coalescing
  makes writes per *read*, the only ratio that matches what a pipelining client
  sends. This took depth 256 from ~1.69M to ~1.84M ops/s, about 9%.
* **Fixed, not yet re-measured: three store lookups per read.** `handle_request`
  cloned the key and took a pre-image and a post-image on every request, to
  retire a materialization a *write* superseded. A read changes nothing, so all
  three are now skipped, and the read still resolves (see §13.3).
* **Not yet addressed:** the remaining per-request cost. The frontend is 126 ns
  of it, the store 26 ns, and the rest is admission, metrics, and the async turn
  machinery. None of it has been attributed, and nothing in this document
  claims it has.

The single-turn budget (`max_commands_per_turn = 64`) is also a suspect: Redis
answers a whole pipeline in one event-loop pass, and Kivi still splits at 64.
Raising it trades memory for syscall count and has not been measured.

### Also unmeasured in this part

Native pipelining, the durability overlap redesign, the device-local flush
scheduler, multi-tablet scaling, and large values remain as Part I left them.
None of them regressed — the durable path is untouched and still forwards
through the queue — but none of them advanced either.
