# Phase 9 Memory Fabric integration: logical-vs-physical ownership, execution paths, and operating guarantees

The architectural claim: Kivi owns exact logical state; physical
representations may move, compress, disappear, reappear, or coexist
without changing any higher-level semantics. Changing a
materialization may affect latency and resource cost. It never affects
logical value, consistency, transaction outcome, durability, fencing,
or topology correctness.

## The three truths

1. **Logical authoritative state** lives in `kivi-state`
   (`StoredObject`: value, version, expiry, representation).
   Deterministic; the only thing consensus replicates and the only
   thing clients observe.
2. **Runtime physical materialization** lives in `kivi-memory`
   (`MemoryFabric`, per engine worker): arenas, compressed blocks,
   NVMe records, shadows. Worker-local, never replicated, never on
   the wire.
3. **Durable/reconstructible sources**, in order: live residence,
   fabric journal (`JournalEntry` per sealed id), WAL history (for
   inline roots), checkpoint bands (refs), immutable chunk packs.

A `ProviderId`, arena slot, NVMe offset, compressed block, or object
id is local physical state. It never enters Mutation IR, consensus
semantics, durable logical formats, client-visible state, or
cross-node topology metadata. `FabricRef{id, logical_len, version}`
is the opaque boundary token: small, `Copy`, carried in mutations and
checkpoints exactly like `ChunkedRef`, resolved only by the owning
worker's engine.

## Size split (system property, enforced at admission)

- `<= 256 B`: inline (`FABRIC_INLINE_MAX == TINY_INLINE_MAX`).
- `257 B..=256 KiB`: managed fabric materializations (medium).
- `> 256 KiB`: immutable chunk system (large; no second architecture).

Semantic primitives that are naturally tiny stay compact. The engine
converts at admission on every front (plain worker, bridge, reactor,
compound batches); the store trusts staged references exactly as it
trusts inline values.

## Offcore execution (no second runtime)

An NVMe miss never blocks the tablet owner. The commit coordinator
suspends the single operation into a parked set (`SuspendedRead`,
keyed by park sequence) and keeps draining everything else; the
bridge parks connection reads on its frontier; reactor tasks
`begin_promote`, await the lane reply, then `finish_promote` with no
borrow held across the await. One Compio runtime; the offcore lane is
an OS thread plus `async_channel`.

Every continuation carries identity plus the pinned logical version.
`poll_parked`/`finish_promote` re-fence id and version against the
live root before serving: mutation, deletion, commit, abort, expiry,
migration, or restart between suspend and resume fails the read
closed (`Unavailable`, stale counted) instead of resurrecting
history. Expired/deleted keys answer absence even when the promotion
succeeded.

## Reads linearize before bytes arrive

A parked read records the root id and version it was authorized to
observe. Pins (`pin`/`unpin`, nesting counts) keep the last
synchronous residence alive while outstanding operations reference
it; reclamation respects pins, and retirement defers around them.
`GetRange` windows apply after resolution at every site (fast path,
suspend resume, ephemeral direct, routed execute, bridge parks), so
range reads slice exactly the authorized bytes.

## Write publication: build, verify, publish, retire

Staging inserts bytes (arena residency, synchronous, bounded) and
returns an id; preparation predicts the authoritative version;
the durability lane persists payload bytes plus journal entries
BEFORE the WAL barrier of the same seal (durable-before-reference);
apply publishes the root exactly once. The old materialization
retires after success (or defers while pinned). Memory pressure can
compress or demote, but never deletes the only usable representation:
lone primaries are never retired, and full eviction needs a recorded
reconstruction source.

## 2PC with fabric values (engine path)

Prepare seals staged bytes under the staging version and pins the id;
commit-finalize re-seals the same bytes under the predicted
authoritative version (superseding the staging record) and releases
the pin; abort releases the pin and retires the id. Seal failure
releases prepare pins (no leak without an intent). Recovery imports
journal records cold and takes the last entry per id, so a restart
between prepare and finalize still commits. A physical fetch never
extends the OCC window: promotion returns the pinned version or
forces revalidation/failure.

## Cluster path: medium values replicate as chunked sidecars

Consensus never holds worker-local ids (`SetFabric` on the propose
path is rejected). The leader stages medium values into content-
addressed chunk sidecars and proposes tiny deterministic roots;
followers fetch missing sidecars through the durability gate before
acking. Same bytes always produce the same manifest, so retries,
snapshots, split/merge, and migration stay location-independent.
The 2PC resolver routes decision records by the unified rule (fiat
while the coordinator is writable, layout routing after retirement)
and CAS-aborts homeless records past the lease, so retired
coordinators cannot wedge intents.

## Checkpoint, recovery, WAL

Checkpoint bands carry refs only: cold objects are never promoted to
serialize. After each publish, fabric journals compact to records for
live roots, pending intents, in-flight payloads, and retained-band
refs (keep-on-doubt per worker). Recovery validates every journaled
record (length plus CRC32C, fail loudly), imports cold with lazy
hydration, and never trusts stale NVMe over WAL/checkpoint truth:
corrupt records fail startup, forgotten tablets skip with a warning.
The WAL never carries fabric bytes; the journal is their only durable
source, and journal GC is driven by exactly what roots and bands name.

## Topology

Topology moves logical bytes, never handles or offsets. Same-process
paths export bytes (`read_anywhere`, no promotion) and re-stage on
import. Outstanding lane operations fence per tablet (`fence_tablet`;
engine tablets are static, so fencing applies at shutdown drain plus
the control-plane-driven lifecycle). Split/merge never hydrate: band
refs clone, bytes stay where they are until touched.

## Runtime, backpressure, NUMA

`tick()` runs bounded worker-local maintenance every 16 wakes:
round-robin planning slices (64), movement under explicit byte/op/CPU
budgets, calibration, demote/promote/compress/reclaim, deferred-retire
sweep. Statistics refresh every 8th tick; arena compaction runs every
64th tick gated on material fragmentation with atomic remap publish.
Every queue is bounded and fails fast with `Overloaded`, distinct
from logical errors; staging has an explicit admission plan (arena
headroom) counted for observability. NUMA topology feeds per-worker
staging preferences (advisory only; single-node degrades to `Any`,
probes never block startup).

## Observability

`GET /v1/fabric`: per-worker counters (objects, bytes by residence,
hits, promotion misses, demotions, promotions, migration bytes,
compression ratio, cancelled, compactions, pins, parks, deferred
retires, stale completions, resumes, admission rejects, locators),
queue depths, arena pressure, and per-tablet footprints. Object-level
`explain` answers why an object sits where it does; pins plus pending
retires answer why memory cannot be reclaimed.

## Scope boundaries (kept)

No redundancy fabric, no erasure coding, no learned placement, no
mandatory CXL/RDMA, no generic cache owning semantics, no second
durability or consensus subsystem. The fabric is not a cache: it is
the physical execution of exact logical state.
