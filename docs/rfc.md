# Future-Native Distributed State Fabric

**Status:** Foundational Architecture Specification
**Supersedes:** all previous architecture RFCs
**Implementation language:** Rust
**Primary target:** Linux, commodity hardware first
**Design horizon:** long-lived architecture intended to absorb future memory, network, storage and compute technologies without changing the logical state model
**Compatibility objective:** Redis-compatible interfaces where valuable, but no Redis architectural constraints
**Primary objective:** build a distributed state system whose architecture is materially more advanced than existing Redis-like systems, while retaining a simple, safe baseline that works without specialized hardware

---

# 1. Vision

This system is not a Redis rewrite.

It is not:

```text
Redis
+
Rust
+
threads
+
io_uring
```

It is a general-purpose, ultra-low-latency **distributed state fabric**.

Its purpose is to provide applications with a single logical state abstraction while independently optimizing:

```text
execution
consistency
ordering
durability
materialization
memory placement
storage representation
redundancy
routing
network transport
acceleration
recovery
caching
computation
```

The central premise is:

> Logical information is not the same thing as its physical representation.

A single logical object may simultaneously exist as:

```text
authoritative mutable state

a consensus log entry

a hot decoded DRAM representation

a compressed DRAM representation

an immutable checkpoint segment

content-addressed chunks

an NVMe-resident copy

a read-only serving materialization

a client cache

a DPA/NIC cache

RLNC coded information

a cold wide-stripe erasure-coded representation

an object-storage archive
```

None of those physical forms independently defines the object.

The logical state does.

---

# 2. The architectural principle

The system follows one overriding principle:

> **Exact State. Adaptive Everything Else.**

Correctness-bearing state is exact.

The following MUST be exact:

```text
logical values

operation semantics

ownership

fencing

transaction outcomes

commit decisions

durability promises

authorization

object identity

tablet lineage
```

Optimization mechanisms MAY be:

```text
approximate

probabilistic

learned

speculative

hardware-assisted

stale

predictive

opportunistic
```

provided that an incorrect optimization decision can only make the system:

```text
slower
more expensive
less cache-efficient
temporarily less balanced
```

and can NEVER cause:

```text
wrong state

lost acknowledged durable data

duplicate non-idempotent execution

unauthorized access

stale data under a stronger read contract

split brain

partial committed transaction
```

This creates a hard boundary:

```text
                  CORRECTNESS DOMAIN

              Semantic Logical State
                       │
                 Mutation Model
                       │
             Ownership / Fencing
                       │
              Consensus / Commit
                       │
                Durable Outcome
                       │

══════════════════════════════════════════════

                  OPTIMIZATION DOMAIN

               Routing Prediction
               Memory Placement
               Cache Admission
               Hot-Key Detection
               Replication Shape
               Network Acceleration
               Compression
               Data Coding
               Learned Policies
               Hardware Offload
```

An optimization MAY fail closed into the correctness path.

It MUST NOT redefine it.

---

# 3. Design goals

The system is designed to provide all of the following within one architecture.

### Extreme single-node performance

A small point operation should ideally traverse:

```text
NIC
 ↓
owning CPU
 ↓
owning tablet
 ↓
local state
```

with:

```text
no global lock
no proxy hop
no control-plane RPC
no unnecessary allocation
no scheduler migration
```

### Horizontal scalability

Data and workload can be distributed across:

```text
cores
NUMA nodes
machines
racks
zones
regions
```

without changing the application API.

### Predictable tail latency

Maintenance work such as:

```text
checkpointing
migration
replica rebuild
coding
archive
scrubbing
compaction
```

is explicitly scheduled and resource-bounded.

It is not allowed to accidentally consume all available I/O or CPU.

### Strong failure semantics

The system should make common ambiguous failure scenarios safer than ordinary Redis-like protocols.

For example:

```text
write committed
reply lost
connection dies
client retries
```

should normally resolve to the original result rather than execute the operation twice.

### Hardware evolution

The same logical engine should work on:

```text
DRAM + ordinary NVMe + TCP
```

while being able to exploit, when available:

```text
CXL
persistent memory
RDMA
SmartNIC/DPA
P4 switches
NVMe FDP
hardware memory telemetry
userspace interrupts
kernel bypass
future memory devices
```

without changing correctness semantics.

### Rich state semantics

Users should be able to request primitives that communicate what they actually mean:

```text
StrictCounter

CommutativeCounter

BoundedCounter

Lease

Semaphore

RateLimit

Map

DistributedMap

StrictStream

ShardedStream
```

rather than expressing everything as arbitrary bytes and forcing the engine to infer impossible optimization opportunities.

---

# 4. Explicit non-goals

The architecture MUST NOT attempt to claim impossible properties.

It does not promise:

```text
infinite scaling of one strict serial key

zero-cost global linearizability

local linearizable reads under every asynchronous crash model

zero-cost durability

zero-copy everywhere

one data representation optimal for every workload

one redundancy code optimal for every lifecycle stage
```

A strict operation that requires total order has a serialization point.

A global strong write requires coordination.

The architecture exists to remove coordination where semantics do not require it, not to pretend coordination does not exist.

The LAW result is an important constraint: fully asynchronous crash-tolerant linearizable replication cannot in general provide arbitrary local reads, so read acceleration requires either additional assumptions, additional communication or a weaker contract.

---

# 5. System decomposition

The engine is decomposed into cooperating fabrics.

```text
                         APPLICATION
                              │
                              ▼
                      SEMANTIC FABRIC
                              │
                              ▼
                        LOGICAL STATE
                              │
                              ▼
                         MUTATION IR
                              │
          ┌───────────────────┼─────────────────────┐
          │                   │                     │
          ▼                   ▼                     ▼

    EXECUTION FABRIC    CONSISTENCY FABRIC    TRANSACTION FABRIC

          │                   │                     │
          └────────────┬──────┴──────────┬──────────┘
                       │                 │
                       ▼                 ▼

               DURABILITY FABRIC   MATERIALIZATION FABRIC
                       │                 │
                       ▼                 ▼

               REDUNDANCY FABRIC    MEMORY FABRIC
                       │                 │
                       └────────┬────────┘
                                ▼

                         STORAGE FABRIC
                                │
                                ▼

                         NETWORK FABRIC
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
          HOST                 DPA                 CXL/P4
            │
            └───────────────────┬───────────────────┘
                                ▼

                       ACCELERATION FABRIC


          OBSERVATION / CONTROL FABRIC
                        │
                        ▼
             adjusts physical decisions


                VERIFICATION FABRIC
                        │
                        ▼
               protects every layer
```

These are conceptual boundaries.

They MAY live in fewer runtime components.

They MUST remain semantically distinct.

---

# 6. Core logical hierarchy

The system hierarchy is:

```text
Cluster
  │
  ├── Region
  │    │
  │    └── Failure Domains
  │
  ├── Node
  │    │
  │    ├── NUMA Domain
  │    │    │
  │    │    └── Worker
  │    │
  │    ├── Memory Resources
  │    ├── Storage Resources
  │    └── Accelerator Resources
  │
  └── Namespace
       │
       └── Tablet
            │
            └── Objects
```

---

# 7. Namespace

A Namespace is the main semantic and policy boundary.

It replaces the weak notion of a Redis database number.

A Namespace declares **what it needs**, rather than how the engine should implement it.

Conceptually:

```rust
NamespaceIntent {
    consistency: ConsistencyIntent,
    durability: DurabilityIntent,
    latency: LatencyIntent,
    availability: AvailabilityIntent,

    geography: GeographyIntent,

    memory: MaterializationIntent,
    redundancy: RedundancyIntent,

    eviction: EvictionIntent,
    ttl: TtlIntent,

    transaction: TransactionIntent,

    security: SecurityIntent,

    cost: CostIntent,

    limits: NamespaceLimits,
}
```

Example:

```text
sessions:

consistency:
    linearizable

durability:
    survive two host failures

read_p99:
    100 µs target

write_p99:
    400 µs target

geo:
    one-region strong
    asynchronous DR

eviction:
    forbidden

memory:
    prefer fast local materialization

cost:
    medium
```

The planner maps that intent onto available hardware and protocols.

This follows the broader idea behind Declarative Memory Services: software should describe desired properties and allow a runtime to choose among heterogeneous memory mechanisms rather than hard-coding device-specific decisions.

---

# 8. Tablet

A Tablet is the normal unit of:

```text
ownership
routing
consensus
movement
replication
split
merge
load accounting
```

A Tablet is NOT a fixed Redis-style hash slot.

Tablets dynamically adapt to workload.

---

# 9. Partition layouts

Namespaces support at least two layouts.

## Hash layout

Default for Redis-compatible state.

```text
key bytes
    │
    ▼
stable PartitionHash
    │
    ▼
adaptive partition tree
    │
    ▼
Tablet
```

Example:

```text
0* → Tablet A
1* → Tablet B
```

A becomes overloaded:

```text
00* → A0
01* → A1
```

The number of tablets is therefore adaptive.

---

## Ordered layout

For:

```text
range scans
ordered maps
indexes
time-series ranges
```

Example:

```text
["a", "g") → A
["g", "p") → B
["p", ∞)   → C
```

---

# 10. Why adaptive tablets

Fixed slot counts make routing simple but permanently constrain movement granularity.

Modern systems have moved toward more flexible tablet/region models. ScyllaDB tablets can split and migrate across both nodes and CPU shards, while TiKV Regions independently form Raft groups and act as movement units.

Our tablet size is determined by a multidimensional cost function rather than bytes alone.

---

# 11. Tablet heat vector

Each tablet tracks approximately:

```text
logical bytes

resident bytes

object count

requests/sec

reads/sec

writes/sec

CPU ns/sec

network input/sec

network output/sec

storage bytes/sec

replication bytes/sec

queue latency

service p50

service p99

service p99.9

consensus lag

checkpoint debt

repair debt
```

It also records hot-key distribution.

---

# 12. Beyond hotness: performance criticality

Frequency alone is not sufficient.

A frequently accessed memory region may have its latency hidden by memory-level parallelism while a less frequently accessed serial dependency may dominate request latency.

Research on Amortized Offcore Latency explicitly incorporates memory latency and memory-level parallelism and demonstrated that hotness is not synonymous with performance criticality.

Therefore the engine's observation model SHOULD evolve toward:

```text
access frequency
+
reuse distance
+
bytes touched
+
CPU cost
+
memory stall contribution
+
memory-level parallelism
+
critical-path contribution
+
tail-latency contribution
```

The system refers to this generalized measure as:

```text
ExecutionCriticality
```

---

# 13. Hotspot classification

Before splitting or moving a tablet, the engine classifies the hotspot.

Possible categories:

```text
ManyKeyHotspot

SingleReadHotKey

SingleStrictWriteHotKey

CommutativeWriteHotKey

StorageHotspot

NetworkHotspot

MemoryCriticalHotspot

BackgroundInterference
```

This distinction is critical.

A tablet dominated by one strict counter cannot be fixed by repeatedly splitting surrounding cold keys.

---

# 14. Execution Fabric

The initial execution model is:

> single mutable owner, adaptive execution.

This preserves the locality advantages of thread-per-core designs without permanently committing to simplistic run-to-completion scheduling.

---

# 15. Worker

A worker is pinned to a CPU.

Conceptually:

```rust
struct Worker {
    id: WorkerId,
    cpu: CpuId,
    numa: NumaId,

    reactor: IoReactor,

    tablets: TabletRegistry,

    allocator: WorkerAllocator,

    scheduler: LocalScheduler,

    timers: TimerSystem,

    telemetry: WorkerTelemetry,
}
```

Normal mutable tablet state is accessed only by its owner worker.

---

# 16. Single-owner invariant

A tablet replica has:

```text
exactly one mutable execution owner
```

at one instant.

Therefore its main object index does NOT require:

```text
Mutex
RwLock
DashMap
concurrent B-tree
```

for ordinary operations.

This reduces:

```text
cache-line bouncing
locking
NUMA traffic
atomic operations
scheduler uncertainty
```

---

# 17. Worker movement

Ownership is movable.

```text
Core 1:
    Tablet A
    Tablet B
    Tablet C
```

If Core 1 is overloaded:

```text
Tablet B
   │
   └────────→ Core 7
```

The transfer is explicitly fenced and performed at an execution boundary.

The Tablet remains single-owned throughout.

---

# 18. Execution classes

Operations are classified.

## NanoOp

Examples:

```text
GET small key
SET small key
INCR
EXISTS
```

Expected behavior:

```text
run to completion
```

No unnecessary async state machine.

---

## YieldableOp

Examples:

```text
large hash scan
large set operation
serialization
large result generation
```

Executed in bounded continuation slices.

```text
work budget
 ↓
yield
 ↓
resume
```

---

## OffcoreOp

Operation waiting for:

```text
NVMe
CXL
remote memory
network
object storage
```

is suspended immediately.

The core runs something else.

---

# 19. Future userspace preemption

Cooperative yielding alone can still produce head-of-line latency when a long operation fails to yield quickly enough.

PreemptDB demonstrated database transaction preemption with Intel user interrupts, while SBB applies user-level timer/NIC interrupts to decentralized network scheduling.

Therefore the runtime reserves an optional:

```text
PreemptionBackend
```

Possible implementations:

```text
Cooperative
UserspaceInterrupt
FutureHardwareMechanism
```

A supported platform may preempt:

```text
YieldableOp
```

for an urgent:

```text
NanoOp
```

without changing tablet ownership.

---

# 20. No generic work stealing of state

A temporary request continuation MAY move only when it does not imply moving mutable tablet state.

The default remains:

```text
state stays with owner worker
```

If scheduling work elsewhere is necessary:

```text
immutable input
or
explicit copied snapshot
```

is sent.

The result returns to the tablet owner for mutation.

---

# 21. Compute Plane

Heavy computation is separated from the latency-critical state plane.

```text
State Worker
    │
    ├── small operations
    │
    └── immutable snapshot/input
             │
             ▼
       Compute Plane
             │
       ┌─────┼─────┐
       │     │     │
      WASM  Search Compression
       │
       └─────────────→ proposed MutationIR
```

Compute workers do not mutate tablet state directly.

---

# 22. Actor frameworks

Frameworks such as Kameo MAY be used for:

```text
placement orchestration

archive jobs

administrative workflows

maintenance coordinators

compute supervision
```

They MUST NOT define:

```text
Tablet execution

DataWorker scheduling

Raft group scheduling

WAL ordering
```

The tablet architecture already provides actor-like ownership without requiring one framework-managed Tokio task per tablet.

---

# 23. State representation

One logical object contains a small authoritative root.

Conceptually:

```rust
ObjectRoot {
    object_type,
    logical_version,
    expiry,
    representation,
}
```

Physical representation is mutable independently from logical state.

---

# 24. Representation ladder

An object can transition between:

```text
Inline
 ↓
ArenaResident
 ↓
Segmented
 ↓
Chunked
 ↓
CompressedMemory
 ↓
LocalStorage
 ↓
RemoteMaterialization
 ↓
Archived
```

without changing the object's logical value.

---

# 25. Small value path

Small values SHOULD remain simple.

```text
key
 ↓
ObjectRoot
 ↓
inline bytes
```

No chunk hash.

No manifest.

No compression metadata unless worthwhile.

The architecture MUST avoid paying large-object machinery for a 32-byte value.

---

# 26. Medium objects

Medium values live in worker-local arenas.

```text
ObjectRoot
    │
    └── Handle(slot, generation)
```

Handles rather than permanent raw pointers allow later:

```text
relocation
compaction
tier movement
object regrouping
```

---

# 27. Large object model

Large values become immutable content-addressed chunks.

```text
ObjectRoot
   │
   ▼
Manifest
   │
   ├── Chunk A
   ├── Chunk B
   ├── Chunk C
   └── Chunk D
```

Updating a subset creates:

```text
new chunks
+
new root
```

Unchanged chunks are reused.

---

# 28. Why immutable chunks are a foundational primitive

One mechanism simultaneously improves:

```text
checkpoint reuse

migration

incremental updates

content verification

deduplication

remote caching

direct bulk reads

storage tiering

repair

network coding

erasure coding

future DPA caching

future CXL materialization
```

It is intentionally one of the most important architectural primitives.

---

# 29. Chunk identity

Conceptually:

```text
ChunkId =
    BLAKE3(
        domain
        || security-domain
        || canonical chunk bytes
    )
```

Deduplication is constrained to a configured security domain.

Cross-tenant deduplication is disabled by default.

---

# 30. Chunk security and RLNC

Network coding introduces a special integrity problem.

A corrupted coded symbol can be recoded by intermediate nodes, spreading corruption through the coding graph.

RLNC literature identifies this class as a pollution attack and studies homomorphic verification mechanisms that can verify coded packets without first decoding them.

Therefore:

> Untrusted RLNC recoding MUST NOT rely only on final generation decode failure.

Initial production RLNC is limited to authenticated cluster nodes and end-to-end generation verification.

Future untrusted recoders require:

```text
homomorphic authentication
or
equivalent pollution-resistant coding verification
```

before becoming valid information sources.

---

# 31. Adaptive data representation

The engine is not required to use one structure for one logical type forever.

Example Set:

```text
tiny set
    → SmallVec

medium sparse set
    → HashSet

dense integer set
    → Roaring bitmap
```

Example ordered map:

```text
dynamic hot
    → B-tree/ART

distribution-friendly
    → learned hybrid

frozen
    → minimal perfect hash + packed values
```

Logical semantics remain identical.

---

# 32. Learned indexes

Learned indexes are not universally superior.

HIRE explicitly combines traditional index robustness with model-accelerated navigation and adaptive leaf structures to improve mixed-workload stability.

Therefore learned indexing is an optional:

```text
IndexRepresentation
```

for ordered/frozen data.

Never the correctness layer.

If its prediction is wrong:

```text
fallback search
```

must still find the correct record.

---

# 33. Frozen minimal-perfect-hash representation

Immutable checkpoint bands have a known fixed key set.

This is exactly the workload where minimal perfect hashing becomes compelling.

Meep Hashing is a recent SIGMOD 2026 large-scale minimal-perfect-hash design and is therefore a research candidate for immutable bands.

Conceptually:

```text
FrozenBand
    │
    ├── MPHF
    ├── fingerprints
    ├── packed keys/offsets
    └── immutable values/chunks
```

A mutable tablet remains on a normal dynamic hash/index structure.

---

# 34. Memory Fabric

The system MUST NOT encode a permanent architecture such as:

```rust
enum MemoryTier {
    Dram,
    Nvme,
    Cxl,
}
```

because that hard-codes today's devices.

Instead it uses declarative materialization requirements.

---

# 35. MaterializationIntent

Conceptual structure:

```rust
MaterializationIntent {
    access_latency_target,
    tail_latency_target,

    expected_read_rate,
    expected_write_rate,

    volatility_allowed,

    durability_required,

    coherence_requirement,

    sharing_scope,

    bandwidth_requirement,

    compute_near_data,

    encryption_requirement,

    locality_requirement,

    cost_weight,

    reclaim_priority,
}
```

A provider advertises capabilities.

---

# 36. Resource providers

Possible providers include:

```text
Local DRAM

Compressed DRAM

CXL-attached memory

Local NVMe

Remote soft memory

DPA memory

Persistent memory

Object storage

future device
```

The namespace/object does not know which physical implementation satisfied the intent.

---

# 37. Memory placement by object behavior

Traditional allocators group by size, which can mix hot and cold objects on the same pages.

OBASE demonstrates object-aware address-space reorganization and reports substantially better page utilization by grouping objects based on access behavior.

Our allocator therefore MAY classify objects by:

```text
size
temperature
mutability
read/write ratio
tablet
tenant
criticality
expected lifetime
```

rather than size alone.

---

# 38. Arena classes

Possible worker-local arenas:

```text
HotMutable

HotReadMostly

Warm

ColdCandidate

ShortLived

LongLivedMetadata

LargeRootMetadata
```

The exact set is internal and adaptive.

---

# 39. Hardware memory telemetry

NEMO demonstrates a hardware memory-controller telemetry model capable of providing flexible policy-specific observations with low CPU overhead.

The architecture therefore defines:

```text
MemoryTelemetryProvider
```

Implementations may use:

```text
software sampling

perf counters

NEMO-like future hardware

CXL telemetry

allocator instrumentation
```

Planner logic consumes generic signals.

---

# 40. Materialization lifecycle

Example:

```text
t=0
Hot:
    decoded DRAM x2

t=20m
Warm:
    decoded DRAM x1
    compressed DRAM x1
    durable NVMe

t=2h
Cold:
    metadata DRAM
    chunks NVMe
    redundancy fabric

t=2d
Archive:
    object storage
    coded redundancy
```

Sudden read burst:

```text
archive
  ↓
NVMe prefetch
  ↓
DRAM promotion
```

Application is unaware.

---

# 41. Durability Fabric

Durability is separate from:

```text
ordering
consensus
materialization
```

This is a major change from a classic WAL-centric architecture.

LogDrive demonstrates the usefulness of explicitly separating sequencing from durability so different durable substrates can be composed under the same ordered abstraction.

---

# 42. Logical commit flow

Conceptually:

```text
Mutation
   │
   ▼
Ordering / Consensus
   │
   ▼
DurabilityIntent
   │
   ▼
DurabilityProvider
   │
   ▼
CommitProof
   │
   ▼
Apply
   │
   ▼
Reply
```

The provider may differ by namespace or deployment.

---

# 43. DurabilityProvider

Conceptual interface:

```rust
trait DurabilityProvider {
    fn persist(intent: PersistIntent) -> PersistFuture;
    fn recover(...);
    fn capabilities() -> DurabilityCapabilities;
}
```

Possible capabilities:

```text
volatile

local-stable

quorum-stable

cross-zone

cross-region

persistent-memory

object-backed
```

---

# 44. Local NVMe durability

The baseline provider works on ordinary NVMe.

Initially:

```text
append
 ↓
durability barrier
 ↓
stable
```

But centralized group commit is NOT an invariant.

---

# 45. Autonomous commit

Modern NVMe devices expose enough parallelism that a single group-commit coordinator can itself become a bottleneck.

Autonomous Commit explores workers performing smaller independent log writes in parallel rather than funneling all commit acknowledgments through a centralized commit mechanism.

Therefore local durability implementations MAY include:

```text
GroupCommit

AutonomousCommit
```

The selection is benchmark-driven.

---

# 46. Cloud durability

Cloud deployments MAY use a quorum durable-log provider backed by cloud storage/services.

The ordering layer remains unchanged.

Conceptually:

```text
Mutation order
     │
     ▼
Durable slot/quorum layer
     │
     ├── provider A
     ├── provider B
     └── provider C
```

This follows the separation demonstrated by LogDrive rather than requiring every node to own identical local disk semantics.

---

# 47. Persistent-memory/CXL durability

If hardware provides a validated persistent memory contract:

```text
PersistIntent
      ↓
PM/CXL provider
```

may replace NVMe for suitable workloads.

The correctness interface remains identical.

---

# 48. Mutation IR

User commands are NOT the canonical replicated/persisted semantics.

Every application mutation is transformed into stable internal Mutation IR.

Example:

```rust
MutationEnvelope {
    version,
    namespace,
    tablet,
    tablet_epoch,

    client_identity,
    idempotency_key,

    operation,
}
```

---

# 49. Mutation IR goals

Mutation IR provides:

```text
deterministic replay

protocol independence

stable replication

upgrade versioning

changefeed source

transaction record

script/WASM normalization

verification target
```

---

# 50. Mutation determinism

Replicas MUST NOT independently re-execute nondeterministic code.

If a server function chooses:

```text
random value
time value
generated identifier
```

the chosen result becomes explicit Mutation IR.

---

# 51. Typestate persistence safety

Rust should enforce critical durability transitions where practical.

SquirrelFS showed that Rust typestate can encode persistence ordering constraints such that invalid update order becomes a compile-time type error.

Our core should use similar ideas.

Example:

```text
Chunk<Volatile>
      │
   persist()
      ▼
Chunk<Durable>
      │
      │ only this type accepted
      ▼
CommittedRootBuilder
```

This makes the invariant:

```text
committed durable root
⇒
required payload durable
```

partially compiler-enforced.

---

# 52. Large-value commit protocol

For a large chunked value:

```text
1. authorize/admit write

2. create immutable chunks

3. hash chunks

4. stage required chunks

5. satisfy durability intent

6. construct root MutationIR

7. replicate/order root mutation

8. commit

9. apply root

10. publish reply
```

A Raft/log record MUST NOT count as durably persisted on a replica when it references required local sidecar content that has not met the replica's required durability contract.

---

# 53. Active logging format

Durable formats are project-owned.

No long-lived canonical format is directly defined by:

```text
bincode
rkyv
postcard
serde
```

Persistent structures include:

```text
magic
major version
minor version
feature flags
identity
length
CRC
```

Canonical immutable artifacts additionally receive a strong content hash.

---

# 54. Integrity algorithms

Default conceptual roles:

```text
CRC32C
    → cheap accidental-corruption detection

BLAKE3-256
    → content identity / strong immutable verification
```

Algorithm IDs are stored.

No algorithm is assumed permanent for the entire lifetime of the project.

---

# 55. Checkpoint model

No `fork()`.

Checkpointing uses logical epochs/MVCC-style version retention.

```text
Applied State at index S
          │
          ├── freeze logically
          │
          ├── continue new writes
          │
          └── serialize S asynchronously
```

Old versions exist only while required.

---

# 56. Incremental checkpoint bands

A tablet is divided into physical checkpoint bands independent of its routing partition.

Dirty bands are rewritten.

Clean bands are reused.

```text
Checkpoint N

Band A hash X
Band B hash Y
Band C hash Z


Checkpoint N+1

Band A → reuse X
Band B → new W
Band C → reuse Z
```

---

# 57. Immutable checkpoint indexing

Frozen bands MAY use different structures from live state.

For example:

```text
Live:
    HashTable

Frozen:
    packed records
    +
    minimal perfect hash
```

This avoids forcing live-state mutability requirements onto archival state.

---

# 58. Checkpoint publication

A checkpoint is visible only after all referenced pieces satisfy its durability contract.

Conceptually:

```text
segments durable
     ↓
chunks durable
     ↓
manifest durable
     ↓
manifest atomic publication
     ↓
checkpoint installed
```

Old log history may then be compacted according to consensus safety.

---

# 59. Consensus Fabric

The baseline strong-consistency mechanism is conservative.

Each strongly replicated Tablet is an independent consensus group.

```text
Tablet A → Group A

Tablet B → Group B

Tablet C → Group C
```

No fixed multi-tablet consensus cohort.

---

# 60. Physical Multi-Raft

Logical groups remain independent, but physical mechanisms are multiplexed.

Many groups may share:

```text
connections
WAL submission
durability operations
timers
batching
snapshot channels
```

TiKV is an established example of many independent Raft Regions operating within one storage node.

---

# 61. Consensus implementation

Do not implement Raft from scratch initially.

The engine keeps a Kivi-owned consensus front:

```rust
trait ConsensusCore
```

The selected backend is OpenRaft 0.10.0-alpha.35, exact-pinned while the 0.10 API remains unstable. The selected configuration uses `single-threaded` OpenRaft on Kivi's Compio reactors, so the backend respects the single-owner invariant.

The selected backend's group-density and integration tests MUST measure:

```text
100 groups/worker

1,000 groups/worker

10,000 groups/worker
```

including:

```text
idle CPU
memory/group
timer overhead
network overhead
storage integration
simulation friendliness
```

No external Raft library type leaks into durable or public engine types.

---

# 62. Strong write baseline

For a durable linearizable namespace:

```text
request
 ↓
tablet authority
 ↓
validate
 ↓
MutationIR
 ↓
consensus proposal
 ↓
required durable quorum
 ↓
commit
 ↓
local ordered apply
 ↓
dedup outcome durable/replicated
 ↓
reply
```

This is the fallback path that all accelerators must be able to fall back to.

---

# 63. Request identity

Native mutations carry:

```text
SessionId
RequestSeq
```

Optional durable idempotency uses:

```text
IdempotencyKey
```

This solves a major practical failure ambiguity.

---

# 64. Lost reply

Example:

```text
SET / INCR
   ↓
commit succeeds
   ↓
server reply lost
   ↓
connection closes
```

Client retries:

```text
same SessionId + RequestSeq
```

The new authority returns the recorded outcome.

It does not execute another mutation.

---

# 65. Durable idempotency

For retries surviving client restart:

```text
IdempotencyKey
```

is stored with:

```text
request fingerprint
result
retention
```

Reuse for a different logical mutation produces:

```text
IDEMPOTENCY_CONFLICT
```

---

# 66. Read contracts

Native reads expose explicit semantics.

```text
Latest

AtLeast(CommitToken)

BoundedStale(duration)

Any
```

No ambiguous “read replica” mode.

---

# 67. Latest baseline

Conservative linearizable read:

```text
establish consensus read barrier
 ↓
wait local applied state >= barrier
 ↓
read
```

Batching may reduce overhead.

---

# 68. Almost-local reads

Given the impossibility result described by the LAW work, almost-local read schemes are an important research path for asynchronous strong replication; the authors report substantial Raft throughput gains while preserving linearizability and asynchrony in their evaluation.

Therefore:

```text
AlmostLocalReadBackend
```

is a Production Candidate.

Not the only baseline.

---

# 69. Roster leases / Bodega mode

In deployments where synchrony assumptions can be enforced, Bodega's roster-leasing approach is particularly interesting: arbitrary subsets of replicas may become designated local linearizable responders instead of restricting local reads to one leader.

The architecture reserves:

```text
ReadAuthorityProvider
```

with potential modes:

```text
Leader

AlmostLocal

RosterLease

WriteGuardCache
```

---

# 70. WriteGuards

WriteGuards are highly aligned with our tablet ownership model.

The mechanism associates writes with ownership/fencing information at range granularity, allowing strongly consistent caches to reject delayed writes from stale owners without adding coordination to ordinary cache reads.

Therefore every authoritative tablet range has a generic:

```text
WriteGuardGeneration
```

derived from ownership state.

A write includes:

```text
TabletId
TabletEpoch
WriteGuardGeneration
```

Stale writes are rejected.

---

# 71. Strong cache architecture

Future strong cache:

```text
               Ownership
                  │
             WriteGuard
                  │
                  ▼
Storage / Tablet Authority


Client
  │
  ▼
Strong Cache
  │
  ├── valid generation → serve
  │
  └── uncertain → fallback
```

The cache cannot manufacture freshness.

If its metadata is insufficient:

```text
fall back to authority
```

---

# 72. Bounded-staleness cache

Not every workload needs linearizability.

Skybridge demonstrates that a lightweight out-of-band replication path can materially tighten staleness bounds for globally distributed caches without replacing the underlying replication system.

The architecture therefore treats:

```text
freshness delivery
```

and:

```text
durability replication
```

as potentially different channels.

---

# 73. Semantic Fabric

The database should not assume that every operation is an opaque:

```text
read bytes
write bytes
```

Built-in operations have semantic properties.

---

# 74. Operation properties

Conceptually:

```rust
OperationSemantics {
    reads,
    writes,

    idempotent,

    commutative_state,
    commutative_result,

    associative,

    partitionable,

    monotonic,

    bounded,

    requires_total_order,

    conflict_scope,
}
```

Only built-ins/proven extensions may set these.

User code cannot simply assert:

```text
trust me, this commutes
```

and bypass consensus ordering.

---

# 75. Strict counter

```text
INCR x
```

returns an exact ordinal result.

Two `INCR`s may produce the same final state regardless of order, but returned values differ.

Therefore:

```text
commutative_state = yes
commutative_result = no
```

and it cannot blindly use an unordered commutative path.

---

# 76. Commutative counter

A native primitive may instead expose:

```text
CounterAddNoOrdinal(+1)
```

where operation result does not reveal ordering.

That operation can exploit stronger optimization opportunities.

---

# 77. CURP-style fast path

CURP demonstrated that commutative operations can separate durability from global ordering and complete in one RTT in appropriate cases; the original work included Redis and RAMCloud implementations. This is a high-value future fast path, not the baseline consensus mechanism.

Rules:

```text
only statically known safe operations

observable results must commute

deduplication required

reconfiguration must fence fast path

conflict → fallback to ordered consensus
```

---

# 78. Escrow / bounded semantics

Some global numerical invariants can be pre-coordinated.

Example:

```text
capacity = 10,000
```

Rights:

```text
Region A: 3000
Region B: 3000
Region C: 4000
```

Local operations consume local rights.

WAN coordination occurs only when rights must be moved.

Native primitives should expose:

```text
BoundedCounter
Semaphore
RateLimit
InventoryCapacity
```

where escrow semantics are appropriate.

---

# 79. CRDT semantics

CRDTs are explicit types.

Examples:

```text
PNCounter

ORSet

LWWRegister

GrowOnlySet
```

An ordinary strict object is never silently converted to a CRDT.

---

# 80. Transaction Fabric

Single-tablet transactions should remain extremely cheap.

Cross-tablet transactions pay distributed coordination only when necessary.

---

# 81. Same-tablet transaction

Because one worker owns the tablet:

```text
read
validate
mutate
```

can be atomically serialized in that local tablet state machine before replication.

---

# 82. Cross-tablet transaction

Initial protocol:

```text
OCC / versioned reads
+
replicated prepare intents
+
2PC
+
durable transaction identity
```

Do not invent a novel transaction protocol for v1.

---

# 83. Transaction coordinator

Coordinator is deterministically chosen from participants, such as:

```text
lowest TabletId
```

rather than a central global coordinator.

---

# 84. Transaction recovery

Prepared participants do not guess.

If coordinator is temporarily unavailable:

```text
transaction waits
```

rather than violating atomicity.

This is an explicit correctness-over-availability choice for strong transactions.

---

# 85. Affinity

Applications can provide:

```text
partition_key
```

to encourage related keys to remain together.

This reduces distributed transaction frequency.

Redis hash tags are supported only as a compatibility mechanism.

---

# 86. RSM-to-RSM communication

Interactions between replicated state machines deserve a first-class primitive.

Picsou introduces Cross-Cluster Consistent Broadcast and quorum acknowledgments to efficiently communicate between RSMs.

We therefore reserve:

```text
RsmChannel
```

for:

```text
cross-tablet transaction messages

geo replication

cluster reconciliation

control transitions

changefeed forwarding
```

Initial implementation may use simpler reliable messaging.

Picsou-like optimization is a strong research candidate.

---

# 87. Redundancy Fabric

This is one of the most unconventional parts of the architecture.

The system MUST NOT model all safety as:

```text
replication factor = 3
```

Instead it manages **recoverable information**.

---

# 88. InformationAsset

A logical durable asset may have physical representations:

```text
FullReplica

LogReplica

CheckpointSegment

ImmutableChunk

CodedSymbol

ReadMaterialization

SoftCache
```

The system distinguishes:

```text
serving capacity

commit durability

recoverability

archive redundancy
```

---

# 89. Lifecycle-specific redundancy

One coding method is not required to serve every lifecycle stage.

```text
ACTIVE MUTABLE
    │
    ▼
Consensus replication

WARM IMMUTABLE
    │
    ▼
RLNC / MSR / selective replication

COLD
    │
    ▼
wide-stripe efficient coding

ARCHIVE
    │
    ▼
object-store / EC / network coding
```

---

# 90. Active state

Normal active strict state uses ordinary replicated consensus.

Do not network-code every `SET`.

Reason:

```text
minimal write latency

simple recovery

simple leadership

simple correctness
```

---

# 91. RLNC

Random Linear Network Coding is a major research path.

A generation consists of original symbols:

```text
A B C D ...
```

A coded symbol is:

```text
αA + βB + γC + δD + ...
```

over a finite field.

Recovery requires enough linearly independent coded information.

Intermediate nodes can recode coded symbols without decoding the original data. Current Rust RLNC implementations expose this recoder model directly.

---

# 92. Why RLNC changes storage semantics

Traditional coding asks:

```text
Which shards are missing?
```

RLNC allows the controller to ask:

```text
How much independent information rank remains?
```

Instead of:

```text
Shard 3
Shard 7
Parity 2
```

the system can reason about:

```text
Generation G
required rank = K
available innovative rank = R
```

---

# 93. Adaptive redundancy

A generation may have:

```text
target information margin
```

rather than a permanently fixed `k+m`.

Example:

```text
normal:
    1.20× information

maintenance expected:
    1.35×

suspected host failure:
    1.50×
```

New coded information can be generated incrementally.

---

# 94. Proactive self-healing

Failure prediction signals MAY increase redundancy before actual failure.

Examples:

```text
SSD SMART degradation

rack maintenance

network instability

node draining

thermal issues
```

Prediction is advisory.

Failure of prediction only changes resource use.

It never permits redundancy below hard safety policy.

---

# 95. Multi-source recoding

Instead of:

```text
source
 ↓
decode
 ↓
re-encode
 ↓
new node
```

a repair path may use:

```text
Node A ── coded info ─┐
Node B ── coded info ─┼──→ recoder ─→ destination
Node C ── coded info ─┘
```

This makes storage and networking cooperate.

---

# 96. Network-aware recoding

If expensive WAN links separate regions:

```text
Region EU
  A B C
   \|/
 recoder
    │
 expensive WAN
    │
 Region US
```

the local region may combine useful information before crossing the scarce link.

This is one of RLNC's most distinctive potential advantages.

---

# 97. RLNC generation sizing

Huge unbounded coding matrices are unacceptable.

RLNC uses bounded generations.

Candidate dimensions might include:

```text
32
64
128
256
```

but no value is fixed architecturally.

Benchmark:

```text
encoding throughput
recoding throughput
rank overhead
coefficient overhead
decode CPU
memory usage
failure recovery latency
```

---

# 98. Systematic RLNC

Full RLNC means normal reads may require decoding.

That is undesirable for many warm-state workloads.

Research should compare:

```text
Full RLNC

Systematic RLNC

Sparse RLNC

Sliding-window RLNC

Fulcrum-style coding
```

A systematic mode retains original symbols for ordinary reads while maintaining coded redundancy.

---

# 99. Sparse/Fulcrum research

Dense RLNC can make decoding expensive.

Sparse network coding and Fulcrum-style constructions attempt to retain recoding/flexibility with lower computational costs.

They remain research candidates.

They MUST NOT define durable archive format until extensively benchmarked and verified.

---

# 100. RLNC Rust candidates

Current experimental candidates include:

```text
rlnc

rlnc-simdx
```

The latter currently advertises SIMD kernels across several x86/ARM instruction sets and provides Encoder/Decoder/Recoder APIs.

These are experiments, not permanent format authorities.

Our own Redundancy Fabric defines the storage format and metadata.

---

# 101. Stripeless redundancy

Traditional erasure coding uses fixed stripes.

Nos/Nostor demonstrates a stripeless model for in-memory erasure-coded storage in which nodes independently replicate and encode data rather than synchronizing through conventional stripe formation.

Our architecture should explore the broader principle:

> Redundancy placement does not require fixed physical stripes.

RLNC is especially compatible with this idea.

---

# 102. Redundancy generation

A RedundancyGeneration may describe:

```text
GenerationId

source content set

required rank

available rank estimate

placement map

failure-domain coverage

coding family

symbol size

verification root

repair policy
```

It is independent from a traditional stripe number.

---

# 103. Network coding for warm blobs

NCBlob revisited non-systematic network-coded MSR storage for warm blob workloads and showed practical repair benefits in its cloud evaluation.

This reinforces the idea that network coding belongs specifically in:

```text
warm immutable / blob-like lifecycle
```

rather than blindly replacing active mutable replication.

---

# 104. Wide-vector cold coding

WiseCode targets very wide stripes with low storage overhead while preserving repair efficiency, demonstrating a different optimum from RLNC for cold large-scale durability.

Therefore cold archival redundancy may select:

```text
WiseCode-like vector coding
MSR
LRC
RS
```

depending on implementation maturity.

---

# 105. Repair-aware coding

Modern coding research such as LESS and DRBoost shows that theoretical storage/bandwidth efficiency alone is insufficient; actual repair performance depends on seeks, I/O pattern and degraded-read scheduling.

The Redundancy Fabric therefore chooses:

```text
code
+
physical placement
+
repair plan
+
helper topology
```

as one decision.

Not just a codec name.

---

# 106. RedundancyProvider capabilities

Conceptually:

```rust
RedundancyCapabilities {
    mds,
    rateless,
    recodable,
    systematic,
    local_repair,
    exact_k_of_n,
    incremental_redundancy,
    stripeless,
    repair_optimal,
}
```

---

# 107. Redundancy modes

Initial modes:

```text
Replication

ReedSolomonBaseline

RLNC

MSR

WideVector

ObjectStoreNative
```

No mode is globally default for all state.

---

# 108. Recovery Fabric

Recovery is distributed rebalancing.

If one node hosted 600 tablet replicas:

```text
do NOT:
    rebuild all 600 onto one machine

do:
    distribute recovery over available cluster capacity
```

---

# 109. Recovery priority

Order:

```text
restore unavailable quorum

restore minimum durability

restore hot serving capacity

restore desired redundancy

restore ideal placement

resume optional archive work
```

---

# 110. Multi-source recovery

Immutable chunks/checkpoint segments may come from any verified source.

```text
Chunk A ← Replica 1

Chunk B ← Archive

Chunk C ← Remote cache

Chunk D ← Reconstructed RLNC information
```

Every immutable piece is verified.

---

# 111. Self-healing

Corruption response:

```text
detect corruption
 ↓
quarantine bad representation
 ↓
find independent valid information
 ↓
verify
 ↓
reconstruct
 ↓
repair
```

Corruption of one replica should not require operator intervention when sufficient information remains elsewhere.

---

# 112. Scrubbing

Background scrubbing validates:

```text
sealed WAL

checkpoint segments

chunk packs

coded generations

archive manifests
```

It is resource-budgeted.

---

# 113. Storage Fabric

Storage providers support capabilities rather than one hard-coded disk model.

```rust
StorageCapabilities {
    stable_write,
    direct_io,
    atomic_write,
    fdp,
    zns,
    uring_cmd,
    persistent_memory,
    remote,
}
```

---

# 114. NVMe FDP

Flexible Data Placement can allow the engine to communicate expected data lifetime to compatible SSDs.

Candidate write classes:

```text
active WAL

sealed WAL

temporary migration

checkpoint

chunk pack

archive staging
```

FDP is an optimization only.

Plain NVMe remains functionally complete.

---

# 115. Declarative I/O

Multiple maintenance jobs often need overlapping data.

Instead of each issuing arbitrary reads independently:

```text
Checkpoint:
    read A B C

Scrubber:
    read B C D

RLNC:
    read A C E
```

they declare intents:

```rust
IoIntent {
    data_set,
    deadline,
    priority,
    operation,
}
```

A storage planner schedules shared/reordered I/O.

Recent Declarative I/O work explicitly explores this style of exposing high-level data requirements so the storage system can optimize across jobs.

---

# 116. Maintenance scheduler

Jobs include:

```text
checkpoint

scrub

repair

migration

archive

compression

redundancy generation

chunk GC

WAL reclamation
```

They share an explicit resource scheduler.

No subsystem is allowed to spawn unlimited background I/O.

---

# 117. Network Fabric

The native data protocol is message-oriented.

TCP remains the deployment baseline, but the logical protocol does not assume byte-stream semantics.

---

# 118. Native protocol frame

Conceptually:

```rust
Frame {
    protocol_version,

    cluster_id,

    session,

    request_id,

    namespace,

    operation,

    routing_hint,

    tablet_epoch,

    deadline,

    flags,

    payload,
}
```

---

# 119. Native protocol features

Native clients support:

```text
multiplexing

out-of-order replies

direct worker routing

route updates

server push

streaming

cancellation

request deduplication

commit tokens

cache metadata

capability negotiation
```

---

# 120. RESP compatibility

RESP2/RESP3 is an adapter.

```text
RESP
 ↓
Redis compatibility semantics
 ↓
Native operation
 ↓
State engine
```

RESP does not define:

```text
partition layout

internal replication

Mutation IR

durable format

native transactions
```

---

# 121. Smart direct routing

Happy path:

```text
Client
  │
  └────────────→ Node
                    │
                    └────────→ Worker
                                 │
                                 └────→ Tablet
```

No central proxy.

Client caches sparse routes.

---

# 122. Stale route

If routing changed:

```text
STALE_ROUTE {
    TabletId,
    TabletEpoch,
    preferred_node,
    preferred_worker,
}
```

Client updates and retries.

---

# 123. Compatibility fallback

Any-node endpoint can accept an unknown request and perform at most a bounded internal forwarding step.

This ensures correctness for naive clients.

Native smart clients avoid this cost.

---

# 124. Kernel networking baseline

Before adopting fully custom user-space networking, benchmark ordinary:

```text
TCP
+
io_uring
```

because compatibility and operational simplicity are valuable.

---

# 125. TUX-style networking

TUX demonstrates a database-specific kernel-bypass design built around message semantics, XDP/eBPF and request pushdown while preserving compatibility with existing NIC drivers.

This motivates a future:

```text
NetworkFastPathProvider
```

that can bypass normal socket processing without requiring a DPDK-only deployment.

---

# 126. QUIC

QUIC is an optional transport for:

```text
WAN

edge

mobile clients

connection migration
```

`quinn-proto` is especially interesting because its protocol state machine is separable from OS networking, which aligns with deterministic simulation.

---

# 127. RDMA

RDMA is an accelerator backend.

It is not a correctness assumption.

Potential use:

```text
bulk chunks

replication

CXL-adjacent deployments

remote memory
```

The same high-level protocol semantics must remain available over TCP.

---

# 128. Traffic classes

At minimum:

```text
ConsensusCritical

Foreground

ReplicaCatchup

FailureRecovery

Migration

Checkpoint

Archive

Scrub
```

Bulk transfers cannot share one uncontrolled FIFO with consensus heartbeats.

---

# 129. DPA / SmartNIC Fabric

DPA-Store demonstrates that modern SmartNIC DPAs can perform meaningful database-path work, including index traversal and NIC-local caching, while leaving heavier structural updates on host CPUs.

Therefore a DPA may become an L0 data-path layer.

---

# 130. DPA responsibilities

Possible:

```text
request parsing

route lookup

hot read cache

immutable chunk cache

fence validation assistance

TLS

checksums

compression

RLNC recoding

bulk transfer
```

Not allowed to be the sole source of:

```text
authoritative mutable state

irrecoverable consensus truth
```

---

# 131. DPA fallback

Every accelerated operation has:

```text
DPA fast path
      │
      └── miss/unsupported/error
              ↓
           host path
```

A DPA failure degrades performance.

It must not corrupt logical availability.

---

# 132. P4 / programmable switches

In-network key/value and consensus research has shown that limited consistency roles can be offloaded to programmable switches, including recent P4KVS work separating a consensus role from normal replicas.

This remains Experimental.

---

# 133. ConsensusAccelerator

Future interface:

```rust
trait ConsensusAccelerator {
    fn capabilities();
    fn fast_path(...);
}
```

Possible providers:

```text
None

DPA

P4

CXLSharedMemory
```

Any uncertain condition falls back to host consensus.

---

# 134. Juneberry-style hardware ACK path

Juneberry demonstrates a particularly aggressive possibility: using NIC hardware acknowledgments as commit signals for an ordered storage queue while server CPU execution happens later.

This is NOT a general baseline commit mechanism.

It is a future:

```text
WireLatencyCommitProvider
```

only when hardware and durability semantics are formally sufficient for the Namespace contract.

---

# 135. CXL Fabric

CXL is not merely a slower DRAM tier.

It may become a coordination fabric.

---

# 136. CXL Pods

Tigon explores transactional database coordination across hosts sharing CXL memory.

The system therefore supports the conceptual topology:

```text
Cluster
  │
  ├── CXL Pod A
  │      ├── Node 1
  │      ├── Node 2
  │      └── Node 3
  │
  └── CXL Pod B
         ├── Node 4
         └── Node 5
```

Within a Pod:

```text
shared coherent metadata
```

may replace network messages for selected mechanisms.

Between Pods:

```text
ordinary distributed protocols
```

remain.

---

# 137. CXL correctness boundary

Do not place entire database semantics into one globally shared mutable CXL region.

Prefer:

```text
tiny high-churn coordination metadata
    → coherent shared region

large state
    → normal materialization/storage
```

This minimizes coherence cost and future hardware dependence.

---

# 138. Kernel-resident state

BPF-DB demonstrates that transactional state can be embedded inside eBPF/kernel contexts and can even act as a storage engine for Redis-compatible workloads.

This suggests an optional:

```text
KernelStateProvider
```

for use cases where kernel programs require ultra-local state.

---

# 139. KernelStateProvider constraints

Kernel-resident state is NOT the universal database.

It is for:

```text
packet-path counters

network policy

rate limits

kernel telemetry

security decisions
```

that benefit from avoiding a userspace transition.

It must expose a synchronization/export boundary back into the normal logical fabric.

---

# 140. Cache Fabric

Cache is not one eviction algorithm.

It contains:

```text
admission

eviction

materialization

replication

consistency

prefetch

client caching
```

---

# 141. Baseline cache algorithms

Research candidates:

```text
SIEVE

S3-FIFO

size-aware TinyLFU
```

SIEVE is especially interesting because it achieves a simple hit path and strong empirical results on large trace suites.

Selection is trace-driven.

---

# 142. Learning-Augmented Heuristics

Do not place a heavy ML model on each cache lookup.

Instead:

```text
simple fast heuristic
        ↑
small parameter vector
        ↑
async learned controller
```

This architecture follows the Learning-Augmented Heuristics idea: preserve simple request-path logic and let learning tune its parameters asynchronously.

This exactly matches:

```text
Exact State
Adaptive Control
```

---

# 143. Learning failure

An ML model may produce:

```text
worse cache parameters

bad prefetch decision

unnecessary migration
```

It may NEVER determine:

```text
whether a committed write exists

who owns a tablet

whether a transaction committed

whether durable data can be discarded
```

---

# 144. Observability Fabric

Observability is multi-layered.

Per:

```text
cluster

region

node

NUMA domain

worker

tablet

namespace

tenant

consensus group

storage device

network peer
```

---

# 145. Hot-path telemetry

No process-global atomic counter per request.

Workers keep local counters/histograms.

Aggregation is periodic.

---

# 146. Approximate telemetry

Hot-key and workload telemetry MAY use:

```text
Count-Min Sketch

SpaceSaving

sampling

probabilistic counters

adaptive sketches
```

Approximation affects optimization only.

---

# 147. Explainability

Administrative API includes:

```text
EXPLAIN KEY

EXPLAIN TABLET

EXPLAIN PLACEMENT

EXPLAIN DURABILITY

EXPLAIN HOTSPOT

EXPLAIN REDUNDANCY
```

---

# 148. EXPLAIN KEY

Example:

```text
Key:
    user:123

Namespace:
    sessions

Tablet:
    918

Tablet epoch:
    381

Authority:
    node-7 / worker-11

Consensus:
    Raft

Durability:
    3-voter local NVMe quorum

Materialization:
    local DRAM
    remote warm replica

Representation:
    Inline

Read contract:
    Latest

Traffic:
    14k reads/s
    80 writes/s

Criticality:
    medium

Last movement:
    3m ago

Reason:
    worker CPU rebalance
```

---

# 149. EXPLAIN REDUNDANCY

Example:

```text
Object generation:
    G8817

Lifecycle:
    warm immutable

Codec:
    RLNC GF(2^8)

Required rank:
    64

Verified rank:
    81

Hard minimum margin:
    8

Placement:
    7 nodes
    3 racks

Current condition:
    healthy

Next transition:
    wide-vector cold archive after 12h
```

---

# 150. Control Fabric

Control Plane stores exact topology/policy state.

Optimization controller makes adaptive physical decisions.

They are not the same component.

---

# 151. Exact Control Plane

Small replicated control state includes:

```text
ClusterId

Node membership

Node incarnation

Namespaces

Tablet directory

Tablet epochs

Replica membership

security policy

feature capability floor
```

---

# 152. Control-plane isolation

GET/SET do not synchronously contact the global control plane.

Existing tablets continue to serve while control quorum is temporarily unavailable.

Blocked operations include:

```text
new namespace

new node admission

split publication

merge publication

policy changes

feature activation
```

---

# 153. Node incarnation

Every process restart increments a durable:

```text
NodeIncarnation
```

Peers reject traffic from an older incarnation.

This prevents a stale restarted process from pretending it is the current node instance.

---

# 154. Placement planner

Planner considers:

```text
hard failure-domain constraints

CPU

DRAM

CXL

NVMe

network

NUMA locality

tail latency

tablet criticality

replication lag

hot keys

migration cost

energy/cost

hardware capabilities
```

---

# 155. Multi-timescale controllers

Different control loops run at different speeds.

```text
µs–ms:
    queueing
    batching
    request admission

100ms–seconds:
    cache tuning
    hot-key classification

seconds:
    worker balancing
    read materialization

tens of seconds/minutes:
    tablet movement
    splits
    replica changes

minutes/hours:
    compression
    tier transitions
    redundancy coding
    archive
```

This prevents one global controller from oscillating everything.

---

# 156. Hysteresis

Every automatic movement policy includes:

```text
minimum improvement

minimum sustained duration

cooldown

movement budget

rollback criteria
```

No object or tablet should bounce between tiers every few seconds.

---

# 157. Resource Fabric

The planner maps declarative intents onto available providers.

Example:

```text
Intent:
    read p99 <= 100 µs
    durable against two hosts
    cost medium

Hardware:
    DRAM
    NVMe
    DPA

Plan:
    authority in DRAM
    hot follower in DRAM
    third durable follower on NVMe
    DPA read cache
```

Different machine:

```text
Hardware:
    CXL pod
    PM
    NVMe

Plan:
    CXL coordination
    local DRAM materialization
    PM durability
```

Same logical namespace.

---

# 158. Geo Fabric

Multi-region operation exposes explicit choices.

---

# 159. Regional strong

Default strong geo mode:

```text
Region A:
    local strong quorum

Region B:
    asynchronous disaster-recovery copy
```

Low write latency.

Recent acknowledged writes may be lost under catastrophic loss of Region A before DR replication catches up.

This is explicitly documented.

---

# 160. Global strong

Quorum crosses regions.

```text
write latency
≈
WAN quorum latency
```

No marketing claim pretends otherwise.

---

# 161. Geo CRDT

Only supported semantic data types can be independently writable in disconnected regions.

Merge behavior is part of the type definition.

---

# 162. Bounded-staleness geo cache

A separate fast replication stream MAY provide:

```text
freshness SLO
```

without being the durability channel.

This mirrors the useful separation demonstrated by Skybridge.

---

# 163. TTL

Expiry is logical metadata:

```text
expires_at
```

not merely an in-process timer.

Read sees expired data as absent even if physical reclamation has not occurred.

---

# 164. Timer system

Per-worker hierarchical timer structures schedule cleanup.

Large expiry storms are processed in bounded batches.

No million-key midnight expiration blocks a worker uninterrupted.

---

# 165. Cache eviction semantics

In replicated logical caches:

```text
leader chooses victim
 ↓
Evict MutationIR
 ↓
replicas apply same logical removal
```

Physical cache admission history need not be replicated.

---

# 166. Lease primitive

Native Lease includes:

```text
LeaseId

FencingToken

Expiry
```

Every acquisition advances the fencing token.

External systems must reject stale fencing tokens.

Time expiry alone is not sufficient for external-resource safety.

---

# 167. Changefeeds

Mutation IR naturally powers changefeeds.

Per-tablet ordering:

```text
TabletId
TabletEpoch
CommitPosition
```

is guaranteed.

A global total order is not fabricated by default.

---

# 168. RSM changefeed delivery

Large cross-cluster changefeed paths may eventually use Picsou-like cross-RSM delivery rather than custom resend mechanisms.

---

# 169. WASM extensions

Extensions run in sandboxed WASM.

They receive:

```text
fuel

memory limit

deadline

declared key scope

deterministic host API
```

They cannot perform arbitrary direct mutation.

---

# 170. WASM mutation model

```text
stable snapshot
 ↓
WASM execution
 ↓
read versions + proposed mutations
 ↓
owner revalidates
 ↓
MutationIR
 ↓
normal commit
```

---

# 171. No trusted user commutativity claims

A WASM module cannot say:

```text
this is commutative
```

and gain a consensus fast path.

Only built-in or formally approved semantic operations receive that capability.

---

# 172. Security model

Cluster peer traffic is authenticated.

Default:

```text
mTLS
```

Client auth is namespace-aware.

Capabilities are scoped.

---

# 173. Direct chunk capabilities

A large-object reader may be redirected to a bulk chunk source.

Authority issues a short-lived capability binding:

```text
tenant/security domain

ChunkId

permission

expiry

optional client/request binding
```

Knowing a content hash is not authorization.

---

# 174. Encryption

At-rest encryption may initially be provided by:

```text
encrypted block devices
cloud disk encryption
filesystem encryption
```

Future native encrypted chunk storage is allowed.

Deduplication never overrides confidentiality policy.

---

# 175. Verification Fabric

This system is too distributed and too adaptive to rely on ordinary integration testing.

Correctness mechanisms are layered.

```text
Rust type system

typestate

property tests

model checking

formal protocol models

deterministic simulation

fuzzing

mutation testing

black-box consistency tests

runtime semantic checkers

integrity hashes

scrubbing
```

---

# 176. Deterministic simulation

This is mandatory from the beginning.

FoundationDB's engineering model demonstrates the power of deterministic whole-cluster simulation: physical interfaces are replaced with simulated ones and entire clusters can be replayed deterministically under machine, network, disk and clock faults.

Our production core therefore MUST abstract:

```text
time

network

storage

randomness

scheduler

failure injection
```

---

# 177. Deterministic core

Correctness logic does not call:

```text
Instant::now()

rand::thread_rng()

TcpStream::connect()

File::write()
```

directly.

Instead:

```text
Event
 ↓
State Machine
 ↓
Effects
```

Production effects use hardware.

Simulation effects use virtual components.

---

# 178. Simulation failures

Must model:

```text
packet drop

packet reorder

packet duplication

one-way partition

connection reset

slow network

node crash

stale process return

torn write

lost unsynced write

bit corruption

misdirected block

ENOSPC

slow disk

clock jump

CPU pause

storage disappearance

control-plane partition

migration failure

split failure
```

---

# 179. Formal protocol models

At minimum:

```text
tablet ownership

split

merge

replica migration

idempotency

transaction prepare/commit

large-chunk durability

RLNC generation state transitions

control-plane directory publication
```

use TLA+/equivalent models.

---

# 180. Executable model checking

Stateright is a strong candidate for executable Rust protocol models between TLA+ and production code.

Use it to explore:

```text
message permutations

failures

duplicate messages

state invariants
```

---

# 181. Storage verification

PoWER demonstrates practical verification techniques for crash consistency and corruption detection and includes a Verus-verified KV store.

Our architecture therefore strongly encourages targeted verification of the smallest most dangerous storage core:

```text
WAL publication

checkpoint install

chunk durable transition

metadata atomic publication

node incarnation

corruption detection
```

Not the entire system.

---

# 182. Runtime semantic checking

T2C demonstrates that tests can be generalized into low-overhead runtime semantic checkers capable of finding silent failures in production systems.

Our test/invariant framework SHOULD be designed so selected invariants can become:

```text
simulator assertion

model-check property

test assertion

sampled production checker
```

from one conceptual definition.

---

# 183. Silent CPU errors

Orthrus shows that silent computation corruption can be detected by asynchronous validation on other cores with bounded overhead in its evaluation.

For high-integrity Namespace profiles, the system MAY support:

```text
sampled shadow apply

cross-core recomputation

critical mutation validation
```

Mismatch quarantines the node/worker.

This is optional, not default.

---

# 184. Mutation testing

Tests should survive:

```text
cargo-mutants
```

style modifications.

Examples that MUST be detected:

```text
epoch < current
    changed to
epoch <= current

checksum check removed

dedup branch removed

quorum condition weakened

transaction abort branch inverted
```

---

# 185. Fuzzing

Permanent fuzz surfaces:

```text
native protocol

RESP

peer protocol

MutationIR

durable log

checkpoint manifest

chunk manifest

RLNC metadata

transaction record
```

---

# 186. Fault model

Production guarantees address crash faults and corruption.

Byzantine cluster members are not the default consensus threat model.

However untrusted external data sources and clients are considered malicious.

---

# 187. RLNC pollution threat

Because recoding can amplify bad coded data, RLNC data paths receive stronger source validation than ordinary immutable chunk copying.

Untrusted peer-to-peer recoding is disabled until an appropriate pollution-resistant authentication design is incorporated.

---

# 188. Node failure behavior

A dead node triggers:

```text
quorum assessment

leadership recovery

placement repair

information-rank assessment

serving-capacity repair
```

These are separate questions.

A tablet may be:

```text
safe but slow

safe but under-replicated

unavailable

fully healthy
```

---

# 189. Health model

No single Boolean `healthy`.

Expose:

```text
process_ready

control_quorum

data_serving

strong_write_capable

under_replicated_tablets

under_coded_generations

corrupt_representations

degraded_materializations

archive_lag
```

---

# 190. Split protocol

Adaptive split is explicit and fenced.

Conservative baseline:

```text
allocate child identities

commit BeginSplit in parent

capture base state

construct child states

mirror parent tail into child builders

choose cut index

commit SealSplit

briefly stop new writes

finish replay

activate children

atomically replace routing

parent becomes redirect tombstone
```

No dual-authority period.

---

# 191. Merge protocol

Conceptually reverse:

```text
allocate merged identity

build merged state

mirror source tails

seal sources

activate merged tablet

atomic directory replacement

sources become redirects
```

---

# 192. Tablet migration

Replica movement:

```text
create destination learner

transfer checkpoint/chunks

catch up ordered tail

verify health

safe consensus membership change

transfer leadership if needed

remove old replica

reclaim old state
```

---

# 193. Worker migration

Same-node worker movement does not change consensus membership.

It changes only local execution ownership.

---

# 194. WriteGuard during topology change

Every topology transition changes/fences the corresponding:

```text
WriteGuardGeneration
```

Old owner:

```text
may still execute code
```

but cannot commit authoritative mutations under the new generation.

---

# 195. Directory

Global routing state is immutable-snapshot based.

Workers consume:

```text
Arc<DirectorySnapshot>
```

published atomically/RCU-style.

No global routing read lock per request.

---

# 196. Sparse client routing

Native clients do not need the complete tablet map.

They cache only routes they actually encounter.

Unknown route:

```text
any known node
 ↓
resolve/forward once
 ↓
route returned
```

---

# 197. Control metadata scaling

Control plane stores compact exact metadata.

Fine-grained runtime telemetry stays outside consensus and is reconstructible.

Do not put every heat counter into control-plane Raft.

---

# 198. Exact vs approximate metadata

Examples:

Exact:

```text
Tablet range

Tablet epoch

Replica membership

Namespace policy
```

Approximate:

```text
hotness

criticality

future demand

cache probability

predicted failure risk
```

This distinction should be visible in type/module boundaries.

---

# 199. Learned control

Any learned controller publishes suggestions:

```text
PlacementSuggestion

CacheParameterSuggestion

PrefetchSuggestion
```

A deterministic policy engine verifies:

```text
hard constraints
```

before execution.

---

# 200. Resource reservations

Before accepting a mutation that grows state, reserve:

```text
memory

durability capacity

response storage

chunk staging

transaction intent state
```

Do not commit something that the leader cannot locally apply because it unexpectedly ran out of memory.

---

# 201. Backpressure

Every queue is bounded.

Every queue must answer:

```text
who owns it?

who pays for it?

what is its maximum?

what happens when full?

can a remote attacker fill it?

can a dead peer prevent reclamation?
```

---

# 202. Overload

Fast rejection is allowed.

```text
OVERLOADED
retry_after=...
```

is preferable to:

```text
wait in queue 10 seconds
then respond
```

for a microsecond-class datastore.

---

# 203. Scheduler resource classes

At minimum:

```text
Foreground

ConsensusCommit

ConsensusCatchup

EmergencyRecovery

Migration

Checkpoint

Coding

Archive

Scrub
```

No pure priority starvation.

Every necessary maintenance class receives a minimum budget.

---

# 204. Deadline-aware durability

Durability batching adapts to workload.

Flush conditions can include:

```text
bytes threshold

oldest request deadline

device queue opportunity

batch size

explicit barrier
```

Not:

```text
always sleep 1 ms
```

---

# 205. Networking fairness

One busy socket cannot monopolize a worker.

Each event-loop iteration enforces:

```text
packet budget

byte budget

CPU budget
```

---

# 206. Large reads

Large object transfer is separated from mutable root authority.

```text
authority validates root/version
 ↓
manifest + capabilities
 ↓
bulk chunks fetched from best valid sources
```

Sources may include:

```text
leader

follower

NVMe chunk server

DPA cache

remote soft cache
```

---

# 207. Direct local transport

Future same-machine clients MAY use:

```text
shared-memory rings
```

instead of TCP.

Immutable chunks are especially suitable for read-only mapped access.

This remains optional.

---

# 208. Local kernel state

Some ultra-fast host-local primitives MAY be mirrored into a BPF-DB-inspired kernel state plane for:

```text
network rate limits

policy state

fast counters
```

but the normal userspace authority remains recoverable.

---

# 209. Dependency architecture

External dependencies provide mechanisms.

The project owns semantics.

Rule:

> Borrow mechanisms. Own semantics.

Examples:

```text
borrow BLAKE3
own ChunkId

borrow Raft
own Tablet lifecycle

borrow io_uring runtime
own worker scheduler

borrow RESP parser
own Redis semantics

borrow RLNC implementation
own redundancy policy and format

borrow WASM runtime
own mutation semantics
```

---

# 210. Core mechanism crates

Current foundational choices include:

```text
bytes

zerocopy

hashbrown

blake3

crc32c

bitflags

arc-swap

jiff

futures

async-channel

crossbeam-channel

core_affinity2

thiserror
```

Runtime and server-specific choices are separated in §211–§212.

---

# 211. Runtime selection

The current data-plane runtime is Compio 0.19. Each networked `DataWorker` owns one single-threaded Compio runtime. OpenRaft consensus uses the official Compio runtime, and the native client remains blocking with no async runtime of its own.

Monoio and a custom io-uring reactor remain alternatives, not current workspace dependencies.

Do not add another data-plane runtime without measurements that justify the change.

---

# 212. Server and control runtime

The current server runtime is Tokio 1 with Axum and Tower. It hosts admin and control HTTP, plus the server-side `Send` fronts used by cluster mode. It does not schedule tablet state or own OpenRaft groups.

Single-node `DataWorker` execution remains on Compio. In cluster mode, Tokio drives native serving and the `Send` consensus fronts while each consensus group remains on its owning Compio reactor. Serde and JSON remain confined to admin and configuration boundaries.

---

# 213. Kameo

Approved outside correctness core.

Useful for:

```text
maintenance actors

placement jobs

archive coordinator

admin workflows

compute supervision
```

Not:

```text
DataWorker

Tablet

Raft state machine
```

---

# 214. Consensus selection

The current selection is:

```text
openraft =0.10.0-alpha.35
openraft-rt-compio =0.10.0-alpha.35
openraft-multi =0.10.0-alpha.35
```

The exact pin is intentional while 0.10 remains alpha. OpenRaft uses `single-threaded` and the official Compio runtime; `openraft-multi` provides shared group routing. Production and spike configurations enable pre-vote. No Tokio runtime is used for consensus group execution.

`raft-rs` is not selected or present in the workspace.

---

# 215. RESP crate

The current compatibility edge is `kivi-resp`, using `redis-protocol` 6.0 behind `kivi-server`'s `redis-compat` feature. It translates protocol syntax into Kivi semantics; it does not define durable formats or core state semantics.

If profiling proves parser CPU significant, replace internally.

---

# 216. RLNC candidates

Experimental:

```text
rlnc

rlnc-simdx
```

No crate's serialized representation becomes the durable format.

---

# 217. Compression

Candidates:

```text
lz4_flex
    hot/warm

zstd
    cold/archive
```

Codec stored in artifact metadata.

---

# 218. Object storage

Use an abstraction such as:

```text
object_store
```

for:

```text
S3

GCS

Azure

local object storage
```

Archive operations do not run directly on data workers.

---

# 219. WASM

Primary candidate:

```text
Wasmtime
```

Potential lightweight benchmark:

```text
Wasmi
```

---

# 220. Testing dependencies

```text
proptest

loom

shuttle

stateright

cargo-fuzz

cargo-mutants

cargo-nextest

cargo-llvm-cov
```

---

# 221. Benchmark tooling

```text
Divan

Criterion

Gungraun

perf-event

hdrhistogram
```

We care about:

```text
instructions/op

cache misses

branches

allocation

CPU/op

p99.99
```

not just average throughput.

---

# 222. Workspace architecture

The workspace is Rust 2024 with an MSRV of 1.98. Its current crates are grouped by responsibility rather than by future fabric names:

```text
crates/
    kivi-types
    kivi-core
    kivi-codec

    kivi-tablet
    kivi-state
    kivi-protocol
    kivi-client
    kivi-resp

    kivi-engine
    kivi-memory
    kivi-server

    kivi-durability
    kivi-checkpoint
    kivi-chunk

    kivi-consensus
    kivi-control
    kivi-redundancy
    kivi-observation

    kivi-sim
    kivi-lab
```

`kivi-server` is the production binary boundary. `kivi-resp` is the optional Redis/RESP edge. `kivi-sim` and `kivi-lab` provide deterministic simulation and test/benchmark support. Experimental mechanisms remain outside the production dependency path until selected.

The current workspace has no separate `experiments/` tree. Future experimental work may use separate crates for:

```text
experiments/

    runtime-compio

    runtime-monoio

    autonomous-commit

    rlnc

    stripeless

    wide-coding

    cxl

    dpa

    p4

    rdma

    userspace-preempt

    learned-index

    mphf

    kernel-state
```

Research code MUST NOT contaminate production core modules prematurely.

---

# 223. Unsafe policy

Unsafe is expected in carefully isolated modules:

```text
io_uring

SIMD

allocator

shared-memory transport

FFI

hardware acceleration
```

Every unsafe module documents:

```text
safety invariants

ownership

aliasing rules

threading assumptions

input trust
```

Distributed/state-machine logic remains safe Rust.

---

# 224. No canonical serde storage

Serde may be used for:

```text
admin JSON

config

tests

debug output
```

not:

```text
MutationIR

WAL

checkpoint canonical format

tablet metadata format
```

---

# 225. Configuration boundary

Local process configuration:

```text
listen address

paths

CPU set

logging

hardware options
```

comes from normal config.

Cluster state:

```text
namespace policy

tablet ownership

replication

security policy

feature activation
```

is replicated control state.

---

# 226. Rolling upgrades

Nodes advertise:

```text
wire protocol versions

MutationIR versions

checkpoint formats

coding formats

hardware capabilities

feature capabilities
```

Cluster computes a minimum compatible feature floor.

---

# 227. Feature activation

Upgrade sequence:

```text
new binaries deployed
 ↓
all necessary participants support feature
 ↓
control plane raises capability floor
 ↓
feature activated
```

Installing one new leader must not cause it to emit data old voters cannot replay.

---

# 228. Durable format evolution

Every durable artifact is versioned.

Old and new representations MAY coexist temporarily.

Large migrations should be:

```text
lazy

incremental

resumable
```

not stop-the-world.

---

# 229. Accelerator versioning

Specialized hardware capability is negotiated dynamically.

If:

```text
DPA v2
```

understands only coding protocol X:

the host planner uses X there or bypasses it.

No cluster-wide format depends on the lowest SmartNIC firmware unless explicitly required.

---

# 230. Research maturity levels

Every feature is assigned one of four maturity classes.

## Foundation

Required in core architecture.

## Production Candidate

Strong research/engineering basis; intended after validation.

## Experimental Accelerator

Potentially transformative, but optional.

## Watchlist

Interesting research with no current implementation commitment.

---

# 231. Foundation features

These belong in the architecture from day one:

```text
Exact State / Adaptive Everything Else

single-owner tablets

adaptive tablets

semantic operations

MutationIR

explicit read contracts

request identity/idempotency

declarative MaterializationIntent

DurabilityProvider abstraction

Redundancy Fabric abstraction

immutable content-addressed chunks

incremental fork-free checkpoints

independent consensus groups

direct smart-client routing

WriteGuard-style fencing primitive

bounded queues/work

deterministic simulation

typed persistence transitions

strong observability
```

---

# 232. Production Candidates

```text
Autonomous Commit

Almost-Local Reads

strong remote cache via WriteGuards

object-aware memory allocation

AOL-based materialization placement

RLNC for warm immutable state

stripeless coding placement

learning-augmented cache control

declarative maintenance I/O

Picsou-like RSM channels

verified storage subcomponents
```

---

# 233. Experimental Accelerators

```text
userspace interrupt preemption

CXL shared coordination

DPA lookup/cache path

DPA RLNC recoding

RDMA bulk path

Juneberry-style NIC commit

P4 consensus role

kernel-resident state

strong client cache leases

learned indexes

MPHF frozen bands
```

---

# 234. Watchlist

```text
future CXL fabrics

future persistent-memory generations

in-switch network coding

advanced homomorphic RLNC integrity

future regenerating codes

new wide-vector codes

new userspace scheduling hardware

processing-in-memory

near-memory compute

novel consensus protocols

quantum-assisted distributed coordination
```

Watchlist items have no implementation commitment.

---

# 235. Rejected foundations

The architecture explicitly rejects the following as universal foundations:

```text
Redis architecture clone

one global concurrent hash map

global work-stealing data path

fixed 16K-style slots

millions of tiny shards by default

pure consistent hashing as complete placement policy

one Raft group per server

one Raft group per key

fixed multi-tablet consensus cohorts

leaderless everything

CRDT everything

global strong consistency by default

universal LSM

universal MVCC

OS swap as tier

fork-based persistence

one global eviction algorithm

RDMA requirement

CXL requirement

DPA requirement

P4 requirement

ML-controlled correctness

one erasure code for every lifecycle

fixed stripes as universal coding unit
```

---

# 236. Benchmark philosophy

The system is not judged by:

```text
redis-benchmark GET
```

alone.

---

# 237. Steady-state workloads

Benchmark:

```text
GET

SET

INCR

mixed 95/5

mixed 50/50

write-heavy

tiny values

1 KB

64 KB

1 MB

large values

uniform access

Zipf

single hot key

many hot keys
```

---

# 238. Disruption workloads

Mandatory:

```text
checkpoint

tablet split

tablet migration

leader failure

node failure

full replica rebuild

RLNC repair

storage corruption

network partition

control-plane failure

memory pressure

TTL storm

NVMe saturation

scale out

scale in
```

---

# 239. Tail degradation metric

Measure:

```text
degradation =
    disrupted p99
    /
    steady p99
```

and similarly:

```text
p99.9

p99.99
```

A system that performs beautifully until checkpointing begins is not considered low-latency.

---

# 240. Competitors

Current comparison set should include contemporary:

```text
Redis

Valkey

Dragonfly

Garnet
```

and relevant specialized systems for specific experiments.

Benchmark configurations must be published.

---

# 241. Runtime experiments

The current baseline is Compio 0.19. Monoio and raw io-uring remain comparison experiments; neither is a current workspace dependency.

Measure:

```text
request latency

cross-core hops

syscalls

CPU/op

allocations

fairness

cancellation

large-stream behavior
```

---

# 242. Consensus experiments

Use the selected OpenRaft 0.10.0-alpha.35 configuration with the official Compio runtime. Measure increasing group counts with the density dimensions in §61, including idle CPU, memory per group, timers, network and storage integration, and simulation cost.

---

# 243. Durability experiments

```text
traditional group commit

Autonomous Commit

direct I/O variants

io_uring

quorum durability backend
```

Measure both:

```text
throughput
and
commit p99
```

---

# 244. Memory experiments

```text
size-only arenas

behavior-aware arenas

compressed DRAM

criticality-aware placement

software telemetry

hardware-assisted telemetry where available
```

---

# 245. Coding experiments

At minimum compare:

```text
3× replication

Leopard/RS

RLNC

systematic RLNC

stripeless XOR baseline

MSR/Clay-like

wide-vector coding where implementation exists
```

---

# 246. Coding metrics

Measure:

```text
storage overhead

encode CPU

decode CPU

recoding CPU

repair network bytes

repair disk bytes

repair wall time

degraded-read latency

metadata overhead

rank deficiency probability

helper count

normal-read penalty
```

---

# 247. RLNC-specific benchmark

Vary:

```text
generation size

symbol size

field

SIMD path

systematic ratio

coding density

recoder fan-in

network topology
```

Measure:

```text
innovative symbols/sec

rank growth

decode CPU

repair traffic
```

---

# 248. DPA experiments

Only after host path is solid.

Compare:

```text
host-only

DPA routing

DPA hot-read cache

DPA immutable chunk cache

DPA recoding
```

Every test includes accelerator failure/fallback.

---

# 249. CXL experiments

Compare:

```text
normal network coordination

CXL shared metadata

CXL state tier

CXL atomics for selected coordination
```

Do not assume the same result across vendors/topologies.

---

# 250. Definition of correctness

At all times:

```text
one valid authoritative lineage per key

no successful strong write without required commit

no committed durable root missing required durable information

same committed MutationIR produces equivalent logical state

retained request identity cannot execute twice

transaction never becomes partially committed

stale owner cannot mutate current state

unsafe optimization uncertainty always falls back
```

---

# 251. Mandatory invariants

### Ownership

```text
one mutable tablet owner per replica
```

### Authority

```text
one authoritative tablet lineage per logical partition
```

### Fencing

```text
stale TabletEpoch/WriteGuard cannot authorize mutation
```

### Strong Commit

```text
success means commit contract satisfied
```

### Apply

```text
success is not returned until required local logical apply
```

### Payload

```text
durably committed reference cannot point to unavailable required data
```

### Dedup

```text
same retained mutation identity cannot commit twice
```

### Transaction

```text
no externally visible partial commit
```

### Queues

```text
all queues bounded
```

### Control

```text
approximate controller cannot weaken safety
```

### Hardware

```text
accelerator failure falls back or degrades availability according
to declared policy; it cannot silently corrupt semantics
```

### Simulation

```text
every correctness-critical state machine must be executable
against deterministic virtual time/network/storage
```

---

# 252. Implementation sequence

The system should be built in layers.

The goal is NOT to implement all research mechanisms at once.

---

# 253. Phase 0 — Formal skeleton

Build:

```text
workspace

typed IDs

canonical codec primitives

effect interfaces

virtual clock

virtual network

virtual storage

simulation scheduler

tablet directory model

MutationIR model

TLA+ ownership model
```

No production server required.

---

# 254. Phase 1 — Single-node state engine

Build:

```text
pinned workers

worker-local allocation

native protocol

RESP adapter

Bytes

Counter

GET

SET

DEL

INCR

TTL

bounded scheduling
```

Benchmark runtime options.

---

# 255. Phase 2 — Persistence core

Build:

```text
DurabilityProvider

local NVMe provider

canonical log format

MutationIR replay

checkpoint

crash recovery

checksums

corruption injection
```

Prototype:

```text
group commit
vs
Autonomous Commit
```

---

# 256. Phase 3 — Large-object state

Build:

```text
content-addressed chunks

manifests

chunk staging

typed durability

chunk GC

incremental checkpoint reuse

streaming read/write
```

---

# 257. Phase 4 — Distributed consensus

Build:

```text
Kivi consensus front

OpenRaft 0.10.0-alpha.35 integration

group-density measurements

3-node cluster

leaders

read barriers

request dedup

replica catchup

failure recovery
```

---

# 258. Phase 5 — Multi-tablet system

Build:

```text
adaptive tablet directory

multiple groups/worker

physical batching

direct worker routing

client route cache
```

---

# 259. Phase 6 — topology

Build:

```text
node membership

incarnations

replica migration

worker migration

split

merge

node drain

placement controller
```

---

# 260. Phase 7 — advanced consistency

Build/experiment:

```text
WriteGuard caches

AtLeast reads

bounded-stale reads

Almost-Local Reads

roster leases
```

---

# 261. Phase 8 — transaction and semantics

Build:

```text
cross-tablet OCC + 2PC

Lease

Semaphore

BoundedCounter

CommutativeCounter

ShardedStream
```

---

# 262. Phase 9 — Memory Fabric

Build:

```text
MaterializationIntent

DRAM provider

compressed DRAM

NVMe demotion

behavior-aware arenas

criticality-aware placement
```

---

# 263. Phase 10 — Redundancy Fabric

Build baseline:

```text
replication

RS/Leopard baseline
```

Then experiments:

```text
RLNC

systematic RLNC

stripeless

MSR

wide-vector coding
```

---

# 264. Phase 11 — self-healing storage

Build:

```text
scrubbing

multi-source repair

coded recovery

adaptive redundancy

archive

object storage
```

---

# 265. Phase 12 — learning/control

Build:

```text
approximate sketches

adaptive cache tuning

predictive placement

failure-risk hints
```

Strict deterministic constraints surround all actions.

---

# 266. Phase 13 — hardware acceleration

Independent experimental tracks:

```text
CXL

DPA

RDMA

AF_XDP/TUX-style

userspace interrupts

P4

kernel state
```

None blocks product viability.

---

# 267. Phase 14 — network-coding fabric

After basic RLNC storage is proven, explore:

```text
intermediate recoding

topology-aware recoding

DPA recoding

WAN coding

adaptive generation placement

pollution-resistant untrusted recoders
```

This is where RLNC can move from an archive codec to a genuine network/storage fabric.

---

# 268. Product completeness baseline

The system is useful before experimental hardware exists.

A complete commodity deployment must work with:

```text
Linux

ordinary x86/ARM CPU

DRAM

NVMe

Ethernet

TCP

3+ servers
```

and provide:

```text
automatic partitioning

HA

durability

smart routing

transactions

Redis compatibility

native semantic API

adaptive memory

large objects

self-healing

observability
```

---

# 269. Future high-end deployment

A future fully accelerated environment could look like:

```text
Client
  │
  ▼
P4 / XDP route assist
  │
  ▼
DPA
  │
  ├── parsing
  ├── L0 cache
  ├── chunk serving
  ├── RLNC recoding
  │
  └── miss
       │
       ▼
Pinned Worker
       │
       ├── userspace-preemptible scheduling
       │
       ├── CXL-aware coordination
       │
       ▼
Tablet Authority
       │
       ▼
Consistency Fabric
       │
       ▼
Durability Fabric
       │
       ├── PM/CXL
       ├── autonomous NVMe
       └── quorum backend
       │
       ▼
Commit
```

Meanwhile:

```text
background state
    ↓
immutable checkpoint
    ↓
content-addressed chunks
    ↓
RLNC warm redundancy
    ↓
stripeless / topology-aware repair
    ↓
wide-vector cold coding
    ↓
object archive
```

And:

```text
memory telemetry
    ↓
criticality estimator
    ↓
learning controller
    ↓
resource planner
    ↓
DRAM / CXL / NVMe / DPA movement
```

None of these alter logical object identity.

---

# 270. Why this architecture is genuinely different

Traditional cache/database design often binds together:

```text
object
=
memory representation
=
authority
=
replica
=
durability copy
=
serving copy
```

This architecture intentionally breaks every equality.

Instead:

```text
Logical State
    ≠
Execution Location

Logical State
    ≠
Materialization

Authority
    ≠
Read Replica

Read Replica
    ≠
Consensus Voter

Consensus Voter
    ≠
Archive Copy

Archive Copy
    ≠
Full Replica

Durability
    ≠
Ordering

Ordering
    ≠
Execution

Redundancy
    ≠
Replication

Routing
    ≠
Physical Data Layout

Cache
    ≠
Weak Consistency
```

That decomposition is the main architectural advantage.

---

# 271. The deepest combination

The most important composition is:

```text
              LOGICAL OBJECT
                    │
                small root
                    │
          ┌─────────┴─────────┐
          │                   │
     inline state      immutable information
                              │
                   ┌──────────┼───────────┐
                   │          │           │
                 DRAM       NVMe       remote
                                          │
                                        DPA
                                          │
                                          ▼
                                    RLNC recoding
                                          │
                                          ▼
                                    cold coding
                                          │
                                          ▼
                                      archive
```

combined with:

```text
Semantic Operation
        │
        ├── strict
        │      ↓
        │   ordered consensus
        │
        ├── commutative
        │      ↓
        │   future CURP fast path
        │
        ├── bounded
        │      ↓
        │   escrow
        │
        └── eventual
               ↓
             CRDT
```

combined with:

```text
MaterializationIntent
        │
        ▼
Resource Planner
        │
        ├── DRAM
        ├── compressed DRAM
        ├── CXL
        ├── NVMe
        ├── remote RAM
        └── accelerator
```

combined with:

```text
Observation
    │
 approximate
    │
    ▼
Optimization Decision
    │
 hard safety validation
    │
    ▼
Physical Change
```

This is the architectural center of the project.

---

# 272. Research foundation

The architecture deliberately incorporates ideas from several independent research directions rather than betting on one system.

Important anchors include:

**Declarative heterogeneous memory:** Declarative Memory Services argues for describing desired memory properties rather than hard-coding devices.

**Object-aware tiering:** OBASE demonstrates the value of reorganizing objects based on behavior rather than size-only allocator layout.

**Criticality beyond hotness:** AOL-based tiering explicitly accounts for latency and memory-level parallelism rather than access count alone.

**Hardware telemetry:** NEMO demonstrates programmable memory-controller-level telemetry for memory policy decisions.

**CXL coordination:** Tigon explores database transactions across shared CXL memory.

**DPA execution:** DPA-Store demonstrates significant KV work on a BlueField DPA while retaining host involvement for complex structural operations.

**Kernel state:** BPF-DB demonstrates transactional database state directly supporting eBPF applications.

**Modern networking:** TUX demonstrates message-oriented database networking over XDP/eBPF while preserving driver compatibility.

**Wire-latency commit:** Juneberry explores hardware ACKs as commit signals with asynchronous CPU execution.

**Modern commit processing:** Autonomous Commit challenges centralized group commit on highly parallel NVMe.

**Composable durability:** LogDrive separates sequencing from durability and composes cloud-storage durability beneath an ordered log.

**Strong caches:** WriteGuards demonstrate range-level fencing supporting strongly consistent memory caches without coordination on ordinary reads.

**Asynchronous read limits:** the LAW theorem precisely constrains what can be achieved for local linearizable reads under asynchronous crash tolerance.

**Localized strong reads:** Bodega explores generalized responder rosters and leases for local linearizable reads.

**Adaptive execution:** PreemptDB and SBB show that emerging userspace interrupt mechanisms can reduce scheduling interference without abandoning userspace runtimes.

**Network coding:** RLNC enables intermediate recoding without decoding and remains actively implemented in modern Rust libraries.

**Network-coded storage:** NCBlob demonstrates practical repair benefits from network-coded MSR storage for warm blobs.

**Stripeless coding:** Nos/Nostor demonstrate that fixed stripes are not a necessary placement abstraction for erasure-coded in-memory storage.

**Wide coding:** WiseCode demonstrates current progress toward practical very-wide vector-coded storage with low overhead.

**Repair-aware coding:** LESS and DRBoost reinforce that repair I/O and degraded-read behavior matter alongside theoretical coding efficiency.

**Simple/adaptive caching:** SIEVE demonstrates that request-path simplicity can coexist with strong cache efficiency.

**Learning outside the hot path:** Learning-Augmented Heuristics demonstrates a useful pattern of asynchronously learning parameters for a simple fast heuristic.

**Crash safety through Rust types:** SquirrelFS demonstrates compile-time enforcement of persistence ordering with typestate.

**Verified storage:** PoWER demonstrates practical formal verification of crash consistency and corruption detection.

**Production semantic checking:** T2C demonstrates deriving runtime semantic checkers from tests.

**Silent compute faults:** Orthrus explores cross-core asynchronous validation to detect silent CPU data corruption.

**Deterministic simulation:** FoundationDB provides the clearest production precedent for deterministic whole-cluster fault simulation as an engineering foundation.

---

# 273. Final engineering rule

When evaluating any future paper, crate, protocol or hardware feature, ask:

```text
Does it improve a real bottleneck?

Can it fit behind an existing fabric?

Does its failure have a safe fallback?

Does it require weakening semantics?

Does it introduce a new single point of failure?

Can deterministic simulation model it?

Can it coexist with commodity hardware?

Can we remove it later without changing logical state?
```

If the answer to the last six questions is unfavorable, the technology probably does not belong in the core architecture regardless of benchmark numbers.

---

# 274. Final architecture statement

The project is a **future-native distributed state fabric**.

Its conservative heart is:

```text
typed logical state

single mutable ownership

explicit semantics

versioned fencing

deterministic mutation

strong baseline consensus

explicit durability

safe retries

bounded execution

verified persistent transitions
```

Around that heart sits a deliberately aggressive adaptive system:

```text
dynamic tablets

declarative materialization

behavior-aware memory

criticality-aware placement

strong cache fabrics

semantic fast paths

network coding

stripeless redundancy

self-healing repair

learning-augmented control

CXL coordination

DPA execution

kernel fast paths

modern NVMe

userspace preemption

future programmable network hardware
```

The aggressive parts are allowed to evolve rapidly because none of them owns the logical truth.

The architecture should therefore remain viable whether the dominant hardware of the next decade becomes:

```text
CXL
DPA
PIM
SmartNIC
new NVMe
new interconnect
new memory
```

or something that does not yet exist.

The system does not attempt to predict that winner.

It defines interfaces that allow the winner to become a provider.

The fundamental contract remains:

> **Keep truth small, explicit, deterministic and provable. Make everything physical adaptive, composable and replaceable.**

That is the architecture.
