# Kivi multi-tablet replicated cluster (product workflow)

Many replicated tablets, three voting replicas each, strong consistency
per tablet. No test code required: three real `kivi-server` processes
plus the normal Kivi client/CLI.

## Layout

```text
Node A: peer 127.0.0.1:9101, native 127.0.0.1:9201, admin 127.0.0.1:9301
Node B: peer 127.0.0.1:9102, native 127.0.0.1:9202, admin 127.0.0.1:9302
Node C: peer 127.0.0.1:9103, native 127.0.0.1:9203, admin 127.0.0.1:9303

ClusterId: any u128 (all three must agree, e.g. 12345)
Tablets: 16 (any nonzero count; all three must agree)
Workers: 4 consensus reactors per node (all three need not agree)
```

Peer endpoints are QUIC/UDP (H3); native/admin/RESP stay TCP. Peer and
native listeners must use distinct ports: native redirects dial
the native endpoints, never the peer mesh.

## Physical architecture

```text
Node
 ├─ ConsensusWorker 0 (one Compio reactor/thread)
 │    ├─ Tablet group 1 ── independent Raft + state machine
 │    ├─ Tablet group 5
 │    └─ ...
 ├─ ConsensusWorker 1 ── Tablet group 2, 6, ...
 └─ ...

one shared H3 peer mesh (control + bulk per node pair, all groups)
one shared WAL writer (cross-group fsync batching, all groups)
```

Tablet `T` lives on worker `(T-1) % workers` (deterministic local
striping; not distributed placement). Thread count scales with workers
plus fixed services — never with tablet count. Every tablet has its own
independent Raft leadership: different tablets routinely have different
leaders; there is no global current leader.

## Start three servers

Terminal A:

```powershell
cargo run -p kivi-server -- `
  --cluster-mode `
  --cluster-id 12345 --node-id 1 `
  --cluster-tablets 16 --cluster-workers 4 `
  --data-dir ./data-a `
  --cluster-peers 1=127.0.0.1:9101,2=127.0.0.1:9102,3=127.0.0.1:9103 `
  --cluster-natives 1=127.0.0.1:9201,2=127.0.0.1:9202,3=127.0.0.1:9203 `
  --cluster-native 127.0.0.1:9201 `
  --admin 127.0.0.1:9301
```

Terminal B: same with `--node-id 2`, `--data-dir ./data-b`,
`--cluster-native 127.0.0.1:9202`, `--admin 127.0.0.1:9302`.

Terminal C: same with `--node-id 3`, `--data-dir ./data-c`,
`--cluster-native 127.0.0.1:9203`, `--admin 127.0.0.1:9303`.

With `--features redis-compat`, add `--redis-listen 127.0.0.1:9401` (etc.)
per node for the RESP edge.

Each process prints one line when ready:

```text
KIVI_READY native=127.0.0.1:9201 peer=127.0.0.1:9101 admin=127.0.0.1:9301 node=1 cluster=12345 incarnation=2 tablets=16 consensus_workers=4 dir_version=...
```

First startup tiles the hash space into `--cluster-tablets` tablets,
forms every group exactly once (initial membership from the static
topology, persisted), and persists the tablet-set identity. Tablet ids
derive from the tiling — they are not operator-chosen — so changing
`--cluster-tablets` (or voters, or identities) on an existing data
directory fails loudly instead of silently forming a different cluster.
Restarts recover every replica and every Raft group (never re-bootstrap)
and fail loudly on any such disagreement — never a silent partial
cluster.

## Normal client use (any node, any tablet)

```powershell
# Build the CLI once
cargo run -p kivi-server --bin kivi-cli -- --help

# Writes route by key: hash -> tablet -> that tablet's leader
# (followers redirect with the tablet's real range; the client caches
# per-tablet routes sparsely and preserves mutation identities)
cargo run -p kivi-server --bin kivi-cli -- --server 127.0.0.1:9202 set foo bar
cargo run -p kivi-server --bin kivi-cli -- --server 127.0.0.1:9202 get foo
# foo == bar (linearizable Latest read on foo's tablet leader)
# With several seeds (leader discovery survives a dead seed):
cargo run -p kivi-server --bin kivi-cli -- --server 127.0.0.1:9201 --seed 127.0.0.1:9202,127.0.0.1:9203 get foo
```

Write keys that land on different tablets (any distinct keys spread
across the hash tiling), then compare per-tablet leaders:

```powershell
curl 127.0.0.1:9301/v1/tablets   # every local group: role, leader, applied, range
```

The client preserves `SessionId`/`RequestSeq` across redirects, retries
bounded (`--max-redirects`, default 8), and surfaces `Overloaded`
(election in flight) instead of looping. There is deliberately no
global leader: each tablet elects independently.

## Transactions and semantic types

Atomic multi-key writes, same API from one key to many:

```powershell
cargo run -p kivi-server --bin kivi-cli -- --server 127.0.0.1:9202 atomic-batch --write a=1,b=2
# both commit or neither does
```

Single-tablet batches commit server-side in one roundtrip. Batches
spanning tablets run client-driven OCC + 2PC with bounded retries: the
coordinator is always the lowest participant tablet (derived, never
configured — recovery re-drives agree with no control-plane
involvement), prepares carry a write-set digest binding the exact
transaction (a conflicting digest under one id is rejected, never
mixed in), and conflicts surface as `TxnConflict` after retries. Range
and predicate transactions are rejected loudly — point writes only.

Beyond bytes and strict counters, keys can hold typed values (client
methods, same routing and contracts as point ops):

```text
commutative counter   order-free adds, no ordinals exposed
                      (commutative_add / commutative_get)
bounded counter       capacity + escrow shares; adds stay inside owned
                      rights, rights move by paired transfer — narrowing
                      is unilateral, widening pairs in one atomic batch
                      (bounded_create / bounded_add / bounded_get)
semaphore             capacity permits, idempotent acquire by client-
                      minted permit id, releases never mint capacity
                      (semaphore_create / semaphore_acquire /
                      semaphore_release / semaphore_inspect)
lease                 fencing tokens advance on every grant; stale
                      tokens are powerless (renew/release reject them),
                      expiry frees the lease but only a newer token
                      retires the old holder
                      (lease_acquire / lease_renew / lease_release /
                      lease_inspect)
sharded stream        per-shard ordered logs; offsets are shard-local
                      and monotonic, trims never rewind, appends never
                      reuse (stream_create / stream_append /
                      stream_read / stream_trim)
```

Failures are machine-readable statuses, never strings:
`BoundedExceeded`, `SemaphoreExhausted`, `LeaseConflict`,
`StaleFencing`, `StreamFull`, `TxnConflict`, `TxnCrossTablet`.
Scans surface typed values as descriptors, never inlined bulk.

## Failover drill (the product promise)

```text
1. SET x 1 via any node, GET x -> 1 (x lives on some tablet T)
2. find T's leader: curl :9301/v1/tablets (or :9302, :9303)
3. kill -9 that leader process
4. every affected tablet elects a replacement independently
   (curl /v1/tablets: all groups have leaders again, others kept theirs)
5. SET x 2 via a survivor -> commits on the new leader
6. GET x -> 2
7. restart the killed process with the same data-dir/config
8. all tablets converge (GET x -> 2 everywhere, per-tablet applied equal)
```

Lost-response exactly-once: retry the same session/sequence after the
kill — the new leader returns the original outcome, the counter advances
once. See `crates/kivi-lab/tests/cluster_replication.rs` (single tablet)
and `cluster_multitablet.rs` (non-default tablet).

## Read contracts

Every native point read names its freshness contract on the wire
(`Latest` when absent — pre-contract clients only knew linearizable
reads, and `Latest` is the strongest contract, so the default never
weakens). Contract retries preserve it across redirects and redials.

```text
Latest
  Conservative linearizable read: a quorum ReadIndex barrier on the
  tablet's leader, then local applied state. A non-leader answers
  NotLeader (shaped as StaleRoute + leader redirect, or retriable
  Overloaded mid-election). Optional Almost-Local / roster-lease fast
  paths reuse explicit evidence and fall back to the barrier.

AtLeast(CommitToken)
  Serves from any replica whose applied state covers the token — no
  leader contact when already covered. A replica behind the token waits
  bounded (deadline-aware, default 5s server-side, hard-capped 10s).
  Tokens are lineage-bound (tablet + epoch): a token from another
  tablet, a superseded epoch (split/merge lineage), or an unassigned
  position rejects as StaleToken without waiting. Valid tokens survive
  failover, restart, and migration; retired lineages reject.

BoundedStale(max_staleness)
  Serves only from a FreshnessReceipt proving the bound (a recent
  authority proof on the server's monotonic clock, same authority,
  coverage applied). Without satisfying evidence the read escalates to
  a fresh barrier — which satisfies any bound — instead of serving weak
  data. If the authority path is unreachable the read fails retriable
  (Overloaded/StaleRoute), never out-of-bound data labeled success.
  Single-node servers are definitionally fresh and serve directly.

Any
  Local applied state with no freshness claim. May be arbitrarily stale;
  never blocks on coverage.
```

Every served read carries a proof (authority triple, applied position,
serving incarnation) on the response trailer. Feed `proof.token()` back
as `AtLeast(token)` to chain reads. Pre-proof servers omit it (treated
as no evidence, never as freshness). Client methods: `get` (Latest),
`get_with_contract`, `get_at_least`, `get_bounded_stale`, `get_any`.
Scans select per-tablet `Latest`/`Any` via the scan-consistency byte
(the multi-tablet scan is not one global snapshot); batches and scans
reject non-`Latest` contracts loudly.

## Partition drill

```powershell
# Suspend peer links (bidirectional partition of node 1)
curl -X POST 127.0.0.1:9301/v1/peers/suspend -d '{"node":2}'
curl -X POST 127.0.0.1:9301/v1/peers/suspend -d '{"node":3}'
curl -X POST 127.0.0.1:9302/v1/peers/suspend -d '{"node":1}'
curl -X POST 127.0.0.1:9303/v1/peers/suspend -d '{"node":1}'

# Majority (2+3) elects per tablet and writes; the isolate cannot acknowledge.
# Heal when done:
curl -X POST 127.0.0.1:9301/v1/peers/resume -d '{"node":2}'
# ... resume every suspended direction, then wait for convergence.
```

Or use the lab harness `Cluster::partition/heal`, which does the same
through the admin plane. See
`crates/kivi-lab/tests/cluster_partition.rs`.

## Snapshot drill (per tablet)

```powershell
curl -X POST 127.0.0.1:9301/v1/snapshot -d '{"tablet":4}'
# Seals a checkpoint snapshot for tablet 4, purges its Raft log through
# the base. Other tablets are untouched (logical snapshots stay
# group-scoped even though the WAL file is shared). A follower behind
# the purge point installs the snapshot and resumes.
```

## Observability

```powershell
curl 127.0.0.1:9301/ready        # ready only when EVERY tablet is healthy
curl 127.0.0.1:9301/v1/node      # cluster, node, tablets[], workers, dir_version
curl 127.0.0.1:9301/v1/tablets   # per tablet: worker owner, role, leader, term,
                                  #   committed, applied, snapshot, health, range
                                  #   + consistency_* counters (see below)
curl 127.0.0.1:9301/v1/tablet    # single-tablet clusters only (400 otherwise)
curl 127.0.0.1:9301/v1/peers     # per-lane H3 connections, requests, bytes, rtt_ms
curl 127.0.0.1:9301/v1/sidecar   # bulk requests, cache hits/misses, inflight
curl 127.0.0.1:9301/v1/durability # shared writer: batches, records/batch,
                                  #   groups/batch, fsyncs, queue, barrier latency
curl 127.0.0.1:9301/v1/preflight # preflight attempts, quorum-ready, fallbacks
```

Per-tablet `consistency_*` counters answer why reads took a given path
without reverse-engineering logs (reads per contract, local vs authority
serves, `AtLeast` waits/timeouts/lineage rejects, bounded-stale
hits/escalations/unprovable, strong-cache hits/fallbacks/invalidations,
Almost-Local and roster-lease hits/fallbacks, fencing rejects). Served
reads also log their path, authority, and position at debug level.

`/ready` is usable-only and multi-tablet aware: `ready` requires every
configured tablet healthy (with `tablets_total` / `tablets_healthy` /
`leaders_known` counts) — partial failure is reported openly, never
hidden behind one boolean.

## Full restart

Stop all three (Ctrl-C or kill), restart with the same data dirs and
flags. No bootstrap reset, no manual per-group repair: every group
reforms, all acknowledged data on every tablet remains, new writes
succeed. See `full_cluster_restart_preserves_state` (single tablet) and
the multi-tablet money flow in `cluster_multitablet.rs`.

## Peer mesh: QUIC/H3

Peer traffic runs over QUIC with HTTP/3 request/response streams (Compio
`quic`/`h3`), two H3 connections per node pair: control (votes,
heartbeats, appends, snapshot fragments, preflight acks) and bulk
(manifests, chunks). The Raft log carries only tiny roots; large values
pre-distribute sidecars over bulk *before* the root proposes (preflight
to a write quorum), so small same-tablet writes stay responsive while
bulk moves. Followers still gate every append on sidecar durability —
preflight is only an optimization, never a correctness dependency
(`KIVI_DISABLE_PREFLIGHT=1` forces the old append-gate path; lab only).

TLS uses per-node self-signed certificates with statically pinned trust
anchors. First boot mints the certificate under `<data-dir>/peer-tls/`;
print it for operator configuration:

```powershell
cargo run -p kivi-server --bin kivi-peer-cert -- --data-dir ./data-a --node-id 1
# node=1 fingerprint=<hex>
# <hex DER>
```

Pass each peer's DER file via `--cluster-peer-certs 2=/path/b.der,...`.
Without pins the node opens only with `KIVI_INSECURE_PEER_TLS=1`, which
is lab-tests-only (loopback) and never production.

Large values replicate end to end: `put_stream`/`get_stream` up to 64 MiB
(and beyond within the manifest entry cap), chunk-aware `SetRange`,
deduplicated sidecar transfer, differential snapshot catch-up. See
`crates/kivi-lab/tests/cluster_large.rs` (single tablet) and
`cluster_multitablet_preflight.rs` / `cluster_multitablet_bench.rs`
(multi-tablet preflight, isolation, and timing decomposition).

## Limits in this stage

* Dynamic placement (migration, repair, split, merge, drain) runs through
  the replicated control plane with learner catch-up, joint-consensus
  membership changes, and tombstoned parents. Read contracts hold across
  all of it: evidence is reconciled against the serving authority on
  every read, retired replicas drop theirs, and foreign lineages reject.
* `Latest` on a non-leader redirects (`StaleRoute` with the tablet's
  range); RESP followers answer retryable `BUSY` (never `MOVED`/`ASK`).
* Consistency fast paths (Almost-Local, roster leases, strong cache) are
  correctness-complete prototypes behind explicit modes: the conservative
  leader barrier is always available and is the default. Roster-lease
  issuance is leader-local (quorum-agreed issuance is future control-plane
  work); the synchrony assumptions are documented on the provider modes.
* Streamed reads (`get_stream`) honor the request contract for their root
  but carry no proof trailer: chain `AtLeast` from a point read.
