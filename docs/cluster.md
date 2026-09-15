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
curl 127.0.0.1:9301/v1/tablet    # single-tablet clusters only (400 otherwise)
curl 127.0.0.1:9301/v1/peers     # per-lane H3 connections, requests, bytes, rtt_ms
curl 127.0.0.1:9301/v1/sidecar   # bulk requests, cache hits/misses, inflight
curl 127.0.0.1:9301/v1/durability # shared writer: batches, records/batch,
                                 #   groups/batch, fsyncs, queue, barrier latency
curl 127.0.0.1:9301/v1/preflight # preflight attempts, quorum-ready, fallbacks
```

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

* Static placement: every node hosts one replica of every tablet, fixed
  3 voters per group. No add/remove voter, learners, split/merge,
  migration, or control-plane Raft.
* `Latest` on a non-leader redirects (`StaleRoute` with the tablet's
  range); RESP followers answer retryable `BUSY` (never `MOVED`/`ASK`).
