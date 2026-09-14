# Kivi 3-node replicated cluster (product workflow)

One replicated tablet, three voting replicas, strong consistency. No test
code required: three real `kivi-server` processes plus the normal Kivi
client/CLI.

## Layout

```text
Node A: peer 127.0.0.1:9101, native 127.0.0.1:9201, admin 127.0.0.1:9301
Node B: peer 127.0.0.1:9102, native 127.0.0.1:9202, admin 127.0.0.1:9302
Node C: peer 127.0.0.1:9103, native 127.0.0.1:9203, admin 127.0.0.1:9303

ClusterId: any u128 (all three must agree, e.g. 12345)
Tablet: 9 (any u64; all three must agree)
```

Peer endpoints are QUIC/UDP (H3); native/admin/RESP stay TCP. Peer and
native listeners must use distinct ports: native redirects dial
the native endpoints, never the peer mesh.

## Start three servers

Terminal A:

```powershell
cargo run -p kivi-server -- `
  --cluster-mode `
  --cluster-id 12345 --node-id 1 --cluster-tablet 9 `
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
KIVI_READY native=127.0.0.1:9201 peer=127.0.0.1:9101 admin=127.0.0.1:9301 node=1 cluster=12345 incarnation=2
```

First startup bootstraps the group exactly once (initial membership from
the static topology, persisted). Restarts reuse membership and fail loudly
on wrong `ClusterId`/`NodeId` — never silently form another cluster.

## Normal client use (any node)

```powershell
# Build the CLI once
cargo run -p kivi-server --bin kivi-cli -- --help

# Write through any node (follower redirects to the leader internally)
cargo run -p kivi-server --bin kivi-cli -- --server 127.0.0.1:9202 set foo bar
cargo run -p kivi-server --bin kivi-cli -- --server 127.0.0.1:9202 get foo
# foo == bar (linearizable Latest read on the leader)
# With several seeds (leader discovery survives a dead seed):
cargo run -p kivi-server --bin kivi-cli -- --server 127.0.0.1:9201 --seed 127.0.0.1:9202,127.0.0.1:9203 get foo
```

The client preserves `SessionId`/`RequestSeq` across redirects, retries
bounded (`--max-redirects`, default 8), and surfaces `Overloaded`
(election in flight) instead of looping. Range routes remain authoritative;
per-`TabletId` leader hints are cached on every redirect for
introspection and future many-tablet routing (replicated clusters use a
root range covering every key, so the range hit already lands on the
leader).

## Failover drill (the product promise)

```text
1. SET x 1 via any node, GET x -> 1
2. kill -9 the leader process (find it via GET /v1/tablet on :9301-:9303)
3. SET x 2 via a survivor -> commits on the new leader
4. GET x -> 2
5. restart the killed process with the same data-dir/config
6. all three converge (GET x -> 2 everywhere, /v1/tablet applied equal)
```

Lost-response exactly-once: retry the same session/sequence after the
kill — the new leader returns the original outcome, the counter advances
once. See `crates/kivi-lab/tests/cluster_replication.rs`.

## Partition drill

```powershell
# Suspend peer links (bidirectional partition of node 1)
curl -X POST 127.0.0.1:9301/v1/peers/suspend -d '{"node":2}'
curl -X POST 127.0.0.1:9301/v1/peers/suspend -d '{"node":3}'
curl -X POST 127.0.0.1:9302/v1/peers/suspend -d '{"node":1}'
curl -X POST 127.0.0.1:9303/v1/peers/suspend -d '{"node":1}'

# Majority (2+3) elects and writes; the isolate cannot acknowledge.
# Heal when done:
curl -X POST 127.0.0.1:9301/v1/peers/resume -d '{"node":2}'
# ... resume every suspended direction, then wait for convergence.
```

Or use the lab harness `Cluster::partition/heal`, which does the same
through the admin plane. See
`crates/kivi-lab/tests/cluster_partition.rs`.

## Snapshot drill

```powershell
curl -X POST 127.0.0.1:9301/v1/snapshot
# Seals a checkpoint snapshot, purges the Raft log through its base.
# A follower behind the purge point installs the snapshot and resumes.
```

## Observability

```powershell
curl 127.0.0.1:9301/ready      # ready, leader_known, election_pending, role
curl 127.0.0.1:9301/v1/node    # cluster, node, incarnation, peer/native endpoints
curl 127.0.0.1:9301/v1/tablet  # role, term, leader, last_log, committed, applied, snapshot, purged, health
curl 127.0.0.1:9301/v1/peers   # per-lane H3 connections, requests, bytes, rtt_ms, active streams
curl 127.0.0.1:9301/v1/sidecar # bulk requests, cache hits/misses, verification failures, inflight
```

`/ready` is usable-only: `ready` requires a healthy replica with loaded
membership and started transport — never just a bound TCP listener.
`leader_known=false` means election in flight (still routable once a
leader emerges).

## Full restart

Stop all three (Ctrl-C or kill), restart with the same data dirs and
flags. No bootstrap reset, no manual repair: the cluster reforms, all
acknowledged data remains, new writes succeed.
See `full_cluster_restart_preserves_state`.

## Peer mesh: QUIC/H3

Peer traffic runs over QUIC with HTTP/3 request/response streams (Compio
`quic`/`h3`), two H3 connections per node pair: control (votes,
heartbeats, appends, snapshot fragments) and bulk (manifests, chunks).
The Raft log carries only tiny roots; followers fetch missing sidecars
over bulk streams and never acknowledge an append before every sidecar
is durable and verified.

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
`crates/kivi-lab/tests/cluster_large.rs` for the money tests and measured
benchmarks.

## Limits in this stage

* One tablet, static 3 voters. No add/remove voter, learners, split/merge.
* `Latest` on a follower redirects (`StaleRoute`); RESP followers answer
  retryable `BUSY` (never `MOVED`/`ASK`).
