# Kivi RESP Compatibility Profile v1

Kivi is **not** a Redis drop-in replacement. This document defines the
intentional subset the optional RESP edge speaks. Future Redis releases do
not change Kivi behavior automatically; every new command or option is an
explicit Kivi compatibility decision.

Architecture:

```text
                 Kivi semantic operations
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
      Native Kivi Protocol        Redis/RESP Adapter
        first-class API            compatibility edge
             │                         │
             ▼                         ▼
       Kivi native client       redis-cli / Redis clients
```

Rules:

- The **native protocol is the unrestricted first-class interface**. Anything
  without a good Redis equivalent (typed `StrictCounter` semantics, native
  streaming, future consistency contracts/commit tokens/leases) stays
  native-only. No bizarre Redis command names are invented for them.
- The adapter **translates** (`RESP → typed Kivi operation → same engine`).
  It never opens a native TCP connection or round-trips a native frame
  inside the process.
- Conditional writes stay **atomic at the owning tablet** via the Kivi-owned
  `SetCondition` (`Always`/`IfAbsent`/`IfPresent`) + `ExpiryPolicy`
  (`Clear`/`Keep`/`ExpireAt`) model. There is no `GET`-then-`SET` in the
  adapter, and no atomic Redis operation is faked with a loop.
- Single-key forms only. Multi-key `DEL`/`EXISTS`/`MGET`/`MSET` are rejected
  with a clear error because Kivi routes keys to different tablets (later
  nodes); a loop would fake atomicity Redis clients could observe.

## Enabling

Compile-time (default-off):

```toml
kivi-server --features redis-compat
```

A build without the feature pulls neither `kivi-resp` nor `redis-protocol`
(verify with `cargo tree -p kivi-server`).

Runtime (even when compiled in, native-only by default):

```sh
kivi-server start --redis-listen 127.0.0.1:6379 --redis-namespace 1
```

No `--redis-listen` means no RESP service. Redis DB 0 maps to the configured
namespace; any other index is rejected (Kivi namespaces are not Redis DBs).

## Protocol versions

Connections start in **RESP2** (normal Redis behavior). `HELLO 2`/`HELLO 3`
switch the per-connection encoding; `HELLO` without args reports the current
context. RESP2 uses legacy nil (`$-1`); RESP3 uses `Null`/`Map`/`Blob`
forms. `HELLO` with `AUTH` always errors clearly (auth is a separate future
feature, never silently accepted).

Bootstrap commands: `PING`, `ECHO`, `HELLO`, `CLIENT SETINFO/SETNAME/GETNAME`,
`SELECT 0`, `QUIT`. `AUTH` errors clearly.

## Support table

`Exact` = verified against official Redis docs. `ProfileDeviation` = works
with a documented difference. `Unsupported` = recognized, clear error, never
faked. This table is generated from `kivi-resp`'s `REGISTRY`, which also
drives dispatch validation and `COMMAND` output — claims cannot drift.

| Command | Status | Notes |
|---|---|---|
| AUTH | Unsupported | no ACL system; always errors, never succeeds silently |
| CLIENT | Exact | SETINFO/SETNAME/GETNAME; other subcommands rejected |
| COMMAND | ProfileDeviation | COUNT/INFO/DOCS/LIST from the registry; not the full Redis metadata universe |
| DECR / DECRBY | Unsupported | `StrictCounter` ≠ Redis String-ints; `GET`-parse-`SET` forbidden |
| DEL | Exact | single-key only |
| ECHO | Exact | binary-safe |
| EXISTS | Exact | single-key only |
| EXPIRE / PEXPIRE / EXPIREAT / PEXPIREAT | Exact | bare forms only; `NX`/`XX`/`GT`/`LT` error instead of racing a read-then-write |
| EXPIRETIME / PEXPIRETIME | Exact | absolute stamp; `-1` immortal, `-2` missing |
| GET | Exact | missing → nil; counters → `WRONGTYPE` |
| GETRANGE | Exact | inclusive, negatives from end, clamped; missing → empty string (not nil) |
| HELLO | ProfileDeviation | version switch + `SETNAME`; `AUTH` errors without switching |
| INCR / INCRBY | Unsupported | `StrictCounter` ≠ Redis String-ints; `GET`-parse-`SET` forbidden |
| MGET / MSET | Unsupported | cross-tablet atomicity not faked |
| PERSIST | Exact | `1` removed, `0` absent/immortal |
| PING | Exact | `PONG` or echo |
| PTTL / TTL | Exact | `-2` missing, `-1` immortal |
| QUIT | Exact | `+OK` then close |
| SELECT | ProfileDeviation | `0` only |
| SET | Exact | `NX`/`XX` + `EX`/`PX`/`EXAT`/`PXAT`/`KEEPTTL` atomic; `GET`/`IFEQ` family unsupported |
| SETRANGE | Exact | zero-pads, answers new length, preserves TTL; empty patch is a pure length read (missing → `0`, never created); counters `WRONGTYPE` |
| STRLEN | Exact | missing → `0`; counters `WRONGTYPE` |

### SET details

- `NX` → `IfAbsent`, `XX` → `IfPresent`; `NX`+`XX` is a syntax error.
- `EX`/`PX` (relative) anchor to absolute stamps at the frontend;
  `EXAT`/`PXAT` are absolute; `KEEPTTL` preserves; default clears
  (successful plain `SET` always discards prior TTL, like Redis).
- Non-positive `EX`/`PX`/`EXAT`/`PXAT` values are rejected where Redis
  rejects them; conflicting expiries are rejected.
- `SET ... GET` (return-previous) is unsupported: it would force
  materialization of arbitrarily large chunked values into a bounded reply.
- Newer `IFEQ`/`IFNE`/`IFDEQ`/`IFDNE` options are unsupported.

### Counters

Redis integer commands operate on String contents; Kivi counters are a
separate `StrictCounter` type (a `GET` on one is `WRONGTYPE` by design).
Mapping `INCR` onto `StrictCounter` would silently change observable types,
so v1 leaves the whole family unsupported with a clear error instead of an
approximate one. A future Kivi-native numeric-bytes semantic — useful on its
own, not just for Redis — could enable exact mappings later.

### Errors

| Kivi outcome | Redis reply |
|---|---|
| missing `GET` | nil (`$-1` / `_`) |
| missing `GETRANGE` | empty string (`$0`) |
| missing `STRLEN` | `0` |
| missing `DEL` | `0` |
| wrong type | `WRONGTYPE ...` |
| bad args/arity | `ERR ...` |
| unknown command | `ERR unknown command '...'` |
| recognized but unsupported | `ERR unsupported command '...' in Kivi RESP profile v1` |
| overload/saturation | `BUSY ...` (retry with backoff) |
| internal | `ERR internal error` |

Rust internals never cross the wire. Every error prefix above is covered by
integration tests over a raw socket.

## Bounds (no bypass of Kivi limits)

- Connection input ≤ 64 MiB buffered; bulk strings ≤ 64 MiB (larger payloads
  belong on native streaming); arrays ≤ 256 elements; commands ≤ 32 args;
  `SET` values ≤ 64 MiB (the legacy ceiling).
- Pipelines are bounded (per-turn budgets, then yield) and always answered
  in request order. One huge pipeline cannot monopolize the frontend.
- Oversized input errors instead of allocating attacker-claimed lengths.

## Benchmarks

`kivi-bench --target native|resp` runs the same `Workload` definitions
through both adapters; `RespTarget` also points at real Redis/Valkey/
Dragonfly for the common subset. Native remains the performance reference:
RESP reasonably costs extra parsing, translation, cross-worker dispatch, and
ordered-pipeline bookkeeping. See the stage report for measured numbers.

## Out of scope (unchanged)

Cluster slots/`MOVED`, Lua, `MULTI`/`EXEC`, Pub/Sub, Streams, modules, ACLs,
and every Redis data type beyond strings (lists/hashes/sets/sorted sets
appear only when Kivi independently owns equivalent semantics).
