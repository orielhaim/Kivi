//! `kivi-bench`: load generator with hdrhistogram latencies.
//!
//! Architecture (for the future Redis comparison suite):
//!
//! ```text
//! Workload (target-agnostic, shared)
//!     ↓
//! Target Adapter
//!        ├── Kivi native (here)
//!        └── future Redis (same workload, different adapter)
//! ```
//!
//! The `workload` section defines everything shared: operation mix,
//! key distribution, value sizes (currently fixed 1-byte `v`; the field
//! exists so value-size distributions land here, not in adapters),
//! thread/client count, pipeline depth (threads == depth: one outstanding
//! op each), duration/op count, and hot-key patterns. It knows nothing
//! about Kivi types (`Key`, `NativeClient`) — keys are plain strings.
//!
//! The `kivi_target` section adapts that workload to Kivi native
//! (`String` → `Key`, `WorkloadOp` → typed client calls). A future Redis
//! adapter implements the same `BenchTarget` trait against the same
//! `WorkloadOp` stream, so identical workloads run against both targets.
//! No Redis dependency exists yet by design.
//!
//! Drives the real TCP path (never the embedded channel). Each thread owns
//! a cloned target (one outstanding request per thread) and records
//! nanosecond latencies locally; histograms merge at the end.

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

use bytes::Bytes;
use clap::{Parser, ValueEnum};
use hdrhistogram::Histogram;
use kivi_client::{ClientConfig, NativeClient};
use kivi_state::Key;
use kivi_types::NamespaceId;

// ---------------------------------------------------------------------------
// Workload: target-agnostic shared definitions.
// ---------------------------------------------------------------------------

/// Shared request mix. Target-agnostic: describes *what* to do, never *how*
/// a specific database executes it. Both the Kivi adapter (here) and the
/// future Redis adapter consume the same mix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Workload {
    /// Repeated GETs of a pre-populated keyspace.
    Get,
    /// SETs of small values.
    Set,
    /// Counter increments.
    Counter,
    /// Mixed GET/SET/DELETE/COUNTER across the keyspace.
    Mixed,
}

/// Shared key distribution. Target-agnostic: hot-key patterns and uniform
/// key counts live here so every adapter hammers the same keyspace shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeySpace {
    Hot,
    Uniform(usize),
}

/// One target-agnostic workload operation. Keys are plain strings; value
/// bytes are fixed 1-byte `v` today (see `BENCH_VALUE`). Adapters map this
/// onto their native calls (`Key` + typed client ops for Kivi).
#[derive(Debug, Clone, PartialEq, Eq)]
enum WorkloadOp {
    Get(String),
    Set(String),
    CounterAdd(String),
    Delete(String),
}

/// Fixed benchmark value (1 byte). Future value-size distributions
/// (1KB/64KB/1MB/large) become a workload parameter here, consumed by every
/// adapter — never hard-wired per target.
const BENCH_VALUE: &[u8] = b"v";

/// Plain key names for one thread's byte keyspace. Counter keys stay
/// separate (see [`counter_key_name`]): mixing types on one key is
/// `WrongType` by design, so workloads seed and hammer the type they
/// measure.
fn thread_key_names(space: KeySpace, thread: usize) -> Vec<String> {
    match space {
        KeySpace::Hot => vec!["hot".to_owned()],
        KeySpace::Uniform(count) => (0..count).map(|i| format!("t{thread}:k{i}")).collect(),
    }
}

/// Dedicated per-thread counter key name (never shares with byte keys).
fn counter_key_name(space: KeySpace, thread: usize) -> String {
    match space {
        KeySpace::Hot => "hot:c".to_owned(),
        KeySpace::Uniform(_) => format!("t{thread}:c"),
    }
}

/// Pure workload generation: the `i`-th op of `workload` over `keys`.
/// No I/O, no client types — identical for every future target adapter.
fn workload_op_for(workload: Workload, i: usize, keys: &[String], counter: &str) -> WorkloadOp {
    let key = keys[i % keys.len()].clone();
    match workload {
        Workload::Get => WorkloadOp::Get(key),
        Workload::Set => WorkloadOp::Set(key),
        Workload::Counter => WorkloadOp::CounterAdd(counter.to_owned()),
        Workload::Mixed => match i % 4 {
            0 => WorkloadOp::Get(key),
            1 => WorkloadOp::Set(key),
            2 => WorkloadOp::CounterAdd(counter.to_owned()),
            _ => WorkloadOp::Delete(key),
        },
    }
}

/// Seed operations for one thread's keyspace, in the exact order the
/// benchmark has always used: byte keys first (in key order), then the
/// counter key. Adapters execute these before measuring.
fn seed_ops_for(space: KeySpace, thread: usize, workload: Workload) -> Vec<WorkloadOp> {
    let mut ops = Vec::new();
    let seed_bytes = matches!(workload, Workload::Get | Workload::Set | Workload::Mixed);
    let seed_counters = matches!(workload, Workload::Counter | Workload::Mixed);
    if seed_bytes {
        if matches!(space, KeySpace::Hot) {
            if thread == 0 {
                ops.push(WorkloadOp::Set("hot".to_owned()));
            }
        } else {
            for key in thread_key_names(space, thread) {
                ops.push(WorkloadOp::Set(key));
            }
        }
    }
    if seed_counters {
        ops.push(WorkloadOp::CounterAdd(counter_key_name(space, thread)));
    }
    ops
}

// ---------------------------------------------------------------------------
// Target Adapter: Kivi native (future Redis adapter implements BenchTarget).
// ---------------------------------------------------------------------------

/// Minimal target interface the workload runner needs. The runner only sees
/// [`WorkloadOp`]; this adapter owns every Kivi-specific conversion.
trait BenchTarget: Clone + Send + Sync + 'static {
    /// Executes one workload op. `Ok` counts as completed (reads of absent
    /// keys are `Ok` — seeding makes absence rare, deletes make it legal).
    fn execute(&self, op: &WorkloadOp) -> Result<(), String>;
}

/// Kivi native adapter: [`WorkloadOp`] → [`Key`] + typed [`NativeClient`].
#[derive(Debug, Clone)]
struct KiviTarget {
    client: NativeClient,
}

impl BenchTarget for KiviTarget {
    fn execute(&self, op: &WorkloadOp) -> Result<(), String> {
        match op {
            WorkloadOp::Get(key) => self
                .client
                .get(&Key::from(key.as_str()))
                .map(|_| ())
                .map_err(|e| e.to_string()),
            WorkloadOp::Set(key) => self
                .client
                .set(&Key::from(key.as_str()), Bytes::from_static(BENCH_VALUE))
                .map_err(|e| e.to_string()),
            WorkloadOp::CounterAdd(key) => self
                .client
                .counter_add(&Key::from(key.as_str()), 1)
                .map(|_| ())
                .map_err(|e| e.to_string()),
            WorkloadOp::Delete(key) => self
                .client
                .delete(&Key::from(key.as_str()))
                .map(|_| ())
                .map_err(|e| e.to_string()),
        }
    }
}

impl KiviTarget {
    /// Counter-add with explicit delta (seeding uses 0 to create without
    /// counting; the workload itself always adds 1 via [`BenchTarget`]).
    fn seed_counter(&self, key: &str) -> Result<(), String> {
        self.client
            .counter_add(&Key::from(key), 0)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

#[derive(Debug, Parser)]
#[command(name = "kivi-bench", about = "Kivi native-protocol benchmark")]
struct Args {
    /// Seed endpoint.
    #[arg(long, default_value = "127.0.0.1:9000")]
    server: String,
    /// Target namespace id.
    #[arg(long, default_value_t = 1)]
    namespace: u64,
    /// Client threads (== pipeline depth: one outstanding op each).
    #[arg(long, default_value_t = 8, value_parser = parse_nonzero)]
    threads: usize,
    /// Operations per thread.
    #[arg(long, default_value_t = 5_000, value_parser = parse_nonzero)]
    ops: usize,
    /// Request mix.
    #[arg(long, value_enum, default_value_t = Workload::Mixed)]
    workload: Workload,
    /// Key distribution: `hot` (one key) or `uniform:<distinct-keys>`.
    #[arg(long, default_value = "uniform:256", value_parser = parse_key_space)]
    keys: KeySpace,
    /// Per-connection in-flight cap (fast-fail past this).
    #[arg(long, default_value_t = 256)]
    pending: usize,
}

/// Parses a nonzero count without panicking: bad input is a Clap usage
/// error, never a panic.
fn parse_nonzero(spec: &str) -> Result<usize, String> {
    let count: usize = spec
        .parse()
        .map_err(|_| format!("invalid count {spec:?}"))?;
    if count > 0 {
        Ok(count)
    } else {
        Err("count must be nonzero".to_owned())
    }
}

fn parse_key_space(spec: &str) -> Result<KeySpace, String> {
    if spec == "hot" {
        Ok(KeySpace::Hot)
    } else if let Some(count) = spec.strip_prefix("uniform:") {
        let count: usize = count
            .parse()
            .map_err(|_| format!("invalid key count in {spec:?}"))?;
        if count > 0 {
            Ok(KeySpace::Uniform(count))
        } else {
            Err(format!("key count must be nonzero in {spec:?}"))
        }
    } else {
        Err(format!("keys must be `hot` or `uniform:<n>`, got {spec:?}"))
    }
}

/// Pre-populates every key the workload touches via the target adapter.
/// Byte keys seed `v`; counters seed `0` (create without counting).
fn seed(
    target: &KiviTarget,
    space: KeySpace,
    threads: usize,
    workload: Workload,
) -> anyhow::Result<()> {
    for thread in 0..threads {
        for op in seed_ops_for(space, thread, workload) {
            match op {
                WorkloadOp::Set(key) => {
                    target
                        .execute(&WorkloadOp::Set(key.clone()))
                        .map_err(|error| {
                            anyhow::anyhow!("seed thread {thread} byte keyspace: {error}")
                        })?;
                }
                WorkloadOp::CounterAdd(key) => {
                    target.seed_counter(&key).map_err(|error| {
                        anyhow::anyhow!("seed thread {thread} counter: {error}")
                    })?;
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn kivi_target(server: &str, namespace: u64, pending: usize) -> anyhow::Result<KiviTarget> {
    use anyhow::Context;
    let client = NativeClient::new(ClientConfig {
        seeds: vec![server.to_owned()],
        namespace: NamespaceId::from_u64(namespace),
        max_pending: pending,
        ..ClientConfig::default()
    })
    .context("client setup failed")?;
    Ok(KiviTarget { client })
}

fn main() -> anyhow::Result<()> {
    use anyhow::Context;
    let args = Args::parse();
    let space = args.keys;
    let server = args.server.clone();
    let target = kivi_target(&server, args.namespace, args.pending)?;
    seed(&target, space, args.threads, args.workload).context("keyspace seeding failed")?;
    let barrier = Arc::new(Barrier::new(args.threads + 1));
    let mut handles = Vec::with_capacity(args.threads);
    for thread in 0..args.threads {
        // One client (and therefore one retry session) per thread.
        // Sessions are sequential streams: sharing one session across
        // threads couples their acknowledgement floors, so a single
        // straggler stalls every sharer. Per-thread sessions isolate
        // faults and let the server batch across them freely.
        let target = kivi_target(&server, args.namespace, args.pending)
            .context("bench client setup failed")?;
        let gate = Arc::clone(&barrier);
        let keys = thread_key_names(space, thread);
        let counter = counter_key_name(space, thread);
        let workload = args.workload;
        let ops = args.ops;
        handles.push(thread::spawn(move || {
            let mut histogram = Histogram::<u64>::new(3).expect("histogram builds");
            let mut errors = 0u64;
            let mut counter_done = 0u64;
            gate.wait();
            for i in 0..ops {
                let op = workload_op_for(workload, i, &keys, &counter);
                let start = Instant::now();
                let outcome = target.execute(&op);
                let nanos = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
                if outcome.is_err() {
                    errors += 1;
                } else {
                    counter_done += 1;
                    histogram.record(nanos).expect("latency fits");
                }
            }
            (histogram, errors, counter_done)
        }));
    }
    let wall = Instant::now();
    barrier.wait();
    let mut merged = Histogram::<u64>::new(3).expect("histogram builds");
    let mut errors = 0u64;
    let mut completed = 0u64;
    for handle in handles {
        let (histogram, thread_errors, thread_done) = handle.join().expect("thread joins");
        merged.add(&histogram).expect("histograms merge");
        errors += thread_errors;
        completed += thread_done;
    }
    let secs = wall.elapsed().as_secs_f64();
    let stats = target.client.stats();
    println!(
        "workload={:?} threads={} ops/thread={} keys={:?}",
        args.workload, args.threads, args.ops, args.keys
    );
    println!(
        "completed={completed} errors={errors} redirects={} elapsed={secs:.2}s",
        stats.redirects
    );
    #[allow(clippy::cast_precision_loss)]
    let ops_per_sec = completed as f64 / secs;
    println!("ops/sec={ops_per_sec:.0}");
    for quantile in [50.0, 95.0, 99.0, 99.9] {
        println!(
            "p{quantile:<4}={}ns",
            merged.value_at_quantile(quantile / 100.0)
        );
    }
    println!("max={}ns", merged.max());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn key_space_parser_accepts_hot_and_uniform() {
        assert_eq!(parse_key_space("hot"), Ok(KeySpace::Hot));
        assert_eq!(parse_key_space("uniform:16"), Ok(KeySpace::Uniform(16)));
        assert!(parse_key_space("uniform:0").is_err());
        assert!(parse_key_space("uniform:lots").is_err());
        assert!(parse_key_space("cold").is_err());
    }

    #[test]
    fn bench_args_parse_end_to_end() {
        let args = Args::try_parse_from([
            "kivi-bench",
            "--threads",
            "4",
            "--ops",
            "100",
            "--workload",
            "get",
            "--keys",
            "hot",
        ])
        .expect("args parse");
        assert_eq!(args.threads, 4);
        assert_eq!(args.ops, 100);
        assert_eq!(args.workload, Workload::Get);
        assert_eq!(args.keys, KeySpace::Hot);
    }

    #[test]
    fn bench_args_reject_bad_input_without_panicking() {
        assert!(Args::try_parse_from(["kivi-bench", "--threads", "0"]).is_err());
        assert!(Args::try_parse_from(["kivi-bench", "--keys", "cold"]).is_err());
        assert!(Args::try_parse_from(["kivi-bench", "--workload", "frobnicate"]).is_err());
    }

    #[test]
    fn workload_key_names_match_legacy_layout() {
        assert_eq!(thread_key_names(KeySpace::Hot, 3), vec!["hot".to_owned()]);
        assert_eq!(
            thread_key_names(KeySpace::Uniform(2), 1),
            vec!["t1:k0".to_owned(), "t1:k1".to_owned()]
        );
        assert_eq!(counter_key_name(KeySpace::Hot, 0), "hot:c");
        assert_eq!(counter_key_name(KeySpace::Uniform(9), 2), "t2:c");
    }

    #[test]
    fn workload_mix_matches_legacy_pattern() {
        let keys = vec!["a".to_owned(), "b".to_owned()];
        let counter = "c";
        assert_eq!(
            workload_op_for(Workload::Get, 0, &keys, counter),
            WorkloadOp::Get("a".to_owned())
        );
        assert_eq!(
            workload_op_for(Workload::Set, 1, &keys, counter),
            WorkloadOp::Set("b".to_owned())
        );
        assert_eq!(
            workload_op_for(Workload::Counter, 7, &keys, counter),
            WorkloadOp::CounterAdd("c".to_owned())
        );
        assert_eq!(
            workload_op_for(Workload::Mixed, 0, &keys, counter),
            WorkloadOp::Get("a".to_owned())
        );
        assert_eq!(
            workload_op_for(Workload::Mixed, 1, &keys, counter),
            WorkloadOp::Set("b".to_owned())
        );
        assert_eq!(
            workload_op_for(Workload::Mixed, 2, &keys, counter),
            WorkloadOp::CounterAdd("c".to_owned())
        );
        assert_eq!(
            workload_op_for(Workload::Mixed, 3, &keys, counter),
            WorkloadOp::Delete("b".to_owned())
        );
    }

    #[test]
    fn seed_ops_cover_measured_types_only() {
        // Byte workloads seed bytes; counter workloads seed counters; mixed
        // seeds both. Hot seeds once (thread 0); uniform seeds per thread.
        assert_eq!(
            seed_ops_for(KeySpace::Hot, 0, Workload::Get),
            vec![WorkloadOp::Set("hot".to_owned())]
        );
        assert!(seed_ops_for(KeySpace::Hot, 1, Workload::Get).is_empty());
        assert_eq!(
            seed_ops_for(KeySpace::Hot, 0, Workload::Counter),
            vec![WorkloadOp::CounterAdd("hot:c".to_owned())]
        );
        let uniform_set = seed_ops_for(KeySpace::Uniform(2), 0, Workload::Set);
        assert_eq!(
            uniform_set,
            vec![
                WorkloadOp::Set("t0:k0".to_owned()),
                WorkloadOp::Set("t0:k1".to_owned())
            ]
        );
    }
}
