//! `kivi-bench`: native-protocol load generator with hdrhistogram latencies.
//!
//! Drives the real TCP path (never the embedded channel) with configurable
//! threads, per-thread operation counts, workloads, and key distributions.
//! Each thread owns a cloned client (one outstanding request per thread, so
//! threads == pipeline depth) and records nanosecond latencies locally;
//! histograms merge at the end for exact quantiles.

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

use bytes::Bytes;
use clap::{Parser, ValueEnum};
use hdrhistogram::Histogram;
use kivi_client::{ClientConfig, NativeClient};
use kivi_state::Key;
use kivi_types::NamespaceId;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeySpace {
    Hot,
    Uniform(usize),
}

fn thread_keys(space: KeySpace, thread: usize) -> Vec<Key> {
    match space {
        KeySpace::Hot => vec![Key::from("hot")],
        KeySpace::Uniform(count) => (0..count)
            .map(|i| Key::from(format!("t{thread}:k{i}")))
            .collect(),
    }
}

/// Dedicated per-thread counter key. Counter ops never share keys with
/// byte ops: mixing types on one key fails with `WrongType` by design, so
/// each workload seeds and hammers the type it measures.
fn counter_key(space: KeySpace, thread: usize) -> Key {
    match space {
        KeySpace::Hot => Key::from("hot:c"),
        KeySpace::Uniform(_) => Key::from(format!("t{thread}:c")),
    }
}

/// Pre-populates every key the workload touches, with the type the
/// workload hammers (byte ops share byte keys; counter ops use a dedicated
/// per-thread counter key — mixing types on one key is `WrongType`).
fn seed(
    client: &NativeClient,
    space: KeySpace,
    threads: usize,
    workload: Workload,
) -> anyhow::Result<()> {
    use anyhow::Context;
    for thread in 0..threads {
        let seed_bytes = matches!(workload, Workload::Get | Workload::Set | Workload::Mixed);
        let seed_counters = matches!(workload, Workload::Counter | Workload::Mixed);
        if seed_bytes {
            if matches!(space, KeySpace::Hot) && thread == 0 {
                client
                    .set(&Key::from("hot"), Bytes::from_static(b"v"))
                    .context("seed hot key")?;
            }
            if !matches!(space, KeySpace::Hot) {
                for key in thread_keys(space, thread) {
                    client
                        .set(&key, Bytes::from_static(b"v"))
                        .with_context(|| format!("seed thread {thread} byte keyspace"))?;
                }
            }
        }
        if seed_counters {
            client
                .counter_add(&counter_key(space, thread), 0)
                .with_context(|| format!("seed thread {thread} counter"))?;
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    use anyhow::Context;
    let args = Args::parse();
    let space = args.keys;
    let client = NativeClient::new(ClientConfig {
        seeds: vec![args.server],
        namespace: NamespaceId::from_u64(args.namespace),
        max_pending: args.pending,
        ..ClientConfig::default()
    })
    .context("client setup failed")?;
    seed(&client, space, args.threads, args.workload).context("keyspace seeding failed")?;
    let barrier = Arc::new(Barrier::new(args.threads + 1));
    let mut handles = Vec::with_capacity(args.threads);
    for thread in 0..args.threads {
        let client = client.clone();
        let gate = Arc::clone(&barrier);
        let keys = thread_keys(space, thread);
        let workload = args.workload;
        let ops = args.ops;
        handles.push(thread::spawn(move || {
            let mut histogram = Histogram::<u64>::new(3).expect("histogram builds");
            let mut errors = 0u64;
            let mut counter = 0u64;
            let counter_key = counter_key(space, thread);
            gate.wait();
            for i in 0..ops {
                let key = &keys[i % keys.len()];
                let start = Instant::now();
                let outcome = match workload {
                    Workload::Get => client.get(key).map(|_| ()),
                    Workload::Set => client.set(key, Bytes::from_static(b"v")),
                    Workload::Counter => client.counter_add(&counter_key, 1).map(|_| ()),
                    Workload::Mixed => match i % 4 {
                        0 => client.get(key).map(|_| ()),
                        1 => client.set(key, Bytes::from_static(b"v")),
                        2 => client.counter_add(&counter_key, 1).map(|_| ()),
                        _ => client.delete(key).map(|_| ()),
                    },
                };
                let nanos = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
                if outcome.is_err() {
                    errors += 1;
                } else {
                    counter += 1;
                    histogram.record(nanos).expect("latency fits");
                }
            }
            (histogram, errors, counter)
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
    let stats = client.stats();
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
}
