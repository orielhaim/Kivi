//! `kivi-lab`: benchmarking and conformance for Kivi vs RESP references.
//!
//! ```text
//! kivi-lab bench --target native --server 127.0.0.1:9000 ...
//! kivi-lab bench --target resp --server 127.0.0.1:6379 ...
//! kivi-lab conformance [--kivi 127.0.0.1:6380 | --spawn] [--redis URL]
//! ```
//!
//! `bench --target resp` speaks the same raw RESP to Kivi RESP, Redis,
//! Valkey, or Dragonfly — the transport is identical, so comparisons
//! isolate server differences. Machine-readable reports use `--format json`
//! (schema [`kivi_lab::metrics::BENCH_SCHEMA`]); serde lives only in this
//! non-production crate.

use std::collections::BTreeMap;

use clap::{Parser, Subcommand, ValueEnum};

use kivi_lab::conformance;
use kivi_lab::metrics::{BENCH_SCHEMA, BenchReport};
use kivi_lab::runner::{self, RunnerConfig};
use kivi_lab::targets::{KiviNativeTarget, RespTarget};
use kivi_lab::workload::{KeySpace, ValueSize, Workload, parse_key_space};

#[derive(Debug, Parser)]
#[command(name = "kivi-lab", about = "Kivi benchmarking and conformance lab")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run a workload against one target and report throughput/latency.
    Bench(BenchArgs),
    /// Run the exact-profile vector matrix against Kivi RESP and a
    /// reference Redis, then diff normalized observables.
    Conformance(ConformanceArgs),
    /// Delete every key under a prefix on a RESP endpoint (benchmark
    /// hygiene for shared instances; uses SCAN, so Redis-only).
    Cleanup(CleanupArgs),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Target {
    /// Kivi native protocol.
    Native,
    /// Any RESP endpoint (Kivi RESP, Redis, Valkey, Dragonfly).
    Resp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    /// Human-readable summary.
    Text,
    /// Machine-readable JSON report.
    Json,
}

#[derive(Debug, Parser)]
struct BenchArgs {
    /// Which frontend to drive.
    #[arg(long, value_enum, default_value_t = Target::Native)]
    target: Target,
    /// Endpoint: native seed endpoint or RESP address.
    #[arg(long, default_value = "127.0.0.1:9000")]
    server: String,
    /// Extra native seed endpoints (comma-separated; the first seed
    /// stays `--server`). A multi-seed native target is how existing
    /// workload definitions run against the replicated cluster: leader
    /// discovery, `NotLeader` retries, and stable mutation identities
    /// are handled inside the target (task AG).
    #[arg(long, value_delimiter = ',')]
    seeds: Vec<String>,
    /// Target namespace id (native only; RESP always uses DB 0).
    #[arg(long, default_value_t = 1)]
    namespace: u64,
    /// Client threads (one target instance each).
    #[arg(long, default_value_t = 8, value_parser = parse_nonzero)]
    threads: usize,
    /// Measured operations per thread.
    #[arg(long, default_value_t = 5_000, value_parser = parse_nonzero)]
    ops: usize,
    /// Warmup operations per thread (unmeasured).
    #[arg(long, default_value_t = 500)]
    warmup: usize,
    /// Request mix.
    #[arg(long, value_enum, default_value_t = Workload::Mixed)]
    workload: Workload,
    /// Key distribution: `hot` or `uniform:<n>`.
    #[arg(long, default_value = "uniform:256", value_parser = parse_key_space)]
    keys: KeySpace,
    /// Key prefix isolating this run (never touch bare shared names on an
    /// externally supplied instance).
    #[arg(long, default_value = "kivi-bench:")]
    prefix: String,
    /// Value size for `Set` ops.
    #[arg(long, value_enum, default_value_t = ValueSize::B16)]
    value_size: ValueSize,
    /// Commands batched per round-trip (latencies are per-batch above 1).
    #[arg(long, default_value_t = 1, value_parser = parse_nonzero)]
    pipeline: usize,
    /// Per-connection in-flight cap, native only.
    #[arg(long, default_value_t = 256)]
    pending: usize,
    /// Output format.
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,
    /// Write the JSON report here instead of stdout.
    #[arg(long)]
    out: Option<String>,
}

#[derive(Debug, Parser)]
struct ConformanceArgs {
    /// Kivi RESP address (skips spawning when given).
    #[arg(long)]
    kivi: Option<String>,
    /// Spawn an ephemeral Kivi server (needs a `redis-compat` binary).
    #[arg(long, default_value_t = false)]
    spawn: bool,
    /// Reference Redis URL (or `KIVI_LAB_REDIS_URL`).
    #[arg(long)]
    redis: Option<String>,
}

#[derive(Debug, Parser)]
struct CleanupArgs {
    /// RESP endpoint holding the keys (Redis/Valkey/Dragonfly).
    #[arg(long)]
    redis: Option<String>,
    /// Key prefix to delete (no `FLUSHALL`, ever).
    #[arg(long)]
    prefix: String,
}

/// Parses a nonzero count without panicking.
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

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Bench(args) => bench(args),
        Command::Conformance(args) => conformance_cmd(args),
        Command::Cleanup(args) => {
            let url = conformance::redis_url(args.redis.as_deref());
            let mut client = kivi_lab::resp_client::RespClient::connect(&url)?;
            let removed = conformance::cleanup_prefix_scan(&mut client, &args.prefix)?;
            println!("removed={removed} prefix={} redis={url}", args.prefix);
            Ok(())
        }
    }
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::needless_pass_by_value)]
fn bench(args: BenchArgs) -> anyhow::Result<()> {
    use anyhow::Context;
    if matches!(args.target, Target::Resp)
        && matches!(args.workload, Workload::Counter | Workload::Mixed)
    {
        eprintln!(
            "note: {:?} includes counter ops, which the RESP profile leaves unsupported; errors expected there (use `compat` for the exact cross-target mix)",
            args.workload
        );
    }
    let config = RunnerConfig {
        workload: args.workload,
        space: args.keys,
        value_len: args.value_size.len(),
        prefix: args.prefix.clone(),
        threads: args.threads,
        ops_per_thread: args.ops,
        warmup_per_thread: args.warmup,
        pipeline: args.pipeline,
    };
    let keys_desc = match args.keys {
        KeySpace::Hot => "hot".to_owned(),
        KeySpace::Uniform(count) => format!("uniform:{count}"),
    };
    let workload_name = format!("{:?}", args.workload);
    let (report, endpoint, kind) = match args.target {
        Target::Native => {
            let server = args.server.clone();
            let mut seeds = vec![server.clone()];
            seeds.extend(args.seeds.iter().cloned());
            let mut seeder = KiviNativeTarget::connect_multi(&seeds, args.namespace, args.pending)?;
            runner::seed(
                &mut seeder,
                args.keys,
                args.threads,
                args.workload,
                args.value_size.len(),
                &args.prefix,
            )
            .context("keyspace seeding failed")?;
            let redirects = seeder.redirects();
            let stats = runner::run(config, || {
                KiviNativeTarget::connect_multi(&seeds, args.namespace, args.pending)
            })?;
            let mut extra = BTreeMap::new();
            extra.insert("redirects".to_owned(), redirects.to_string());
            (
                finish_report(
                    &stats,
                    "native",
                    &server,
                    &workload_name,
                    &keys_desc,
                    &args,
                    extra,
                ),
                server,
                "native",
            )
        }
        Target::Resp => {
            let server = args.server.clone();
            let mut seeder = RespTarget::lazy(&server);
            runner::seed(
                &mut seeder,
                args.keys,
                args.threads,
                args.workload,
                args.value_size.len(),
                &args.prefix,
            )
            .context("keyspace seeding failed")?;
            let stats = runner::run(config, || Ok(RespTarget::lazy(&server)))?;
            let extra = BTreeMap::new();
            (
                finish_report(
                    &stats,
                    "resp",
                    &server,
                    &workload_name,
                    &keys_desc,
                    &args,
                    extra,
                ),
                server,
                "resp",
            )
        }
    };
    let _ = (endpoint, kind);
    match args.format {
        Format::Text => print!("{}", report.text()),
        Format::Json => {
            let document = serde_json::to_string_pretty(&report)?;
            match &args.out {
                Some(path) => std::fs::write(path, format!("{document}\n"))?,
                None => println!("{document}"),
            }
        }
    }
    if report.errors > 0 {
        eprintln!("warning: {} op errors (see errors=)", report.errors);
    }
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn finish_report(
    stats: &runner::RunStats,
    target: &str,
    endpoint: &str,
    workload: &str,
    keys: &str,
    args: &BenchArgs,
    extra: BTreeMap<String, String>,
) -> BenchReport {
    BenchReport {
        schema: BENCH_SCHEMA,
        target: target.to_owned(),
        endpoint: endpoint.to_owned(),
        workload: workload.to_owned(),
        keys: keys.to_owned(),
        prefix: args.prefix.clone(),
        value_len: args.value_size.len(),
        threads: args.threads,
        pipeline: args.pipeline,
        ops_per_thread: args.ops,
        completed: stats.completed,
        errors: stats.errors,
        ops_per_sec: stats.completed as f64 / stats.secs,
        latency_unit: if args.pipeline == 1 {
            "ns/op"
        } else {
            "ns/batch"
        },
        p50_ns: stats.histogram.value_at_quantile(0.50),
        p95_ns: stats.histogram.value_at_quantile(0.95),
        p99_ns: stats.histogram.value_at_quantile(0.99),
        p999_ns: stats.histogram.value_at_quantile(0.999),
        max_ns: stats.histogram.max(),
        elapsed_secs: stats.secs,
        bytes_out: stats.bytes.map(|bytes| bytes.0),
        bytes_in: stats.bytes.map(|bytes| bytes.1),
        extra,
    }
}

fn conformance_cmd(args: ConformanceArgs) -> anyhow::Result<()> {
    use anyhow::Context;
    // Resolve the Kivi side: explicit address, or a spawned ephemeral
    // server (which hard-fails without a `redis-compat` binary).
    let _spawned: Option<kivi_lab::process::Server>;
    let kivi_addr = match (args.kivi, args.spawn) {
        (Some(addr), _) => {
            _spawned = None;
            addr
        }
        (None, true) => {
            let server = kivi_lab::process::Server::spawn_ephemeral_resp()
                .context("spawning Kivi with RESP")?;
            let addr = server.resp_endpoint().expect("resp endpoint");
            _spawned = Some(server);
            addr
        }
        (None, false) => {
            anyhow::bail!("pass --kivi <addr> or --spawn to provide the Kivi RESP side");
        }
    };
    let redis_url = conformance::redis_url(args.redis.as_deref());
    let mut kivi = kivi_lab::resp_client::RespClient::connect(&kivi_addr)?;
    let mut reference = kivi_lab::resp_client::RespClient::connect(&redis_url)
        .with_context(|| format!("reference Redis at {redis_url} unreachable"))?;
    let config = conformance::redis_config(&mut reference)?;
    println!(
        "reference: version={} appendonly={} appendfsync={} save=[{}] maxmemory-policy={}",
        config.version, config.appendonly, config.appendfsync, config.save, config.maxmemory_policy
    );
    let prefix = conformance::run_scope();
    conformance::setup_edge_keys(&mut kivi, &prefix)?;
    conformance::setup_edge_keys(&mut reference, &prefix)?;
    let left = conformance::run_vectors(&mut kivi, &prefix)?;
    let right = conformance::run_vectors(&mut reference, &prefix)?;
    let mismatches = conformance::diff_vectors(&left, &right);
    let removed_kivi = conformance::cleanup_prefix(&mut kivi, &prefix)?;
    let removed_ref = conformance::cleanup_prefix(&mut reference, &prefix)?;
    println!(
        "vectors={} mismatches={} cleaned kivi={removed_kivi} redis={removed_ref} prefix={prefix}",
        left.len(),
        mismatches.len()
    );
    for (argv, kivi_outcome, redis_outcome) in &mismatches {
        println!("MISMATCH {argv:?}: kivi={kivi_outcome:?} redis={redis_outcome:?}");
    }
    if mismatches.is_empty() {
        Ok(())
    } else {
        anyhow::bail!("{} conformance mismatches", mismatches.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn bench_args_parse_end_to_end() {
        let cli = Cli::try_parse_from([
            "kivi-lab",
            "bench",
            "--threads",
            "4",
            "--ops",
            "100",
            "--workload",
            "get",
            "--keys",
            "hot",
            "--value-size",
            "kb64",
            "--format",
            "json",
        ])
        .expect("args parse");
        let Command::Bench(args) = cli.command else {
            panic!("bench subcommand");
        };
        assert_eq!(args.threads, 4);
        assert_eq!(args.ops, 100);
        assert_eq!(args.workload, Workload::Get);
        assert_eq!(args.value_size, ValueSize::Kb64);
        assert_eq!(args.format, Format::Json);
    }

    #[test]
    fn bench_args_reject_bad_input_without_panicking() {
        assert!(Cli::try_parse_from(["kivi-lab", "bench", "--threads", "0"]).is_err());
        assert!(Cli::try_parse_from(["kivi-lab", "bench", "--keys", "cold"]).is_err());
        assert!(Cli::try_parse_from(["kivi-lab", "bench", "--workload", "frobnicate"]).is_err());
    }

    #[test]
    fn conformance_args_parse() {
        let cli = Cli::try_parse_from([
            "kivi-lab",
            "conformance",
            "--spawn",
            "--redis",
            "redis://localhost:6379",
        ])
        .expect("args parse");
        assert!(matches!(cli.command, Command::Conformance(_)));
    }
}
