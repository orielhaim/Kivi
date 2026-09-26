//! Benchmark runner: concurrency, pipelining, warmup, timing, histograms.
//!
//! The runner is deliberately separate from [`crate::targets::BenchTarget`]
//! so methodology is byte-identical across Native and RESP targets: the
//! same thread count, the same batching, the same warmup, the same clock,
//! and the same histogram. Only the target implementation differs.
//!
//! Methodology: each thread owns one target instance (one connection or
//! session), runs `warmup` ops unmeasured, waits on a start barrier, then
//! issues measured batches of `pipeline` ops. One histogram sample covers
//! one batch round-trip; `completed`/`errors` count individual ops, so
//! throughput stays per-op exact at every pipeline depth.

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use hdrhistogram::Histogram;

use crate::targets::{BenchTarget, TargetStats};
use crate::workload::{
    KeySpace, PayloadCache, Workload, counter_key_name, thread_key_names, workload_op_for,
};

/// Runner configuration (shared by all targets).
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Workload mix.
    pub workload: Workload,
    /// Key distribution.
    pub space: KeySpace,
    /// Value length in bytes for `Set` ops.
    pub value_len: usize,
    /// Client threads (one target instance each).
    pub threads: usize,
    /// Key prefix isolating this run (never bare shared names against an
    /// externally supplied instance).
    pub prefix: String,
    /// Measured operations per thread.
    pub ops_per_thread: usize,
    /// Warmup operations per thread (unmeasured).
    pub warmup_per_thread: usize,
    /// Commands batched per round-trip.
    pub pipeline: usize,
}

/// Per-thread outcome before merging.
#[derive(Debug)]
pub struct ThreadStats {
    /// Per-batch latency histogram (nanoseconds per batch round-trip).
    pub histogram: Histogram<u64>,
    /// Failed individual ops.
    pub errors: u64,
    /// Completed individual ops.
    pub completed: u64,
    /// Failed warmup operations.
    pub warmup_errors: u64,
    /// Bytes written/read, when the target accounts them.
    pub bytes: Option<(u64, u64)>,
    /// Target-specific counters, when the target exposes them.
    pub target_stats: Option<TargetStats>,
}

/// Merged outcome of one run.
#[derive(Debug)]
pub struct RunStats {
    /// Merged per-batch latency histogram.
    pub histogram: Histogram<u64>,
    /// Failed individual ops across threads.
    pub errors: u64,
    /// Completed individual ops across threads.
    pub completed: u64,
    /// Completed individual ops for each worker, in worker order.
    pub completed_per_thread: Vec<u64>,
    /// Failed warmup operations, which invalidate a trial.
    pub warmup_errors: u64,
    /// Wall-clock seconds for the measured phase.
    pub secs: f64,
    /// Bytes written/read, when the target accounts them.
    pub bytes: Option<(u64, u64)>,
    /// Merged target-specific counters, when available.
    pub target_stats: Option<TargetStats>,
}

/// Drives one fixed-count benchmark configuration.
///
/// # Panics
///
/// Panics when a worker thread fails to spawn/join or a target factory
/// fails (setup errors are programmer errors here, never runtime noise).
///
/// # Errors
///
/// Returns the first target-factory failure.
pub fn run<T: BenchTarget + 'static>(
    config: RunnerConfig,
    make_target: impl FnMut() -> anyhow::Result<T> + Send + Sync,
) -> anyhow::Result<RunStats> {
    let limit = config.ops_per_thread;
    run_inner(config, RunLimit::Count(limit), make_target)
}

/// Drives a benchmark until a wall-clock deadline.
///
/// # Panics
///
/// Panics when a worker thread fails to spawn/join or a target factory
/// fails (setup errors are programmer errors here, never runtime noise).
///
/// # Errors
///
/// Returns an error when `duration` is zero or target construction fails.
pub fn run_for_duration<T: BenchTarget + 'static>(
    config: RunnerConfig,
    duration: Duration,
    make_target: impl FnMut() -> anyhow::Result<T> + Send + Sync,
) -> anyhow::Result<RunStats> {
    if duration.is_zero() {
        anyhow::bail!("duration must be nonzero");
    }
    run_inner(config, RunLimit::Duration(duration), make_target)
}

#[derive(Debug, Clone, Copy)]
enum RunLimit {
    Count(usize),
    Duration(Duration),
}

#[allow(clippy::too_many_lines)]
fn run_inner<T: BenchTarget + 'static>(
    config: RunnerConfig,
    limit: RunLimit,
    mut make_target: impl FnMut() -> anyhow::Result<T> + Send + Sync,
) -> anyhow::Result<RunStats> {
    let RunnerConfig {
        workload,
        space,
        value_len,
        prefix,
        threads,
        ops_per_thread: _,
        warmup_per_thread,
        pipeline,
    } = config;
    if threads == 0 || pipeline == 0 {
        anyhow::bail!("threads and pipeline must be nonzero");
    }
    let barrier = Arc::new(Barrier::new(threads + 1));
    let payload = Arc::new(PayloadCache::new(value_len));
    let mut handles = Vec::with_capacity(threads);
    let mut targets = Vec::with_capacity(threads);
    for _ in 0..threads {
        targets.push(make_target()?);
    }
    for (thread, mut target) in targets.into_iter().enumerate() {
        let gate = Arc::clone(&barrier);
        let worker_payload = Arc::clone(&payload);
        let keys = thread_key_names(space, thread, &prefix);
        let counter = counter_key_name(space, thread, &prefix);
        handles.push(thread::spawn(move || {
            let mut warmup_errors = 0u64;
            let mut index = 0usize;
            while index < warmup_per_thread {
                let end = (index + pipeline).min(warmup_per_thread);
                let chunk: Vec<_> = (index..end)
                    .map(|i| workload_op_for(workload, value_len, i, &keys, &counter))
                    .collect();
                let outcomes = target.execute_batch_with_payload(&chunk, worker_payload.as_bytes());
                warmup_errors += outcomes.iter().filter(|outcome| outcome.is_err()).count() as u64;
                index = end;
            }
            let baseline_stats = target.take_stats();
            let mut histogram = Histogram::<u64>::new(3).expect("histogram builds");
            let mut errors = 0u64;
            let mut completed = 0u64;
            let _ = target.take_bytes();
            gate.wait();
            let run_started = Instant::now();
            let mut index = 0usize;
            loop {
                let batch_len = match limit {
                    RunLimit::Count(limit) => {
                        if index >= limit {
                            break;
                        }
                        (index + pipeline).min(limit) - index
                    }
                    RunLimit::Duration(_) => pipeline,
                };
                let end = index.saturating_add(batch_len);
                let chunk: Vec<_> = (index..end)
                    .map(|i| workload_op_for(workload, value_len, i, &keys, &counter))
                    .collect();
                let start = Instant::now();
                let outcomes = target.execute_batch_with_payload(&chunk, worker_payload.as_bytes());
                let nanos = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
                let failed = outcomes.iter().filter(|outcome| outcome.is_err()).count() as u64;
                errors += failed;
                completed += (outcomes.len() as u64).saturating_sub(failed);
                histogram.record(nanos).expect("latency fits");
                index = end;
                if matches!(limit, RunLimit::Duration(duration) if run_started.elapsed() >= duration) {
                    break;
                }
            }
            let target_stats = target.take_stats().map(|current| TargetStats {
                requests: current.requests.saturating_sub(
                    baseline_stats.map_or(0, |baseline| baseline.requests),
                ),
                redirects: current.redirects.saturating_sub(
                    baseline_stats.map_or(0, |baseline| baseline.redirects),
                ),
                errors: current.errors.saturating_sub(
                    baseline_stats.map_or(0, |baseline| baseline.errors),
                ),
            });
            ThreadStats {
                histogram,
                errors,
                completed,
                warmup_errors,
                bytes: target.take_bytes(),
                target_stats,
            }
        }));
    }
    barrier.wait();
    let wall = Instant::now();
    let mut merged = Histogram::<u64>::new(3).expect("histograms merge");
    let mut errors = 0u64;
    let mut completed = 0u64;
    let mut bytes_out = 0u64;
    let mut bytes_in = 0u64;
    let mut completed_per_thread = Vec::with_capacity(threads);
    let mut warmup_errors = 0u64;
    let mut bytes_known = true;
    let mut target_stats_known = true;
    let mut target_stats = TargetStats::default();
    for handle in handles {
        let stats = handle.join().expect("thread joins");
        merged.add(&stats.histogram).expect("histograms merge");
        errors += stats.errors;
        completed += stats.completed;
        warmup_errors += stats.warmup_errors;
        completed_per_thread.push(stats.completed);
        match stats.bytes {
            Some((out, inn)) => {
                bytes_out += out;
                bytes_in += inn;
            }
            None => bytes_known = false,
        }
        match stats.target_stats {
            Some(current) => {
                target_stats.requests = target_stats.requests.saturating_add(current.requests);
                target_stats.redirects = target_stats.redirects.saturating_add(current.redirects);
                target_stats.errors = target_stats.errors.saturating_add(current.errors);
            }
            None => target_stats_known = false,
        }
    }
    Ok(RunStats {
        histogram: merged,
        errors,
        completed,
        completed_per_thread,
        warmup_errors,
        secs: wall.elapsed().as_secs_f64(),
        bytes: bytes_known.then_some((bytes_out, bytes_in)),
        target_stats: target_stats_known.then_some(target_stats),
    })
}

/// Seeds every key the workload touches through `target` (byte keys at
/// `value_len`, counters at their creation point). Overload retries with
/// bounded backoff, the way production loaders pace bursts: pressure
/// scenarios establish their working set instead of failing the burst.
/// Any other seeding failure aborts loudly.
///
/// # Errors
///
/// Returns the first non-overload seeding failure, labeled by thread, or
/// overload that outlasts the retry budget.
pub fn seed<T: BenchTarget>(
    target: &mut T,
    space: KeySpace,
    threads: usize,
    workload: Workload,
    value_len: usize,
    prefix: &str,
) -> anyhow::Result<()> {
    use crate::workload::seed_ops_for;
    let payload = PayloadCache::new(value_len);
    for thread in 0..threads {
        for op in seed_ops_for(space, thread, workload, value_len, prefix) {
            let mut attempt = 0u32;
            loop {
                match target.seed_one_with_payload(&op, payload.as_bytes()) {
                    Ok(()) => break,
                    Err(error) if is_overload(&error) && attempt < 30 => {
                        attempt += 1;
                        thread::sleep(std::time::Duration::from_millis(50));
                    }
                    Err(error) => {
                        return Err(anyhow::anyhow!("seed thread {thread} keyspace: {error}"));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Whether a seed failure is backpressure (retryable) rather than a
/// logical error (fatal): overload surfaces through several shapings.
fn is_overload(error: &str) -> bool {
    error.contains("overload")
        || error.contains("Overload")
        || error.contains("window exhausted")
        || error.contains("busy")
}
