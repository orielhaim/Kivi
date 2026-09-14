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
use std::time::Instant;

use hdrhistogram::Histogram;

use crate::targets::BenchTarget;
use crate::workload::{KeySpace, Workload, counter_key_name, thread_key_names, workload_op_for};

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
    /// Bytes written/read, when the target accounts them.
    pub bytes: Option<(u64, u64)>,
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
    /// Wall-clock seconds for the measured phase.
    pub secs: f64,
    /// Bytes written/read, when the target accounts them.
    pub bytes: Option<(u64, u64)>,
}

/// Drives one benchmark configuration.
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
    mut make_target: impl FnMut() -> anyhow::Result<T> + Send + Sync,
) -> anyhow::Result<RunStats> {
    let RunnerConfig {
        workload,
        space,
        value_len,
        prefix,
        threads,
        ops_per_thread,
        warmup_per_thread,
        pipeline,
    } = config;
    let barrier = Arc::new(Barrier::new(threads + 1));
    let mut handles = Vec::with_capacity(threads);
    // Factories are `FnMut`; build all instances up front on this thread.
    let mut targets = Vec::with_capacity(threads);
    for _ in 0..threads {
        targets.push(make_target()?);
    }
    for (thread, mut target) in targets.into_iter().enumerate() {
        let gate = Arc::clone(&barrier);
        let keys = thread_key_names(space, thread, &prefix);
        let counter = counter_key_name(space, thread, &prefix);
        handles.push(thread::spawn(move || {
            // Warmup: identical op stream, unmeasured, so steady state
            // (route caches, connections, allocator) is reached first.
            let mut index = 0usize;
            while index < warmup_per_thread {
                let end = (index + pipeline).min(warmup_per_thread);
                let chunk: Vec<_> = (index..end)
                    .map(|i| workload_op_for(workload, value_len, i, &keys, &counter))
                    .collect();
                let _ = target.execute_batch(&chunk);
                index = end;
            }
            let mut histogram = Histogram::<u64>::new(3).expect("histogram builds");
            let mut errors = 0u64;
            let mut completed = 0u64;
            // Reset transport counters after warmup so I/O metrics cover
            // the measured phase only.
            let _ = target.take_bytes();
            gate.wait();
            let mut index = 0usize;
            while index < ops_per_thread {
                let end = (index + pipeline).min(ops_per_thread);
                let chunk: Vec<_> = (index..end)
                    .map(|i| workload_op_for(workload, value_len, i, &keys, &counter))
                    .collect();
                let start = Instant::now();
                let outcomes = target.execute_batch(&chunk);
                let nanos = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
                let failed = outcomes.iter().filter(|outcome| outcome.is_err()).count() as u64;
                errors += failed;
                completed += (outcomes.len() as u64).saturating_sub(failed);
                histogram.record(nanos).expect("latency fits");
                index = end;
            }
            ThreadStats {
                histogram,
                errors,
                completed,
                bytes: target.take_bytes(),
            }
        }));
    }
    let wall = Instant::now();
    barrier.wait();
    let mut merged = Histogram::<u64>::new(3).expect("histograms merge");
    let mut errors = 0u64;
    let mut completed = 0u64;
    let mut bytes_out = 0u64;
    let mut bytes_in = 0u64;
    let mut bytes_known = true;
    for handle in handles {
        let stats = handle.join().expect("thread joins");
        merged.add(&stats.histogram).expect("histograms merge");
        errors += stats.errors;
        completed += stats.completed;
        match stats.bytes {
            Some((out, inn)) => {
                bytes_out += out;
                bytes_in += inn;
            }
            None => bytes_known = false,
        }
    }
    Ok(RunStats {
        histogram: merged,
        errors,
        completed,
        secs: wall.elapsed().as_secs_f64(),
        bytes: bytes_known.then_some((bytes_out, bytes_in)),
    })
}

/// Seeds every key the workload touches through `target` (byte keys at
/// `value_len`, counters at their creation point).
///
/// # Errors
///
/// Returns the first seeding failure, labeled by thread.
pub fn seed<T: BenchTarget>(
    target: &mut T,
    space: KeySpace,
    threads: usize,
    workload: Workload,
    value_len: usize,
    prefix: &str,
) -> anyhow::Result<()> {
    use crate::workload::seed_ops_for;
    for thread in 0..threads {
        for op in seed_ops_for(space, thread, workload, value_len, prefix) {
            target
                .seed_one(&op)
                .map_err(|error| anyhow::anyhow!("seed thread {thread} keyspace: {error}"))?;
        }
    }
    Ok(())
}
