//! Benchmark metrics: histogram helpers and the machine-readable schema.
//!
//! Human text goes to stdout for operators; [`BenchReport`] goes to a file
//! (or stdout with `--format json`) for tooling. JSON here is lab output,
//! never a Kivi canonical storage format — serde lives only in this
//! non-production crate.

use std::collections::BTreeMap;

/// Schema identifier for [`BenchReport`]. Bump on any breaking field change
/// so historical comparisons and CI regressions compare like with like.
pub const BENCH_SCHEMA: &str = "kivi-lab-bench/1";

/// Machine-readable result of one benchmark run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchReport {
    /// Schema/version marker (see [`BENCH_SCHEMA`]).
    pub schema: &'static str,
    /// Target kind: `native` or `resp`.
    pub target: String,
    /// Target endpoint that was driven.
    pub endpoint: String,
    /// Workload mix name.
    pub workload: String,
    /// Key distribution description.
    pub keys: String,
    /// Key prefix isolating this run.
    pub prefix: String,
    /// Value length in bytes for `Set` ops.
    pub value_len: usize,
    /// Client threads.
    pub threads: usize,
    /// Commands batched per round-trip.
    pub pipeline: usize,
    /// Measured ops per thread (warmup excluded).
    pub ops_per_thread: usize,
    /// Completed individual ops.
    pub completed: u64,
    /// Failed individual ops.
    pub errors: u64,
    /// Throughput in completed ops per second.
    pub ops_per_sec: f64,
    /// Latency unit: `ns/op` (pipeline 1) or `ns/batch`.
    pub latency_unit: &'static str,
    /// p50 batch latency in nanoseconds.
    pub p50_ns: u64,
    /// p95 batch latency in nanoseconds.
    pub p95_ns: u64,
    /// p99 batch latency in nanoseconds.
    pub p99_ns: u64,
    /// p99.9 batch latency in nanoseconds.
    pub p999_ns: u64,
    /// Max observed batch latency in nanoseconds.
    pub max_ns: u64,
    /// Elapsed wall-clock seconds for the measured phase.
    pub elapsed_secs: f64,
    /// Transport bytes written, when the target accounts them.
    pub bytes_out: Option<u64>,
    /// Transport bytes read, when the target accounts them.
    pub bytes_in: Option<u64>,
    /// Target-specific extras (native redirects, reference server version).
    /// Shared fields stay top-level; nothing target-specific leaks into
    /// the workload abstraction.
    pub extra: BTreeMap<String, String>,
}

impl BenchReport {
    /// Renders the human-readable summary block.
    #[must_use]
    pub fn text(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "target={} workload={} threads={} ops/thread={} keys={} batch={} value_len={} prefix={}",
            self.target,
            self.workload,
            self.threads,
            self.ops_per_thread,
            self.keys,
            self.pipeline,
            self.value_len,
            self.prefix,
        );
        let _ = writeln!(
            out,
            "completed={} errors={} elapsed={:.2}s",
            self.completed, self.errors, self.elapsed_secs
        );
        let _ = writeln!(out, "ops/sec={:.0}", self.ops_per_sec);
        for (name, value) in [
            ("p50", self.p50_ns),
            ("p95", self.p95_ns),
            ("p99", self.p99_ns),
            ("p99.9", self.p999_ns),
        ] {
            let _ = writeln!(out, "{name:<4}={value}{}", self.latency_unit);
        }
        let _ = writeln!(out, "max={}ns", self.max_ns);
        if let (Some(written), Some(read)) = (self.bytes_out, self.bytes_in) {
            #[allow(clippy::cast_precision_loss)]
            let completed = self.completed.max(1) as f64;
            #[allow(clippy::cast_precision_loss)]
            let (written, read) = (written as f64, read as f64);
            let _ = writeln!(
                out,
                "bytes_out={written:.0} bytes_in={read:.0} bytes_per_op={:.1}",
                (written + read) / completed
            );
        }
        for (key, value) in &self.extra {
            let _ = writeln!(out, "{key}={value}");
        }
        out
    }
}
