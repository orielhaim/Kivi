//! Memory telemetry inputs (DAMON / NEMO as signals, not policy).
//!
//! Linux DAMON access bitmaps, PMU stall counters, and future NEMO
//! hardware rules are *telemetry*: they fill
//! [`AccessSignals`](crate::criticality::AccessSignals) through the
//! [`TelemetrySource`](crate::criticality::TelemetrySource) trait. They
//! never decide placement, never own materialization semantics, and never
//! appear in the correctness path. This module provides the software
//! sampler used on commodity hardware plus the extension trait future
//! hardware sources implement.

use std::collections::HashMap;

use crate::criticality::{AccessSignals, Mutability, TelemetrySource};

/// Per-object software access counters, maintained by the owning worker
/// on the fast path (plain integer increments, no atomics: the worker is
/// the single owner) and drained once per observation window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AccessCounters {
    /// Reads since the last drain.
    pub reads: u64,
    /// Writes since the last drain.
    pub writes: u64,
    /// CPU nanoseconds attributed since the last drain.
    pub cpu_ns: u64,
    /// Bytes touched (sum) since the last drain.
    pub bytes_touched: u64,
}

/// Software telemetry sampler: aggregates [`AccessCounters`] into
/// [`AccessSignals`] with configured object metadata.
///
/// This is the commodity-hardware baseline. A DAMON integration would
/// replace the reuse-distance estimator with age-bit scans; a PMU
/// integration would replace the stall/MLP constants with measured
/// counters; a NEMO integration would push the same fields from hardware
/// rules. All three keep this struct's output schema unchanged.
#[derive(Debug, Default)]
pub struct SoftwareTelemetry {
    counters: HashMap<u64, AccessCounters>,
    mutability: HashMap<u64, Mutability>,
    logical_bytes: HashMap<u64, u64>,
}

impl SoftwareTelemetry {
    /// Creates an empty sampler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            counters: HashMap::new(),
            mutability: HashMap::new(),
            logical_bytes: HashMap::new(),
        }
    }

    /// Records one access. Called on the owning worker's fast path.
    pub fn record(&mut self, object: u64, write: bool, bytes: u64, cpu_ns: u64) {
        let entry = self.counters.entry(object).or_default();
        if write {
            entry.writes = entry.writes.saturating_add(1);
        } else {
            entry.reads = entry.reads.saturating_add(1);
        }
        entry.bytes_touched = entry.bytes_touched.saturating_add(bytes);
        entry.cpu_ns = entry.cpu_ns.saturating_add(cpu_ns);
    }

    /// Registers object metadata (mutability class and size).
    pub fn register(&mut self, object: u64, mutability: Mutability, logical_bytes: u64) {
        self.mutability.insert(object, mutability);
        self.logical_bytes.insert(object, logical_bytes);
    }

    /// Removes an object from telemetry.
    pub fn remove(&mut self, object: u64) {
        self.counters.remove(&object);
        self.mutability.remove(&object);
        self.logical_bytes.remove(&object);
    }

    /// Drains all counters into signals, clearing per-window counts.
    /// Metadata persists across windows.
    pub fn drain(&mut self) -> Vec<(u64, AccessSignals)> {
        let mut out = Vec::with_capacity(self.counters.len());
        for (object, counters) in &self.counters {
            let total = counters.reads.saturating_add(counters.writes);
            if total == 0 {
                continue;
            }
            let bytes = self.logical_bytes.get(object).copied().unwrap_or(64);
            let mean_touched = counters.bytes_touched / total.max(1);
            out.push((
                *object,
                AccessSignals {
                    reads: counters.reads,
                    writes: counters.writes,
                    reuse_distance: 0,
                    bytes_touched: mean_touched,
                    cpu_ns: counters.cpu_ns,
                    stall_fraction: 0.5,
                    mlp: 1.0,
                    critical_path_fraction: 0.5,
                    tail_fraction: 0.0,
                    mutability: self
                        .mutability
                        .get(object)
                        .copied()
                        .unwrap_or(Mutability::Mutable),
                    logical_bytes: bytes,
                    reconstruction_cost_ns: 5_000,
                    compression_ratio: None,
                },
            ));
        }
        for counters in self.counters.values_mut() {
            *counters = AccessCounters::default();
        }
        out
    }
}

impl TelemetrySource for SoftwareTelemetry {
    fn sample(&mut self, object: u64) -> Option<AccessSignals> {
        let counters = self.counters.get(&object).copied().unwrap_or_default();
        let total = counters.reads.saturating_add(counters.writes);
        if total == 0 {
            return None;
        }
        Some(AccessSignals {
            reads: counters.reads,
            writes: counters.writes,
            reuse_distance: 0,
            bytes_touched: counters.bytes_touched / total.max(1),
            cpu_ns: counters.cpu_ns,
            stall_fraction: 0.5,
            mlp: 1.0,
            critical_path_fraction: 0.5,
            tail_fraction: 0.0,
            mutability: self
                .mutability
                .get(&object)
                .copied()
                .unwrap_or(Mutability::Mutable),
            logical_bytes: self.logical_bytes.get(&object).copied().unwrap_or(64),
            reconstruction_cost_ns: 5_000,
            compression_ratio: None,
        })
    }
}

/// Extension point for hardware telemetry (DAMON age bits, PMU counters,
/// CXL/NEMO rules).
///
/// Implementors translate device-specific observations into the generic
/// [`AccessSignals`] schema. The trait is intentionally narrow: hardware
/// proposes signals, the deterministic planner disposes.
pub trait HardwareTelemetry {
    /// Converts a hardware observation into generic access signals.
    ///
    /// Returns `None` when the observation carries no usable signal.
    fn translate(&self, object: u64) -> Option<AccessSignals>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drain_clears_counts_but_keeps_metadata() {
        let mut telemetry = SoftwareTelemetry::new();
        telemetry.register(7, Mutability::ReadMostly, 128);
        telemetry.record(7, false, 128, 50);
        let drained = telemetry.drain();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].1.reads, 1);
        // Second drain without new accesses yields nothing.
        assert!(telemetry.drain().is_empty());
        // Metadata survived: a new access still reports the class.
        telemetry.record(7, true, 128, 10);
        let signal = telemetry.sample(7).expect("signal");
        assert_eq!(signal.mutability, Mutability::ReadMostly);
    }
}
