//! Bounded, deterministic software access telemetry.
//!
//! The sampler is worker-local and mutable by contract. It retains a fixed
//! number of objects, evicts the least recently observed object on ties by
//! object id, and drains in ascending object id order. It never relies on
//! hash-map iteration order or process-global state.

use std::collections::BTreeMap;

use kivi_observation::ExecutionCriticalitySignal;

use crate::criticality::{AccessSignals, Mutability, TelemetrySource};

/// Default number of objects retained by software telemetry.
pub const DEFAULT_OBJECT_CAPACITY: usize = 4096;

/// Per-object software access counters, maintained by the owning worker
/// on the fast path and drained once per observation window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AccessCounters {
    /// Reads since the last drain.
    pub reads: u64,
    /// Writes since the last drain.
    pub writes: u64,
    /// CPU nanoseconds attributed since the last drain.
    pub cpu_ns: u64,
    /// Bytes touched since the last drain.
    pub bytes_touched: u64,
}

/// Fixed capacity limits for [`SoftwareTelemetry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TelemetryLimits {
    /// Maximum number of retained object entries.
    pub object_capacity: usize,
}

impl Default for TelemetryLimits {
    fn default() -> Self {
        Self::new(super::DEFAULT_OBJECT_CAPACITY)
    }
}

impl TelemetryLimits {
    /// Creates limits with the supplied object capacity.
    #[must_use]
    pub const fn new(object_capacity: usize) -> Self {
        Self { object_capacity }
    }

    /// Returns the maximum number of retained objects.
    #[must_use]
    pub const fn object_capacity(self) -> usize {
        self.object_capacity
    }

    /// Returns the maximum number of retained objects.
    #[must_use]
    pub const fn max_objects(self) -> usize {
        self.object_capacity
    }

    /// Returns the configured object capacity.
    #[must_use]
    pub const fn capacity(self) -> usize {
        self.object_capacity
    }

    /// Returns the configured object capacity.
    #[must_use]
    pub const fn limit(self) -> usize {
        self.object_capacity
    }
}

#[derive(Debug, Clone, Copy)]
struct TelemetryEntry {
    counters: AccessCounters,
    mutability: Mutability,
    logical_bytes: u64,
    last_seen: u64,
}

impl TelemetryEntry {
    fn new(mutability: Mutability, logical_bytes: u64, last_seen: u64) -> Self {
        Self {
            counters: AccessCounters::default(),
            mutability,
            logical_bytes,
            last_seen,
        }
    }
}

/// Bounded software telemetry sampler for worker-local access counters.
///
/// A capacity of zero is valid and retains no objects. This makes the
/// bound explicit rather than silently replacing it with an unbounded map.
#[derive(Debug)]
pub struct SoftwareTelemetry {
    entries: BTreeMap<u64, TelemetryEntry>,
    limits: TelemetryLimits,
    clock: u64,
}

impl Default for SoftwareTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftwareTelemetry {
    /// Creates an empty sampler with the default object capacity.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_OBJECT_CAPACITY)
    }

    /// Creates an empty sampler with a fixed object capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            limits: TelemetryLimits::new(capacity),
            clock: 0,
        }
    }

    /// Creates an empty sampler from explicit limits.
    #[must_use]
    pub fn with_limits(limits: TelemetryLimits) -> Self {
        Self {
            entries: BTreeMap::new(),
            limits,
            clock: 0,
        }
    }

    /// Returns the immutable capacity limits.
    #[must_use]
    pub const fn limits(&self) -> TelemetryLimits {
        self.limits
    }

    /// Returns the maximum number of retained objects.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.limits.object_capacity()
    }

    /// Returns the maximum number of retained objects.
    #[must_use]
    pub const fn object_capacity(&self) -> usize {
        self.capacity()
    }

    /// Returns the maximum number of retained objects.
    #[must_use]
    pub const fn max_objects(&self) -> usize {
        self.capacity()
    }

    /// Returns the maximum number of retained objects.
    #[must_use]
    pub const fn limit(&self) -> usize {
        self.capacity()
    }

    /// Returns the number of retained objects.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether no objects are retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns whether the fixed object capacity is reached.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity()
    }

    /// Returns whether an object currently has metadata or counters.
    #[must_use]
    pub fn contains(&self, object: u64) -> bool {
        self.entries.contains_key(&object)
    }

    /// Records one access on the owning worker.
    pub fn record(&mut self, object: u64, write: bool, bytes: u64, cpu_ns: u64) {
        if !self.entries.contains_key(&object) {
            self.admit(object, Mutability::Mutable, 64);
        }
        if self.capacity() == 0 {
            return;
        }
        self.clock = self.clock.saturating_add(1);
        let clock = self.clock;
        let Some(entry) = self.entries.get_mut(&object) else {
            return;
        };
        entry.last_seen = clock;
        if write {
            entry.counters.writes = entry.counters.writes.saturating_add(1);
        } else {
            entry.counters.reads = entry.counters.reads.saturating_add(1);
        }
        entry.counters.bytes_touched = entry.counters.bytes_touched.saturating_add(bytes);
        entry.counters.cpu_ns = entry.counters.cpu_ns.saturating_add(cpu_ns);
    }

    /// Registers or updates object metadata without recording an access.
    pub fn register(&mut self, object: u64, mutability: Mutability, logical_bytes: u64) {
        if let Some(entry) = self.entries.get_mut(&object) {
            entry.mutability = mutability;
            entry.logical_bytes = logical_bytes;
            return;
        }
        self.admit(object, mutability, logical_bytes);
    }

    /// Removes all counters and metadata for an object.
    pub fn remove(&mut self, object: u64) {
        self.entries.remove(&object);
    }

    /// Drains counters in ascending object id order and clears the window.
    ///
    /// Metadata remains registered for later windows. Objects without an
    /// access in the window are omitted from the returned vector.
    pub fn drain(&mut self) -> Vec<(u64, AccessSignals)> {
        let mut out = Vec::with_capacity(self.entries.len());
        for (&object, entry) in &self.entries {
            let total = entry.counters.reads.saturating_add(entry.counters.writes);
            if total == 0 {
                continue;
            }
            out.push((object, Self::signals_for(entry)));
        }
        for entry in self.entries.values_mut() {
            entry.counters = AccessCounters::default();
        }
        out
    }

    /// Drains access signals in ascending object id order.
    #[must_use]
    pub fn drain_access_signals(&mut self) -> Vec<(u64, AccessSignals)> {
        self.drain()
    }

    /// Converts one object to the typed Phase 12 criticality signal.
    #[must_use]
    pub fn observation_signal(&self, object: u64) -> Option<ExecutionCriticalitySignal> {
        self.entries
            .get(&object)
            .filter(|entry| entry.counters.reads != 0 || entry.counters.writes != 0)
            .map(Self::observation_signal_for)
    }

    /// Drains typed Phase 12 criticality signals in ascending object order.
    pub fn drain_observation_signals(&mut self) -> Vec<(u64, ExecutionCriticalitySignal)> {
        self.drain()
            .into_iter()
            .map(|(object, signals)| (object, Self::observation_signal_from_access(signals)))
            .collect()
    }

    fn admit(&mut self, object: u64, mutability: Mutability, logical_bytes: u64) {
        if self.capacity() == 0 {
            return;
        }
        if self.entries.len() >= self.capacity()
            && let Some(victim) = self
                .entries
                .iter()
                .min_by_key(|(candidate, entry)| (entry.last_seen, **candidate))
                .map(|(candidate, _)| *candidate)
        {
            self.entries.remove(&victim);
        }
        self.entries
            .insert(object, TelemetryEntry::new(mutability, logical_bytes, 0));
    }

    fn signals_for(entry: &TelemetryEntry) -> AccessSignals {
        let total = entry.counters.reads.saturating_add(entry.counters.writes);
        AccessSignals {
            reads: entry.counters.reads,
            writes: entry.counters.writes,
            reuse_distance: 0,
            bytes_touched: entry.counters.bytes_touched / total.max(1),
            cpu_ns: entry.counters.cpu_ns,
            stall_fraction: 0.5,
            mlp: 1.0,
            critical_path_fraction: 0.5,
            tail_fraction: 0.0,
            mutability: entry.mutability,
            logical_bytes: entry.logical_bytes,
            reconstruction_cost_ns: 5_000,
            compression_ratio: None,
        }
    }

    fn observation_signal_for(entry: &TelemetryEntry) -> ExecutionCriticalitySignal {
        let total = entry.counters.reads.saturating_add(entry.counters.writes);
        let latency = entry.counters.cpu_ns.checked_div(total).unwrap_or(1);
        ExecutionCriticalitySignal {
            accesses: total,
            serial_accesses: total,
            critical_path_accesses: total.saturating_add(1) / 2,
            tail_accesses: 0,
            stall_time_ns: entry.counters.cpu_ns,
            access_latency_ns: latency,
            logical_bytes: entry.logical_bytes,
        }
    }

    /// Converts generic access signals into the typed Phase 12 signal.
    #[must_use]
    pub fn observation_signal_from_access(signals: AccessSignals) -> ExecutionCriticalitySignal {
        let total = signals.frequency();
        let latency = signals.cpu_ns.checked_div(total).unwrap_or(1);
        ExecutionCriticalitySignal {
            accesses: total,
            serial_accesses: total,
            critical_path_accesses: total.saturating_add(1) / 2,
            tail_accesses: 0,
            stall_time_ns: signals.cpu_ns,
            access_latency_ns: latency,
            logical_bytes: signals.logical_bytes,
        }
    }
}

impl TelemetrySource for SoftwareTelemetry {
    fn sample(&mut self, object: u64) -> Option<AccessSignals> {
        self.entries
            .get(&object)
            .filter(|entry| entry.counters.reads != 0 || entry.counters.writes != 0)
            .map(Self::signals_for)
    }
}

/// Extension point for hardware telemetry (DAMON age bits, PMU counters,
/// CXL/NEMO rules).
pub trait HardwareTelemetry {
    /// Converts a hardware observation into generic access signals.
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
        assert!(telemetry.drain().is_empty());
        telemetry.record(7, true, 128, 10);
        let signal = telemetry.sample(7).expect("signal");
        assert_eq!(signal.mutability, Mutability::ReadMostly);
    }

    #[test]
    fn capacity_is_explicit_and_drain_is_sorted() {
        let mut telemetry = SoftwareTelemetry::with_capacity(2);
        telemetry.register(1, Mutability::Mutable, 64);
        telemetry.record(1, false, 64, 1);
        telemetry.register(2, Mutability::Mutable, 64);
        telemetry.record(2, false, 64, 1);
        telemetry.register(3, Mutability::Mutable, 64);
        telemetry.record(3, false, 64, 1);
        assert_eq!(telemetry.capacity(), 2);
        assert_eq!(telemetry.len(), 2);
        assert!(!telemetry.contains(1));
        assert!(telemetry.contains(2));
        assert!(telemetry.contains(3));
        let ids: Vec<u64> = telemetry.drain().into_iter().map(|(id, _)| id).collect();
        assert_eq!(ids, vec![2, 3]);
    }

    #[test]
    fn typed_observation_signal_is_derived_deterministically() {
        let mut telemetry = SoftwareTelemetry::with_capacity(1);
        telemetry.register(9, Mutability::ReadMostly, 256);
        telemetry.record(9, false, 256, 400);
        let signal = telemetry.observation_signal(9).expect("typed signal");
        assert_eq!(signal.accesses, 1);
        assert_eq!(signal.serial_accesses, 1);
        assert_eq!(signal.logical_bytes, 256);
    }
}
