//! Execution criticality: performance impact beyond hotness.
//!
//! Hotness (access frequency) alone misleads tiering: a frequently
//! accessed region with high memory-level parallelism may hide its latency
//! while a rare serial dependency dominates request latency (AOL, OSDI
//! 2025). [`ExecutionCriticality`] generalizes the RFC §12 heat vector
//! into one cost-model input: frequency plus reuse distance, bytes,
//! CPU/stall contribution, parallelism, critical-path and tail weights,
//! mutability, and reconstruction cost.
//!
//! Future DAMON, PMU, CXL/NEMO, and learned controllers contribute signals
//! through [`TelemetrySource`] without changing policy semantics: they
//! fill the same struct fields the planner already scores.

/// How an object is mutated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mutability {
    /// Never mutated after creation (frozen, checkpoint-adjacent).
    Immutable,
    /// Rarely mutated (configuration, metadata).
    ReadMostly,
    /// Regularly mutated.
    Mutable,
}

/// Raw access signals for one object over one observation window.
///
/// All fields are advisory optimization inputs. Wrong values make
/// placement slower or more expensive, never incorrect.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AccessSignals {
    /// Reads observed in the window.
    pub reads: u64,
    /// Writes observed in the window.
    pub writes: u64,
    /// Mean reuse distance in distinct intervening objects (`0` = hot).
    /// `u64::MAX` means never re-accessed in the window.
    pub reuse_distance: u64,
    /// Bytes touched per access (mean).
    pub bytes_touched: u64,
    /// CPU nanoseconds spent serving this object in the window.
    pub cpu_ns: u64,
    /// Fraction of request stall time attributed to this object,
    /// in `[0, 1]`.
    pub stall_fraction: f64,
    /// Memory-level parallelism: how many concurrent accesses overlap
    /// this object's misses. Higher MLP hides latency (less critical).
    pub mlp: f64,
    /// Fraction of accesses on the request critical path, `[0, 1]`.
    pub critical_path_fraction: f64,
    /// Weight of this object in tail-latency samples, `[0, 1]`.
    pub tail_fraction: f64,
    /// How the object is mutated.
    pub mutability: Mutability,
    /// Bytes of the object.
    pub logical_bytes: u64,
    /// Cost to reconstruct the object from another representation or
    /// source, in nanoseconds. `u64::MAX` means irreconstructible except
    /// from the current representation (never evict without a copy).
    pub reconstruction_cost_ns: u64,
    /// Estimated compressed size ratio (stored/logical) from calibration,
    /// or `None` when unknown.
    pub compression_ratio: Option<f64>,
}

impl AccessSignals {
    /// Zero signals: cold, untouched, unknown cost.
    #[must_use]
    pub const fn cold(logical_bytes: u64) -> Self {
        Self {
            reads: 0,
            writes: 0,
            reuse_distance: u64::MAX,
            bytes_touched: 0,
            cpu_ns: 0,
            stall_fraction: 0.0,
            mlp: 1.0,
            critical_path_fraction: 0.0,
            tail_fraction: 0.0,
            mutability: Mutability::Mutable,
            logical_bytes,
            reconstruction_cost_ns: u64::MAX,
            compression_ratio: None,
        }
    }

    /// Read/write ratio in `[0, 1]` (1.0 = read-only in the window).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn read_ratio(&self) -> f64 {
        let total = self.reads.saturating_add(self.writes);
        if total == 0 {
            return 0.5;
        }
        self.reads as f64 / total as f64
    }

    /// Access frequency (reads + writes) in the window.
    #[must_use]
    pub const fn frequency(&self) -> u64 {
        self.reads.saturating_add(self.writes)
    }
}

/// The planner's single cost-model input per object.
///
/// Computed from [`AccessSignals`] by [`criticality_of`]. Future signal
/// sources (DAMON age bits, PMU stall counters, NEMO rules, learned
/// predictors) adjust the signals, never the scoring function signature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExecutionCriticality {
    /// Composite score: higher means "keep fast". Nonnegative, unbounded.
    /// The planner compares scores relatively; absolute values carry no
    /// meaning across workloads.
    pub score: f64,
    /// Amortized offcore-latency style term: stall-weighted latency per
    /// byte after MLP discounting. Higher = more performance-critical.
    pub amortized_latency_per_byte_ns: f64,
    /// Whether compression is expected to pay (ratio good enough and CPU
    /// cheap enough relative to criticality).
    pub compression_worthy: bool,
    /// Whether the object is write-hot (migration must handle mutation).
    pub write_hot: bool,
}

impl ExecutionCriticality {
    /// Minimum score: cold, never accessed.
    #[must_use]
    pub const fn cold() -> Self {
        Self {
            score: 0.0,
            amortized_latency_per_byte_ns: 0.0,
            compression_worthy: false,
            write_hot: false,
        }
    }
}

/// Scores access signals into an [`ExecutionCriticality`].
///
/// The model follows AOL (OSDI 2025) in spirit: frequency is divided by
/// memory-level parallelism (parallel misses hide latency), multiplied by
/// stall and critical-path contribution, normalized per byte so a 1 MiB
/// cold blob does not outrank a 64 B serial dependency. Compression
/// worthiness additionally requires a measured ratio below 0.85 and a
/// non-write-hot pattern (compressing write-hot objects churns CPU).
///
/// `dram_read_ns` is the calibrated DRAM read latency used as the
/// latency unit; callers pass the current calibrated value so the score
/// tracks loaded hardware.
///
/// This function is pure and deterministic: same signals and latency
/// produce the same criticality, which keeps simulation replayable.
///
/// Counters are window-bounded (never near 2^53), so `u64`-to-`f64`
/// conversion is exact for all reachable inputs.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn criticality_of(signals: &AccessSignals, dram_read_ns: f64) -> ExecutionCriticality {
    let frequency = signals.frequency() as f64;
    let mlp = signals.mlp.clamp(1.0, 64.0);
    let stall = signals.stall_fraction.clamp(0.0, 1.0);
    let critical = signals.critical_path_fraction.clamp(0.0, 1.0);
    let tail = signals.tail_fraction.clamp(0.0, 1.0);
    let bytes = (signals.logical_bytes.max(1)) as f64;
    let latency = dram_read_ns.max(1.0);

    // Effective serial accesses: frequency discounted by parallelism,
    // then weighted by where the accesses sit (stall, critical path,
    // tail). An object accessed 1000× off the critical path with high
    // MLP scores below one accessed 10× serially on it: off-path
    // accesses count roughly 10× less per access, so placement never
    // mistakes parallelism-hidden hotness for criticality.
    let serial = frequency / mlp;
    let weight = 0.1 + 0.9 * (0.4 * stall + 0.4 * critical + 0.2 * tail).clamp(0.0, 1.0);
    let amortized = serial * weight * latency / bytes;
    // Score keeps an absolute term (serial work exists even for big
    // objects) plus the per-byte term (small serial dependencies win).
    let score = serial * weight * (1.0 + 8.0 / bytes.sqrt());

    let ratio = signals.compression_ratio.unwrap_or(1.0);
    let write_hot = signals.writes > signals.reads && signals.writes > 4;
    let compression_worthy =
        ratio < 0.85 && !write_hot && signals.mutability != Mutability::Mutable;

    ExecutionCriticality {
        score,
        amortized_latency_per_byte_ns: amortized,
        compression_worthy,
        write_hot,
    }
}

/// A source of access signals.
///
/// Software sampling implements this today; future DAMON age-bit scans,
/// PMU stall counters, CXL/NEMO telemetry rules, and learned predictors
/// implement the same method and fill the same [`AccessSignals`] fields.
/// Policy code never branches on the concrete source.
pub trait TelemetrySource {
    /// Samples signals for one object over the last window.
    ///
    /// Returns `None` when the source has no data (cold objects report
    /// [`AccessSignals::cold`] from the caller instead).
    fn sample(&mut self, object: u64) -> Option<AccessSignals>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_dependency_outranks_parallel_hot_object() {
        let serial = AccessSignals {
            reads: 10,
            writes: 0,
            reuse_distance: 2,
            bytes_touched: 64,
            cpu_ns: 1_000,
            stall_fraction: 0.9,
            mlp: 1.0,
            critical_path_fraction: 1.0,
            tail_fraction: 0.8,
            mutability: Mutability::ReadMostly,
            logical_bytes: 64,
            reconstruction_cost_ns: 500,
            compression_ratio: None,
        };
        let parallel_hot = AccessSignals {
            reads: 1_000,
            writes: 0,
            reuse_distance: 0,
            bytes_touched: 64,
            cpu_ns: 1_000,
            stall_fraction: 0.05,
            mlp: 16.0,
            critical_path_fraction: 0.0,
            tail_fraction: 0.0,
            mutability: Mutability::ReadMostly,
            logical_bytes: 64,
            reconstruction_cost_ns: 500,
            compression_ratio: None,
        };
        let serial_score = criticality_of(&serial, 90.0).score;
        let hot_score = criticality_of(&parallel_hot, 90.0).score;
        assert!(
            serial_score > hot_score,
            "serial {serial_score} should beat parallel-hot {hot_score}"
        );
    }

    #[test]
    fn write_hot_objects_are_not_compression_worthy() {
        let signals = AccessSignals {
            reads: 1,
            writes: 100,
            reuse_distance: 0,
            bytes_touched: 128,
            cpu_ns: 0,
            stall_fraction: 0.5,
            mlp: 1.0,
            critical_path_fraction: 0.5,
            tail_fraction: 0.0,
            mutability: Mutability::Mutable,
            logical_bytes: 128,
            reconstruction_cost_ns: 1_000,
            compression_ratio: Some(0.3),
        };
        let criticality = criticality_of(&signals, 90.0);
        assert!(criticality.write_hot);
        assert!(!criticality.compression_worthy);
    }
}
