//! Live provider calibration.
//!
//! Static capability sheets lie under load: an `NVMe` device at 95%
//! bandwidth looks nothing like the same device idle, and a compression
//! ratio measured on text tells nothing about counters. The calibrator
//! continuously folds observations (operation latencies, bandwidth
//! utilization, compression ratios, CPU costs) into each provider's
//! advertised capabilities with bounded-memory exponential moving
//! averages. Placement scores calibrated capabilities, never static
//! sheets.
//!
//! All updates are explicit method calls with caller-supplied timestamps;
//! the calibrator never reads a hardware clock, so deterministic
//! simulation drives it with virtual time.

use crate::provider::ProviderId;

/// Exponential moving average over `f64` samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ewma {
    /// Current estimate.
    value: f64,
    /// Weight of each new sample in `[0, 1]`.
    alpha: f64,
    /// Whether any sample has been observed yet.
    initialized: bool,
}

impl Ewma {
    /// Creates an uninitialized average with the given weight.
    ///
    /// `alpha` near 1.0 reacts fast and forgets fast; near 0.0 is stable
    /// but slow. Calibration uses fast averages for latency (0.2) and
    /// slow ones for compression ratios (0.05).
    #[must_use]
    pub const fn new(alpha: f64) -> Self {
        Self {
            value: 0.0,
            alpha,
            initialized: false,
        }
    }

    /// Observes one sample.
    pub fn observe(&mut self, sample: f64) {
        if self.initialized {
            self.value += self.alpha * (sample - self.value);
        } else {
            self.value = sample;
            self.initialized = true;
        }
    }

    /// Returns the current estimate, or `default` before any sample.
    #[must_use]
    pub const fn get_or(&self, default: f64) -> f64 {
        if self.initialized {
            self.value
        } else {
            default
        }
    }

    /// Returns the current estimate, or `None` before any sample.
    #[must_use]
    pub const fn get(&self) -> Option<f64> {
        if self.initialized {
            Some(self.value)
        } else {
            None
        }
    }
}

/// Per-provider live measurements.
#[derive(Debug, Clone)]
pub struct ProviderCalibration {
    /// Which provider these measurements belong to.
    pub provider: ProviderId,
    /// Mean read latency estimate in nanoseconds.
    pub read_latency_ns: Ewma,
    /// Mean write latency estimate in nanoseconds.
    pub write_latency_ns: Ewma,
    /// Bytes read since process start (monotonic counter).
    pub bytes_read: u64,
    /// Bytes written since process start (monotonic counter).
    pub bytes_written: u64,
    /// Recent read bandwidth estimate, bytes per second.
    pub read_bps: Ewma,
    /// Recent write bandwidth estimate, bytes per second.
    pub write_bps: Ewma,
    /// Observed compression ratio (stored/logical) for compressed
    /// providers. `None` before any observation.
    pub compression_ratio: Ewma,
    /// Observed decompression CPU cost, nanoseconds per byte.
    pub decompress_ns_per_byte: Ewma,
    /// Operation count (for hit/miss accounting by the caller).
    pub ops: u64,
    /// Error count.
    pub errors: u64,
}

impl ProviderCalibration {
    /// Creates an uninitialized calibration for a provider.
    #[must_use]
    pub const fn new(provider: ProviderId) -> Self {
        Self {
            provider,
            read_latency_ns: Ewma::new(0.2),
            write_latency_ns: Ewma::new(0.2),
            bytes_read: 0,
            bytes_written: 0,
            read_bps: Ewma::new(0.1),
            write_bps: Ewma::new(0.1),
            compression_ratio: Ewma::new(0.05),
            decompress_ns_per_byte: Ewma::new(0.1),
            ops: 0,
            errors: 0,
        }
    }

    /// Records one completed read of `bytes` taking `latency_ns`.
    ///
    /// Nanosends and byte counts are far below 2^53, so float
    /// conversion is exact for all reachable inputs.
    #[allow(clippy::cast_precision_loss)]
    pub fn observe_read(&mut self, bytes: u64, latency_ns: u64) {
        self.ops = self.ops.saturating_add(1);
        self.bytes_read = self.bytes_read.saturating_add(bytes);
        self.read_latency_ns.observe(latency_ns as f64);
    }

    /// Records one completed write of `bytes` taking `latency_ns`.
    #[allow(clippy::cast_precision_loss)]
    pub fn observe_write(&mut self, bytes: u64, latency_ns: u64) {
        self.ops = self.ops.saturating_add(1);
        self.bytes_written = self.bytes_written.saturating_add(bytes);
        self.write_latency_ns.observe(latency_ns as f64);
    }

    /// Records one compression outcome: `stored` bytes for `logical`
    /// bytes at `cpu_ns_per_byte` decompression cost.
    #[allow(clippy::cast_precision_loss)]
    pub fn observe_compression(&mut self, logical: u64, stored: u64, cpu_ns_per_byte: f64) {
        if logical == 0 {
            return;
        }
        self.compression_ratio
            .observe(stored as f64 / logical as f64);
        self.decompress_ns_per_byte.observe(cpu_ns_per_byte);
    }

    /// Records one failure.
    pub fn observe_error(&mut self) {
        self.errors = self.errors.saturating_add(1);
    }

    /// Error rate in `[0, 1]`, or `0.0` before any operation.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn error_rate(&self) -> f64 {
        if self.ops == 0 {
            0.0
        } else {
            self.errors as f64 / self.ops as f64
        }
    }

    /// Loaded latency multiplier: `1.0` when unloaded, growing with
    /// observed throughput relative to `provisioned_bps`. The planner uses
    /// this to avoid piling migrations onto a saturated device.
    ///
    /// `observed_bps` should be the recent throughput estimate; callers
    /// that do not track it pass `0.0` for no penalty.
    #[must_use]
    pub fn loaded_penalty(observed_bps: f64, provisioned_bps: f64) -> f64 {
        if provisioned_bps <= 0.0 {
            return 1.0;
        }
        let utilization = (observed_bps / provisioned_bps).clamp(0.0, 1.0);
        1.0 + utilization * utilization * 4.0
    }
}

/// Owns one calibration per provider, keyed by [`ProviderId`].
#[derive(Debug, Default)]
pub struct Calibrator {
    entries: Vec<ProviderCalibration>,
}

impl Calibrator {
    /// Creates an empty calibrator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Returns the calibration for a provider, creating it on first use.
    pub fn for_provider(&mut self, provider: ProviderId) -> &mut ProviderCalibration {
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.provider == provider)
        {
            return &mut self.entries[index];
        }
        self.entries.push(ProviderCalibration::new(provider));
        let last = self.entries.len() - 1;
        &mut self.entries[last]
    }

    /// Returns the calibration for a provider, if any observation exists.
    #[must_use]
    pub fn get(&self, provider: ProviderId) -> Option<&ProviderCalibration> {
        self.entries.iter().find(|entry| entry.provider == provider)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ewma_converges_toward_samples() {
        let mut average = Ewma::new(0.5);
        assert_eq!(average.get(), None);
        average.observe(100.0);
        assert_eq!(average.get(), Some(100.0));
        average.observe(200.0);
        assert_eq!(average.get(), Some(150.0));
    }

    #[test]
    fn loaded_penalty_is_one_when_idle() {
        let idle = ProviderCalibration::loaded_penalty(0.0, 1.0e9);
        assert!((idle - 1.0).abs() < 1e-9);
        let saturated = ProviderCalibration::loaded_penalty(1.0e9, 1.0e9);
        assert!(saturated > 4.0);
    }
}
