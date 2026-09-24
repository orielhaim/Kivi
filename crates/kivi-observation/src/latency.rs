//! Fixed-capacity rolling latency windows and exact retained quantiles.

use std::collections::VecDeque;

use crate::signals::LatencySignal;

/// Exact quantiles over the currently retained bounded latency window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LatencySummary {
    /// Number of retained observations.
    pub sample_count: usize,
    /// Nearest-rank median in nanoseconds.
    pub p50_ns: u64,
    /// Nearest-rank 99th percentile in nanoseconds.
    pub p99_ns: u64,
    /// Nearest-rank 99.9th percentile in nanoseconds.
    pub p999_ns: u64,
    /// Maximum retained latency in nanoseconds.
    pub max_ns: u64,
}

#[derive(Debug)]
pub(crate) struct LatencyWindow {
    values: VecDeque<u64>,
    capacity: usize,
}

impl LatencyWindow {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            values: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub(crate) fn observe(&mut self, signal: LatencySignal) {
        self.observe_value(signal.duration_ns);
    }

    pub(crate) fn observe_value(&mut self, duration_ns: u64) {
        if self.values.len() == self.capacity {
            self.values.pop_front();
        }
        self.values.push_back(duration_ns);
    }

    pub(crate) fn summary(&self) -> Option<LatencySummary> {
        if self.values.is_empty() {
            return None;
        }
        let mut values: Vec<u64> = self.values.iter().copied().collect();
        values.sort_unstable();
        Some(LatencySummary {
            sample_count: values.len(),
            p50_ns: nearest_rank(&values, 50, 100),
            p99_ns: nearest_rank(&values, 99, 100),
            p999_ns: nearest_rank(&values, 999, 1_000),
            max_ns: *values.last().unwrap_or(&0),
        })
    }
}

fn nearest_rank(values: &[u64], numerator: usize, denominator: usize) -> u64 {
    let quotient = values.len() / denominator;
    let remainder = values.len() % denominator;
    let rank = quotient
        .saturating_mul(numerator)
        .saturating_add(remainder.saturating_mul(numerator).div_ceil(denominator))
        .max(1)
        .min(values.len());
    values[rank - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_window_discards_oldest_observations() {
        let mut window = LatencyWindow::new(3);
        for value in 1..=5 {
            window.observe(LatencySignal { duration_ns: value });
        }
        let summary = window.summary().expect("summary");
        assert_eq!(summary.sample_count, 3);
        assert_eq!(summary.p50_ns, 4);
        assert_eq!(summary.p99_ns, 5);
        assert_eq!(summary.p999_ns, 5);
        assert_eq!(summary.max_ns, 5);
    }
}
