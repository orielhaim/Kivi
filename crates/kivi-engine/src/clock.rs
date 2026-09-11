//! Production wall clock: the single sanctioned reader of system time.
//!
//! [`SystemClock::wall_now`] is the only production path that touches
//! `SystemTime`. Tablet and store logic never call it — timestamps travel
//! explicitly from the engine/client boundary inward (and tests pass fixed
//! values), so identical logic runs in real execution, unit tests with
//! controlled time, and deterministic simulation alike.
//!
//! Wall time is used exclusively for expiry comparison. Fencing, ordering,
//! versions, and routing never depend on it, so clock skew can only make an
//! expiry fire early or late — never corrupt state.

use std::time::{SystemTime, UNIX_EPOCH};

use kivi_types::UnixMicros;

/// Production wall-clock source.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct SystemClock;

impl SystemClock {
    /// Reads the current wall time as microseconds since the Unix epoch.
    ///
    /// Clamps pre-epoch readings to zero and saturates past `u64::MAX`
    /// (both practically unreachable; explicit rather than panicking).
    #[must_use]
    pub fn wall_now() -> UnixMicros {
        // Plain match (rather than combinators) keeps both pedantic lints
        // quiet without hiding the pre-epoch fallback.
        let micros = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(span) => span.as_micros(),
            Err(_) => 0,
        };
        UnixMicros::from_micros(u64::try_from(micros).unwrap_or(u64::MAX))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wall_clock_reads_sane_time() {
        let now = SystemClock::wall_now();
        // Past 2020-01-01 and well short of u64 exhaustion: the clock works.
        assert!(now.as_micros() > 1_577_836_800_000_000);
    }
}
