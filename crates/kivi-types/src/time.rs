//! Time value types.
//!
//! These are inert spans and stamps: nothing here reads a clock. Callers pass
//! timestamps in explicitly so production can supply hardware time while
//! deterministic simulation supplies virtual time (RFC §176, §177).
//!
//! Two types prevent mixing unrelated origins:
//!
//! * [`Ticks`] — monotonic virtual microseconds since a clock-defined origin.
//! * [`UnixMicros`] — microseconds since the Unix epoch, for logical metadata
//!   such as expiry (RFC §163).

use core::fmt;
use core::time::Duration;

/// Monotonic virtual timestamp in microseconds.
///
/// The origin is defined by the clock implementation (`kivi-core` provides the
/// `Clock` trait and a manual clock for simulation); only differences and
/// ordering are meaningful across implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ticks(u64);

impl Ticks {
    /// Wraps a raw microsecond count.
    #[must_use]
    pub const fn from_micros(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw microsecond count.
    #[must_use]
    pub const fn as_micros(self) -> u64 {
        self.0
    }

    /// Advances by `delta`, saturating instead of wrapping or truncating.
    ///
    /// Durations wider than 64 bits clamp to `u64::MAX` (fail closed: time
    /// jumps forward to the end rather than landing somewhere arbitrary).
    #[must_use]
    pub fn advance_by(self, delta: Duration) -> Self {
        let wide = delta.as_micros();
        let micros = u64::try_from(wide).unwrap_or(u64::MAX);
        Self(self.0.saturating_add(micros))
    }

    /// Elapsed microseconds from `earlier` to `self`, saturating at zero if
    /// `earlier` is newer (clock skew fails closed to "no time passed").
    #[must_use]
    pub const fn saturating_since(self, earlier: Self) -> Duration {
        Duration::from_micros(self.0.saturating_sub(earlier.0))
    }
}

impl fmt::Display for Ticks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}us", self.0)
    }
}

impl From<Ticks> for u64 {
    /// Returns the raw microsecond count.
    fn from(ticks: Ticks) -> Self {
        ticks.0
    }
}

impl From<u64> for Ticks {
    /// Wraps a raw microsecond count.
    fn from(value: u64) -> Self {
        Self(value)
    }
}

/// Microseconds since the Unix epoch, for logical metadata.
///
/// Stored with the data it describes (e.g. [`Expiry`]); a read treats expired
/// data as absent even if physical reclamation has not run yet (RFC §163).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnixMicros(u64);

impl UnixMicros {
    /// Wraps a raw microsecond count.
    #[must_use]
    pub const fn from_micros(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw microsecond count.
    #[must_use]
    pub const fn as_micros(self) -> u64 {
        self.0
    }
}

impl fmt::Display for UnixMicros {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}us", self.0)
    }
}

impl From<UnixMicros> for u64 {
    /// Returns the raw microsecond count.
    fn from(stamp: UnixMicros) -> Self {
        stamp.0
    }
}

impl From<u64> for UnixMicros {
    /// Wraps a raw microsecond count.
    fn from(value: u64) -> Self {
        Self(value)
    }
}

/// Logical expiration attached to stored data (RFC §163).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Expiry(Option<UnixMicros>);

impl Expiry {
    /// No expiration: the data never logically expires.
    pub const NEVER: Self = Self(None);

    /// Expires at the given logical timestamp.
    #[must_use]
    pub const fn at(when: UnixMicros) -> Self {
        Self(Some(when))
    }

    /// Returns the expiration stamp, or `None` for [`NEVER`](Self::NEVER).
    #[must_use]
    pub const fn as_stamp(self) -> Option<UnixMicros> {
        self.0
    }

    /// Whether the data is logically absent at `now`.
    ///
    /// Pure comparison: the caller supplies `now` from whatever clock (real
    /// or virtual) governs the execution context.
    #[must_use]
    pub const fn is_expired(self, now: UnixMicros) -> bool {
        match self.0 {
            None => false,
            Some(when) => now.as_micros() >= when.as_micros(),
        }
    }
}

impl fmt::Display for Expiry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            None => write!(f, "never"),
            Some(when) => write!(f, "at {when}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_advance_and_measure_spans() {
        let start = Ticks::from_micros(1_000);
        let later = start.advance_by(Duration::from_millis(2));
        assert_eq!(later.as_micros(), 3_000);
        assert_eq!(later.saturating_since(start), Duration::from_millis(2));
        assert_eq!(start.saturating_since(later), Duration::ZERO);
    }

    #[test]
    fn expiry_is_never_or_a_stamp() {
        assert!(!Expiry::NEVER.is_expired(UnixMicros::from_micros(u64::MAX)));
        let deadline = UnixMicros::from_micros(500);
        let expiry = Expiry::at(deadline);
        assert_eq!(expiry.as_stamp(), Some(deadline));
        assert!(!expiry.is_expired(UnixMicros::from_micros(499)));
        assert!(expiry.is_expired(UnixMicros::from_micros(500)));
        assert!(expiry.is_expired(UnixMicros::from_micros(501)));
    }

    #[test]
    fn display_formats() {
        assert_eq!(Ticks::from_micros(7).to_string(), "7us");
        assert_eq!(Expiry::NEVER.to_string(), "never");
    }
}
