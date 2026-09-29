//! Wall-clock reads at production boundaries.
//!
//! Correctness logic never reads a hardware clock. Wall time arrives through
//! the [`WallClock`] trait, which production implements over the system clock
//! and deterministic contexts satisfy with fixed [`WallTimestamp`]s.
//!
//! Wall reads are fallible ([`WallError`]) - a broken system clock fails loudly
//! at the boundary instead of silently becoming epoch zero or maximum time. The
//! one fail-closed exception is [`wall_now_or_max`], for expiry comparisons
//! where "time unknown" must mean "treat as expired", stated explicitly at
//! the call site.

use jiff::Timestamp;
use kivi_types::WallTimestamp;

/// Source of wall-clock time for production boundaries.
///
/// Implemented once per process over the real system clock (see
/// [`SystemClock`]); deterministic contexts inject fixed
/// [`WallTimestamp`]s or a scripted stub instead. State/tablet logic never
/// touches this - timestamps travel explicitly from the boundary inward.
pub trait WallClock {
    /// Reads the current wall time.
    ///
    /// # Errors
    ///
    /// Returns [`WallError`] when the system clock is unreadable or lies
    /// outside Jiff's representable range.
    fn wall_now(&self) -> Result<WallTimestamp, WallError>;
}

/// Production wall-clock source: the single sanctioned reader of system
/// time.
///
/// Stateless and unit-constructible (`#[derive(Default)]`); call sites hold
/// no clock state, so wall reads cannot entangle with deterministic logic.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct SystemClock;

impl WallClock for SystemClock {
    fn wall_now(&self) -> Result<WallTimestamp, WallError> {
        // `Timestamp::now()` panics on nonsensical system-clock values, so
        // the fallible `TryFrom<SystemTime>` is the boundary instead: a
        // broken clock is an error here, never a panic, never a silent
        // zero. Truncation to micros delegates to
        // `WallTimestamp::from_jiff`, the same rounding every other
        // boundary uses.
        let stamp = Timestamp::try_from(std::time::SystemTime::now()).map_err(|error| {
            WallError::Unrepresentable {
                detail: error.to_string(),
            }
        })?;
        WallTimestamp::from_jiff(stamp).map_err(|_| WallError::OutOfRange {
            micros: stamp.as_microsecond(),
        })
    }
}

/// Wall-clock read failure: explicit, never a silent zero or maximum.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum WallError {
    /// The system clock lies outside Jiff's ±9999-year range (practically
    /// unreachable; explicit rather than panicking like
    /// `Timestamp::now()`).
    #[error("system wall time is outside the representable range: {detail}")]
    Unrepresentable {
        /// The underlying Jiff conversion failure.
        detail: String,
    },
    /// The instant needs more than 64 bits of microseconds (unreachable
    /// for Jiff-range values; checked rather than assumed).
    #[error("wall timestamp {micros}us is outside the storable range")]
    OutOfRange {
        /// The offending raw microsecond count.
        micros: i64,
    },
}

/// Reads the wall clock, failing closed to [`WallTimestamp::MAX`] when the
/// system clock is broken.
///
/// Use only for expiry comparisons, where "time unknown" must mean "treat
/// as expired" (fail closed). Everywhere else, propagate [`WallError`].
/// The fallback is explicit in the name so silent clamping cannot hide at
/// a call site.
#[must_use]
pub fn wall_now_or_max(clock: &impl WallClock) -> WallTimestamp {
    clock.wall_now().unwrap_or(WallTimestamp::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A broken clock must read as maximum time, not as epoch zero, or every
    /// expiry comparison would silently treat live data as unexpired.
    #[test]
    fn fail_closed_fallback_means_expired() {
        struct Broken;
        impl WallClock for Broken {
            fn wall_now(&self) -> Result<WallTimestamp, WallError> {
                Err(WallError::Unrepresentable {
                    detail: "broken".to_owned(),
                })
            }
        }
        let now = wall_now_or_max(&Broken);
        assert_eq!(now, WallTimestamp::MAX);
        assert!(kivi_types::Expiry::at(WallTimestamp::from_micros(1)).is_expired(now));
    }
}
