//! Virtualized time (RFC §176, §177).
//!
//! Correctness logic never reads a hardware clock. It receives time through
//! the [`Clock`] trait: production implements it over the system clock, while
//! deterministic simulation advances a virtual clock explicitly — including
//! clock jumps, which the simulator models as events (RFC §178), not as
//! surprising observations.
//!
//! Two clocks, two types: [`Clock`] yields virtual monotonic [`Ticks`];
//! [`WallClock`] yields wall-clock [`WallTimestamp`]s. Wall reads are
//! fallible ([`WallError`]) — a broken system clock fails loudly at the
//! boundary instead of silently becoming epoch zero or maximum time. The one
//! fail-closed exception is [`wall_now_or_max`], for expiry comparisons
//! where "time unknown" must mean "treat as expired", stated explicitly at
//! the call site.

use core::time::Duration;

use jiff::Timestamp;
use kivi_types::{Ticks, WallTimestamp};

/// Source of time for deterministic logic.
///
/// Object-safe and trivially mockable: simulation and tests use
/// [`ManualClock`], production will supply a hardware-backed implementation
/// in a later phase.
pub trait Clock {
    /// Returns the current timestamp. Pure observation — no side effects.
    fn now(&self) -> Ticks;
}

/// Manually advanced clock for simulation and tests.
///
/// Starts at an explicit origin and moves only forward; time passes if and
/// only if someone advances it, which makes executions exactly replayable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ManualClock {
    now: Ticks,
}

impl ManualClock {
    /// Creates a clock starting at `origin`.
    #[must_use]
    pub const fn new(origin: Ticks) -> Self {
        Self { now: origin }
    }

    /// Moves the clock forward by `delta`.
    pub fn advance_by(&mut self, delta: Duration) {
        self.now = self.now.advance_by(delta);
    }

    /// Moves the clock forward to `target` if it is newer, returning the
    /// previous timestamp.
    ///
    /// Backward targets are ignored (monotonicity is structural): clock jumps
    /// backward are modeled as explicit simulator failure events, never as
    /// silent clock behavior.
    pub fn advance_to(&mut self, target: Ticks) -> Ticks {
        let previous = self.now;
        if target > self.now {
            self.now = target;
        }
        previous
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Ticks {
        self.now
    }
}

/// Source of wall-clock time for production boundaries.
///
/// Implemented once per process over the real system clock (see
/// [`SystemClock`]); deterministic contexts inject fixed
/// [`WallTimestamp`]s or a scripted stub instead. State/tablet logic never
/// touches this — timestamps travel explicitly from the boundary inward.
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

/// Manual wall clock for tests and simulation: yields a scripted stamp,
/// advancing only when told.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ManualWallClock {
    /// The stamp [`wall_now`](WallClock::wall_now) returns.
    now: WallTimestamp,
}

impl ManualWallClock {
    /// Creates a clock fixed at `origin`.
    #[must_use]
    pub const fn new(origin: WallTimestamp) -> Self {
        Self { now: origin }
    }

    /// Moves the clock to `target`.
    pub fn advance_to(&mut self, target: WallTimestamp) {
        self.now = target;
    }
}

impl WallClock for ManualWallClock {
    fn wall_now(&self) -> Result<WallTimestamp, WallError> {
        Ok(self.now)
    }
}

#[cfg(test)]
mod wall_tests {
    use super::*;

    #[test]
    fn system_clock_reads_sane_time() {
        let now = SystemClock.wall_now().expect("system clock works");
        assert!(now > WallTimestamp::from_micros(1_577_836_800_000_000));
        // Renders human-readable (in Jiff range), not raw micros.
        assert!(now.to_string().contains('T'));
    }

    #[test]
    fn manual_wall_clock_is_deterministic() {
        let origin = WallTimestamp::from_micros(1_000_000);
        let clock = ManualWallClock::new(origin);
        assert_eq!(clock.wall_now().expect("manual never fails"), origin);
        let mut clock = clock;
        clock.advance_to(WallTimestamp::from_micros(2_000_000));
        assert_eq!(
            clock.wall_now().expect("manual never fails"),
            WallTimestamp::from_micros(2_000_000)
        );
    }

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
        // MAX expires everything: fail closed.
        assert!(kivi_types::Expiry::at(WallTimestamp::from_micros(1)).is_expired(now));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_clock_advances_only_when_told() {
        let origin = Ticks::from_micros(1_000_000);
        let clock = ManualClock::new(origin);
        assert_eq!(clock.now(), origin);
        let mut clock = clock;
        clock.advance_by(Duration::from_millis(5));
        assert_eq!(clock.now(), Ticks::from_micros(1_005_000));
    }

    #[test]
    fn advance_to_is_monotonic_and_reports_previous() {
        let mut clock = ManualClock::new(Ticks::from_micros(100));
        let previous = clock.advance_to(Ticks::from_micros(250));
        assert_eq!(previous, Ticks::from_micros(100));
        assert_eq!(clock.now(), Ticks::from_micros(250));
        // Backward jumps are ignored, not applied.
        let previous = clock.advance_to(Ticks::from_micros(10));
        assert_eq!(previous, Ticks::from_micros(250));
        assert_eq!(clock.now(), Ticks::from_micros(250));
    }
}
