//! Virtualized time (RFC §176, §177).
//!
//! Correctness logic never reads a hardware clock. It receives time through
//! the [`Clock`] trait: production implements it over the system clock, while
//! deterministic simulation advances a virtual clock explicitly — including
//! clock jumps, which the simulator models as events (RFC §178), not as
//! surprising observations.

use core::time::Duration;

use kivi_types::Ticks;

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
