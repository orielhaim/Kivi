//! Built-in deterministic fault policies.
//!
//! Small, reviewable policies implementing [`FaultPolicy`]
//! for tests and early simulators. Each policy's decisions are a pure function
//! of its visible state plus the supplied [`RandomSource`]:
//! same construction plus same RNG stream replays the same drops, always.

use core::num::NonZeroU64;

use kivi_core::{EventView, FaultDecision, FaultPolicy, RandomSource};

use crate::net::NetDeliveryContext;

/// Never interferes: every event is allowed. The replay baseline — a faulted
/// run must reduce to the unfaulted run when the policy allows everything.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AllowAll;

impl FaultPolicy for AllowAll {
    fn decide(&mut self, _: &EventView, (): &(), _: &mut dyn RandomSource) -> FaultDecision {
        FaultDecision::Allow
    }
}

// Context-ignoring policies serve every domain: the same rule applies with
// or without delivery context, via one shared implementation.
impl FaultPolicy<NetDeliveryContext> for AllowAll {
    fn decide(
        &mut self,
        view: &EventView,
        _: &NetDeliveryContext,
        rng: &mut dyn RandomSource,
    ) -> FaultDecision {
        FaultPolicy::<()>::decide(self, view, &(), rng)
    }
}

/// Drops every `every`-th decision (1st, 2nd, ... counted from construction).
///
/// With `every = 1` everything is dropped; with `every = N` the Nth, 2Nth,
/// ... decisions drop. Ignores randomness entirely, so drop positions are
/// a fixed function of event order — ideal for testing fencing and retry
/// paths deterministically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropEveryNth {
    every: NonZeroU64,
    seen: u64,
}

impl DropEveryNth {
    /// Builds a policy dropping every `every`-th decision.
    ///
    /// `NonZeroU64` makes a zero period unrepresentable at the type level.
    #[must_use]
    pub const fn new(every: NonZeroU64) -> Self {
        Self { every, seen: 0 }
    }

    /// Returns the drop period.
    #[must_use]
    pub fn every(&self) -> NonZeroU64 {
        self.every
    }

    /// Shared rule for every context instantiation.
    fn decide_impl(&mut self) -> FaultDecision {
        // Diagnostic counter only: saturation keeps the decided pattern a
        // deterministic function of decisions made (never wraps the period).
        self.seen = self.seen.saturating_add(1);
        if self.seen.is_multiple_of(self.every.get()) {
            FaultDecision::Drop
        } else {
            FaultDecision::Allow
        }
    }
}

impl FaultPolicy for DropEveryNth {
    fn decide(&mut self, _: &EventView, (): &(), _: &mut dyn RandomSource) -> FaultDecision {
        self.decide_impl()
    }
}

impl FaultPolicy<NetDeliveryContext> for DropEveryNth {
    fn decide(
        &mut self,
        _: &EventView,
        _: &NetDeliveryContext,
        _: &mut dyn RandomSource,
    ) -> FaultDecision {
        self.decide_impl()
    }
}

/// Maximum parts-per-million value (100% drop probability).
pub const MAX_PPM: u32 = 1_000_000;

/// Drops each event independently with probability `ppm / 1_000_000`, drawn
/// from the supplied RNG (`below_u64(1_000_000) < ppm`).
///
/// Values above [`MAX_PPM`] are clamped to it in [`new`](Self::new) (total
/// constructor, documented); `0` never drops, `MAX_PPM` always drops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DropWithProbability {
    threshold_ppm: u32,
}

impl DropWithProbability {
    /// Builds a probabilistic drop policy, clamping above [`MAX_PPM`].
    #[must_use]
    pub const fn new(ppm: u32) -> Self {
        Self {
            threshold_ppm: if ppm > MAX_PPM { MAX_PPM } else { ppm },
        }
    }

    /// Returns the effective threshold in parts per million.
    #[must_use]
    pub const fn threshold_ppm(self) -> u32 {
        self.threshold_ppm
    }
}

impl FaultPolicy for DropWithProbability {
    fn decide(&mut self, _: &EventView, (): &(), rng: &mut dyn RandomSource) -> FaultDecision {
        self.decide_impl(rng)
    }
}

impl FaultPolicy<NetDeliveryContext> for DropWithProbability {
    fn decide(
        &mut self,
        _: &EventView,
        _: &NetDeliveryContext,
        rng: &mut dyn RandomSource,
    ) -> FaultDecision {
        self.decide_impl(rng)
    }
}

impl DropWithProbability {
    /// Shared rule for every context instantiation.
    fn decide_impl(&mut self, rng: &mut dyn RandomSource) -> FaultDecision {
        if rng.below_u64(u64::from(MAX_PPM)) < u64::from(self.threshold_ppm) {
            FaultDecision::Drop
        } else {
            FaultDecision::Allow
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SimRng;
    use kivi_core::{EventId, RandomSource};
    use kivi_types::Ticks;

    struct ZeroRng;

    impl RandomSource for ZeroRng {
        fn next_u64(&mut self) -> u64 {
            0
        }
    }

    fn view(n: u64) -> EventView {
        EventView::new(EventId::from_u64(n), Ticks::from_micros(n))
    }

    #[test]
    fn allow_all_never_interferes() {
        let mut policy = AllowAll;
        let mut rng = ZeroRng;
        for n in 0..16 {
            assert_eq!(policy.decide(&view(n), &(), &mut rng), FaultDecision::Allow);
        }
    }

    #[test]
    fn drop_every_nth_has_exact_period() {
        use FaultDecision::{Allow, Drop};

        let every = NonZeroU64::new(3).expect("nonzero");
        let mut policy = DropEveryNth::new(every);
        let mut rng = ZeroRng;
        let decisions: Vec<FaultDecision> = (0..9)
            .map(|n| policy.decide(&view(n), &(), &mut rng))
            .collect();
        assert_eq!(
            decisions,
            vec![Allow, Allow, Drop, Allow, Allow, Drop, Allow, Allow, Drop]
        );
    }

    #[test]
    fn probability_policy_is_deterministic_given_the_stream() {
        let mut left = DropWithProbability::new(500_000);
        let mut right = DropWithProbability::new(500_000);
        let mut left_rng = SimRng::seed_from_u64(11);
        let mut right_rng = SimRng::seed_from_u64(11);
        for n in 0..64 {
            assert_eq!(
                left.decide(&view(n), &(), &mut left_rng),
                right.decide(&view(n), &(), &mut right_rng)
            );
        }
        assert_eq!(DropWithProbability::new(u32::MAX).threshold_ppm(), MAX_PPM);
        let mut never = DropWithProbability::new(0);
        let mut always = DropWithProbability::new(MAX_PPM);
        let mut rng = SimRng::seed_from_u64(1);
        for n in 0..16 {
            assert_eq!(never.decide(&view(n), &(), &mut rng), FaultDecision::Allow);
            assert_eq!(always.decide(&view(n), &(), &mut rng), FaultDecision::Drop);
        }
    }
}
