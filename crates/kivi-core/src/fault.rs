//! Deterministic fault-injection vocabulary.
//!
//! Faults are decisions, not mechanisms. Before the kernel runs a scheduled
//! event it asks a [`FaultPolicy`], which sees an [`EventView`] (identifier
//! plus scheduled tick — never payload semantics), optionally some typed
//! domain context, and answers with a [`FaultDecision`]. The layering stays:
//!
//! ```text
//! simulation kernel
//!        ↓
//! generic fault decision machinery (this module)
//!        ↓
//! domain-specific context (network/storage/node types in kivi-sim)
//! ```
//!
//! `kivi-core` never names network, storage, or node concepts: domains plug
//! in through the `Ctx` type parameter. The default `Ctx = ()` covers
//! domain-neutral policies; there is exactly one policy trait, never a
//! parallel fault system per subsystem.

use core::fmt;

use kivi_types::Ticks;

use crate::rng::RandomSource;
use crate::schedule::EventId;

/// What a fault policy sees about one scheduled event: its stable identifier
/// and its scheduled tick. Payload type and meaning stay hidden so policies
/// remain reusable across future network, storage, and node simulators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventView {
    id: EventId,
    at: Ticks,
}

impl EventView {
    /// Builds a view of one scheduled event.
    #[must_use]
    pub const fn new(id: EventId, at: Ticks) -> Self {
        Self { id, at }
    }

    /// Returns the event identifier.
    #[must_use]
    pub const fn id(self) -> EventId {
        self.id
    }

    /// Returns the scheduled tick.
    #[must_use]
    pub const fn at(self) -> Ticks {
        self.at
    }
}

impl fmt::Display for EventView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.id, self.at)
    }
}

/// Kernel-level fault decision for one scheduled event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaultDecision {
    /// Run the event now.
    Allow,
    /// Skip the event (recorded in the trace; never silently lost).
    Drop,
    /// Run the event later instead, re-scheduled at `until`.
    ///
    /// Interpreters route deferrals through the scheduler's checked
    /// scheduling path, so a deferral into the past is rejected rather than
    /// applied.
    Defer {
        /// Rescheduled tick (must not precede the current tick).
        until: Ticks,
    },
    /// Run an identical copy of the event at the same tick under a fresh
    /// scheduler identity (duplicated delivery, retried operation).
    Duplicate,
}

impl fmt::Display for FaultDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allow => write!(f, "allow"),
            Self::Drop => write!(f, "drop"),
            Self::Defer { until } => write!(f, "defer-until({until})"),
            Self::Duplicate => write!(f, "duplicate"),
        }
    }
}

/// Decides the fate of each scheduled event before it runs.
///
/// Policies are stateful (`&mut self`) and draw only from the supplied
/// [`RandomSource`]: identical policy state plus identical RNG stream yields
/// identical decisions, which is what makes faulted runs replayable. Policies
/// must not read wall clocks, thread-local RNG, or any other ambient input.
///
/// The `Ctx` parameter carries optional typed domain context (e.g. a network
/// delivery descriptor) defined entirely outside `kivi-core`; `()` means no
/// context. One trait serves every domain — domain simulators instantiate it
/// with their context type rather than inventing parallel machinery.
pub trait FaultPolicy<Ctx = ()> {
    /// Returns the decision for `event` with domain context `ctx`,
    /// optionally drawing from `rng`.
    fn decide(&mut self, event: &EventView, ctx: &Ctx, rng: &mut dyn RandomSource)
    -> FaultDecision;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scripted {
        decisions: Vec<FaultDecision>,
    }

    impl FaultPolicy for Scripted {
        fn decide(&mut self, _: &EventView, (): &(), _: &mut dyn RandomSource) -> FaultDecision {
            self.decisions.pop().unwrap_or(FaultDecision::Allow)
        }
    }

    struct NullRng;

    impl RandomSource for NullRng {
        fn next_u64(&mut self) -> u64 {
            0
        }
    }

    #[test]
    fn policy_sees_only_identity_and_tick() {
        let view = EventView::new(EventId::from_u64(9), Ticks::from_micros(150));
        assert_eq!(view.id(), EventId::from_u64(9));
        assert_eq!(view.at(), Ticks::from_micros(150));
        let mut policy = Scripted {
            decisions: vec![FaultDecision::Drop],
        };
        let mut rng = NullRng;
        assert_eq!(policy.decide(&view, &(), &mut rng), FaultDecision::Drop);
        assert_eq!(policy.decide(&view, &(), &mut rng), FaultDecision::Allow);
    }
}
