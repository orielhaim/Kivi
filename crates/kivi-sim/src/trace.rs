//! Deterministic trace and replay representation.
//!
//! A [`Trace`] records, per popped event in run order: the stable event
//! identifier, the virtual tick it ran at, the fault decision injected, and
//! the event payload itself. Together with the seed it was constructed with,
//! a trace fully explains a run and replays it: re-driving the same seed,
//! events, and policy must produce an equal trace.
//!
//! Traces are plain data (`PartialEq`, `Clone`, `Debug`) with no I/O, clocks,
//! or maps involved - equality across repeated runs is the replay check.

use core::fmt;

use kivi_core::{EventId, FaultDecision};
use kivi_types::Ticks;

/// What the kernel did with one popped event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecordedDecision {
    /// The event ran.
    Ran,
    /// The event was skipped by fault injection.
    Dropped,
    /// The event was re-scheduled at `until` (under a fresh identifier).
    Deferred {
        /// Tick the event was deferred to.
        until: Ticks,
    },
    /// An identical copy of the event was scheduled at the same tick.
    Duplicated,
}

impl fmt::Display for RecordedDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ran => write!(f, "ran"),
            Self::Dropped => write!(f, "dropped"),
            Self::Deferred { until } => write!(f, "deferred-until({until})"),
            Self::Duplicated => write!(f, "duplicated"),
        }
    }
}

impl From<FaultDecision> for RecordedDecision {
    /// Maps a policy decision to its recorded form (`Allow` records as `Ran`).
    fn from(decision: FaultDecision) -> Self {
        match decision {
            FaultDecision::Allow => Self::Ran,
            FaultDecision::Drop => Self::Dropped,
            FaultDecision::Defer { until } => Self::Deferred { until },
            FaultDecision::Duplicate => Self::Duplicated,
        }
    }
}

/// One recorded event: identity, tick, injected fault, and payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceEntry<E> {
    id: EventId,
    at: Ticks,
    decision: RecordedDecision,
    event: E,
}

impl<E> TraceEntry<E> {
    /// Returns the recorded fault decision.
    #[must_use]
    pub const fn decision(&self) -> RecordedDecision {
        self.decision
    }
}

impl<E: fmt::Display> fmt::Display for TraceEntry<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}@{} {} {}",
            self.id, self.at, self.decision, self.event
        )
    }
}

/// Ordered record of a whole simulation run, plus the seed reproducing it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trace<E> {
    seed: u64,
    entries: Vec<TraceEntry<E>>,
}

impl<E> Trace<E> {
    /// Creates an empty trace for a run driven by `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            entries: Vec::new(),
        }
    }

    /// Returns the seed reproducing the recorded run.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Appends one entry in run order.
    pub fn record(&mut self, id: EventId, at: Ticks, decision: RecordedDecision, event: E) {
        self.entries.push(TraceEntry {
            id,
            at,
            decision,
            event,
        });
    }

    /// Iterates over entries in run order.
    pub fn iter(&self) -> core::slice::Iter<'_, TraceEntry<E>> {
        self.entries.iter()
    }
}

impl<'a, E> IntoIterator for &'a Trace<E> {
    type Item = &'a TraceEntry<E>;
    type IntoIter = core::slice::Iter<'a, TraceEntry<E>>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_answers_map_onto_recorded_decisions() {
        use RecordedDecision as R;
        assert_eq!(R::from(FaultDecision::Allow), R::Ran);
        assert_eq!(R::from(FaultDecision::Drop), R::Dropped);
        assert_eq!(R::from(FaultDecision::Duplicate), R::Duplicated);
        assert_eq!(
            R::from(FaultDecision::Defer {
                until: Ticks::from_micros(7)
            }),
            R::Deferred {
                until: Ticks::from_micros(7)
            }
        );
    }
}
