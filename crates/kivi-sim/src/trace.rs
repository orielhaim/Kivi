//! Deterministic trace and replay representation.
//!
//! A [`Trace`] records, per popped event in run order: the stable event
//! identifier, the virtual tick it ran at, the fault decision injected, and
//! the event payload itself. Together with the seed it was constructed with,
//! a trace fully explains a run and replays it: re-driving the same seed,
//! events, and policy must produce an equal trace.
//!
//! Traces are plain data (`PartialEq`, `Clone`, `Debug`) with no I/O, clocks,
//! or maps involved — equality across repeated runs is the replay check.

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
    /// Returns the event identifier.
    #[must_use]
    pub const fn id(&self) -> EventId {
        self.id
    }

    /// Returns the virtual tick the event ran (or would have run) at.
    #[must_use]
    pub const fn at(&self) -> Ticks {
        self.at
    }

    /// Returns the recorded fault decision.
    #[must_use]
    pub const fn decision(&self) -> RecordedDecision {
        self.decision
    }

    /// Returns the event payload.
    #[must_use]
    pub fn event(&self) -> &E {
        &self.event
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

    /// Returns all entries in run order.
    #[must_use]
    pub fn entries(&self) -> &[TraceEntry<E>] {
        &self.entries
    }

    /// Iterates over entries in run order.
    pub fn iter(&self) -> core::slice::Iter<'_, TraceEntry<E>> {
        self.entries.iter()
    }

    /// Returns the number of recorded entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing has been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
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
    fn trace_records_identity_tick_decision_and_payload() {
        let mut trace = Trace::new(42);
        assert!(trace.is_empty());
        assert_eq!(trace.seed(), 42);
        trace.record(
            EventId::from_u64(3),
            Ticks::from_micros(90),
            RecordedDecision::Dropped,
            "packet",
        );
        assert_eq!(trace.len(), 1);
        let entry = &trace.entries()[0];
        assert_eq!(entry.id(), EventId::from_u64(3));
        assert_eq!(entry.decision(), RecordedDecision::Dropped);
        assert_eq!(entry.event(), &"packet");
        assert_eq!(trace.iter().count(), 1);
    }

    #[test]
    fn decisions_map_from_policy_answers() {
        assert_eq!(
            RecordedDecision::from(FaultDecision::Allow),
            RecordedDecision::Ran
        );
        assert_eq!(
            RecordedDecision::from(FaultDecision::Drop),
            RecordedDecision::Dropped
        );
        assert_eq!(
            RecordedDecision::from(FaultDecision::Defer {
                until: Ticks::from_micros(7)
            }),
            RecordedDecision::Deferred {
                until: Ticks::from_micros(7)
            }
        );
    }
}
