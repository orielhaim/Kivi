//! Deterministic event scheduler: total order over `(tick, identifier)`.
//!
//! Events scheduled for the same virtual tick run in schedule-call
//! (insertion) order — the identifier minted at scheduling doubles as the
//! tie-break, so the order key never has ties and replay needs no
//! additional disambiguation. Time only moves forward: popping an event
//! advances the scheduler clock to its tick, scheduling into the past is
//! rejected, and identifier/tick arithmetic is checked rather than wrapping.
//!
//! The scheduler stores payloads opaquely (`E` is unconstrained) and only
//! pops them; the driver loop — pop, consult [`FaultPolicy`](kivi_core::FaultPolicy),
//! run or skip, record to [`Trace`](crate::Trace) — lives with the future
//! network/storage/node simulators and their tests.

use core::time::Duration;

use kivi_core::{EventId, EventIdExhausted, EventView};
use kivi_types::Ticks;

/// One scheduled event as popped from the queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scheduled<E> {
    id: EventId,
    at: Ticks,
    event: E,
}

impl<E> Scheduled<E> {
    /// Returns the stable event identifier.
    #[must_use]
    pub const fn id(&self) -> EventId {
        self.id
    }

    /// Returns the tick this event was scheduled for.
    #[must_use]
    pub const fn at(&self) -> Ticks {
        self.at
    }

    /// Returns the payload by reference.
    #[must_use]
    pub fn event(&self) -> &E {
        &self.event
    }

    /// Consumes the wrapper and returns the payload.
    #[must_use]
    pub fn into_event(self) -> E {
        self.event
    }

    /// Builds the fault-policy view of this event (identity and tick only).
    #[must_use]
    pub const fn view(&self) -> EventView {
        EventView::new(self.id, self.at)
    }
}

/// Synchronous deterministic scheduler over opaque payloads.
///
/// Backed solely by ordered maps, so iteration and popping are fully
/// determined by contents — never by hashing, threading, or wall clocks.
#[derive(Debug, Clone)]
pub struct Scheduler<E> {
    queue: std::collections::BTreeMap<(Ticks, EventId), E>,
    ticks_by_id: std::collections::BTreeMap<EventId, Ticks>,
    next_id: EventId,
    now: Ticks,
}

impl<E> Scheduler<E> {
    /// Creates an empty scheduler whose clock starts at `origin`.
    #[must_use]
    pub fn new(origin: Ticks) -> Self {
        Self {
            queue: std::collections::BTreeMap::new(),
            ticks_by_id: std::collections::BTreeMap::new(),
            next_id: EventId::FIRST,
            now: origin,
        }
    }

    /// Returns the current virtual tick (the last popped event's tick, or
    /// the origin while nothing has run).
    #[must_use]
    pub const fn now(&self) -> Ticks {
        self.now
    }

    /// Returns the number of scheduled (not yet popped or cancelled) events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether no events are scheduled.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Schedules `event` at tick `at`, returning its stable identifier.
    ///
    /// # Errors
    ///
    /// Returns [`ScheduleError::TickInPast`] if `at` precedes the current
    /// tick, or [`ScheduleError::EventIdExhausted`] if no identifier remains.
    /// Nothing is inserted on failure.
    pub fn schedule_at(&mut self, at: Ticks, event: E) -> Result<EventId, ScheduleError> {
        if at < self.now {
            return Err(ScheduleError::TickInPast { at, now: self.now });
        }
        let id = self.next_id;
        self.next_id = id.next()?;
        self.queue.insert((at, id), event);
        self.ticks_by_id.insert(id, at);
        Ok(id)
    }

    /// Schedules `event` at `delay` after the current tick.
    ///
    /// # Errors
    ///
    /// Returns [`ScheduleError::TickOverflow`] if the target tick exceeds
    /// `u64::MAX` microseconds, or the [`schedule_at`](Self::schedule_at)
    /// errors otherwise.
    pub fn schedule_after(&mut self, delay: Duration, event: E) -> Result<EventId, ScheduleError> {
        let delta = u64::try_from(delay.as_micros()).unwrap_or(u64::MAX);
        let target = self
            .now
            .as_micros()
            .checked_add(delta)
            .ok_or(ScheduleError::TickOverflow)?;
        self.schedule_at(Ticks::from_micros(target), event)
    }

    /// Cancels a scheduled event. Returns `false` if `id` is unknown,
    /// already popped, or already cancelled.
    pub fn cancel(&mut self, id: EventId) -> bool {
        if let Some(at) = self.ticks_by_id.remove(&id) {
            self.queue.remove(&(at, id));
            true
        } else {
            false
        }
    }

    /// Peeks at the next event's identifier and tick without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<(EventId, Ticks)> {
        self.queue.iter().next().map(|((at, id), _)| (*id, *at))
    }

    /// Pops the next event in `(tick, identifier)` order and advances the
    /// clock to its tick. Returns `None` when empty.
    pub fn pop_next(&mut self) -> Option<Scheduled<E>> {
        let ((at, id), event) = self.queue.pop_first()?;
        self.ticks_by_id.remove(&id);
        debug_assert!(
            at >= self.now,
            "scheduler clock moved backward: popped {at} before {now}",
            now = self.now
        );
        self.now = at;
        Some(Scheduled { id, at, event })
    }
}

/// Scheduler failure modes. All are explicit rejections, never silent
/// mis-scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum ScheduleError {
    /// The requested tick precedes the scheduler clock.
    #[error("cannot schedule at {at} before current tick {now}")]
    TickInPast {
        /// Requested tick.
        at: Ticks,
        /// Current tick.
        now: Ticks,
    },
    /// `now + delay` exceeds the 64-bit microsecond range.
    #[error("scheduled tick overflows u64 microseconds")]
    TickOverflow,
    /// No event identifier remains to mint.
    #[error("no event identifier remains to mint")]
    EventIdExhausted(#[from] EventIdExhausted),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_tick_events_run_in_schedule_order() {
        let mut scheduler = Scheduler::new(Ticks::from_micros(0));
        let tick = Ticks::from_micros(100);
        let first = scheduler.schedule_at(tick, "first").expect("schedule");
        let second = scheduler.schedule_at(tick, "second").expect("schedule");
        let third = scheduler.schedule_at(tick, "third").expect("schedule");
        assert!(first < second && second < third);
        let popped: Vec<&str> = std::iter::from_fn(|| scheduler.pop_next())
            .map(Scheduled::into_event)
            .collect();
        assert_eq!(popped, vec!["first", "second", "third"]);
        assert_eq!(scheduler.now(), tick);
    }

    #[test]
    fn earlier_ticks_run_first_and_advance_the_clock() {
        let mut scheduler = Scheduler::new(Ticks::from_micros(1_000));
        scheduler
            .schedule_at(Ticks::from_micros(9_000), "late")
            .expect("schedule");
        scheduler
            .schedule_at(Ticks::from_micros(1_500), "early")
            .expect("schedule");
        scheduler
            .schedule_after(Duration::from_millis(1), "after")
            .expect("schedule");
        assert_eq!(scheduler.pop_next().expect("early").into_event(), "early");
        assert_eq!(scheduler.now(), Ticks::from_micros(1_500));
        assert_eq!(scheduler.pop_next().expect("after").into_event(), "after");
        assert_eq!(scheduler.now(), Ticks::from_micros(2_000));
        assert_eq!(scheduler.pop_next().expect("late").into_event(), "late");
        assert!(scheduler.pop_next().is_none());
        assert!(scheduler.is_empty());
    }

    #[test]
    fn scheduling_into_the_past_is_rejected() {
        let mut scheduler: Scheduler<&str> = Scheduler::new(Ticks::from_micros(100));
        assert!(matches!(
            scheduler.schedule_at(Ticks::from_micros(50), "past"),
            Err(ScheduleError::TickInPast { .. })
        ));
        scheduler
            .schedule_at(Ticks::from_micros(100), "present")
            .expect("present is fine");
        assert_eq!(
            scheduler.schedule_at(Ticks::from_micros(10), "past"),
            Err(ScheduleError::TickInPast {
                at: Ticks::from_micros(10),
                now: Ticks::from_micros(100),
            })
        );
    }

    #[test]
    fn tick_overflow_is_explicit() {
        let mut scheduler: Scheduler<&str> = Scheduler::new(Ticks::from_micros(u64::MAX - 1_000));
        assert_eq!(
            scheduler.schedule_after(Duration::from_millis(2), "boom"),
            Err(ScheduleError::TickOverflow)
        );
        assert!(scheduler.is_empty());
    }

    #[test]
    fn cancellation_removes_exactly_one_event() {
        let mut scheduler = Scheduler::new(Ticks::from_micros(0));
        let tick = Ticks::from_micros(10);
        let keep = scheduler.schedule_at(tick, "keep").expect("schedule");
        let drop = scheduler.schedule_at(tick, "drop").expect("schedule");
        assert!(scheduler.cancel(drop));
        assert!(!scheduler.cancel(drop));
        assert!(!scheduler.cancel(EventId::from_u64(999)));
        assert_eq!(scheduler.len(), 1);
        let only = scheduler.pop_next().expect("one remains");
        assert_eq!(only.id(), keep);
        assert_eq!(only.into_event(), "keep");
    }

    #[test]
    fn identifier_exhaustion_is_explicit() {
        let mut scheduler: Scheduler<&str> = Scheduler {
            queue: std::collections::BTreeMap::new(),
            ticks_by_id: std::collections::BTreeMap::new(),
            next_id: EventId::from_u64(u64::MAX),
            now: Ticks::from_micros(0),
        };
        assert_eq!(
            scheduler.schedule_at(Ticks::from_micros(1), "last"),
            Err(ScheduleError::EventIdExhausted(EventIdExhausted))
        );
        assert!(scheduler.is_empty());
        scheduler.next_id = EventId::from_u64(u64::MAX - 1);
        scheduler
            .schedule_at(Ticks::from_micros(1), "penultimate")
            .expect("fits");
        assert_eq!(
            scheduler.schedule_at(Ticks::from_micros(2), "one too many"),
            Err(ScheduleError::EventIdExhausted(EventIdExhausted))
        );
    }

    #[test]
    fn scheduled_views_carry_identity_and_tick() {
        let mut scheduler = Scheduler::new(Ticks::from_micros(5));
        let id = scheduler
            .schedule_at(Ticks::from_micros(9), "x")
            .expect("schedule");
        let peeked = scheduler.peek().expect("peek");
        assert_eq!(peeked, (id, Ticks::from_micros(9)));
        let popped = scheduler.pop_next().expect("pop");
        assert_eq!(
            popped.view(),
            kivi_core::EventView::new(id, Ticks::from_micros(9))
        );
    }
}
