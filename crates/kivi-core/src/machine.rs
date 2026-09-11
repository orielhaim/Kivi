//! Deterministic state-machine vocabulary.
//!
//! A [`StateMachine`] consumes one [`StateMachine::Event`] at a time and
//! appends resulting [`StateMachine::Effect`]s to a bounded [`Effects`]
//! collector in causal order. Effects are data, not actions: the surrounding
//! executor (production or simulation) interprets them.
//!
//! Every queue is bounded (RFC §201): [`Effects`] carries an explicit
//! capacity and [`EffectSink::push`] reports [`EffectOverflow`] instead of
//! growing without limit. There is deliberately no unbounded sink, not even
//! for tests — a machine that cannot describe its output bound has not
//! finished its design.

/// Collector of effects with an explicit capacity bound.
#[derive(Debug, Clone)]
pub struct Effects<E> {
    inner: Vec<E>,
    capacity: usize,
}

impl<E> Effects<E> {
    /// Creates an empty collector that accepts at most `capacity` effects.
    ///
    /// A zero capacity is valid: it describes a step that must be pure.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Vec::new(),
            capacity,
        }
    }

    /// Returns the maximum number of effects this collector accepts.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of effects collected so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether no effects have been collected yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterates over collected effects in causal (push) order.
    pub fn iter(&self) -> core::slice::Iter<'_, E> {
        self.inner.iter()
    }

    /// Removes and yields all collected effects in causal order,
    /// leaving the collector empty with its capacity unchanged.
    pub fn drain(&mut self) -> std::vec::Drain<'_, E> {
        self.inner.drain(..)
    }

    /// Consumes the collector and returns the effects in causal order.
    #[must_use]
    pub fn into_vec(self) -> Vec<E> {
        self.inner
    }
}

impl<'a, E> IntoIterator for &'a Effects<E> {
    type Item = &'a E;
    type IntoIter = core::slice::Iter<'a, E>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

impl<E> EffectSink<E> for Effects<E> {
    fn push(&mut self, effect: E) -> Result<(), EffectOverflow> {
        if self.inner.len() >= self.capacity {
            return Err(EffectOverflow {
                capacity: self.capacity,
            });
        }
        self.inner.push(effect);
        Ok(())
    }
}

/// Destination for effects emitted by one [`StateMachine`] step.
///
/// Implementations must be bounded: [`push`](Self::push) fails with
/// [`EffectOverflow`] rather than growing without limit.
pub trait EffectSink<E> {
    /// Appends `effect`, preserving causal order.
    ///
    /// # Errors
    ///
    /// Returns [`EffectOverflow`] if the sink is full. The effect is dropped;
    /// the machine decides whether that is fatal or backpressure.
    fn push(&mut self, effect: E) -> Result<(), EffectOverflow>;
}

/// Reported when an [`EffectSink`] is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[error("effect sink full (capacity {capacity})")]
pub struct EffectOverflow {
    capacity: usize,
}

impl EffectOverflow {
    /// Returns the capacity of the sink that overflowed.
    #[must_use]
    pub const fn capacity(self) -> usize {
        self.capacity
    }
}

/// Deterministic state machine: one event in, zero or more effects out.
///
/// Implementations must be pure with respect to the outside world: the only
/// inputs are the current state, the event, and explicitly passed
/// abstractions (clocks, inputs); the only outputs are the mutated state and
/// pushed effects. No wall-clock reads, no RNG, no I/O.
pub trait StateMachine {
    /// Input decided by the executor (client request, timer firing, peer
    /// message, completion of a previously emitted effect, ...).
    type Event;
    /// Output describing what the executor must do next (send, persist,
    /// schedule, reply, ...). Interpreted by the executor, never acted on
    /// directly.
    type Effect;

    /// Applies `event` to `self`, pushing resulting effects to `effects` in
    /// causal order.
    ///
    /// `S` is generic (not a trait object) so machines compose statically
    /// with zero dispatch cost on the hot path.
    fn step<S: EffectSink<Self::Effect>>(&mut self, event: Self::Event, effects: &mut S);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only example proving the interface works; not shipped logic.
    #[derive(Debug, Default)]
    struct CounterMachine {
        value: i64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum CounterEvent {
        Increment,
        Reset,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum CounterEffect {
        Changed(i64),
    }

    impl StateMachine for CounterMachine {
        type Event = CounterEvent;
        type Effect = CounterEffect;

        fn step<S: EffectSink<Self::Effect>>(&mut self, event: Self::Event, effects: &mut S) {
            match event {
                CounterEvent::Increment => {
                    self.value += 1;
                    let _ = effects.push(CounterEffect::Changed(self.value));
                }
                CounterEvent::Reset => {
                    self.value = 0;
                    let _ = effects.push(CounterEffect::Changed(self.value));
                }
            }
        }
    }

    #[test]
    fn steps_apply_in_order_and_emit_causal_effects() {
        let mut machine = CounterMachine::default();
        let mut effects = Effects::new(8);
        machine.step(CounterEvent::Increment, &mut effects);
        machine.step(CounterEvent::Increment, &mut effects);
        machine.step(CounterEvent::Reset, &mut effects);
        assert_eq!(machine.value, 0);
        assert_eq!(
            effects.into_vec(),
            vec![
                CounterEffect::Changed(1),
                CounterEffect::Changed(2),
                CounterEffect::Changed(0),
            ]
        );
    }

    #[test]
    fn full_sink_reports_overflow_and_keeps_prior_effects() {
        let mut effects: Effects<CounterEffect> = Effects::new(1);
        assert_eq!(effects.capacity(), 1);
        assert!(effects.is_empty());
        effects
            .push(CounterEffect::Changed(1))
            .expect("room for one");
        assert_eq!(
            effects.push(CounterEffect::Changed(2)),
            Err(EffectOverflow { capacity: 1 })
        );
        assert_eq!(effects.len(), 1);
        assert_eq!(effects.iter().count(), 1);
        let drained: Vec<_> = effects.drain().collect();
        assert_eq!(drained, vec![CounterEffect::Changed(1)]);
        assert!(effects.is_empty());
    }

    #[test]
    fn zero_capacity_sink_is_pure_by_construction() {
        let mut effects: Effects<CounterEffect> = Effects::new(0);
        assert_eq!(
            effects.push(CounterEffect::Changed(1)),
            Err(EffectOverflow { capacity: 0 })
        );
    }
}
