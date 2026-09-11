//! Synthetic state machine driven through thousands of scheduled, faulted
//! events: the replay-equivalence proof for the simulation kernel.
//!
//! The driver loop below is the exact shape future virtual network/storage
//! simulators will use — pop, consult the fault policy, run or skip, record
//! to the trace — over their own event payloads.

use core::num::NonZeroU64;
use core::time::Duration;

use kivi_core::{EventView, FaultDecision, FaultPolicy, RandomSource};
use kivi_sim::{
    AllowAll, DropEveryNth, DropWithProbability, RecordedDecision, Scheduler, SimRng, Trace,
};
use kivi_types::Ticks;

#[derive(Debug, Clone, PartialEq, Eq)]
enum MachineEvent {
    Add(i64),
    Spawn { count: u64, delay_ms: u64 },
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Machine {
    total: i64,
    applied: u64,
    spawned: u64,
}

/// Defers the first event once, then allows everything.
struct DeferOnce {
    done: bool,
}

impl FaultPolicy for DeferOnce {
    fn decide(&mut self, event: &EventView, (): &(), _: &mut dyn RandomSource) -> FaultDecision {
        if self.done {
            FaultDecision::Allow
        } else {
            self.done = true;
            FaultDecision::Defer {
                until: Ticks::from_micros(event.at().as_micros() + 25),
            }
        }
    }
}

/// Drives one full run: 1,200 initial events (a fifth of them spawners),
/// faulted per `policy`, traced throughout.
fn drive<P: FaultPolicy>(seed: u64, mut policy: P) -> (Machine, Trace<MachineEvent>) {
    let mut rng = SimRng::seed_from_u64(seed);
    let mut scheduler = Scheduler::new(Ticks::from_micros(0));
    for i in 0..1200u64 {
        // Heavy same-tick collisions: 1,200 events over 400 ticks.
        let at = Ticks::from_micros((i * 37) % 400);
        if i % 5 == 0 {
            scheduler
                .schedule_at(
                    at,
                    MachineEvent::Spawn {
                        count: 3,
                        delay_ms: 10 + (i % 50),
                    },
                )
                .expect("initial schedule fits");
        } else {
            scheduler
                .schedule_at(at, MachineEvent::Add(1))
                .expect("initial schedule fits");
        }
    }
    let mut trace = Trace::new(seed);
    let mut machine = Machine::default();
    while let Some(scheduled) = scheduler.pop_next() {
        let id = scheduled.id();
        let at = scheduled.at();
        let event = scheduled.into_event();
        let decision = policy.decide(&EventView::new(id, at), &(), &mut rng);
        let duplicate = matches!(decision, FaultDecision::Duplicate);
        match decision {
            FaultDecision::Allow | FaultDecision::Duplicate => {
                match event.clone() {
                    MachineEvent::Add(n) => {
                        machine.total += n;
                        machine.applied += 1;
                    }
                    MachineEvent::Spawn { count, delay_ms } => {
                        machine.spawned += 1;
                        for k in 0..count {
                            scheduler
                                .schedule_after(
                                    Duration::from_millis(delay_ms + k),
                                    MachineEvent::Add(1),
                                )
                                .expect("spawned schedule fits");
                        }
                    }
                }
                trace.record(id, at, RecordedDecision::Ran, event.clone());
                if duplicate {
                    let copy = scheduler.schedule_at(at, event.clone()).expect("copy fits");
                    trace.record(copy, at, RecordedDecision::Duplicated, event);
                }
            }
            FaultDecision::Drop => {
                trace.record(id, at, RecordedDecision::Dropped, event);
            }
            FaultDecision::Defer { until } => {
                scheduler
                    .schedule_at(until, event.clone())
                    .expect("deferral fits");
                trace.record(id, at, RecordedDecision::Deferred { until }, event);
            }
        }
    }
    (machine, trace)
}

#[test]
fn allow_all_baseline_has_exact_counts() {
    let (machine, trace) = drive(9, AllowAll);
    // 960 initial adds + 240 spawners x 3 children each, nothing dropped.
    assert_eq!(
        machine,
        Machine {
            total: 1680,
            applied: 1680,
            spawned: 240,
        }
    );
    assert_eq!(trace.len(), 1920);
    assert!(trace.iter().all(|e| e.decision() == RecordedDecision::Ran));
}

#[test]
fn replay_is_identical_with_lossy_policy() {
    let (first_state, first_trace) = drive(42, DropWithProbability::new(200_000));
    let (second_state, second_trace) = drive(42, DropWithProbability::new(200_000));
    assert_eq!(first_state, second_state);
    assert_eq!(first_trace, second_trace);
    // Faults were actually injected (not a vacuous replay of all-Allow).
    assert!(
        first_trace
            .iter()
            .any(|e| e.decision() == RecordedDecision::Dropped)
    );
    assert!(
        first_trace
            .iter()
            .any(|e| e.decision() == RecordedDecision::Ran)
    );
}

#[test]
fn replay_is_identical_with_periodic_drops() {
    let every = NonZeroU64::new(4).expect("nonzero");
    let (first_state, first_trace) = drive(7, DropEveryNth::new(every));
    let (second_state, second_trace) = drive(7, DropEveryNth::new(every));
    assert_eq!(first_state, second_state);
    assert_eq!(first_trace, second_trace);
}

#[test]
fn different_seeds_diverge() {
    let (_, trace_42) = drive(42, DropWithProbability::new(200_000));
    let (_, trace_43) = drive(43, DropWithProbability::new(200_000));
    assert_ne!(trace_42, trace_43);
}

#[test]
fn defer_once_still_replays_exactly() {
    let (first_state, first_trace) = drive(5, DeferOnce { done: false });
    let (second_state, second_trace) = drive(5, DeferOnce { done: false });
    assert_eq!(first_state, second_state);
    assert_eq!(first_trace, second_trace);
    assert_eq!(
        first_trace
            .iter()
            .filter(|e| matches!(e.decision(), RecordedDecision::Deferred { .. }))
            .count(),
        1
    );
}
