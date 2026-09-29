//! Synthetic state machine driven through thousands of scheduled, faulted
//! events: the replay-equivalence proof for the simulation kernel.
//!
//! The driver loop below is the exact shape future virtual network/storage
//! simulators will use - pop, consult the fault policy, run or skip, record
//! to the trace - over their own event payloads.

use core::num::NonZeroU64;
use core::time::Duration;

use kivi_core::{EventView, FaultDecision, FaultPolicy, RandomSource};
use kivi_sim::{
    AllowAll, DropEveryNth, DropWithProbability, RecordedDecision, Scheduler, SimRng, Trace,
};
use kivi_types::Ticks;
use rstest::rstest;

#[derive(Debug, Clone, PartialEq, Eq)]
enum MachineEvent {
    Add(i64),
    Spawn { count: u64, delay_ms: u64 },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
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
fn drive(seed: u64, mut policy: Box<dyn FaultPolicy>) -> (Machine, Trace<MachineEvent>) {
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

/// One scripted policy family, re-constructible so the same scenario can be
/// driven twice from scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    Lossy,
    EveryFourth,
    DeferOnce,
}

impl Fault {
    fn build(self) -> Box<dyn FaultPolicy> {
        match self {
            Self::None => Box::new(AllowAll),
            Self::Lossy => Box::new(DropWithProbability::new(200_000)),
            Self::EveryFourth => Box::new(DropEveryNth::new(NonZeroU64::new(4).expect("nonzero"))),
            Self::DeferOnce => Box::new(DeferOnce { done: false }),
        }
    }
}

/// The replay-equivalence proof: the whole kernel - scheduler order, event
/// identities, fault decisions, and the driven machine - is a pure function
/// of (seed, policy). A hash-map iteration order, a wall clock, or a
/// consumed-counter leak anywhere in the loop breaks these.
#[rstest]
#[case(Fault::None, 1920, 0, 0)]
#[case(Fault::Lossy, 0, 0, 0)]
#[case(Fault::EveryFourth, 0, 0, 0)]
#[case(Fault::DeferOnce, 0, 0, 1)]
fn kernel_replays_exactly(
    #[case] fault: Fault,
    #[case] all_ran: usize,
    #[case] all_dropped: usize,
    #[case] deferred: usize,
) {
    let (machine, trace) = drive(42, fault.build());
    assert_eq!(drive(42, fault.build()), (machine.clone(), trace.clone()));
    let count = |wanted: RecordedDecision| {
        trace
            .iter()
            .filter(|entry| entry.decision() == wanted)
            .count()
    };
    if all_ran > 0 {
        assert_eq!(count(RecordedDecision::Ran), all_ran);
        assert_eq!(count(RecordedDecision::Dropped), all_dropped);
    } else {
        // Not a vacuous all-allow replay: the scripted fault actually bit.
        assert!(count(RecordedDecision::Ran) > 0, "the run did work");
        assert!(
            count(RecordedDecision::Dropped) + deferred > 0,
            "fault {fault:?} injected nothing"
        );
    }
    assert_eq!(
        trace
            .iter()
            .filter(|entry| matches!(entry.decision(), RecordedDecision::Deferred { .. }))
            .count(),
        deferred,
        "fault {fault:?} deferred the wrong number of events"
    );
    // Every `Add(1)` that ran contributed exactly once, so the running total
    // matches the apply count: a double-apply or a lost add breaks it.
    assert_eq!(machine.total, i64::try_from(machine.applied).expect("fits"));
    if all_ran > 0 {
        // Unfaulted: the whole 1,200-event script drains, spawns included.
        assert_eq!(machine.spawned, 240);
        assert_eq!(machine.applied, 1680);
    }
}

#[test]
fn the_seed_is_load_bearing() {
    let (_, first) = drive(42, Fault::Lossy.build());
    let (_, second) = drive(43, Fault::Lossy.build());
    assert_ne!(first, second, "an ignored seed makes every replay vacuous");
}
