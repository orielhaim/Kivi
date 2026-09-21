//! Minimal simulator reuse: the production Mutation IR and tablet state
//! logic driven deterministically through the simulation kernel.
//!
//! There is exactly one logical implementation (`kivi_state`); the
//! scheduler, seeded RNG, fault policy, and trace below are only a
//! deterministic *driver* for it. If production semantics ever depended on
//! ambient time, randomness, or hash order, these replay assertions would
//! catch it — no `SimTablet` fork exists or may be introduced.

use kivi_core::{EventView, FaultDecision, FaultPolicy, RandomSource};
use kivi_sim::{AllowAll, DropEveryNth, Scheduler, SimRng, Trace};
use kivi_state::{Key, Mutation, ObjectStore};
use kivi_types::WallTimestamp;
use rstest::rstest;
use std::num::NonZeroU64;

/// Builds a deterministic mutation script from the seed: puts, counter adds
/// (including extremes), deletes, and expiry sets over eight keys.
fn script(seed: u64, count: u64) -> Vec<Mutation> {
    let mut rng = SimRng::seed_from_u64(seed);
    let mut mutations = Vec::with_capacity(usize::try_from(count).expect("tiny script"));
    for _ in 0..count {
        let key = Key::from(format!("k{}", rng.below_u64(8)));
        match rng.below_u64(5) {
            0 => mutations.push(Mutation::PutBytes {
                key,
                value: bytes::Bytes::from(rng.below_u64(1_000).to_le_bytes().to_vec()),
            }),
            1 => mutations.push(Mutation::CounterAdd {
                key,
                delta: match rng.below_u64(4) {
                    0 => i64::MAX,
                    1 => i64::MIN,
                    _ => i64::try_from(rng.below_u64(20)).expect("tiny") - 10,
                },
            }),
            2 => mutations.push(Mutation::Delete { key }),
            3 => mutations.push(Mutation::SetExpiry {
                key,
                expiry: kivi_types::Expiry::at(WallTimestamp::from_micros(
                    i64::try_from(rng.below_u64(500_000)).expect("tiny fits"),
                )),
            }),
            _ => mutations.push(Mutation::SetExpiry {
                key,
                expiry: kivi_types::Expiry::NEVER,
            }),
        }
    }
    mutations
}

/// Drives one run: each mutation executes at its scheduled tick with the
/// tick as logical time, under `policy`. The policy is consulted exactly
/// once per event. Returns the final snapshot plus the trace.
fn drive<P: FaultPolicy>(
    seed: u64,
    mut policy: P,
) -> (Vec<(Key, kivi_state::StoredObject)>, Trace<Mutation>) {
    use kivi_sim::RecordedDecision;

    let mutations = script(seed, 300);
    let mut rng = SimRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut scheduler = Scheduler::new(kivi_types::Ticks::from_micros(0));
    for (index, mutation) in mutations.into_iter().enumerate() {
        let tick = u64::try_from(index).expect("tiny script fits");
        let at = kivi_types::Ticks::from_micros(tick * 10);
        scheduler.schedule_at(at, mutation).expect("schedule fits");
    }
    let mut store = ObjectStore::new();
    let mut trace = Trace::new(seed);
    while let Some(scheduled) = scheduler.pop_next() {
        let id = scheduled.id();
        let at = scheduled.at();
        let mutation = scheduled.into_event();
        let decision = policy.decide(&EventView::new(id, at), &(), &mut rng);
        match decision {
            FaultDecision::Allow | FaultDecision::Duplicate => {
                // Duplicates re-run the identical deterministic apply and
                // schedule an identical copy: same rule as the cluster.
                let duplicate = matches!(decision, FaultDecision::Duplicate);
                let _ = store.apply(
                    &mutation,
                    WallTimestamp::from_micros(i64::try_from(at.as_micros()).expect("tick fits")),
                );
                trace.record(id, at, RecordedDecision::Ran, mutation.clone());
                if duplicate {
                    scheduler.schedule_at(at, mutation).expect("copy fits");
                }
            }
            FaultDecision::Drop => {
                trace.record(id, at, RecordedDecision::Dropped, mutation);
            }
            FaultDecision::Defer { until } => {
                scheduler
                    .schedule_at(until, mutation.clone())
                    .expect("defer fits");
                trace.record(id, at, RecordedDecision::Deferred { until }, mutation);
            }
        }
    }
    (store.snapshot_sorted(), trace)
}

#[rstest]
#[case(11)]
#[case(22)]
fn production_mutations_replay_identically(#[case] seed: u64) {
    let (first_state, first_trace) = drive(seed, AllowAll);
    let (second_state, second_trace) = drive(seed, AllowAll);
    assert_eq!(first_state, second_state);
    assert_eq!(first_trace, second_trace);
    assert!(!first_state.is_empty(), "script must touch state");
}

#[rstest]
#[case(11)]
#[case(22)]
fn faulted_mutation_runs_replay_identically(#[case] seed: u64) {
    let every = NonZeroU64::new(7).expect("nonzero");
    let (first_state, first_trace) = drive(seed, DropEveryNth::new(every));
    let (second_state, second_trace) = drive(seed, DropEveryNth::new(every));
    assert_eq!(first_state, second_state);
    assert_eq!(first_trace, second_trace);
}
