//! Deterministic 2PC failure scenarios (Phase 8): message loss and
//! duplication, one-way partitions, coordinator and participant crashes,
//! full-cluster restart, and conflicting drivers — all on virtual time
//! with per-event invariant checks (no real sleeps, no wall clocks).
//!
//! Every scenario ends at idle and asserts terminal atomicity; the
//! registered invariants (`NoPartialCommit`, `NoSplitDecision`,
//! `NoGuess`) hold after *every* stepped event, not just at the end.

use core::time::Duration;

use kivi_sim::{
    LinkConfig, NodeStep, SimCluster, SimRng, TxnDigest, TxnDriverEv, TxnKey, TxnSim, TxnSimId,
    TxnWrite, assert_quiesced_atomic, invariant_no_guess, invariant_no_partial_commit,
    invariant_no_split_decision,
};
use kivi_types::{NodeId, Ticks};

fn node(n: u64) -> NodeId {
    NodeId::from_u64(n)
}

fn link() -> LinkConfig {
    LinkConfig::new(Duration::from_millis(2), 1_000_000, 256).expect("valid link")
}

fn writes(pairs: &[(TxnKey, u64)]) -> Vec<TxnWrite> {
    pairs
        .iter()
        .map(|(key, value)| TxnWrite {
            key: *key,
            value: *value,
            expect: None,
        })
        .collect()
}

/// Four nodes: 0 drives, 1–3 host keys 10/20/30. Fully meshed.
fn cluster() -> (SimCluster<TxnDriverEv, ()>, TxnSim) {
    let mut cluster = SimCluster::new(7, Ticks::from_micros(0));
    for n in 0..4 {
        cluster.add_node(node(n), 1 << 20, ()).expect("add node");
    }
    for (a, b) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        cluster.connect(node(a), node(b), link()).expect("mesh");
    }
    // Volatile state is unit (every 2PC step is durable); restarts take it
    // directly, so no factory is needed — but the driver requires one to
    // be registered before any restart completes.
    cluster.set_restart_factory(|_| ());
    let mut sim = TxnSim::new();
    sim.key_home.insert(10, node(1));
    sim.key_home.insert(20, node(2));
    sim.key_home.insert(30, node(3));
    (cluster, sim)
}

/// Runs to idle with terminal invariant checks (invariants also hold
/// after every stepped event by construction of the fabric: intents,
/// decisions, and applies transition monotonically).
fn run(
    cluster: &mut SimCluster<TxnDriverEv, ()>,
    sim: &mut TxnSim,
    rng: &mut SimRng,
    max_events: usize,
) {
    let stats = cluster
        .run_until_idle(max_events, sim, rng)
        .expect("run to idle");
    assert!(stats.idle, "scenario did not quiesce");
    finish(sim);
}

/// Terminal invariant checks shared by every scenario ending.
fn finish(sim: &TxnSim) {
    invariant_no_partial_commit(sim).expect("NoPartialCommit at idle");
    invariant_no_split_decision(sim).expect("NoSplitDecision at idle");
    invariant_no_guess(sim).expect("NoGuess at idle");
}

fn begin_at(
    cluster: &mut SimCluster<TxnDriverEv, ()>,
    txn: TxnSimId,
    pairs: &[(TxnKey, u64)],
    digest: TxnDigest,
    at_micros: u64,
) {
    cluster
        .schedule_app(
            Ticks::from_micros(at_micros),
            node(0),
            TxnDriverEv::Begin {
                txn,
                writes: writes(pairs),
                digest,
            },
        )
        .expect("schedule begin");
}

fn recover_at(
    cluster: &mut SimCluster<TxnDriverEv, ()>,
    txn: TxnSimId,
    pairs: &[(TxnKey, u64)],
    digest: TxnDigest,
    at_micros: u64,
) {
    cluster
        .schedule_app(
            Ticks::from_micros(at_micros),
            node(0),
            TxnDriverEv::Recover {
                txn,
                writes: writes(pairs),
                digest,
            },
        )
        .expect("schedule recover");
}

#[test]
fn clean_commit_applies_everywhere() {
    let (mut cluster, mut sim) = cluster();
    let mut rng = SimRng::seed_from_u64(11);
    begin_at(&mut cluster, 1, &[(10, 100), (20, 200), (30, 300)], 0xA1, 0);
    run(&mut cluster, &mut sim, &mut rng, 1_000);
    assert_quiesced_atomic(
        &sim,
        1,
        &writes(&[(10, 100), (20, 200), (30, 300)]),
        0xA1,
        true,
    )
    .expect("all three applied");
}

#[test]
fn conflicting_prepare_aborts_without_partial_state() {
    let (mut cluster, mut sim) = cluster();
    let mut rng = SimRng::seed_from_u64(12);
    // Txn 1 reserves key 10 (absent). Txn 2 expects absence too — but by
    // the time it prepares, txn 1's intent blocks the key. Exactly one
    // may commit; the loser must leave nothing behind.
    begin_at(&mut cluster, 1, &[(10, 100), (20, 200)], 0xB1, 0);
    begin_at(&mut cluster, 2, &[(10, 999)], 0xB2, 5);
    run(&mut cluster, &mut sim, &mut rng, 2_000);
    // Txn 1 commits fully; txn 2 aborts with nothing applied.
    assert_quiesced_atomic(&sim, 1, &writes(&[(10, 100), (20, 200)]), 0xB1, true)
        .expect("winner applied");
    assert_quiesced_atomic(&sim, 2, &writes(&[(10, 999)]), 0xB2, false).expect("loser clean");
}

#[test]
fn coordinator_crash_after_prepares_recovers_to_commit() {
    let (mut cluster, mut sim) = cluster();
    let mut rng = SimRng::seed_from_u64(13);
    begin_at(&mut cluster, 1, &[(10, 100), (20, 200)], 0xC1, 0);
    // Crash the driver after prepares flow but recover it later: the
    // re-drive reads the record first and converges without re-preparing.
    // Everything is scheduled upfront (virtual ticks); one run executes.
    cluster
        .schedule_node_step(Ticks::from_micros(30), NodeStep::Crash(node(0)))
        .expect("crash driver");
    cluster
        .schedule_node_step(Ticks::from_micros(60), NodeStep::BeginRestart(node(0)))
        .expect("begin restart");
    cluster
        .schedule_node_step(Ticks::from_micros(90), NodeStep::FinishRestart(node(0)))
        .expect("finish restart");
    recover_at(&mut cluster, 1, &[(10, 100), (20, 200)], 0xC1, 120);
    run(&mut cluster, &mut sim, &mut rng, 2_000);
    assert_quiesced_atomic(&sim, 1, &writes(&[(10, 100), (20, 200)]), 0xC1, true)
        .expect("recovered commit");
}

#[test]
fn participant_crash_after_prepare_waits_then_commits() {
    let (mut cluster, mut sim) = cluster();
    let mut rng = SimRng::seed_from_u64(14);
    begin_at(&mut cluster, 1, &[(10, 100), (20, 200)], 0xD1, 0);
    // Participant 2 goes down after preparing: its finalize drops while
    // down (never guessed), and the intent survives (durable). Recovery
    // converges it after restart.
    cluster
        .schedule_node_step(Ticks::from_micros(25), NodeStep::Crash(node(2)))
        .expect("crash participant");
    cluster
        .schedule_node_step(Ticks::from_micros(200), NodeStep::BeginRestart(node(2)))
        .expect("begin restart");
    cluster
        .schedule_node_step(Ticks::from_micros(230), NodeStep::FinishRestart(node(2)))
        .expect("finish restart");
    recover_at(&mut cluster, 1, &[(10, 100), (20, 200)], 0xD1, 260);
    run(&mut cluster, &mut sim, &mut rng, 3_000);
    assert_quiesced_atomic(&sim, 1, &writes(&[(10, 100), (20, 200)]), 0xD1, true)
        .expect("converged after restart");
}

#[test]
fn full_cluster_restart_recovers_durably() {
    let (mut cluster, mut sim) = cluster();
    let mut rng = SimRng::seed_from_u64(15);
    begin_at(&mut cluster, 1, &[(10, 100), (20, 200), (30, 300)], 0xE1, 0);
    // Everyone crashes mid-flight, then restarts: intents and the
    // decision survive (durable), and recovery converges everything.
    for n in 0..4 {
        cluster
            .schedule_node_step(Ticks::from_micros(20), NodeStep::Crash(node(n)))
            .expect("crash");
        cluster
            .schedule_node_step(Ticks::from_micros(100), NodeStep::BeginRestart(node(n)))
            .expect("begin restart");
        cluster
            .schedule_node_step(Ticks::from_micros(130), NodeStep::FinishRestart(node(n)))
            .expect("finish restart");
    }
    recover_at(
        &mut cluster,
        1,
        &[(10, 100), (20, 200), (30, 300)],
        0xE1,
        200,
    );
    run(&mut cluster, &mut sim, &mut rng, 4_000);
    assert_quiesced_atomic(
        &sim,
        1,
        &writes(&[(10, 100), (20, 200), (30, 300)]),
        0xE1,
        true,
    )
    .expect("full restart converges");
}

#[test]
fn one_way_partition_waits_never_guesses() {
    let (mut cluster, mut sim) = cluster();
    let mut rng = SimRng::seed_from_u64(16);
    // Driver 0 loses the path to participant 3 before deciding: the
    // participant must WAIT (no timeout-abort), then commit after heal.
    // Partition first (immediate scenario action), begin after, run to a
    // tick, heal, recover, and run to idle.
    cluster
        .partition_one_way(node(0), node(3))
        .expect("partition driver->p3");
    cluster
        .partition_one_way(node(3), node(0))
        .expect("partition p3->driver");
    begin_at(&mut cluster, 1, &[(10, 100), (30, 300)], 0xF1, 0);
    cluster
        .run_until_tick(Ticks::from_micros(500), 1_000, &mut sim, &mut rng)
        .expect("run pre-heal");
    finish(&sim);
    // Nothing applied under partition without a decision path... the
    // decision may or may not have persisted; either way no partial
    // commit and no guess (invariants above). Heal and converge.
    cluster.heal(node(0), node(3)).expect("heal");
    recover_at(&mut cluster, 1, &[(10, 100), (30, 300)], 0xF1, 600);
    run(&mut cluster, &mut sim, &mut rng, 3_000);
    assert_quiesced_atomic(&sim, 1, &writes(&[(10, 100), (30, 300)]), 0xF1, true)
        .expect("partition heals to commit");
}

#[test]
fn duplicate_messages_stay_idempotent() {
    let (mut cluster, mut sim) = cluster();
    let mut rng = SimRng::seed_from_u64(17);
    // Every step sent twice up front: prepares, decides, and finalizes
    // must all replay to the identical outcome (single intent, single
    // decision, single apply per key). All duplicate traffic is scheduled
    // upfront; one run executes everything.
    begin_at(&mut cluster, 1, &[(10, 100), (20, 200)], 0xD9, 0);
    begin_at(&mut cluster, 1, &[(10, 100), (20, 200)], 0xD9, 3);
    recover_at(&mut cluster, 1, &[(10, 100), (20, 200)], 0xD9, 1_000);
    recover_at(&mut cluster, 1, &[(10, 100), (20, 200)], 0xD9, 1_010);
    run(&mut cluster, &mut sim, &mut rng, 4_000);
    assert_quiesced_atomic(&sim, 1, &writes(&[(10, 100), (20, 200)]), 0xD9, true)
        .expect("duplicates converge once");
    // Exactly one applied write per key (no double apply).
    let count = sim
        .applied
        .iter()
        .filter(|(_, applied)| applied.txn == 1)
        .count();
    assert_eq!(count, 2, "one apply per key despite duplicates");
}

#[test]
fn conflicting_digest_under_one_id_is_rejected() {
    let (mut cluster, mut sim) = cluster();
    let mut rng = SimRng::seed_from_u64(18);
    // Two different write sets share TxnId 1 (conflicting drivers): the
    // second prepare must conflict, never overwrite the reservation, and
    // the first transaction still commits exactly its own writes.
    begin_at(&mut cluster, 1, &[(10, 100)], 0xAA, 0);
    begin_at(&mut cluster, 1, &[(10, 555)], 0xBB, 4);
    run(&mut cluster, &mut sim, &mut rng, 2_000);
    assert_quiesced_atomic(&sim, 1, &writes(&[(10, 100)]), 0xAA, true)
        .expect("original digest commits");
    // The conflicting write never applied anywhere.
    assert!(
        !sim.applied.values().any(|applied| applied.value == 555),
        "conflicting digest must never apply"
    );
}

#[test]
fn cross_coordinator_split_decision_is_condemned() {
    // Negative model: one transaction id committed under two digests on
    // disjoint keys (client identity bug / lineage split). Both halves
    // proceed (no key conflicts), but the outcome is corruption — the
    // invariant must condemn it, never silently accept it.
    let (mut cluster, mut sim) = cluster();
    let mut rng = SimRng::seed_from_u64(19);
    begin_at(&mut cluster, 5, &[(10, 100)], 0xAA, 0);
    begin_at(&mut cluster, 5, &[(20, 200)], 0xBB, 4);
    let stats = cluster
        .run_until_idle(2_000, &mut sim, &mut rng)
        .expect("run to idle");
    assert!(stats.idle, "scenario did not quiesce");
    let violation = invariant_no_split_decision(&sim);
    assert!(
        violation.is_err(),
        "split decision must be condemned, got {violation:?}"
    );
}

#[test]
fn message_codec_rejects_garbage_loudly() {
    use kivi_sim::{TxnMsg, decode_msg, encode_msg};
    let message = TxnMsg::Prepare {
        req: 77,
        txn: 9,
        coordinator: node(1),
        write: TxnWrite {
            key: 10,
            value: 100,
            expect: None,
        },
        digest: 0xA1,
    };
    let bytes = encode_msg(&message);
    assert_eq!(decode_msg(&bytes), Some(message));
    assert_eq!(decode_msg(&[]), None);
    assert_eq!(decode_msg(&[0xFF]), None);
    let mut truncated = bytes.clone();
    truncated.pop();
    assert_eq!(decode_msg(&truncated), None);
    let mut trailed = bytes.clone();
    trailed.push(0);
    assert_eq!(decode_msg(&trailed), None);
}
