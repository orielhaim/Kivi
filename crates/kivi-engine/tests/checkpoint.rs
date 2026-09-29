//! Checkpoint end-to-end through the embedded engine: capture, build,
//! publish, restart, restore, tail replay, reclamation.
//!
//! No processes or sockets here - these prove the checkpoint/recovery
//! contract through the public `LocalEngine` API (manual triggers,
//! CURRENT polling, restart restores). Real crash/kill coverage lives in
//! `kivi-lab/tests/durability.rs`.

mod harness;

use harness::{NS, ROOT, await_current, durable_config, root_snapshot};
use kivi_engine::LocalEngine;
use kivi_state::Key;

fn start(dir: &std::path::Path) -> LocalEngine {
    LocalEngine::start(durable_config(
        dir,
        kivi_engine::ChunkFabricConfig::default(),
        kivi_engine::FabricConfig::default(),
    ))
    .expect("durable engine starts")
}

#[test]
fn checkpoint_restart_restores_state_and_skips_covered_tail() {
    let scratch = tempfile::tempdir().expect("scratch");
    let engine = start(scratch.path());
    let client = engine.client();
    for index in 0..10u32 {
        client
            .set(
                &Key::from(format!("k:{index:03}")),
                bytes::Bytes::from(format!("v{index}")),
            )
            .expect("set");
    }
    assert_eq!(client.counter_add(&Key::from("n"), 41).expect("add"), 41);
    // Trigger a checkpoint covering all 11 mutations and wait for install.
    engine.request_checkpoint(None);
    let current = await_current(scratch.path(), 11);
    assert_eq!(current.cut, 11);
    assert!(
        current.previous.is_none(),
        "first checkpoint has no previous"
    );
    // More writes after the cut (new WAL tail).
    client
        .set(&Key::from("after"), bytes::Bytes::from_static(b"yes"))
        .expect("set");
    engine.shutdown().expect("clean shutdown");
    // Restart: checkpoint restores 11, WAL tail replays 1.
    let engine = start(scratch.path());
    let client = engine.client();
    for index in 0..10u32 {
        assert_eq!(
            client
                .get(&Key::from(format!("k:{index:03}")))
                .expect("get"),
            Some(bytes::Bytes::from(format!("v{index}"))),
            "checkpointed key survives"
        );
    }
    assert_eq!(client.counter_get(&Key::from("n")).expect("read"), Some(41));
    assert_eq!(
        client.get(&Key::from("after")).expect("get"),
        Some(bytes::Bytes::from_static(b"yes")),
        "post-cut tail replays"
    );
    let recovery = engine
        .admin_handle()
        .durability()
        .expect("durable admin info")
        .clone();
    assert_eq!(
        recovery.recovery.records_replayed, 1,
        "only the tail replays"
    );
    assert_eq!(
        recovery.checkpoint_recovery.tablets_loaded, 1,
        "one tablet restored from checkpoint"
    );
    assert_eq!(
        recovery.checkpoint_recovery.obsolete_skipped, 11,
        "covered records skipped, never replayed"
    );
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn second_checkpoint_reclaims_old_segments_and_keeps_serving() {
    let scratch = tempfile::tempdir().expect("scratch");
    let engine = start(scratch.path());
    let client = engine.client();
    for index in 0..5u32 {
        client
            .set(
                &Key::from(format!("k:{index}")),
                bytes::Bytes::from_static(b"v"),
            )
            .expect("set");
    }
    engine.request_checkpoint(None);
    let first = await_current(scratch.path(), 5);
    for index in 5..10u32 {
        client
            .set(
                &Key::from(format!("k:{index}")),
                bytes::Bytes::from_static(b"v"),
            )
            .expect("set");
    }
    engine.request_checkpoint(None);
    let second = await_current(scratch.path(), 10);
    assert_eq!(second.cut, 10);
    assert_eq!(
        second.previous,
        Some(first.manifest),
        "retention chain links current to previous"
    );
    engine.shutdown().expect("clean shutdown");
    // Restart works on the reclaimed layout with all state intact.
    let engine = start(scratch.path());
    let client = engine.client();
    for index in 0..10u32 {
        assert_eq!(
            client.get(&Key::from(format!("k:{index}"))).expect("get"),
            Some(bytes::Bytes::from_static(b"v")),
            "all keys survive two checkpoints plus reclaim"
        );
    }
    engine.shutdown().expect("clean shutdown");
}

/// Live state at the cut must equal the restored checkpoint state exactly,
/// across objects, versions, expiries and dedup.
#[test]
fn live_state_at_cut_equals_restored_checkpoint_state() {
    use kivi_engine::LiveTablet;
    use kivi_state::{Operation, StoredObject};
    use kivi_types::{TabletAuthority, TabletEpoch, WallTimestamp, WriteGuardGeneration};
    let now = WallTimestamp::from_micros(1_000_000);
    let authority = TabletAuthority::new(ROOT, TabletEpoch::INITIAL, WriteGuardGeneration::INITIAL);
    let descriptor = root_snapshot().get(ROOT).expect("descriptor").clone();
    let mut live = LiveTablet::from_descriptor(&descriptor, authority).expect("live");
    // Mixed workload: bytes, counters, expiries, deletes.
    for (key, value) in [("a", "1"), ("b", "22"), ("c", "333")] {
        live.execute(
            &Operation::Set {
                key: Key::from(key),
                value: bytes::Bytes::from(value),
            },
            now,
        )
        .expect("set");
    }
    live.execute(
        &Operation::CounterAdd {
            key: Key::from("n"),
            delta: 7,
        },
        now,
    )
    .expect("add");
    live.execute(
        &Operation::ExpireAt {
            key: Key::from("a"),
            expires_at: WallTimestamp::from_micros(9_000_000),
        },
        now,
    )
    .expect("expire");
    live.execute(
        &Operation::Delete {
            key: Key::from("b"),
        },
        now,
    )
    .expect("delete");
    let live_snapshot: Vec<(Key, StoredObject)> = live.store().snapshot_sorted();
    assert_eq!(live_snapshot.len(), 3, "a, c, n remain");
    // Capture + build + publish + load through the real artifacts.
    let scratch = tempfile::tempdir().expect("scratch");
    let view = live.capture_view(NS);
    let identity = kivi_checkpoint::CheckpointIdentity {
        cluster: kivi_types::ClusterId::from_u128(1),
        node: kivi_types::NodeId::from_u64(2),
        incarnation: kivi_types::NodeIncarnation::INITIAL,
        namespace: NS,
        tablet: ROOT,
    };
    let built = kivi_checkpoint::build_tablet(
        &view,
        None,
        &kivi_checkpoint::CheckpointPolicy::DEFAULT,
        &identity,
    )
    .expect("builds");
    let (installed, _) =
        kivi_checkpoint::publish_tablet(scratch.path(), &built, None).expect("publishes");
    let loaded = kivi_checkpoint::load_installed(scratch.path(), ROOT)
        .expect("loads")
        .expect("installed");
    assert!(!loaded.fell_back);
    assert_eq!(loaded.current.cut, installed.cut);
    // Restored objects equal live objects exactly (key, value, version,
    // expiry - full StoredObject equality).
    let mut restored: Vec<(Key, StoredObject)> = loaded
        .bands
        .iter()
        .flat_map(|band| {
            band.records
                .iter()
                .map(|record| (record.key.clone(), record.object.clone()))
        })
        .collect();
    restored.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(restored, live_snapshot, "checkpoint restores exact state");
}
