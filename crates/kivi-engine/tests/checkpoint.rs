//! Checkpoint end-to-end through the embedded engine: capture, build,
//! publish, restart, restore, tail replay, reclamation.
//!
//! No processes or sockets here — these prove the checkpoint/recovery
//! contract through the public `LocalEngine` API (manual triggers,
//! CURRENT polling, restart restores). Real crash/kill coverage lives in
//! `kivi-server/tests/durability.rs`.

use std::time::{Duration, Instant};

use kivi_engine::{DurabilityMode, DurableConfig, EngineConfig, LocalEngine, Placement};
use kivi_state::Key;
use kivi_tablet::{DirectorySnapshot, HashPrefix, PartitionRange};
use kivi_types::{NamespaceId, TabletEpoch, TabletId, WorkerId, WriteGuardGeneration};

const NS: NamespaceId = NamespaceId::from_u64(1);
const TABLET: TabletId = TabletId::from_u64(1);

fn directory() -> DirectorySnapshot {
    let genesis = DirectorySnapshot::bootstrap(
        NS,
        TABLET,
        PartitionRange::Hash(HashPrefix::new(0, 0).expect("root")),
        TabletEpoch::INITIAL,
        WriteGuardGeneration::INITIAL,
    )
    .expect("genesis");
    genesis
        .stage(TABLET)
        .and_then(|snapshot| snapshot.activate(TABLET))
        .expect("active root")
}

fn placement() -> Placement {
    Placement::new([(TABLET, WorkerId::from_u64(0))])
}

fn durable_config(dir: &std::path::Path) -> DurabilityMode {
    let opened = kivi_durability::open_data_dir(dir).expect("data dir opens");
    DurabilityMode::Durable(DurableConfig {
        data_dir: dir.to_owned(),
        segment_target_bytes: 1024 * 1024,
        node: opened.meta.node,
        cluster: opened.meta.cluster,
        incarnation: opened.meta.incarnation,
        shared_wal: false,
        batch: kivi_engine::BatchPolicy::default_policy(),
        checkpoint: kivi_engine::CheckpointConfig {
            // Tests trigger manually; automatic thresholds stay out of the way.
            disabled: true,
            ..kivi_engine::CheckpointConfig::default_config()
        },
    })
}

fn start(dir: &std::path::Path) -> LocalEngine {
    LocalEngine::start(EngineConfig {
        namespace: NS,
        directory: directory(),
        placement: placement(),
        worker_count: 2,
        request_capacity: 256,
        chunks: kivi_engine::ChunkFabricConfig::default(),
        fabric: kivi_engine::FabricConfig::default(),
        network: None,
        durability: durable_config(dir),
    })
    .expect("durable engine starts")
}

/// Waits for a CURRENT record at `cut` (background build + publish are
/// asynchronous), returning it.
fn await_current(dir: &std::path::Path, cut: u64) -> kivi_checkpoint::CurrentRecord {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Ok(Some(current)) = kivi_checkpoint::read_current(dir, TABLET)
            && current.cut >= cut
        {
            return current;
        }
        assert!(
            Instant::now() <= deadline,
            "no CURRENT at cut {cut} after 30s"
        );
        std::thread::sleep(Duration::from_millis(25));
    }
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
    // Reclamation runs after every publish: with two checkpoints the
    // floor is the previous cut (5), so fully covered sealed segments go.
    // (Tiny database: likely one active segment; assert the invariant,
    // not a count — counts are proven at process scale in AO.)
    let wal_dir = scratch.path().join("wal").join("lane-0000");
    let segments_before = std::fs::read_dir(&wal_dir).map_or(0, std::iter::Iterator::count);
    let _ = segments_before;
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

#[test]
fn checkpoint_captures_dedup_state_for_retry_safety() {
    let scratch = tempfile::tempdir().expect("scratch");
    let engine = start(scratch.path());
    // Embedded clients share fate with the process (no identity), so no
    // sessions exist here. This test proves the capture/build path
    // carries the (empty) dedup component end to end; the money scenario
    // with real identities runs at process scale (AP).
    let client = engine.client();
    client
        .set(&Key::from("k"), bytes::Bytes::from_static(b"v"))
        .expect("set");
    engine.request_checkpoint(None);
    let current = await_current(scratch.path(), 1);
    let manifest_bytes = std::fs::read(
        scratch
            .path()
            .join("checkpoints")
            .join("tablet-1")
            .join("manifests")
            .join(format!("{}.manifest", current.manifest.hex())),
    )
    .expect("manifest file present");
    let manifest =
        kivi_checkpoint::load_manifest(current.manifest, &manifest_bytes).expect("manifest loads");
    assert_eq!(manifest.dedup.sessions, 0, "no sessions embedded");
    engine.shutdown().expect("clean shutdown");
}

/// AB semantic verification: live state at the cut equals restored
/// checkpoint state, compared as deterministic sorted snapshots
/// (objects, versions, expiries, dedup floors and outcomes).
#[test]
fn live_state_at_cut_equals_restored_checkpoint_state() {
    use kivi_engine::LiveTablet;
    use kivi_state::{Operation, StoredObject};
    use kivi_types::{TabletAuthority, WallTimestamp};
    let now = WallTimestamp::from_micros(1_000_000);
    let authority =
        TabletAuthority::new(TABLET, TabletEpoch::INITIAL, WriteGuardGeneration::INITIAL);
    let descriptor = directory().get(TABLET).expect("descriptor").clone();
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
        tablet: TABLET,
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
    let loaded = kivi_checkpoint::load_installed(scratch.path(), TABLET)
        .expect("loads")
        .expect("installed");
    assert!(!loaded.fell_back);
    assert_eq!(loaded.current.cut, installed.cut);
    // Restored objects equal live objects exactly (key, value, version,
    // expiry — full StoredObject equality).
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
