//! Memory Fabric engine integration tests: medium values stay logically
//! exact through staging, slicing, arena pressure with `NVMe` demotion,
//! cross-tablet batches, checkpoints, and durable restart.
//!
//! All values here exceed [`kivi_engine::FABRIC_INLINE_MAX`], so they ride
//! the fabric path; assertions compare full logical bytes through the
//! public [`kivi_engine::LocalClient`] API only.

use std::path::Path;
use std::time::{Duration, Instant};

use bytes::Bytes;
use kivi_engine::{
    BatchPolicy, CheckpointConfig, DurabilityMode, DurableConfig, EngineConfig, EngineError,
    FabricConfig, FabricPaths, LocalClient, LocalEngine, Placement,
};
use kivi_state::{Key, ObjectVersion, TxnExpect, TxnWrite, TxnWriteKind};
use kivi_tablet::{DirectorySnapshot, HashPrefix, PartitionRange};
use kivi_types::{NamespaceId, TabletEpoch, TabletId, WorkerId, WriteGuardGeneration};

const NS: NamespaceId = NamespaceId::from_u64(1);
const TABLET: TabletId = TabletId::from_u64(1);
const EPOCH: TabletEpoch = TabletEpoch::INITIAL;
const GUARD: WriteGuardGeneration = WriteGuardGeneration::INITIAL;

fn hash_range(bits: u128, len: u8) -> PartitionRange {
    PartitionRange::Hash(HashPrefix::new(bits, len).expect("range"))
}

fn activate(snapshot: &DirectorySnapshot, tablet: TabletId) -> DirectorySnapshot {
    snapshot
        .clone()
        .stage(tablet)
        .and_then(|staged| staged.activate(tablet))
        .expect("stage+activate")
}

fn root_snapshot() -> DirectorySnapshot {
    let genesis =
        DirectorySnapshot::bootstrap(NS, TABLET, hash_range(0, 0), EPOCH, GUARD).expect("genesis");
    activate(&genesis, TABLET)
}

fn split_snapshot() -> DirectorySnapshot {
    let mut dir = root_snapshot();
    dir = dir
        .allocate(TabletId::from_u64(2), hash_range(0, 1), EPOCH, GUARD)
        .and_then(|staged| staged.stage(TabletId::from_u64(2)))
        .expect("child 0*");
    dir = dir
        .allocate(
            TabletId::from_u64(3),
            hash_range(1u128 << 127, 1),
            EPOCH,
            GUARD,
        )
        .and_then(|staged| staged.stage(TabletId::from_u64(3)))
        .expect("child 1*");
    dir = dir.seal(TABLET).expect("seal");
    dir = dir.activate(TabletId::from_u64(2)).expect("activate 2");
    dir = dir.activate(TabletId::from_u64(3)).expect("activate 3");
    dir.retire(
        TABLET,
        kivi_tablet::Redirect::new(vec![TabletId::from_u64(2), TabletId::from_u64(3)]),
    )
    .expect("retire")
}

fn worker(index: u64) -> WorkerId {
    WorkerId::from_u64(index)
}

/// Deterministic constant-fill pattern (`(seed % 251)` repeated).
fn fill_pattern(seed: u64, len: usize) -> Bytes {
    let byte = u8::try_from(seed % 251).expect("mod 251 fits in u8");
    Bytes::from(vec![byte; len])
}

/// Deterministic position-varying pattern, so range windows are meaningful.
fn slice_pattern(seed: u64, len: usize) -> Bytes {
    let mut out = Vec::with_capacity(len);
    for index in 0..len {
        let position = u64::try_from(index).expect("index fits in u64");
        out.push(u8::try_from((seed + position) % 251).expect("mod 251 fits in u8"));
    }
    Bytes::from(out)
}

/// Deterministic incompressible bytes (fixed-seed xorshift): pressure must
/// demote these to `NVMe`, since there is nothing to compress away.
fn incompressible_pattern(seed: u64, len: usize) -> Bytes {
    let mut state = if seed == 0 { 1 } else { seed };
    let mut out = vec![0u8; len];
    for byte in &mut out {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = u8::try_from((state >> 11) & 0xFF).expect("masked to one byte");
    }
    Bytes::from(out)
}

/// Small arena plus roomy demotion lane, forcing background movement.
fn pressured_fabric() -> FabricConfig {
    FabricConfig {
        arena_bytes_per_worker: 65536,
        demotion_bytes_per_worker: 1 << 20,
    }
}

fn start_ephemeral(
    directory: DirectorySnapshot,
    placement: Placement,
    worker_count: usize,
    fabric: FabricConfig,
) -> LocalEngine {
    LocalEngine::start(EngineConfig {
        namespace: NS,
        directory,
        placement,
        worker_count,
        request_capacity: 256,
        chunks: kivi_engine::ChunkFabricConfig::default(),
        fabric,
        network: None,
        durability: DurabilityMode::Ephemeral,
    })
    .expect("ephemeral engine starts")
}

fn start_single_ephemeral(fabric: FabricConfig) -> LocalEngine {
    start_ephemeral(
        root_snapshot(),
        Placement::new([(TABLET, worker(0))]),
        1,
        fabric,
    )
}

/// Durable config with automatic checkpoints stilled: tests trigger
/// manually, so cuts stay deterministic.
fn durable(dir: &Path, fabric: FabricConfig) -> EngineConfig {
    let opened = kivi_durability::open_data_dir(dir).expect("data dir opens");
    EngineConfig {
        namespace: NS,
        directory: root_snapshot(),
        placement: Placement::new([(TABLET, worker(0))]),
        worker_count: 2,
        request_capacity: 256,
        chunks: kivi_engine::ChunkFabricConfig::default(),
        fabric,
        network: None,
        durability: DurabilityMode::Durable(DurableConfig {
            data_dir: dir.to_owned(),
            segment_target_bytes: 1024 * 1024,
            node: opened.meta.node,
            cluster: opened.meta.cluster,
            incarnation: opened.meta.incarnation,
            shared_wal: false,
            batch: BatchPolicy::default_policy(),
            checkpoint: CheckpointConfig {
                disabled: true,
                ..CheckpointConfig::default_config()
            },
        }),
    }
}

/// Waits for an installed CURRENT at `cut` (background build plus publish
/// are asynchronous), sharing the test's overall deadline.
fn await_current(dir: &Path, cut: u64, deadline: Instant) {
    loop {
        if let Ok(Some(current)) = kivi_checkpoint::read_current(dir, TABLET)
            && current.cut >= cut
        {
            return;
        }
        assert!(
            Instant::now() <= deadline,
            "no CURRENT at cut {cut} before deadline"
        );
        std::thread::sleep(Duration::from_millis(25));
    }
}

fn is_backpressure(error: &EngineError) -> bool {
    matches!(
        error,
        EngineError::Overloaded | EngineError::SessionOverloaded
    )
}

fn set_retry(client: &LocalClient, key: &Key, value: &Bytes, deadline: Instant) {
    loop {
        match client.set(key, value.clone()) {
            Ok(()) => return,
            Err(error) if is_backpressure(&error) => {
                assert!(
                    Instant::now() <= deadline,
                    "set {key:?} backpressured past deadline: {error:?}"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(error) => panic!("set {key:?} failed: {error:?}"),
        }
    }
}

fn get_retry(client: &LocalClient, key: &Key, deadline: Instant) -> Option<Bytes> {
    loop {
        match client.get(key) {
            Ok(value) => return value,
            Err(error) if is_backpressure(&error) => {
                assert!(
                    Instant::now() <= deadline,
                    "get {key:?} backpressured past deadline: {error:?}"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(error) => panic!("get {key:?} failed: {error:?}"),
        }
    }
}

fn clamped_slice(value: &Bytes, offset: u64, len: u64) -> Bytes {
    let total = u64::try_from(value.len()).expect("length fits in u64");
    let start = offset.min(total);
    let end = offset.saturating_add(len).min(total);
    let from = usize::try_from(start).expect("offset fits in usize");
    let until = usize::try_from(end).expect("offset fits in usize");
    value.slice(from..until)
}

#[test]
fn medium_value_round_trips_ephemeral() {
    let engine = start_single_ephemeral(FabricConfig::default());
    let client = engine.client();
    let key = Key::from("fabric:medium:1k");
    let value = fill_pattern(9, 1024);
    client.set(&key, value.clone()).expect("set");
    assert_eq!(client.get(&key).expect("get"), Some(value), "1KB exact");
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn get_range_slices_fabric_value() {
    let engine = start_single_ephemeral(FabricConfig::default());
    let client = engine.client();
    let key = Key::from("fabric:range:4k");
    let value = slice_pattern(5, 4096);
    client.set(&key, value.clone()).expect("set");
    for (offset, len) in [
        (0u64, 512u64),
        (1000, 2000),
        (4000, 512),
        (4096, 64),
        (5000, 100),
        (123, 0),
        (0, 4096),
    ] {
        assert_eq!(
            client.get_range(&key, offset, len).expect("range"),
            Some(clamped_slice(&value, offset, len)),
            "window [{offset}, {len})"
        );
    }
    engine.shutdown().expect("clean shutdown");
}

fn write_pressure_keys(client: &LocalClient, deadline: Instant) -> Vec<(Key, Bytes)> {
    let mut expected = Vec::with_capacity(20);
    for index in 0..20u64 {
        let key = Key::from(format!("fabric:pressure:{index:02}"));
        let value = incompressible_pattern(1000 + index, 4096);
        set_retry(client, &key, &value, deadline);
        expected.push((key, value));
    }
    expected
}

fn drive_until_demoted(
    engine: &LocalEngine,
    client: &LocalClient,
    expected: &[(Key, Bytes)],
    deadline: Instant,
) -> bool {
    let admin = engine.admin_handle();
    let scratch = Key::from("fabric:pressure:churn");
    let mut round = 0u64;
    while Instant::now() <= deadline {
        for (key, value) in expected {
            match client.get(key) {
                Ok(seen) => assert_eq!(seen.as_ref(), Some(value), "byte-exact while pressured"),
                Err(error) if is_backpressure(&error) => {}
                Err(error) => panic!("get {key:?} failed: {error:?}"),
            }
        }
        // Best-effort churn: under full pressure promotions refill the
        // arena as fast as demotions drain it, so a traffic set may
        // backpressure indefinitely; that only means keep polling.
        match client.set(&scratch, incompressible_pattern(round, 1024)) {
            Ok(()) => {}
            Err(error) if is_backpressure(&error) => {}
            Err(error) => panic!("churn set failed: {error:?}"),
        }
        round += 1;
        let stats = admin.fabric_stats_snapshot();
        if stats.iter().any(|(_, report)| report.fabric.nvme_bytes > 0) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    false
}

fn verify_pressure_keys(client: &LocalClient, expected: &[(Key, Bytes)], deadline: Instant) {
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let fresh = Key::from("fabric:pressure:fresh");
            let value = fill_pattern(7, 1024);
            set_retry(client, &fresh, &value, deadline);
            assert_eq!(
                get_retry(client, &fresh, deadline),
                Some(value),
                "fresh key exact during reads"
            );
        });
        for (key, value) in expected {
            assert_eq!(
                get_retry(client, key, deadline),
                Some(value.clone()),
                "post-demotion exact"
            );
        }
    });
}

#[test]
fn pressure_demotes_and_resume_is_exact() {
    let engine = start_ephemeral(
        root_snapshot(),
        Placement::new([(TABLET, worker(0))]),
        2,
        pressured_fabric(),
    );
    let client = engine.client();
    let deadline = Instant::now() + Duration::from_secs(30);
    let expected = write_pressure_keys(&client, deadline);
    if !drive_until_demoted(&engine, &client, &expected, deadline) {
        let dump = format!("{:#?}", engine.admin_handle().fabric_stats_snapshot());
        engine.shutdown().expect("clean shutdown");
        panic!("no worker reported nvme_bytes>0 within 30s; stats: {dump}");
    }
    verify_pressure_keys(&client, &expected, deadline);
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn small_set_range_restages_after_offcore_fabric_demotion() {
    let engine = start_ephemeral(
        root_snapshot(),
        Placement::new([(TABLET, worker(0))]),
        2,
        pressured_fabric(),
    );
    let client = engine.client();
    let deadline = Instant::now() + Duration::from_secs(30);
    let expected = write_pressure_keys(&client, deadline);
    if !drive_until_demoted(&engine, &client, &expected, deadline) {
        let dump = format!("{:#?}", engine.admin_handle().fabric_stats_snapshot());
        let _ = engine.shutdown();
        panic!("no worker reported nvme_bytes>0 within 30s; stats: {dump}");
    }
    let (key, base) = &expected[0];
    let patch = Bytes::from_static(b"patch");
    let mut patched = base.to_vec();
    patched[..patch.len()].copy_from_slice(&patch);
    loop {
        match client.set_range(key, 0, patch.clone()) {
            Ok(()) => break,
            Err(error) if is_backpressure(&error) => {
                assert!(
                    Instant::now() <= deadline,
                    "set-range backpressured past deadline: {error:?}"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(error) => panic!("set-range failed: {error:?}"),
        }
    }
    assert_eq!(
        get_retry(&client, key, deadline),
        Some(Bytes::from(patched))
    );
    engine.shutdown().expect("clean shutdown");
}

fn probe_one_key_per_tablet(engine: &LocalEngine) -> Vec<(TabletId, Key)> {
    let mut probes = Vec::new();
    for index in 0..1000u64 {
        let key = Key::from(format!("fabric:batch:{index}"));
        let (tablet, _) = engine.route_key(&key).expect("covered");
        if !probes.iter().any(|(known, _)| *known == tablet) {
            probes.push((tablet, key));
        }
        if probes.len() == 2 {
            break;
        }
    }
    probes
}

#[test]
fn cross_tablet_batch_with_medium_values() {
    let engine = start_ephemeral(
        split_snapshot(),
        Placement::new([
            (TabletId::from_u64(2), worker(0)),
            (TabletId::from_u64(3), worker(1)),
        ]),
        2,
        FabricConfig::default(),
    );
    let client = engine.client();
    let probes = probe_one_key_per_tablet(&engine);
    assert_eq!(probes.len(), 2, "both tablets must be reachable");
    assert_ne!(probes[0].0, probes[1].0, "probes span tablets");
    let alpha = fill_pattern(11, 1024);
    let beta = fill_pattern(22, 1024);
    let writes = [
        TxnWrite {
            key: probes[0].1.clone(),
            kind: TxnWriteKind::Put(alpha.clone()),
            expect: TxnExpect::Any,
        },
        TxnWrite {
            key: probes[1].1.clone(),
            kind: TxnWriteKind::Put(beta.clone()),
            expect: TxnExpect::Any,
        },
    ];
    let versions = client.atomic_batch(&writes).expect("batch commits");
    assert_eq!(versions.len(), 2, "one version per write");
    assert!(versions.iter().all(Option::is_some), "both puts versioned");
    assert_eq!(
        client.get(&probes[0].1).expect("get"),
        Some(alpha.clone()),
        "tablet key one exact"
    );
    assert_eq!(
        client.get(&probes[1].1).expect("get"),
        Some(beta.clone()),
        "tablet key two exact"
    );
    let live_alpha = client
        .get_version(&probes[0].1)
        .expect("version")
        .expect("present");
    let live_beta = client
        .get_version(&probes[1].1)
        .expect("version")
        .expect("present");
    let conflict = [
        TxnWrite {
            key: probes[0].1.clone(),
            kind: TxnWriteKind::Put(fill_pattern(99, 1024)),
            expect: TxnExpect::Version(ObjectVersion::from_u64(live_alpha.as_u64() + 1000)),
        },
        TxnWrite {
            key: probes[1].1.clone(),
            kind: TxnWriteKind::Put(fill_pattern(98, 1024)),
            expect: TxnExpect::Version(ObjectVersion::from_u64(live_beta.as_u64() + 1000)),
        },
    ];
    assert!(
        client.atomic_batch(&conflict).is_err(),
        "wrong-version batch must fail"
    );
    assert_eq!(
        client.get(&probes[0].1).expect("get"),
        Some(alpha),
        "failed batch changes nothing"
    );
    assert_eq!(
        client.get(&probes[1].1).expect("get"),
        Some(beta),
        "failed batch changes nothing"
    );
    engine.shutdown().expect("clean shutdown");
}

fn flip_worker_zero_material(data_dir: &Path) {
    let path = FabricPaths::worker(data_dir, 0)
        .material_dir()
        .join("materializations.dat");
    let mut raw = std::fs::read(&path).expect("worker-0 material file present");
    assert!(raw.len() > 64, "material file holds sealed records");
    let flip_at = raw.len() - 8;
    raw[flip_at] ^= 0xFF;
    std::fs::write(&path, raw).expect("clobber material record");
}

#[test]
fn durable_supersede_checkpoint_restart() {
    let scratch = tempfile::tempdir().expect("scratch");
    let deadline = Instant::now() + Duration::from_secs(30);
    let key = Key::from("fabric:durable:super");
    let first = fill_pattern(3, 2048);
    let second = fill_pattern(4, 2048);
    let engine = LocalEngine::start(durable(scratch.path(), FabricConfig::default()))
        .expect("durable engine starts");
    let client = engine.client();
    client.set(&key, first).expect("set v1");
    client.set(&key, second.clone()).expect("set v2");
    engine.request_checkpoint(None);
    await_current(scratch.path(), 2, deadline);
    engine.shutdown().expect("clean shutdown");
    let engine = LocalEngine::start(durable(scratch.path(), FabricConfig::default()))
        .expect("restart serves");
    assert_eq!(
        engine.client().get(&key).expect("get"),
        Some(second.clone()),
        "v2 survives checkpoint plus restart"
    );
    engine.shutdown().expect("clean shutdown");
    let engine = LocalEngine::start(durable(scratch.path(), FabricConfig::default()))
        .expect("reopen serves");
    assert_eq!(
        engine.client().get(&key).expect("get"),
        Some(second),
        "v2 survives reopen"
    );
    flip_worker_zero_material(scratch.path());
    engine.shutdown().expect("clean shutdown");
    match LocalEngine::start(durable(scratch.path(), FabricConfig::default())) {
        Ok(running) => {
            let _ = running.shutdown();
            panic!("restart on corrupt fabric material must fail, never serve absence");
        }
        Err(error) => {
            let detail = format!("{error:?}");
            assert!(
                detail.contains("Fabric")
                    || detail.contains("Corrupt")
                    || detail.contains("corrupt"),
                "loud fabric failure, got {detail}"
            );
        }
    }
}

/// Durable config over the split directory (two tablets, two workers).
fn durable_split(dir: &Path, fabric: FabricConfig) -> EngineConfig {
    let opened = kivi_durability::open_data_dir(dir).expect("data dir opens");
    EngineConfig {
        namespace: NS,
        directory: split_snapshot(),
        placement: Placement::new([
            (TabletId::from_u64(2), worker(0)),
            (TabletId::from_u64(3), worker(1)),
        ]),
        worker_count: 2,
        request_capacity: 256,
        chunks: kivi_engine::ChunkFabricConfig::default(),
        fabric,
        network: None,
        durability: DurabilityMode::Durable(DurableConfig {
            data_dir: dir.to_owned(),
            segment_target_bytes: 1024 * 1024,
            node: opened.meta.node,
            cluster: opened.meta.cluster,
            incarnation: opened.meta.incarnation,
            shared_wal: false,
            batch: BatchPolicy::default_policy(),
            checkpoint: CheckpointConfig {
                disabled: true,
                ..CheckpointConfig::default_config()
            },
        }),
    }
}

#[test]
fn two_pc_medium_batch_survives_restart() {
    let scratch = tempfile::tempdir().expect("scratch");
    let engine = LocalEngine::start(durable_split(scratch.path(), FabricConfig::default()))
        .expect("durable split engine starts");
    let probes = probe_one_key_per_tablet(&engine);
    assert_eq!(probes.len(), 2, "both tablets must be reachable");
    let alpha = fill_pattern(31, 2048);
    let beta = fill_pattern(32, 2048);
    // Cross-tablet batch drives prepare (seal plus pin) and finalize
    // (re-seal plus unpin) on both participants.
    let versions = engine
        .client()
        .atomic_batch(&[
            TxnWrite {
                key: probes[0].1.clone(),
                kind: TxnWriteKind::Put(alpha.clone()),
                expect: TxnExpect::Any,
            },
            TxnWrite {
                key: probes[1].1.clone(),
                kind: TxnWriteKind::Put(beta.clone()),
                expect: TxnExpect::Any,
            },
        ])
        .expect("2PC batch commits");
    assert!(versions.iter().all(Option::is_some), "both versioned");
    engine.shutdown().expect("clean shutdown");
    // Restart replays prepare/finalize records and re-links the
    // re-sealed journal entries (prepare staging records superseded).
    let engine = LocalEngine::start(durable_split(scratch.path(), FabricConfig::default()))
        .expect("restart serves");
    let client = engine.client();
    assert_eq!(client.get(&probes[0].1).expect("get"), Some(alpha));
    assert_eq!(client.get(&probes[1].1).expect("get"), Some(beta));
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn checkpoint_during_movement_stays_exact() {
    let scratch = tempfile::tempdir().expect("scratch");
    let deadline = Instant::now() + Duration::from_secs(30);
    let engine =
        LocalEngine::start(durable(scratch.path(), pressured_fabric())).expect("engine starts");
    let client = engine.client();
    let mut expected = Vec::with_capacity(20);
    let mut applied = 0u64;
    for index in 0..20u64 {
        let key = Key::from(format!("fabric:moving:{index:02}"));
        let value = incompressible_pattern(5000 + index, 4096);
        set_retry(&client, &key, &value, deadline);
        applied += 1;
        expected.push((key, value));
        if index % 4 == 3 {
            engine.request_checkpoint(None);
        }
    }
    engine.request_checkpoint(None);
    await_current(scratch.path(), applied, deadline);
    engine.shutdown().expect("clean shutdown");
    let engine =
        LocalEngine::start(durable(scratch.path(), pressured_fabric())).expect("restart serves");
    let client = engine.client();
    for (key, value) in &expected {
        assert_eq!(
            client.get(key).expect("get"),
            Some(value.clone()),
            "movement plus checkpoints stay exact"
        );
    }
    engine.shutdown().expect("clean shutdown");
}
