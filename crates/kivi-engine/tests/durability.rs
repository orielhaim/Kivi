//! Durable engine tests: embedded write → shutdown → reopen → read back.
//!
//! No processes or sockets here — these prove the engine/durability
//! contract (WAL persist, recovery replay, identity continuity) through
//! the public `LocalEngine` API. Real crash/kill coverage lives in
//! `kivi-server/tests/durability.rs`.

use kivi_engine::{DurabilityMode, DurableConfig, EngineConfig, LocalEngine, Placement};
use kivi_state::Key;
use kivi_tablet::{DirectorySnapshot, HashPrefix, PartitionRange};
use kivi_types::{
    ClusterId, NamespaceId, NodeId, NodeIncarnation, TabletEpoch, TabletId, WorkerId,
    WriteGuardGeneration,
};

const NS: NamespaceId = NamespaceId::from_u64(1);

fn directory() -> DirectorySnapshot {
    let genesis = DirectorySnapshot::bootstrap(
        NS,
        TabletId::from_u64(1),
        PartitionRange::Hash(HashPrefix::new(0, 0).expect("root")),
        TabletEpoch::INITIAL,
        WriteGuardGeneration::INITIAL,
    )
    .expect("genesis");
    genesis
        .stage(TabletId::from_u64(1))
        .and_then(|snapshot| snapshot.activate(TabletId::from_u64(1)))
        .expect("active root")
}

fn placement() -> Placement {
    Placement::new([(TabletId::from_u64(1), WorkerId::from_u64(0))])
}

fn durable_config(dir: &std::path::Path) -> (DurabilityMode, NodeId, ClusterId, NodeIncarnation) {
    let opened = kivi_durability::open_data_dir(dir).expect("data dir opens");
    let mode = DurabilityMode::Durable(DurableConfig {
        data_dir: dir.to_owned(),
        segment_target_bytes: 1024 * 1024,
        node: opened.meta.node,
        cluster: opened.meta.cluster,
        incarnation: opened.meta.incarnation,
        shared_wal: false,
    });
    (
        mode,
        opened.meta.node,
        opened.meta.cluster,
        opened.meta.incarnation,
    )
}

fn start(dir: &std::path::Path) -> (LocalEngine, NodeIncarnation) {
    let (durability, _, _, incarnation) = durable_config(dir);
    let engine = LocalEngine::start(EngineConfig {
        namespace: NS,
        directory: directory(),
        placement: placement(),
        worker_count: 2,
        request_capacity: 64,
        network: None,
        durability,
    })
    .expect("durable engine starts");
    (engine, incarnation)
}

#[test]
fn embedded_write_restart_read_back_exact() {
    let scratch = tempfile::tempdir().expect("scratch");
    let (engine, first_incarnation) = start(scratch.path());
    let client = engine.client();
    client
        .set(&Key::from("a"), bytes::Bytes::from_static(b"1"))
        .expect("set");
    assert_eq!(client.counter_add(&Key::from("n"), 41).expect("add"), 41);
    engine.shutdown().expect("clean shutdown");
    // Reopen the same directory: acknowledged state must come back, the
    // incarnation must advance, and new writes must continue the log.
    let (engine, second_incarnation) = start(scratch.path());
    assert!(second_incarnation.supersedes(first_incarnation));
    let client = engine.client();
    assert_eq!(
        client.get(&Key::from("a")).expect("get"),
        Some(bytes::Bytes::from_static(b"1"))
    );
    assert_eq!(client.counter_get(&Key::from("n")).expect("read"), Some(41));
    assert_eq!(client.counter_add(&Key::from("n"), 1).expect("add"), 42);
    let durability = engine
        .admin_handle()
        .durability()
        .expect("durable admin info")
        .clone();
    assert_eq!(durability.recovery.records_replayed, 2);
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn recovery_rejects_forgotten_tablet() {
    let scratch = tempfile::tempdir().expect("scratch");
    let (engine, _) = start(scratch.path());
    engine
        .client()
        .set(&Key::from("a"), bytes::Bytes::from_static(b"1"))
        .expect("set");
    engine.shutdown().expect("clean shutdown");
    // Reopen with tablet 1 sealed (Active->Fenced, unwritable): its WAL
    // history has nowhere to replay, so startup must fail rather than
    // discard it.
    let sealed = directory().seal(TabletId::from_u64(1)).expect("seal");
    let (durability, _, _, _) = durable_config(scratch.path());
    let err = LocalEngine::start(EngineConfig {
        namespace: NS,
        directory: sealed,
        placement: placement(),
        worker_count: 2,
        request_capacity: 64,
        network: None,
        durability,
    })
    .expect_err("forgotten tablet fails startup");
    let message = format!("{err:?}");
    assert!(
        message.contains("UnknownTablet") || message.contains("unknown tablet"),
        "clear error, got {message}"
    );
}
