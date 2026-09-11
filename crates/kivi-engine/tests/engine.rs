//! Engine integration tests: bootstrap, typed CRUD, routing, replacement,
//! shutdown, and failure surfacing over real worker threads.

use bytes::Bytes;
use kivi_engine::{DurabilityMode, EngineConfig, EngineError, LocalEngine, Placement};
use kivi_state::Key;
use kivi_tablet::{DirectorySnapshot, HashPrefix, PartitionRange, TabletState};
use kivi_types::{NamespaceId, TabletEpoch, TabletId, WorkerId, WriteGuardGeneration};

const NS: NamespaceId = NamespaceId::from_u64(1);
const EPOCH: TabletEpoch = TabletEpoch::INITIAL;
const GUARD: WriteGuardGeneration = WriteGuardGeneration::INITIAL;

fn hash_range(bits: u128, len: u8) -> PartitionRange {
    PartitionRange::Hash(HashPrefix::new(bits, len).expect("range"))
}

fn activate(snapshot: &DirectorySnapshot, tablet: TabletId) -> DirectorySnapshot {
    snapshot
        .clone()
        .stage(tablet)
        .and_then(|s| s.activate(tablet))
        .expect("stage+activate")
}

fn root_snapshot() -> DirectorySnapshot {
    let genesis =
        DirectorySnapshot::bootstrap(NS, TabletId::from_u64(1), hash_range(0, 0), EPOCH, GUARD)
            .expect("genesis");
    activate(&genesis, TabletId::from_u64(1))
}

fn split_snapshot() -> DirectorySnapshot {
    // Root 1 split into children 2 (0*) and 3 (1*), parent retired.
    let mut dir = root_snapshot();
    dir = dir
        .allocate(TabletId::from_u64(2), hash_range(0, 1), EPOCH, GUARD)
        .and_then(|s| s.stage(TabletId::from_u64(2)))
        .expect("child 0*");
    dir = dir
        .allocate(
            TabletId::from_u64(3),
            hash_range(1u128 << 127, 1),
            EPOCH,
            GUARD,
        )
        .and_then(|s| s.stage(TabletId::from_u64(3)))
        .expect("child 1*");
    dir = dir.seal(TabletId::from_u64(1)).expect("seal");
    dir = dir.activate(TabletId::from_u64(2)).expect("activate 2");
    dir = dir.activate(TabletId::from_u64(3)).expect("activate 3");
    dir.retire(
        TabletId::from_u64(1),
        kivi_tablet::Redirect::new(vec![TabletId::from_u64(2), TabletId::from_u64(3)]),
    )
    .expect("retire")
}

fn worker(n: u64) -> WorkerId {
    WorkerId::from_u64(n)
}

fn start_root(workers: usize) -> LocalEngine {
    LocalEngine::start(EngineConfig {
        namespace: NS,
        directory: root_snapshot(),
        placement: Placement::new([(TabletId::from_u64(1), worker(0))]),
        worker_count: workers,
        request_capacity: 64,
        network: None,
        durability: DurabilityMode::Ephemeral,
    })
    .expect("root engine starts")
}

#[test]
fn invalid_configurations_fail_before_serving() {
    let root = root_snapshot();
    let placement = Placement::new([(TabletId::from_u64(1), worker(0))]);
    // Zero workers.
    assert!(matches!(
        LocalEngine::start(EngineConfig {
            namespace: NS,
            directory: root.clone(),
            placement: placement.clone(),
            worker_count: 0,
            request_capacity: 64,
            network: None,
            durability: DurabilityMode::Ephemeral,
        }),
        Err(EngineError::InvalidConfig { .. })
    ));
    // Zero capacity.
    assert!(matches!(
        LocalEngine::start(EngineConfig {
            namespace: NS,
            directory: root.clone(),
            placement: placement.clone(),
            worker_count: 1,
            request_capacity: 0,
            network: None,
            durability: DurabilityMode::Ephemeral,
        }),
        Err(EngineError::InvalidConfig { .. })
    ));
    // Missing owner for the active tablet.
    assert!(matches!(
        LocalEngine::start(EngineConfig {
            namespace: NS,
            directory: root.clone(),
            placement: Placement::new([]),
            worker_count: 1,
            request_capacity: 64,
            network: None,
            durability: DurabilityMode::Ephemeral,
        }),
        Err(EngineError::Routing(_))
    ));
    // Worker out of range.
    assert!(matches!(
        LocalEngine::start(EngineConfig {
            namespace: NS,
            directory: root,
            placement: Placement::new([(TabletId::from_u64(1), worker(9))]),
            worker_count: 1,
            request_capacity: 64,
            network: None,
            durability: DurabilityMode::Ephemeral,
        }),
        Err(EngineError::Routing(_))
    ));
}

#[test]
fn typed_crud_round_trip_with_ttl() {
    let engine = start_root(1);
    let client = engine.client();
    let key = Key::from("user:1");
    assert_eq!(client.get(&key).expect("get"), None);
    assert!(!client.exists(&key).expect("exists"));
    client.set(&key, Bytes::from_static(b"ada")).expect("set");
    assert_eq!(
        client.get(&key).expect("get"),
        Some(Bytes::from_static(b"ada"))
    );
    assert!(client.exists(&key).expect("exists"));
    assert_eq!(
        client.counter_get(&Key::from("n")).expect("absent counter"),
        None
    );
    assert!(
        client.counter_get(&key).is_err(),
        "bytes key is not a counter"
    );
    assert_eq!(client.counter_add(&Key::from("n"), 5).expect("add"), 5);
    assert_eq!(client.counter_add(&Key::from("n"), -2).expect("add"), 3);
    assert_eq!(client.counter_get(&Key::from("n")).expect("read"), Some(3));
    // Expiry far in the future: live.
    let far = kivi_types::UnixMicros::from_micros(u64::MAX / 2);
    assert!(client.expire_at(&key, far).expect("expire"));
    assert_eq!(
        client.get(&key).expect("live").expect("value"),
        Bytes::from_static(b"ada")
    );
    assert!(client.persist_expiry(&key).expect("persist"));
    assert!(!client.persist_expiry(&key).expect("persist again"));
    assert!(client.delete(&key).expect("delete"));
    assert!(!client.delete(&key).expect("re-delete"));
    let report = engine.shutdown().expect("clean shutdown");
    assert_eq!(report.workers_joined, 1);
}

#[test]
fn multi_tablet_routing_reaches_both_owners() {
    let engine = LocalEngine::start(EngineConfig {
        namespace: NS,
        directory: split_snapshot(),
        placement: Placement::new([
            (TabletId::from_u64(2), worker(0)),
            (TabletId::from_u64(3), worker(1)),
        ]),
        worker_count: 2,
        request_capacity: 64,
        network: None,
        durability: DurabilityMode::Ephemeral,
    })
    .expect("split engine starts");
    assert_eq!(engine.worker_of(TabletId::from_u64(2)), Some(worker(0)));
    assert_eq!(engine.worker_of(TabletId::from_u64(3)), Some(worker(1)));
    // Find keys landing on each side deterministically.
    let client = engine.client();
    let mut probes: Vec<(TabletId, Key)> = Vec::new();
    for i in 0..1000u64 {
        let key = Key::from(format!("key:{i}"));
        let (tablet, _) = engine.route_key(&key).expect("covered");
        if !probes.iter().any(|(t, _)| *t == tablet) {
            probes.push((tablet, key));
        }
        if probes.len() == 2 {
            break;
        }
    }
    assert_eq!(probes.len(), 2, "both tablets must be reachable");
    for (tablet, key) in &probes {
        client.set(key, Bytes::from_static(b"v")).expect("set");
        assert_eq!(
            client.get(key).expect("get"),
            Some(Bytes::from_static(b"v"))
        );
        let _ = tablet;
    }
    let report = engine.shutdown().expect("clean shutdown");
    assert_eq!(report.workers_joined, 2);
}

#[test]
fn routing_replacement_is_atomic_and_validated() {
    let engine = start_root(1);
    let client = engine.client();
    let version_before = engine.routing_version();
    // No-op replacement (same content) succeeds.
    engine
        .replace_routing(
            root_snapshot(),
            Placement::new([(TabletId::from_u64(1), worker(0))]),
        )
        .expect("no-op swap");
    assert!(engine.routing_version() >= version_before);
    // Incoherent replacement fails and the old publication keeps serving.
    assert!(matches!(
        engine.replace_routing(root_snapshot(), Placement::new([])),
        Err(EngineError::Routing(_))
    ));
    let key = Key::from("still-here");
    client
        .set(&key, Bytes::from_static(b"v"))
        .expect("serves after failed swap");
    assert_eq!(
        client.get(&key).expect("get"),
        Some(Bytes::from_static(b"v"))
    );
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn shutdown_surfaces_worker_loss_to_clients() {
    let engine = start_root(1);
    let client = engine.client();
    client
        .set(&Key::from("k"), Bytes::from_static(b"v"))
        .expect("set");
    engine.shutdown().expect("clean shutdown");
    assert!(matches!(
        client.get(&Key::from("k")),
        Err(EngineError::WorkerDown { .. })
    ));
}

#[test]
fn directory_states_are_visible_for_debugging() {
    let snapshot = root_snapshot();
    let tablet = snapshot.get(TabletId::from_u64(1)).expect("tablet");
    assert_eq!(tablet.state(), TabletState::Active);
}
