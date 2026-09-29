//! Engine integration tests: bootstrap, typed CRUD, routing, replacement, and
//! shutdown over real worker threads.

mod harness;

use bytes::Bytes;
use harness::{
    LEFT, RIGHT, ROOT, config, root_placement, root_snapshot, split_placement, split_snapshot,
    worker,
};
use kivi_engine::{EngineConfig, EngineError, LocalEngine, Placement};
use kivi_state::Key;

#[test]
fn invalid_configurations_fail_before_serving() {
    let root = root_snapshot();
    let placement = root_placement();
    // Zero workers.
    assert!(matches!(
        LocalEngine::start(EngineConfig {
            worker_count: 0,
            ..config(root.clone(), placement.clone(), 0)
        }),
        Err(EngineError::InvalidConfig { .. })
    ));
    // Zero request capacity.
    assert!(matches!(
        LocalEngine::start(EngineConfig {
            request_capacity: 0,
            ..config(root.clone(), placement.clone(), 1)
        }),
        Err(EngineError::InvalidConfig { .. })
    ));
    // Missing owner for the active tablet.
    assert!(matches!(
        LocalEngine::start(config(root.clone(), Placement::new([]), 1)),
        Err(EngineError::Routing(_))
    ));
    // Worker out of range.
    assert!(matches!(
        LocalEngine::start(config(root, Placement::new([(ROOT, worker(9))]), 1)),
        Err(EngineError::Routing(_))
    ));
}

fn start_root(workers: usize) -> LocalEngine {
    LocalEngine::start(config(root_snapshot(), root_placement(), workers))
        .expect("root engine starts")
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
    let far = kivi_types::WallTimestamp::from_micros(i64::MAX / 2);
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
    let engine = LocalEngine::start(config(split_snapshot(), split_placement(), 2))
        .expect("split engine starts");
    assert_eq!(engine.worker_of(LEFT), Some(worker(0)));
    assert_eq!(engine.worker_of(RIGHT), Some(worker(1)));
    // Find one key per side, then round-trip each through the worker that owns
    // it. A route that named the wrong owner, or a retired parent, would fail
    // one of these.
    let client = engine.client();
    let mut probes = Vec::new();
    for index in 0..1000u64 {
        let key = Key::from(format!("key:{index}"));
        let tablet = engine.route_key(&key).expect("covered").0;
        if !probes
            .iter()
            .any(|other: &Key| engine.route_key(other).expect("covered").0 == tablet)
        {
            probes.push(key);
        }
        if probes.len() == 2 {
            break;
        }
    }
    assert_eq!(probes.len(), 2, "both tablets must be reachable");
    for key in &probes {
        client.set(key, Bytes::from_static(b"v")).expect("set");
        assert_eq!(
            client.get(key).expect("get"),
            Some(Bytes::from_static(b"v"))
        );
    }
    let report = engine.shutdown().expect("clean shutdown");
    assert_eq!(report.workers_joined, 2);
}

#[test]
fn routing_replacement_is_atomic_and_validated() {
    let engine = start_root(1);
    let client = engine.client();
    engine
        .replace_routing(root_snapshot(), root_placement())
        .expect("no-op swap");
    // An incoherent replacement fails and the old publication keeps serving.
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
