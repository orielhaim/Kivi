//! Real multithreaded tests over OS threads: exact concurrent counters,
//! multi-tablet parallelism, and a mixed workload validated against a
//! sequential reference model.
//!
//! Client calls that hit a full bounded queue return `Overloaded`, which the
//! harnesses below retry with backoff-freedag yields (bounded attempts):
//! overload is expected under contention and must never corrupt results.

use std::collections::BTreeMap;
use std::sync::{Arc, Barrier};
use std::thread;

use bytes::Bytes;
use kivi_engine::{DurabilityMode, EngineConfig, EngineError, LocalClient, LocalEngine, Placement};
use kivi_state::{Key, ObjectStore, Operation};
use kivi_tablet::{DirectorySnapshot, HashPrefix, PartitionRange};
use kivi_types::{NamespaceId, TabletEpoch, TabletId, UnixMicros, WorkerId, WriteGuardGeneration};

const NS: NamespaceId = NamespaceId::from_u64(1);
const EPOCH: TabletEpoch = TabletEpoch::INITIAL;
const GUARD: WriteGuardGeneration = WriteGuardGeneration::INITIAL;
const PAST: UnixMicros = UnixMicros::from_micros(1);
const FUTURE: UnixMicros = UnixMicros::from_micros(u64::MAX / 2);

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
    activate(
        &DirectorySnapshot::bootstrap(NS, TabletId::from_u64(1), hash_range(0, 0), EPOCH, GUARD)
            .expect("genesis"),
        TabletId::from_u64(1),
    )
}

fn split_snapshot() -> DirectorySnapshot {
    let mut dir = root_snapshot();
    for (id, bits) in [(2u64, 0u128), (3u64, 1u128 << 127)] {
        dir = dir
            .allocate(TabletId::from_u64(id), hash_range(bits, 1), EPOCH, GUARD)
            .and_then(|s| s.stage(TabletId::from_u64(id)))
            .expect("child staged");
    }
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

/// Runs `op` to a non-overload outcome, retrying bounded-queue overloads
/// with yields. Panics after the attempt budget.
fn run_result<T>(
    client: &LocalClient,
    mut op: impl FnMut(&LocalClient) -> Result<T, EngineError>,
) -> Result<T, EngineError> {
    for _ in 0..100_000 {
        match op(client) {
            Err(EngineError::Overloaded) => thread::yield_now(),
            result => return result,
        }
    }
    panic!("overload retries exhausted");
}

/// Runs `op` to success, retrying bounded-queue overloads with yields.
/// Panics on any non-overload error or after the attempt budget.
fn run<T>(client: &LocalClient, mut op: impl FnMut(&LocalClient) -> Result<T, EngineError>) -> T {
    run_result(client, &mut op).expect("non-overload engine error")
}

/// Runs `op`, retrying overloads but tolerating legitimate semantic
/// rejections (wrong-type, overflow): the reference model produces and
/// ignores the same outcomes, so final-state comparison stays exact.
fn run_tolerant<T>(
    client: &LocalClient,
    mut op: impl FnMut(&LocalClient) -> Result<T, EngineError>,
) {
    for _ in 0..100_000 {
        match op(client) {
            Err(EngineError::Overloaded) => thread::yield_now(),
            Err(EngineError::Tablet(_)) | Ok(_) => return,
            Err(error) => panic!("transport engine error: {error:?}"),
        }
    }
    panic!("overload retries exhausted");
}

#[test]
fn concurrent_counter_is_exact() {
    let engine = LocalEngine::start(EngineConfig {
        namespace: NS,
        directory: root_snapshot(),
        placement: Placement::new([(TabletId::from_u64(1), worker(0))]),
        worker_count: 1,
        request_capacity: 16,
        chunks: kivi_engine::ChunkFabricConfig::default(),
        network: None,
        durability: DurabilityMode::Ephemeral,
    })
    .expect("engine starts");
    let key = Key::from("counter");
    let barrier = Arc::new(Barrier::new(9));
    let mut threads = Vec::new();
    for _ in 0..8 {
        let client = engine.client();
        let gate = Arc::clone(&barrier);
        let key = key.clone();
        threads.push(thread::spawn(move || {
            gate.wait();
            for _ in 0..250 {
                run(&client, |c| c.counter_add(&key, 1));
            }
        }));
    }
    barrier.wait();
    for thread in threads {
        thread.join().expect("worker thread joins");
    }
    let client = engine.client();
    assert_eq!(run(&client, |c| c.counter_get(&key)), Some(8 * 250));
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn multi_tablet_parallelism_makes_independent_progress() {
    let engine = LocalEngine::start(EngineConfig {
        namespace: NS,
        directory: split_snapshot(),
        placement: Placement::new([
            (TabletId::from_u64(2), worker(0)),
            (TabletId::from_u64(3), worker(1)),
        ]),
        worker_count: 2,
        request_capacity: 64,
        chunks: kivi_engine::ChunkFabricConfig::default(),
        network: None,
        durability: DurabilityMode::Ephemeral,
    })
    .expect("engine starts");
    assert_ne!(
        engine.worker_of(TabletId::from_u64(2)),
        engine.worker_of(TabletId::from_u64(3))
    );
    // One thread per tablet family: route probe keys, then hammer each side.
    let probe = |prefix: &str| -> Key {
        for i in 0..10_000u64 {
            let key = Key::from(format!("{prefix}:{i}"));
            if let Ok((tablet, _)) = engine.route_key(&key) {
                let want = if prefix == "even" {
                    TabletId::from_u64(2)
                } else {
                    TabletId::from_u64(3)
                };
                if tablet == want {
                    return key;
                }
            }
        }
        panic!("no key routed to {prefix}");
    };
    let key_a = probe("even");
    let key_b = probe("odd");
    let barrier = Arc::new(Barrier::new(3));
    let mut threads = Vec::new();
    for (key, delta) in [(key_a.clone(), 1i64), (key_b.clone(), 2i64)] {
        let client = engine.client();
        let gate = Arc::clone(&barrier);
        threads.push(thread::spawn(move || {
            gate.wait();
            for _ in 0..500 {
                run(&client, |c| c.counter_add(&key, delta));
            }
        }));
    }
    barrier.wait();
    for thread in threads {
        thread.join().expect("worker thread joins");
    }
    let client = engine.client();
    // Tablet placement decides which key saw which delta; read both back.
    let total: i64 = [key_a, key_b]
        .iter()
        .map(|key| run(&client, |c| c.counter_get(key)).expect("counter present"))
        .sum();
    assert_eq!(total, 1_500, "500 x +1 plus 500 x +2");
    engine.shutdown().expect("clean shutdown");
}

/// Deterministic per-thread op script over disjoint keys.
fn thread_script(thread: u64, ops: usize) -> Vec<(Key, u8, Vec<u8>, i64)> {
    // (key, op_tag, bytes, number): tag selects the operation family.
    let mut script = Vec::with_capacity(ops);
    let mut rng = 0x9E37_79B9_7F4A_7C15u64.wrapping_add(thread.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    for i in 0..ops {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let key = Key::from(format!("t{thread}:k{}", i % 32));
        let tag = (rng % 9) as u8;
        let mut bytes = (rng % 97) as u8;
        if bytes == 0 {
            bytes = 1;
        }
        let width = usize::try_from(rng % 5).expect("tiny test value fits");
        let delta = (rng % 7).cast_signed() - 3;
        script.push((key, tag, vec![bytes; 1 + width], delta));
    }
    script
}

#[test]
fn mixed_workload_matches_sequential_reference() {
    const THREADS: u64 = 4;
    const OPS: usize = 400;
    let engine = LocalEngine::start(EngineConfig {
        namespace: NS,
        directory: split_snapshot(),
        placement: Placement::new([
            (TabletId::from_u64(2), worker(0)),
            (TabletId::from_u64(3), worker(1)),
        ]),
        worker_count: 2,
        request_capacity: 64,
        chunks: kivi_engine::ChunkFabricConfig::default(),
        network: None,
        durability: DurabilityMode::Ephemeral,
    })
    .expect("engine starts");
    // Reference: the same scripts applied sequentially per disjoint key set.
    // Wall-clock-sensitive expiries use far-future/far-past stamps so the
    // real engine clock and the reference agree by construction.
    let mut reference = ObjectStore::new();
    let scripts: Vec<_> = (0..THREADS).map(|t| thread_script(t, OPS)).collect();
    for script in &scripts {
        for (key, tag, bytes, number) in script {
            apply_reference(&mut reference, key, *tag, bytes, *number);
        }
    }
    // Concurrent execution across threads (disjoint keys per thread).
    let barrier = Arc::new(Barrier::new(
        usize::try_from(THREADS + 1).expect("few threads"),
    ));
    let mut threads = Vec::new();
    for script in scripts {
        let client = engine.client();
        let gate = Arc::clone(&barrier);
        threads.push(thread::spawn(move || {
            gate.wait();
            for (key, tag, bytes, number) in &script {
                apply_concurrent(&client, key, *tag, bytes, *number);
            }
        }));
    }
    barrier.wait();
    for thread in threads {
        thread.join().expect("worker thread joins");
    }
    // Read back the full key space and compare against the reference dump.
    // Get/counter-get are mutually exclusive by stored type; exactly one of
    // the two succeeds per live key (wrong-type outcomes are the signal).
    let client = engine.client();
    let mut observed = BTreeMap::new();
    for thread in 0..THREADS {
        for i in 0..32 {
            let key = Key::from(format!("t{thread}:k{i}"));
            let value = match run_result(&client, |c| c.get(&key)) {
                Ok(value) => value,
                Err(EngineError::Tablet(_)) => None,
                Err(error) => panic!("transport engine error: {error:?}"),
            };
            let counter = match run_result(&client, |c| c.counter_get(&key)) {
                Ok(value) => value,
                Err(EngineError::Tablet(_)) => None,
                Err(error) => panic!("transport engine error: {error:?}"),
            };
            let expiry = run(&client, |c| c.get_expiry(&key));
            observed.insert(key.as_bytes().to_vec(), (value, counter, expiry));
        }
    }
    for (key, object) in reference.snapshot_sorted() {
        let raw = key.as_bytes().to_vec();
        let entry = observed.remove(&raw).expect("key observed");
        // Expiries in this test are only NEVER, PAST, or FUTURE, so liveness
        // agrees between the fixed reference clock and the real wall clock.
        if object.expiry().as_stamp() == Some(PAST) {
            assert_eq!(entry.0, None, "expired key {key} reads absent");
            assert_eq!(entry.1, None, "expired key {key} reads absent");
            assert_eq!(entry.2, None, "expired key {key} reads absent");
            continue;
        }
        match object.value() {
            kivi_state::LogicalValue::Bytes(value) => {
                assert_eq!(entry.0, Some(value.clone()), "bytes for {key}");
                assert_eq!(entry.1, None, "no counter for {key}");
            }
            // Unreachable here: the scripted ops are inline-only (Set,
            // CounterAdd, expiry, delete) on an empty store, so no chunked
            // root can exist. Chunked equivalence is covered by the state
            // model test and the chunked engine tests.
            kivi_state::LogicalValue::Chunked(_) => {
                panic!("no scripted op produces chunked roots for {key}")
            }
            kivi_state::LogicalValue::StrictCounter(value) => {
                assert_eq!(entry.1, Some(*value), "counter for {key}");
                assert_eq!(entry.0, None, "no bytes for {key}");
            }
        }
        assert_eq!(entry.2, Some(object.expiry()), "expiry for {key}");
    }
    // Keys the reference dropped must read absent.
    for (raw, (value, counter, expiry)) in &observed {
        assert_eq!(value, &None, "dropped key {raw:?} reads absent");
        assert_eq!(counter, &None, "dropped key {raw:?} reads absent");
        assert_eq!(expiry, &None, "dropped key {raw:?} reads absent");
    }
    engine.shutdown().expect("clean shutdown");
}

/// Applies one scripted op to the sequential reference model.
fn apply_reference(store: &mut ObjectStore, key: &Key, tag: u8, bytes: &[u8], number: i64) {
    // Fixed logical timestamp: far from both PAST and FUTURE sentinels used
    // below, so wall-clock drift cannot matter.
    let now = UnixMicros::from_micros(1_000_000_000);
    let op = match tag {
        0 => Operation::Set {
            key: key.clone(),
            value: Bytes::from(bytes.to_vec()),
        },
        1 => Operation::Get { key: key.clone() },
        2 => Operation::Delete { key: key.clone() },
        3 => Operation::Exists { key: key.clone() },
        4 => Operation::CounterAdd {
            key: key.clone(),
            delta: number,
        },
        5 => Operation::CounterGet { key: key.clone() },
        6 => Operation::ExpireAt {
            key: key.clone(),
            expires_at: if number % 2 == 0 { PAST } else { FUTURE },
        },
        7 => Operation::PersistExpiry { key: key.clone() },
        _ => Operation::GetExpiry { key: key.clone() },
    };
    match store.prepare(&op, now) {
        Ok(kivi_state::Prepared::Read(_)) | Err(_) => {}
        Ok(kivi_state::Prepared::Write(mutation)) => {
            let _ = store.apply(&mutation, now);
        }
    }
}

/// Applies one scripted op concurrently (retrying bounded-queue overloads,
/// tolerating semantic rejections exactly like the reference).
fn apply_concurrent(client: &LocalClient, key: &Key, tag: u8, bytes: &[u8], number: i64) {
    match tag {
        0 => run_tolerant(client, |c| c.set(key, Bytes::from(bytes.to_vec()))),
        1 => run_tolerant(client, |c| c.get(key)),
        2 => run_tolerant(client, |c| c.delete(key)),
        3 => run_tolerant(client, |c| c.exists(key)),
        4 => run_tolerant(client, |c| c.counter_add(key, number)),
        5 => run_tolerant(client, |c| c.counter_get(key)),
        6 => run_tolerant(client, |c| {
            c.expire_at(key, if number % 2 == 0 { PAST } else { FUTURE })
        }),
        7 => run_tolerant(client, |c| c.persist_expiry(key)),
        _ => run_tolerant(client, |c| c.get_expiry(key)),
    }
}
