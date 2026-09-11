//! Baseline throughput benchmarks: direct tablet cost vs. embedded-client
//! channel cost. Repeatable locally with `cargo bench -p kivi-engine`.
//!
//! These numbers pin the pre-optimization baseline so future runtime, index,
//! and allocator changes measure rather than guess. Run in release mode for
//! meaningful figures; debug numbers only prove the harness works.

use divan::{Bencher, black_box, counter::ItemsCount};

use bytes::Bytes;
use kivi_engine::{EngineConfig, LiveTablet, LocalEngine, Placement};
use kivi_state::{Key, Operation};
use kivi_tablet::{DirectorySnapshot, HashPrefix, PartitionRange};
use kivi_types::{
    NamespaceId, TabletAuthority, TabletEpoch, TabletId, UnixMicros, WorkerId, WriteGuardGeneration,
};

const NS: NamespaceId = NamespaceId::from_u64(1);
const NOW: UnixMicros = UnixMicros::from_micros(1_000_000);
const KEY: &str = "bench:key";
const VALUE_16: &[u8] = b"0123456789abcdef";

fn live_tablet() -> LiveTablet {
    let tablet = TabletId::from_u64(1);
    let authority =
        TabletAuthority::new(tablet, TabletEpoch::INITIAL, WriteGuardGeneration::INITIAL);
    let snapshot = DirectorySnapshot::bootstrap(
        NS,
        tablet,
        PartitionRange::Hash(HashPrefix::new(0, 0).expect("root")),
        TabletEpoch::INITIAL,
        WriteGuardGeneration::INITIAL,
    )
    .expect("genesis");
    let descriptor = snapshot.get(tablet).expect("descriptor").clone();
    LiveTablet::from_descriptor(&descriptor, authority).expect("live tablet")
}

fn root_engine() -> LocalEngine {
    let tablet = TabletId::from_u64(1);
    let directory = DirectorySnapshot::bootstrap(
        NS,
        tablet,
        PartitionRange::Hash(HashPrefix::new(0, 0).expect("root")),
        TabletEpoch::INITIAL,
        WriteGuardGeneration::INITIAL,
    )
    .and_then(|s| s.stage(tablet))
    .and_then(|s| s.activate(tablet))
    .expect("active root");
    LocalEngine::start(EngineConfig {
        namespace: NS,
        directory,
        placement: Placement::new([(tablet, WorkerId::from_u64(0))]),
        worker_count: 1,
        request_capacity: 1024,
        network: None,
    })
    .expect("engine starts")
}

fn main() {
    divan::main();
}

#[divan::bench]
fn direct_set(bencher: Bencher) {
    let mut tablet = live_tablet();
    let op = Operation::Set {
        key: Key::from(KEY),
        value: Bytes::from_static(VALUE_16),
    };
    bencher
        .counter(ItemsCount::new(1u64))
        .bench_local(|| black_box(tablet.execute(&op, NOW).expect("set")));
}

#[divan::bench]
fn direct_get(bencher: Bencher) {
    let mut tablet = live_tablet();
    tablet
        .execute(
            &Operation::Set {
                key: Key::from(KEY),
                value: Bytes::from_static(VALUE_16),
            },
            NOW,
        )
        .expect("seed");
    let op = Operation::Get {
        key: Key::from(KEY),
    };
    bencher
        .counter(ItemsCount::new(1u64))
        .bench_local(|| black_box(tablet.execute(&op, NOW).expect("get")));
}

#[divan::bench]
fn direct_counter_add(bencher: Bencher) {
    let mut tablet = live_tablet();
    let op = Operation::CounterAdd {
        key: Key::from(KEY),
        delta: 1,
    };
    bencher
        .counter(ItemsCount::new(1u64))
        .bench_local(|| black_box(tablet.execute(&op, NOW).expect("add")));
}

#[divan::bench]
fn client_get(bencher: Bencher) {
    let engine = root_engine();
    let client = engine.client();
    let key = Key::from(KEY);
    client
        .set(&key, Bytes::from_static(VALUE_16))
        .expect("seed");
    bencher
        .counter(ItemsCount::new(1u64))
        .bench_local(|| black_box(client.get(&key).expect("get")));
}

#[divan::bench]
fn client_counter_add(bencher: Bencher) {
    let engine = root_engine();
    let client = engine.client();
    let key = Key::from(KEY);
    bencher
        .counter(ItemsCount::new(1u64))
        .bench_local(|| black_box(client.counter_add(&key, 1).expect("add")));
}
