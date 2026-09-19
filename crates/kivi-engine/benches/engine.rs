//! Baseline throughput benchmarks: direct tablet cost vs. embedded-client
//! channel cost. Repeatable locally with `cargo bench -p kivi-engine`.
//!
//! These numbers pin the pre-optimization baseline so future runtime, index,
//! and allocator changes measure rather than guess. Run in release mode for
//! meaningful figures; debug numbers only prove the harness works.

use divan::{Bencher, black_box, counter::ItemsCount};

use bytes::Bytes;
use kivi_engine::{DurabilityMode, EngineConfig, LiveTablet, LocalEngine, Placement};
use kivi_state::{Key, Operation};
use kivi_tablet::{DirectorySnapshot, HashPrefix, PartitionRange};
use kivi_types::{
    ClusterId, NamespaceId, NodeId, NodeIncarnation, TabletAuthority, TabletEpoch, TabletId,
    UnixMicros, WorkerId, WriteGuardGeneration,
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
        chunks: kivi_engine::ChunkFabricConfig::default(),
        network: None,
        durability: DurabilityMode::Ephemeral,
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

/// Large-value benchmarks: a 300 KiB value stages one chunk on set and
/// resolves it on get; the range patch exercises the inline splice path.
/// These pin the chunk-fabric overhead against the point-op baselines
/// above (staging cost, lane round trip, splice cost).
const BIG_LEN: usize = 300_000;

fn big_value() -> Bytes {
    Bytes::from(vec![0xABu8; BIG_LEN])
}

#[divan::bench]
fn client_set_chunked(bencher: Bencher) {
    let engine = root_engine();
    let client = engine.client();
    let key = Key::from(KEY);
    let value = big_value();
    bencher.counter(ItemsCount::new(1u64)).bench_local(|| {
        client.set(&key, value.clone()).expect("set");
        black_box(());
    });
}

#[divan::bench]
fn client_get_chunked(bencher: Bencher) {
    let engine = root_engine();
    let client = engine.client();
    let key = Key::from(KEY);
    client.set(&key, big_value()).expect("seed");
    bencher
        .counter(ItemsCount::new(1u64))
        .bench_local(|| black_box(client.get(&key).expect("get")));
}

/// Networked engine: one worker serving loopback TCP plus the embedded
/// channel bridge. `LocalClient` calls below traverse the bridge — the
/// event-driven wakeup path — so these benches pin the A1 latency floor
/// fix (the old 100 µs bridge poll quantum showed up here directly).
fn net_engine() -> LocalEngine {
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
        chunks: kivi_engine::ChunkFabricConfig::default(),
        network: Some(kivi_engine::EngineNetwork {
            base_port: 0,
            ports: Vec::new(),
            bind_ip: [127, 0, 0, 1].into(),
            max_frame: kivi_protocol::DEFAULT_MAX_FRAME,
            affinity: kivi_engine::AffinityMode::Disabled,
            node_id: NodeId::from_u64(1),
            cluster_id: ClusterId::from_u128(1),
            incarnation: NodeIncarnation::INITIAL,
            conn: kivi_engine::ConnLimits::default(),
            turn: kivi_engine::TurnBudget::default(),
        }),
        durability: DurabilityMode::Ephemeral,
    })
    .expect("networked engine starts")
}

#[divan::bench]
fn bridge_set(bencher: Bencher) {
    let engine = net_engine();
    let client = engine.client();
    let key = Key::from(KEY);
    let value = Bytes::from_static(VALUE_16);
    bencher.counter(ItemsCount::new(1u64)).bench_local(|| {
        client.set(&key, value.clone()).expect("set");
        black_box(());
    });
}

#[divan::bench]
fn bridge_get(bencher: Bencher) {
    let engine = net_engine();
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
fn client_set_range_inline(bencher: Bencher) {
    let engine = root_engine();
    let client = engine.client();
    let key = Key::from(KEY);
    client
        .set(&key, Bytes::from_static(VALUE_16))
        .expect("seed");
    let patch = Bytes::from_static(b"XY");
    bencher.counter(ItemsCount::new(1u64)).bench_local(|| {
        client.set_range(&key, 4, patch.clone()).expect("patch");
        black_box(());
    });
}

/// Same-tablet atomic batch end to end: four blind puts through the
/// embedded client, exercising the cheap `TxnCommitLocal` path (one
/// ordered mutation, no 2PC) against the point-op baselines above.
#[divan::bench]
fn client_atomic_batch_local(bencher: Bencher) {
    use kivi_state::{TxnExpect, TxnWrite, TxnWriteKind};
    let engine = root_engine();
    let client = engine.client();
    let writes: Vec<TxnWrite> = (0..4)
        .map(|index| TxnWrite {
            key: Key::from(format!("batch:{index}")),
            kind: TxnWriteKind::Put(Bytes::from_static(VALUE_16)),
            expect: TxnExpect::Any,
        })
        .collect();
    bencher.counter(ItemsCount::new(4u64)).bench_local(|| {
        black_box(client.atomic_batch(black_box(&writes)).expect("batch"));
    });
}
