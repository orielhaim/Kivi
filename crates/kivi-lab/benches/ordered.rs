//! Ordered data-plane benchmarks over a single-node TCP engine (same
//! substrate as `ordered_engine` money tests): range scans, atomic
//! batches (single/multi tablet), and secondary index maintenance +
//! queries through the normal native client paths.
//!
//! Run in release mode for meaningful figures:
//! `cargo bench -p kivi-lab --bench ordered`.

use divan::{Bencher, black_box, counter::ItemsCount};

use bytes::Bytes;
use kivi_client::{
    ClientConfig, NativeClient,
    ordered::{
        BatchExpect, BatchWriteKind, BatchWriteSpec, IndexTermSet, ScanConsistency, ScanDirection,
        ScanProjection,
    },
};
use kivi_engine::{
    ConnLimits, DurabilityMode, EngineConfig, EngineNetwork, LocalEngine, Placement, TurnBudget,
};
use kivi_state::{IndexId, IndexKind, Key};
use kivi_tablet::DirectorySnapshot;
use kivi_types::{ClusterId, NamespaceId, NodeId, NodeIncarnation, TabletId, WorkerId};

const NS: NamespaceId = NamespaceId::from_u64(1);

/// Divan runs benches in parallel threads: each bench starts its own
/// engine, and parallel engines contend (ports, reactor pacing), which
/// shows up as transport flakes in setup. Serialize bench bodies; per
/// sample timings are unaffected (setup runs once outside sampling).
static BENCH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    BENCH_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Three ordered tablets: 1:[empty, m), 2:[m, t), 3:[t, +inf).
fn start_ordered() -> LocalEngine {
    let directory = DirectorySnapshot::static_ordered_tiles(NS, &[b"m".to_vec(), b"t".to_vec()])
        .expect("ordered tiling builds");
    LocalEngine::start(EngineConfig {
        namespace: NS,
        directory,
        placement: Placement::new([
            (TabletId::from_u64(1), WorkerId::from_u64(0)),
            (TabletId::from_u64(2), WorkerId::from_u64(1)),
            (TabletId::from_u64(3), WorkerId::from_u64(0)),
        ]),
        worker_count: 2,
        request_capacity: 4096,
        chunks: kivi_engine::ChunkFabricConfig::default(),
        network: Some(EngineNetwork {
            base_port: 0,
            ports: Vec::new(),
            bind_ip: [127, 0, 0, 1].into(),
            max_frame: kivi_protocol::DEFAULT_MAX_FRAME,
            affinity: kivi_engine::AffinityMode::Disabled,
            node_id: NodeId::from_u64(1),
            cluster_id: ClusterId::from_u128(1),
            incarnation: NodeIncarnation::INITIAL,
            conn: ConnLimits::default(),
            turn: TurnBudget::default(),
        }),
        durability: DurabilityMode::Ephemeral,
    })
    .expect("ordered engine starts")
}

fn client_for(engine: &LocalEngine) -> NativeClient {
    let seed = engine.worker_addrs()[0].to_string();
    NativeClient::new(ClientConfig {
        seeds: vec![seed],
        namespace: NS,
        ..ClientConfig::default()
    })
    .expect("client builds")
}

fn key(prefix: u8, index: usize) -> Vec<u8> {
    let mut key = vec![prefix];
    key.extend_from_slice(format!("{index:06}").as_bytes());
    key
}

/// Seeds plain keys across all three tablets plus one indexed family
/// (`city` non-unique terms) for the index benches. Seed writes retry on
/// transport ambiguity (fresh-engine startup races a first dial; puts and
/// indexed sets are idempotent/OCC-guarded, so retries never duplicate).
fn seed(client: &NativeClient) {
    for prefix in *b"anz" {
        for index in 0..1000usize {
            let key = Key::from(key(prefix, index));
            let value = Bytes::from(vec![0x76; 16]);
            let mut tries = 0u32;
            loop {
                match client.set(&key, value.clone()) {
                    Ok(()) => break,
                    Err(kivi_client::ClientError::AmbiguousOutcome) if tries < 20 => {
                        tries += 1;
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                    Err(error) => panic!("seed set works: {error:?}"),
                }
            }
        }
    }
    let terms = |city: &str| IndexTermSet {
        index: IndexId::from_u64(1),
        kind: IndexKind::NonUnique,
        terms: vec![city.as_bytes().to_vec()],
    };
    for index in 0..500usize {
        let city = ["berlin", "oslo", "zurich"][index % 3];
        let mut tries = 0u32;
        loop {
            match client.indexed_set(
                NS,
                &key(b'i', 10_000 + index),
                format!("profile:{index}").into_bytes(),
                &[terms(city)],
            ) {
                Ok(_) => break,
                Err(kivi_client::ClientError::AmbiguousOutcome) if tries < 20 => {
                    tries += 1;
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(error) => panic!("seed indexed set works: {error:?}"),
            }
        }
    }
}

fn main() {
    divan::main();
}

/// One 100-entry keys-only scan page (single tablet window).
#[divan::bench]
fn scan_page_100(bencher: Bencher) {
    let _guard = lock();
    let engine = start_ordered();
    let client = client_for(&engine);
    seed(&client);
    bencher.counter(ItemsCount::new(100u64)).bench_local(|| {
        black_box(
            client
                .scan_page(
                    NS,
                    Some(key(b'a', 0)),
                    None,
                    ScanDirection::Forward,
                    ScanConsistency::LatestPerTablet,
                    ScanProjection::KeysOnly,
                    100,
                    1 << 20,
                )
                .expect("scan page works"),
        )
    });
    drop(engine);
}

/// 4-key single-tablet atomic batch (fast path).
#[divan::bench]
fn atomic_batch_single_tablet(bencher: Bencher) {
    let _guard = lock();
    let engine = start_ordered();
    let client = client_for(&engine);
    seed(&client);
    let mut seq: usize = 0;
    bencher.counter(ItemsCount::new(4u64)).bench_local(|| {
        seq += 1;
        let writes: Vec<BatchWriteSpec> = (0..4)
            .map(|slot| BatchWriteSpec {
                key: key(b'a', 50_000 + seq * 4 + slot),
                kind: BatchWriteKind::Put(vec![0x76; 16]),
                expect: BatchExpect::Absent,
            })
            .collect();
        black_box(client.atomic_batch(NS, &writes).expect("batch commits"))
    });
    drop(engine);
}

/// 4-key cross-tablet atomic batch (client-driven 2PC).
#[divan::bench]
fn atomic_batch_multi_tablet(bencher: Bencher) {
    let _guard = lock();
    let engine = start_ordered();
    let client = client_for(&engine);
    seed(&client);
    let mut seq: usize = 0;
    bencher.counter(ItemsCount::new(4u64)).bench_local(|| {
        seq += 1;
        let writes: Vec<BatchWriteSpec> = b"anza"
            .iter()
            .enumerate()
            .map(|(slot, prefix)| BatchWriteSpec {
                key: key(*prefix, 60_000 + seq * 4 + slot),
                kind: BatchWriteKind::Put(vec![0x76; 16]),
                expect: BatchExpect::Absent,
            })
            .collect();
        black_box(client.atomic_batch(NS, &writes).expect("batch commits"))
    });
    drop(engine);
}

/// Full indexed write: primary + index entry + projection, one transaction.
#[divan::bench]
fn indexed_set(bencher: Bencher) {
    let _guard = lock();
    let engine = start_ordered();
    let client = client_for(&engine);
    seed(&client);
    let mut seq: usize = 0;
    bencher.bench_local(|| {
        seq += 1;
        black_box(
            client
                .indexed_set(
                    NS,
                    &key(b'i', 70_000 + seq),
                    b"profile".to_vec(),
                    &[IndexTermSet {
                        index: IndexId::from_u64(1),
                        kind: IndexKind::NonUnique,
                        terms: vec![b"berlin".to_vec()],
                    }],
                )
                .expect("indexed set works"),
        )
    });
    drop(engine);
}

/// Non-unique equality lookup over the seeded family (limit 100).
#[divan::bench]
fn index_equal_100(bencher: Bencher) {
    let _guard = lock();
    let engine = start_ordered();
    let client = client_for(&engine);
    seed(&client);
    bencher.counter(ItemsCount::new(100u64)).bench_local(|| {
        black_box(
            client
                .index_equal(NS, IndexId::from_u64(1), b"berlin", 100)
                .expect("equality works"),
        )
    });
    drop(engine);
}
