//! Fabric acceptance gate: multi-node workload proving the Memory
//! Fabric (and the cluster medium-to-sidecar path) never changes logical
//! semantics.
//!
//! A single deterministic driver holds a sequential reference model while
//! a 3-process ordered cluster absorbs member kill/restart cycles,
//! snapshots, and a full restart. Every acknowledged write mirrors into
//! the model; the finale reads back every model key exactly, covers the
//! range with a scan, and asserts no stuck intents. No wall-clock sleeps:
//! only harness waits.
//!
//! Split/merge churn is deliberately OUT of this gate: live topology
//! reorganization is covered by the elastic suite, and combined
//! split+kill chaos by `cluster_acceptance.rs` (currently catching
//! pre-existing native-plane defects). This gate proves the fabric claim:
//! placement, sidecars, failover, and restart affect latency and cost,
//! never value, consistency, or durability.

use std::collections::{BTreeSet, HashMap};

use bytes::Bytes;
use kivi_client::ordered::{
    BatchExpect, BatchWriteKind, BatchWriteSpec, ScanConsistency, ScanDirection, ScanOptions,
    ScanProjection,
};
use kivi_lab::cluster::Cluster;
use kivi_state::Key;
use kivi_types::NamespaceId;

const NS: NamespaceId = NamespaceId::from_u64(1);

/// Genesis tiling rotation for load spread.
fn tablet_of(index: usize) -> u8 {
    u8::try_from(index % 4).unwrap_or(0)
}

/// Genesis tiling prefixes, mirroring the acceptance suite.
fn spread_key(tablet: u8, index: usize) -> Vec<u8> {
    let prefix = match tablet {
        0 => 0x10u8,
        1 => b'A',
        2 => 0x90u8,
        _ => 0xF0u8,
    };
    let mut key = vec![prefix];
    key.extend_from_slice(format!("{index:06}").as_bytes());
    key
}

/// Deterministic incompressible bytes (medium tier must not compress
/// into the inline path or dodge the sidecar/device tiers).
fn medium_value(index: usize) -> Vec<u8> {
    let len = 2048 + (index % 3) * 1024;
    let mut state = (index as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1)
        | 1;
    let mut out = vec![0u8; len];
    for byte in &mut out {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = u8::try_from((state >> 11) & 0xFF).unwrap_or(0xFF);
    }
    out
}

fn put_checked(
    client: &kivi_client::NativeClient,
    model: &mut HashMap<Vec<u8>, Vec<u8>>,
    key: Vec<u8>,
    value: Vec<u8>,
) {
    client
        .set(&Key::from(key.clone()), Bytes::copy_from_slice(&value))
        .unwrap_or_else(|error| panic!("set failed: {error:?}"));
    model.insert(key, value);
}

fn batch_checked(
    client: &kivi_client::NativeClient,
    model: &mut HashMap<Vec<u8>, Vec<u8>>,
    writes: Vec<(Vec<u8>, Vec<u8>)>,
) {
    let specs: Vec<BatchWriteSpec> = writes
        .iter()
        .map(|(key, value)| BatchWriteSpec {
            key: key.clone(),
            kind: BatchWriteKind::Put(value.clone()),
            expect: BatchExpect::Any,
        })
        .collect();
    client
        .atomic_batch(NS, &specs)
        .unwrap_or_else(|error| panic!("batch failed: {error:?}"));
    for (key, value) in writes {
        model.insert(key, value);
    }
}

fn scan_all_keys(client: &kivi_client::NativeClient) -> BTreeSet<Vec<u8>> {
    let options = ScanOptions {
        direction: ScanDirection::Forward,
        consistency: ScanConsistency::LatestPerTablet,
        projection: ScanProjection::KeysOnly,
        max_items_per_page: 500,
        max_bytes_per_page: 1 << 20,
        limit: None,
        ..ScanOptions::default()
    };
    client
        .scan(NS, &options)
        .expect("scan serves")
        .into_iter()
        .map(|entry| entry.key)
        .collect()
}

#[test]
fn fabric_transparent_across_failover_and_restart() {
    let mut cluster = Cluster::spawn_ordered(4).expect("ordered cluster spawns");
    let mut model: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();
    let mut counters: HashMap<Vec<u8>, i64> = HashMap::new();

    // Phase 1: mixed load across all four ranges (small inline, medium
    // sidecar-tier, counters).
    let client = cluster.client();
    for index in 0..100usize {
        let key = spread_key(tablet_of(index), index);
        put_checked(
            &client,
            &mut model,
            key,
            format!("small-{index:06}").into_bytes(),
        );
    }
    for index in 0..150usize {
        let key = spread_key(tablet_of(index), 10_000 + index);
        put_checked(&client, &mut model, key, medium_value(index));
    }
    for index in 0..10usize {
        let key = spread_key(tablet_of(index), 50_000 + index);
        let live = client
            .counter_add(&Key::from(key.clone()), 1)
            .expect("counter commits");
        counters.insert(key, live);
    }

    // Phase 2: kill/restart cycle with traffic on both sides, plus one
    // cross-tablet medium batch and per-member snapshots.
    cluster.kill(1);
    cluster.restart(1).expect("member restarts");
    let _ = cluster.wait_all_leaders();
    cluster.wait_converged_all();
    let client = cluster.client();
    for index in 0..50usize {
        let key = spread_key(tablet_of(index + 1), 20_000 + index);
        put_checked(&client, &mut model, key, medium_value(1000 + index));
    }
    batch_checked(
        &client,
        &mut model,
        [0u8, 1, 2, 3]
            .iter()
            .enumerate()
            .map(|(slot, tablet)| {
                (
                    spread_key(*tablet, 30_000 + slot),
                    medium_value(2000 + slot),
                )
            })
            .collect(),
    );
    for tablet in 1..=4u64 {
        for index in 0..cluster.member_count() {
            if cluster.alive(index) {
                let _ = cluster.snapshot_tablet(index, tablet);
            }
        }
    }

    // Phase 3: full restart, then exact read-back of everything.
    cluster.kill_all();
    cluster.restart_all().expect("cluster restarts");
    cluster.wait_converged_all();
    let client = cluster.client();
    for (key, value) in &model {
        assert_eq!(
            client.get(&Key::from(key.clone())).expect("get"),
            Some(Bytes::copy_from_slice(value)),
            "key survives failover plus full restart"
        );
    }
    for (key, expected) in &counters {
        let live = client
            .counter_add(&Key::from(key.clone()), 0)
            .expect("counter reads");
        assert_eq!(&live, expected, "counter exact after restart");
    }
    let scanned = scan_all_keys(&client);
    let modelled: BTreeSet<Vec<u8>> = model.keys().cloned().collect();
    assert!(
        modelled.is_subset(&scanned),
        "every model key scans (model {}, scanned {})",
        modelled.len(),
        scanned.len()
    );
    assert_eq!(cluster.total_intents(), 0, "no stuck intents");
}
