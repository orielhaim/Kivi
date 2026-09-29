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
use kivi_client::ordered::{ScanConsistency, ScanDirection, ScanOptions, ScanProjection};
use kivi_lab::cluster::Cluster;
use kivi_lab::testkit::{NS, batch_checked, put_checked, spread_key, tablet_of};
use kivi_state::Key;

/// Deterministic incompressible bytes (medium tier must not compress
/// into the inline path or dodge the sidecar/device tiers).
fn incompressible_value(index: usize) -> Vec<u8> {
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
            "fabric",
        );
    }
    for index in 0..150usize {
        let key = spread_key(tablet_of(index), 10_000 + index);
        put_checked(
            &client,
            &mut model,
            key,
            incompressible_value(index),
            "fabric",
        );
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
        put_checked(
            &client,
            &mut model,
            key,
            incompressible_value(1000 + index),
            "fabric",
        );
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
                    incompressible_value(2000 + slot),
                )
            })
            .collect(),
        "fabric",
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

/// A value written with a plain `set` reaches the redundancy fabric.
///
/// The acceptance test above proves the cluster survives member loss. It cannot
/// prove *why*: a missing whole copy is only recoverable if the fragments
/// exist somewhere, and before the propose path handed its staged roots to the
/// plane, a plain `set` produced no fragments at all. The streaming path was
/// the only one that protected anything.
///
/// So this asks the fabric directly: after ordinary medium writes, are the
/// manifests in the published catalog? If this fails, the acceptance test is
/// passing for a reason unrelated to redundancy, and a real member loss will
/// still be unrecoverable.
#[test]
fn a_plain_medium_set_reaches_the_redundancy_fabric() {
    /// Medium writes issued; each stages one manifest that must be protected.
    const WRITES: usize = 12;

    let cluster = Cluster::spawn_ordered(3).expect("ordered cluster spawns");
    let client = cluster.client();
    let mut model: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();
    for index in 0..WRITES {
        let key = spread_key(tablet_of(index), 70_000 + index);
        put_checked(
            &client,
            &mut model,
            key,
            incompressible_value(5000 + index),
            "fabric-protect",
        );
    }
    cluster.wait_converged_all();

    // `AssetKind::ChunkManifest`'s discriminant, taken from the crate rather
    // than written as a literal: a reordering must break the build here rather
    // than quietly turn the filter into one that matches nothing and passes.
    let manifest_kind = u64::from(kivi_redundancy::AssetKind::ChunkManifest.as_u8());

    // Protection is asynchronous by design, so poll a bounded window rather
    // than sleeping a fixed amount. The wait counts *manifests*, not assets:
    // one write publishes a chunk and a manifest, so waiting on the total
    // crosses the threshold at about half the point the assertion needs, and
    // reads the catalog mid-protection.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut published: Vec<serde_json::Value> = Vec::new();
    let mut manifest_assets = 0usize;
    while std::time::Instant::now() < deadline {
        let (status, body) = cluster.admin_get(0, "/v1/redundancy/catalog");
        assert_eq!(status, 200, "GET /v1/redundancy/catalog");
        published = body["assets"]
            .as_array()
            .cloned()
            .unwrap_or_else(|| panic!("catalog has no assets array: {body}"));
        manifest_assets = published
            .iter()
            .filter(|entry| entry["kind"].as_u64() == Some(manifest_kind))
            .count();
        if manifest_assets >= WRITES {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    assert!(
        manifest_assets >= WRITES,
        "a plain medium set must reach the fabric, found {manifest_assets} of {WRITES} \
         published manifest assets: {published:?}"
    );
}
