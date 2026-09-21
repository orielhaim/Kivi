//! Native-plane rejoin lifecycle: one long-lived client survives repeated
//! kill/restart cycles with no client recreation and no fixed settle sleeps.
//!
//! Root-cause context: the old "native plane goes deaf after repeated
//! kill/restart cycles" failure was substantially explained by the restart
//! directory-replay bug (eligible splits replayed before merges, so a
//! restarted member permanently diverged its serving directory and produced
//! contradictory tombstone redirects). This test proves whether any
//! independent restart/incarnation bug remains: 10 kill/restart cycles
//! across all members with continuous mixed traffic on one client.
//!
//! Serving-ready definition used here (not just "leaders exist"): the
//! restarted member's incarnation advanced past its pre-kill value, every
//! live member agrees on the exact active tiling (stable across polls), all
//! tablets have one stable leader each, per-tablet applied indexes converge
//! across live members, the restarted member's directory version is at or
//! past its pre-kill floor, and mixed traffic (point reads/writes,
//! batches, scans, streaming values) succeeds on the same pooled client.

use std::collections::{BTreeSet, HashMap};
use std::time::{Duration, Instant};

use bytes::Bytes;
use kivi_client::ordered::{
    BatchExpect, BatchWriteKind, BatchWriteSpec, ScanConsistency, ScanDirection, ScanOptions,
    ScanProjection,
};
use kivi_lab::cluster::Cluster;
use kivi_state::Key;
use kivi_types::NamespaceId;

const NS: NamespaceId = NamespaceId::from_u64(1);
const CYCLES: usize = 10;
const SEED_KEYS: usize = 40;
const PER_PHASE_WRITES: usize = 8;

fn spread_key(slot: usize, index: usize) -> Vec<u8> {
    let prefix = match slot % 4 {
        0 => 0x10u8,
        1 => b'A',
        2 => 0x90u8,
        _ => 0xF0u8,
    };
    let mut key = vec![prefix];
    key.extend_from_slice(format!("{index:06}").as_bytes());
    key
}

fn value_for(index: usize) -> Vec<u8> {
    format!("rejoin-{index:06}").into_bytes()
}

fn scan_keys(client: &kivi_client::NativeClient) -> Vec<Vec<u8>> {
    let options = ScanOptions {
        direction: ScanDirection::Forward,
        consistency: ScanConsistency::LatestPerTablet,
        projection: ScanProjection::KeysOnly,
        max_items_per_page: 100,
        max_bytes_per_page: 1 << 20,
        limit: None,
        ..ScanOptions::default()
    };
    client
        .scan(NS, &options)
        .expect("covering scan works")
        .into_iter()
        .map(|entry| entry.key)
        .collect()
}

fn write_checked(
    client: &kivi_client::NativeClient,
    model: &mut HashMap<Vec<u8>, Vec<u8>>,
    key: Vec<u8>,
    value: Vec<u8>,
    context: &str,
) {
    client
        .set(&Key::from(key.clone()), Bytes::copy_from_slice(&value))
        .unwrap_or_else(|error| panic!("{context}: set failed: {error:?}"));
    model.insert(key, value);
}

fn read_checked(
    client: &kivi_client::NativeClient,
    model: &HashMap<Vec<u8>, Vec<u8>>,
    key: &[u8],
    context: &str,
) {
    let got = client
        .get(&Key::from(key.to_vec()))
        .unwrap_or_else(|error| panic!("{context}: get failed: {error:?}"));
    assert_eq!(
        got.as_deref(),
        model.get(key).map(Vec::as_slice),
        "{context}: value mismatch"
    );
}

fn batch_checked(
    client: &kivi_client::NativeClient,
    model: &mut HashMap<Vec<u8>, Vec<u8>>,
    writes: Vec<(Vec<u8>, Vec<u8>)>,
    context: &str,
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
        .unwrap_or_else(|error| panic!("{context}: batch failed: {error:?}"));
    for (key, value) in writes {
        model.insert(key, value);
    }
}

fn stream_checked(
    client: &kivi_client::NativeClient,
    model: &mut HashMap<Vec<u8>, Vec<u8>>,
    key: &[u8],
    bytes: &[u8],
    context: &str,
) {
    // NOTE (found bug, deferred): `put_stream` uploads do not follow
    // `StaleRoute` at commit — the commit is single-attempt, so a
    // leadership/topology move between route-probe and commit surfaces as
    // `Internal("unexpected stale-route here")` instead of converging
    // (reproduced on a fresh 4-tablet ordered cluster, no churn). The
    // streaming *download* path (`get_stream`) does follow redirects, so
    // this helper writes through the redirect-following point path and
    // reads back through both `get` and the streaming download path.
    // Full upload-redirect (with source replay) belongs to the
    // Sidecar/streaming lifecycle work, not this rejoin regression.
    write_checked(client, model, key.to_vec(), bytes.to_vec(), context);
    let back = client
        .get_stream(&Key::from(key.to_vec()))
        .unwrap_or_else(|error| panic!("{context}: get_stream failed: {error:?}"))
        .expect("streamed value present");
    assert_eq!(back.as_ref(), bytes, "{context}: stream mismatch");
}

fn traffic_phase(
    client: &kivi_client::NativeClient,
    model: &mut HashMap<Vec<u8>, Vec<u8>>,
    base: usize,
    context: &str,
) {
    for index in 0..PER_PHASE_WRITES {
        let key = spread_key(index, base + index);
        write_checked(client, model, key, value_for(base + index), context);
    }
    // Read back a deterministic subset (proves the serving path, not just ack).
    for index in (0..PER_PHASE_WRITES).step_by(3) {
        let key = spread_key(index, base + index);
        read_checked(client, model, &key, context);
    }
    let batch: Vec<(Vec<u8>, Vec<u8>)> = [0usize, 1, 3]
        .iter()
        .enumerate()
        .map(|(slot, tablet)| {
            (
                spread_key(*tablet, base + 500 + slot),
                value_for(base + 500 + slot),
            )
        })
        .collect();
    batch_checked(client, model, batch, context);
}

fn directory_version(cluster: &Cluster, index: usize) -> u64 {
    cluster
        .directory(index)
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0)
}

#[test]
fn rejoin_lifecycle_survives_ten_restart_cycles() {
    let mut cluster = Cluster::spawn_ordered(4).expect("ordered cluster spawns");
    let _ = cluster.wait_all_leaders();
    let _ = cluster.set_policy(&serde_json::json!({
        "auto_split": false,
        "auto_merge": false,
    }));

    // One long-lived client for the entire run: pooled connections are
    // reused across every kill/restart; no recreation, no warmup.
    let client = cluster.client();
    let mut model: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();

    // Seed state across all four tablets plus one streamed medium value.
    for index in 0..SEED_KEYS {
        let key = spread_key(index, index);
        write_checked(&client, &mut model, key, value_for(index), "seed");
    }
    let stream_key = b"\x10rejoin-stream".to_vec();
    let stream_bytes: Vec<u8> = (0..32 * 1024)
        .map(|i| u8::try_from(i % 251).unwrap_or(0xFF))
        .collect();
    stream_checked(
        &client,
        &mut model,
        &stream_key,
        &stream_bytes,
        "seed-stream",
    );
    cluster.wait_converged_all();
    cluster.wait_tiling_agreed(4, Instant::now() + Duration::from_secs(300));

    for cycle in 0..CYCLES {
        let victim = cycle % cluster.member_count();
        let incarnation_before = cluster.incarnation(victim);
        let dir_before = directory_version(&cluster, victim);
        assert!(incarnation_before > 0, "incarnation present before kill");

        cluster.kill(victim);
        // Steady traffic continues against the surviving quorum on the
        // same client: proves the kill alone never strands the pool.
        let down_base = 100_000 + cycle * 1_000;
        traffic_phase(
            &client,
            &mut model,
            down_base,
            &format!("cycle{cycle}-down"),
        );

        cluster
            .restart(victim)
            .unwrap_or_else(|error| panic!("node {victim} restarts: {error:?}"));

        // Semantic rejoin readiness via the one coherent predicate:
        // incarnation fencing, stable leaders, converged applied indexes,
        // exact tiling agreement, and the replay version floor. Never
        // "process is listening" and never a fixed sleep.
        cluster.wait_member_rejoined(
            victim,
            incarnation_before,
            dir_before,
            4,
            Duration::from_secs(300),
        );

        // Post-rejoin traffic on the same client: proves the restarted
        // member actually serves and the pool recovered (same endpoint
        // string, new incarnation behind it).
        let up_base = 200_000 + cycle * 1_000;
        traffic_phase(&client, &mut model, up_base, &format!("cycle{cycle}-up"));
        // Stream round-trip every third cycle (sidecar/chunk path rejoins too).
        if cycle % 3 == 0 {
            let key = format!("rejoin-cycle-{cycle:02}").into_bytes();
            let bytes: Vec<u8> = (0..16 * 1024)
                .map(|i| u8::try_from((i + cycle) % 251).unwrap_or(0xFF))
                .collect();
            stream_checked(
                &client,
                &mut model,
                &key,
                &bytes,
                &format!("cycle{cycle}-stream"),
            );
        }
        // Scan covers everything written so far: no loss across the rejoin.
        let scanned: BTreeSet<Vec<u8>> = scan_keys(&client).into_iter().collect();
        for key in model.keys() {
            assert!(
                scanned.contains(key),
                "cycle {cycle}: model key missing from scan after rejoin"
            );
        }
    }

    // Final equivalence: every model key reads back exactly, scan matches.
    for (key, expected) in &model {
        let got = client
            .get(&Key::from(key.clone()))
            .expect("final get works");
        assert_eq!(
            got.as_deref(),
            Some(expected.as_slice()),
            "final equivalence"
        );
    }
    let mut got_sorted = scan_keys(&client);
    got_sorted.sort();
    let mut expected_sorted: Vec<Vec<u8>> = model.keys().cloned().collect();
    expected_sorted.sort();
    assert_eq!(got_sorted, expected_sorted, "final scan matches model");
    assert_eq!(cluster.total_intents(), 0, "no stuck intents");
    eprintln!("rejoin: {CYCLES} restart cycles verified on one client");
    cluster.kill_all();
}
