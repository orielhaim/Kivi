//! Live fourth-node admission on an ordered cluster: one long-lived client
//! keeps serving traffic straight through admission, directory catch-up,
//! lineage churn, and rebalance — no client recreation, no rebalance
//! warmup, no `TooManyRedirects`, no routing black holes.
//!
//! Lifecycle model under test:
//!
//! * process reachable ≠ serving-ready: the joiner starts data-empty and
//!   must never be treated as a routing candidate merely for listening.
//!   Before it hosts anything it answers `StaleRoute` toward a founder
//!   (control desired when known, static founders otherwise) instead of
//!   `Overloaded`, so it can never burn a caller's redirect budget.
//! * control member: explicit admission (`/v1/control/nodes`) moves it
//!   `Joining → Active`; the mesh and control-learner paths converge
//!   without restarts.
//! * directory-aware: committed split/merge cutovers replay continuously
//!   on every node, so the joiner converges from genesis to the live
//!   tiling without any republish of past plans.
//! * tablet host: only an explicit migration (`move_tablet`) makes it
//!   serve a tablet, after learner catch-up through the normal plan flow.
//!
//! Every traffic operation must succeed first try on the pre-admission
//! client (whose seeds name only the three founders): any
//! `TooManyRedirects`, `Io("all known endpoints failed")`, or unexpected
//! error fails the test loudly with the tablet/leader state attached.

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

/// One traffic burst: point writes + spot reads + a cross-tablet batch +
/// a covering scan. Every op must succeed on the long-lived client; the
/// context names the admission phase for failure messages.
#[allow(clippy::too_many_lines)]
fn traffic_burst(
    cluster: &Cluster,
    client: &kivi_client::NativeClient,
    model: &mut HashMap<Vec<u8>, Vec<u8>>,
    base: usize,
    context: &str,
) {
    for index in 0..12usize {
        let key = spread_key(index, base + index);
        let value = format!("admit-{base}-{index:02}").into_bytes();
        client
            .set(&Key::from(key.clone()), Bytes::copy_from_slice(&value))
            .unwrap_or_else(|error| {
                panic!(
                    "{context}: set failed: {error:?} [{}]",
                    liveness(cluster, client)
                )
            });
        model.insert(key, value);
    }
    for index in (0..12usize).step_by(4) {
        let key = spread_key(index, base + index);
        let expected = model.get(&key).cloned().unwrap_or_default();
        let got = client
            .get(&Key::from(key.clone()))
            .unwrap_or_else(|error| {
                panic!(
                    "{context}: get failed: {error:?} [{}]",
                    liveness(cluster, client)
                )
            })
            .unwrap_or_default();
        assert_eq!(
            got.as_ref(),
            expected.as_slice(),
            "{context}: value mismatch"
        );
    }
    // Cross-tablet batch (spans three tablets → client 2PC): exercises
    // the known-decision recovery path under churn. A stale cached route
    // may probe a retired tablet while the server serves on its live
    // successor; the driver keeps served-elsewhere prepares in the abort
    // set, evicts the proven-stale routes, and converges on re-probe —
    // no intent is ever orphaned without a record or an owner.
    // (`txn_abort.rs` holds the deterministic abort regression.)
    let batch: Vec<BatchWriteSpec> = [0usize, 1, 3]
        .iter()
        .enumerate()
        .map(|(slot, tablet)| BatchWriteSpec {
            key: spread_key(*tablet, base + 900 + slot),
            kind: BatchWriteKind::Put(format!("admit-batch-{base}-{slot}").into_bytes()),
            expect: BatchExpect::Any,
        })
        .collect();
    let values: Vec<(Vec<u8>, Vec<u8>)> = batch
        .iter()
        .map(|spec| {
            (
                spec.key.clone(),
                match &spec.kind {
                    BatchWriteKind::Put(value) => value.clone(),
                    _ => Vec::new(),
                },
            )
        })
        .collect();
    // A batch that lands on a just-split/merged directory can lose the
    // OCC race transiently (`TxnConflict`) even though nothing semantic
    // conflicts: the probe planned against a retired tablet. Redrive with
    // a fresh transaction id after convergence (same contract as the
    // acceptance driver); persistent conflicts fail loudly with per-key
    // probes attached.
    let mut attempt = 0u32;
    loop {
        match client.atomic_batch(NS, &batch) {
            Ok(_) => break,
            Err(kivi_client::ClientError::TxnConflict) if attempt < 5 => {
                attempt += 1;
                cluster.wait_converged_all();
            }
            Err(error) => {
                let probes: Vec<String> = batch
                    .iter()
                    .map(|spec| format!("{:02x?}", &spec.key[..1]))
                    .collect();
                // Single-key probe batch per key: distinguishes a wedged
                // key (foreign intent → Conflict on exactly that key) from
                // an emergent coordinator/decide failure (all singles Ok).
                let mut singles = Vec::new();
                for spec in &batch {
                    let single = [BatchWriteSpec {
                        key: spec.key.clone(),
                        kind: BatchWriteKind::Put(b"admit-probe".to_vec()),
                        expect: BatchExpect::Any,
                    }];
                    singles.push(match client.atomic_batch(NS, &single) {
                        Ok(_) => "Ok".to_owned(),
                        Err(error) => format!("{error:?}"),
                    });
                }
                panic!(
                    "{context}: batch failed: {error:?} [{}] tries={} keys={:?} singles={:?}",
                    liveness(cluster, client),
                    attempt + 1,
                    probes,
                    singles,
                );
            }
        }
    }
    for (key, value) in values {
        model.insert(key, value);
    }
    let options = ScanOptions {
        direction: ScanDirection::Forward,
        consistency: ScanConsistency::LatestPerTablet,
        projection: ScanProjection::KeysOnly,
        max_items_per_page: 100,
        max_bytes_per_page: 1 << 20,
        limit: None,
        ..ScanOptions::default()
    };
    let scanned: BTreeSet<Vec<u8>> = client
        .scan(NS, &options)
        .unwrap_or_else(|error| {
            panic!(
                "{context}: scan failed: {error:?} [{}]",
                liveness(cluster, client)
            )
        })
        .into_iter()
        .map(|entry| entry.key)
        .collect();
    // Coverage every burst: a key that vanishes between bursts pinpoints
    // the exact loss window (post-churn, migration, rebalance) instead of
    // surfacing only at final equivalence.
    let missing_cover: Vec<String> = model
        .keys()
        .filter(|key| !scanned.contains(*key))
        .take(8)
        .map(|key| hex_key(key))
        .collect();
    let missing_total = model.keys().filter(|key| !scanned.contains(*key)).count();
    assert!(
        missing_total == 0,
        "{context}: scan lost {missing_total} model keys (e.g. {missing_cover:?}) [{}]",
        liveness(cluster, client),
    );
}

fn liveness(cluster: &Cluster, client: &kivi_client::NativeClient) -> String {
    let mut parts = Vec::new();
    for index in 0..cluster.member_count() {
        if !cluster.alive(index) {
            parts.push(format!("m{index}:down"));
            continue;
        }
        match cluster.tablets_status_opt(index) {
            Some(status) => {
                let entries = status.as_array().cloned().unwrap_or_default();
                let mut holding: Vec<String> = Vec::new();
                let mut member_intents = 0u64;
                let mut roles: Vec<String> = Vec::new();
                for entry in &entries {
                    let count = entry["intent_count"].as_u64().unwrap_or(0);
                    member_intents += count;
                    if count > 0 {
                        let keys = entry["intent_keys"]
                            .as_array()
                            .cloned()
                            .unwrap_or_default()
                            .iter()
                            .map(|intent| {
                                format!(
                                    "{}@{}:{}s",
                                    intent["key"].as_str().unwrap_or("?"),
                                    intent["coordinator"].as_u64().unwrap_or(0),
                                    intent["age_micros"].as_u64().unwrap_or(0) / 1_000_000,
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(",");
                        holding.push(format!(
                            "{}:{}[{}]",
                            entry["group"].as_u64().unwrap_or(u64::MAX),
                            count,
                            keys,
                        ));
                    }
                    roles.push(format!(
                        "{}:{}/{}/t{}",
                        entry["group"].as_u64().unwrap_or(u64::MAX),
                        entry["role"].as_str().unwrap_or("?"),
                        entry["leader"]
                            .as_u64()
                            .map_or("none".to_owned(), |l| l.to_string()),
                        entry["term"].as_u64().unwrap_or(0),
                    ));
                }
                parts.push(format!(
                    "m{index}:up({}t,{}i[{}],{})",
                    entries.len(),
                    member_intents,
                    holding.join(","),
                    roles.join(" ")
                ));
            }
            None => parts.push(format!("m{index}:up(?)")),
        }
    }
    let stats = client.stats();
    format!(
        "{} req={} redir={} errs={}",
        parts.join(" "),
        stats.requests,
        stats.redirects,
        stats.errors
    )
}

/// Short hex for failure messages (first bytes only).
fn hex_key(key: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for byte in key.iter().take(8) {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Per-member Raft state for one tablet group: role/term/leader plus
/// log positions — distinguishes "no leader yet" from "divergent logs"
/// when a key goes missing after migration.
fn tablet_state_dump(cluster: &Cluster, tablet: u64) -> String {
    let mut parts = Vec::new();
    for index in 0..cluster.member_count() {
        if !cluster.alive(index) {
            parts.push(format!("m{index}:down"));
            continue;
        }
        let row = cluster.tablets_status_opt(index).and_then(|status| {
            status
                .as_array()
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .find(|entry| {
                    entry.get("group").and_then(serde_json::Value::as_u64) == Some(tablet)
                })
        });
        match row {
            Some(entry) => parts.push(format!(
                "m{index}:role={} leader={} term={} last_log={} committed={} applied={}",
                entry["role"].as_str().unwrap_or("?"),
                entry["leader"]
                    .as_u64()
                    .map_or("none".to_owned(), |l| l.to_string()),
                entry["term"].as_u64().unwrap_or(0),
                entry["last_log"].as_u64().unwrap_or(0),
                entry["committed"].as_u64().unwrap_or(0),
                entry["applied"].as_u64().unwrap_or(0),
            )),
            None => parts.push(format!("m{index}:no-group")),
        }
    }
    parts.join(" ")
}

/// Active tablet set + directory version of one member's serving
/// directory (data-empty joiners report a version too: convergence is
/// defined over directories, not hosted groups).
fn directory_view(cluster: &Cluster, index: usize) -> (u64, Vec<u64>) {
    let directory = cluster.directory(index);
    let version = directory
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let mut active: Vec<u64> = directory
        .get("tablets")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter(|tablet| tablet.get("state").and_then(serde_json::Value::as_str) == Some("Active"))
        .filter_map(|tablet| tablet.get("tablet").and_then(serde_json::Value::as_u64))
        .collect();
    active.sort_unstable();
    (version, active)
}

/// Waits until every live member serves the same directory: equal
/// versions and equal active tablet sets, stable across polls. This is
/// the admission convergence predicate — it holds for data-empty members
/// (genesis + replayed cutovers) as well as tablet hosts.
fn wait_directory_agreed(cluster: &Cluster, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    let mut last: Vec<(u64, Vec<u64>)> = Vec::new();
    let mut stable = 0u32;
    loop {
        let mut views = Vec::new();
        for index in 0..cluster.member_count() {
            if !cluster.alive(index) {
                continue;
            }
            views.push(directory_view(cluster, index));
        }
        if !views.is_empty() && views.iter().all(|view| *view == views[0]) {
            if views == last {
                stable += 1;
                if stable >= 2 {
                    return;
                }
            } else {
                stable = 0;
                last = views;
            }
        } else {
            stable = 0;
        }
        assert!(
            Instant::now() < deadline,
            "serving directories never agreed"
        );
        std::thread::sleep(Duration::from_millis(500));
    }
}

#[allow(clippy::too_many_lines)]
#[test]
fn live_fourth_node_admission_without_traffic_loss() {
    let mut cluster = Cluster::spawn_ordered(4).expect("ordered cluster spawns");
    let _ = cluster.set_policy(&serde_json::json!({
        "auto_split": false,
        "auto_merge": false,
    }));

    // Seeds name only the three founders; this client never changes.
    let client = cluster.client();
    let mut model: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();
    traffic_burst(&cluster, &client, &mut model, 0, "pre-admission");

    // Admit the fourth process: data-empty, no tablets, no warmup.
    let fourth = cluster.add_node();
    assert_eq!(fourth, 3, "fourth member slot");
    cluster.wait_control_nodes(4, Duration::from_secs(90));

    // Traffic straight through admission: the empty member must neither
    // attract routes nor poison them.
    for round in 0..4usize {
        traffic_burst(
            &cluster,
            &client,
            &mut model,
            10_000 + round * 1_000,
            &format!("admitting-{round}"),
        );
    }

    // The joiner converges its serving directory via continuous replay
    // (genesis + every committed cutover, no republish needed).
    wait_directory_agreed(&cluster, Duration::from_secs(300));
    traffic_burst(&cluster, &client, &mut model, 20_000, "admitted");

    // Lineage churn with four members present: split one tablet and merge
    // it back; the joiner must track the cutovers like a founder.
    let tablets: Vec<u64> = cluster
        .tablet_ids(cluster.live_index())
        .into_iter()
        .filter(|id| *id != 0)
        .collect();
    let parent = *tablets.iter().max().expect("a tablet to split");
    let reply = cluster.split_tablet(parent);
    assert_eq!(
        reply.get("ok").and_then(serde_json::Value::as_bool),
        Some(true),
        "split accepted: {reply}"
    );
    let left = reply
        .get("left")
        .and_then(serde_json::Value::as_u64)
        .expect("split left child");
    let right = reply
        .get("right")
        .and_then(serde_json::Value::as_u64)
        .expect("split right child");
    cluster.wait_splits_done(Duration::from_secs(600));
    traffic_burst(&cluster, &client, &mut model, 30_000, "split");
    let merge = cluster.merge_tablets(left, right);
    assert_eq!(
        merge.get("ok").and_then(serde_json::Value::as_bool),
        Some(true),
        "merge accepted: {merge}"
    );
    cluster.wait_merges_done(Duration::from_secs(600));
    wait_directory_agreed(&cluster, Duration::from_secs(300));
    traffic_burst(&cluster, &client, &mut model, 40_000, "merged");

    // Rebalance one tablet replica onto the new member through the
    // persisted migration pathway, with traffic online throughout. The
    // moved tablet must be live: the split parent above was merged away
    // (its children became the merged tablet), so re-read the active set
    // and move the newest (highest-id) live tablet.
    let moved: u64 = cluster
        .tablet_ids(cluster.live_index())
        .into_iter()
        .filter(|id| *id != 0)
        .max()
        .expect("a live tablet to move");
    let _ = cluster.move_tablet(moved, 1, 4);
    traffic_burst(&cluster, &client, &mut model, 50_000, "migrating");
    cluster.wait_tablet_voters(moved, &[2, 3, 4], Duration::from_secs(240));
    traffic_burst(&cluster, &client, &mut model, 60_000, "migrated-voters");
    cluster.wait_migrations_done(Duration::from_secs(240));
    cluster.wait_converged_all();
    wait_directory_agreed(&cluster, Duration::from_secs(300));
    traffic_burst(&cluster, &client, &mut model, 70_000, "rebalanced");

    // The new member actually hosts the moved tablet now.
    let groups: Vec<u64> = cluster
        .tablets_status(fourth)
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter_map(|tablet| tablet.get("group").and_then(serde_json::Value::as_u64))
        .collect();
    assert!(
        groups.contains(&moved),
        "node 4 hosts moved tablet {moved}: {groups:?}"
    );

    // Final equivalence on the same client: every model key reads back,
    // the covering scan matches exactly, no stuck intents. Missing keys
    // dump the moved tablet's per-member Raft state (data loss through
    // migration is the highest-severity failure this test can catch).
    let mut missing: Vec<String> = Vec::new();
    for (key, expected) in &model {
        let got = client
            .get(&Key::from(key.clone()))
            .unwrap_or_else(|error| panic!("final get failed: {error:?}"));
        if got.as_deref() != Some(expected.as_slice()) {
            missing.push(hex_key(key));
        }
    }
    assert!(
        missing.is_empty(),
        "final equivalence lost {} keys (e.g. {:?}) [{}] moved-tablet state: {}",
        missing.len(),
        &missing[..missing.len().min(8)],
        liveness(&cluster, &client),
        tablet_state_dump(&cluster, moved),
    );
    let options = ScanOptions {
        direction: ScanDirection::Forward,
        consistency: ScanConsistency::LatestPerTablet,
        projection: ScanProjection::KeysOnly,
        max_items_per_page: 100,
        max_bytes_per_page: 1 << 20,
        limit: None,
        ..ScanOptions::default()
    };
    let mut got_sorted: Vec<Vec<u8>> = client
        .scan(NS, &options)
        .expect("final scan works")
        .into_iter()
        .map(|entry| entry.key)
        .collect();
    got_sorted.sort();
    let mut expected_sorted: Vec<Vec<u8>> = model.keys().cloned().collect();
    expected_sorted.sort();
    let scanned_set: BTreeSet<&Vec<u8>> = got_sorted.iter().collect();
    assert!(
        model.keys().all(|key| scanned_set.contains(key)),
        "final scan covers the model"
    );
    assert_eq!(got_sorted, expected_sorted, "final scan matches model");
    assert_eq!(cluster.total_intents(), 0, "no stuck intents");
    eprintln!(
        "admission: 4th node joined, tracked churn, hosts tablet {moved}; {} keys verified",
        model.len()
    );
}
