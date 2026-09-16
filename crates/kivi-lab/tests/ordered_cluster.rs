//! Ordered data-plane money tests over real 3-node Raft clusters:
//! cross-tablet range scans under splits/failover/restarts, atomic
//! batches with all-or-none commit, secondary indexes with uniqueness,
//! and chained ordered splits with cutover timing.
//!
//! Everything runs through the normal native server/client paths (no
//! test-only state).

use std::time::{Duration, Instant};

use bytes::Bytes;
use kivi_client::ordered::{
    BatchExpect, BatchWriteKind, BatchWriteSpec, ClientScanValue, IndexTermSet, ScanConsistency,
    ScanDirection, ScanOptions, ScanProjection,
};
use kivi_lab::cluster::Cluster;
use kivi_state::{IndexId, IndexKind, Key};
use kivi_types::NamespaceId;

const NS: NamespaceId = NamespaceId::from_u64(1);

/// Keys spread over the genesis tiling (4 tablets: [0,64), [64,128),
/// [128,192), [192,+inf)): first-byte prefixes per tablet.
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

fn put(client: &kivi_client::NativeClient, key: &[u8], value: &[u8]) {
    client
        .set(&Key::from(key.to_vec()), Bytes::copy_from_slice(value))
        .expect("set works");
}

fn full_scan(client: &kivi_client::NativeClient) -> Vec<Vec<u8>> {
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
        .expect("scan works")
        .into_iter()
        .map(|entry| entry.key)
        .collect()
}

fn assert_sorted_unique(keys: &[Vec<u8>]) {
    for pair in keys.windows(2) {
        assert!(pair[0] < pair[1], "scan strictly ordered");
    }
}

#[test]
fn ordered_range_scan_money() {
    let cluster = Cluster::spawn_ordered(4).expect("ordered cluster spawns");
    let _ = cluster.wait_all_leaders();
    let client = cluster.client();
    // 1200 keys spread over all four tablets.
    let mut expected: Vec<Vec<u8>> = Vec::new();
    for tablet in 0..4u8 {
        for index in 0..300usize {
            let key = spread_key(tablet, index);
            put(&client, &key, b"v");
            expected.push(key);
        }
    }
    expected.sort();
    // Full forward scan: complete and strictly ordered.
    let got = full_scan(&client);
    // Filter system keys (transaction machinery may add none here, but be
    // explicit: user scans skip `\xff` anyway server-side).
    assert_eq!(got.len(), expected.len(), "no omissions");
    assert_eq!(got, expected, "no duplicates, strict order");
    assert_sorted_unique(&got);
    // Full reverse scan: exact reverse.
    let options = ScanOptions {
        direction: ScanDirection::Reverse,
        consistency: ScanConsistency::LatestPerTablet,
        projection: ScanProjection::KeysOnly,
        max_items_per_page: 100,
        max_bytes_per_page: 1 << 20,
        limit: None,
        ..ScanOptions::default()
    };
    let mut reverse = client
        .scan(NS, &options)
        .expect("reverse scan works")
        .into_iter()
        .map(|entry| entry.key)
        .collect::<Vec<_>>();
    reverse.reverse();
    assert_eq!(reverse, expected);
    // Prefix/sub-range scan with values.
    let prefix = vec![b'A'];
    let mut end = prefix.clone();
    end.push(0xFF);
    let options = ScanOptions {
        start: Some(prefix),
        end: Some(end),
        direction: ScanDirection::Forward,
        consistency: ScanConsistency::LatestPerTablet,
        projection: ScanProjection::KeysAndValues,
        max_items_per_page: 50,
        max_bytes_per_page: 1 << 20,
        limit: None,
    };
    let entries = client.scan(NS, &options).expect("prefix scan works");
    assert_eq!(entries.len(), 300);
    for entry in &entries {
        assert!(matches!(entry.value, ClientScanValue::Inline(_)));
    }
    assert_eq!(cluster.total_intents(), 0, "no stray intents");
}

#[test]
fn ordered_scan_survives_split_during_scan() {
    let cluster = Cluster::spawn_ordered(4).expect("ordered cluster spawns");
    let _ = cluster.wait_all_leaders();
    let client = cluster.client();
    let mut expected: Vec<Vec<u8>> = Vec::new();
    for tablet in 0..4u8 {
        for index in 0..200usize {
            let key = spread_key(tablet, index);
            put(&client, &key, b"v");
            expected.push(key);
        }
    }
    expected.sort();
    // Baseline: exact full set, no concurrent writes anywhere below.
    assert_eq!(full_scan(&client), expected);
    // Continuous background scans while the main thread splits every
    // tablet: with no concurrent writes every scan must equal the exact
    // full set (LatestPerTablet is exact absent writes).
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_scan = std::sync::Arc::clone(&stop);
    let scanner = cluster.client();
    let expected_scan = expected.clone();
    let scans = std::thread::spawn(move || {
        let mut completed = 0usize;
        while !stop_scan.load(std::sync::atomic::Ordering::Relaxed) {
            let got = full_scan(&scanner);
            assert_sorted_unique(&got);
            assert_eq!(got.len(), expected_scan.len(), "scan omission under split");
            assert_eq!(got, expected_scan, "scan mismatch under split");
            completed += 1;
        }
        completed
    });
    // Split every genesis tablet (median pivots), waiting for each.
    let tablets: Vec<u64> = cluster
        .tablet_ids(cluster.live_index())
        .into_iter()
        .filter(|id| *id != 0)
        .collect();
    for tablet in &tablets {
        let reply = cluster.split_tablet(*tablet);
        assert_eq!(
            reply.get("ok").and_then(serde_json::Value::as_bool),
            Some(true),
            "split accepted: {reply}"
        );
    }
    cluster.wait_splits_done(Duration::from_secs(600));
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let completed = scans.join().expect("scanner joins");
    assert!(completed >= 1, "scans completed during splits");
    // Post-split: exact set again, coverage tiles without gaps.
    assert_eq!(full_scan(&client), expected);
    // Directory cutover converges per node after plan completion:
    // poll for the doubled width with an exact-coverage check.
    let deadline = Instant::now() + Duration::from_secs(300);
    let sorted = loop {
        let ranges = cluster.ordered_ranges(cluster.live_index());
        if ranges.len() >= 8 {
            let mut sorted = ranges;
            sorted.sort_by(|left, right| left.1.cmp(&right.1));
            let complete = sorted[0].1.is_empty()
                && sorted.last().expect("ranges").2.is_none()
                && sorted
                    .windows(2)
                    .all(|pair| pair[0].2.as_deref() == Some(pair[1].1.as_slice()));
            if complete {
                break sorted;
            }
        }
        assert!(
            Instant::now() < deadline,
            "ordered coverage never converged after splits"
        );
        std::thread::sleep(Duration::from_millis(500));
    };
    assert!(sorted[0].1.is_empty(), "coverage starts at empty");
    assert!(
        sorted.last().expect("ranges").2.is_none(),
        "coverage ends at +inf"
    );
    for pair in sorted.windows(2) {
        assert_eq!(
            pair[0].2.as_deref(),
            Some(pair[1].1.as_slice()),
            "adjacent ranges share exact boundaries"
        );
    }
    assert_eq!(cluster.total_intents(), 0, "no stray intents");
}

#[test]
fn cross_tablet_batch_money() {
    let cluster = Cluster::spawn_ordered(4).expect("ordered cluster spawns");
    let _ = cluster.wait_all_leaders();
    let client = cluster.client();
    // Two-tablet batch.
    let result = client
        .atomic_batch(
            NS,
            &[
                BatchWriteSpec {
                    key: spread_key(0, 1),
                    kind: BatchWriteKind::Put(b"two-a".to_vec()),
                    expect: BatchExpect::Any,
                },
                BatchWriteSpec {
                    key: spread_key(1, 1),
                    kind: BatchWriteKind::Put(b"two-b".to_vec()),
                    expect: BatchExpect::Any,
                },
            ],
        )
        .expect("two-tablet batch commits");
    assert_eq!(result.versions.len(), 2);
    // Four-tablet batch with a counter.
    let result = client
        .atomic_batch(
            NS,
            &[
                BatchWriteSpec {
                    key: spread_key(0, 2),
                    kind: BatchWriteKind::Put(b"four-a".to_vec()),
                    expect: BatchExpect::Any,
                },
                BatchWriteSpec {
                    key: spread_key(1, 2),
                    kind: BatchWriteKind::Put(b"four-b".to_vec()),
                    expect: BatchExpect::Any,
                },
                BatchWriteSpec {
                    key: spread_key(2, 2),
                    kind: BatchWriteKind::Delete,
                    expect: BatchExpect::Any,
                },
                BatchWriteSpec {
                    key: spread_key(3, 2),
                    kind: BatchWriteKind::CounterAdd(7),
                    expect: BatchExpect::Any,
                },
            ],
        )
        .expect("four-tablet batch commits");
    assert_eq!(result.versions.len(), 4);
    // Conflict aborts everything: stale version on one key.
    let version = client
        .get_version(NS, &spread_key(0, 1))
        .expect("version works")
        .expect("present");
    let error = client
        .atomic_batch(
            NS,
            &[
                BatchWriteSpec {
                    key: spread_key(0, 1),
                    kind: BatchWriteKind::Put(b"lost".to_vec()),
                    expect: BatchExpect::Version(version + 50),
                },
                BatchWriteSpec {
                    key: spread_key(3, 99),
                    kind: BatchWriteKind::Put(b"lost-too".to_vec()),
                    expect: BatchExpect::Any,
                },
            ],
        )
        .expect_err("stale batch conflicts");
    assert!(
        matches!(error, kivi_client::ClientError::TxnConflict),
        "got {error:?}"
    );
    assert_eq!(
        client.get(&Key::from(spread_key(0, 1))).expect("get works"),
        Some(Bytes::from("two-a")),
        "aborted batch applied nothing"
    );
    assert_eq!(
        client
            .get(&Key::from(spread_key(3, 99)))
            .expect("get works"),
        None
    );
    assert_eq!(cluster.total_intents(), 0, "no stuck intents");
}

#[test]
fn batch_survives_node_kill_and_rejoins() {
    let mut cluster = Cluster::spawn_ordered(4).expect("ordered cluster spawns");
    let _ = cluster.wait_all_leaders();
    let client = cluster.client();
    for index in 0..50usize {
        put(&client, &spread_key(0, index), b"pre");
        put(&client, &spread_key(3, index), b"pre");
    }
    // Kill one member mid-workload; batches keep committing through the
    // survivors, then the member rejoins.
    let victim = 1usize;
    cluster.kill(victim);
    for index in 50..100usize {
        let result = client
            .atomic_batch(
                NS,
                &[
                    BatchWriteSpec {
                        key: spread_key(0, index),
                        kind: BatchWriteKind::Put(b"post".to_vec()),
                        expect: BatchExpect::Any,
                    },
                    BatchWriteSpec {
                        key: spread_key(3, index),
                        kind: BatchWriteKind::Put(b"post".to_vec()),
                        expect: BatchExpect::Any,
                    },
                ],
            )
            .expect("batch commits during outage");
        assert_eq!(result.versions.len(), 2);
    }
    cluster.restart(victim).expect("member rejoins");
    let _ = cluster.wait_all_leaders();
    for index in 0..100usize {
        let expected = if index < 50 { "pre" } else { "post" };
        assert_eq!(
            client
                .get(&Key::from(spread_key(0, index)))
                .expect("get works"),
            Some(Bytes::from(expected)),
            "all-or-none held across kill"
        );
    }
    assert_eq!(cluster.total_intents(), 0, "no stuck intents");
}

#[test]
fn secondary_index_cluster_money() {
    let mut cluster = Cluster::spawn_ordered(4).expect("ordered cluster spawns");
    let _ = cluster.wait_all_leaders();
    let client = cluster.client();
    let term = |index: u64, kind: IndexKind, terms: &[&str]| IndexTermSet {
        index: IndexId::from_u64(index),
        kind,
        terms: terms.iter().map(|term| term.as_bytes().to_vec()).collect(),
    };
    // 40 indexed records across tablets.
    for index in 0..40usize {
        let tablet = u8::try_from(index % 4).expect("small tablet selector");
        let key = spread_key(tablet, 1000 + index);
        let city = match index % 3 {
            0 => "berlin",
            1 => "oslo",
            _ => "zurich",
        };
        client
            .indexed_set(
                NS,
                &key,
                format!("profile:{index}").into_bytes(),
                &[term(1, IndexKind::NonUnique, &[city])],
            )
            .expect("indexed set works");
    }
    let berlin = client
        .index_equal(NS, IndexId::from_u64(1), b"berlin", 100)
        .expect("equality works");
    assert_eq!(berlin.len(), 14, "berlin has 14 primaries (0,3,...,39)");
    // Split every tablet while querying: results stay version-valid.
    let tablets: Vec<u64> = cluster
        .tablet_ids(cluster.live_index())
        .into_iter()
        .filter(|id| *id != 0)
        .collect();
    for tablet in &tablets {
        let _ = cluster.split_tablet(*tablet);
    }
    let oslo = client
        .index_range(NS, IndexId::from_u64(1), b"o", b"p", 100)
        .expect("range works during splits");
    assert_eq!(oslo.len(), 13);
    cluster.wait_splits_done(Duration::from_secs(600));
    let zurich = client
        .index_equal(NS, IndexId::from_u64(1), b"zurich", 100)
        .expect("equality works after splits");
    assert_eq!(zurich.len(), 13);
    // Unique race: exactly one claimant wins across threads.
    let results: Vec<_> = std::thread::scope(|scope| {
        [0u8, 1]
            .iter()
            .map(|tablet| {
                let client = client.clone();
                let key = spread_key(*tablet, 5000);
                scope.spawn(move || {
                    client.indexed_set(
                        NS,
                        &key,
                        b"owner".to_vec(),
                        &[term(2, IndexKind::Unique, &["solo@example.com"])],
                    )
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().expect("thread joins"))
            .collect()
    });
    let violations = results
        .iter()
        .filter(|result| matches!(result, Err(kivi_client::ClientError::UniqueViolation)))
        .count();
    let committed = results.iter().filter(|result| result.is_ok()).count();
    assert_eq!(committed, 1, "exactly one unique owner");
    assert_eq!(violations, 1, "loser sees a stable violation");
    // Full restart: scans and index queries remain correct.
    cluster.restart_all().expect("cluster restarts");
    let _ = cluster.wait_all_leaders();
    let berlin = client
        .index_equal(NS, IndexId::from_u64(1), b"berlin", 100)
        .expect("equality works after restart");
    assert_eq!(berlin.len(), 14);
    assert_eq!(cluster.total_intents(), 0, "no stuck intents after restart");
}

#[test]
fn chained_ordered_splits_with_timing() {
    let cluster = Cluster::spawn_ordered(4).expect("ordered cluster spawns");
    let _ = cluster.wait_all_leaders();
    let client = cluster.client();
    // Steady dataset across the keyspace.
    for tablet in 0..4u8 {
        for index in 0..150usize {
            put(&client, &spread_key(tablet, index), b"v");
        }
    }
    let mut tablet_count = 4;
    let mut pauses: Vec<Duration> = Vec::new();
    // Two chained rounds: 4 → 8 → 16, splitting every tablet each round.
    for _round in 0..2 {
        let tablets: Vec<u64> = cluster
            .tablet_ids(cluster.live_index())
            .into_iter()
            .filter(|id| *id != 0)
            .collect();
        assert_eq!(
            tablets.len(),
            tablet_count,
            "round starts at expected width"
        );
        for tablet in &tablets {
            let start = Instant::now();
            let reply = cluster.split_tablet(*tablet);
            assert_eq!(
                reply.get("ok").and_then(serde_json::Value::as_bool),
                Some(true),
                "split accepted: {reply}"
            );
            cluster.wait_splits_done(Duration::from_secs(600));
            pauses.push(start.elapsed());
        }
        tablet_count *= 2;
        // Coverage stays exact after every round (cutover converges per
        // node after plan completion: poll for the doubled width).
        let deadline = Instant::now() + Duration::from_secs(300);
        loop {
            let ranges = cluster.ordered_ranges(cluster.live_index());
            if ranges.len() == tablet_count {
                let mut sorted = ranges;
                sorted.sort_by(|left, right| left.1.cmp(&right.1));
                let complete = sorted[0].1.is_empty()
                    && sorted.last().expect("ranges").2.is_none()
                    && sorted
                        .windows(2)
                        .all(|pair| pair[0].2.as_deref() == Some(pair[1].1.as_slice()));
                if complete {
                    break;
                }
            }
            assert!(
                Instant::now() < deadline,
                "ordered coverage never converged at width {tablet_count}"
            );
            std::thread::sleep(Duration::from_millis(500));
        }
        // Full scan stays exact at the new width.
        let got = full_scan(&client);
        assert_eq!(got.len(), 600, "no omissions at width {tablet_count}");
        assert_sorted_unique(&got);
    }
    pauses.sort();
    let percentile = |num: usize, den: usize| {
        pauses[pauses.len() * num / den].min(*pauses.last().expect("pauses"))
    };
    eprintln!(
        "ordered chained-split cutover: n={} p50={:?} p95={:?} max={:?}",
        pauses.len(),
        percentile(1, 2),
        percentile(95, 100),
        pauses.last().expect("pauses"),
    );
    assert_eq!(cluster.total_intents(), 0, "no stuck intents");
}

#[test]
fn full_restart_preserves_ordered_state() {
    let mut cluster = Cluster::spawn_ordered(4).expect("ordered cluster spawns");
    let _ = cluster.wait_all_leaders();
    let client = cluster.client();
    for tablet in 0..4u8 {
        for index in 0..100usize {
            put(&client, &spread_key(tablet, index), b"v");
        }
    }
    client
        .atomic_batch(
            NS,
            &[
                BatchWriteSpec {
                    key: spread_key(0, 9999),
                    kind: BatchWriteKind::Put(b"txn-a".to_vec()),
                    expect: BatchExpect::Any,
                },
                BatchWriteSpec {
                    key: spread_key(3, 9999),
                    kind: BatchWriteKind::Put(b"txn-b".to_vec()),
                    expect: BatchExpect::Any,
                },
            ],
        )
        .expect("batch commits before restart");
    cluster.restart_all().expect("cluster restarts");
    let _ = cluster.wait_all_leaders();
    let got = full_scan(&client);
    assert_eq!(got.len(), 402, "all keys survive full restart");
    assert_eq!(
        client
            .get(&Key::from(spread_key(0, 9999)))
            .expect("get works"),
        Some(Bytes::from("txn-a"))
    );
    assert_eq!(cluster.total_intents(), 0, "no stuck intents");
}

#[test]
fn ordered_split_then_merge_round_trip() {
    let mut cluster = Cluster::spawn_ordered(4).expect("ordered cluster spawns");
    let _ = cluster.wait_all_leaders();
    let client = cluster.client();
    // Dataset across the keyspace.
    let mut expected: Vec<Vec<u8>> = Vec::new();
    for tablet in 0..4u8 {
        for index in 0..50usize {
            let key = spread_key(tablet, index);
            put(&client, &key, b"v");
            expected.push(key);
        }
    }
    expected.sort();
    // Split tablet 2 (median path) and wait for cutover.
    let reply = cluster.split_tablet(2);
    let (Some(left), Some(right)) = (
        reply.get("left").and_then(serde_json::Value::as_u64),
        reply.get("right").and_then(serde_json::Value::as_u64),
    ) else {
        panic!("split accepted with children: {reply}");
    };
    cluster.wait_splits_done(Duration::from_secs(600));
    let mut got = full_scan(&client);
    got.sort();
    assert_eq!(got, expected, "no omissions after split");
    // Merge the children back and wait for cutover.
    let _ = cluster.merge_tablets(left, right);
    cluster.wait_merges_done(Duration::from_secs(600));
    let mut got = full_scan(&client);
    got.sort();
    assert_eq!(got, expected, "no omissions after merge");
    // Durability: the merged base (seeded out-of-band, snapshotted under
    // fence) survives a full hard restart.
    cluster.restart_all().expect("cluster restarts");
    let _ = cluster.wait_all_leaders();
    let mut got = full_scan(&client);
    got.sort();
    assert_eq!(got, expected, "merged state survives restart");
    assert_eq!(cluster.total_intents(), 0, "no stuck intents");
}

#[test]
fn ordered_automatic_split_then_merge() {
    let cluster = Cluster::spawn_ordered(4).expect("ordered cluster spawns");
    let _ = cluster.wait_all_leaders();
    let live = cluster.live_index();
    let client = cluster.client();
    // Background tablets stay quiet: 20 small keys each (above the merge
    // object threshold, below the split byte threshold — never eligible).
    for tablet in [0u8, 2, 3] {
        for index in 0..20usize {
            put(&client, &spread_key(tablet, index), b"v");
        }
    }
    // Tiny byte threshold; tablet 2 ('A' keys) carries 60 x 200B = 12KB.
    let _ = cluster.set_policy(&serde_json::json!({
        "auto_split": true,
        "auto_merge": false,
        "split_bytes_threshold": 4096,
    }));
    let big: Vec<u8> = vec![0x78; 200];
    let mut hot: Vec<Vec<u8>> = Vec::new();
    for index in 0..60usize {
        let key = spread_key(1, 2000 + index);
        put(&client, &key, &big);
        hot.push(key);
    }
    // An automatic ordered split plan appears with no manual command.
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        let splits = cluster.splits(live);
        let plans = splits
            .get("splits")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        if !plans.is_empty() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "automatic ordered split never planned"
        );
        std::thread::sleep(Duration::from_secs(2));
    }
    cluster.wait_splits_done(Duration::from_secs(600));
    // Cool the hot children to mergeable size, then enable auto-merge.
    for key in &hot {
        let _ = client.delete(&Key::from(key.clone()));
    }
    let _ = cluster.set_policy(&serde_json::json!({
        "auto_split": false,
        "auto_merge": true,
        "merge_object_threshold": 10,
    }));
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        let merges = cluster.merges(live);
        let plans = merges
            .get("merges")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        if !plans.is_empty() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "automatic ordered merge never planned"
        );
        std::thread::sleep(Duration::from_secs(2));
    }
    cluster.wait_merges_done(Duration::from_secs(600));
    // Only the quiet background keys remain, exactly.
    let mut got = full_scan(&client);
    got.sort();
    assert_eq!(got.len(), 60, "hot keys deleted, background intact");
    assert_sorted_unique(&got);
    assert_eq!(cluster.total_intents(), 0, "no stuck intents");
}
