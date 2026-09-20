//! Ordered data-plane money tests over a single-node TCP engine:
//! ordered scans, atomic batches, OCC conflicts, and secondary indexes
//! through the normal native server/client paths (no test-only state).

use bytes::Bytes;
use kivi_client::{
    ClientConfig, NativeClient,
    ordered::{
        AtomicBatchResult, BatchExpect, BatchWriteKind, BatchWriteSpec, ClientScanValue,
        IndexTermSet, ScanConsistency, ScanDirection, ScanOptions, ScanProjection,
    },
};
use kivi_engine::{
    ConnLimits, DurabilityMode, EngineConfig, EngineNetwork, LocalEngine, Placement, TurnBudget,
};
use kivi_state::{IndexId, IndexKind, Key};
use kivi_tablet::DirectorySnapshot;
use kivi_types::{ClusterId, NamespaceId, NodeId, NodeIncarnation, TabletId, WorkerId};

const NS: NamespaceId = NamespaceId::from_u64(1);

/// Three ordered tablets: 1:[empty, m), 2:[m, t), 3:[t, +inf).
fn ordered_snapshot() -> DirectorySnapshot {
    DirectorySnapshot::static_ordered_tiles(NS, &[b"m".to_vec(), b"t".to_vec()])
        .expect("ordered tiling builds")
}

fn network() -> EngineNetwork {
    EngineNetwork {
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
    }
}

fn start_ordered() -> LocalEngine {
    LocalEngine::start(EngineConfig {
        namespace: NS,
        directory: ordered_snapshot(),
        placement: Placement::new([
            (TabletId::from_u64(1), WorkerId::from_u64(0)),
            (TabletId::from_u64(2), WorkerId::from_u64(1)),
            (TabletId::from_u64(3), WorkerId::from_u64(0)),
        ]),
        worker_count: 2,
        request_capacity: 1024,
        chunks: kivi_engine::ChunkFabricConfig::default(),
        fabric: kivi_engine::FabricConfig::default(),
        network: Some(network()),
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

fn put(client: &NativeClient, key: &str, value: &str) {
    client
        .set(&Key::from(key), Bytes::from(value.to_owned()))
        .expect("set works");
}

fn scan_all(
    client: &NativeClient,
    direction: ScanDirection,
    projection: ScanProjection,
) -> Vec<(Vec<u8>, ClientScanValue)> {
    let options = ScanOptions {
        direction,
        projection,
        consistency: ScanConsistency::LatestPerTablet,
        max_items_per_page: 7,
        max_bytes_per_page: 1 << 16,
        limit: None,
        ..ScanOptions::default()
    };
    client
        .scan(NS, &options)
        .expect("scan works")
        .into_iter()
        .map(|entry| (entry.key, entry.value))
        .collect()
}

#[test]
fn ordered_scan_forward_reverse_prefix_paginate() {
    let engine = start_ordered();
    let client = client_for(&engine);
    // Keys spread across all three tablets (a-g → T1, m-s → T2, t-z → T3).
    let mut keys: Vec<String> = Vec::new();
    for word in [
        "apple", "berry", "cherry", "mango", "melon", "olive", "peach", "tango", "ultra", "zebra",
    ] {
        put(&client, word, &format!("v:{word}"));
        keys.push(word.to_owned());
    }
    // Full forward scan: strictly ordered, complete.
    let forward = scan_all(&client, ScanDirection::Forward, ScanProjection::KeysOnly);
    let forward_keys: Vec<String> = forward
        .iter()
        .map(|(key, _)| String::from_utf8_lossy(key).into_owned())
        .collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(forward_keys, sorted);
    // Full reverse scan: exact reverse.
    let reverse = scan_all(&client, ScanDirection::Reverse, ScanProjection::KeysOnly);
    let reverse_keys: Vec<String> = reverse
        .iter()
        .map(|(key, _)| String::from_utf8_lossy(key).into_owned())
        .collect();
    sorted.reverse();
    assert_eq!(reverse_keys, sorted);
    // Prefix/sub-range scan with values.
    let options = ScanOptions {
        start: Some(b"m".to_vec()),
        end: Some(b"p".to_vec()),
        direction: ScanDirection::Forward,
        consistency: ScanConsistency::LatestPerTablet,
        projection: ScanProjection::KeysAndValues,
        max_items_per_page: 2,
        max_bytes_per_page: 1 << 16,
        limit: None,
    };
    let entries = client.scan(NS, &options).expect("range scan works");
    let got: Vec<String> = entries
        .iter()
        .map(|entry| String::from_utf8_lossy(&entry.key).into_owned())
        .collect();
    assert_eq!(got, vec!["mango", "melon", "olive"]);
    for entry in &entries {
        match &entry.value {
            ClientScanValue::Inline(bytes) => {
                assert_eq!(
                    &bytes[..],
                    format!("v:{}", String::from_utf8_lossy(&entry.key)).as_bytes()
                );
            }
            other => panic!("expected inline value, got {other:?}"),
        }
    }
    // Reverse sub-range.
    let options = ScanOptions {
        start: None,
        end: Some(b"m".to_vec()),
        direction: ScanDirection::Reverse,
        limit: Some(2),
        ..options
    };
    let entries = client.scan(NS, &options).expect("reverse range works");
    let got: Vec<String> = entries
        .iter()
        .map(|entry| String::from_utf8_lossy(&entry.key).into_owned())
        .collect();
    assert_eq!(got, vec!["cherry", "berry"]);
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn atomic_batch_single_and_multi_tablet() {
    let engine = start_ordered();
    let client = client_for(&engine);
    // Single-tablet batch (all keys on tablet 1).
    let result = client
        .atomic_batch(
            NS,
            &[
                BatchWriteSpec {
                    key: b"a:1".to_vec(),
                    kind: BatchWriteKind::Put(b"v1".to_vec()),
                    expect: BatchExpect::Any,
                },
                BatchWriteSpec {
                    key: b"a:2".to_vec(),
                    kind: BatchWriteKind::Put(b"v2".to_vec()),
                    expect: BatchExpect::Any,
                },
            ],
        )
        .expect("single-tablet batch commits");
    assert!(mixer(&result));
    assert_eq!(result.versions.len(), 2);
    // Multi-tablet batch spanning all three tablets (client-driven 2PC).
    let result = client
        .atomic_batch(
            NS,
            &[
                BatchWriteSpec {
                    key: b"a:x".to_vec(),
                    kind: BatchWriteKind::Put(b"1".to_vec()),
                    expect: BatchExpect::Any,
                },
                BatchWriteSpec {
                    key: b"m:x".to_vec(),
                    kind: BatchWriteKind::Put(b"2".to_vec()),
                    expect: BatchExpect::Any,
                },
                BatchWriteSpec {
                    key: b"z:x".to_vec(),
                    kind: BatchWriteKind::CounterAdd(41),
                    expect: BatchExpect::Any,
                },
            ],
        )
        .expect("multi-tablet batch commits");
    assert_eq!(result.versions.len(), 3);
    assert_eq!(
        client.get(&Key::from("m:x")).expect("get works"),
        Some(Bytes::from("2"))
    );
    assert_eq!(
        client
            .counter_get(&Key::from("z:x"))
            .expect("counter works"),
        Some(41)
    );
    engine.shutdown().expect("clean shutdown");
}

fn mixer(result: &AtomicBatchResult) -> bool {
    result.versions.iter().all(Option::is_some)
}

#[test]
fn batch_conflict_aborts_cleanly_with_nothing_applied() {
    let engine = start_ordered();
    let client = client_for(&engine);
    put(&client, "a:1", "original");
    let version = client
        .get_version(NS, b"a:1")
        .expect("version works")
        .expect("present");
    // Stale expectation on one key aborts the whole batch.
    let error = client
        .atomic_batch(
            NS,
            &[
                BatchWriteSpec {
                    key: b"a:1".to_vec(),
                    kind: BatchWriteKind::Put(b"lost".to_vec()),
                    expect: BatchExpect::Version(version + 100),
                },
                BatchWriteSpec {
                    key: b"m:1".to_vec(),
                    kind: BatchWriteKind::Put(b"lost-too".to_vec()),
                    expect: BatchExpect::Any,
                },
            ],
        )
        .expect_err("stale batch conflicts");
    assert!(
        matches!(error, kivi_client::ClientError::TxnConflict),
        "conflict surfaces, got {error:?}"
    );
    // Nothing applied anywhere.
    assert_eq!(
        client.get(&Key::from("a:1")).expect("get works"),
        Some(Bytes::from("original"))
    );
    assert_eq!(client.get(&Key::from("m:1")).expect("get works"), None);
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn concurrent_batches_have_exactly_one_winner() {
    let engine = start_ordered();
    let client = client_for(&engine);
    put(&client, "race", "v0");
    let version = client
        .get_version(NS, b"race")
        .expect("version works")
        .expect("present");
    // Two threads race blind-versioned writes with the same expectation.
    let results: Vec<_> = std::thread::scope(|scope| {
        [1, 2]
            .iter()
            .map(|id| {
                let client = client.clone();
                scope.spawn(move || {
                    client.atomic_batch(
                        NS,
                        &[BatchWriteSpec {
                            key: b"race".to_vec(),
                            kind: BatchWriteKind::Put(format!("winner{id}").into_bytes()),
                            expect: BatchExpect::Version(version),
                        }],
                    )
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().expect("thread joins"))
            .collect()
    });
    let committed = results.iter().filter(|result| result.is_ok()).count();
    assert_eq!(committed, 1, "exactly one compatible result wins");
    let value = client.get(&Key::from("race")).expect("get works");
    assert!(value == Some(Bytes::from("winner1")) || value == Some(Bytes::from("winner2")));
    engine.shutdown().expect("clean shutdown");
}

fn term_set(index: u64, kind: IndexKind, terms: &[&str]) -> IndexTermSet {
    IndexTermSet {
        index: IndexId::from_u64(index),
        kind,
        terms: terms.iter().map(|term| term.as_bytes().to_vec()).collect(),
    }
}

#[test]
fn secondary_index_money_flow() {
    let engine = start_ordered();
    let client = client_for(&engine);
    // Populate indexed records across tablets (primaries everywhere,
    // terms shared so index entries scatter too).
    for (user, city) in [
        ("user:amy", "berlin"),
        ("user:bob", "berlin"),
        ("user:cid", "oslo"),
        ("user:dan", "oslo"),
        ("user:eve", "oslo"),
        ("user:zed", "zurich"),
    ] {
        client
            .indexed_set(
                NS,
                user.as_bytes(),
                format!("profile:{user}").into_bytes(),
                &[term_set(1, IndexKind::NonUnique, &[city])],
            )
            .expect("indexed set works");
    }
    // Equality lookup.
    let mut berlin = client
        .index_equal(NS, IndexId::from_u64(1), b"berlin", 100)
        .expect("equality works");
    berlin.sort();
    assert_eq!(berlin, vec![b"user:amy".to_vec(), b"user:bob".to_vec()]);
    // Range lookup over terms.
    let oslo = client
        .index_range(NS, IndexId::from_u64(1), b"o", b"p", 100)
        .expect("range works");
    assert_eq!(oslo.len(), 3);
    // Update an indexed term: old entry disappears, new appears.
    client
        .indexed_set(
            NS,
            b"user:amy",
            b"profile:user:amy".to_vec(),
            &[term_set(1, IndexKind::NonUnique, &["oslo"])],
        )
        .expect("update works");
    let berlin = client
        .index_equal(NS, IndexId::from_u64(1), b"berlin", 100)
        .expect("equality works");
    assert_eq!(berlin, vec![b"user:bob".to_vec()]);
    let oslo = client
        .index_equal(NS, IndexId::from_u64(1), b"oslo", 100)
        .expect("equality works");
    assert_eq!(oslo.len(), 4);
    // Delete removes the primary and its index entries.
    client
        .indexed_delete(NS, b"user:bob")
        .expect("delete works");
    assert_eq!(client.get(&Key::from("user:bob")).expect("get works"), None);
    let berlin = client
        .index_equal(NS, IndexId::from_u64(1), b"berlin", 100)
        .expect("equality works");
    assert!(berlin.is_empty());
    // Unique index: second claimant loses stably.
    client
        .indexed_set(
            NS,
            b"user:amy",
            b"profile:user:amy".to_vec(),
            &[term_set(2, IndexKind::Unique, &["amy@example.com"])],
        )
        .expect("unique claim works");
    let error = client
        .indexed_set(
            NS,
            b"user:cid",
            b"profile:user:cid".to_vec(),
            &[term_set(2, IndexKind::Unique, &["amy@example.com"])],
        )
        .expect_err("dual claim fails");
    assert!(
        matches!(error, kivi_client::ClientError::UniqueViolation),
        "stable violation, got {error:?}"
    );
    // The unique term still resolves to its one owner.
    let owner = client
        .index_equal_unique(NS, IndexId::from_u64(2), b"amy@example.com")
        .expect("unique lookup works");
    assert_eq!(owner, Some(b"user:amy".to_vec()));
    engine.shutdown().expect("clean shutdown");
}
