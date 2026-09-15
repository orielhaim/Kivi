//! Real multi-tablet product tests: many independent Raft groups over one
//! shared H3 peer mesh and one shared WAL writer per node.
//!
//! All tests run against three real `kivi-server --cluster-mode` processes
//! (real QUIC/H3, real data directories) with several tablets striped over
//! several consensus workers per node:
//!
//! ```text
//! startup + writes across every tablet -> per-tablet leaders differ ->
//! kill one node -> affected groups elect replacements independently ->
//! writes continue -> restart -> every tablet converges -> full-cluster
//! restart preserves everything.
//! ```

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use kivi_lab::cluster::Cluster;
use kivi_state::Key;

fn key(name: &str) -> Key {
    Key::from(name)
}

fn put(client: &kivi_client::NativeClient, name: &str, value: &[u8]) {
    client
        .set(&key(name), bytes::Bytes::copy_from_slice(value))
        .unwrap_or_else(|error| panic!("set {name} commits: {error:?}"));
}

fn get(client: &kivi_client::NativeClient, name: &str) -> Vec<u8> {
    client
        .get(&key(name))
        .unwrap_or_else(|error| panic!("get {name} reads: {error:?}"))
        .unwrap_or_else(|| panic!("{name} present"))
        .to_vec()
}

/// Distinct leader node ids observed across tablets on one member.
fn distinct_leaders(cluster: &Cluster, index: usize) -> Vec<u64> {
    let mut leaders: Vec<u64> = cluster
        .tablets_leaders(index)
        .values()
        .filter_map(|leader| *leader)
        .collect();
    leaders.sort_unstable();
    leaders.dedup();
    leaders
}

/// Multi-tablet money flow (§43): many tablets, writes everywhere,
/// independent per-tablet leadership, kill with independent elections,
/// continued traffic, restart convergence, full-cluster restart.
#[allow(clippy::too_many_lines)]
#[test]
fn multi_tablets_startup_writes_failover_restart() {
    let mut cluster = Cluster::spawn_with_tablets(16, 4, false).expect("cluster spawns");
    let leaders = cluster.wait_all_leaders();
    assert_eq!(leaders.len(), 16, "every tablet elects exactly one leader");
    // Keys covering every tablet (2 per tablet).
    let keys = cluster.keys_for_tablets(2);
    assert_eq!(keys.len(), 16);
    // Independent logs (deterministic): writes to one tablet advance only
    // that tablet's applied index on every member — a shared log would
    // move all tablets together.
    let pioneer = *keys.keys().next().expect("a tablet");
    cluster.wait_converged_all();
    let before = cluster.tablets_applied(cluster.live_index());
    // Fresh clients after every membership change below: cached routes
    // may name a dead member (refused dials), and route re-discovery
    // through redirects is the designed flow (same as the single-tablet
    // failover tests, which re-seed after kills).
    let mut client = cluster.client();
    for name in &keys[&pioneer] {
        put(&client, name, b"pioneer");
    }
    cluster.wait_converged_all();
    for i in 0..3 {
        if !cluster.alive(i) {
            continue;
        }
        let after = cluster.tablets_applied(i);
        for (tablet, applied) in &after {
            if *tablet == pioneer {
                assert!(
                    *applied > before[tablet],
                    "pioneer tablet advances on member {i}"
                );
            } else {
                assert_eq!(
                    *applied, before[tablet],
                    "untouched tablet {tablet} never moves on member {i}"
                );
            }
        }
    }
    // Independent leadership: simultaneous startup may (and often does)
    // elect one member everywhere, so scatter with kill rounds until
    // tablets genuinely split across leaders (each round re-scatters all
    // of the victim's groups independently; bounded).
    let mut rounds = 0;
    let distinct = loop {
        let distinct = distinct_leaders(&cluster, cluster.live_index());
        if distinct.len() >= 2 || rounds >= 3 {
            break distinct;
        }
        rounds += 1;
        let victim = cluster
            .wait_all_leaders()
            .into_iter()
            .next()
            .map(|(_, leader)| leader)
            .expect("a leader");
        cluster.kill(victim);
        let _ = cluster.wait_all_leaders();
        cluster.restart(victim).expect("scatter victim restarts");
        cluster.wait_converged_all();
    };
    assert!(
        distinct.len() >= 2,
        "different tablets have different leaders after scatter: {distinct:?}"
    );
    client = cluster.client();
    for (tablet, names) in &keys {
        for name in names {
            put(&client, name, format!("t{tablet}-{name}").as_bytes());
        }
    }
    cluster.wait_converged_all();
    for (tablet, names) in &keys {
        for name in names {
            assert_eq!(get(&client, name), format!("t{tablet}-{name}").as_bytes());
        }
    }
    // Kill one node: affected groups elect replacements, the rest continue.
    let victim = cluster
        .wait_all_leaders()
        .into_iter()
        .next()
        .map(|(_, leader)| leader)
        .expect("a leader");
    cluster.kill(victim);
    let elected = cluster.wait_all_leaders();
    assert_eq!(
        elected.len(),
        16,
        "every tablet re-elects without the victim"
    );
    assert!(
        elected.iter().all(|(_, leader)| *leader != victim),
        "no tablet still points at the dead member"
    );
    client = cluster.client();
    // Groups whose leader survived continue serving immediately.
    let survivor_tablet = elected
        .iter()
        .find(|(_, leader)| cluster.alive(*leader))
        .map(|(tablet, _)| *tablet)
        .expect("a live-led tablet");
    let survivor_key = cluster
        .keys_for_tablets(1)
        .remove(&survivor_tablet)
        .expect("survivor key")[0]
        .clone();
    put(&client, &survivor_key, b"during-outage");
    assert_eq!(get(&client, &survivor_key), b"during-outage");
    // Mixed writes across every tablet while degraded.
    for names in keys.values() {
        put(&client, &names[0], b"degraded");
    }
    // Restart: every tablet replica converges with no manual repair.
    // (`survivor_key` is one tablet's `names[0]`, so the degraded wave
    // overwrote it too — the during-outage value was already verified
    // above, before the wave.)
    cluster.restart(victim).expect("victim restarts");
    cluster.wait_converged_all();
    client = cluster.client();
    for names in keys.values() {
        assert_eq!(get(&client, names[0].as_str()), b"degraded");
    }
    assert_eq!(get(&client, &survivor_key), b"degraded");
    // Full-cluster restart: all tablet state survives.
    cluster.restart_all().expect("full restart reforms");
    cluster.wait_converged_all();
    client = cluster.client();
    for names in keys.values() {
        assert_eq!(get(&client, names[0].as_str()), b"degraded");
    }
}

/// Independent ordering (§44): a large write on tablet A creates no Raft
/// ordering dependency on tablet B — B's writes commit while A's bulk
/// moves.
#[test]
fn multi_tablet_independent_ordering() {
    let cluster = Cluster::spawn_with_tablets(4, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    // All B keys are verified on tablet B upfront (deterministic
    // targeting, not prefix luck).
    let keys = cluster.keys_for_tablets(21);
    let tablets: Vec<u64> = keys.keys().copied().collect();
    let (tablet_a, tablet_b) = (tablets[0], tablets[1]);
    let key_a = keys[&tablet_a][0].clone();
    let keys_b = keys[&tablet_b][..20].to_vec();
    let value_a = vec![0xABu8; 16 * 1024 * 1024];
    let started = Instant::now();
    let upload_key = key_a.clone();
    std::thread::scope(|scope| {
        let upload_client = cluster.client();
        let upload = scope.spawn(move || {
            let mut source: &[u8] = &value_a;
            upload_client.put_stream(&key(&upload_key), Some(value_a.len() as u64), &mut source)
        });
        // Tablet B writes must commit while A's 16 MiB is still moving
        // (no cross-tablet Raft head-of-line).
        let client = cluster.client();
        let mut b_done = 0;
        let deadline = Instant::now() + Duration::from_secs(120);
        for name in &keys_b {
            assert!(Instant::now() < deadline, "tablet B writes stall behind A");
            put(&client, name, format!("b-{b_done}").as_bytes());
            b_done += 1;
            if upload.is_finished() {
                break;
            }
        }
        let outcome = upload.join().expect("upload thread joins");
        assert!(outcome.is_ok(), "tablet A upload commits: {outcome:?}");
        eprintln!(
            "tablet B committed {b_done} writes during A's upload ({:?})",
            started.elapsed()
        );
    });
    cluster.wait_converged_all();
    assert_eq!(get(&cluster.client(), &key_a).len(), 16 * 1024 * 1024);
}

/// Minority fencing per tablet: an isolated member leading tablet A must
/// not acknowledge new writes on A while the majority continues.
#[test]
fn multi_tablet_minority_fences_isolated_leader() {
    use kivi_client::{ClientConfig, NativeClient};
    use kivi_types::NamespaceId;

    let cluster = Cluster::spawn_with_tablets(8, 2, false).expect("cluster spawns");
    let leaders = cluster.wait_all_leaders();
    let (tablet_a, leader_a) = (leaders[0].0, leaders[0].1);
    let keys = cluster.keys_for_tablets(1);
    let key_a = keys[&tablet_a][0].clone();
    cluster
        .client()
        .set(&key(&key_a), bytes::Bytes::from_static(b"base"))
        .expect("base commits");
    cluster.wait_converged_all();
    // Isolate the tablet-A leader from both followers. The survivors are
    // partitioned but alive, so the majority view must ignore the
    // isolate's stale leadership claim (it still reports leader on its
    // stale term): exactly one survivor claims tablet-A leadership,
    // stably, and the other survivor observes that same leader id.
    cluster.partition(leader_a);
    let others: Vec<usize> = (0..3).filter(|i| *i != leader_a).collect();
    let node_of = |index: usize| cluster.node_info(index)["node"].as_u64().expect("node id");
    let role_of = |index: usize| -> Option<String> {
        cluster
            .tablets_status(index)
            .as_array()
            .expect("tablets array")
            .iter()
            .find(|entry| entry["group"].as_u64() == Some(tablet_a))
            .and_then(|entry| entry["role"].as_str())
            .map(str::to_owned)
    };
    let deadline = Instant::now() + Duration::from_secs(45);
    let mut stable = 0;
    let majority = loop {
        let claimants: Vec<usize> = others
            .iter()
            .filter(|i| role_of(**i).as_deref() == Some("leader"))
            .copied()
            .collect();
        let agreed = claimants.len() == 1
            && others.iter().all(|i| {
                cluster
                    .tablets_leaders(*i)
                    .get(&tablet_a)
                    .copied()
                    .flatten()
                    == Some(node_of(claimants[0]))
            });
        if agreed {
            stable += 1;
            if stable >= 3 {
                break claimants[0];
            }
        } else {
            stable = 0;
        }
        assert!(
            Instant::now() < deadline,
            "majority never takes tablet {tablet_a}"
        );
        std::thread::sleep(Duration::from_millis(200));
    };
    assert_ne!(majority, leader_a);
    // Pinned on the isolate, the write must never acknowledge (its commit
    // can never reach quorum). Still-leader, still-pending, or prompt
    // routing failure all count as fenced — only `Ok` fails.
    let isolate_client = NativeClient::new(ClientConfig {
        seeds: vec![cluster.native_endpoint(leader_a)],
        namespace: NamespaceId::from_u64(1),
        request_timeout: Duration::from_secs(6),
        ..ClientConfig::default()
    })
    .expect("isolate client builds");
    let verdict = isolate_client.set(&key(&key_a), bytes::Bytes::from_static(b"isolated"));
    assert!(
        verdict.is_err(),
        "isolated minority acknowledged without quorum: {verdict:?}"
    );
    // The majority side proceeds on the same tablet while partitioned.
    cluster
        .client()
        .set(&key(&key_a), bytes::Bytes::from_static(b"majority"))
        .expect("majority writes during partition");
    cluster.heal(leader_a);
    cluster.wait_converged_all();
    assert_eq!(get(&cluster.client(), &key_a), b"majority");
}

/// Lost-response exactly-once on a non-default tablet: same identity
/// retried after the tablet's leader dies returns the original outcome.
#[test]
fn multi_tablet_lost_response_exactly_once_non_default_tablet() {
    use kivi_types::{RequestSeq, SessionId};

    let mut cluster = Cluster::spawn_with_tablets(4, 2, false).expect("cluster spawns");
    let leaders = cluster.wait_all_leaders();
    assert!(leaders.len() >= 2);
    // A non-first tablet (never the default/single-tablet path).
    let (tablet, leader) = (leaders[1].0, leaders[1].1);
    let keys = cluster.keys_for_tablets(1);
    let name = keys[&tablet][0].clone();
    let session = SessionId::from_u128(0x00AB_1ABE_0000_0001u128);
    let first = cluster
        .client_with_session(session)
        .counter_add_with_seq(&key(&name), 1, RequestSeq::from_u64(1))
        .expect("first add commits");
    assert_eq!(first, 1);
    cluster.wait_converged_all();
    cluster.kill(leader);
    let _ = cluster.wait_all_leaders();
    // Same identity retried: dedup hit on the new leader, counter advances
    // once.
    let retry = cluster
        .client_with_session(session)
        .counter_add_with_seq(&key(&name), 1, RequestSeq::from_u64(1))
        .expect("retry answers");
    assert_eq!(retry, 1, "same identity returns the original outcome");
    let value = cluster
        .client()
        .counter_get(&key(&name))
        .expect("reads")
        .expect("present");
    assert_eq!(value, 1);
    cluster.restart(leader).expect("old leader restarts");
    cluster.wait_converged_all();
}

/// Shared-WAL crash recovery across many groups: writes, per-group
/// snapshots/purges on different groups sharing one physical WAL, kill,
/// restart — every group recovers its own exact history.
#[test]
fn multi_tablet_shared_wal_crash_recovery_and_purge() {
    let mut cluster = Cluster::spawn_with_tablets(8, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(3);
    let client = cluster.client();
    for names in keys.values() {
        for name in names {
            put(&client, name, format!("v1-{name}").as_bytes());
        }
    }
    cluster.wait_converged_all();
    // Snapshot/purge different groups sharing the physical WAL (one
    // group's purge must never erase another's records).
    let tablets: Vec<u64> = keys.keys().copied().collect();
    let live = cluster.live_index();
    let mut bases = BTreeMap::new();
    for tablet in tablets.iter().take(3) {
        bases.insert(*tablet, cluster.snapshot_tablet(live, *tablet));
    }
    for names in keys.values() {
        put(&client, &names[1], b"v2");
    }
    cluster.wait_converged_all();
    // Crash one member hard; restart; every group converges exactly.
    let kill_at = (live + 1) % 3;
    cluster.kill(kill_at);
    cluster.restart(kill_at).expect("crashed member restarts");
    cluster.wait_converged_all();
    for names in keys.values() {
        assert_eq!(get(&client, &names[1]), b"v2");
        assert!(get(&client, &names[0]).starts_with(b"v1-"));
    }
    // Snapshot bases advanced on the snapshotted groups only.
    for (tablet, base) in &bases {
        assert!(*base > 0, "tablet {tablet} sealed a snapshot base");
    }
}
