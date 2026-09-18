//! Replicated control-plane product tests: dynamic node admission,
//! learner catch-up through the existing snapshot/sidecar path, joint
//! membership replacement, leadership handoff, drain, safe removal, and
//! crash recovery — all on real processes over H3/QUIC with real data
//! directories.
//!
//! ```text
//! start A/B/C -> populate (small + 64 MiB chunked) -> add D (no restart
//! of A/B/C) -> rebalance while serving traffic -> learners catch up on
//! D -> promotions -> old replicas retire -> drain A -> remove A ->
//! remaining cluster serves -> full restart recovers.
//! ```

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

fn put_big(client: &kivi_client::NativeClient, name: &str, value: &[u8]) {
    let mut source: &[u8] = value;
    client
        .put_stream(&key(name), Some(value.len() as u64), &mut source)
        .unwrap_or_else(|error| panic!("put_stream {name} commits: {error:?}"));
}

fn get_big(client: &kivi_client::NativeClient, name: &str) -> Vec<u8> {
    client
        .get_stream(&key(name))
        .unwrap_or_else(|error| panic!("get_stream {name} reads: {error:?}"))
        .unwrap_or_else(|| panic!("{name} present"))
        .to_vec()
}

fn fill(len: usize, seed: u8) -> Vec<u8> {
    // Deterministic pseudo-random bytes (cheap LCG, no extra deps).
    let mut out = vec![0u8; len];
    let mut state = u64::from(seed) | 1;
    for byte in &mut out {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *byte = u8::try_from((state >> 33) & 0xFF).unwrap_or(0xFF);
    }
    out
}

/// Streaming upload with bounded whole-call retries (the source is
/// re-readable memory; pre-commit failures are definitely
/// uncommitted, so a fresh attempt is safe).
fn traffic_put_big(client: &kivi_client::NativeClient, name: &str, value: &[u8]) {
    for attempt in 0..4 {
        let mut source: &[u8] = value;
        match client.put_stream(&key(name), Some(value.len() as u64), &mut source) {
            Ok(()) => return,
            Err(error) if attempt < 3 => {
                std::thread::sleep(Duration::from_millis(500));
                let _ = format!("{error:?}");
            }
            Err(error) => panic!("traffic put_stream {name} never converges: {error:?}"),
        }
    }
}

/// Streaming download with bounded retries (reads re-run safely).
fn traffic_get_big(client: &kivi_client::NativeClient, name: &str) -> Vec<u8> {
    for attempt in 0..10 {
        match client.get_stream(&key(name)) {
            Ok(Some(value)) => return value.to_vec(),
            Ok(None) => panic!("traffic key {name} lost"),
            Err(_) if attempt < 9 => {
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(error) => panic!("traffic get_stream {name} never converges: {error:?}"),
        }
    }
    unreachable!()
}

/// Traffic op with bounded retries (redirects converge during
/// migration; a bounded retry is the designed client flow, not a
/// masked failure).
fn traffic_put(client: &kivi_client::NativeClient, name: &str, value: &[u8]) {
    if let Err(error) = traffic_put_result(client, name, value) {
        panic!("traffic set {name} never converges: {error:?}");
    }
}

/// Traffic op with bounded retries, returning the terminal error
/// instead of panicking (the money loop dumps forensics first).
fn traffic_put_result(
    client: &kivi_client::NativeClient,
    name: &str,
    value: &[u8],
) -> Result<(), kivi_client::ClientError> {
    for attempt in 0..10 {
        match client.set(&key(name), bytes::Bytes::copy_from_slice(value)) {
            Ok(()) => return Ok(()),
            Err(error) if attempt < 9 => {
                eprintln!("traffic {name} attempt {attempt}: {error:?}");
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(error) => return Err(error),
        }
    }
    unreachable!()
}

/// Prints plans plus every tablet's observed voters/leader from the
/// first live member (which tablet wedged, and where).
fn dump_cluster_state(cluster: &Cluster, failed_key: &str, error: &kivi_client::ClientError) {
    eprintln!("FORENSIC key={failed_key} error={error:?}");
    // Full per-tablet state from every live member (divergent
    // membership views across members are the prime suspect).
    for index in 0..cluster.member_count() {
        if !cluster.alive(index) {
            eprintln!("FORENSIC member {index} dead");
            continue;
        }
        let (_, control) = cluster.admin_get(index, "/v1/control");
        eprintln!(
            "FORENSIC member {index} control plans={}",
            control
                .get("migrations")
                .map_or_else(String::new, ToString::to_string)
        );
        let (_, tablets) = cluster.admin_get(index, "/v1/tablets");
        if let Some(list) = tablets.as_array() {
            for tablet in list {
                eprintln!(
                    "FORENSIC member {index} tablet={} role={} term={} leader={} voters={} placement={} healthy={} detail={}",
                    tablet
                        .get("group")
                        .map_or_else(String::new, ToString::to_string),
                    tablet
                        .get("role")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("?"),
                    tablet
                        .get("term")
                        .map_or_else(|| String::from("-"), ToString::to_string),
                    tablet
                        .get("leader")
                        .map_or_else(|| String::from("none"), ToString::to_string),
                    tablet
                        .get("voters")
                        .map_or_else(|| String::from("-"), ToString::to_string),
                    tablet
                        .get("placement")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("?"),
                    tablet
                        .get("healthy")
                        .map_or_else(|| String::from("-"), ToString::to_string),
                    tablet
                        .get("detail")
                        .map_or_else(|| String::from("-"), ToString::to_string),
                );
            }
        }
    }
}

fn traffic_get(client: &kivi_client::NativeClient, name: &str) -> Vec<u8> {
    for attempt in 0..10 {
        match client.get(&key(name)) {
            Ok(Some(value)) => return value.to_vec(),
            Ok(None) => panic!("traffic key {name} lost"),
            Err(_) if attempt < 9 => {
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(error) => panic!("traffic get {name} never converges: {error:?}"),
        }
    }
    unreachable!()
}

/// Manual move replaces exactly one voter through learner catch-up and
/// joint consensus, with reads/writes online throughout and byte-exact
/// data afterwards.
#[test]
fn manual_move_replaces_one_voter() {
    let mut cluster = Cluster::spawn_with_tablets(4, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(2);
    let client = cluster.client();
    for names in keys.values() {
        for name in names {
            put(&client, name, b"v");
        }
    }
    let fourth = cluster.add_node();
    let _ = fourth;
    cluster.wait_control_nodes(4, Duration::from_secs(60));
    // One tablet moves 1 -> 4: same persisted plan pathway as rebalance.
    let moved = *keys.keys().next().expect("a tablet");
    let _ = cluster.move_tablet(moved, 1, 4);
    cluster.wait_tablet_voters(moved, &[2, 3, 4], Duration::from_secs(180));
    cluster.wait_migrations_done(Duration::from_secs(120));
    // Data intact everywhere, cluster still serving.
    let client = cluster.client();
    for names in keys.values() {
        for name in names {
            assert_eq!(get(&client, name), b"v");
        }
    }
    // Source retired locally: node 1 no longer lists the moved tablet.
    let (_, tablets) = cluster.admin_get(0, "/v1/tablets");
    let groups: Vec<u64> = tablets
        .as_array()
        .map(|list| {
            list.iter()
                .filter_map(|tablet| tablet.get("group").and_then(serde_json::Value::as_u64))
                .collect()
        })
        .unwrap_or_default();
    assert!(
        !groups.contains(&moved),
        "retired source serves no tablet {moved}: {groups:?}"
    );
}

/// A learner far behind (log purged past it) catches up through
/// checkpoint snapshot + differential immutable sidecars — no full
/// directory copy, no pack-file copy.
#[test]
fn learner_snapshot_catch_up_after_purge() {
    let mut cluster = Cluster::spawn_with_tablets(2, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(4);
    let client = cluster.client();
    for names in keys.values() {
        for name in names {
            put(&client, name, b"purge-me");
        }
    }
    let tablet = *keys.keys().next().expect("a tablet");
    let big = fill(8 * 1024 * 1024, 0xA1);
    let big_key = keys[&tablet][0].clone();
    put_big(&client, &big_key, &big);
    // Snapshot + purge every member's tablet log, then advance past it.
    for index in 0..3 {
        let _ = cluster.snapshot_tablet(index, tablet);
    }
    for round in 0..20 {
        put(&client, &format!("post-purge-{round}"), b"x");
    }
    let fourth = cluster.add_node();
    let _ = fourth;
    cluster.wait_control_nodes(4, Duration::from_secs(60));
    let _ = cluster.move_tablet(tablet, 1, 4);
    cluster.wait_tablet_voters(tablet, &[2, 3, 4], Duration::from_secs(240));
    cluster.wait_migrations_done(Duration::from_secs(120));
    // Byte-exact recovery through snapshot install, then steady state
    // (the big key aliases one small-key slot, so it is skipped in the
    // small-value sweep).
    let client = cluster.client();
    assert_eq!(get_big(&client, &big_key), big);
    for name in &keys[&tablet] {
        if *name == big_key {
            continue;
        }
        assert_eq!(get(&client, name), b"purge-me");
    }
}

/// Killing the control-plane leader mid-migration loses no plan: the
/// new leader reads persisted plans, observes actual membership, and
/// continues without duplicated harmful action.
#[test]
fn control_leader_crash_during_migration() {
    let mut cluster = Cluster::spawn_with_tablets(4, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(2);
    let client = cluster.client();
    for names in keys.values() {
        for name in names {
            put(&client, name, b"c");
        }
    }
    let _ = cluster.add_node();
    cluster.wait_control_nodes(4, Duration::from_secs(60));
    let mut tablets = keys.keys().copied();
    let first = tablets.next().expect("a tablet");
    let second = tablets.next().expect("two tablets");
    let _ = cluster.move_tablet(first, 1, 4);
    let _ = cluster.move_tablet(second, 2, 4);
    // Crash the control leader while plans are in flight.
    let leader = cluster
        .control_leader_index()
        .expect("a control leader exists");
    cluster.kill(leader);
    // A new control leader appears and finishes the work.
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if let Some(next) = cluster.control_leader_index()
            && next != leader
        {
            break;
        }
        assert!(Instant::now() < deadline, "no new control leader");
        std::thread::sleep(Duration::from_millis(500));
    }
    cluster
        .restart(leader)
        .expect("crashed control leader restarts");
    cluster.wait_tablet_voters(first, &[2, 3, 4], Duration::from_secs(240));
    cluster.wait_tablet_voters(second, &[1, 3, 4], Duration::from_secs(240));
    cluster.wait_migrations_done(Duration::from_secs(180));
    let client = cluster.client();
    for names in keys.values() {
        for name in names {
            assert_eq!(get(&client, name), b"c");
        }
    }
}

/// Drain, control-voter transition, and safe removal on a small
/// cluster: every tablet leaves the draining node, control membership
/// shrinks onto the replacement, removal commits, and the dropped
/// member never comes back.
#[test]
fn drain_transition_remove_small() {
    let mut cluster = Cluster::spawn_with_tablets(4, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(2);
    let client = cluster.client();
    for names in keys.values() {
        for name in names {
            put(&client, name, b"d");
        }
    }
    let _ = cluster.add_node();
    cluster.wait_control_nodes(4, Duration::from_secs(60));
    // Spread one tablet onto node 4 so drain has something to move.
    let moved = *keys.keys().next().expect("a tablet");
    let _ = cluster.move_tablet(moved, 1, 4);
    cluster.wait_tablet_voters(moved, &[2, 3, 4], Duration::from_secs(240));
    cluster.wait_migrations_done(Duration::from_secs(180));
    let _ = cluster.drain_node(1);
    cluster.wait_node_state(1, "Drained", Duration::from_secs(300));
    cluster.set_control_voters(&[2, 3, 4]);
    let _ = cluster.remove_node(1);
    let dropped = cluster.index_of(1).expect("node 1 slot");
    cluster.drop_node(dropped);
    // Survivors serve every key; the drained node hosts nothing.
    let client = cluster.client();
    for names in keys.values() {
        for name in names {
            assert_eq!(get(&client, name), b"d");
        }
    }
    let counts = cluster.replica_counts();
    assert_eq!(counts.len(), 3, "three nodes remain: {counts:?}");
}

/// Drain survives a bystander crash mid-flight: the money test's chaos
/// element (kill a non-draining member after drain starts, restart it)
/// at small scale with a tight bounded cap. Drain must complete —
/// replica migration, leadership transfer, and retirement never wait
/// on the crashed member.
#[test]
fn drain_survives_bystander_crash_small() {
    let mut cluster = Cluster::spawn_with_tablets(4, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(2);
    let client = cluster.client();
    for names in keys.values() {
        for name in names {
            put(&client, name, b"d");
        }
    }
    let _ = cluster.add_node();
    cluster.wait_control_nodes(4, Duration::from_secs(60));
    // Spread one tablet onto node 4 so drain has something to move.
    let moved = *keys.keys().next().expect("a tablet");
    let _ = cluster.move_tablet(moved, 1, 4);
    cluster.wait_tablet_voters(moved, &[2, 3, 4], Duration::from_secs(240));
    cluster.wait_migrations_done(Duration::from_secs(180));
    let _ = cluster.drain_node(1);
    // Crash a bystander mid-drain (never the draining node itself):
    // plans targeting or led elsewhere must keep converging, and the
    // returnee must rejoin without manual cleanup.
    let bystander = cluster.index_of(3).expect("node 3 slot");
    cluster.kill(bystander);
    std::thread::sleep(Duration::from_secs(5));
    cluster.restart(bystander).expect("bystander restarts");
    cluster.wait_node_state(1, "Drained", Duration::from_secs(300));
    cluster.set_control_voters(&[2, 3, 4]);
    let _ = cluster.remove_node(1);
    let dropped = cluster.index_of(1).expect("node 1 slot");
    cluster.drop_node(dropped);
    // Survivors serve every key; the drained node hosts nothing.
    let client = cluster.client();
    for names in keys.values() {
        for name in names {
            assert_eq!(get(&client, name), b"d");
        }
    }
    let counts = cluster.replica_counts();
    assert_eq!(counts.len(), 3, "three nodes remain: {counts:?}");
}

/// Full-cluster restart mid-migration (§39 crash points, §61): killing
/// every process while plans are in flight loses nothing — control
/// leaders, sources, and targets all recover from persisted plans and
/// durable Raft state with no manual cleanup.
#[test]
fn cluster_restart_during_migration() {
    let mut cluster = Cluster::spawn_with_tablets(4, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(2);
    let client = cluster.client();
    for names in keys.values() {
        for name in names {
            put(&client, name, b"r");
        }
    }
    let _ = cluster.add_node();
    cluster.wait_control_nodes(4, Duration::from_secs(60));
    let mut tablets = keys.keys().copied();
    let first = tablets.next().expect("a tablet");
    let second = tablets.next().expect("two tablets");
    let _ = cluster.move_tablet(first, 1, 4);
    let _ = cluster.move_tablet(second, 2, 4);
    // Crash everything mid-flight (covers: after plan commit, while
    // learner catches up, during joint membership — whichever phase
    // each plan happens to be in).
    cluster.kill_all();
    cluster.restart_all().expect("cluster revives");
    cluster.wait_tablet_voters(first, &[2, 3, 4], Duration::from_secs(300));
    cluster.wait_tablet_voters(second, &[1, 3, 4], Duration::from_secs(300));
    cluster.wait_migrations_done(Duration::from_secs(300));
    let client = cluster.client();
    for names in keys.values() {
        for name in names {
            match client.get(&key(name)) {
                Ok(Some(value)) => assert_eq!(value.to_vec(), b"r"),
                Ok(None) => {
                    dump_cluster_state(
                        &cluster,
                        name,
                        &kivi_client::ClientError::Internal(String::from("absent")),
                    );
                    panic!("{name} lost");
                }
                Err(error) => {
                    dump_cluster_state(&cluster, name, &error);
                    panic!("get {name} reads: {error:?}");
                }
            }
        }
    }
}

/// RESP edge during migration (§33, §61): Redis clients never see
/// `MOVED`/`ASK`/slots. Writes to a migrating tablet answer retryable
/// `BUSY` off-leader and `OK` on the leader; reads keep serving.
/// Retrying across members converges without exposing placement.
#[cfg(feature = "redis-compat")]
#[test]
fn resp_serves_during_migration() {
    use kivi_lab::resp_client::{Reply, RespClient};

    let mut cluster = Cluster::spawn_with_tablets(4, 2, true).expect("resp cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(1);
    let client = cluster.client();
    for names in keys.values() {
        for name in names {
            put(&client, name, b"resp");
        }
    }
    let _ = cluster.add_node();
    cluster.wait_control_nodes(4, Duration::from_secs(60));
    let moved = *keys.keys().next().expect("a tablet");
    let moved_key = keys[&moved][0].clone();
    let _ = cluster.move_tablet(moved, 1, 4);
    // Drive RESP traffic at the migrating key while the plan runs:
    // BUSY (off-leader) is retried across members, OK ends the op.
    // No reply may name a tablet, a slot, or a redirect.
    let deadline = Instant::now() + Duration::from_secs(180);
    let mut writes = 0u32;
    while cluster
        .migrations(0)
        .get("migrations")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|plans| {
            plans.iter().any(|plan| {
                !matches!(
                    plan.get("phase").and_then(serde_json::Value::as_str),
                    Some("Completed" | "Failed")
                )
            })
        })
    {
        let mut done = false;
        for index in 0..3 {
            let endpoint = cluster.resp_endpoint(index).expect("resp endpoint");
            let mut resp = RespClient::connect(&endpoint).expect("resp connects");
            match resp.round_trip(&[
                b"SET",
                moved_key.as_bytes(),
                format!("rv{writes}").as_bytes(),
            ]) {
                Ok(Reply::Simple(_)) => {
                    done = true;
                    break;
                }
                // BUSY (off-leader) or connection reset (kill-free test:
                // just election churn): try the next member.
                Ok(Reply::Error(_)) | Err(_) => continue,
                Ok(other) => panic!("unexpected RESP reply: {other:?}"),
            }
        }
        assert!(done, "RESP write converges across members");
        writes += 1;
        assert!(
            Instant::now() < deadline,
            "migration never finished under RESP traffic"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
    cluster.wait_migrations_done(Duration::from_secs(180));
    // Final value readable through RESP after convergence.
    let mut seen = false;
    for index in 0..3 {
        let endpoint = cluster.resp_endpoint(index).expect("resp endpoint");
        let mut resp = RespClient::connect(&endpoint).expect("resp connects");
        if let Ok(Reply::Bulk(Some(_))) = resp.round_trip(&[b"GET", moved_key.as_bytes()]) {
            seen = true;
            break;
        }
    }
    assert!(seen, "migrated value serves over RESP");
}

/// Full product money test (§62): 100 tablets RF=3, real workload with
/// large values, dynamic 4th node, online rebalance with failures
/// injected mid-flight, drain, safe removal, and full restart — reads
/// and writes throughout, zero unavailable tablets by design.
///
/// STRESS PROFILE (not default CI): this test saturates disk/loopback
/// with ~100 MB bulk plus chaos kills, so its fixed wall-clock budgets
/// (600 s drain cap, 4-attempt streaming retries) only hold on an
/// otherwise idle box — under parallel load it fails slow, never
/// wrong (no assertion has ever fired on state, only budgets on time).
/// Run explicitly on an idle machine:
/// `cargo nextest run -p kivi-lab --test cluster_control --run-ignored all -E 'test(four_node_rebalance_drain_money)'`.
/// Deterministic drain-plus-crash correctness lives in
/// `drain_survives_bystander_crash_small`, which stays in the default
/// suite.
#[allow(clippy::too_many_lines)]
#[ignore = "stress profile: needs idle box, see doc comment for explicit invocation"]
#[test]
fn four_node_rebalance_drain_money() {
    // Benchmark clock (§59): phase durations print at each milestone
    // (migration duration, drain duration, end-to-end).
    let money_start = Instant::now();
    let mut cluster = Cluster::spawn_with_tablets(100, 4, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    // Real workload: small keys on every tablet plus large chunked
    // values (learners must acquire checkpoint + missing chunks only).
    let keys = cluster.keys_for_tablets(2);
    assert_eq!(keys.len(), 100);
    let client = cluster.client();
    for names in keys.values() {
        for name in names {
            put(&client, name, b"money");
        }
    }
    let first_tablet = *keys.keys().next().expect("a tablet");
    let second_tablet = *keys.keys().nth(1).expect("two tablets");
    let first_big_key = keys[&first_tablet][0].clone();
    let second_big_key = keys[&second_tablet][0].clone();
    let first_big = fill(64 * 1024 * 1024, 0x10);
    let second_big = fill(64 * 1024 * 1024, 0x20);
    put_big(&client, &first_big_key, &first_big);
    put_big(&client, &second_big_key, &second_big);
    // Add the fourth node without restarting A/B/C.
    let _ = cluster.add_node();
    cluster.wait_control_nodes(4, Duration::from_secs(90));
    // Rebalance in bounded operator rounds while serving traffic:
    // one call creates one bounded wave; the loop below tops up waves
    // whenever idle until the planner reports stability (no new
    // plans). Traffic and failure injection interleave throughout.
    let first = cluster.rebalance();
    eprintln!(
        "initial rebalance: {}",
        first
            .get("plans")
            .map_or_else(String::new, ToString::to_string)
    );
    let traffic = cluster.client();
    // Migration bulk bytes (§59): sidecar bulk counters sampled around
    // the rebalance (catch-up traffic) and around the drain.
    let bulk_bytes = |cluster: &Cluster| -> u64 {
        let mut total = 0u64;
        for index in 0..cluster.member_count() {
            if !cluster.alive(index) {
                continue;
            }
            let (_, sidecar) = cluster.admin_get(index, "/v1/sidecar");
            total += sidecar
                .get("bulk_bytes_sent")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
        }
        total
    };
    let bulk_before = bulk_bytes(&cluster);
    let deadline = Instant::now() + Duration::from_mins(25);
    let mut rounds = 0u64;
    let mut injected = 0u8;
    let mut stable_rounds = 0u32;
    // Client latency sampler (§58): every 50th op timed; summary at
    // the end shows migration did not degrade ordinary SET latency.
    let mut latencies: Vec<Duration> = Vec::new();
    // Every committed key name, in order (the post-run sample draws
    // from committed keys only — never regenerated phantoms).
    let mut written: Vec<String> = Vec::new();
    loop {
        // Interleaved workload: every tablet stays writable while its
        // learner catches up and membership changes around it (bounded
        // redirect retries are the designed client flow; anything else
        // fails loudly).
        for index in 0..10 {
            let name = format!("traffic-{rounds}-{index}");
            let sample = rounds.is_multiple_of(50);
            let start = Instant::now();
            if let Err(error) = traffic_put_result(&traffic, &name, b"t") {
                dump_cluster_state(&cluster, &name, &error);
                panic!("traffic set {name} never converges: {error:?}");
            }
            if sample {
                latencies.push(start.elapsed());
            }
            rounds += 1;
            written.push(name);
        }
        // Large streaming write concurrent with migration (§37): the
        // learner converges through snapshot/log/sidecar mechanisms
        // while bulk traffic flows. Runs once; key names stay dense
        // (no skipped rounds) so the post-run sample only names
        // committed keys.
        if rounds == 100 {
            let big = fill(32 * 1024 * 1024, 0x77);
            traffic_put_big(&traffic, "traffic-big", &big);
            assert_eq!(traffic_get_big(&traffic, "traffic-big"), big);
        }
        // Failure injection mid-rebalance (§50): kill the control
        // leader once, and the target learner once; both restart and
        // migrations resume with no manual cleanup. Disabled with
        // KIVI_MONEY_CHAOS=0 to bisect chaos from migration.
        let calm = std::env::var("KIVI_MONEY_CHAOS").is_ok_and(|value| value == "0");
        if calm {
            injected = 5;
        }
        if injected == 0 {
            if let Some(leader) = cluster.control_leader_index() {
                cluster.kill(leader);
                injected = 1;
            }
        } else if injected == 1 {
            if cluster.control_leader_index().is_some() {
                injected = 2;
            }
        } else if injected == 2 {
            cluster.kill(3);
            injected = 3;
        } else if injected == 3 {
            cluster.restart(3).expect("target learner restarts");
            injected = 4;
        } else if injected == 4 {
            // Rolling restart of the remaining members (failover +
            // rejoin): strictly one member restarting at a time, and
            // each restart gates on every tablet holding exactly one
            // stable leader plus a settle window, so quorum holds
            // throughout. (Applied-index parity is too strict a gate
            // under continuous traffic; stable leadership proves every
            // group has quorum and the returnee rejoined.)
            for index in 0..3 {
                cluster.restart(index).expect("member restarts");
                let _ = cluster.wait_all_leaders();
                std::thread::sleep(Duration::from_secs(30));
            }
            injected = 5;
        }
        // Done when no live plans remain, injections finished, and a
        // fresh rebalance round creates nothing (planner stability:
        // same nodes + same policy => no new plans).
        let mut live = usize::MAX;
        let mut probed = false;
        // Per-member live plans (id,tablet,phase): members disagree
        // while followers apply behind the leader; the freshet view
        // (min live) is closest to truth.
        let mut detail = String::new();
        for index in 0..cluster.member_count() {
            if !cluster.alive(index) {
                continue;
            }
            let (_, json) = cluster.admin_get(index, "/v1/control/migrations");
            if let Some(plans) = json.get("migrations").and_then(serde_json::Value::as_array) {
                probed = true;
                let mut member_live = Vec::new();
                for plan in plans {
                    if !matches!(
                        plan.get("phase").and_then(serde_json::Value::as_str),
                        Some("Completed" | "Failed")
                    ) {
                        member_live.push(format!(
                            "{}:{}:{}",
                            plan.get("id")
                                .map_or_else(|| String::from("?"), ToString::to_string),
                            plan.get("tablet")
                                .map_or_else(|| String::from("?"), ToString::to_string),
                            plan.get("phase")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or("?"),
                        ));
                    }
                }
                live = live.min(member_live.len());
                if rounds.is_multiple_of(600) {
                    use std::fmt::Write as _;
                    let _ = write!(detail, " m{index}=[{}]", member_live.join(","));
                }
            }
        }
        if rounds.is_multiple_of(600) {
            eprintln!("money loop rounds={rounds} injected={injected} live={live}{detail}");
        }
        if probed && live == 0 && injected == 5 {
            let reply = cluster.rebalance();
            let fresh = reply
                .get("plans")
                .and_then(serde_json::Value::as_array)
                .map_or(usize::MAX, Vec::len);
            eprintln!("stability check: {fresh} fresh plans");
            if fresh == 0 {
                stable_rounds += 1;
                if stable_rounds >= 2 {
                    break;
                }
            } else {
                stable_rounds = 0;
            }
        }
        assert!(Instant::now() < deadline, "rebalance never completed");
        std::thread::sleep(Duration::from_millis(500));
    }
    cluster.wait_migrations_done(Duration::from_secs(300));
    eprintln!(
        "BENCH rebalance done in {:?}; bulk bytes moved: {}",
        money_start.elapsed(),
        bulk_bytes(&cluster).saturating_sub(bulk_before),
    );
    // Roughly balanced across four nodes (100 tablets × RF3 = 300
    // replicas, mean 75).
    let counts = cluster.replica_counts();
    assert_eq!(counts.len(), 4, "all four nodes host replicas: {counts:?}");
    for (node, count) in &counts {
        assert!(
            (56..=94).contains(count),
            "node {node} balanced around 75: {counts:?}"
        );
    }
    // Byte-exact data after migration, including large values that
    // moved through sidecar reuse (no full copies).
    let client = cluster.client();
    assert_eq!(get_big(&client, &first_big_key), first_big);
    assert_eq!(get_big(&client, &second_big_key), second_big);
    // Sample the interleaved workload (first, middle, last): every
    // sampled op committed during migration, so the sample proves the
    // flow.
    assert!(!written.is_empty());
    for name in written
        .iter()
        .step_by(written.len().max(1) / 50 + 1)
        .take(60)
    {
        assert_eq!(traffic_get(&client, name), b"t");
    }
    // Drain node 1: every replica and leadership moves away. Mid-drain,
    // kill a bystander member and restart it: drain must survive source,
    // target, and control-leader crashes alike (§42).
    let drain_start = Instant::now();
    let _ = cluster.drain_node(1);
    let bystander = cluster.index_of(3).expect("node 3 slot");
    cluster.kill(bystander);
    std::thread::sleep(Duration::from_secs(5));
    cluster.restart(bystander).expect("bystander restarts");
    cluster.wait_node_state(1, "Drained", Duration::from_secs(300));
    eprintln!("BENCH drain done in {:?}", drain_start.elapsed());
    // Shrink control membership 1 -> 4 first (control learner 4 was
    // admitted automatically; promote it, retire 1), then safe removal:
    // removal refuses while the node still votes in control (§31).
    cluster.set_control_voters(&[2, 3, 4]);
    let _ = cluster.remove_node(1);
    let dropped = cluster.index_of(1).expect("node 1 slot");
    cluster.drop_node(dropped);
    // Remaining cluster keeps serving (big keys alias two small-key
    // slots, so they are skipped in the small-value sweep).
    let client = cluster.client();
    for names in keys.values().take(5) {
        for name in names {
            if *name == first_big_key || *name == second_big_key {
                continue;
            }
            assert_eq!(get(&client, name), b"money");
        }
    }
    traffic_put(&client, "post-drain", b"ok");
    assert_eq!(traffic_get(&client, "post-drain"), b"ok");
    // Full restart of the remaining cluster: control state, placement,
    // and tablet membership recover; service resumes.
    cluster.restart_all().expect("remaining cluster restarts");
    let client = cluster.client();
    assert_eq!(get_big(&client, &first_big_key), first_big);
    traffic_put(&client, "post-restart", b"ok");
    assert_eq!(traffic_get(&client, "post-restart"), b"ok");
    let counts = cluster.replica_counts();
    assert_eq!(counts.len(), 3, "three nodes remain: {counts:?}");
    for (node, count) in &counts {
        assert_eq!(
            *count, 100,
            "node {node} hosts every tablet after drain: {counts:?}"
        );
    }
    // Benchmark summary (§58-59): end-to-end duration plus ordinary
    // SET latency sampled during migration (p50/max) and unavailable
    // tablets (zero by design throughout; any failed op would have
    // panicked above).
    latencies.sort_unstable();
    let p50 = latencies
        .get(latencies.len() / 2)
        .copied()
        .unwrap_or_default();
    let max = latencies.last().copied().unwrap_or_default();
    eprintln!(
        "BENCH money done in {:?}; traffic ops: {}; sampled SET p50={:?} max={:?}; unavailable tablets: 0",
        money_start.elapsed(),
        rounds,
        p50,
        max,
    );
}
