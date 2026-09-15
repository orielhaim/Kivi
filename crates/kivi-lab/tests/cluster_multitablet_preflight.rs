//! Sidecar preflight over real multi-tablet clusters: large values move
//! before the tiny Raft root proposes, so same-tablet small writes stay
//! responsive and other tablets stay isolated.
//!
//! Covers the preflight money tests: same-tablet latency (§45),
//! cross-tablet isolation (§46), quorum-ready with a slow follower (§47),
//! append-gate fallback with preflight disabled (§48), leader loss during
//! preflight (§49), and the stale-state race (§50).

use std::time::{Duration, Instant};

use kivi_lab::cluster::Cluster;
use kivi_state::Key;
use kivi_types::{RequestSeq, SessionId};

fn key(name: &str) -> Key {
    Key::from(name)
}

/// Deterministic pseudo-random fill (splitmix64).
fn fill(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut out = vec![0u8; len];
    for byte in &mut out {
        state = state
            .wrapping_add(0x9E37_79B9_7F4A_7C15)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state ^= state >> 29;
        *byte = (state % 251) as u8;
    }
    out
}

fn put_stream(client: &kivi_client::NativeClient, name: &str, value: &[u8]) {
    let mut source: &[u8] = value;
    client
        .put_stream(&key(name), Some(value.len() as u64), &mut source)
        .unwrap_or_else(|error| panic!("put_stream {name} commits: {error:?}"));
}

fn get_stream(client: &kivi_client::NativeClient, name: &str) -> Vec<u8> {
    client
        .get_stream(&key(name))
        .unwrap_or_else(|error| panic!("get_stream {name} reads: {error:?}"))
        .unwrap_or_else(|| panic!("{name} present"))
        .to_vec()
}

/// Sorted percentiles (ms) of client-measured latencies.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn percentiles(mut latencies: Vec<Duration>) -> (f64, f64, f64, f64) {
    latencies.sort_unstable();
    let at = |q: f64| {
        let rank = (q * latencies.len() as f64)
            .ceil()
            .max(1.0)
            .min(latencies.len() as f64);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let index = rank as usize - 1;
        latencies[index].as_secs_f64() * 1000.0
    };
    (at(0.50), at(0.95), at(0.99), at(0.999))
}

/// Sum of `/v1/preflight` attempts across live members.
fn preflight_attempts(cluster: &Cluster) -> u64 {
    (0..3)
        .filter(|i| cluster.alive(*i))
        .map(|i| {
            let (_, body) = cluster.admin_get(i, "/v1/preflight");
            body["attempts"].as_u64().unwrap_or(0)
        })
        .sum()
}

/// Sum of quorum-ready rounds across live members.
fn preflight_ready(cluster: &Cluster) -> u64 {
    (0..3)
        .filter(|i| cluster.alive(*i))
        .map(|i| {
            let (_, body) = cluster.admin_get(i, "/v1/preflight");
            body["quorum_ready"].as_u64().unwrap_or(0)
        })
        .sum()
}

/// Sum of append-gate fallbacks across live members.
fn preflight_fallbacks(cluster: &Cluster) -> u64 {
    (0..3)
        .filter(|i| cluster.alive(*i))
        .map(|i| {
            let (_, body) = cluster.admin_get(i, "/v1/preflight");
            body["fallbacks"].as_u64().unwrap_or(0)
        })
        .sum()
}

/// Same-tablet preflight (§45): while a 64 MiB value's sidecars move
/// before its root proposes, 100 small SETs on the SAME tablet keep
/// committing — the seconds-long queue of the old append-blocking design
/// (p99 ≈ 2.3 s) is gone.
#[test]
fn preflight_same_tablet_small_writes_stay_fast() {
    let cluster = Cluster::spawn_with_tablets(4, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(101);
    let tablet_a = *keys.keys().next().expect("a tablet");
    let names = keys[&tablet_a].clone();
    let (big_name, small_names) = (names[0].clone(), names[1..].to_vec());
    let value = fill(64 * 1024 * 1024, 0x45);
    let started = Instant::now();
    let latencies = std::thread::scope(|scope| {
        let upload_client = cluster.client();
        let upload = scope.spawn(move || {
            let mut source: &[u8] = &value;
            upload_client.put_stream(&key(&big_name), Some(value.len() as u64), &mut source)
        });
        // Give the upload a head start so the small writes overlap bulk
        // movement (staging 64 MiB takes seconds in debug).
        std::thread::sleep(Duration::from_secs(2));
        let client = cluster.client();
        let mut latencies = Vec::with_capacity(100);
        for (i, name) in small_names.iter().enumerate() {
            let at = Instant::now();
            client
                .set(&key(name), bytes::Bytes::from(format!("s-{i}")))
                .unwrap_or_else(|error| panic!("small set {i} commits during upload: {error:?}"));
            latencies.push(at.elapsed());
        }
        let outcome = upload.join().expect("upload thread joins");
        assert!(outcome.is_ok(), "64 MiB upload commits: {outcome:?}");
        latencies
    });
    let total = started.elapsed();
    let (p50, p95, p99, p999) = percentiles(latencies);
    eprintln!("same-tablet small writes during 64 MiB upload: total {total:?}");
    eprintln!("  p50 {p50:.2} ms, p95 {p95:.2} ms, p99 {p99:.2} ms, p99.9 {p999:.2} ms");
    eprintln!(
        "  preflight attempts {}, quorum-ready {}",
        preflight_attempts(&cluster),
        preflight_ready(&cluster)
    );
    assert!(
        p99 < 1500.0,
        "same-tablet p99 {p99:.2} ms must stay far below the old 2.3 s queue"
    );
    assert!(preflight_ready(&cluster) >= 1, "preflight reached quorum");
    cluster.wait_converged_all();
    assert_eq!(
        get_stream(&cluster.client(), &names[0]).len(),
        64 * 1024 * 1024
    );
}

/// Cross-tablet isolation (§46): a 32 MiB upload on tablet A leaves
/// tablet B latency nearly untouched — no Raft-log head-of-line crosses
/// tablet boundaries (only shared CPU/storage/network couple them).
#[test]
fn preflight_other_tablet_stays_isolated() {
    let cluster = Cluster::spawn_with_tablets(4, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(51);
    let mut tablets = keys.keys().copied();
    let (tablet_a, tablet_b) = (tablets.next().expect("A"), tablets.next().expect("B"));
    let names_a = keys[&tablet_a].clone();
    let names_b = keys[&tablet_b][..50].to_vec();
    let value = fill(32 * 1024 * 1024, 0x46);
    let upload_name = names_a[0].clone();
    let latencies = std::thread::scope(|scope| {
        let upload_client = cluster.client();
        let upload = scope.spawn(move || {
            let mut source: &[u8] = &value;
            upload_client.put_stream(&key(&upload_name), Some(value.len() as u64), &mut source)
        });
        std::thread::sleep(Duration::from_secs(2));
        let client = cluster.client();
        let mut latencies = Vec::with_capacity(50);
        for (i, name) in names_b.iter().enumerate() {
            let at = Instant::now();
            client
                .set(&key(name), bytes::Bytes::from(format!("b-{i}")))
                .unwrap_or_else(|error| panic!("tablet B set {i} commits: {error:?}"));
            latencies.push(at.elapsed());
        }
        let outcome = upload.join().expect("upload thread joins");
        assert!(outcome.is_ok(), "tablet A upload commits: {outcome:?}");
        latencies
    });
    let (p50, p95, p99, p999) = percentiles(latencies);
    eprintln!("other-tablet writes during 32 MiB upload on A:");
    eprintln!("  p50 {p50:.2} ms, p95 {p95:.2} ms, p99 {p99:.2} ms, p99.9 {p999:.2} ms");
    assert!(
        p99 < 1000.0,
        "cross-tablet p99 {p99:.2} ms shows no Raft HOL"
    );
    cluster.wait_converged_all();
    assert_eq!(
        get_stream(&cluster.client(), &names_a[0]).len(),
        32 * 1024 * 1024
    );
}

/// Quorum-ready with a slow follower (§47): with the A↔C bulk path down,
/// A+B (quorum) still commit the root; C catches up later through the
/// append gate.
#[test]
fn preflight_quorum_with_slow_follower_commits() {
    let cluster = Cluster::spawn_with_tablets(4, 2, false).expect("cluster spawns");
    let leaders = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(1);
    let tablet_a = *keys.keys().next().expect("a tablet");
    let leader_a = leaders
        .iter()
        .find(|(tablet, _)| *tablet == tablet_a)
        .map(|(_, leader)| *leader)
        .expect("A leader");
    let node_of = |index: usize| cluster.node_info(index)["node"].as_u64().expect("node id");
    let follower_c = (0..3).find(|i| *i != leader_a).expect("a follower");
    // Sever only the leader↔C links (B stays connected to both).
    cluster.suspend_peer(leader_a, node_of(follower_c));
    cluster.suspend_peer(follower_c, node_of(leader_a));
    let name = keys[&tablet_a][0].clone();
    let value = fill(8 * 1024 * 1024, 0x47);
    put_stream(&cluster.client(), &name, &value);
    // Committed without C's preflight: quorum (leader + B) sufficed.
    // Heal both directions (heal() resumes every link toward/away).
    cluster.heal(leader_a);
    cluster.wait_converged_all();
    assert_eq!(get_stream(&cluster.client(), &name), value);
    // C's applied coverage for A proves its append gate fetched the
    // sidecars after healing.
    let applied: Vec<u64> = (0..3)
        .map(|i| cluster.tablets_applied(i)[&tablet_a])
        .collect();
    assert_eq!(applied[0], applied[1]);
    assert_eq!(applied[1], applied[2]);
}

/// Preflight is only an optimization (§48): with preflight disabled, the
/// old safe path (root arrives → gate fetches → durable → ACK) still
/// commits correctly.
#[test]
fn preflight_disabled_falls_back_to_append_gate() {
    let cluster = Cluster::spawn_full(4, 2, false, &[("KIVI_DISABLE_PREFLIGHT", "1")])
        .expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(1);
    let name = keys
        .keys()
        .next()
        .map(|tablet| keys[tablet][0].clone())
        .expect("a key");
    let value = fill(8 * 1024 * 1024, 0x48);
    put_stream(&cluster.client(), &name, &value);
    cluster.wait_converged_all();
    assert_eq!(get_stream(&cluster.client(), &name), value);
    assert!(
        preflight_fallbacks(&cluster) >= 1,
        "disabled preflight counts append-gate fallbacks"
    );
}

/// Leader loss during preflight (§49): sidecars alone never commit — the
/// key stays absent or fully present, and the same identity retried on
/// the new leader converges exactly once.
#[test]
fn preflight_leader_loss_abandons_without_commit() {
    let mut cluster = Cluster::spawn_with_tablets(4, 2, false).expect("cluster spawns");
    let leaders = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(1);
    let tablet_a = *keys.keys().next().expect("a tablet");
    let leader_a = leaders
        .iter()
        .find(|(tablet, _)| *tablet == tablet_a)
        .map(|(_, leader)| *leader)
        .expect("A leader");
    let session = SessionId::from_u128(0x9E1E_F117_0000_0001u128);
    let name = keys[&tablet_a][0].clone();
    let value = fill(32 * 1024 * 1024, 0x49);
    let client = cluster.client_with_session(session);
    let upload_name = name.clone();
    let attempt = std::thread::scope(|scope| {
        let upload = scope.spawn(|| {
            let mut source: &[u8] = &value;
            client.put_stream_with_seq(
                &key(&upload_name),
                Some(value.len() as u64),
                &mut source,
                Some(RequestSeq::from_u64(1)),
            )
        });
        std::thread::sleep(Duration::from_millis(200));
        cluster.kill(leader_a);
        upload.join().expect("upload thread joins")
    });
    let _ = cluster.wait_all_leaders();
    if attempt.is_ok() {
        cluster.wait_converged_all();
        assert_eq!(get_stream(&cluster.client(), &name), value);
    } else {
        cluster.wait_converged_all();
        assert!(
            cluster
                .client()
                .get_stream(&key(&name))
                .expect("reads")
                .is_none(),
            "abandoned preflight commits nothing"
        );
        // New leader reuses distributed chunks by manifest id; the same
        // identity converges exactly once.
        let mut retry: &[u8] = &value;
        cluster
            .client_with_session(session)
            .put_stream_with_seq(
                &key(&name),
                Some(value.len() as u64),
                &mut retry,
                Some(RequestSeq::from_u64(1)),
            )
            .expect("retry converges");
        assert_eq!(get_stream(&cluster.client(), &name), value);
    }
    cluster.restart(leader_a).expect("old leader restarts");
    cluster.wait_converged_all();
}

/// Stale-state race (§50): a large write started before a small write to
/// the same key still obeys proposal ordering — the large root proposes
/// after its preflight, so it wins deterministically.
#[test]
fn preflight_stale_state_race_last_proposal_wins() {
    let cluster = Cluster::spawn_with_tablets(4, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(1);
    let name = keys
        .keys()
        .next()
        .map(|tablet| keys[tablet][0].clone())
        .expect("a key");
    let large = fill(16 * 1024 * 1024, 0x50);
    let small = b"small".to_vec();
    let upload_name = name.clone();
    std::thread::scope(|scope| {
        let upload_client = cluster.client();
        let upload = scope.spawn(move || {
            let mut source: &[u8] = &large;
            upload_client.put_stream(&key(&upload_name), Some(large.len() as u64), &mut source)
        });
        // Wait until the leader's preflight round actually started
        // (condition-driven, not a blind sleep), then commit the small
        // write mid-preflight.
        let deadline = Instant::now() + Duration::from_secs(120);
        while preflight_attempts(&cluster) == 0 {
            assert!(Instant::now() < deadline, "preflight never starts");
            std::thread::sleep(Duration::from_millis(100));
        }
        cluster
            .client()
            .set(&key(&name), bytes::Bytes::from(small.clone()))
            .expect("small write commits during preflight");
        let outcome = upload.join().expect("upload thread joins");
        assert!(outcome.is_ok(), "large upload commits after: {outcome:?}");
    });
    cluster.wait_converged_all();
    assert_eq!(
        get_stream(&cluster.client(), &name).len(),
        16 * 1024 * 1024,
        "the later proposal (large) wins; no stale overwrite the other way"
    );
}
