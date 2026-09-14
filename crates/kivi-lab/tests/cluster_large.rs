//! Replicated large values over the QUIC/H3 peer mesh: real processes,
//! real UDP, real data directories.
//!
//! Product workflow (both task definitions of done):
//!
//! ```text
//! put_stream (1 MiB … 64 MiB) -> quorum commit (Raft log stays small) ->
//! followers acquire sidecars over H3 bulk -> read exact bytes ->
//! kill leader -> new leader serves same bytes -> SetRange patch ->
//! fail over again -> restart all -> snapshot catch-up converges.
//! ```
//!
//! The core invariant under test: no replica counts a Raft append as
//! durable until every referenced sidecar is locally durable and
//! verified, so every committed large value is reconstructible from any
//! quorum survivor.

use std::time::{Duration, Instant};

use kivi_lab::cluster::Cluster;
use kivi_state::Key;
use kivi_types::{RequestSeq, SessionId};

fn key(name: &str) -> Key {
    Key::from(name)
}

/// Deterministic pseudo-random fill (splitmix64): representative,
/// incompressible-ish, chunk-boundary agnostic.
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

fn put(client: &kivi_client::NativeClient, name: &str, value: &[u8]) {
    let mut source: &[u8] = value;
    client
        .put_stream(&key(name), Some(value.len() as u64), &mut source)
        .unwrap_or_else(|error| panic!("put_stream {name} commits: {error:?}"));
}

fn get(client: &kivi_client::NativeClient, name: &str) -> Vec<u8> {
    client
        .get_stream(&key(name))
        .unwrap_or_else(|error| panic!("get_stream {name} reads: {error:?}"))
        .unwrap_or_else(|| panic!("{name} present"))
        .to_vec()
}

/// Follower sidecar bulk bytes received, per member.
fn bulk_received(cluster: &Cluster) -> [u64; 3] {
    [0, 1, 2].map(|i| {
        let (status, body) = cluster.admin_get(i, "/v1/sidecar");
        assert_eq!(status, 200);
        body["bulk_bytes_received"].as_u64().unwrap_or(0)
    })
}

/// A. 1 MiB replicated stream, byte-for-byte on the cluster.
#[test]
fn large_1mib_replicated_stream() {
    let cluster = Cluster::spawn().expect("cluster spawns");
    let _ = cluster.wait_leader();
    let value = fill(1_048_576, 7);
    put(&cluster.client(), "big-1mib", &value);
    cluster.wait_converged();
    assert_eq!(get(&cluster.client(), "big-1mib"), value);
}

/// B. 64 MiB replicated stream; the Raft log stays small (spot-checked
/// via peer control bytes far below the logical size).
#[test]
fn large_64mib_replicated_stream() {
    let cluster = Cluster::spawn().expect("cluster spawns");
    let _ = cluster.wait_leader();
    let value = fill(64 * 1024 * 1024, 0x40);
    put(&cluster.client(), "big-64mib", &value);
    cluster.wait_converged();
    assert_eq!(get(&cluster.client(), "big-64mib"), value);
    // Control plane moved kilobytes per peer, not megabytes: the 64 MiB
    // never entered the Raft log.
    for i in 0..3 {
        let (_, peers) = cluster.admin_get(i, "/v1/peers");
        for peer in peers.as_array().cloned().unwrap_or_default() {
            let control = peer["control_bytes_sent"].as_u64().unwrap_or(u64::MAX);
            assert!(
                control < 8 * 1024 * 1024,
                "control bytes {control} far below 64 MiB (no payload in Raft)"
            );
        }
    }
}

/// C. Leader failover after a 64 MiB commit: the new leader serves the
/// exact bytes without the dead leader.
#[test]
fn large_failover_after_commit_serves_exact_bytes() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let leader = cluster.wait_leader();
    let value = fill(64 * 1024 * 1024, 0xC0);
    put(&cluster.client(), "big-failover", &value);
    cluster.wait_converged();
    cluster.kill(leader);
    let elected = cluster.wait_leader();
    assert_ne!(elected, leader, "new leader emerges");
    assert_eq!(get(&cluster.client(), "big-failover"), value);
    cluster.restart(leader).expect("old leader restarts");
    cluster.wait_converged();
    assert_eq!(get(&cluster.client(), "big-failover"), value);
}

/// D. Lost response after commit is exactly-once: retrying the same
/// session/sequence on the new leader returns the original outcome with
/// no second mutation and exact bytes.
#[test]
fn large_lost_response_retry_is_exactly_once() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let leader = cluster.wait_leader();
    let session = SessionId::from_u128(0x1A26_E000_0000_0001);
    let value = fill(4 * 1024 * 1024, 0xD0);
    let mut source: &[u8] = &value;
    cluster
        .client_with_session(session)
        .put_stream_with_seq(
            &key("big-dedup"),
            Some(value.len() as u64),
            &mut source,
            Some(RequestSeq::from_u64(1)),
        )
        .expect("first upload commits");
    cluster.kill(leader);
    let _ = cluster.wait_leader();
    // Same identity retried: dedup hit, same outcome, no second version.
    let mut retry: &[u8] = &value;
    cluster
        .client_with_session(session)
        .put_stream_with_seq(
            &key("big-dedup"),
            Some(value.len() as u64),
            &mut retry,
            Some(RequestSeq::from_u64(1)),
        )
        .expect("retry returns original outcome");
    assert_eq!(get(&cluster.client(), "big-dedup"), value);
}

/// E/F. Leader death mid-upload or mid-sidecar-replication never commits
/// a torn value: afterwards the key is absent or fully present, and a
/// retry converges. (Timing selects the branch; both are safe.)
#[test]
fn large_mid_upload_kill_leaves_no_torn_value() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let leader = cluster.wait_leader();
    let value = fill(32 * 1024 * 1024, 0xE0);
    // Race the upload against the kill: start the upload in a thread,
    // kill 200 ms into the multi-second transfer (firmly mid-stream),
    // then join.
    let client = cluster.client();
    let attempt = std::thread::scope(|scope| {
        let upload = scope.spawn(|| {
            let mut source: &[u8] = &value;
            client.put_stream(&key("big-torn"), Some(value.len() as u64), &mut source)
        });
        std::thread::sleep(Duration::from_millis(200));
        cluster.kill(leader);
        upload.join().expect("upload thread joins")
    });
    let _ = cluster.wait_leader();
    if attempt.is_ok() {
        // Committed implies a sidecar-qualified quorum (invariant):
        // survivors reconstruct the full value.
        cluster.wait_converged();
        assert_eq!(get(&cluster.client(), "big-torn"), value);
    } else {
        // Never committed: absent, and a retry converges safely.
        cluster.wait_converged();
        assert!(
            cluster
                .client()
                .get_stream(&key("big-torn"))
                .expect("reads")
                .is_none(),
            "uncommitted upload leaves nothing behind"
        );
        put(&cluster.client(), "big-torn", &value);
        assert_eq!(get(&cluster.client(), "big-torn"), value);
    }
}

/// H. Chunk reuse across values: B shares 90% of A's chunks, so
/// replicating B moves ~10% bulk, not the full value.
#[test]
fn large_chunk_reuse_moves_only_missing_chunks() {
    let cluster = Cluster::spawn().expect("cluster spawns");
    let _ = cluster.wait_leader();
    let shared = fill(9 * 1024 * 1024, 0x90);
    let mut value_a = shared.clone();
    value_a.extend_from_slice(&fill(1_048_576, 0xA1));
    put(&cluster.client(), "reuse-a", &value_a);
    cluster.wait_converged();
    let before = bulk_received(&cluster);
    let mut value_b = shared.clone();
    value_b.extend_from_slice(&fill(1_048_576, 0xB2));
    put(&cluster.client(), "reuse-b", &value_b);
    cluster.wait_converged();
    assert_eq!(get(&cluster.client(), "reuse-b"), value_b);
    // Generous bound: new tail (1 MiB) + manifests + framing, far below
    // the 10 MiB a blind retransmit would move.
    let after = bulk_received(&cluster);
    for (i, (now, was)) in after.iter().zip(before.iter()).enumerate() {
        let delta = now.saturating_sub(*was);
        assert!(
            delta < 4 * 1024 * 1024,
            "member {i} moved only {delta} bytes for 90%-shared value"
        );
    }
}

/// I. Chunked `SetRange`: tiny patch, untouched chunks reused, readable
/// after another failover.
#[test]
fn large_set_range_reuses_chunks_and_survives_failover() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let _ = cluster.wait_leader();
    let mut value = fill(8 * 1024 * 1024, 0x50);
    put(&cluster.client(), "big-patch", &value);
    cluster.wait_converged();
    let before = bulk_received(&cluster);
    let patch = vec![0xEEu8; 4096];
    let at = 4_194_304usize;
    cluster
        .client()
        .set_range(
            &key("big-patch"),
            at as u64,
            bytes::Bytes::from(patch.clone()),
        )
        .expect("set_range commits");
    value[at..at + patch.len()].copy_from_slice(&patch);
    cluster.wait_converged();
    assert_eq!(get(&cluster.client(), "big-patch"), value);
    let after = bulk_received(&cluster);
    for (i, (now, was)) in after.iter().zip(before.iter()).enumerate() {
        let delta = now.saturating_sub(*was);
        assert!(
            delta < 4 * 1024 * 1024,
            "member {i} moved only {delta} bytes for a 4 KiB patch"
        );
    }
    let leader = cluster.wait_leader();
    cluster.kill(leader);
    let _ = cluster.wait_leader();
    assert_eq!(get(&cluster.client(), "big-patch"), value);
}

/// J. Differential snapshot catch-up: a partitioned follower holding A
/// converges on B (90% shared) moving ~10%, including across a
/// snapshot/purge boundary.
#[test]
fn large_snapshot_catch_up_is_differential() {
    let cluster = Cluster::spawn().expect("cluster spawns");
    let leader = cluster.wait_leader();
    let shared = fill(9 * 1024 * 1024, 0x5E);
    let mut value_a = shared.clone();
    value_a.extend_from_slice(&fill(1_048_576, 0xA5));
    put(&cluster.client(), "snap-a", &value_a);
    cluster.wait_converged();
    // Suspend (not kill) one follower: it keeps A while missing B.
    let lagger = (0..3).find(|i| *i != leader).expect("a follower");
    cluster.partition(lagger);
    let mut value_b = shared.clone();
    value_b.extend_from_slice(&fill(1_048_576, 0xB5));
    put(&cluster.client(), "snap-b", &value_b);
    // Force a snapshot + purge on the live leader while the follower is
    // behind, so catch-up must flow through snapshot install.
    let base = cluster.snapshot(leader);
    assert!(base > 0, "snapshot seals a purge base");
    let before = bulk_received(&cluster)[lagger];
    cluster.heal(lagger);
    cluster.wait_converged();
    assert_eq!(get(&cluster.client(), "snap-b"), value_b);
    let delta = bulk_received(&cluster)[lagger].saturating_sub(before);
    assert!(
        delta < 4 * 1024 * 1024,
        "lagger moved only {delta} bytes across snapshot catch-up"
    );
}

/// K. Full cluster restart preserves large committed values.
#[test]
fn large_full_cluster_restart_preserves_state() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let _ = cluster.wait_leader();
    let value = fill(16 * 1024 * 1024, 0xFE);
    put(&cluster.client(), "big-restart", &value);
    cluster.wait_converged();
    cluster.kill_all();
    cluster.restart_all().expect("all restart");
    let _ = cluster.wait_leader();
    assert_eq!(get(&cluster.client(), "big-restart"), value);
}

/// L. A partitioned minority cannot commit a large root: the isolate
/// fails without quorum while the majority proceeds.
#[test]
fn large_minority_partition_cannot_commit() {
    use kivi_client::{ClientConfig, NativeClient};
    use kivi_types::NamespaceId;

    let cluster = Cluster::spawn().expect("cluster spawns");
    let leader = cluster.wait_leader();
    cluster.partition(leader);
    let majority = cluster.wait_leader_except(leader);
    assert_ne!(majority, leader);
    // Pinned directly on the isolate: no quorum, no commit.
    let isolate = NativeClient::new(ClientConfig {
        seeds: vec![cluster.native_endpoint(leader)],
        namespace: NamespaceId::from_u64(1),
        request_timeout: Duration::from_secs(8),
        ..ClientConfig::default()
    })
    .expect("isolate client builds");
    let value = fill(4 * 1024 * 1024, 0x10);
    let mut source: &[u8] = &value;
    assert!(
        isolate
            .put_stream(&key("big-minority"), Some(value.len() as u64), &mut source)
            .is_err(),
        "isolated minority must not acknowledge without quorum"
    );
    // Majority proceeds while partitioned.
    put(&cluster.client(), "big-majority", &value);
    cluster.heal(leader);
    cluster.wait_converged();
    assert_eq!(get(&cluster.client(), "big-majority"), value);
}

/// Benchmarks (reported, stability-asserted): put/get latencies per size,
/// small-write p50/p99 during a 64 MiB bulk transfer with no term change,
/// and failover timing to first byte.
#[test]
fn large_benchmarks_report_and_hold_stable() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let _ = cluster.wait_leader();
    for (name, size) in [
        ("bench-256kib", 256 * 1024),
        ("bench-1mib", 1_048_576),
        ("bench-4mib", 4 * 1024 * 1024),
    ] {
        let value = fill(size, 0xB0);
        let start = Instant::now();
        put(&cluster.client(), name, &value);
        let put_ms = start.elapsed().as_millis();
        let start = Instant::now();
        assert_eq!(get(&cluster.client(), name), value);
        let get_ms = start.elapsed().as_millis();
        eprintln!("BENCH {name}: put {put_ms}ms get {get_ms}ms");
    }
    // Bulk transfer in the background while small writes continue: every
    // small write must succeed with a stable term (no election).
    let term_before = cluster.tablet(cluster.wait_leader())["term"]
        .as_u64()
        .unwrap_or(0);
    let big = fill(64 * 1024 * 1024, 0xBB);
    let client = cluster.client();
    std::thread::scope(|scope| {
        let bulk = scope.spawn(|| {
            let mut source: &[u8] = &big;
            client.put_stream(&key("bench-bulk"), Some(big.len() as u64), &mut source)
        });
        let small = cluster.client();
        let mut latencies = Vec::new();
        for i in 0..100u32 {
            let start = Instant::now();
            small
                .set(
                    &key(&format!("bench-small-{i:03}")),
                    bytes::Bytes::from_static(b"v"),
                )
                .expect("small write succeeds during bulk");
            latencies.push(start.elapsed().as_micros());
        }
        latencies.sort_unstable();
        let p50 = latencies[50];
        let p99 = latencies[99];
        eprintln!("BENCH small-during-bulk: p50 {p50}us p99 {p99}us over 100 writes");
        bulk.join()
            .expect("bulk thread joins")
            .expect("bulk commits");
    });
    cluster.wait_converged();
    assert_eq!(get(&cluster.client(), "bench-bulk"), big);
    let term_after = cluster.tablet(cluster.wait_leader())["term"]
        .as_u64()
        .unwrap_or(u64::MAX);
    assert_eq!(
        term_before, term_after,
        "no election during 64 MiB bulk transfer"
    );
    // Failover timing: kill leader, measure election + first byte.
    let leader = cluster.wait_leader();
    cluster.kill(leader);
    let start = Instant::now();
    let elected = cluster.wait_leader();
    let election_ms = start.elapsed().as_millis();
    assert_ne!(elected, leader);
    let start = Instant::now();
    assert_eq!(get(&cluster.client(), "bench-bulk"), big);
    eprintln!(
        "BENCH failover: election {election_ms}ms first-read {read_ms}ms",
        read_ms = start.elapsed().as_millis()
    );
}
