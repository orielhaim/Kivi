//! Real-process partition, fencing, stale-hint, and snapshot catch-up.
//!
//! All tests run against three real `kivi-server --cluster-mode` processes
//! over real TCP using the admin suspend/resume/snapshot hooks:
//!
//! * minority fencing: an isolated old leader must not acknowledge new
//!   strong mutations while the majority continues;
//! * stale-hint routing: a client holding the old leader hint retries
//!   safely (same identity, bounded budget) and converges on the new
//!   leader;
//! * log-conflict repair: the isolated leader's uncommitted suffix is
//!   truncated after heal and the authoritative history wins;
//! * snapshot catch-up: a follower behind the purge point recovers via
//!   checkpoint snapshot install and resumes log replication.

use std::time::Duration;

use kivi_lab::cluster::Cluster;
use kivi_state::Key;

fn key(name: &str) -> Key {
    Key::from(name)
}

/// Minority fencing over real processes: isolate the leader, prove the
/// minority cannot acknowledge while the majority continues, heal, and
/// prove convergence with no dual acknowledged histories. Mirrors the
/// in-process `isolated_leader_cannot_acknowledge`: the isolate may still
/// believe it leads (stale term), but without quorum its commit can never
/// complete.
#[test]
fn minority_partition_fences_old_leader() {
    use kivi_client::{ClientConfig, NativeClient};
    use kivi_types::NamespaceId;

    let cluster = Cluster::spawn().expect("cluster spawns");
    let leader = cluster.wait_leader();
    cluster
        .client()
        .set(&key("fenced"), bytes::Bytes::from_static(b"base"))
        .expect("base commits");
    cluster.wait_converged();

    // Isolate the old leader from both followers (bidirectional suspends;
    // each returns only after its sockets closed, so the partition is
    // airtight for the fencing verdict below).
    cluster.partition(leader);
    // The majority elects without the isolated node (the isolate itself
    // may still report `leader` on its stale term — that is the fenced
    // minority, not the majority).
    let majority = cluster.wait_leader_except(leader);
    assert_ne!(majority, leader, "majority elects without the isolate");

    // The isolated minority must not acknowledge a new strong mutation:
    // without quorum its commit can never complete. Pin a short-timeout
    // client on the isolate alone so no majority redirect can rescue it.
    // Still-leader, still-pending, or prompt routing failure all count as
    // fenced — only an `Ok` acknowledgement fails the test.
    let isolate_client = NativeClient::new(ClientConfig {
        seeds: vec![cluster.native_endpoint(leader)],
        namespace: NamespaceId::from_u64(1),
        request_timeout: Duration::from_secs(6),
        ..ClientConfig::default()
    })
    .expect("isolate client builds");
    let verdict = isolate_client.set(&key("fenced"), bytes::Bytes::from_static(b"isolated"));
    assert!(
        verdict.is_err(),
        "isolated minority acknowledged without quorum: {verdict:?}"
    );

    // The majority side proceeds while partitioned.
    cluster
        .client()
        .set(&key("fenced"), bytes::Bytes::from_static(b"majority"))
        .expect("majority writes during partition");

    // Heal: the isolated uncommitted suffix truncates and every replica
    // converges on the majority history — no dual acknowledged histories.
    cluster.heal(leader);
    cluster.wait_converged();
    let value = cluster
        .client()
        .get(&key("fenced"))
        .expect("reads")
        .expect("present");
    assert_eq!(value.as_ref(), b"majority");
}

/// Stale leader hints retry safely: a client that learned the old leader
/// keeps its mutation identity across the failover, updates its hint, and
/// either succeeds on the new leader or fails bounded (never loops).
///
/// Under a full parallel suite the post-kill election and mesh heal are
/// load-sensitive (Windows fsync + 3-process QUIC/H3 formation), so this
/// test is condition-driven throughout: it converges the first write
/// before the kill, waits for the new leader to cover that watermark, and
/// retries the second write on retryable routing errors instead of
/// assuming the first post-election attempt lands.
#[test]
fn stale_leader_hint_recovers_with_same_identity() {
    use kivi_types::{RequestSeq, SessionId};

    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let leader = cluster.wait_leader();
    let session = SessionId::from_u128(0x0005_7A1E_0000_0000_0001u128);
    // Learn the old leader hint with a first write, then converge it so
    // the kill cannot truncate an uncommitted suffix (that path is covered
    // separately by the log-conflict test).
    cluster
        .client_with_session(session)
        .counter_add_with_seq(&key("hint"), 1, RequestSeq::from_u64(1))
        .expect("first write learns hint");
    cluster.wait_converged();
    let watermark = cluster.applied(leader);
    // Kill the hinted leader: every cached hint is now stale.
    cluster.kill(leader);
    let elected = cluster.wait_leader();
    assert_ne!(elected, leader, "new leader emerges");
    // The stable-role poll can observe the new leader before it covers
    // the pre-kill watermark; wait for coverage so the next proposal does
    // not race replication catch-up.
    cluster.wait_applied(watermark);
    // Retry a NEW identity through the stale-hint client: bounded
    // condition-driven retries must land on the new leader (identity
    // preserved exactly once). Election-in-flight answers (`Overloaded`,
    // timeouts) are retryable here — only a terminal acknowledgement or
    // the deadline decides.
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let retry = loop {
        match cluster.client_with_session(session).counter_add_with_seq(
            &key("hint"),
            1,
            RequestSeq::from_u64(2),
        ) {
            Ok(value) => break value,
            Err(error) => {
                let retryable = matches!(
                    error,
                    kivi_client::ClientError::Overloaded
                        | kivi_client::ClientError::Timeout
                        | kivi_client::ClientError::Io(_)
                        | kivi_client::ClientError::SessionOverloaded
                );
                assert!(
                    retryable,
                    "stale hint fails only retryably while electing: {error:?}"
                );
                assert!(
                    std::time::Instant::now() < deadline,
                    "stale hint never recovered: {error:?}"
                );
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    };
    assert_eq!(retry, 2, "second add applies once on the new leader");
    let value = cluster
        .client()
        .counter_get(&key("hint"))
        .expect("reads")
        .expect("present");
    assert_eq!(value, 2);
    cluster.restart(leader).expect("old leader restarts");
    cluster.wait_converged();
}

/// Snapshot catch-up: purge history on the leader while a follower is
/// down, then restart it behind the purge point — it must install the
/// checkpoint snapshot and resume replication.
#[test]
fn lagging_follower_recovers_via_snapshot_after_purge() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let leader = cluster.wait_leader();
    cluster
        .client()
        .set(&key("snap-base"), bytes::Bytes::from_static(b"0"))
        .expect("base commits");
    cluster.wait_converged();

    // Take one follower down (the highest index that is not the leader).
    let lagger = (0..3).find(|i| *i != leader).expect("a follower");
    cluster.kill(lagger);

    // Advance the log while the follower is down.
    let client = cluster.client();
    for i in 0..150u32 {
        let name = format!("snap-k{i:03}");
        client
            .set(&key(&name), bytes::Bytes::from_static(b"v"))
            .unwrap_or_else(|_| panic!("write {name} commits without the lagger"));
    }
    // Force a checkpoint snapshot + purge on the live leader.
    let base = cluster.snapshot(leader);
    assert!(base > 0, "snapshot seals a purge base");
    let tablet = cluster.tablet(leader);
    let purged = tablet["purged"].as_u64().unwrap_or(0);
    assert!(purged > 0, "purge advanced past history: {tablet}");

    // Restart the lagger behind the purge point: it must snapshot-install
    // and converge without manual repair.
    cluster.restart(lagger).expect("lagger restarts");
    cluster.wait_converged();
    let read = cluster
        .client()
        .get(&key("snap-base"))
        .expect("reads")
        .expect("present");
    assert_eq!(read.as_ref(), b"0");
    let read = cluster
        .client()
        .get(&key("snap-k149"))
        .expect("reads")
        .expect("present");
    assert_eq!(read.as_ref(), b"v");
}
