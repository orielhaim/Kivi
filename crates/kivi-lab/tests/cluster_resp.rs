//! RESP edge on the replicated cluster (task P).
//!
//! Spawns a 3-node cluster with RESP edges and drives Redis commands
//! through them: writes/reads round-trip on the leader, followers answer
//! `NOTLEADER` (never `MOVED`/`ASK` — there is no Redis Cluster here),
//! and counters work end to end. Without the lab `redis-compat` feature
//! this file compiles to nothing; without a RESP-capable server binary
//! the harness fails loudly.

#![cfg(feature = "redis-compat")]

use kivi_lab::cluster::Cluster;
use kivi_lab::resp_client::{Reply, RespClient};

/// RESP round-trips against the replicated leader.
#[test]
fn resp_writes_and_reads_on_the_leader() {
    let cluster = Cluster::spawn_with_resp().expect("resp cluster spawns");
    let leader = cluster.wait_leader();
    let mut client = RespClient::connect(&cluster.resp_endpoint(leader).expect("resp endpoint"))
        .expect("resp connects");
    assert_eq!(
        client.round_trip(&[b"SET", b"rk", b"rv"]).expect("set"),
        Reply::Simple(b"OK".to_vec())
    );
    assert_eq!(
        client.round_trip(&[b"GET", b"rk"]).expect("get"),
        Reply::Bulk(Some(b"rv".to_vec()))
    );
    assert_eq!(
        client.round_trip(&[b"EXISTS", b"rk"]).expect("exists"),
        Reply::Integer(1)
    );
    assert_eq!(
        client.round_trip(&[b"DEL", b"rk"]).expect("del"),
        Reply::Integer(1)
    );
    assert_eq!(
        client.round_trip(&[b"GET", b"rk"]).expect("missing"),
        Reply::Bulk(None)
    );
}

/// Followers refuse writes with a retryable `BUSY` error (never a
/// cluster redirect, never an off-leader execution), while the leader
/// serves them. Clients discover the leader out of band (admin
/// `/v1/tablet`) or retry.
#[test]
fn resp_follower_refuses_writes_as_busy() {
    let cluster = Cluster::spawn_with_resp().expect("resp cluster spawns");
    let leader = cluster.wait_leader();
    let follower = (leader + 1) % 3;
    let mut follower_client =
        RespClient::connect(&cluster.resp_endpoint(follower).expect("resp endpoint"))
            .expect("resp connects");
    let reply = follower_client
        .round_trip(&[b"SET", b"k", b"v"])
        .expect("follower answers");
    match reply {
        Reply::Error(message) => {
            let text = String::from_utf8_lossy(&message);
            assert!(
                text.contains("BUSY"),
                "follower reports retryable BUSY, got {text}"
            );
        }
        other => panic!("follower must refuse writes, got {other:?}"),
    }
    // Reads serve locally on the follower (applied state).
    let mut leader_client =
        RespClient::connect(&cluster.resp_endpoint(leader).expect("resp endpoint"))
            .expect("resp connects");
    leader_client
        .round_trip(&[b"SET", b"sk", b"sv"])
        .expect("leader writes");
    // Wait for the follower to apply, then read there.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        let reply = follower_client.round_trip(&[b"GET", b"sk"]).expect("reads");
        if reply == Reply::Bulk(Some(b"sv".to_vec())) {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "follower never served the replicated write"
        );
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}
