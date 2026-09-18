//! Phase 7 read contracts end to end over a real 3-process cluster:
//! every contract serves its documented semantics through failover and
//! restart, proofs chain `AtLeast` reads, and foreign tokens fail as
//! `StaleToken` instead of serving weak data.
//!
//! Same harness discipline as `cluster_replication.rs`: each test spawns
//! its own cluster (fresh directories, fixed test cluster id); drops kill
//! every child, so no test leaks processes.

use std::time::Duration;

use kivi_lab::cluster::Cluster;
use kivi_state::Key;
use kivi_types::{CommitPosition, CommitToken, TabletEpoch, TabletId};

fn key(name: &str) -> Key {
    Key::from(name)
}

/// All four contracts serve their semantics, proofs chain, and the chain
/// survives leader failover and restart without weakening.
#[test]
fn read_contracts_hold_through_failover_and_restart() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let _leader = cluster.wait_leader();
    let client = cluster.client();
    client
        .set(&key("x"), bytes::Bytes::from_static(b"1"))
        .expect("x=1 commits");

    // Latest returns the value with a proof to chain from.
    let (value, proof) = client
        .get_with_contract(&key("x"), kivi_types::ReadContract::Latest)
        .expect("latest serves");
    assert_eq!(value.expect("present").as_ref(), b"1");
    let proof = proof.expect("servers attach read proofs");
    let token = proof.token();

    // AtLeast(token) serves the same state from any replica.
    let (value, chained) = client
        .get_at_least(&key("x"), token)
        .expect("at-least serves");
    assert_eq!(value.expect("present").as_ref(), b"1");
    let chained = chained.expect("chained proof");
    assert!(
        chained.position.as_u64() >= token.position().as_u64(),
        "chained receipt covers the token"
    );

    // Bounded-stale and Any serve present state too.
    let (value, _) = client
        .get_bounded_stale(&key("x"), Duration::from_secs(60))
        .expect("bounded-stale serves");
    assert_eq!(value.expect("present").as_ref(), b"1");
    let (value, _) = client.get_any(&key("x")).expect("any serves");
    assert_eq!(value.expect("present").as_ref(), b"1");

    // A token for a foreign tablet fails as StaleToken, never as data.
    let foreign = CommitToken::new(
        TabletId::from_u64(9999),
        TabletEpoch::INITIAL,
        CommitPosition::FIRST,
    );
    let error = client
        .get_at_least(&key("x"), foreign)
        .expect_err("foreign token rejects");
    assert_eq!(error, kivi_client::ClientError::StaleToken);

    // Failover: kill the leader, write on the successor, and prove the
    // pre-failover token still covers (same lineage) while reads stay
    // exact.
    let leader = cluster.wait_leader();
    cluster.kill(leader);
    let _elected = cluster.wait_leader();
    cluster
        .client()
        .set(&key("x"), bytes::Bytes::from_static(b"2"))
        .expect("x=2 commits on the new leader");
    let (value, _) = cluster
        .client()
        .get_at_least(&key("x"), token)
        .expect("pre-failover token survives failover");
    assert_eq!(value.expect("present").as_ref(), b"2");
    let (value, _) = cluster
        .client()
        .get_bounded_stale(&key("x"), Duration::from_secs(60))
        .expect("bounded-stale serves after failover");
    assert_eq!(value.expect("present").as_ref(), b"2");

    // Restart the dead replica from its directory: it catches up and the
    // cluster converges on the same state under every contract.
    cluster.restart(leader).expect("old leader restarts");
    cluster.wait_converged();
    let (value, _) = cluster
        .client()
        .get_with_contract(&key("x"), kivi_types::ReadContract::Latest)
        .expect("latest after catch-up");
    assert_eq!(value.expect("present").as_ref(), b"2");
    let (value, _) = cluster
        .client()
        .get_at_least(&key("x"), token)
        .expect("token covers after catch-up");
    assert_eq!(value.expect("present").as_ref(), b"2");
}
