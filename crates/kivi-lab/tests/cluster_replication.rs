//! Real 3-process replicated Kivi deployment (tasks X–Z, AC–AD).
//!
//! Three `kivi-server --cluster-mode` processes with independent data
//! directories form one tablet group over real peer TCP. These are the
//! money tests: real election, `kill -9` leader failover, lost-response
//! exactly-once across failover, full-cluster restart, and incarnation
//! discipline — all against real processes, never in-process Raft.
//!
//! Each test spawns its own cluster (fresh directories, fixed test
//! cluster id); drops kill every child, so no test leaks processes.
//! Client routing follows leader redirects with stable mutation
//! identities; TCP-death failover re-seeds a fresh client on the live
//! members (documented client contract: hint-driven retry is
//! transparent, dead-seed redial is the application's to re-seed).

use std::time::Duration;

use kivi_lab::cluster::Cluster;
use kivi_state::Key;
use kivi_types::{RequestSeq, SessionId};

fn key(name: &str) -> Key {
    Key::from(name)
}

/// Applied index of the current leader (the commit watermark followers
/// must reach).
fn leader_applied(cluster: &Cluster) -> u64 {
    let leader = cluster.wait_leader();
    cluster.tablet(leader)["applied"].as_u64().unwrap_or(0)
}

/// Real election (task X): exactly one leader, a write through it, a
/// linearizable read of the value, and full application everywhere.
#[test]
fn election_first_write_and_convergence() {
    let cluster = Cluster::spawn().expect("cluster spawns");
    let _leader = cluster.wait_leader();
    let client = cluster.client();
    client
        .set(&key("x"), bytes::Bytes::from_static(b"1"))
        .expect("write through leader");
    let value = client
        .get(&key("x"))
        .expect("latest reads")
        .expect("present");
    assert_eq!(value.as_ref(), b"1");
    cluster.wait_applied(leader_applied(&cluster));
    cluster.wait_converged();
}

/// Hard leader failover (task Y): `kill -9` the leader mid-service, a
/// new leader emerges, writes continue, the old leader restarts and
/// catches up, and all three replicas agree.
#[test]
fn hard_leader_failover_and_catch_up() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let leader = cluster.wait_leader();
    cluster
        .client()
        .set(&key("x"), bytes::Bytes::from_static(b"1"))
        .expect("x=1 commits");
    // `kill -9`: no shutdown handshake, no extra flush.
    cluster.kill(leader);
    let elected = cluster.wait_leader();
    assert_ne!(elected, leader, "a new leader emerges");
    // Fresh client on the live members (the dead seed is gone).
    cluster
        .client()
        .set(&key("x"), bytes::Bytes::from_static(b"2"))
        .expect("x=2 commits on the new leader");
    let value = cluster
        .client()
        .get(&key("x"))
        .expect("reads")
        .expect("present");
    assert_eq!(value.as_ref(), b"2");
    // Restart the old leader from its directory: no re-bootstrap, it
    // catches up, and every replica agrees.
    cluster.restart(leader).expect("old leader restarts");
    cluster.wait_converged();
    let value = cluster
        .client()
        .get(&key("x"))
        .expect("reads after catch-up")
        .expect("present");
    assert_eq!(value.as_ref(), b"2");
}

/// Lost-response failover (task Z, the money test): a counter mutation
/// commits on the leader, the leader dies around the response, and the
/// same identity retried on the new leader returns the original outcome
/// without executing twice.
#[test]
fn lost_response_failover_is_exactly_once() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let leader = cluster.wait_leader();
    let session = SessionId::from_u128(0x505E_D9E5_0000_0001);
    // First attempt races the kill: it may commit (response lost) or
    // fail outright — both are the ambiguous-delivery case. The client
    // is built before the scope so the kill (mutable) never contends
    // with the attempt's borrow.
    let racing = cluster.client_with_session(session);
    let first = std::thread::scope(|scope| {
        let attempt =
            scope.spawn(|| racing.counter_add_with_seq(&key("n"), 1, RequestSeq::from_u64(1)));
        std::thread::sleep(Duration::from_millis(200));
        cluster.kill(leader);
        attempt.join().expect("attempt joins")
    });
    let elected = cluster.wait_leader();
    assert_ne!(elected, leader, "a new leader emerges");
    // Retry the SAME identity on the new leader: the response must equal
    // the original outcome, and the counter must read exactly 1.
    let retry = cluster
        .client_with_session(session)
        .counter_add_with_seq(&key("n"), 1, RequestSeq::from_u64(1))
        .expect("same-identity retry succeeds");
    if let Ok(first) = first {
        assert_eq!(
            retry, first,
            "retry returns the original outcome, not a re-execution"
        );
    }
    let value = cluster
        .client()
        .counter_get(&key("n"))
        .expect("reads")
        .expect("counter present");
    assert_eq!(value, 1, "counter executed exactly once");
    // Restart the old leader and prove full convergence afterwards.
    cluster.restart(leader).expect("old leader restarts");
    cluster.wait_converged();
    let value = cluster
        .client()
        .counter_get(&key("n"))
        .expect("reads after convergence")
        .expect("counter present");
    assert_eq!(value, 1);
}

/// Full cluster restart (task AC): acknowledged state survives stopping
/// every process, no member re-bootstraps, and new writes succeed.
#[test]
fn full_cluster_restart_preserves_state() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let client = cluster.client();
    for (name, value) in [("a", "1"), ("b", "2"), ("c", "3")] {
        client
            .set(&key(name), bytes::Bytes::from_static(value.as_bytes()))
            .unwrap_or_else(|_| panic!("{name}={value} commits"));
    }
    cluster.wait_converged();
    cluster.restart_all().expect("cluster restarts");
    // State preserved across the full restart.
    let client = cluster.client();
    for (name, value) in [("a", "1"), ("b", "2"), ("c", "3")] {
        let read = client
            .get(&key(name))
            .unwrap_or_else(|error| panic!("{name} reads: {error:?}"))
            .expect("present");
        assert_eq!(read.as_ref(), value.as_bytes());
    }
    // And new writes succeed on the reformed cluster.
    client
        .set(&key("d"), bytes::Bytes::from_static(b"4"))
        .expect("post-restart write commits");
    let read = client.get(&key("d")).expect("reads").expect("present");
    assert_eq!(read.as_ref(), b"4");
}

/// Node incarnation (task AD): a restarted member keeps its identity
/// but advances its incarnation, and the cluster reconverges around it.
#[test]
fn restarted_member_advances_incarnation() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let before: Vec<u64> = (0..3).map(|i| cluster.incarnation(i)).collect();
    cluster.restart(1).expect("member restarts");
    let after = cluster.incarnation(1);
    assert!(
        after > before[1],
        "incarnation advances across restarts ({} -> {after})",
        before[1]
    );
    // Identity is stable while the generation advances.
    let info = cluster.node_info(1);
    assert_eq!(info["node"].as_u64().unwrap_or(0), 2);
    cluster.wait_converged();
    let _ = cluster.wait_leader();
}
