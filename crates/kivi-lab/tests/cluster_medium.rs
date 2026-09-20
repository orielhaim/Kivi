//! Medium values replicate as chunked sidecars over a real 3-process
//! cluster: a plain 2 KiB `set` (256 B < len <= 256 KiB) restages to a
//! chunked root, never inline bytes, never fabric ids.
//!
//! Uses the real multi-node [`Cluster`] harness (spawn, client, put/get):
//! put 2 KiB via plain `set`, get exact bytes, prove the replicated form
//! is chunked via node sidecar state (`/v1/sidecar` bulk moved), then kill
//! the leader and restart a follower while re-reading exact (sidecar fetch
//! path). Snapshot-level proof (no inline 2 KiB `Bytes`, no `Fabric`,
//! deterministic manifest) lives at the propose level in
//! `kivi-consensus::node::tests::medium_set_replicates_as_chunked_sidecar`.
//!
//! Deterministic: bounded retry polls only (`wait_leader`,
//! `wait_converged`); no wall-clock sleeps.

use kivi_lab::cluster::Cluster;
use kivi_state::Key;

fn key(name: &str) -> Key {
    Key::from(name)
}

/// Deterministic 2 KiB value (matches the consensus unit test fill).
fn medium_value() -> Vec<u8> {
    (0..2048u32)
        .map(|i| u8::try_from(i % 251).expect("remainder fits in u8"))
        .collect()
}

/// Follower sidecar bulk bytes received, per member.
fn bulk_received(cluster: &Cluster) -> [u64; 3] {
    [0, 1, 2].map(|i| {
        let (status, body) = cluster.admin_get(i, "/v1/sidecar");
        assert_eq!(status, 200);
        body["bulk_bytes_received"].as_u64().unwrap_or(0)
    })
}

/// Sidecar request counters per member: `(manifest_requests, chunk_requests)`.
fn sidecar_requests(cluster: &Cluster) -> [(u64, u64); 3] {
    [0, 1, 2].map(|i| {
        let (status, body) = cluster.admin_get(i, "/v1/sidecar");
        assert_eq!(status, 200);
        (
            body["manifest_requests"].as_u64().unwrap_or(0),
            body["chunk_requests"].as_u64().unwrap_or(0),
        )
    })
}

/// Medium 2 KiB `set` replicates as a chunked sidecar and survives
/// failover plus restart with exact bytes.
#[test]
fn medium_2kib_replicates_as_chunked_sidecar() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let leader = cluster.wait_leader();
    let value = medium_value();
    assert_eq!(value.len(), 2048);
    let before_bulk = bulk_received(&cluster);
    // Plain `set` (not `put_stream`): the propose path restages medium
    // values to `SetChunked` before Raft ordering.
    cluster
        .client()
        .set(&key("medium"), bytes::Bytes::from(value.clone()))
        .expect("medium set commits");
    cluster.wait_converged();
    // Exact bytes back through the normal `get` (sidecar-resolved).
    let read = cluster
        .client()
        .get(&key("medium"))
        .expect("reads")
        .expect("present");
    assert_eq!(read.as_ref(), value.as_slice(), "medium round-trips exact");
    // Replicated form is a chunked root: sidecars moved over the bulk
    // lane. Inline replication would move zero sidecar bulk; a
    // worker-local fabric id could never resolve on another replica (this
    // `get` succeeding already rules fabric out). Preflight needs only one
    // follower for quorum (the straggler catches up through the append
    // gate), so assert cluster-wide movement, never an exact per-member
    // shape that races elections.
    let after_bulk = bulk_received(&cluster);
    let total_delta: u64 = after_bulk
        .iter()
        .zip(before_bulk.iter())
        .map(|(after, before)| after.saturating_sub(*before))
        .sum();
    assert!(
        total_delta >= 2048,
        "sidecars moved for the 2 KiB value (total delta {total_delta})"
    );
    let requests = sidecar_requests(&cluster);
    let total_manifests: u64 = requests.iter().map(|(m, _)| *m).sum();
    let total_chunks: u64 = requests.iter().map(|(_, c)| *c).sum();
    assert!(
        total_manifests >= 1 && total_chunks >= 1,
        "followers fetched manifest + chunks: {requests:?}"
    );
    // Kill the leader: the survivors serve the exact bytes from their
    // durable sidecars (sidecar fetch path, no dead leader needed).
    cluster.kill(leader);
    let elected = cluster.wait_leader();
    assert_ne!(elected, leader, "a new leader emerges");
    let read = cluster
        .client()
        .get(&key("medium"))
        .expect("reads after failover")
        .expect("present");
    assert_eq!(read.as_ref(), value.as_slice(), "exact after failover");
    // Restart the dead leader from its directory: it catches up and every
    // replica agrees on the same bytes.
    cluster.restart(leader).expect("old leader restarts");
    cluster.wait_converged();
    let read = cluster
        .client()
        .get(&key("medium"))
        .expect("reads after catch-up")
        .expect("present");
    assert_eq!(read.as_ref(), value.as_slice(), "exact after catch-up");
}
