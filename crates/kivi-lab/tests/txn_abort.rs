//! Deterministic transaction-abort recovery: every OCC-aborted batch
//! must release its prepared intents through the persisted Abort
//! decision + durable driver abort wave — never by waiting out the 60s
//! transaction lease.
//!
//! Each iteration forces an abort deterministically (a cross-tablet batch
//! whose second key carries a wrong-version OCC expectation: the first
//! key prepares, the second fails validation, the driver persists Abort
//! and finalize-aborts the prepared key). The test then requires zero
//! prepared intents cluster-wide within a short semantic bound (grace +
//! margin, far below lease expiry) and verifies the prepared write never
//! applied.
//!
//! This is the black-box contract for the driver abort path; it guards
//! the shape (decision persist → durable wave → resolver convergence).
//! Crash-mid-wave recovery (driver gone, resolver + lease fallback) is
//! covered by the chaos suites, not here.

use std::time::{Duration, Instant};

use bytes::Bytes;
use kivi_client::ordered::{BatchExpect, BatchWriteKind, BatchWriteSpec};
use kivi_lab::cluster::Cluster;
use kivi_state::Key;
use kivi_types::NamespaceId;

const NS: NamespaceId = NamespaceId::from_u64(1);
const ITERATIONS: usize = 10;
/// Short semantic bound for intent clearance: resolver grace is 2s, so
/// 20s is grace + ample margin yet 3× below the 60s transaction lease.
/// Clearing inside this bound proves no lease wait.
const DRAIN_BOUND: Duration = Duration::from_secs(20);

fn wait_zero_intents(cluster: &Cluster, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        if cluster.total_intents() == 0 {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "intents never cleared (stuck {})",
            cluster.total_intents()
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

#[test]
fn occ_abort_releases_intents_without_lease_wait() {
    let cluster = Cluster::spawn_ordered(4).expect("ordered cluster spawns");
    let _ = cluster.wait_all_leaders();
    let _ = cluster.set_policy(&serde_json::json!({
        "auto_split": false,
        "auto_merge": false,
    }));
    let client = cluster.client();

    // Two keys on different ordered tablets (genesis tiling).
    let key_a = [0x10u8, b'a'];
    let key_b = [0xF0u8, b'b'];
    client
        .set(&Key::from(key_a.to_vec()), Bytes::from_static(b"base-a"))
        .expect("seed a");
    client
        .set(&Key::from(key_b.to_vec()), Bytes::from_static(b"base-b"))
        .expect("seed b");

    for round in 0..ITERATIONS {
        // Cross-tablet batch: A puts cleanly, B's OCC expectation is
        // deliberately wrong (no live version is u64::MAX), so B's
        // prepare fails after A prepared → deterministic abort path.
        let writes = [
            BatchWriteSpec {
                key: key_a.to_vec(),
                kind: BatchWriteKind::Put(format!("round-{round}").into_bytes()),
                expect: BatchExpect::Any,
            },
            BatchWriteSpec {
                key: key_b.to_vec(),
                kind: BatchWriteKind::Put(b"never".to_vec()),
                expect: BatchExpect::Version(u64::MAX),
            },
        ];
        match client.atomic_batch(NS, &writes) {
            Err(kivi_client::ClientError::TxnConflict) => {}
            other => panic!("round {round}: expected TxnConflict, got {other:?}"),
        }
        // Abort applied nothing: A keeps its pre-batch value.
        let got = client
            .get(&Key::from(key_a.to_vec()))
            .expect("readback works");
        assert_eq!(
            got.as_deref(),
            Some(b"base-a".as_slice()),
            "round {round}: aborted write stayed out"
        );
        // Prepared intents clear via the Abort path, not the lease.
        wait_zero_intents(&cluster, DRAIN_BOUND);
    }
    assert_eq!(cluster.total_intents(), 0, "no stuck intents");
    eprintln!("abort: {ITERATIONS} forced aborts cleared without lease wait");
}
