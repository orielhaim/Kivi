//! Multi-tablet product invariants over a real 3-node cluster: the 64 MiB
//! preflight decomposition, where the expensive sidecar movement must land
//! before the tiny Raft root proposes.
//!
//! Timings print to stderr; the assertions are the invariants, not the
//! numbers.

use std::time::Instant;

use kivi_lab::cluster::Cluster;
use kivi_lab::testkit::{fill, key, put_stream};

/// 64 MiB preflight timing decomposition: upload time, preflight
/// quorum evidence, bulk vs Raft bytes.
#[test]
fn multi_tablet_large_value_preflight_decomposition() {
    let cluster = Cluster::spawn_with_tablets(4, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(1);
    let tablet = *keys.keys().next().expect("a tablet");
    let name = keys[&tablet][0].clone();
    let value = fill(64 * 1024 * 1024, 0x62);
    // Preflight runs on the tablet leader; bulk lands on followers.
    // Control (tiny roots) is read on the leader, bulk summed everywhere.
    let leader = cluster.leader_of(tablet);
    let control_of = |index: usize| {
        let (_, peers) = cluster.admin_get(index, "/v1/peers");
        peers
            .as_array()
            .cloned()
            .unwrap_or_default()
            .iter()
            .map(|peer| peer["control_bytes_sent"].as_u64().unwrap_or(0))
            .sum::<u64>()
    };
    let bulk_everywhere = || {
        (0..3)
            .filter(|index| cluster.alive(*index))
            .map(|index| {
                let (_, sidecar) = cluster.admin_get(index, "/v1/sidecar");
                sidecar["bulk_bytes_received"].as_u64().unwrap_or(0)
            })
            .sum::<u64>()
    };
    let control_before = control_of(leader);
    let bulk_before = bulk_everywhere();
    let started = Instant::now();
    put_stream(&cluster.client(), &name, &value);
    let upload = started.elapsed();
    cluster.wait_converged_all();
    let control = control_of(cluster.leader_of(tablet)).saturating_sub(control_before);
    let bulk = bulk_everywhere().saturating_sub(bulk_before);
    let (_, preflight) = cluster.admin_get(cluster.leader_of(tablet), "/v1/preflight");
    eprintln!("64 MiB replicated value timing decomposition:");
    eprintln!("  client put_stream (stage+preflight+root commit): {upload:?}");
    eprintln!(
        "  leader preflight attempts={} quorum_ready={} fallbacks={} latency_us={}",
        preflight["attempts"],
        preflight["quorum_ready"],
        preflight["fallbacks"],
        preflight["latency_us"],
    );
    eprintln!("  bulk bytes received across members (sidecars): {bulk}");
    eprintln!("  leader control bytes sent (Raft, tiny roots): {control}");
    assert!(
        control < 8 * 1024 * 1024,
        "Raft control stays tiny while 64 MiB moves on bulk"
    );
    assert!(
        bulk >= 64 * 1024 * 1024,
        "followers received the sidecars: {bulk}"
    );
    assert_eq!(
        preflight["fallbacks"].as_u64().unwrap_or(1),
        0,
        "clean run preflights to quorum (no append-gate fallback)"
    );
    assert!(
        preflight["quorum_ready"].as_u64().unwrap_or(0) >= 1,
        "leader reached preflight quorum before proposing the root"
    );
    let back = cluster
        .client()
        .get_stream(&key(&name))
        .expect("reads")
        .expect("present")
        .to_vec();
    assert_eq!(back, value, "byte-exact round trip");
}
