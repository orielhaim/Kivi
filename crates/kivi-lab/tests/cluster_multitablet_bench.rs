//! Multi-tablet product benchmarks over real 3-node clusters: normal
//! mixed workloads, shared-writer batching evidence, 64 MiB preflight
//! decomposition, and tablet-scale startup characterization.
//!
//! These are measurement tests with lenient guards (they must stay green
//! across machines): percentiles, throughput, batching counters, and
//! timings print to stderr for the milestone report. The 100/1000-tablet
//! density probes are `#[ignore]`d (explicit runs only — never blocking).

use std::time::{Duration, Instant};

use kivi_lab::cluster::Cluster;
use kivi_state::Key;

fn key(name: &str) -> Key {
    Key::from(name)
}

/// Sorted latencies (ms) with (p50, p95, p99, count).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn summarize(name: &str, mut latencies: Vec<Duration>) {
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
    eprintln!(
        "{name}: n={} p50={:.2}ms p95={:.2}ms p99={:.2}ms",
        latencies.len(),
        at(0.50),
        at(0.95),
        at(0.99),
    );
}

/// Max `max_groups_per_batch` observed across live members (shared-writer
/// batching evidence: >1 proves cross-group physical batching happened).
fn max_groups_per_batch(cluster: &Cluster) -> u64 {
    (0..3)
        .filter(|i| cluster.alive(*i))
        .map(|i| {
            let (_, body) = cluster.admin_get(i, "/v1/durability");
            body["max_groups_per_batch"].as_u64().unwrap_or(0)
        })
        .max()
        .unwrap_or(0)
}

/// Normal mixed workload (§61) over 16 tablets: `GET` Latest, `SET`,
/// `CounterAdd` with 1 client then 8 concurrent clients, spanning the
/// tablet set.
#[test]
fn multi_tablet_mixed_workload_benchmark() {
    let spawned = Instant::now();
    let cluster = Cluster::spawn_with_tablets(16, 4, false).expect("cluster spawns");
    let elected = cluster.wait_all_leaders();
    eprintln!(
        "16 tablets x 4 workers: spawn+elect={:?} groups={}",
        spawned.elapsed(),
        elected.len()
    );
    let keys = cluster.keys_for_tablets(16);
    let flat: Vec<String> = keys.values().flatten().cloned().collect();
    assert_eq!(flat.len(), 256);
    // Phase 1: one client, sequential mixed ops.
    let client = cluster.client();
    let mut latencies = Vec::new();
    let phase = Instant::now();
    for (i, name) in flat.iter().take(128).enumerate() {
        let at = Instant::now();
        client
            .set(&key(name), bytes::Bytes::from(format!("v-{i}")))
            .expect("set commits");
        latencies.push(at.elapsed());
    }
    for name in flat.iter().take(128) {
        let at = Instant::now();
        client.get(&key(name)).expect("get reads").expect("present");
        latencies.push(at.elapsed());
    }
    for (i, name) in flat.iter().take(64).enumerate() {
        let at = Instant::now();
        let value = client
            .counter_add(&key(&format!("c-{name}")), 1)
            .expect("counter commits");
        assert_eq!(value, 1, "fresh counter starts at 1 (pass {i})");
        latencies.push(at.elapsed());
    }
    let elapsed = phase.elapsed();
    summarize("mixed/1-client", latencies);
    eprintln!(
        "mixed/1-client: 320 ops in {elapsed:?} ({:.0} ops/s)",
        320.0 / elapsed.as_secs_f64()
    );
    // Phase 2: eight concurrent clients over disjoint key slices.
    let phase = Instant::now();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker in 0..8 {
            let client = cluster.client();
            let slice: Vec<String> = flat.iter().skip(worker * 32).take(32).cloned().collect();
            handles.push(scope.spawn(move || {
                let mut latencies = Vec::new();
                for (i, name) in slice.iter().enumerate() {
                    let at = Instant::now();
                    if (i + worker) % 3 == 2 {
                        client.get(&key(name)).expect("get reads");
                    } else {
                        client
                            .set(&key(name), bytes::Bytes::from(format!("w{worker}-{i}")))
                            .expect("set commits");
                    }
                    latencies.push(at.elapsed());
                }
                latencies
            }));
        }
        let mut latencies = Vec::new();
        for handle in handles {
            latencies.extend(handle.join().expect("worker joins"));
        }
        let elapsed = phase.elapsed();
        summarize("mixed/8-client", latencies);
        eprintln!(
            "mixed/8-client: 256 ops in {elapsed:?} ({:.0} ops/s)",
            256.0 / elapsed.as_secs_f64()
        );
    });
    cluster.wait_converged_all();
    let batching = max_groups_per_batch(&cluster);
    eprintln!("shared-writer max_groups_per_batch across members: {batching}");
    let (_, durability) = cluster.admin_get(cluster.live_index(), "/v1/durability");
    eprintln!(
        "shared-writer totals: batches={} records={} barriers={} group_appearances={} barrier_latency_us={}",
        durability["physical_batches"],
        durability["records"],
        durability["barriers"],
        durability["group_appearances"],
        durability["barrier_latency_us"],
    );
}

/// 64 MiB preflight timing decomposition (§62): upload time, preflight
/// quorum evidence, bulk vs Raft bytes — the expensive sidecar movement
/// must sit BEFORE the tiny root proposal.
#[test]
fn multi_tablet_large_value_preflight_decomposition() {
    let cluster = Cluster::spawn_with_tablets(4, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(1);
    let tablet = *keys.keys().next().expect("a tablet");
    let name = keys[&tablet][0].clone();
    let mut value = vec![0u8; 64 * 1024 * 1024];
    let mut state = 0x62u64;
    for byte in &mut value {
        state = state
            .wrapping_add(0x9E37_79B9_7F4A_7C15)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state ^= state >> 29;
        *byte = (state % 251) as u8;
    }
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
            .filter(|i| cluster.alive(*i))
            .map(|i| {
                let (_, sidecar) = cluster.admin_get(i, "/v1/sidecar");
                sidecar["bulk_bytes_received"].as_u64().unwrap_or(0)
            })
            .sum::<u64>()
    };
    let control_before = control_of(leader);
    let bulk_before = bulk_everywhere();
    let started = Instant::now();
    let mut source: &[u8] = &value;
    cluster
        .client()
        .put_stream(&key(&name), Some(value.len() as u64), &mut source)
        .expect("64 MiB commits");
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

/// 100-tablet startup/density characterization (§42, §60): real product
/// path (must pass); prints startup, election, RSS-adjacent timings.
#[test]
#[ignore = "density characterization: explicit runs only, never blocking"]
fn multi_tablet_100_startup_characterization() {
    let spawned = Instant::now();
    let cluster = Cluster::spawn_with_tablets(100, 4, false).expect("cluster spawns");
    let leaders = cluster.wait_all_leaders();
    eprintln!(
        "100 tablets x 4 workers: spawn+elect={:?} groups={}",
        spawned.elapsed(),
        leaders.len()
    );
    assert_eq!(leaders.len(), 100);
    let keys = cluster.keys_for_tablets(1);
    assert_eq!(keys.len(), 100);
    let client = cluster.client();
    let phase = Instant::now();
    for (tablet, names) in &keys {
        client
            .set(&key(&names[0]), bytes::Bytes::from(format!("t{tablet}")))
            .unwrap_or_else(|error| panic!("tablet {tablet} writes: {error:?}"));
    }
    eprintln!("100 tablets x 1 write each: {:?}", phase.elapsed());
    cluster.wait_converged_all();
    eprintln!("100 tablets converged");
}

/// ~1000-tablet startup smoke (§42, §60): density characterization only —
/// generous deadlines, explicit runs, never blocking.
#[test]
#[ignore = "density characterization: explicit runs only, never blocking"]
fn multi_tablet_1000_startup_smoke() {
    let spawned = Instant::now();
    let cluster =
        Cluster::spawn_bare(1000, 4, false, &[], Duration::from_secs(300)).expect("cluster spawns");
    eprintln!("1000 tablets x 4 workers: spawn={:?}", spawned.elapsed());
    // Generous convergence poll (debug elections for 1000 groups take a
    // while; the standard 45 s window is for the blocking suites). Smoke
    // bar: some member knows a leader for every tablet. The 5 s admin
    // ceiling may 503 under a 1000-group debug fan-out — a slow member is
    // polled again, never failed.
    let deadline = Instant::now() + Duration::from_mins(15);
    loop {
        let mut known = 0;
        for i in 0..3 {
            if !cluster.alive(i) {
                continue;
            }
            let (status, body) = cluster.admin_get(i, "/v1/tablets");
            if status != 200 {
                continue;
            }
            known = known.max(
                body.as_array()
                    .cloned()
                    .unwrap_or_default()
                    .iter()
                    .filter(|entry| entry["leader"].as_u64().is_some())
                    .count(),
            );
        }
        if known == 1000 {
            eprintln!(
                "1000 tablets all leader-known after {:?}",
                spawned.elapsed()
            );
            break;
        }
        eprintln!("1000-tablet convergence: best member knows {known}/1000 leaders");
        assert!(Instant::now() < deadline, "1000 tablets never all elect");
        std::thread::sleep(Duration::from_secs(5));
    }
}
