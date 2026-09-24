//! Networked redundancy benchmarks over a real 3-node cluster.
//!
//! Measurement tests with lenient guards (they must stay green across
//! machines): protect/read/repair latency percentiles, cross-node byte
//! accounting, and storage amplification print to stderr for the milestone
//! report. Correctness is asserted only where a regression would be real:
//! bytes must actually cross the peer mesh, reads must stay exact, repair
//! must restore the lost copy, and a coded layout must store less than a
//! replicated one.

use std::time::{Duration, Instant};

use kivi_lab::cluster::Cluster;
use serde_json::{Value, json};

/// Generous deadlines: process spawn plus Raft commits dominates wall time.
const DEADLINE: Duration = Duration::from_secs(180);
const POLL: Duration = Duration::from_millis(250);

/// Asset family discriminant for generic immutable bytes (BLAKE3 identity).
const KIND_GENERIC: u8 = 5;

/// Bench payload size: 64 KiB is the replication/coding threshold, so this
/// size is replicated under an intent (cheap direct reads).
const ASSET_BYTES: usize = 64 * 1024;

/// Assets per benchmark phase.
const ASSETS: usize = 8;

/// Coded-layout comparison payload (above the threshold, so the baseline
/// planner picks erasure coding).
const CODED_BYTES: usize = 512 * 1024;

/// Deterministic pseudo-random payload (splitmix64): incompressible enough
/// to be representative, fully reproducible.
fn payload(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.extend_from_slice(&z.to_le_bytes());
    }
    out.truncate(len);
    out
}

/// Standard padded base64 (the admin plane's image encoding).
fn b64(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Decodes standard padded base64.
fn unb64(text: &str) -> Option<Vec<u8>> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.decode(text).ok()
}

/// Lowercase hex identity of a generic immutable asset.
fn asset_hex(bytes: &[u8]) -> String {
    use core::fmt::Write as _;
    let digest = kivi_codec::integrity::blake3_256(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Fragment store root of one cluster member.
fn fragments_root(cluster: &Cluster, index: usize) -> std::path::PathBuf {
    cluster.data_dir(index).join("redundancy").join("fragments")
}

/// Protects bytes through the admin plane.
fn protect(cluster: &Cluster, index: usize, bytes: &[u8], tolerance: u8) -> (u16, Value) {
    let body = json!({
        "kind": KIND_GENERIC,
        "domain": 0,
        "bytes_b64": b64(bytes),
        "tolerance": tolerance,
    });
    cluster.admin_post(index, "/v1/redundancy/protect", &body)
}

/// Reads an asset back through the admin plane.
fn read(cluster: &Cluster, index: usize, hash: &str) -> (u16, Value) {
    cluster.admin_get(
        index,
        &format!("/v1/redundancy/read?kind={KIND_GENERIC}&domain=0&hash_hex={hash}"),
    )
}

/// Reads one asset's health.
fn health(cluster: &Cluster, index: usize, hash: &str) -> (u16, Value) {
    cluster.admin_get(
        index,
        &format!("/v1/redundancy/health?kind={KIND_GENERIC}&domain=0&hash_hex={hash}"),
    )
}

/// Repairs one asset.
fn repair(cluster: &Cluster, index: usize, hash: &str) -> (u16, Value) {
    cluster.admin_post(
        index,
        "/v1/redundancy/repair",
        &json!({ "kind": KIND_GENERIC, "domain": 0, "hash_hex": hash }),
    )
}

/// Transitions one asset to an explicit scheme.
fn transition(cluster: &Cluster, index: usize, hash: &str, scheme: &Value) -> (u16, Value) {
    cluster.admin_post(
        index,
        "/v1/redundancy/transition",
        &json!({ "kind": KIND_GENERIC, "domain": 0, "hash_hex": hash, "scheme": scheme }),
    )
}

/// Snapshot of one node's fabric metrics.
fn metrics(cluster: &Cluster, index: usize) -> Value {
    let (status, body) = cluster.admin_get(index, "/v1/redundancy/metrics");
    assert_eq!(status, 200, "metrics: {body}");
    body
}

/// Total fragment bytes the catalog says every holder stores (absolute
/// gauge derived from published layouts, so it nets retired generations).
fn stored_bytes(body: &Value) -> u64 {
    body["by_node_bytes"]
        .as_array()
        .expect("by_node_bytes")
        .iter()
        .filter_map(|entry| entry["bytes"].as_u64())
        .sum()
}

/// Sorted latencies (ms) reported as p50/p95/p99.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
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

/// Polls until `check` returns a value or the deadline passes.
fn await_until<T>(label: &str, mut check: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + DEADLINE;
    loop {
        if let Some(value) = check() {
            return value;
        }
        assert!(
            Instant::now() < deadline,
            "timed out after {DEADLINE:?} waiting for {label}"
        );
        std::thread::sleep(POLL);
    }
}

/// Waits until an asset reports `wanted` health on a node.
fn await_health(cluster: &Cluster, index: usize, hash: &str, wanted: &str) -> Value {
    await_until(&format!("health {wanted} on node {index}"), || {
        let (status, body) = health(cluster, index, hash);
        (status == 200 && body["health"].as_str() == Some(wanted)).then_some(body)
    })
}

/// Protect throughput and cross-node byte accounting: eight 64 KiB assets,
/// three copies each, all placed on distinct holders.
///
/// The real signal is `remote_bytes_sent`: a "distributed" protect that
/// quietly kept every copy local would show zero, so the guard requires at
/// least half the stored bytes to have crossed the mesh (two of three
/// copies per asset are remote by construction).
#[test]
fn protect_throughput_and_cross_node_bytes() {
    let cluster = Cluster::spawn().expect("cluster spawns");
    let _leader = cluster.wait_leader();

    let mut latencies = Vec::new();
    let mut hashes = Vec::new();
    for index in 0..ASSETS {
        let bytes = payload(ASSET_BYTES, 0xA11CE + u64::try_from(index).unwrap_or(0));
        let hash = asset_hex(&bytes);
        let started = Instant::now();
        let (status, body) = protect(&cluster, 0, &bytes, 2);
        latencies.push(started.elapsed());
        assert_eq!(status, 200, "protect: {body}");
        let fragments = body["fragments"].as_array().expect("fragments");
        assert_eq!(fragments.len(), 3, "tolerance 2 replicates three copies");
        let mut holders: Vec<u64> = fragments
            .iter()
            .filter_map(|entry| entry["node"].as_u64())
            .collect();
        holders.sort_unstable();
        holders.dedup();
        assert_eq!(holders.len(), 3, "one copy per holder: {holders:?}");
        hashes.push(hash);
    }
    summarize("protect 64KiB x3", latencies);

    let after = metrics(&cluster, 0);
    let sent = after["remote_bytes_sent"].as_u64().unwrap_or(0);
    let local = after["local_bytes_written"].as_u64().unwrap_or(0);
    let stored = stored_bytes(&after);
    let logical = u64::try_from(ASSET_BYTES).unwrap_or(0) * ASSETS as u64;
    eprintln!(
        "protect bytes: logical={logical} stored={stored} remote_sent={sent} local_written={local} amplification={}",
        after["amplification"]
    );
    assert_eq!(stored, logical * 3, "three copies of every asset");
    assert_eq!(sent + local, logical * 3, "every byte landed on a holder");
    assert!(
        sent >= logical,
        "at least two of three copies crossed the mesh: sent={sent} logical={logical}"
    );
    assert_eq!(
        after["assets_replicated"].as_u64().unwrap_or(0),
        ASSETS as u64,
        "the catalog counts every replicated layout"
    );
    assert_eq!(after["assets_coded"], json!(0));
    assert_eq!(after["logical_bytes"].as_u64().unwrap_or(0), logical);

    // Every asset reads back exact on the coordinator that placed it.
    for hash in &hashes {
        let (status, body) = read(&cluster, 0, hash);
        assert_eq!(status, 200, "read: {body}");
    }
}

/// Read latency on the healthy path versus the degraded path (one holder
/// wiped: the coordinator must fetch a verified copy from a peer process).
#[test]
fn read_latency_healthy_and_degraded() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let _leader = cluster.wait_leader();
    let mut hashes = Vec::new();
    for index in 0..ASSETS {
        let bytes = payload(ASSET_BYTES, 0xB0B + u64::try_from(index).unwrap_or(0));
        let hash = asset_hex(&bytes);
        let (status, body) = protect(&cluster, 0, &bytes, 2);
        assert_eq!(status, 200, "protect: {body}");
        hashes.push((hash, bytes));
    }

    let before_healthy = metrics(&cluster, 0);
    let mut healthy = Vec::new();
    for (hash, bytes) in &hashes {
        let started = Instant::now();
        let (status, body) = read(&cluster, 0, hash);
        healthy.push(started.elapsed());
        assert_eq!(status, 200, "healthy read: {body}");
        assert_eq!(body["degraded"], json!(false));
        assert_eq!(
            unb64(body["bytes_b64"].as_str().expect("image")).as_deref(),
            Some(bytes.as_slice())
        );
    }
    summarize("read healthy (local holder)", healthy);
    let after_healthy = metrics(&cluster, 0);
    assert_eq!(
        after_healthy["fetch_count"].as_u64().unwrap_or(0)
            - before_healthy["fetch_count"].as_u64().unwrap_or(0),
        ASSETS as u64,
        "every read verifies exactly one holder copy"
    );
    assert_eq!(
        after_healthy["remote_bytes_received"], before_healthy["remote_bytes_received"],
        "a healthy layout is served by the local holder: nothing crosses the mesh"
    );
    assert_eq!(
        after_healthy["local_bytes_read"].as_u64().unwrap_or(0)
            - before_healthy["local_bytes_read"].as_u64().unwrap_or(0),
        u64::try_from(ASSET_BYTES).unwrap_or(0) * ASSETS as u64,
        "every healthy read came from this node's own holder"
    );
    // Wipe the coordinator's own store: its local copy is gone, so reads
    // must reconstruct from peer processes. The background repair lane may
    // win the race and refill the local copy, so the guard is the wire
    // evidence (fragments fetched across the mesh), not the degraded flag.
    cluster.kill(0);
    std::fs::remove_dir_all(fragments_root(&cluster, 0)).expect("coordinator wiped");
    cluster.restart(0).expect("coordinator restarts");
    let before_remote = metrics(&cluster, 0);

    let mut remote = Vec::new();
    let mut degraded_reads = 0u64;
    for (hash, bytes) in &hashes {
        let started = Instant::now();
        let (status, body) = read(&cluster, 0, hash);
        remote.push(started.elapsed());
        assert_eq!(status, 200, "read after local loss: {body}");
        if body["degraded"] == json!(true) {
            degraded_reads += 1;
        }
        assert_eq!(
            unb64(body["bytes_b64"].as_str().expect("image")).as_deref(),
            Some(bytes.as_slice()),
            "reconstruction from peers is exact"
        );
    }
    summarize("read after local loss (remote fetch)", remote);
    let after_remote = metrics(&cluster, 0);
    let fetched = after_remote["fetch_count"].as_u64().unwrap_or(0)
        - before_remote["fetch_count"].as_u64().unwrap_or(0);
    let received = after_remote["remote_bytes_received"].as_u64().unwrap_or(0)
        - before_remote["remote_bytes_received"].as_u64().unwrap_or(0);
    eprintln!(
        "after local loss: degraded_reads={degraded_reads} fetched={fetched} received={received}"
    );
    assert!(
        fetched > 0 && received > 0,
        "the missing local copy was refetched across the mesh: fetched={fetched} received={received}"
    );
}

/// Repair latency after a holder loses its store: every asset must regain
/// full health, and the rebuilt fragments must cross the mesh.
#[test]
fn repair_latency_after_holder_loss() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let _leader = cluster.wait_leader();
    let mut hashes = Vec::new();
    for index in 0..ASSETS {
        let bytes = payload(ASSET_BYTES, 0xC0DE + u64::try_from(index).unwrap_or(0));
        let hash = asset_hex(&bytes);
        let (status, body) = protect(&cluster, 0, &bytes, 2);
        assert_eq!(status, 200, "protect: {body}");
        hashes.push(hash);
    }

    cluster.kill(1);
    std::fs::remove_dir_all(fragments_root(&cluster, 1)).expect("holder wiped");
    cluster.restart(1).expect("holder restarts");
    for hash in &hashes {
        await_health(&cluster, 0, hash, "degraded");
    }
    let before = metrics(&cluster, 0);

    let mut latencies = Vec::new();
    for hash in &hashes {
        let started = Instant::now();
        let (status, body) = repair(&cluster, 0, hash);
        latencies.push(started.elapsed());
        assert_eq!(status, 200, "repair: {body}");
    }
    summarize("repair 64KiB", latencies);

    for hash in &hashes {
        let healthy = await_health(&cluster, 0, hash, "healthy");
        assert_eq!(healthy["usable"], json!(3), "all three pieces restored");
    }
    let after = metrics(&cluster, 0);
    let sent = after["remote_bytes_sent"].as_u64().unwrap_or(0)
        - before["remote_bytes_sent"].as_u64().unwrap_or(0);
    eprintln!(
        "repair bytes: remote_sent={sent} repair_pending={}",
        after["repair_bytes_done"]
    );
    assert!(
        sent >= u64::try_from(ASSET_BYTES).unwrap_or(0),
        "rebuilt fragments crossed the mesh: sent={sent}"
    );
    assert_eq!(after["repair_bytes_pending"], json!(0));
}

/// Storage cost of erasure coding versus replication on the same bytes.
///
/// A 512 KiB asset plans `RS(2,1)` on three domains (1.5x storage); the
/// explicit `replication(3)` transition costs 3x. Both must read back
/// exact, and the coded layout must have stored strictly less.
#[test]
fn coded_layout_stores_less_than_replicated() {
    let cluster = Cluster::spawn().expect("cluster spawns");
    let _leader = cluster.wait_leader();
    let bytes = payload(CODED_BYTES, 0x5EED);
    let hash = asset_hex(&bytes);

    let (status, body) = protect(&cluster, 0, &bytes, 1);
    assert_eq!(status, 200, "protect: {body}");
    assert_eq!(
        body["scheme"],
        json!("reed-solomon"),
        "large assets plan coded: {body}"
    );
    let coded = metrics(&cluster, 0);
    let logical = u64::try_from(CODED_BYTES).unwrap_or(0);
    eprintln!(
        "coded stored={} logical={} amplification={}",
        stored_bytes(&coded),
        logical,
        coded["amplification"]
    );

    // Reads stay exact on the coded layout.
    let (status, body) = read(&cluster, 0, &hash);
    assert_eq!(status, 200, "coded read: {body}");
    assert_eq!(
        unb64(body["bytes_b64"].as_str().expect("image")).as_deref(),
        Some(bytes.as_slice())
    );

    let scheme = json!({ "replication": { "copies": 3 } });
    let (status, body) = transition(&cluster, 0, &hash, &scheme);
    assert_eq!(status, 200, "transition: {body}");
    assert_eq!(body["scheme"], json!("replication"));
    let replicated = metrics(&cluster, 0);
    eprintln!(
        "replicated stored={} logical={} amplification={}",
        stored_bytes(&replicated),
        logical,
        replicated["amplification"]
    );

    let coded_stored = stored_bytes(&coded);
    let replicated_stored = stored_bytes(&replicated);
    assert_eq!(coded_stored, logical + logical / 2, "RS(2,1) stores 1.5x");
    assert_eq!(replicated_stored, logical * 3, "three copies store 3x");
    assert!(
        coded_stored < replicated_stored,
        "coding wins on storage: coded={coded_stored} replicated={replicated_stored}"
    );
    assert_eq!(coded["assets_coded"], json!(1));
    assert_eq!(replicated["assets_replicated"], json!(1));
    assert_eq!(replicated["assets_coded"], json!(0));

    // The replicated generation still reads exact.
    let (status, body) = read(&cluster, 0, &hash);
    assert_eq!(status, 200, "replicated read: {body}");
    assert_eq!(
        unb64(body["bytes_b64"].as_str().expect("image")).as_deref(),
        Some(bytes.as_slice())
    );
    let healthy = await_health(&cluster, 0, &hash, "healthy");
    assert_eq!(healthy["usable"], json!(3));
}
