//! Real multi-process distributed redundancy acceptance suite.
//!
//! Three `kivi-server --cluster-mode` processes with independent data
//! directories form the cluster. Fragments live under each node's
//! `<data_dir>/redundancy/fragments`; the control group (Raft) owns which
//! generation is published. Nothing here pretends a shared directory is a
//! remote node: every kill/restart is a real process, every wipe removes
//! only that process's fragment store, and every read crosses the real
//! peer mesh.
//!
//! Coverage:
//! * replication: protect across nodes, wipe a holder's store, remote read,
//!   repair, healthy again;
//! * Reed-Solomon: explicit `RS(2,1)` transition, cross-process shard
//!   reconstruction with exact bytes, repair, fail-closed beyond tolerance;
//! * restart: coordinator + holder restart, catalog + fragments recover;
//! * drain: replacement protection established elsewhere, old fragment
//!   retired, no read gap;
//! * transition with a node down: fails closed, the old layout keeps
//!   serving (no protection gap);
//! * corruption: a bit-flipped remote fragment is detected, never served,
//!   and repaired;
//! * duplicate repair: two concurrent repairs converge idempotently;
//! * orphans: staged fragments without authority leak but never answer reads.

use std::time::{Duration, Instant};

use kivi_lab::cluster::Cluster;
use serde_json::{Value, json};

/// Generous deadlines: spawning three real processes plus elections,
/// fragment transfer, and Raft commits dominates wall time.
const DEADLINE: Duration = Duration::from_secs(180);
const POLL: Duration = Duration::from_millis(250);

/// Asset family discriminant for generic immutable bytes (BLAKE3 identity).
const KIND_GENERIC: u8 = 5;

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

/// Fragment store root of one live cluster member.
fn fragments_root(cluster: &Cluster, index: usize) -> std::path::PathBuf {
    cluster.data_dir(index).join("redundancy").join("fragments")
}

/// Protects bytes through the admin plane, failing loudly with the body.
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

/// Polls until `check` returns true or the deadline passes, then panics with
/// the last observed body.
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

/// Replication across real nodes: protect, wipe a holder's physical store
/// while it is down, restart it, read REMOTELY (it holds no fragment), then
/// repair back to healthy.
#[test]
fn replication_remote_read_and_repair() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let _leader = cluster.wait_leader();
    let bytes = payload(64 * 1024, 0xA11CE);
    let hash = asset_hex(&bytes);
    let (status, body) = protect(&cluster, 0, &bytes, 2);
    assert_eq!(status, 200, "protect: {body}");
    let fragments = body["fragments"].as_array().expect("fragment list");
    assert_eq!(fragments.len(), 3, "tolerance 2 replicates three copies");
    let mut holders: Vec<u64> = fragments
        .iter()
        .filter_map(|entry| entry["node"].as_u64())
        .collect();
    holders.sort_unstable();
    holders.dedup();
    assert_eq!(
        holders.len(),
        3,
        "each copy lives on a distinct node: {holders:?} fragments={fragments:?} body={body}"
    );

    // Wipe node 0's physical store while it is down: the copy is gone from
    // disk, but the catalog still names node 0 (the real failure case).
    cluster.kill(0);
    std::fs::remove_dir_all(fragments_root(&cluster, 0)).expect("fragments wiped");
    cluster.restart(0).expect("node 0 restarts");

    // Node 0 has no usable fragment: the read must fetch a remote copy.
    let (status, body) = read(&cluster, 0, &hash);
    assert_eq!(status, 200, "remote read: {body}");
    assert_eq!(
        unb64(body["bytes_b64"].as_str().expect("image")).as_deref(),
        Some(bytes.as_slice())
    );

    let observed = await_until("replication health after restart", || {
        let (status, body) = health(&cluster, 0, &hash);
        (status == 200 && matches!(body["health"].as_str(), Some("degraded" | "healthy")))
            .then_some(body)
    });
    if observed["health"] == "degraded" {
        let actual = observed["actual"].as_array().expect("actual placement");
        assert!(
            actual
                .iter()
                .any(|entry| entry["node"] == json!(1) && entry["healthy"] == json!(false)),
            "the wiped holder is reported unhealthy: {actual:?}"
        );
        assert!(
            actual
                .iter()
                .filter(|entry| entry["node"] == json!(2) || entry["node"] == json!(3))
                .all(|entry| entry["healthy"] == json!(true)),
            "surviving holders stay healthy: {actual:?}"
        );
    } else {
        assert_eq!(
            observed["usable"],
            json!(3),
            "automatic repair won the race"
        );
    }
    let (status, body) = repair(&cluster, 0, &hash);
    assert_eq!(status, 200, "repair: {body}");
    let healthy = await_health(&cluster, 0, &hash, "healthy");
    assert_eq!(healthy["usable"], json!(3), "all three pieces restored");
    // And a second read now needs no remote fetch.
    let (status, body) = read(&cluster, 0, &hash);
    assert_eq!(status, 200);
    assert_eq!(body["degraded"], json!(false), "local piece restored");
}

/// Reed-Solomon across real processes: transition to `RS(2,1)`, wipe one
/// shard, reconstruct from the other two processes, repair, then kill both
/// remaining holders and fail closed beyond tolerance.
#[test]
fn reed_solomon_cross_process_reconstruction() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let _leader = cluster.wait_leader();
    let bytes = payload(512 * 1024, 0xBEEF);
    let hash = asset_hex(&bytes);
    let (status, body) = protect(&cluster, 0, &bytes, 1);
    assert_eq!(status, 200, "protect: {body}");

    // Three nodes, so the explicit baseline is RS(2,1) (three fragments on
    // three distinct nodes, one failure tolerated).
    let scheme = json!({ "reed_solomon": {
        "data": 2,
        "parity": 1,
        "fragment_len": kivi_redundancy::RsParams::shard_for_len(bytes.len() as u64, 2),
    }});
    let (status, body) = transition(&cluster, 0, &hash, &scheme);
    assert_eq!(status, 200, "transition: {body}");
    assert_eq!(body["scheme"], json!("reed-solomon"));
    let shards = body["params"].as_str().unwrap_or_default();
    assert!(
        shards.contains("ReedSolomon"),
        "coded layout published: {shards}"
    );

    // Wipe node 0's shards: reconstruction must cross two other processes.
    cluster.kill(0);
    std::fs::remove_dir_all(fragments_root(&cluster, 0)).expect("shards wiped");
    cluster.restart(0).expect("node 0 restarts");
    let (status, body) = read(&cluster, 0, &hash);
    assert_eq!(status, 200, "degraded RS read: {body}");
    assert_eq!(
        unb64(body["bytes_b64"].as_str().expect("image")).as_deref(),
        Some(bytes.as_slice()),
        "reconstructed bytes are exact"
    );

    let (status, body) = repair(&cluster, 0, &hash);
    assert_eq!(status, 200, "repair: {body}");
    await_health(&cluster, 0, &hash, "healthy");

    // Beyond tolerance: two dead holders with parity 1 fails closed and
    // never returns wrong bytes.
    let mut dead = Vec::new();
    for index in 1..3usize {
        let (_, body) = health(&cluster, 0, &hash);
        let actual = body["actual"].as_array().expect("actual");
        if actual
            .iter()
            .any(|entry| entry["node"] == json!(index as u64) && entry["healthy"] == json!(true))
        {
            dead.push(index);
        }
    }
    assert_eq!(dead.len(), 2, "two live holders besides node 0");
    for index in dead {
        cluster.kill(index);
    }
    let (status, body) = read(&cluster, 0, &hash);
    assert_ne!(status, 200, "beyond tolerance fails closed: {body}");
    assert!(
        body.get("bytes_b64").is_none(),
        "no bytes are ever returned beyond tolerance: {body}"
    );
}

/// Restart safety: the layout authority (control log) and the fragment
/// store (files) recover independently; reads still work.
#[test]
fn restart_recovers_catalog_and_fragments() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let _leader = cluster.wait_leader();
    let bytes = payload(32 * 1024, 0xC0DE);
    let hash = asset_hex(&bytes);
    let (status, body) = protect(&cluster, 1, &bytes, 2);
    assert_eq!(status, 200, "protect on node 1: {body}");

    // Restart the coordinator (node 1 served protect) and a holder (node 2).
    cluster.restart(1).expect("coordinator restarts");
    cluster.restart(2).expect("holder restarts");

    // Catalog recovered through control replay: every node answers health.
    for index in 0..3usize {
        await_health(&cluster, index, &hash, "healthy");
    }
    let (status, body) = read(&cluster, 0, &hash);
    assert_eq!(status, 200, "read after restart: {body}");
    assert_eq!(
        unb64(body["bytes_b64"].as_str().expect("image")).as_deref(),
        Some(bytes.as_slice())
    );
    let (status, body) = cluster.admin_get(0, "/v1/redundancy/catalog");
    assert_eq!(status, 200, "catalog: {body}");
    let assets = body["assets"].as_array().expect("catalog assets");
    assert!(
        assets.iter().any(|entry| entry["asset"] == json!(hash)),
        "published layout survives coordinator restart: {assets:?}"
    );
}

/// Drain moves protection elsewhere and retires the old fragment only after
/// the replacement is verified; reads keep working throughout.
#[test]
fn drain_moves_protection_without_gap() {
    let cluster = Cluster::spawn().expect("cluster spawns");
    let _leader = cluster.wait_leader();
    let bytes = payload(48 * 1024, 0xD4A1);
    let hash = asset_hex(&bytes);
    // Two copies on two of the three nodes leaves a replacement target.
    let (status, body) = protect(&cluster, 0, &bytes, 1);
    assert_eq!(status, 200, "protect: {body}");
    let holders: Vec<u64> = body["fragments"]
        .as_array()
        .expect("fragments")
        .iter()
        .filter_map(|entry| entry["node"].as_u64())
        .collect();
    assert_eq!(holders.len(), 2, "tolerance 1 replicates twice");
    let victim = holders[0];

    // Reads succeed before, during (polled by the repair path), and after.
    let (status, body) = read(&cluster, 0, &hash);
    assert_eq!(status, 200, "read before drain: {body}");
    let (status, body) = cluster.admin_post(0, "/v1/redundancy/drain", &json!({ "node": victim }));
    assert_eq!(status, 200, "drain: {body}");
    assert_eq!(
        body["moved"],
        json!(1),
        "one asset moved off the drained node"
    );

    let (status, body) = read(&cluster, 0, &hash);
    assert_eq!(status, 200, "read after drain: {body}");
    assert_eq!(
        unb64(body["bytes_b64"].as_str().expect("image")).as_deref(),
        Some(bytes.as_slice())
    );
    let healthy = await_health(&cluster, 0, &hash, "healthy");
    let actual: Vec<u64> = healthy["actual"]
        .as_array()
        .expect("actual")
        .iter()
        .filter_map(|entry| entry["node"].as_u64())
        .collect();
    assert!(
        !actual.contains(&victim),
        "drained node holds no current fragment: {actual:?}"
    );
}

/// A transition that cannot place its fragments fails closed while the old
/// layout keeps serving: no protection gap, no partial publication.
#[test]
fn transition_without_enough_nodes_fails_closed() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let _leader = cluster.wait_leader();
    let bytes = payload(40 * 1024, 0xF00D);
    let hash = asset_hex(&bytes);
    let (status, body) = protect(&cluster, 0, &bytes, 2);
    assert_eq!(status, 200, "protect: {body}");
    let before = await_health(&cluster, 0, &hash, "healthy");

    // Kill one holder: RS(3,1) needs four nodes, so the transition cannot be
    // established and must fail without touching the published layout.
    cluster.kill(2);
    let scheme = json!({ "reed_solomon": {
        "data": 3,
        "parity": 1,
        "fragment_len": kivi_redundancy::RsParams::shard_for_len(bytes.len() as u64, 3),
    }});
    let (status, body) = transition(&cluster, 0, &hash, &scheme);
    assert_ne!(status, 200, "unsatisfiable transition fails: {body}");

    // The replicated layout still serves exactly, degraded but recoverable.
    let (status, body) = read(&cluster, 0, &hash);
    assert_eq!(status, 200, "old layout keeps serving: {body}");
    assert_eq!(
        unb64(body["bytes_b64"].as_str().expect("image")).as_deref(),
        Some(bytes.as_slice())
    );
    let observed = await_until("old layout remains authoritative", || {
        let (status, body) = health(&cluster, 0, &hash);
        (status == 200 && matches!(body["health"].as_str(), Some("degraded" | "healthy")))
            .then_some(body)
    });
    assert_eq!(
        observed["generation"], before["generation"],
        "failed transition did not publish a new generation"
    );
}

/// Corruption: a bit-flipped remote fragment is detected by holder
/// verification, never served, and repaired from the remaining pieces.
#[test]
fn corrupted_remote_fragment_is_detected_and_repaired() {
    let cluster = Cluster::spawn().expect("cluster spawns");
    let _leader = cluster.wait_leader();
    let bytes = payload(64 * 1024, 0x0BAD_F00D);
    let hash = asset_hex(&bytes);
    let (status, body) = protect(&cluster, 0, &bytes, 2);
    assert_eq!(status, 200, "protect: {body}");

    // Flip a byte inside one live holder's committed payload. Only committed
    // fragments are touched (`frag-<gen>-<idx>.bin`); the layout file is the
    // control mirror, not the fragment under test. Corrupting in place (no
    // restart) exercises fetch-time verification, which is what protects
    // reads from a holder that rotted underneath the catalog.
    let root = fragments_root(&cluster, 1);
    let mut corrupted = None;
    for entry in walk(&root) {
        // The name is lowercased first, so the extension test is exact (the
        // lint only wants the case handling spelled out).
        #[allow(clippy::case_sensitive_file_extension_comparisons)]
        let is_fragment = entry
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                let name = name.to_ascii_lowercase();
                name.starts_with("frag-")
                    && std::path::Path::new(&name)
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
            });
        if !is_fragment {
            continue;
        }
        let Ok(mut raw) = std::fs::read(&entry) else {
            continue;
        };
        if raw.len() > 32 {
            let middle = raw.len() / 2;
            raw[middle] ^= 0xFF;
            std::fs::write(&entry, &raw).expect("corrupt write");
            corrupted = Some(entry);
            break;
        }
    }
    let corrupted = corrupted.expect("a fragment file was found to corrupt");

    // Reads stay exact: the healthy local copy answers, so the rotten copy
    // is never even fetched (and never served).
    let (status, body) = read(&cluster, 0, &hash);
    assert_eq!(status, 200, "read tolerates one corrupt copy: {body}");
    assert_eq!(
        unb64(body["bytes_b64"].as_str().expect("image")).as_deref(),
        Some(bytes.as_slice()),
        "corrupt bytes are never exposed"
    );

    // Health probing verifies every holder, so the rot becomes visible
    // there (this is how repair learns about it).
    let degraded = await_health(&cluster, 0, &hash, "degraded");
    assert!(
        degraded["actual"]
            .as_array()
            .expect("actual placement")
            .iter()
            .any(|entry| entry["healthy"] == json!(false)),
        "the corrupt holder is reported unhealthy: {degraded}"
    );
    let (status, metrics) = cluster.admin_get(0, "/v1/redundancy/metrics");
    assert_eq!(status, 200, "metrics: {metrics}");
    assert!(
        metrics["corruption_detected"].as_u64().unwrap_or(0) >= 1,
        "corruption was observed: {metrics}"
    );

    // Repair rewrites the payload on disk: the same file is verified again.
    let (status, body) = repair(&cluster, 0, &hash);
    assert_eq!(status, 200, "repair: {body}");
    let healthy = await_health(&cluster, 0, &hash, "healthy");
    assert_eq!(healthy["usable"], json!(3));
    let restored = std::fs::read(&corrupted).expect("repaired fragment readable");
    assert_eq!(
        restored, bytes,
        "repair rewrote the authoritative bytes on disk, not just the index"
    );
}

/// Background maintenance discovers corruption and repairs it without an
/// operator repair request.
#[test]
fn autonomous_corruption_heals_without_admin_call() {
    let cluster = Cluster::spawn().expect("cluster spawns");
    let _leader = cluster.wait_leader();
    let bytes = payload(48 * 1024, 0xA11CE);
    let hash = asset_hex(&bytes);
    let (status, body) = protect(&cluster, 0, &bytes, 2);
    assert_eq!(status, 200, "protect: {body}");
    let root = fragments_root(&cluster, 1);
    let fragment = walk(&root)
        .into_iter()
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with("frag-")
                        && std::path::Path::new(&name)
                            .extension()
                            .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
                })
        })
        .expect("fragment");
    let mut damaged = std::fs::read(&fragment).expect("fragment bytes");
    let middle = damaged.len() / 2;
    damaged[middle] ^= 0xFF;
    std::fs::write(&fragment, damaged).expect("corrupt write");

    let started = Instant::now();
    let _healthy = await_until("autonomous healthy", || {
        let (status, body) = health(&cluster, 0, &hash);
        if started.elapsed() > Duration::from_secs(30)
            && status == 200
            && body["health"] != "healthy"
        {
            let metrics = cluster.admin_get(0, "/v1/redundancy/metrics").1;
            let maintenance = cluster.admin_get(0, "/v1/redundancy/maintenance").1;
            panic!(
                "autonomous repair stalled: health={body} metrics={metrics} maintenance={maintenance}"
            );
        }
        (status == 200 && body["health"] == "healthy").then_some(body)
    });
    let (status, body) = read(&cluster, 0, &hash);
    assert_eq!(status, 200, "read after autonomous repair: {body}");
    assert_eq!(
        unb64(body["bytes_b64"].as_str().expect("image")).as_deref(),
        Some(bytes.as_slice())
    );
    let (status, metrics) = cluster.admin_get(0, "/v1/redundancy/metrics");
    assert_eq!(status, 200, "metrics: {metrics}");
    assert!(metrics["repair_attempts"].as_u64().unwrap_or(0) >= 1);
    assert!(metrics["scrub_runs"].as_u64().unwrap_or(0) >= 1);
}

/// Background maintenance discovers a physically missing fragment and heals
/// it without an operator repair request.
#[test]
fn autonomous_missing_fragment_heals_without_admin_call() {
    let cluster = Cluster::spawn().expect("cluster spawns");
    let _leader = cluster.wait_leader();
    let bytes = payload(40 * 1024, 0x0BAD_C0DE);
    let hash = asset_hex(&bytes);
    let (status, body) = protect(&cluster, 0, &bytes, 2);
    assert_eq!(status, 200, "protect: {body}");
    let fragment = walk(&fragments_root(&cluster, 2))
        .into_iter()
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with("frag-")
                        && std::path::Path::new(&name)
                            .extension()
                            .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
                })
        })
        .expect("fragment");
    std::fs::remove_file(&fragment).expect("fragment delete");

    let _healthy = await_health(&cluster, 0, &hash, "healthy");
    let (status, body) = read(&cluster, 0, &hash);
    assert_eq!(status, 200, "read after autonomous repair: {body}");
    assert_eq!(
        unb64(body["bytes_b64"].as_str().expect("image")).as_deref(),
        Some(bytes.as_slice())
    );
}

/// Two workers repairing the same deficit converge idempotently.
#[test]
fn duplicate_repair_is_idempotent() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let _leader = cluster.wait_leader();
    let bytes = payload(32 * 1024, 0xD00D);
    let hash = asset_hex(&bytes);
    let (status, body) = protect(&cluster, 0, &bytes, 2);
    assert_eq!(status, 200, "protect: {body}");
    cluster.kill(2);
    std::fs::remove_dir_all(fragments_root(&cluster, 2)).expect("fragments wiped");
    cluster.restart(2).expect("node 2 restarts");
    await_health(&cluster, 0, &hash, "degraded");

    // Two coordinators repair concurrently: both may report work, the end
    // state is one healthy layout with exactly three verified pieces.
    let first = cluster.admin_post(
        0,
        "/v1/redundancy/repair",
        &json!({ "kind": KIND_GENERIC, "domain": 0, "hash_hex": hash }),
    );
    let second = cluster.admin_post(
        1,
        "/v1/redundancy/repair",
        &json!({ "kind": KIND_GENERIC, "domain": 0, "hash_hex": hash }),
    );
    assert_eq!(first.0, 200, "first repair: {:?}", first.1);
    assert_eq!(second.0, 200, "second repair: {:?}", second.1);
    let healthy = await_health(&cluster, 0, &hash, "healthy");
    assert_eq!(healthy["usable"], json!(3), "no duplicated pieces");
    assert_eq!(healthy["total"], json!(3));
}

/// Orphan fragments (staged bytes without authority) leak but never answer
/// reads; the sweep reclaims them without touching protection.
#[test]
fn orphan_fragments_never_become_authoritative() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let _leader = cluster.wait_leader();
    let bytes = payload(16 * 1024, 0x0BAD);
    let hash = asset_hex(&bytes);
    let (status, body) = protect(&cluster, 0, &bytes, 2);
    assert_eq!(status, 200, "protect: {body}");

    // Plant an orphan fragment for a generation the catalog never published.
    cluster.kill(1);
    let asset_dir = fragments_root(&cluster, 1)
        .join("assets")
        .join(format!("generic-immutable-{hash}"));
    std::fs::create_dir_all(&asset_dir).expect("orphan dir");
    std::fs::write(
        asset_dir.join("frag-999-0000.bin"),
        b"orphan bytes nobody published",
    )
    .expect("orphan planted");
    cluster.restart(1).expect("node 1 restarts");

    // Reads still answer from the published generation only.
    let (status, body) = read(&cluster, 0, &hash);
    assert_eq!(status, 200, "orphan never answers: {body}");
    assert_eq!(
        unb64(body["bytes_b64"].as_str().expect("image")).as_deref(),
        Some(bytes.as_slice())
    );
    // Health reports exactly the published pieces, never the orphan.
    let healthy = await_health(&cluster, 0, &hash, "healthy");
    assert_eq!(healthy["total"], json!(3), "orphan is not in the layout");

    // The sweep reclaims it on the holder that leaked it (cluster-wide, not
    // just the coordinator's own disk).
    let swept = await_until("orphan sweep", || {
        let (status, body) = cluster.admin_post(0, "/v1/redundancy/sweep-orphans", &json!({}));
        (status == 200 && body["swept"].as_u64().unwrap_or(0) >= 1).then_some(body)
    });
    assert!(swept["swept"].as_u64().unwrap_or(0) >= 1);
    assert!(
        swept["unreachable"]
            .as_array()
            .expect("unreachable list")
            .is_empty(),
        "every holder answered the sweep: {swept}"
    );
    assert!(
        !asset_dir.join("frag-999-0000.bin").exists(),
        "the leaked bytes are gone from the holder's disk"
    );
    // Protection is untouched by the sweep.
    let (status, body) = read(&cluster, 0, &hash);
    assert_eq!(status, 200, "reads still exact after sweep: {body}");
}

/// Bounded mixed workload: repeated protected writes, a holder restart,
/// deletion, corruption, and eventual convergence without manual repair.
#[test]
fn bounded_self_healing_chaos_converges() {
    let mut cluster = Cluster::spawn().expect("cluster spawns");
    let _leader = cluster.wait_leader();
    let mut assets = Vec::new();
    for index in 0..4u64 {
        let bytes = payload(32 * 1024, 0x00C0_FFEE + index);
        let hash = asset_hex(&bytes);
        let (status, body) = protect(&cluster, 0, &bytes, 2);
        assert_eq!(status, 200, "protect {index}: {body}");
        assets.push((hash, bytes));
    }
    cluster.kill(1);
    cluster.restart(1).expect("holder restart");
    for (index, (hash, _)) in assets.iter().enumerate() {
        let root = fragments_root(&cluster, index % 3)
            .join("assets")
            .join(format!("generic-immutable-{hash}"));
        let files = walk(&root);
        let fragment = files
            .into_iter()
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.starts_with("frag-")
                            && std::path::Path::new(&name)
                                .extension()
                                .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
                    })
            })
            .expect("fragment");
        if index % 2 == 0 {
            std::fs::remove_file(fragment).expect("delete fragment");
        } else {
            let mut bytes = std::fs::read(&fragment).expect("fragment bytes");
            let middle = bytes.len() / 2;
            bytes[middle] ^= 0x5A;
            std::fs::write(fragment, bytes).expect("corrupt fragment");
        }
        let _ = hash;
    }
    for (hash, bytes) in &assets {
        let started = Instant::now();
        let _healthy = await_until("chaos healthy", || {
            let (status, body) = health(&cluster, 0, hash);
            if started.elapsed() > Duration::from_secs(90)
                && status == 200
                && body["health"] != "healthy"
            {
                let metrics = cluster.admin_get(0, "/v1/redundancy/metrics").1;
                let node_metrics: Vec<_> = (0..3)
                    .map(|index| cluster.admin_get(index, "/v1/redundancy/metrics").1)
                    .collect();
                let ticks: Vec<_> = (0..3)
                    .map(|index| {
                        cluster
                            .admin_post(index, "/v1/redundancy/maintenance/tick", &json!({}))
                            .1
                    })
                    .collect();
                let tick = ticks;
                let peers = cluster.admin_get(0, "/v1/peers").1;
                let maintenance_nodes: Vec<_> = (0..3)
                    .map(|index| cluster.admin_get(index, "/v1/redundancy/maintenance").1)
                    .collect();
                panic!(
                    "chaos repair stalled for {hash}: health={body} metrics={metrics} nodes={node_metrics:?} tick={tick:?} peers={peers} maintenance={maintenance_nodes:?}"
                );
            }
            (status == 200 && body["health"] == "healthy").then_some(body)
        });
        let (status, body) = read(&cluster, 0, hash);
        assert_eq!(status, 200, "read {hash}: {body}");
        assert_eq!(
            unb64(body["bytes_b64"].as_str().expect("image")).as_deref(),
            Some(bytes.as_slice())
        );
    }
    let (status, metrics) = cluster.admin_get(0, "/v1/redundancy/metrics");
    assert_eq!(status, 200, "metrics: {metrics}");
    assert_eq!(metrics["unrecoverable"], json!(0));
    assert_eq!(metrics["pending_repairs"], json!(0));
}

/// Yields every regular file under `root` (deterministic order).
fn walk(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return out;
    };
    let mut paths: Vec<std::path::PathBuf> = entries.flatten().map(|entry| entry.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            out.extend(walk(&path));
        } else {
            out.push(path);
        }
    }
    out
}
