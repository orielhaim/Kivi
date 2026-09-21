//! Full-cluster acceptance: logical equivalence across topology chaos.
//!
//! A single deterministic driver holds a sequential reference model
//! (`HashMap` for byte keys, `HashMap` for counters) while an ordered
//! cluster (`spawn_ordered(4)`: 3 processes, 4 tablets, the convention
//! every ordered test uses) absorbs splits, merges, kills, and restarts.
//! Every acknowledged write mirrors into the model; the finale reads back
//! every model key exactly, covers the range with a scan, and asserts
//! `total_intents() == 0`. No wall-clock sleeps: only harness waits.
//!
//! One long-lived client drives every phase (load, chaos rounds, verify)
//! over reused pooled connections: retired tablets redirect transitively
//! through tombstone chains, the client detects redirect non-progress
//! (`A → B → A`, older generations) and refreshes from seeds, and every
//! restart waits for exact tiling agreement before traffic resumes.
//!
//! Live fourth-node admission is covered by `node_admission` (one
//! long-lived client through admission, directory catch-up, lineage
//! churn, and rebalance with no warmup); this driver stays on the
//! 3-process topology and focuses on split/merge/kill/restart chaos.
//!
//! Split targeting is largest-first by design: cross-tablet batches
//! coordinate on the smallest participant id, so the coordinator tablet
//! never retires and any transiently leaked intent stays resolvable (the
//! server resolver reaps it past the transaction lease; the driver drains
//! intents every round to prove it). Smallest-first rotation retires
//! coordinators; the resolver follows migrated records through unified
//! routing, covered by `coordinator_retirement_resolves_intents`.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use bytes::Bytes;
use kivi_client::ordered::{
    BatchExpect, BatchWriteKind, BatchWriteSpec, ScanConsistency, ScanDirection, ScanOptions,
    ScanProjection,
};
use kivi_lab::cluster::Cluster;
use kivi_state::Key;
use kivi_types::NamespaceId;

const NS: NamespaceId = NamespaceId::from_u64(1);

const SMALL_COUNT: usize = 200;
const MEDIUM_COUNT: usize = 200;
const COUNTER_COUNT: usize = 20;
const ROUND_KEYS: usize = 50;
const ROUNDS: usize = 3;

/// Genesis tiling prefixes: `[0,64)`, `[64,128)`, `[128,192)`, `[192,+inf)`.
fn spread_key(tablet: u8, index: usize) -> Vec<u8> {
    let prefix = match tablet {
        0 => 0x10u8,
        1 => b'A',
        2 => 0x90u8,
        _ => 0xF0u8,
    };
    let mut key = vec![prefix];
    key.extend_from_slice(format!("{index:06}").as_bytes());
    key
}

/// Deterministic LCG bytes (no randomness, seeded patterns only).
fn fill(len: usize, seed: u8) -> Vec<u8> {
    let mut out = vec![0u8; len];
    let mut state = u64::from(seed) | 1;
    for byte in &mut out {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *byte = u8::try_from((state >> 33) & 0xFF).unwrap_or(0xFF);
    }
    out
}

fn small_value(index: usize) -> Vec<u8> {
    format!("small-{index:06}").into_bytes()
}

fn medium_value(index: usize) -> Vec<u8> {
    let len = 2048 + (index % 3) * 1024;
    let seed = u8::try_from(index % 251).unwrap_or(0x5A);
    fill(len, seed)
}

fn tablet_of(index: usize) -> u8 {
    u8::try_from(index % 4).unwrap_or(0)
}

/// Best-effort liveness snapshot for failure messages (never masks the
/// original error with a second panic: dead slots are skipped, wedged
/// admin answers degrade to `up(?)`). Client counters distinguish a
/// redirect storm (`redirects` far above `requests`) from dead dials.
fn liveness(cluster: &Cluster, client: &kivi_client::NativeClient) -> String {
    let mut parts = Vec::new();
    let mut tablets = 0usize;
    let mut intents = 0u64;
    for index in 0..cluster.member_count() {
        if !cluster.alive(index) {
            parts.push(format!("m{index}:down"));
            continue;
        }
        match cluster.tablets_status_opt(index) {
            Some(status) => {
                let entries = status.as_array().cloned().unwrap_or_default();
                tablets += entries.len();
                let mut holding: Vec<String> = Vec::new();
                let mut member_intents = 0u64;
                for entry in &entries {
                    let count = entry["intent_count"].as_u64().unwrap_or(0);
                    member_intents += count;
                    if count > 0 {
                        holding.push(format!(
                            "{}:{}",
                            entry["group"].as_u64().unwrap_or(u64::MAX),
                            count
                        ));
                    }
                }
                intents += member_intents;
                parts.push(format!(
                    "m{index}:up({}t,{}i[{}])",
                    entries.len(),
                    member_intents,
                    holding.join(",")
                ));
            }
            None => parts.push(format!("m{index}:up(?)")),
        }
    }
    let stats = client.stats();
    format!(
        "{} tablets_total={tablets} intents={intents} req={} redir={} errs={}",
        parts.join(" "),
        stats.requests,
        stats.redirects,
        stats.errors
    )
}

/// Point write that must ack; mirrors into the model only on ack.
fn put_checked(
    cluster: &Cluster,
    client: &kivi_client::NativeClient,
    model: &mut HashMap<Vec<u8>, Vec<u8>>,
    key: Vec<u8>,
    value: Vec<u8>,
) {
    client
        .set(&Key::from(key.clone()), Bytes::copy_from_slice(&value))
        .unwrap_or_else(|error| {
            panic!(
                "set {} commits: {error:?} [{}]",
                hex_key(&key),
                liveness(cluster, client)
            )
        });
    model.insert(key, value);
}

/// Counter add that must ack; mirrors the new value into the model.
fn counter_checked(
    cluster: &Cluster,
    client: &kivi_client::NativeClient,
    model: &mut HashMap<Vec<u8>, i64>,
    key: Vec<u8>,
    delta: i64,
) {
    let live = client
        .counter_add(&Key::from(key.clone()), delta)
        .unwrap_or_else(|error| {
            panic!(
                "counter_add {} commits: {error:?} [{}]",
                hex_key(&key),
                liveness(cluster, client)
            )
        });
    model.insert(key, live);
}

fn hex_key(key: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for byte in key.iter().take(8) {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Where one key probes right now: `(tablet, dir_version)` or the error.
/// Uses the same client cache the batch driver trusts.
fn probe_of(client: &kivi_client::NativeClient, key: &[u8]) -> String {
    match client.scan_page(
        NS,
        Some(key.to_vec()),
        Some(kivi_client::key_successor(key)),
        ScanDirection::Forward,
        ScanConsistency::LatestPerTablet,
        ScanProjection::KeysOnly,
        1,
        64,
    ) {
        Ok(page) => format!("t{}@v{}", page.tablet, page.dir_version),
        Err(error) => format!("err({error:?})"),
    }
}

/// Failure-path wedge diagnosis (runs only when the batch already failed,
/// so its own mutations cannot taint the verdict): for each batch key, a
/// point `set` on an adjacent probe key (same range, fresh key) tests
/// whether the tablet serves writes at all, and a single-key batch on the
/// exact key tests whether a foreign intent wedges exactly that key.
/// `put=Ok/batch=Ok` on all three means the keys are clean and the
/// conflict is emergent (coordinator/decide level); `batch=Conflict`
/// names the wedged key; `put=Err` names a dead tablet.
fn diagnose_wedge(client: &kivi_client::NativeClient, writes: &[(Vec<u8>, Vec<u8>)]) -> String {
    let mut parts = Vec::new();
    for (key, _) in writes {
        let mut probe_key = key.clone();
        probe_key.push(0x00);
        let put = match client.set(
            &Key::from(probe_key),
            bytes::Bytes::from_static(b"wedge-probe"),
        ) {
            Ok(()) => "Ok".to_owned(),
            Err(error) => format!("Err({error:?})"),
        };
        let single = [BatchWriteSpec {
            key: key.clone(),
            kind: BatchWriteKind::Put(b"wedge-probe".to_vec()),
            expect: BatchExpect::Any,
        }];
        let batch = match client.atomic_batch(NS, &single) {
            Ok(_) => "Ok".to_owned(),
            Err(error) => format!("{error:?}"),
        };
        parts.push(format!("{}:put={put}:batch={batch}", hex_key(key)));
    }
    parts.join(" ")
}

/// Cross-tablet batch; mirrors every put into the model only on ack.
///
/// A batch that lands on a just-split/merged directory can lose the OCC
/// race transiently (`TxnConflict`) even though nothing semantic conflicts:
/// the probe planned against a retired tablet. Worse, a conflicted attempt
/// can leak its already-prepared intents (the abort wave is best-effort),
/// and any later attempt touching those keys conflicts until the server
/// resolver reaps the lease-expired orphans. So every conflict retry first
/// converges the cluster, then drains intents (instant when clean, up to
/// the lease horizon when a leak must be reaped), and only then redrives
/// under a fresh id. Every other error fails loudly, and exhaustion fails
/// loudly too. Mirroring only the acked outcome keeps the model exact.
fn batch_checked(
    cluster: &Cluster,
    client: &kivi_client::NativeClient,
    model: &mut HashMap<Vec<u8>, Vec<u8>>,
    writes: Vec<(Vec<u8>, Vec<u8>)>,
) {
    let specs: Vec<BatchWriteSpec> = writes
        .iter()
        .map(|(key, value)| BatchWriteSpec {
            key: key.clone(),
            kind: BatchWriteKind::Put(value.clone()),
            expect: BatchExpect::Any,
        })
        .collect();
    let first_probes: Vec<String> = writes
        .iter()
        .map(|(key, _)| format!("{}={}", hex_key(key), probe_of(client, key)))
        .collect();
    let mut attempt = 0u32;
    loop {
        match client.atomic_batch(NS, &specs) {
            Ok(result) => {
                assert_eq!(
                    result.versions.len(),
                    writes.len(),
                    "batch versions cover every write"
                );
                for (key, value) in writes {
                    model.insert(key, value);
                }
                return;
            }
            Err(kivi_client::ClientError::TxnConflict) if attempt < 8 => {
                attempt += 1;
                cluster.wait_converged_all();
                wait_intents_drained(cluster, Duration::from_secs(150));
            }
            Err(error) => {
                let last_probes: Vec<String> = writes
                    .iter()
                    .map(|(key, _)| format!("{}={}", hex_key(key), probe_of(client, key)))
                    .collect();
                panic!(
                    "atomic_batch commits: {error:?} [{}] tries={} first=[{}] last=[{}] wedge=[{}]",
                    liveness(cluster, client),
                    attempt + 1,
                    first_probes.join(" "),
                    last_probes.join(" "),
                    diagnose_wedge(client, &writes),
                );
            }
        }
    }
}

fn try_full_scan_keys(
    client: &kivi_client::NativeClient,
) -> Result<Vec<Vec<u8>>, kivi_client::ClientError> {
    let options = ScanOptions {
        direction: ScanDirection::Forward,
        consistency: ScanConsistency::LatestPerTablet,
        projection: ScanProjection::KeysOnly,
        max_items_per_page: 100,
        max_bytes_per_page: 1 << 20,
        limit: None,
        ..ScanOptions::default()
    };
    client
        .scan(NS, &options)
        .map(|entries| entries.into_iter().map(|entry| entry.key).collect())
}

/// Per-range scan diagnostics (genesis tiling): counts keys per range or
/// reports the range's error. Distinguishes a dead native plane (every
/// range fails) from a range-specific routing hole (only the churned
/// lineage fails) using a fresh client, so cached routes cannot skew it.
fn diagnose_ranges(client: &kivi_client::NativeClient) -> String {
    let bounds = [
        ("[,@)", None, Some(vec![0x40])),
        ("[@,80)", Some(vec![0x40]), Some(vec![0x80])),
        ("[80,C0)", Some(vec![0x80]), Some(vec![0xC0])),
        ("[C0,)", Some(vec![0xC0]), None),
    ];
    bounds
        .iter()
        .map(|(label, start, end)| {
            let options = ScanOptions {
                start: start.clone(),
                end: end.clone(),
                direction: ScanDirection::Forward,
                consistency: ScanConsistency::LatestPerTablet,
                projection: ScanProjection::KeysOnly,
                max_items_per_page: 100,
                max_bytes_per_page: 1 << 20,
                limit: None,
            };
            match client.scan(NS, &options) {
                Ok(entries) => format!("{label}={}keys", entries.len()),
                Err(error) => format!("{label}=Err({error:?})"),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn full_scan_keys(cluster: &Cluster, client: &kivi_client::NativeClient) -> Vec<Vec<u8>> {
    match try_full_scan_keys(client) {
        Ok(keys) => keys,
        Err(error) => {
            // One fresh-client verdict before failing: fresh-ok means the
            // reused client's pool/cache poisoned the call (client gap),
            // fresh-fail means the native plane itself is down (server
            // gap). Ports tell dead listeners apart from wedged protocol.
            let fresh = cluster.client();
            let fresh_verdict = match try_full_scan_keys(&fresh) {
                Ok(healed) => format!("Ok({} keys)", healed.len()),
                Err(error) => format!("Err({error:?})"),
            };
            let stats = client.stats();
            panic!(
                "covering scan works: {error:?} [{}] \
                 req={} redir={} errs={} fresh client: {fresh_verdict} \
                 ports: {} ranges: {}",
                liveness(cluster, client),
                stats.requests,
                stats.redirects,
                stats.errors,
                probe_ports(cluster),
                diagnose_ranges(&fresh),
            );
        }
    }
}

/// In-situ transport probe: TCP-connects (2 s budget) to every member's
/// native endpoint and reports open/refused/timeout while the servers are
/// still alive to answer. The panic context already proves admin answers
/// (via `liveness`), so native-refused means dead native listeners and
/// native-open means the protocol above TCP is wedged.
fn probe_ports(cluster: &Cluster) -> String {
    use std::net::ToSocketAddrs as _;
    let mut parts = Vec::new();
    for index in 0..cluster.member_count() {
        let endpoint = cluster.native_endpoint(index);
        let state = endpoint
            .to_socket_addrs()
            .ok()
            .and_then(|mut addrs| addrs.next())
            .map_or(
                "unresolvable",
                |addr| match std::net::TcpStream::connect_timeout(&addr, Duration::from_secs(2)) {
                    Ok(_) => "open",
                    Err(error) if error.kind() == std::io::ErrorKind::ConnectionRefused => {
                        "refused"
                    }
                    Err(_) => "timeout",
                },
            );
        let liveness = if cluster.alive(index) {
            "alive"
        } else {
            "down"
        };
        parts.push(format!("m{index}={state}({liveness})"));
    }
    parts.join(" ")
}

/// Splittable tablet id (0 is never a user tablet): smallest or largest.
///
/// Largest-first is the acceptance default: the cross-tablet batches
/// coordinate on the smallest participant id, so splitting the largest
/// keeps every coordinator tablet live for the whole test. A coordinator
/// that retires makes its transactions' intents unresolvable (the
/// resolver reads decisions off the coordinator tablet), turning any
/// transient abort/commit-finalize loss into a permanent per-key wedge.
/// Smallest-first rotation retires coordinators and deterministically
/// wedges (kept as an ignored regression test for that product bug).
fn pick_split_target(cluster: &Cluster, largest: bool) -> u64 {
    let mut tablets: Vec<u64> = cluster
        .tablet_ids(cluster.live_index())
        .into_iter()
        .filter(|id| *id != 0)
        .collect();
    tablets.sort_unstable();
    if largest {
        tablets
            .into_iter()
            .next_back()
            .expect("a splittable tablet")
    } else {
        tablets.into_iter().next().expect("a splittable tablet")
    }
}

/// Waits until no live member reports any prepared intent. A leaked abort
/// (finalize-abort lost under churn, outcome ignored by design) is reaped
/// by the server resolver only after the transaction lease expires, so
/// this can take a minute; when clean it returns on the first poll. Loud
/// deadline, no hope-sleeps. Members that skip a poll round (slow admin
/// fan-out) simply do not count as drained yet.
fn wait_intents_drained(cluster: &Cluster, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        let mut total = 0u64;
        let mut polled = false;
        for index in 0..cluster.member_count() {
            if !cluster.alive(index) {
                continue;
            }
            let Some(status) = cluster.tablets_status_opt(index) else {
                continue;
            };
            polled = true;
            for entry in status.as_array().cloned().unwrap_or_default() {
                total += entry["intent_count"].as_u64().unwrap_or(0);
            }
        }
        if polled && total == 0 {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "stuck intents never drained (last total {total})"
        );
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// Split one tablet, then merge its children back. Waits are harness-only.
///
/// Plan completion is not cutover convergence, so after both plans land
/// the driver waits for exact tiling agreement at the restored width
/// before any client write may assume the new directory.
fn split_then_merge(cluster: &Cluster, parent: u64) {
    let reply = cluster.split_tablet(parent);
    assert_eq!(
        reply.get("ok").and_then(serde_json::Value::as_bool),
        Some(true),
        "split accepted: {reply}"
    );
    let left = reply
        .get("left")
        .and_then(serde_json::Value::as_u64)
        .expect("split left child");
    let right = reply
        .get("right")
        .and_then(serde_json::Value::as_u64)
        .expect("split right child");
    cluster.wait_splits_done(Duration::from_secs(600));
    let merge = cluster.merge_tablets(left, right);
    assert_eq!(
        merge.get("ok").and_then(serde_json::Value::as_bool),
        Some(true),
        "merge accepted: {merge}"
    );
    cluster.wait_merges_done(Duration::from_secs(600));
    cluster.wait_tiling_agreed(4, Instant::now() + Duration::from_secs(300));
}

fn load_phase1(
    cluster: &Cluster,
    client: &kivi_client::NativeClient,
    bytes_model: &mut HashMap<Vec<u8>, Vec<u8>>,
    counter_model: &mut HashMap<Vec<u8>, i64>,
) {
    for index in 0..SMALL_COUNT {
        let key = spread_key(tablet_of(index), index);
        put_checked(cluster, client, bytes_model, key, small_value(index));
    }
    for index in 0..MEDIUM_COUNT {
        let key = spread_key(tablet_of(index), 10_000 + index);
        put_checked(cluster, client, bytes_model, key, medium_value(index));
    }
    for index in 0..COUNTER_COUNT {
        let key = spread_key(tablet_of(index), 50_000 + index);
        let delta = i64::try_from(index % 7).unwrap_or(0) + 1;
        counter_checked(cluster, client, counter_model, key, delta);
    }
}

fn chaos_round(
    cluster: &mut Cluster,
    client: &kivi_client::NativeClient,
    bytes_model: &mut HashMap<Vec<u8>, Vec<u8>>,
    counter_model: &mut HashMap<Vec<u8>, i64>,
    round: usize,
    largest_first: bool,
) {
    let base = 20_000 + round * 1_000;
    for index in 0..ROUND_KEYS {
        let key = spread_key(tablet_of(index), base + index);
        let value = fill(64, u8::try_from((round * 31 + index) % 251).unwrap_or(1));
        put_checked(cluster, client, bytes_model, key, value);
    }
    // One cross-tablet batch of medium puts (spans three tablets).
    let batch: Vec<(Vec<u8>, Vec<u8>)> = [0u8, 1, 3]
        .iter()
        .enumerate()
        .map(|(slot, tablet)| {
            let key = spread_key(*tablet, 30_000 + round * 10 + slot);
            let value = medium_value(round * 10 + slot);
            (key, value)
        })
        .collect();
    batch_checked(cluster, client, bytes_model, batch);
    // Counters keep moving through the chaos window.
    for index in 0..4usize {
        let key = spread_key(
            tablet_of(index),
            50_000 + (round * 4 + index) % COUNTER_COUNT,
        );
        let prior = counter_model.get(&key).copied().unwrap_or(0);
        let live = client
            .counter_add(&Key::from(key.clone()), 2)
            .unwrap_or_else(|error| panic!("chaos counter commits: {error:?}"));
        assert_eq!(live, prior + 2, "counter adds linearly");
        counter_model.insert(key, live);
    }
    split_then_merge(cluster, pick_split_target(cluster, largest_first));
    let victim = round % cluster.member_count();
    let incarnation_before = cluster.incarnation(victim);
    let dir_before = cluster
        .directory(victim)
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    cluster.kill(victim);
    cluster
        .restart(victim)
        .unwrap_or_else(|error| panic!("node {victim} restarts: {error:?}"));
    // One coherent serving-ready predicate (incarnation fencing, stable
    // leaders, converged applied, exact tiling agreement, replay floor):
    // traffic resumes only on real directory state, never on a sleep, so
    // the long-lived client's cached routes cannot outrun a lagging member.
    cluster.wait_member_rejoined(
        victim,
        incarnation_before,
        dir_before,
        4,
        Duration::from_secs(300),
    );
    // Every round ends with zero intents: attributes any leak to the round
    // that made it (and reaps it via the resolver) instead of letting it
    // wedge a later round's batch from afar. Point writes after the
    // restart refresh client routes per key, so no scan is needed here
    // (post-restart scans proved fragile against divergent members).
    wait_intents_drained(cluster, Duration::from_secs(150));
}

/// Scan-first verification: the scan walks the live directory by key, so
/// a model key absent from the scan is lost state, while a key present in
/// the scan but missed by its point `get` is a stale-route read. The
/// report separates the two instead of lumping both as "missing".
fn verify_phase3(
    cluster: &Cluster,
    client: &kivi_client::NativeClient,
    bytes_model: &HashMap<Vec<u8>, Vec<u8>>,
    counter_model: &HashMap<Vec<u8>, i64>,
) {
    use std::collections::BTreeSet;
    let scan_keys = full_scan_keys(cluster, client);
    let scan_set: BTreeSet<&Vec<u8>> = scan_keys.iter().collect();
    let mut scan_missing: Vec<String> = Vec::new();
    for key in bytes_model.keys().chain(counter_model.keys()) {
        if !scan_set.contains(key) && scan_missing.len() < 8 {
            scan_missing.push(hex_key(key));
        }
    }
    let scan_missing_total = bytes_model
        .keys()
        .chain(counter_model.keys())
        .filter(|key| !scan_set.contains(*key))
        .count();
    let mut get_missing: Vec<String> = Vec::new();
    let mut get_mismatch: Vec<String> = Vec::new();
    let mut get_missing_total = 0usize;
    let mut mismatch_total = 0usize;
    for (key, expected) in bytes_model {
        match client.get(&Key::from(key.clone())) {
            Ok(Some(got)) => {
                if got.as_ref() != expected.as_slice() {
                    mismatch_total += 1;
                    if get_mismatch.len() < 8 {
                        get_mismatch.push(hex_key(key));
                    }
                }
            }
            Ok(None) => {
                get_missing_total += 1;
                if get_missing.len() < 8 {
                    get_missing.push(hex_key(key));
                }
            }
            Err(error) => panic!("get {} reads: {error:?}", hex_key(key)),
        }
    }
    let mut counter_missing = 0usize;
    for (key, expected) in counter_model {
        match client.counter_get(&Key::from(key.clone())) {
            Ok(Some(got)) => assert_eq!(got, *expected, "counter-exact"),
            Ok(None) => counter_missing += 1,
            Err(error) => panic!("counter_get {} reads: {error:?}", hex_key(key)),
        }
    }
    assert!(
        scan_missing_total == 0
            && get_missing_total == 0
            && mismatch_total == 0
            && counter_missing == 0,
        "equivalence break: scan_missing={scan_missing_total}{scan_missing:?} \
         get_missing={get_missing_total}{get_missing:?} \
         value_mismatch={mismatch_total}{get_mismatch:?} \
         counter_missing={counter_missing} scan_total={} model_total={}",
        scan_keys.len(),
        bytes_model.len() + counter_model.len(),
    );
    let mut got_sorted = scan_keys;
    got_sorted.sort();
    let mut expected_sorted: Vec<Vec<u8>> = bytes_model
        .keys()
        .chain(counter_model.keys())
        .cloned()
        .collect();
    expected_sorted.sort();
    assert_eq!(
        got_sorted, expected_sorted,
        "scan keys match the model exactly"
    );
    assert_eq!(cluster.total_intents(), 0, "no stuck intents");
}

fn run_acceptance(largest_first: bool) {
    let mut cluster = Cluster::spawn_ordered(4).expect("ordered cluster spawns");
    let _ = cluster.wait_all_leaders();
    // Driver-owned topology: background auto-split/merge would race the
    // manual splits below and churn the directory under single-attempt
    // writes, so it stays off for the whole test.
    let _ = cluster.set_policy(&serde_json::json!({
        "auto_split": false,
        "auto_merge": false,
    }));

    let mut bytes_model: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();
    let mut counter_model: HashMap<Vec<u8>, i64> = HashMap::new();

    // One long-lived client across all phases (same seeds, reused pooled
    // connections, one route cache, one session): retired tablets redirect
    // transitively, non-progress redirects refresh from seeds, and every
    // restart gates on tiling agreement — so cached routes converge
    // instead of pinning dead ranges. The model (not fresh caches)
    // carries continuity across phases.
    let client = cluster.client();
    load_phase1(&cluster, &client, &mut bytes_model, &mut counter_model);
    cluster.wait_converged_all();

    for round in 0..ROUNDS {
        chaos_round(
            &mut cluster,
            &client,
            &mut bytes_model,
            &mut counter_model,
            round,
            largest_first,
        );
    }

    // One checkpoint trigger on a live tablet: equivalence must survive it.
    let live = cluster.live_index();
    let tablet = pick_split_target(&cluster, largest_first);
    let _ = cluster.snapshot_tablet(live, tablet);
    cluster.wait_converged_all();

    verify_phase3(&cluster, &client, &bytes_model, &counter_model);
    eprintln!(
        "acceptance: {} byte keys + {} counters verified, {} rounds of chaos",
        bytes_model.len(),
        counter_model.len(),
        ROUNDS
    );
    cluster.kill_all();
}

#[test]
fn full_cluster_acceptance_across_topology_chaos() {
    run_acceptance(true);
}

/// Regression catcher for the coordinator-retirement intent wedge:
/// smallest-first rotation retires 2PC coordinator tablets, and any
/// leaked prepare intent must still resolve (the resolver routes the
/// decision record by the unified rule instead of pinning the retired
/// coordinator). Formerly ignored while the wedge was a product bug;
/// now a gate.
#[test]
fn coordinator_retirement_resolves_intents() {
    run_acceptance(false);
}
