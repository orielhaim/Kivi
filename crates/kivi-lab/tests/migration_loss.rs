//! Focused migration-loss reproducer: tight loop of acknowledged writes
//! followed by a leadership-handoff + membership migration of their
//! tablet, verifying exact readback after every step.
//!
//! Background: one `node_admission` run lost a single acknowledged key
//! after migration 1→4 completed (14/15 green). The loss window is
//! somewhere in write → handoff → reconfig → retire → serve, so this
//! test verifies per-replica presence (pinned `get_any` per member)
//! as well as routed reads, and dumps the full Raft timeline
//! (leader/term/membership/applied per member, migration phases,
//! directory versions) on the first mismatch.
//!
//! One long-lived client for routed traffic; per-member pinned clients
//! only for forensic presence checks (they never mutate).

use std::collections::{BTreeSet, HashMap};
use std::time::Duration;

use bytes::Bytes;
use kivi_client::{ClientConfig, NativeClient};
use kivi_lab::cluster::Cluster;
use kivi_state::Key;
use kivi_types::NamespaceId;

const NS: NamespaceId = NamespaceId::from_u64(1);
const ITERATIONS: usize = 50;
const KEYS_PER_ITER: usize = 10;

fn spread_key(slot: usize, index: usize) -> Vec<u8> {
    let prefix = match slot % 4 {
        0 => 0x10u8,
        1 => b'A',
        2 => 0x90u8,
        _ => 0xF0u8,
    };
    let mut key = vec![prefix];
    key.extend_from_slice(format!("{index:06}").as_bytes());
    key
}

fn hex_key(key: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for byte in key.iter().take(8) {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Client pinned to one member's native endpoint: `get_any` serves from
/// that member's local applied state when it hosts the tablet (no
/// barrier, no cross-member redirect), exposing per-replica presence.
fn pinned_client(cluster: &Cluster, index: usize) -> NativeClient {
    NativeClient::new(ClientConfig {
        seeds: vec![cluster.native_endpoint(index)],
        namespace: NS,
        ..ClientConfig::default()
    })
    .expect("pinned client builds")
}

fn presence(cluster: &Cluster, key: &[u8]) -> Vec<(usize, Option<Vec<u8>>)> {
    let mut out = Vec::new();
    for index in 0..cluster.member_count() {
        if !cluster.alive(index) {
            out.push((index, None));
            continue;
        }
        let value = pinned_client(cluster, index)
            .get_any(&Key::from(key.to_vec()))
            .ok()
            .and_then(|(value, _)| value)
            .map(|bytes| bytes.to_vec());
        out.push((index, value));
    }
    out
}

/// Per-member Raft state for one tablet group.
fn tablet_state_dump(cluster: &Cluster, tablet: u64) -> String {
    let mut parts = Vec::new();
    for index in 0..cluster.member_count() {
        if !cluster.alive(index) {
            parts.push(format!("m{index}:down"));
            continue;
        }
        let row = cluster.tablets_status_opt(index).and_then(|status| {
            status
                .as_array()
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .find(|entry| {
                    entry.get("group").and_then(serde_json::Value::as_u64) == Some(tablet)
                })
        });
        match row {
            Some(entry) => parts.push(format!(
                "m{index}:role={} leader={} term={} last_log={} committed={} applied={}",
                entry["role"].as_str().unwrap_or("?"),
                entry["leader"]
                    .as_u64()
                    .map_or("none".to_owned(), |l| l.to_string()),
                entry["term"].as_u64().unwrap_or(0),
                entry["last_log"].as_u64().unwrap_or(0),
                entry["committed"].as_u64().unwrap_or(0),
                entry["applied"].as_u64().unwrap_or(0),
            )),
            None => parts.push(format!("m{index}:no-group")),
        }
    }
    parts.join(" ")
}

fn directory_versions(cluster: &Cluster) -> String {
    let mut parts = Vec::new();
    for index in 0..cluster.member_count() {
        if !cluster.alive(index) {
            parts.push(format!("m{index}:down"));
            continue;
        }
        let version = cluster
            .directory(index)
            .get("version")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        parts.push(format!("m{index}:v{version}"));
    }
    parts.join(" ")
}

fn migration_plans(cluster: &Cluster) -> String {
    let (_, json) = cluster.admin_get(cluster.live_index(), "/v1/control/migrations");
    json.get("migrations")
        .and_then(serde_json::Value::as_array)
        .map_or("[]".to_owned(), |plans| {
            plans
                .iter()
                .map(|plan| {
                    format!(
                        "{}:{}->{}:{}",
                        plan["tablet"].as_u64().unwrap_or(0),
                        plan["from"].as_u64().unwrap_or(0),
                        plan["to"].as_u64().unwrap_or(0),
                        plan["phase"].as_str().unwrap_or("?"),
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        })
}

#[allow(clippy::too_many_lines)]
#[test]
fn migration_preserves_acked_writes_across_handoff() {
    let mut cluster = Cluster::spawn_ordered(4).expect("ordered cluster spawns");
    let _ = cluster.set_policy(&serde_json::json!({
        "auto_split": false,
        "auto_merge": false,
    }));
    let client = cluster.client();

    // Fourth member joins once up front (admission itself is covered by
    // `node_admission`; here it is the standing migration target).
    let _fourth = cluster.add_node();
    cluster.wait_control_nodes(4, Duration::from_secs(90));

    // Migrate the highest-id live tablet back and forth 1<->4.
    let tablet: u64 = cluster
        .tablet_ids(cluster.live_index())
        .into_iter()
        .filter(|id| *id != 0)
        .max()
        .expect("a live tablet");
    // Sanity: the migrating tablet's whole range serves before we start.
    let mut model: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();

    for iter in 0..ITERATIONS {
        let base = iter * KEYS_PER_ITER;
        // Keys spread over all four genesis slots; slot 3 (0xF0) lands
        // on the highest tablet in the genesis tiling.
        for index in 0..KEYS_PER_ITER {
            let key = spread_key(index, 500_000 + base + index);
            let value = format!("migloss-{base}-{index:02}").into_bytes();
            client
                .set(&Key::from(key.clone()), Bytes::copy_from_slice(&value))
                .unwrap_or_else(|error| panic!("iter {iter}: set failed: {error:?}"));
            model.insert(key, value);
        }
        // Alternate direction each iteration.
        let (from, to, voters) = if iter % 2 == 0 {
            (1u64, 4u64, vec![2u64, 3, 4])
        } else {
            (4u64, 1u64, vec![1u64, 2, 3])
        };
        let _ = cluster.move_tablet(tablet, from, to);
        cluster.wait_tablet_voters(tablet, &voters, Duration::from_secs(240));
        cluster.wait_migrations_done(Duration::from_secs(240));

        // Routed readback for this iteration's keys.
        for index in 0..KEYS_PER_ITER {
            let key = spread_key(index, 500_000 + base + index);
            let expected = model.get(&key).expect("model has key");
            let got = client
                .get(&Key::from(key.clone()))
                .unwrap_or_else(|error| panic!("iter {iter}: get failed: {error:?}"));
            if got.as_deref() != Some(expected.as_slice()) {
                let present: Vec<String> = presence(&cluster, &key)
                    .into_iter()
                    .map(|(member, value)| {
                        format!(
                            "m{member}:{}",
                            value.map_or("absent".to_owned(), |v| format!("{}b", v.len()))
                        )
                    })
                    .collect();
                panic!(
                    "iter {iter}: acked key {} lost after migration {from}->{to}\n\
                     routed got: {:?}\nper-replica: [{}]\ntablet state: {}\n\
                     directories: {}\nplans: {}",
                    hex_key(&key),
                    got.as_deref().map(hex_key),
                    present.join(" "),
                    tablet_state_dump(&cluster, tablet),
                    directory_versions(&cluster),
                    migration_plans(&cluster),
                );
            }
        }
        // Per-replica presence for the migrating tablet's keys: every
        // retained voter must hold them (catch-up completeness), so a
        // silent divergence is caught even when routed reads still hit
        // a healthy replica.
        for index in 0..KEYS_PER_ITER {
            let key = spread_key(index, 500_000 + base + index);
            let expected = model.get(&key).expect("model has key");
            for member in 0..cluster.member_count() {
                if !cluster.alive(member) {
                    continue;
                }
                // Only members hosting the tablet must hold the key;
                // others (retired source) are exempt.
                let hosts = cluster
                    .tablets_status_opt(member)
                    .and_then(|status| {
                        status
                            .as_array()
                            .cloned()
                            .unwrap_or_default()
                            .into_iter()
                            .find(|entry| {
                                entry.get("group").and_then(serde_json::Value::as_u64)
                                    == Some(tablet)
                            })
                    })
                    .is_some();
                if !hosts {
                    continue;
                }
                let value = pinned_client(&cluster, member)
                    .get_any(&Key::from(key.clone()))
                    .ok()
                    .and_then(|(value, _)| value);
                assert_eq!(
                    value.as_deref(),
                    Some(expected.as_slice()),
                    "iter {iter}: member {member} replica diverges on {} [{}]",
                    hex_key(&key),
                    tablet_state_dump(&cluster, tablet),
                );
            }
        }
        if iter % 5 == 4 {
            eprintln!(
                "migration-loss probe: {}/{} iterations clean",
                iter + 1,
                ITERATIONS
            );
        }
    }
    // Full-model equivalence at the end.
    let mut missing = 0usize;
    for (key, expected) in &model {
        let got = client
            .get(&Key::from(key.clone()))
            .expect("final get works");
        if got.as_deref() != Some(expected.as_slice()) {
            missing += 1;
        }
    }
    assert_eq!(missing, 0, "final equivalence lost {missing} keys");
    assert_eq!(cluster.total_intents(), 0, "no stuck intents");
    let scanned: BTreeSet<Vec<u8>> = client
        .scan(
            NS,
            &kivi_client::ordered::ScanOptions {
                direction: kivi_client::ordered::ScanDirection::Forward,
                consistency: kivi_client::ordered::ScanConsistency::LatestPerTablet,
                projection: kivi_client::ordered::ScanProjection::KeysOnly,
                max_items_per_page: 100,
                max_bytes_per_page: 1 << 20,
                limit: None,
                ..kivi_client::ordered::ScanOptions::default()
            },
        )
        .expect("final scan works")
        .into_iter()
        .map(|entry| entry.key)
        .collect();
    assert!(
        model.keys().all(|key| scanned.contains(key)),
        "final scan covers the model"
    );
    eprintln!(
        "migration-loss probe: {ITERATIONS}/{ITERATIONS} clean, {} keys verified",
        model.len()
    );
}
