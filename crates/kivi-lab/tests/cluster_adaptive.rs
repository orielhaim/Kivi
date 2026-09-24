//! Multi-process acceptance coverage for the cluster adaptive shadow runtime.

use std::time::{Duration, Instant};

use kivi_lab::cluster::Cluster;
use kivi_state::Key;

fn put(client: &kivi_client::NativeClient, name: &str, value: &[u8]) {
    client
        .set(&Key::from(name), bytes::Bytes::copy_from_slice(value))
        .unwrap_or_else(|error| panic!("set {name} commits: {error:?}"));
}

fn get(client: &kivi_client::NativeClient, name: &str) -> Vec<u8> {
    client
        .get(&Key::from(name))
        .unwrap_or_else(|error| panic!("get {name} reads: {error:?}"))
        .unwrap_or_else(|| panic!("{name} present"))
        .to_vec()
}

#[test]
fn adaptive_shadow_preserves_cluster_data_and_bounds_status() {
    let cluster = Cluster::spawn_full(2, 2, false, &[("KIVI_CLUSTER_ADAPTIVE_SHADOW", "1")])
        .expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let client = cluster.client();
    let keys = cluster.keys_for_tablets(2);
    for names in keys.values() {
        for name in names {
            put(&client, name, b"adaptive-shadow");
        }
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let mut ready = true;
        for index in 0..cluster.member_count() {
            let (status, snapshot) = cluster.admin_get(index, "/v1/control/adaptive");
            assert_eq!(status, 200);
            assert_eq!(snapshot["enabled"], true);
            assert_eq!(snapshot["mode"], "shadow");
            assert_eq!(snapshot["learned_shadow"], true);
            assert_eq!(snapshot["actuated"], 0);
            assert!(
                snapshot["traces"]
                    .as_array()
                    .is_some_and(|trace| trace.len() <= 64)
            );
            assert!(
                snapshot["recommendations"]
                    .as_array()
                    .is_some_and(|recommendations| recommendations.len() <= 64)
            );
            ready &= snapshot["intervals"].as_u64().unwrap_or(0) > 0;
        }
        if ready || Instant::now() >= deadline {
            assert!(ready, "adaptive runtime did not publish an interval");
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    for names in keys.values() {
        for name in names {
            assert_eq!(get(&client, name), b"adaptive-shadow");
        }
    }
}
