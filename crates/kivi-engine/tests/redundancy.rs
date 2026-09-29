//! Redundancy fabric wiring through the public `LocalEngine` API: durable
//! bootstrap opens the fabric (ephemeral mode holds none), chunk staging
//! protects chunks plus manifests, checkpoint publication protects bands
//! plus dedup plus manifest, and restarts recover published layouts.
//!
//! Channel-only engines (plain worker threads, no sockets): the embedded
//! path exercises the same worker/chunk-lane/checkpoint flow as production.

mod harness;

use std::time::{Duration, Instant};

use harness::{
    await_current, config, durable_config, pattern_bytes, root_placement, root_snapshot,
};
use kivi_engine::{ChunkFabricConfig, LocalEngine, RedundancyAdminSnapshot};
use kivi_state::Key;

fn start(dir: &std::path::Path) -> LocalEngine {
    LocalEngine::start(durable_config(
        dir,
        ChunkFabricConfig::default(),
        kivi_engine::FabricConfig::default(),
    ))
    .expect("durable engine starts")
}

/// Total protected assets in the snapshot (replicated plus coded).
fn assets(snapshot: &RedundancyAdminSnapshot) -> u64 {
    snapshot.metrics.replicated_assets + snapshot.metrics.coded_assets
}

/// Waits until the fabric holds at least `min` protected assets (checkpoint
/// protection runs on the background thread after CURRENT installs).
fn await_assets(engine: &LocalEngine, min: u64) -> RedundancyAdminSnapshot {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let snapshot = engine
            .redundancy_snapshot()
            .expect("durable engine holds a fabric");
        if assets(&snapshot) >= min {
            return snapshot;
        }
        assert!(
            Instant::now() <= deadline,
            "fabric holds {} assets, wanted {min}",
            assets(&snapshot)
        );
        std::thread::sleep(Duration::from_millis(25));
    }
}

#[test]
fn ephemeral_holds_no_redundancy_fabric() {
    let engine = LocalEngine::start(config(root_snapshot(), root_placement(), 2))
        .expect("ephemeral engine starts");
    assert!(
        engine.redundancy_snapshot().is_none(),
        "ephemeral mode stages nothing durable, so it holds no fabric"
    );
    assert!(
        engine.admin_handle().redundancy_snapshot().is_none(),
        "admin twin agrees"
    );
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn durable_staging_protects_chunks_and_manifest() {
    let scratch = tempfile::tempdir().expect("scratch");
    let engine = start(scratch.path());
    assert!(
        engine.redundancy_snapshot().is_some(),
        "durable bootstrap opens the fabric"
    );
    let client = engine.client();
    // A 300 KiB value crosses the default 256 KiB threshold: chunked
    // staging protects each chunk plus the manifest before the set proves.
    let big = pattern_bytes(300_000, 0xB16);
    client.set(&Key::from("big"), big.clone()).expect("set");
    assert_eq!(
        client.get(&Key::from("big")).expect("get"),
        Some(big),
        "chunked bytes read back exactly"
    );
    let snapshot = await_assets(&engine, 2);
    assert!(
        snapshot.metrics.logical_bytes > 0,
        "protected logical bytes are counted"
    );
    assert!(
        snapshot.census_fragments > 0 && snapshot.census_nodes >= 1,
        "fragments are placed and counted"
    );
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn checkpoint_publication_protects_artifacts_and_restart_recovers() {
    let scratch = tempfile::tempdir().expect("scratch");
    let engine = start(scratch.path());
    let client = engine.client();
    for index in 0..3u32 {
        client
            .set(
                &Key::from(format!("k:{index:03}")),
                bytes::Bytes::from(format!("v{index}")),
            )
            .expect("set");
    }
    engine.request_checkpoint(None);
    let current = await_current(scratch.path(), 3);
    assert_eq!(current.cut, 3);
    // Bands (all newly written on a first checkpoint) plus dedup plus
    // manifest are submitted to the background lane at publish; the asset
    // count flips only after the lane job runs, so polling (not the
    // checkpoint-completed counter) proves protection done.
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if engine
            .admin_handle()
            .checkpoint_state()
            .checkpoints_completed
            >= 1
        {
            break;
        }
        assert!(
            Instant::now() <= deadline,
            "checkpoint never completed after 60s"
        );
        std::thread::sleep(Duration::from_millis(25));
    }
    let before = await_assets(&engine, kivi_checkpoint::BANDS_PER_TABLET as u64 + 2);
    assert_eq!(
        assets(&before),
        kivi_checkpoint::BANDS_PER_TABLET as u64 + 2,
        "first checkpoint protects every new band plus dedup plus manifest"
    );
    engine.shutdown().expect("clean shutdown");
    // Restart recovers published layouts: the asset count survives intact.
    let engine = start(scratch.path());
    let snapshot = engine
        .redundancy_snapshot()
        .expect("restarted engine holds a fabric");
    assert_eq!(
        assets(&snapshot),
        assets(&before),
        "restart recovers every published layout"
    );
    assert_eq!(
        engine.client().get(&Key::from("k:001")).expect("get"),
        Some(bytes::Bytes::from("v1")),
        "checkpoint state restores across the restart"
    );
    engine.shutdown().expect("clean shutdown");
}
