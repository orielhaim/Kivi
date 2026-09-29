//! Shared engine fixtures: directory layouts, placement, and durable configs.
//!
//! Every integration test in this crate drives a real `LocalEngine` over real
//! worker threads, so the same bootstrap (one active tablet, or a two-way
//! split, plus a per-worker chunk lane and fabric) is needed everywhere. The
//! helpers here build it once so each test file states only what it is about.

#![allow(dead_code, reason = "each test binary uses a different subset")]

use kivi_engine::{
    ChunkFabricConfig, DurabilityMode, DurableConfig, EngineConfig, FabricConfig, HardwareConfig,
    LocalEngine, Placement,
};
use kivi_tablet::{DirectorySnapshot, HashPrefix, PartitionRange};
use kivi_types::{NamespaceId, TabletEpoch, TabletId, WorkerId, WriteGuardGeneration};

/// The namespace every test serves.
pub const NS: NamespaceId = NamespaceId::from_u64(1);
/// The root tablet before any split.
pub const ROOT: TabletId = TabletId::from_u64(1);
/// Left child of the root split (hash prefix `0*`).
pub const LEFT: TabletId = TabletId::from_u64(2);
/// Right child of the root split (hash prefix `1*`).
pub const RIGHT: TabletId = TabletId::from_u64(3);

fn hash_range(bits: u128, len: u8) -> PartitionRange {
    PartitionRange::Hash(HashPrefix::new(bits, len).expect("range"))
}

fn activate(snapshot: &DirectorySnapshot, tablet: TabletId) -> DirectorySnapshot {
    snapshot
        .clone()
        .stage(tablet)
        .and_then(|staged| staged.activate(tablet))
        .expect("stage+activate")
}

/// One active root tablet covering the whole hash space.
#[must_use]
pub fn root_snapshot() -> DirectorySnapshot {
    activate(
        &DirectorySnapshot::bootstrap(
            NS,
            ROOT,
            hash_range(0, 0),
            TabletEpoch::INITIAL,
            WriteGuardGeneration::INITIAL,
        )
        .expect("genesis"),
        ROOT,
    )
}

/// Root split into `LEFT` and `RIGHT`, parent sealed and retired - the shape a
/// namespace has after its first rebalance.
#[must_use]
pub fn split_snapshot() -> DirectorySnapshot {
    let mut dir = root_snapshot();
    for (id, bits) in [(LEFT, 0u128), (RIGHT, 1u128 << 127)] {
        dir = dir
            .allocate(
                id,
                hash_range(bits, 1),
                TabletEpoch::INITIAL,
                WriteGuardGeneration::INITIAL,
            )
            .and_then(|staged| staged.stage(id))
            .expect("child staged");
    }
    // A child cannot go active while the parent still covers its prefix, so
    // the parent seals first.
    dir = dir.seal(ROOT).expect("seal root");
    dir = dir.activate(LEFT).expect("activate left");
    dir = dir.activate(RIGHT).expect("activate right");
    dir.retire(ROOT, kivi_tablet::Redirect::new(vec![LEFT, RIGHT]))
        .expect("retire root")
}

/// One active root tablet, all of it on worker 0.
#[must_use]
pub fn root_placement() -> Placement {
    Placement::new([(ROOT, worker(0))])
}

/// `LEFT` on worker 0, `RIGHT` on worker 1.
#[must_use]
pub fn split_placement() -> Placement {
    Placement::new([(LEFT, worker(0)), (RIGHT, worker(1))])
}

#[must_use]
pub fn worker(index: u64) -> WorkerId {
    WorkerId::from_u64(index)
}

/// An engine config over `directory`/`placement`, ephemeral and
/// channel-only. Callers set `chunks`, `fabric`, `durability`, and
/// `worker_count` when they need something other than the defaults.
#[must_use]
pub fn config(
    directory: DirectorySnapshot,
    placement: Placement,
    worker_count: usize,
) -> EngineConfig {
    EngineConfig {
        namespace: NS,
        hardware: HardwareConfig::default(),
        directory,
        placement,
        worker_count,
        request_capacity: 64,
        chunks: ChunkFabricConfig::default(),
        fabric: FabricConfig::default(),
        network: None,
        durability: DurabilityMode::Ephemeral,
    }
}

/// A single-tablet ephemeral engine, optionally with a custom fabric policy.
#[must_use]
pub fn start_single(fabric: FabricConfig) -> LocalEngine {
    let mut config = config(root_snapshot(), root_placement(), 1);
    config.fabric = fabric;
    LocalEngine::start(config).expect("engine starts")
}

/// A single-tablet durable engine over `dir`, with automatic checkpoints
/// stilled so tests can trigger cuts themselves.
#[must_use]
pub fn start_durable(
    dir: &std::path::Path,
    chunks: ChunkFabricConfig,
    fabric: FabricConfig,
) -> LocalEngine {
    LocalEngine::start(durable_config(dir, chunks, fabric)).expect("durable engine starts")
}

/// A durable engine config over the single root tablet, automatic
/// checkpoints stilled.
#[must_use]
pub fn durable_config(
    dir: &std::path::Path,
    chunks: ChunkFabricConfig,
    fabric: FabricConfig,
) -> EngineConfig {
    let mut config = config(root_snapshot(), root_placement(), 2);
    config.chunks = chunks;
    config.fabric = fabric;
    config.request_capacity = 256;
    let opened = kivi_durability::open_data_dir(dir, None).expect("data dir opens");
    config.durability = DurabilityMode::Durable(DurableConfig {
        data_dir: dir.to_owned(),
        segment_target_bytes: 1024 * 1024,
        node: opened.meta.node,
        cluster: opened.meta.cluster,
        incarnation: opened.meta.incarnation,
        shared_wal: false,
        batch: kivi_engine::BatchPolicy::default_policy(),
        checkpoint: kivi_engine::CheckpointConfig {
            disabled: true,
            ..kivi_engine::CheckpointConfig::default_config()
        },
    });
    config
}

/// A durable engine config over the two-way split, one tablet per worker.
#[must_use]
pub fn durable_split_config(dir: &std::path::Path, fabric: FabricConfig) -> EngineConfig {
    let mut config = durable_config(dir, ChunkFabricConfig::default(), fabric);
    config.directory = split_snapshot();
    config.placement = split_placement();
    config
}

/// Deterministic pseudo-random bytes (splitmix64): representative,
/// incompressible-ish content - never zeros-only.
#[must_use]
pub fn pattern_bytes(len: usize, seed: u64) -> bytes::Bytes {
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
    bytes::Bytes::from(out)
}

/// Waits for an installed CURRENT at `cut` (background build plus publish are
/// asynchronous) and returns it.
#[must_use]
pub fn await_current(dir: &std::path::Path, cut: u64) -> kivi_checkpoint::CurrentRecord {
    use std::time::{Duration, Instant};
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Ok(Some(current)) = kivi_checkpoint::read_current(dir, ROOT)
            && current.cut >= cut
        {
            return current;
        }
        assert!(
            Instant::now() <= deadline,
            "no CURRENT at cut {cut} after 30s"
        );
        std::thread::sleep(Duration::from_millis(25));
    }
}
