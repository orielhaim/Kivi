//! Chunked large values through the public `LocalEngine` API: representation
//! transparency, threshold conversion, durable restart recovery, and loud
//! dependency failure.
//!
//! Channel-only engines (plain worker threads, no sockets): the embedded
//! path exercises the same tablet/commit/chunk-lane flow as production,
//! with blocking lane calls on worker threads that already block. Native
//! streaming-protocol coverage lives in `kivi-server` process tests.

use kivi_engine::{
    ChunkFabricConfig, DurabilityMode, DurableConfig, EngineConfig, EngineError, FabricConfig,
    LocalEngine, Placement,
};
use kivi_state::Key;
use kivi_tablet::{DirectorySnapshot, HashPrefix, PartitionRange};
use kivi_types::{NamespaceId, TabletEpoch, TabletId, WorkerId, WriteGuardGeneration};

const NS: NamespaceId = NamespaceId::from_u64(1);

fn directory() -> DirectorySnapshot {
    let genesis = DirectorySnapshot::bootstrap(
        NS,
        TabletId::from_u64(1),
        PartitionRange::Hash(HashPrefix::new(0, 0).expect("root")),
        TabletEpoch::INITIAL,
        WriteGuardGeneration::INITIAL,
    )
    .expect("genesis");
    genesis
        .stage(TabletId::from_u64(1))
        .and_then(|snapshot| snapshot.activate(TabletId::from_u64(1)))
        .expect("active root")
}

fn placement() -> Placement {
    Placement::new([(TabletId::from_u64(1), WorkerId::from_u64(0))])
}

/// Deterministic pseudo-random bytes (splitmix64): representative,
/// incompressible-ish content — never zeros-only.
fn pattern_bytes(len: usize, seed: u64) -> bytes::Bytes {
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

fn ephemeral(chunks: ChunkFabricConfig) -> LocalEngine {
    LocalEngine::start(EngineConfig {
        namespace: NS,
        directory: directory(),
        placement: placement(),
        worker_count: 2,
        request_capacity: 64,
        chunks,
        fabric: kivi_engine::FabricConfig::default(),
        network: None,
        durability: DurabilityMode::Ephemeral,
    })
    .expect("ephemeral engine starts")
}

fn durable_config(dir: &std::path::Path) -> DurabilityMode {
    let opened = kivi_durability::open_data_dir(dir).expect("data dir opens");
    DurabilityMode::Durable(DurableConfig {
        data_dir: dir.to_owned(),
        segment_target_bytes: 1024 * 1024,
        node: opened.meta.node,
        cluster: opened.meta.cluster,
        incarnation: opened.meta.incarnation,
        shared_wal: false,
        batch: kivi_engine::BatchPolicy::default_policy(),
        checkpoint: kivi_engine::CheckpointConfig::default_config(),
    })
}

fn durable(dir: &std::path::Path, chunks: ChunkFabricConfig) -> LocalEngine {
    LocalEngine::start(EngineConfig {
        namespace: NS,
        directory: directory(),
        placement: placement(),
        worker_count: 2,
        request_capacity: 64,
        chunks,
        fabric: kivi_engine::FabricConfig::default(),
        network: None,
        durability: durable_config(dir),
    })
    .expect("durable engine starts")
}

/// Test offsets as indices (test values always fit the address space).
fn idx(offset: u64) -> usize {
    usize::try_from(offset).expect("test offsets fit")
}

/// Total chunk records indexed across lanes (proof that staging happened).
fn total_chunks(engine: &LocalEngine) -> u64 {
    engine
        .admin_handle()
        .chunk_stats_snapshot()
        .into_iter()
        .map(|(_, stats)| stats.map_or(0, |stats| stats.store.chunks))
        .sum()
}

#[test]
fn large_values_read_identically_whatever_the_representation() {
    let engine = ephemeral(ChunkFabricConfig::default());
    let client = engine.client();
    // Small values stay simple.
    client
        .set(&Key::from("small"), bytes::Bytes::from_static(b"v"))
        .expect("set");
    assert_eq!(total_chunks(&engine), 0, "small sets stage nothing");
    // A 300 KiB value crosses the default 256 KiB threshold: one chunk.
    let big = pattern_bytes(300_000, 0xB16);
    client.set(&Key::from("big"), big.clone()).expect("set");
    assert_eq!(
        client.get(&Key::from("big")).expect("get"),
        Some(big.clone()),
        "chunked bytes read back exactly"
    );
    assert_eq!(total_chunks(&engine), 1, "one chunk staged");
    // Overwrites both directions preserve logical semantics.
    client
        .set(&Key::from("big"), bytes::Bytes::from_static(b"tiny"))
        .expect("shrink");
    assert_eq!(
        client.get(&Key::from("big")).expect("get"),
        Some(bytes::Bytes::from_static(b"tiny"))
    );
    client.set(&Key::from("big"), big.clone()).expect("regrow");
    assert_eq!(client.get(&Key::from("big")).expect("get"), Some(big));
    // Typed operations treat chunked bytes as bytes.
    assert!(client.counter_add(&Key::from("big"), 1).is_err());
    assert!(matches!(
        client.counter_add(&Key::from("big"), 1),
        Err(EngineError::Tablet(_))
    ));
    assert!(client.exists(&Key::from("big")).expect("exists"));
    assert!(
        client
            .expire_at(
                &Key::from("big"),
                // Far-future stamp: expiry attaches without expiring the key
                // under the real wall clock this test runs against.
                kivi_types::UnixMicros::from_micros(9_999_999_999_999_999)
            )
            .expect("expire")
    );
    // Deletion removes the root immediately; chunks go quiet later.
    assert!(client.delete(&Key::from("big")).expect("delete"));
    assert_eq!(client.get(&Key::from("big")).expect("get"), None);
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn representation_boundary_follows_the_threshold() {
    let chunks = ChunkFabricConfig {
        inline_threshold: 1024,
        ..ChunkFabricConfig::default()
    };
    let engine = ephemeral(chunks);
    let client = engine.client();
    client
        .set(&Key::from("at"), pattern_bytes(1024, 1))
        .expect("set");
    assert_eq!(total_chunks(&engine), 0, "threshold value stays inline");
    client
        .set(&Key::from("over"), pattern_bytes(1025, 2))
        .expect("set");
    assert_eq!(total_chunks(&engine), 1, "threshold+1 chunks");
    assert_eq!(
        client.get(&Key::from("over")).expect("get"),
        Some(pattern_bytes(1025, 2))
    );
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn chunked_durable_restart_recovers_bytes_exactly() {
    let scratch = tempfile::tempdir().expect("scratch");
    let engine = durable(scratch.path(), ChunkFabricConfig::default());
    let client = engine.client();
    // Multi-chunk value: 2.5 MiB at 1 MiB chunks is 1 MiB + 1 MiB + rest.
    let huge = pattern_bytes(2_500_000, 0x4400);
    let big = pattern_bytes(300_000, 0xB16);
    client.set(&Key::from("huge"), huge.clone()).expect("set");
    client.set(&Key::from("big"), big.clone()).expect("set");
    client
        .set(&Key::from("small"), bytes::Bytes::from_static(b"s"))
        .expect("set");
    assert_eq!(client.counter_add(&Key::from("n"), 7).expect("add"), 7);
    engine.shutdown().expect("clean shutdown");
    // Restart on the same directory: every acknowledged root — inline or
    // chunked — comes back byte-exact before serving.
    let engine = durable(scratch.path(), ChunkFabricConfig::default());
    let client = engine.client();
    assert_eq!(client.get(&Key::from("huge")).expect("get"), Some(huge));
    assert_eq!(client.get(&Key::from("big")).expect("get"), Some(big));
    assert_eq!(
        client.get(&Key::from("small")).expect("get"),
        Some(bytes::Bytes::from_static(b"s"))
    );
    assert_eq!(client.counter_get(&Key::from("n")).expect("read"), Some(7));
    // The log continues past the replay.
    client
        .set(&Key::from("after"), bytes::Bytes::from_static(b"yes"))
        .expect("set");
    assert_eq!(
        client.get(&Key::from("after")).expect("get"),
        Some(bytes::Bytes::from_static(b"yes"))
    );
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn missing_chunk_packs_fail_startup_loudly() {
    let scratch = tempfile::tempdir().expect("scratch");
    let engine = durable(scratch.path(), ChunkFabricConfig::default());
    engine
        .client()
        .set(&Key::from("big"), pattern_bytes(300_000, 9))
        .expect("set");
    engine.shutdown().expect("clean shutdown");
    // Destroy the bulk information while keeping the WAL: the committed
    // root names chunks that no longer exist. Restart must refuse to
    // serve, never fabricate absence.
    std::fs::remove_dir_all(scratch.path().join("chunks")).expect("chunks destroyed");
    let error = LocalEngine::start(EngineConfig {
        namespace: NS,
        directory: directory(),
        placement: placement(),
        worker_count: 2,
        request_capacity: 64,
        chunks: ChunkFabricConfig::default(),
        fabric: FabricConfig::default(),
        network: None,
        durability: durable_config(scratch.path()),
    })
    .expect_err("missing chunks fail startup");
    let message = format!("{error:?}");
    assert!(
        message.contains("MissingManifest") || message.contains("MissingChunk"),
        "loud chunk dependency failure, got {message}"
    );
}

#[test]
fn set_range_patches_inline_values_with_set_semantics() {
    let engine = ephemeral(ChunkFabricConfig::default());
    let client = engine.client();
    // Absent state reads as empty: a patch at zero creates the value.
    client
        .set_range(&Key::from("k"), 0, bytes::Bytes::from_static(b"hello"))
        .expect("create");
    assert_eq!(
        client.get(&Key::from("k")).expect("get"),
        Some(bytes::Bytes::from_static(b"hello"))
    );
    // In-place overwrite keeps prefix and suffix.
    client
        .set_range(&Key::from("k"), 1, bytes::Bytes::from_static(b"ELL"))
        .expect("overwrite");
    assert_eq!(
        client.get(&Key::from("k")).expect("get"),
        Some(bytes::Bytes::from_static(b"hELLo"))
    );
    // Past-the-end gaps zero-pad; nothing stages for small results.
    client
        .set_range(&Key::from("k"), 8, bytes::Bytes::from_static(b"!"))
        .expect("gap");
    assert_eq!(
        client.get(&Key::from("k")).expect("get"),
        Some(bytes::Bytes::from_static(b"hELLo\0\0\0!"))
    );
    assert_eq!(total_chunks(&engine), 0, "small splices stage nothing");
    // Counters reject range patches with wrong-type, state untouched.
    client.counter_add(&Key::from("n"), 4).expect("counter");
    assert!(matches!(
        client.set_range(&Key::from("n"), 0, bytes::Bytes::from_static(b"x")),
        Err(EngineError::Tablet(_))
    ));
    assert_eq!(client.counter_get(&Key::from("n")).expect("read"), Some(4));
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn set_range_restages_chunked_bases_reusing_prefix_chunks() {
    let engine = ephemeral(ChunkFabricConfig::default());
    let client = engine.client();
    // Three chunks at 1 MiB: patching the last chunk must retain the
    // first two by reference — one new chunk, never a rewrite.
    let base = pattern_bytes(3_000_000, 0x5EED);
    client.set(&Key::from("big"), base.clone()).expect("set");
    assert_eq!(total_chunks(&engine), 3, "three chunks staged");
    let patch = pattern_bytes(1000, 0x9A7C);
    let at = 2_900_000u64;
    client
        .set_range(&Key::from("big"), at, patch.clone())
        .expect("patch tail");
    let mut expected = base.to_vec();
    expected[idx(at)..idx(at) + patch.len()].copy_from_slice(&patch);
    assert_eq!(
        client.get(&Key::from("big")).expect("get"),
        Some(bytes::Bytes::from(expected)),
        "patched bytes read back exactly"
    );
    assert_eq!(
        total_chunks(&engine),
        4,
        "prefix chunks reused: only the patched tail stages anew"
    );
    // A patch straddling the first boundary re-chunks from that boundary:
    // the first chunk survives, the rest streams through the new grid.
    let patch = pattern_bytes(100, 0xBEEF);
    let at = 1_048_500u64;
    client
        .set_range(&Key::from("big"), at, patch.clone())
        .expect("patch straddle");
    let expected = client
        .get(&Key::from("big"))
        .expect("get")
        .expect("present");
    assert_eq!(&expected[idx(at)..idx(at) + patch.len()], &patch);
    assert_eq!(expected.len(), 3_000_000, "length preserved");
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn set_range_growing_past_the_threshold_restages() {
    let chunks = ChunkFabricConfig {
        inline_threshold: 1024,
        ..ChunkFabricConfig::default()
    };
    let engine = ephemeral(chunks);
    let client = engine.client();
    client
        .set(&Key::from("k"), pattern_bytes(1000, 3))
        .expect("set");
    assert_eq!(total_chunks(&engine), 0, "base stays inline");
    // The patched result (1100 bytes) crosses the 1 KiB threshold: the
    // range restages as a full value through the ordinary large-`Set`
    // path instead of WALing an oversize splice.
    let patch = pattern_bytes(200, 4);
    client
        .set_range(&Key::from("k"), 900, patch.clone())
        .expect("grow");
    // The patch runs past the old end: head plus patch, length 1100.
    let mut expected = pattern_bytes(1000, 3).to_vec();
    expected.truncate(900);
    expected.extend_from_slice(&patch);
    assert_eq!(
        client.get(&Key::from("k")).expect("get"),
        Some(bytes::Bytes::from(expected))
    );
    assert_eq!(total_chunks(&engine), 1, "grown result stages one chunk");
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn set_range_rejects_absurd_ranges_with_state_untouched() {
    let engine = ephemeral(ChunkFabricConfig::default());
    let client = engine.client();
    assert!(matches!(
        client.set_range(&Key::from("k"), u64::MAX, bytes::Bytes::from_static(b"x")),
        Err(EngineError::InvalidRequest { .. })
    ));
    // A result past the 64 MiB range bound rejects before allocating any
    // zero pad: rewrite through a streaming upload instead.
    assert!(matches!(
        client.set_range(&Key::from("k"), 64 * 1024 * 1024 + 1, bytes::Bytes::new()),
        Err(EngineError::InvalidRequest { .. })
    ));
    assert_eq!(client.get(&Key::from("k")).expect("get"), None);
    assert_eq!(total_chunks(&engine), 0, "rejections stage nothing");
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn set_range_chunked_patch_survives_durable_restart() {
    let scratch = tempfile::tempdir().expect("scratch");
    let engine = durable(scratch.path(), ChunkFabricConfig::default());
    let client = engine.client();
    let base = pattern_bytes(2_500_000, 0xD0AB);
    client.set(&Key::from("huge"), base.clone()).expect("set");
    let patch = pattern_bytes(5000, 0xF00D);
    let at = 2_000_000u64;
    client
        .set_range(&Key::from("huge"), at, patch.clone())
        .expect("patch");
    let mut expected = base.to_vec();
    expected[idx(at)..idx(at) + patch.len()].copy_from_slice(&patch);
    let expected = bytes::Bytes::from(expected);
    assert_eq!(
        client.get(&Key::from("huge")).expect("get"),
        Some(expected.clone())
    );
    engine.shutdown().expect("clean shutdown");
    // The spliced root is a small WAL record over durable packs: restart
    // recovers the patched bytes exactly.
    let engine = durable(scratch.path(), ChunkFabricConfig::default());
    assert_eq!(
        engine.client().get(&Key::from("huge")).expect("get"),
        Some(expected)
    );
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn chunked_values_checkpoint_and_restore() {
    let scratch = tempfile::tempdir().expect("scratch");
    let engine = durable(scratch.path(), ChunkFabricConfig::default());
    let client = engine.client();
    let big = pattern_bytes(300_000, 77);
    client.set(&Key::from("big"), big.clone()).expect("set");
    engine.request_checkpoint(None);
    // Wait for the checkpoint to install (background worker, bounded).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let installed = loop {
        let state = engine.admin_handle().checkpoint_state();
        if state.checkpoints_completed >= 1 {
            break true;
        }
        if std::time::Instant::now() > deadline {
            break false;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    };
    assert!(installed, "checkpoint installs");
    engine.shutdown().expect("clean shutdown");
    // Checkpoint restore plus empty WAL tail: bytes come back from the
    // checkpoint's small roots plus the surviving packs.
    let engine = durable(scratch.path(), ChunkFabricConfig::default());
    assert_eq!(
        engine.client().get(&Key::from("big")).expect("get"),
        Some(big)
    );
    engine.shutdown().expect("clean shutdown");
}
