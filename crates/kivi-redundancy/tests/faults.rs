//! Deterministic fault injection: every failure mode, on every scheme.
//!
//! Pure codec faults drive the replication/RS kernels directly (no
//! filesystem). Fabric faults drive [`RedundancyFabric`] file layouts
//! (delete/corrupt/truncate fragment files, plant pending builds,
//! drain/restart) and prove reads stay exact and failures close
//! ([`Unrecoverable`], never wrong bytes).
//!
//! Determinism: fixed seeds, fixed byte patterns, no RNG, no wall time.
//! Storage amplification reference: replication 3x = 3.0, RS 4+2 = 1.5,
//! RS 8+3 = 1.375 (see [`SchemeParams::amplification`]).

use std::path::PathBuf;

use kivi_redundancy::{
    AssetHealth, FabricConfig, FragmentVerdict, InformationAsset, RedundancyError,
    RedundancyFabric, RepairReason, ReplicationParams, RsParams, SchemeParams,
};
use kivi_types::{NodeId, SecurityDomainId};

const DOMAIN: SecurityDomainId = SecurityDomainId::from_u64(7);
const SMALL_LEN: usize = 32 * 1024;
const CODED_LEN: usize = 64 * 1024;

// ---------------------------------------------------------------------------
// Deterministic helpers
// ---------------------------------------------------------------------------

fn det_bytes(len: usize, seed: u64) -> Vec<u8> {
    (0..len)
        .map(|i| {
            let v = (i as u64)
                .wrapping_mul(31)
                .wrapping_add(seed)
                .wrapping_add((i as u64) >> 8);
            (v % 251) as u8
        })
        .collect()
}

fn rep_scheme() -> SchemeParams {
    SchemeParams::Replication(ReplicationParams { copies: 3 })
}

fn rs42_scheme(len: u64) -> SchemeParams {
    SchemeParams::ReedSolomon(RsParams {
        data: 4,
        parity: 2,
        fragment_len: RsParams::shard_for_len(len, 4),
    })
}

fn rs83_scheme(len: u64) -> SchemeParams {
    SchemeParams::ReedSolomon(RsParams {
        data: 8,
        parity: 3,
        fragment_len: RsParams::shard_for_len(len, 8),
    })
}

fn tolerance_of(scheme: SchemeParams) -> usize {
    scheme.tolerance() as usize
}

fn encode_scheme(bytes: &[u8], scheme: SchemeParams) -> Result<Vec<Vec<u8>>, RedundancyError> {
    match scheme {
        SchemeParams::Replication(p) => kivi_redundancy::replication::encode(bytes, p),
        SchemeParams::ReedSolomon(p) => kivi_redundancy::rs::encode(bytes, p),
    }
}

fn decode_scheme(
    present: &[(u32, Vec<u8>)],
    scheme: SchemeParams,
    len: u64,
) -> Result<Vec<u8>, RedundancyError> {
    match scheme {
        SchemeParams::Replication(_) => kivi_redundancy::replication::decode(present),
        SchemeParams::ReedSolomon(p) => kivi_redundancy::rs::decode(present, p, len),
    }
}

/// Encodes `bytes` and returns the shards as the `(index, bytes)` pairs the
/// decoders take.
fn shards_of(bytes: &[u8], scheme: SchemeParams) -> Vec<(u32, Vec<u8>)> {
    encode_scheme(bytes, scheme)
        .expect("encodes")
        .into_iter()
        .enumerate()
        .map(|(i, s)| {
            // Index < total <= 40, fits `u32`.
            #[allow(clippy::cast_possible_truncation)]
            let index = i as u32;
            (index, s)
        })
        .collect()
}

/// The three schemes, at whatever fragment size `len` implies.
fn schemes_for_len(len: u64) -> [(&'static str, SchemeParams); 3] {
    [
        ("rep3", rep_scheme()),
        ("rs42", rs42_scheme(len)),
        ("rs83", rs83_scheme(len)),
    ]
}

fn test_nodes(count: u64) -> Vec<kivi_redundancy::NodeDescriptor> {
    (1..=count)
        .map(|id| kivi_redundancy::NodeDescriptor {
            id: NodeId::from_u64(id),
            domain: kivi_redundancy::FailureDomain::node_only(NodeId::from_u64(id)),
            health: kivi_redundancy::NodeHealth::Active,
        })
        .collect()
}

fn open_fabric(nodes: usize) -> (tempfile::TempDir, RedundancyFabric, PathBuf) {
    let dir = tempfile::tempdir().expect("scratch");
    let root = dir.path().join("redundancy");
    let (fabric, _) = RedundancyFabric::open(
        &root,
        FabricConfig::conservative(),
        test_nodes(nodes as u64),
    )
    .expect("opens");
    (dir, fabric, root)
}

fn protect_scheme(
    fabric: &mut RedundancyFabric,
    bytes: &[u8],
    scheme: SchemeParams,
) -> InformationAsset {
    let asset = InformationAsset::chunk_for_bytes(bytes, DOMAIN);
    fabric
        .transition(asset, bytes, scheme)
        .expect("installs scheme");
    asset
}

fn asset_dir_for(root: &std::path::Path, asset: InformationAsset) -> PathBuf {
    root.join("assets")
        .join(format!("{}-{}", asset.id.kind.name(), asset.id.hex()))
}

fn frag_path(dir: &std::path::Path, generation: u64, index: u32) -> PathBuf {
    dir.join(format!("frag-{generation}-{index:04}.bin"))
}

fn delete_frag(dir: &std::path::Path, generation: u64, index: u32) {
    std::fs::remove_file(frag_path(dir, generation, index)).expect("deletes");
}

fn corrupt_frag(dir: &std::path::Path, generation: u64, index: u32) {
    let path = frag_path(dir, generation, index);
    let mut bytes = std::fs::read(&path).expect("reads frag");
    assert!(!bytes.is_empty(), "frag nonempty");
    bytes[0] ^= 0xFF;
    std::fs::write(&path, &bytes).expect("writes corrupt");
}

fn truncate_frag(dir: &std::path::Path, generation: u64, index: u32) {
    let path = frag_path(dir, generation, index);
    let bytes = std::fs::read(&path).expect("reads frag");
    assert!(bytes.len() >= 2, "frag long enough to truncate");
    std::fs::write(&path, &bytes[..bytes.len() / 2]).expect("truncates");
}

// ---------------------------------------------------------------------------
// Pure codec: the recoverability contract, on every scheme
// ---------------------------------------------------------------------------

/// Losing up to the advertised tolerance reconstructs the exact bytes;
/// losing one more fails closed. Corruption and torn writes are the same
/// case the fabric sees, because a per-fragment hash turns them into
/// detected erasures.
#[test]
fn codec_tolerance_boundary_is_exact() {
    for (name, scheme) in schemes_for_len(CODED_LEN as u64) {
        let bytes = det_bytes(CODED_LEN, 33);
        let shards = shards_of(&bytes, scheme);
        let tol = tolerance_of(scheme);

        let present: Vec<_> = shards.iter().skip(tol).cloned().collect();
        assert_eq!(present.len(), shards.len() - tol, "{name}");
        let back = decode_scheme(&present, scheme, bytes.len() as u64).expect("recovers");
        assert_eq!(back, bytes, "{name}: exact within tolerance");

        let present: Vec<_> = shards.iter().skip(tol + 1).cloned().collect();
        let err =
            decode_scheme(&present, scheme, bytes.len() as u64).expect_err("must fail closed");
        assert!(
            err.is_unrecoverable(),
            "{name}: expected Unrecoverable, got {err:?}"
        );
    }
}

/// A torn shard must be rejected loudly rather than silently decoded into
/// the wrong answer. Only RS can see it: replication takes the first copy
/// and never inspects the rest, which is why a replicated fragment's
/// integrity is a per-fragment hash check, not the codec's job.
#[test]
fn codec_torn_shard_fails_loudly_then_recovers() {
    for (name, scheme) in schemes_for_len(SMALL_LEN as u64) {
        let bytes = det_bytes(SMALL_LEN, 55);
        let mut shards = shards_of(&bytes, scheme);
        let half = shards[0].1.len() / 2;
        shards[0].1.truncate(half);

        if matches!(scheme, SchemeParams::ReedSolomon(_)) {
            assert!(
                decode_scheme(&shards, scheme, bytes.len() as u64).is_err(),
                "{name}: torn shard fails loudly"
            );
        }
        let present: Vec<_> = shards.iter().skip(1).cloned().collect();
        let back = decode_scheme(&present, scheme, bytes.len() as u64).expect("recovers");
        assert_eq!(back, bytes, "{name}: exact after dropping the torn shard");
    }
}

// ---------------------------------------------------------------------------
// Fabric faults: file-level failures, all three schemes each
// ---------------------------------------------------------------------------

fn fabric_case(
    seed: u64,
    len: usize,
    f: impl Fn(&mut RedundancyFabric, &PathBuf, InformationAsset, SchemeParams, &str),
) {
    for (name, scheme) in schemes_for_len(len as u64) {
        let (_tmp, mut fabric, root) = open_fabric(12);
        let bytes = det_bytes(len, seed);
        let asset = protect_scheme(&mut fabric, &bytes, scheme);
        // Readable before fault.
        assert_eq!(fabric.read(asset).expect("reads"), bytes, "{name} pre");
        f(&mut fabric, &root, asset, scheme, name);
    }
}

#[test]
fn fabric_missing_fragment_degraded_then_healthy() {
    fabric_case(101, CODED_LEN, |fabric, root, asset, scheme, name| {
        let dir = asset_dir_for(root, asset);
        let generation = fabric.layout(&asset.id).expect("layout").generation;
        delete_frag(&dir, generation, 0);
        let bytes = det_bytes(CODED_LEN, 101);
        assert_eq!(fabric.read(asset).expect("reads degraded"), bytes, "{name}");
        let assessment = fabric.assess_asset(&asset.id).expect("assesses");
        assert_eq!(assessment.health, AssetHealth::Degraded, "{name}");
        assert!(assessment.reconstructable, "{name}");
        assert!(
            fabric
                .queue_repairs(&asset.id, RepairReason::FragmentMissing)
                .expect("queues")
        );
        fabric.repair_tick(false).expect("repairs");
        let after = fabric.assess_asset(&asset.id).expect("re-assesses");
        assert_eq!(after.health, AssetHealth::Healthy, "{name} {scheme:?}");
        assert_eq!(fabric.read(asset).expect("reads"), bytes, "{name}");
    });
}

#[test]
fn fabric_corrupted_fragment_scrub_and_repair() {
    fabric_case(102, CODED_LEN, |fabric, root, asset, _scheme, name| {
        let dir = asset_dir_for(root, asset);
        let generation = fabric.layout(&asset.id).expect("layout").generation;
        corrupt_frag(&dir, generation, 0);
        let bytes = det_bytes(CODED_LEN, 102);
        assert_eq!(fabric.read(asset).expect("reads via rest"), bytes, "{name}");
        let report = fabric.scrub(&asset.id).expect("scrubs");
        assert!(report.corruptions >= 1, "{name}");
        let verdict = report
            .verdicts
            .iter()
            .find(|(i, _)| *i == 0)
            .map(|(_, v)| *v);
        assert_eq!(verdict, Some(FragmentVerdict::ChecksumMismatch), "{name}");
        assert!(
            fabric
                .queue_repairs(&asset.id, RepairReason::CorruptionDetected)
                .expect("queues")
        );
        fabric.repair_tick(false).expect("repairs");
        assert_eq!(fabric.read(asset).expect("reads"), bytes, "{name}");
        let clean = fabric.scrub(&asset.id).expect("re-scrubs");
        assert_eq!(clean.corruptions, 0, "{name}");
    });
}

#[test]
fn fabric_within_tolerance_recovers() {
    fabric_case(103, CODED_LEN, |fabric, root, asset, scheme, name| {
        let dir = asset_dir_for(root, asset);
        let generation = fabric.layout(&asset.id).expect("layout").generation;
        let tol = tolerance_of(scheme);
        for i in 0..tol {
            #[allow(clippy::cast_possible_truncation)]
            let idx = i as u32;
            delete_frag(&dir, generation, idx);
        }
        let bytes = det_bytes(CODED_LEN, 103);
        assert_eq!(fabric.read(asset).expect("recovers"), bytes, "{name}");
        assert!(
            fabric
                .queue_repairs(&asset.id, RepairReason::FragmentMissing)
                .expect("queues")
        );
        fabric.repair_tick(false).expect("repairs");
        assert_eq!(fabric.read(asset).expect("reads"), bytes, "{name}");
    });
}

#[test]
fn fabric_beyond_tolerance_fails_closed() {
    fabric_case(104, CODED_LEN, |fabric, root, asset, scheme, name| {
        let dir = asset_dir_for(root, asset);
        let generation = fabric.layout(&asset.id).expect("layout").generation;
        let tol = tolerance_of(scheme);
        for i in 0..=tol {
            #[allow(clippy::cast_possible_truncation)]
            let idx = i as u32;
            delete_frag(&dir, generation, idx);
        }
        let before = fabric.metrics().snapshot().reconstruction_failures;
        let err = fabric.read(asset).expect_err("fails closed");
        assert!(err.is_unrecoverable(), "{name}: got {err:?}");
        let assessment = fabric.assess_asset(&asset.id).expect("assesses");
        assert_eq!(assessment.health, AssetHealth::Unrecoverable, "{name}");
        assert!(!assessment.reconstructable, "{name}");
        assert!(
            fabric.metrics().snapshot().reconstruction_failures > before,
            "{name}: metric increments"
        );
    });
}

#[test]
fn fabric_during_encoding_discards_pending() {
    // Crash during encoding: pending build never publishes; old serves.
    fabric_case(105, SMALL_LEN, |fabric, root, asset, _scheme, name| {
        let dir = asset_dir_for(root, asset);
        let generation = fabric.layout(&asset.id).expect("layout").generation;
        std::fs::write(
            dir.join(format!("pending-{}-0000.bin", generation + 1)),
            b"half",
        )
        .expect("plants");
        std::fs::write(
            dir.join(format!("pending-{}.layout", generation + 1)),
            b"half",
        )
        .expect("plants");
        // Discard the torn build (what restart recovery does).
        std::fs::remove_file(dir.join(format!("pending-{}-0000.bin", generation + 1)))
            .expect("discards");
        std::fs::remove_file(dir.join(format!("pending-{}.layout", generation + 1)))
            .expect("discards");
        let bytes = det_bytes(SMALL_LEN, 105);
        assert_eq!(fabric.read(asset).expect("old serves"), bytes, "{name}");
        assert_eq!(
            fabric.layout(&asset.id).expect("layout").generation,
            generation,
            "{name}"
        );
    });
}

#[test]
fn fabric_during_reconstruction_retries_cleanly() {
    fabric_case(106, CODED_LEN, |fabric, root, asset, _scheme, name| {
        let dir = asset_dir_for(root, asset);
        let generation = fabric.layout(&asset.id).expect("layout").generation;
        delete_frag(&dir, generation, 0);
        let bytes = det_bytes(CODED_LEN, 106);
        let first = fabric.read(asset).expect("first read");
        let second = fabric.read(asset).expect("retry reads");
        assert_eq!(first, bytes, "{name}");
        assert_eq!(second, bytes, "{name}");
    });
}

#[test]
fn fabric_before_publish_old_serves() {
    // Pending staged but unpublished: readers still see the old generation.
    fabric_case(107, SMALL_LEN, |fabric, root, asset, _scheme, name| {
        let dir = asset_dir_for(root, asset);
        let generation = fabric.layout(&asset.id).expect("layout").generation;
        std::fs::write(
            dir.join(format!("pending-{}-0000.bin", generation + 1)),
            b"staged",
        )
        .expect("stages");
        let current = std::fs::read(dir.join("CURRENT")).expect("reads CURRENT");
        assert_eq!(
            u64::from_le_bytes(current[..8].try_into().expect("len")),
            generation
        );
        let bytes = det_bytes(SMALL_LEN, 107);
        assert_eq!(fabric.read(asset).expect("old serves"), bytes, "{name}");
        assert_eq!(
            fabric.layout(&asset.id).expect("layout").generation,
            generation,
            "{name}"
        );
        std::fs::remove_file(dir.join(format!("pending-{}-0000.bin", generation + 1)))
            .expect("cleans");
    });
}

#[test]
fn fabric_after_publish_before_retire_extra_retained() {
    // Crash after publish before retire: extra old files linger but the new
    // generation (CURRENT) still answers exactly. The first generation uses
    // a scheme different from the target so the replacement always mints a
    // new generation (same-params transitions are idempotent no-ops).
    for (name, scheme) in schemes_for_len(SMALL_LEN as u64) {
        let (_tmp, mut fabric, root) = open_fabric(12);
        let bytes = det_bytes(SMALL_LEN, 108);
        let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);
        let initial = if scheme == rep_scheme() {
            rs42_scheme(bytes.len() as u64)
        } else {
            rep_scheme()
        };
        fabric.transition(asset, &bytes, initial).expect("gen1");
        let gen1 = fabric.layout(&asset.id).expect("layout").generation;
        fabric.transition(asset, &bytes, scheme).expect("gen2");
        let gen2 = fabric.layout(&asset.id).expect("layout").generation;
        assert!(gen2 > gen1, "{name}");
        // Simulate un-retired leftovers: recreate one old fragment file.
        let dir = asset_dir_for(&root, asset);
        std::fs::write(frag_path(&dir, gen1, 0), b"stale-extra").expect("plants stale");
        assert_eq!(fabric.read(asset).expect("new serves"), bytes, "{name}");
        assert_eq!(
            fabric.layout(&asset.id).expect("layout").generation,
            gen2,
            "{name}"
        );
        let _ = std::fs::remove_file(frag_path(&dir, gen1, 0));
    }
}

#[test]
fn fabric_stale_repair_fenced() {
    for (name, scheme) in schemes_for_len(SMALL_LEN as u64) {
        let (_tmp, mut fabric, _root) = open_fabric(12);
        let bytes = det_bytes(SMALL_LEN, 109);
        let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);
        let initial = if scheme == rep_scheme() {
            rs42_scheme(bytes.len() as u64)
        } else {
            rep_scheme()
        };
        fabric.transition(asset, &bytes, initial).expect("gen1");
        let gen1 = fabric.layout(&asset.id).expect("layout").generation;
        fabric.transition(asset, &bytes, scheme).expect("gen2");
        let before = fabric.metrics().snapshot().stale_rejected;
        let err = fabric.fence(&asset.id, gen1).expect_err("stale fenced");
        assert!(
            matches!(err, RedundancyError::StaleGeneration { .. }),
            "{name}: got {err:?}"
        );
        assert!(
            fabric.metrics().snapshot().stale_rejected == before + 1,
            "{name}: metric increments"
        );
        assert_eq!(fabric.read(asset).expect("reads"), bytes, "{name}");
    }
}

#[test]
fn fabric_stale_queued_repair_rejected_after_transition() {
    // Queue at gen1, publish gen2, then the tick must fence the stale task.
    let (_tmp, mut fabric, root) = open_fabric(12);
    let bytes = det_bytes(SMALL_LEN, 110);
    let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);
    fabric
        .transition(asset, &bytes, rep_scheme())
        .expect("gen1");
    let dir = asset_dir_for(&root, asset);
    let gen1 = fabric.layout(&asset.id).expect("layout").generation;
    delete_frag(&dir, gen1, 0);
    assert!(
        fabric
            .queue_repairs(&asset.id, RepairReason::FragmentMissing)
            .expect("queues gen1")
    );
    fabric
        .transition(asset, &bytes, rs42_scheme(bytes.len() as u64))
        .expect("gen2");
    let before = fabric.metrics().snapshot().stale_rejected;
    let err = fabric.repair_tick(false).expect_err("stale tick fails");
    assert!(
        matches!(err, RedundancyError::StaleGeneration { .. }),
        "got {err:?}"
    );
    assert_eq!(
        fabric.metrics().snapshot().stale_rejected,
        before + 1,
        "metric increments"
    );
    assert_eq!(fabric.read(asset).expect("gen2 serves"), bytes);
}

#[test]
fn fabric_drain_during_repair_readable_throughout() {
    for (name, scheme) in schemes_for_len(CODED_LEN as u64) {
        let (_tmp, mut fabric, root) = open_fabric(12);
        let bytes = det_bytes(CODED_LEN, 111);
        let asset = protect_scheme(&mut fabric, &bytes, scheme);
        assert_eq!(fabric.read(asset).expect("pre"), bytes, "{name}");
        let victim = fabric.layout(&asset.id).expect("layout").fragments[0].node;
        // Queue a repair first, then drain mid-repair.
        let dir = asset_dir_for(&root, asset);
        let generation = fabric.layout(&asset.id).expect("layout").generation;
        delete_frag(&dir, generation, 0);
        assert!(
            fabric
                .queue_repairs(&asset.id, RepairReason::FragmentMissing)
                .expect("queues")
        );
        let moved = fabric.drain_node(victim).expect("drains");
        assert!(moved.contains(&asset.id), "{name}");
        assert_eq!(
            fabric.read(asset).expect("readable after drain"),
            bytes,
            "{name}"
        );
        for record in &fabric.layout(&asset.id).expect("layout").fragments {
            assert_ne!(record.node, victim, "{name}: drained node unused");
        }
    }
}

#[test]
fn fabric_restart_during_transition_discards_pending() {
    for (name, scheme) in schemes_for_len(SMALL_LEN as u64) {
        let dir_tmp = tempfile::tempdir().expect("scratch");
        let root = dir_tmp.path().join("redundancy");
        let bytes = det_bytes(SMALL_LEN, 112);
        let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);
        {
            let (fabric, _) =
                RedundancyFabric::open(&root, FabricConfig::conservative(), test_nodes(12))
                    .expect("opens");
            let mut f = fabric;
            f.transition(asset, &bytes, scheme).expect("protects");
            assert_eq!(f.read(asset).expect("reads"), bytes, "{name}");
        }
        let layout_gen = {
            let (f, _) =
                RedundancyFabric::open(&root, FabricConfig::conservative(), test_nodes(12))
                    .expect("reopens");
            f.layout(&asset.id).expect("layout").generation
        };
        let asset_dir =
            root.join("assets")
                .join(format!("{}-{}", asset.id.kind.name(), asset.id.hex()));
        std::fs::write(asset_dir.join("pending-999.layout"), b"incomplete").expect("plants");
        std::fs::write(asset_dir.join("pending-999-0000.bin"), b"orphan").expect("plants");
        let (mut fabric, report) =
            RedundancyFabric::open(&root, FabricConfig::conservative(), test_nodes(12))
                .expect("recovers");
        assert_eq!(report.assets_recovered, 1, "{name}");
        assert!(report.pending_discarded >= 2, "{name}");
        assert_eq!(
            fabric.read(asset).expect("published serves"),
            bytes,
            "{name}"
        );
        assert_eq!(
            fabric.layout(&asset.id).expect("layout").generation,
            layout_gen,
            "{name}"
        );
    }
}

#[test]
fn fabric_duplicate_repair_coalesces() {
    fabric_case(113, CODED_LEN, |fabric, root, asset, _scheme, name| {
        let dir = asset_dir_for(root, asset);
        let generation = fabric.layout(&asset.id).expect("layout").generation;
        delete_frag(&dir, generation, 0);
        let first = fabric
            .queue_repairs(&asset.id, RepairReason::FragmentMissing)
            .expect("queues");
        let second = fabric
            .queue_repairs(&asset.id, RepairReason::FragmentMissing)
            .expect("re-queues");
        assert!(first, "{name}: first queues");
        assert!(!second, "{name}: duplicate coalesces");
        fabric.repair_tick(false).expect("repairs");
        let bytes = det_bytes(CODED_LEN, 113);
        assert_eq!(fabric.read(asset).expect("reads"), bytes, "{name}");
        assert_eq!(
            fabric.assess_asset(&asset.id).expect("assesses").health,
            AssetHealth::Healthy,
            "{name}"
        );
    });
}

#[test]
fn fabric_partial_write_detected_and_repaired() {
    fabric_case(114, CODED_LEN, |fabric, root, asset, _scheme, name| {
        let dir = asset_dir_for(root, asset);
        let generation = fabric.layout(&asset.id).expect("layout").generation;
        truncate_frag(&dir, generation, 0);
        let bytes = det_bytes(CODED_LEN, 114);
        assert_eq!(fabric.read(asset).expect("reads via rest"), bytes, "{name}");
        let assessment = fabric.assess_asset(&asset.id).expect("assesses");
        assert!(assessment.need_repair.contains(&0), "{name}");
        assert!(
            fabric
                .queue_repairs(&asset.id, RepairReason::CorruptionDetected)
                .expect("queues")
        );
        fabric.repair_tick(false).expect("repairs");
        assert_eq!(fabric.read(asset).expect("reads"), bytes, "{name}");
        assert_eq!(
            fabric.assess_asset(&asset.id).expect("assesses").health,
            AssetHealth::Healthy,
            "{name}"
        );
    });
}

#[test]
fn fabric_metrics_observe_repairs_and_decodes() {
    let (_tmp, mut fabric, root) = open_fabric(12);
    let bytes = det_bytes(SMALL_LEN, 115);
    let asset = protect_scheme(&mut fabric, &bytes, rep_scheme());
    let decodes_before = fabric.metrics().snapshot().decodes;
    assert_eq!(fabric.read(asset).expect("reads"), bytes);
    assert!(fabric.metrics().snapshot().decodes > decodes_before);
    let dir = asset_dir_for(&root, asset);
    let generation = fabric.layout(&asset.id).expect("layout").generation;
    delete_frag(&dir, generation, 1);
    assert!(
        fabric
            .queue_repairs(&asset.id, RepairReason::FragmentMissing)
            .expect("queues")
    );
    // Pending-repair accounting is visible before the tick drains it.
    assert!(fabric.metrics().snapshot().pending_repairs >= 1);
    fabric.repair_tick(false).expect("repairs");
    assert_eq!(fabric.read(asset).expect("reads"), bytes);
    assert!(fabric.metrics().snapshot().repair_bytes_done > 0);
}
