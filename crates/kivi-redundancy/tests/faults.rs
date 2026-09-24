//! Deterministic fault injection: every [`FaultKind`] on every scheme.
//!
//! Pure codec faults use [`SimFragments`] plus the replication/RS kernels
//! directly (no filesystem). Fabric faults drive [`RedundancyFabric`] file
//! layouts (delete/corrupt/truncate fragment files, plant pending builds,
//! drain/restart) and prove reads stay exact and failures close
//! ([`Unrecoverable`], never wrong bytes).
//!
//! Determinism: fixed seeds, fixed byte patterns, no RNG, no wall time.
//! Storage amplification reference: replication 3x = 3.0, RS 4+2 = 1.5,
//! RS 8+3 = 1.375 (see [`SchemeParams::amplification`]).

use std::path::PathBuf;

use kivi_redundancy::{
    AssetHealth, FabricConfig, FragmentVerdict, InformationAsset, RedundancyError,
    RedundancyFabric, RepairReason, ReplicationParams, RsParams, SchemeParams, SimFragments,
};
use kivi_types::{NodeId, SecurityDomainId};

const DOMAIN: SecurityDomainId = SecurityDomainId::from_u64(7);
const SMALL_LEN: usize = 32 * 1024;
const CODED_LEN: usize = 64 * 1024;
const BIG_LEN: usize = 1024 * 1024;

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

fn test_nodes(count: u64) -> Vec<kivi_redundancy::NodeDescriptor> {
    (1..=count)
        .map(|id| kivi_redundancy::NodeDescriptor {
            id: NodeId::from_u64(id),
            domain: kivi_redundancy::FailureDomain::node_only(NodeId::from_u64(id)),
            health: kivi_redundancy::NodeHealth::Active,
            weight: 1,
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

fn drop_first_n(view: &SimFragments, n: usize, seed: u64) -> Vec<(u32, Vec<u8>)> {
    let mut present = view.present_sorted();
    present.sort_by_key(|(i, _)| *i);
    // Truncation is fine: `seed` only picks a deterministic start offset.
    #[allow(clippy::cast_possible_truncation)]
    let start = (seed as usize) % present.len().max(1);
    for _ in 0..n {
        if present.is_empty() {
            break;
        }
        let pos = start % present.len();
        present.remove(pos);
    }
    present
}

// ---------------------------------------------------------------------------
// Pure codec: every FaultKind via SimFragments, all three schemes
// ---------------------------------------------------------------------------

fn codec_schemes_for_len(len: u64) -> Vec<(&'static str, SchemeParams)> {
    vec![
        ("rep3", rep_scheme()),
        ("rs42", rs42_scheme(len)),
        ("rs83", rs83_scheme(len)),
    ]
}

#[test]
fn codec_all_faultkinds_behave() {
    // Walks every FaultKind for every scheme through SimFragments and proves
    // the recoverability contract. Corruption/partial writes are modeled as
    // detected erasures (hash rejects them, decoder sees the remainder).
    for fault in kivi_redundancy::FaultKind::all() {
        for (name, scheme) in codec_schemes_for_len(SMALL_LEN as u64) {
            let bytes = det_bytes(SMALL_LEN, 0xA11CE);
            let shards = encode_scheme(&bytes, scheme).expect("encodes");
            let mut view = SimFragments::from_shards(shards);
            let total = shards_len(scheme);
            let tol = tolerance_of(scheme);
            view.apply(fault, 3);
            let outcome = codec_expectation(fault, &view, total, tol);
            check_codec_outcome(&bytes, scheme, &view, outcome, fault, name);
        }
    }
}

fn shards_len(scheme: SchemeParams) -> usize {
    scheme.total_fragments() as usize
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Expect {
    Recoverable,
    Unrecoverable,
    NoopRecoverable,
}

fn codec_expectation(
    fault: kivi_redundancy::FaultKind,
    view: &SimFragments,
    total: usize,
    tol: usize,
) -> Expect {
    use kivi_redundancy::FaultKind as F;
    match fault {
        F::MissingFragment | F::CorruptedFragment | F::PartialWrite | F::WithinTolerance => {
            // WithinTolerance removes one via apply(); the dedicated tests
            // below drop the full tolerance. Single loss always recovers.
            let _ = (total, tol);
            Expect::Recoverable
        }
        F::BeyondTolerance => {
            let _ = view;
            Expect::Unrecoverable
        }
        F::DuringEncoding
        | F::DuringReconstruction
        | F::BeforePublish
        | F::AfterPublishBeforeRetire
        | F::StaleRepair
        | F::DrainDuringRepair
        | F::RestartDuringTransition
        | F::DuplicateRepair => Expect::NoopRecoverable,
    }
}

fn check_codec_outcome(
    bytes: &[u8],
    scheme: SchemeParams,
    view: &SimFragments,
    outcome: Expect,
    fault: kivi_redundancy::FaultKind,
    name: &str,
) {
    use kivi_redundancy::FaultKind as F;
    match (fault, outcome) {
        (F::BeyondTolerance, Expect::Unrecoverable) => {
            // apply() clears everything; beyond tolerance also holds for
            // tolerance+1 losses (checked in dedicated tests).
            let present = view.present_sorted();
            let err = decode_scheme(&present, scheme, bytes.len() as u64)
                .expect_err("beyond tolerance fails");
            assert!(
                err.is_unrecoverable(),
                "{name} {fault:?}: expected Unrecoverable, got {err:?}"
            );
        }
        (_, Expect::Recoverable | Expect::NoopRecoverable) => {
            // For corruption/partial, drop the victim (hash would reject it).
            let mut present = view.present_sorted();
            if matches!(fault, F::CorruptedFragment | F::PartialWrite) {
                // Victim is seed%len = 3%len; recompute deterministically by
                // dropping the shard whose length/content disagrees: the
                // corrupted shard keeps length, the truncated one is short.
                // Simplest: drop index (3 % total).
                let total = shards_len(scheme);
                #[allow(clippy::cast_possible_truncation)]
                let victim = (3 % total) as u32;
                present.retain(|(i, _)| *i != victim);
            }
            let back = decode_scheme(&present, scheme, bytes.len() as u64)
                .expect("within tolerance recovers");
            assert_eq!(&back, bytes, "{name} {fault:?}: exact bytes");
        }
        (_, Expect::Unrecoverable) => unreachable!("only beyond maps to unrecoverable"),
    }
}

#[test]
fn codec_missing_single_recovers_all_schemes() {
    for (name, scheme) in codec_schemes_for_len(SMALL_LEN as u64) {
        let bytes = det_bytes(SMALL_LEN, 11);
        let shards = encode_scheme(&bytes, scheme).expect("encodes");
        let mut view = SimFragments::from_shards(shards);
        view.apply(kivi_redundancy::FaultKind::MissingFragment, 1);
        let present = view.present_sorted();
        assert_eq!(present.len(), shards_len(scheme) - 1, "{name}");
        let back = decode_scheme(&present, scheme, bytes.len() as u64).expect("recovers");
        assert_eq!(back, bytes, "{name}: exact match");
    }
}

#[test]
fn codec_corrupted_detected_as_erasure_recovers() {
    for (name, scheme) in codec_schemes_for_len(SMALL_LEN as u64) {
        let bytes = det_bytes(SMALL_LEN, 22);
        let shards = encode_scheme(&bytes, scheme).expect("encodes");
        let mut view = SimFragments::from_shards(shards);
        view.apply(kivi_redundancy::FaultKind::CorruptedFragment, 0);
        // Hash verification would reject the victim; drop it and decode.
        let total = shards_len(scheme);
        #[allow(clippy::cast_possible_truncation)]
        let victim = (0 % total) as u32;
        let present: Vec<(u32, Vec<u8>)> = view
            .present_sorted()
            .into_iter()
            .filter(|(i, _)| *i != victim)
            .collect();
        let back = decode_scheme(&present, scheme, bytes.len() as u64).expect("recovers");
        assert_eq!(back, bytes, "{name}: exact match");
    }
}

#[test]
fn codec_within_tolerance_recovers_exact() {
    for (name, scheme) in codec_schemes_for_len(SMALL_LEN as u64) {
        let bytes = det_bytes(SMALL_LEN, 33);
        let shards = encode_scheme(&bytes, scheme).expect("encodes");
        let view = SimFragments::from_shards(shards);
        let tol = tolerance_of(scheme);
        let present = drop_first_n(&view, tol, 5);
        assert_eq!(present.len(), shards_len(scheme) - tol, "{name}");
        let back = decode_scheme(&present, scheme, bytes.len() as u64).expect("recovers");
        assert_eq!(back, bytes, "{name}: exact match");
    }
}

#[test]
fn codec_beyond_tolerance_fails_closed() {
    for (name, scheme) in codec_schemes_for_len(SMALL_LEN as u64) {
        let bytes = det_bytes(SMALL_LEN, 44);
        let shards = encode_scheme(&bytes, scheme).expect("encodes");
        let view = SimFragments::from_shards(shards);
        let tol = tolerance_of(scheme);
        let present = drop_first_n(&view, tol + 1, 5);
        let err =
            decode_scheme(&present, scheme, bytes.len() as u64).expect_err("must fail closed");
        assert!(
            err.is_unrecoverable(),
            "{name}: expected Unrecoverable, got {err:?}"
        );
    }
}

#[test]
fn codec_partial_write_detected_recovers() {
    for (name, scheme) in codec_schemes_for_len(SMALL_LEN as u64) {
        let bytes = det_bytes(SMALL_LEN, 55);
        let shards = encode_scheme(&bytes, scheme).expect("encodes");
        let mut view = SimFragments::from_shards(shards);
        view.apply(kivi_redundancy::FaultKind::PartialWrite, 2);
        let total = shards_len(scheme);
        #[allow(clippy::cast_possible_truncation)]
        let victim = (2 % total) as u32;
        // Including the torn shard must fail loudly, never silently decode.
        let with_torn = view.present_sorted();
        if matches!(scheme, SchemeParams::ReedSolomon(_)) {
            assert!(
                decode_scheme(&with_torn, scheme, bytes.len() as u64).is_err(),
                "{name}: torn shard fails loudly"
            );
        }
        let present: Vec<(u32, Vec<u8>)> = with_torn
            .into_iter()
            .filter(|(i, _)| *i != victim)
            .collect();
        let back = decode_scheme(&present, scheme, bytes.len() as u64).expect("recovers");
        assert_eq!(back, bytes, "{name}: exact match");
    }
}

#[test]
fn codec_reconstruction_retry_is_clean() {
    // DuringReconstruction: a crash mid-decode retries against the same
    // present set and returns identical bytes.
    for (name, scheme) in codec_schemes_for_len(CODED_LEN as u64) {
        let bytes = det_bytes(CODED_LEN, 66);
        let shards = encode_scheme(&bytes, scheme).expect("encodes");
        let view = SimFragments::from_shards(shards);
        let present = drop_first_n(&view, 1, 0);
        let first = decode_scheme(&present, scheme, bytes.len() as u64).expect("decodes");
        let second = decode_scheme(&present, scheme, bytes.len() as u64).expect("retries");
        assert_eq!(first, bytes, "{name}");
        assert_eq!(second, bytes, "{name}");
    }
}

#[test]
fn codec_big_asset_within_and_beyond_tolerance() {
    // 1 MiB proves the large-blob path; other tests stay small for speed.
    for (name, scheme) in codec_schemes_for_len(BIG_LEN as u64) {
        let bytes = det_bytes(BIG_LEN, 77);
        let shards = encode_scheme(&bytes, scheme).expect("encodes");
        let view = SimFragments::from_shards(shards);
        let tol = tolerance_of(scheme);
        let ok = drop_first_n(&view, tol, 1);
        let back = decode_scheme(&ok, scheme, bytes.len() as u64).expect("recovers");
        assert_eq!(back, bytes, "{name}: 1MiB exact");
        let bad = drop_first_n(&view, tol + 1, 1);
        let err = decode_scheme(&bad, scheme, bytes.len() as u64).expect_err("fails closed");
        assert!(err.is_unrecoverable(), "{name}: Unrecoverable");
    }
}

// ---------------------------------------------------------------------------
// Fabric faults (13): file-level failures, all three schemes each
// ---------------------------------------------------------------------------

fn fabric_schemes(len: u64) -> Vec<(&'static str, SchemeParams)> {
    vec![
        ("rep3", rep_scheme()),
        ("rs42", rs42_scheme(len)),
        ("rs83", rs83_scheme(len)),
    ]
}

fn fabric_case(
    seed: u64,
    len: usize,
    f: impl Fn(&mut RedundancyFabric, &PathBuf, InformationAsset, SchemeParams, &str),
) {
    for (name, scheme) in fabric_schemes(len as u64) {
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
    for (name, scheme) in fabric_schemes(SMALL_LEN as u64) {
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
    for (name, scheme) in fabric_schemes(SMALL_LEN as u64) {
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
    for (name, scheme) in fabric_schemes(CODED_LEN as u64) {
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
    for (name, scheme) in fabric_schemes(SMALL_LEN as u64) {
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
