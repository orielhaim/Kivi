//! Scheme transitions while readable: replicated <-> coded, coded <-> coded.
//!
//! Every transition keeps the logical [`AssetId`] unchanged, serves exact
//! bytes before/during/after (no caller reference rewrite), publishes the
//! target params, retires the old generation (old files gone, `CURRENT`
//! points at the new generation), bumps `transitions` metrics, and moves
//! storage amplification as expected (rep 3x = 3.0, RS 4+2 = 1.5,
//! RS 8+3 = 1.375).

use kivi_redundancy::{
    FabricConfig, InformationAsset, RedundancyFabric, ReplicationParams, RsParams, SchemeParams,
};
use kivi_types::{NodeId, SecurityDomainId};

const DOMAIN: SecurityDomainId = SecurityDomainId::from_u64(7);
const LEN: usize = 256 * 1024;

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

fn nodes(count: u64) -> Vec<kivi_redundancy::NodeDescriptor> {
    (1..=count)
        .map(|id| kivi_redundancy::NodeDescriptor {
            id: NodeId::from_u64(id),
            domain: kivi_redundancy::FailureDomain::node_only(NodeId::from_u64(id)),
            health: kivi_redundancy::NodeHealth::Active,
            weight: 1,
        })
        .collect()
}

fn rep3() -> SchemeParams {
    SchemeParams::Replication(ReplicationParams { copies: 3 })
}

fn rs42(len: u64) -> SchemeParams {
    SchemeParams::ReedSolomon(RsParams {
        data: 4,
        parity: 2,
        fragment_len: RsParams::shard_for_len(len, 4),
    })
}

fn rs83(len: u64) -> SchemeParams {
    SchemeParams::ReedSolomon(RsParams {
        data: 8,
        parity: 3,
        fragment_len: RsParams::shard_for_len(len, 8),
    })
}

fn amplification_f64(params: SchemeParams) -> f64 {
    let (num, den) = params.amplification();
    // Amplification ratios are tiny (<=8); precision loss is irrelevant.
    #[allow(clippy::cast_precision_loss)]
    let out = num as f64 / den as f64;
    out
}

fn asset_dir(root: &std::path::Path, asset: InformationAsset) -> std::path::PathBuf {
    root.join("assets")
        .join(format!("{}-{}", asset.id.kind.name(), asset.id.hex()))
}

fn frag_name(generation: u64, index: u32) -> String {
    format!("frag-{generation}-{index:04}.bin")
}

fn current_gen(dir: &std::path::Path) -> u64 {
    let bytes = std::fs::read(dir.join("CURRENT")).expect("reads CURRENT");
    u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes"))
}

fn old_files_gone(dir: &std::path::Path, old_gen: u64, old_total: u32) {
    assert!(
        !dir.join(format!("gen-{old_gen}.layout")).exists(),
        "old layout retired"
    );
    for i in 0..old_total {
        assert!(
            !dir.join(frag_name(old_gen, i)).exists(),
            "old frag {i} retired"
        );
    }
}

fn check_transition(from: SchemeParams, to: SchemeParams, seed: u64, label: &str) {
    let tmp = tempfile::tempdir().expect("scratch");
    let root = tmp.path().join("redundancy");
    let (mut fabric, _) =
        RedundancyFabric::open(&root, FabricConfig::conservative(), nodes(12)).expect("opens");
    let bytes = det_bytes(LEN, seed);
    let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);

    let first = fabric
        .transition(asset, &bytes, from)
        .expect("installs from");
    assert_eq!(fabric.read(asset).expect("reads before"), bytes, "{label}");
    let asset_id_before = first.asset;
    let gen_before = first.generation;
    let old_total = from.total_fragments();
    let transitions_before = fabric.metrics().snapshot().transitions;

    let second = fabric.transition(asset, &bytes, to).expect("transitions");
    // Logical identity unchanged: no caller reference rewrite.
    assert_eq!(second.asset, asset_id_before, "{label}");
    assert_eq!(asset_id_before, asset.id, "{label}");
    assert!(second.generation > gen_before, "{label}");
    assert_eq!(second.params, to, "{label}");
    assert_eq!(fabric.read(asset).expect("reads after"), bytes, "{label}");

    let dir = asset_dir(&root, asset);
    assert_eq!(current_gen(&dir), second.generation, "{label}");
    old_files_gone(&dir, gen_before, old_total);
    // New fragments all present.
    for i in 0..to.total_fragments() {
        assert!(
            dir.join(frag_name(second.generation, i)).exists(),
            "{label} frag {i}"
        );
    }
    assert!(
        fabric.metrics().snapshot().transitions > transitions_before,
        "{label}: transitions increment"
    );
    assert_eq!(
        fabric.layout(&asset.id).expect("layout").params,
        to,
        "{label}"
    );
}

#[test]
fn replicated_to_coded_stays_readable() {
    check_transition(rep3(), rs42(LEN as u64), 201, "rep->rs42");
    assert!((amplification_f64(rep3()) - 3.0).abs() < f64::EPSILON);
    assert!((amplification_f64(rs42(LEN as u64)) - 1.5).abs() < f64::EPSILON);
}

#[test]
fn coded_to_replicated_stays_readable() {
    check_transition(rs42(LEN as u64), rep3(), 202, "rs42->rep");
}

#[test]
fn coded_width_change_stays_readable() {
    check_transition(rs42(LEN as u64), rs83(LEN as u64), 203, "rs42->rs83");
    assert!((amplification_f64(rs83(LEN as u64)) - 1.375).abs() < f64::EPSILON);
    // And back down.
    check_transition(rs83(LEN as u64), rs42(LEN as u64), 204, "rs83->rs42");
}

#[test]
fn safe_removal_only_after_replacement() {
    // Staging a pending build must not disturb readers: the old generation
    // serves until publish completes. Publish here is synchronous, so the
    // mid-transition state is simulated by hand-staged pending files.
    let tmp = tempfile::tempdir().expect("scratch");
    let root = tmp.path().join("redundancy");
    let (mut fabric, _) =
        RedundancyFabric::open(&root, FabricConfig::conservative(), nodes(12)).expect("opens");
    let bytes = det_bytes(LEN, 205);
    let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);
    let first = fabric.transition(asset, &bytes, rep3()).expect("protects");
    let dir = asset_dir(&root, asset);

    // Stage pending files for the next generation by hand.
    let pending_gen = first.generation + 1;
    std::fs::write(
        dir.join(format!("pending-{pending_gen}-0000.bin")),
        b"staged",
    )
    .expect("stages");
    // Both the published files and the staged pending file exist, yet reads
    // still answer from the published generation.
    assert!(dir.join(frag_name(first.generation, 0)).exists());
    assert!(dir.join(format!("pending-{pending_gen}-0000.bin")).exists());
    assert_eq!(
        fabric.read(asset).expect("old serves mid-transition"),
        bytes
    );
    assert_eq!(
        fabric.layout(&asset.id).expect("layout").generation,
        first.generation
    );
    assert_eq!(current_gen(&dir), first.generation);

    // Deleting the staged pending build (failed transition) changes nothing.
    std::fs::remove_file(dir.join(format!("pending-{pending_gen}-0000.bin"))).expect("discards");
    assert_eq!(fabric.read(asset).expect("old serves after discard"), bytes);

    // The real replacement then publishes and retires the old generation.
    let transitions_before = fabric.metrics().snapshot().transitions;
    let second = fabric
        .transition(asset, &bytes, rs42(bytes.len() as u64))
        .expect("replaces");
    assert_eq!(second.asset, asset.id, "identity unchanged");
    assert_eq!(fabric.read(asset).expect("new serves"), bytes);
    assert_eq!(current_gen(&dir), second.generation);
    old_files_gone(&dir, first.generation, rep3().total_fragments());
    assert!(
        fabric.metrics().snapshot().transitions > transitions_before,
        "transition counted"
    );
    assert!(
        fabric.metrics().snapshot().transitions_with_overlap
            >= fabric.metrics().snapshot().transitions,
        "overlap tracked"
    );
}
