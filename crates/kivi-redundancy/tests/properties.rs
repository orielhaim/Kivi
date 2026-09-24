//! Property tests for coding parameter combinations.
//!
//! Covers RS widths, asset lengths, and loss patterns (within tolerance
//! recovers exactly, beyond tolerance fails closed), plus fragment-length
//! derivation, layout round-trips, and placement determinism. Proptest cases
//! are capped at 32 for speed; seeds make loss selection deterministic.

use std::collections::HashMap;

use kivi_redundancy::{
    AssetKind, FailureScope, FragmentId, FragmentRecord, InformationAsset, RedundancyLayout,
    ReplicationParams, RsParams, SchemeParams, plan_placement,
};
use kivi_types::{NodeId, SecurityDomainId};
use proptest::prelude::*;

fn det_bytes(len: usize, seed: u64) -> Vec<u8> {
    (0..len)
        .map(|i| {
            let v = (i as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(seed)
                .wrapping_add((i as u64) * 31);
            ((v >> 11) % 251) as u8
        })
        .collect()
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

fn drop_deterministic(
    mut present: Vec<(u32, Vec<u8>)>,
    losses: usize,
    seed: u64,
) -> Vec<(u32, Vec<u8>)> {
    present.sort_by_key(|(i, _)| *i);
    for _ in 0..losses {
        if present.is_empty() {
            break;
        }
        #[allow(clippy::cast_possible_truncation)]
        let pos = (seed as usize) % present.len();
        present.remove(pos);
    }
    present
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn rs_within_tolerance_recovers_exact(
        data in 1u8..=8,
        parity in 1u8..=4,
        len in 1usize..=65536,
        seed in 0u64..=u64::MAX,
    ) {
        let flen = RsParams::shard_for_len(len as u64, data);
        let params = RsParams { data, parity, fragment_len: flen };
        prop_assert!(params.validate().is_ok());
        let bytes = det_bytes(len, seed);
        let shards = kivi_redundancy::rs::encode(&bytes, params).expect("encodes");
        prop_assert_eq!(shards.len(), (data + parity) as usize);
        for shard in &shards {
            prop_assert_eq!(shard.len(), flen as usize);
        }
        // Drop exactly `parity` shards at deterministic positions.
        let present: Vec<(u32, Vec<u8>)> = shards
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                #[allow(clippy::cast_possible_truncation)]
                let idx = i as u32;
                (idx, s)
            })
            .collect();
        let kept = drop_deterministic(present, parity as usize, seed);
        prop_assert_eq!(kept.len(), data as usize);
        let back = kivi_redundancy::rs::decode(&kept, params, len as u64).expect("recovers");
        prop_assert_eq!(back, bytes);
    }

    #[test]
    fn rs_beyond_tolerance_fails_closed(
        data in 1u8..=8,
        parity in 1u8..=4,
        len in 1usize..=16384,
        seed in 0u64..=u64::MAX,
    ) {
        let flen = RsParams::shard_for_len(len as u64, data);
        let params = RsParams { data, parity, fragment_len: flen };
        let bytes = det_bytes(len, seed ^ 0x9E37_79B9_7F4A_7C15);
        let shards = kivi_redundancy::rs::encode(&bytes, params).expect("encodes");
        let present: Vec<(u32, Vec<u8>)> = shards
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                #[allow(clippy::cast_possible_truncation)]
                let idx = i as u32;
                (idx, s)
            })
            .collect();
        // Drop parity+1: fewer than `data` remain, must fail closed.
        let kept = drop_deterministic(present, parity as usize + 1, seed);
        let err = kivi_redundancy::rs::decode(&kept, params, len as u64).expect_err("fails closed");
        prop_assert!(err.is_unrecoverable(), "got {err:?}");
    }

    #[test]
    fn rs_fragment_len_derivation_holds(
        data in 1u8..=8,
        len in 1u64..=65536,
    ) {
        let flen = RsParams::shard_for_len(len, data);
        prop_assert!(flen.is_multiple_of(RsParams::ALIGNMENT));
        prop_assert!(flen >= 2);
        prop_assert!((u64::from(data) * u64::from(flen)) >= len);
        // Encoding with the derived length succeeds; off-by-alignment fails.
        let params = RsParams { data, parity: 1, fragment_len: flen };
        prop_assert!(params.validate().is_ok());
        let bad = RsParams { data, parity: 1, fragment_len: flen.saturating_add(1) };
        if !bad.fragment_len.is_multiple_of(RsParams::ALIGNMENT) {
            prop_assert!(bad.validate().is_err());
        }
    }

    #[test]
    fn layout_round_trip_preserves_identity(
        copies in 1u8..=4,
        seed in 0u64..=u64::MAX,
    ) {
        let asset = kivi_redundancy::AssetId::new(
            AssetKind::Chunk,
            7,
            det_bytes(32, seed).try_into().expect("32 bytes"),
        ).expect("asset");
        let params = SchemeParams::Replication(ReplicationParams { copies });
        let fragments: Vec<FragmentRecord> = (0..u32::from(copies))
            .map(|index| FragmentRecord {
                id: FragmentId::for_bytes(asset, 9, index, params, b"payload"),
                node: NodeId::from_u64(u64::from(index) + 1),
                stored_len: 7,
            })
            .collect();
        let layout = RedundancyLayout::new(asset, 7, 9, params, fragments).expect("builds");
        let bytes = layout.encode();
        let back = RedundancyLayout::decode(&bytes).expect("decodes");
        prop_assert_eq!(&back, &layout);
        prop_assert_eq!(back.encode(), bytes);
    }

    #[test]
    fn placement_is_deterministic(
        generation in 1u64..=100,
        total in 1u32..=11,
        seed in 0u64..=u64::MAX,
    ) {
        let mut hash = det_bytes(32, seed);
        // Avoid the zero-sentinel hash.
        if hash.iter().all(|b| *b == 0) {
            hash[0] = 1;
        }
        let asset = kivi_redundancy::AssetId::new(
            AssetKind::Chunk,
            7,
            hash.try_into().expect("32 bytes"),
        ).expect("asset");
        let cluster = test_nodes(12);
        let first = plan_placement(asset, generation, total, FailureScope::Node, &cluster, &[])
            .expect("plans");
        let second = plan_placement(asset, generation, total, FailureScope::Node, &cluster, &[])
            .expect("plans");
        prop_assert_eq!(&first, &second);
        prop_assert_eq!(first.fragments.len(), total as usize);
    }
}

#[test]
fn replication_single_copy_suffices_many_seeds() {
    // Deterministic loop (no proptest): 16 fixed seeds, exact bytes.
    for seed in 0..16u64 {
        // Seed <16, truncation is intentional for a small deterministic size.
        #[allow(clippy::cast_possible_truncation)]
        let len = 1024 + (seed as usize) * 137;
        let bytes = det_bytes(len, seed);
        let shards = kivi_redundancy::replication::encode(&bytes, ReplicationParams { copies: 3 })
            .expect("encodes");
        assert_eq!(shards.len(), 3);
        for (i, shard) in shards.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let idx = i as u32;
            let back =
                kivi_redundancy::replication::decode(&[(idx, shard.clone())]).expect("decodes");
            assert_eq!(back, bytes, "seed {seed} copy {i}");
        }
        // Empty fails closed.
        let err = kivi_redundancy::replication::decode(&[]).expect_err("empty fails");
        assert!(err.is_unrecoverable());
    }
}

#[test]
fn content_hash_binds_asset_across_schemes() {
    // Same bytes keep the same AssetId under replication and RS; a lying
    // length fails verification instead of addressing new data.
    let bytes = det_bytes(8192, 0x00C0_FFEE);
    let asset = InformationAsset::chunk_for_bytes(&bytes, SecurityDomainId::from_u64(7));
    asset.verify_bytes(&bytes).expect("verifies");
    let lying = InformationAsset::new(asset.id, bytes.len() as u64 + 1);
    assert!(lying.verify_bytes(&bytes).is_err());
    // Both schemes decode to the same bytes that verify against one id.
    let rep = kivi_redundancy::replication::encode(&bytes, ReplicationParams { copies: 3 })
        .expect("encodes");
    let rep_back = kivi_redundancy::replication::decode(&[(0, rep[0].clone())]).expect("decodes");
    let rs_params = RsParams {
        data: 4,
        parity: 2,
        fragment_len: RsParams::shard_for_len(bytes.len() as u64, 4),
    };
    let coded = kivi_redundancy::rs::encode(&bytes, rs_params).expect("encodes");
    let present: Vec<(u32, Vec<u8>)> = coded
        .into_iter()
        .enumerate()
        .map(|(i, s)| {
            #[allow(clippy::cast_possible_truncation)]
            let idx = i as u32;
            (idx, s)
        })
        .collect();
    let rs_back =
        kivi_redundancy::rs::decode(&present, rs_params, bytes.len() as u64).expect("decodes");
    assert_eq!(rep_back, bytes);
    assert_eq!(rs_back, bytes);
    asset.verify_bytes(&rep_back).expect("verifies");
    asset.verify_bytes(&rs_back).expect("verifies");
}

#[test]
fn placement_independence_never_overcounts() {
    // Collided domains (2 nodes, 3 fragments) never inflate survivability.
    let cluster = test_nodes(2);
    let asset = kivi_redundancy::AssetId::new(AssetKind::Chunk, 7, [11; 32]).expect("asset");
    let placement = plan_placement(asset, 1, 3, FailureScope::Node, &cluster, &[]).expect("plans");
    let by_id: HashMap<NodeId, kivi_redundancy::NodeDescriptor> =
        cluster.into_iter().map(|n| (n.id, n)).collect();
    let params = SchemeParams::Replication(ReplicationParams { copies: 3 });
    let survivable = kivi_redundancy::independent_survivable(
        params,
        FailureScope::Node,
        &placement,
        &[0, 1, 2],
        &by_id,
    );
    assert_eq!(survivable, 1);
}
