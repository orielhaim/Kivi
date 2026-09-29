//! Control-to-fabric mapping: registry records into placement descriptors.
//!
//! The control plane owns desired topology; the fabric owns physical
//! protection. This adapter converts [`crate::NodeRecord`] snapshots into
//! [`kivi_redundancy::NodeDescriptor`] snapshots without building a second
//! membership system and without leaking fabric internals back into control
//! state. Joins, drains, migrations, and failures flow through
//! [`descriptors_from_registry`] plus [`kivi_redundancy::RedundancyFabric::set_nodes`]
//! and [`kivi_redundancy::RedundancyFabric::drain_node`].

use kivi_redundancy::{AssetId, FailureDomain, NodeDescriptor, NodeHealth, PublishedLayout};

use crate::layouts::LayoutKey;
use crate::state::ControlState;
use crate::{NodeRecord, NodeState};

/// Maps one control lifecycle state onto the fabric health vocabulary.
///
/// The mapping is one-to-one: every control state has exactly one fabric
/// counterpart, so no liveness nuance is lost or invented here.
const fn health_of(state: NodeState) -> NodeHealth {
    match state {
        NodeState::Active => NodeHealth::Active,
        NodeState::Draining => NodeHealth::Draining,
        NodeState::Unavailable => NodeHealth::Unavailable,
        NodeState::Suspect => NodeHealth::Suspect,
        NodeState::Joining => NodeHealth::Joining,
        NodeState::Drained => NodeHealth::Drained,
        NodeState::Removed => NodeHealth::Removed,
    }
}

/// Converts registry records into fabric placement descriptors.
///
/// The control plane's single failure-domain label maps onto the fabric
/// rack slot (unknown zone and region collapse, which never over-claims
/// independence); weights below one normalize to one.
#[must_use]
pub fn descriptors_from_registry(records: &[NodeRecord]) -> Vec<NodeDescriptor> {
    records
        .iter()
        .map(|record| NodeDescriptor {
            id: record.node,
            domain: FailureDomain::full(
                record.node,
                record.failure_domain.clone(),
                String::new(),
                String::new(),
            ),
            health: health_of(record.state),
        })
        .collect()
}

/// Derives the control map key for an asset's published redundancy layout.
#[must_use]
pub fn layout_key(asset: &AssetId) -> LayoutKey {
    LayoutKey::from_asset(asset)
}

/// Returns the published layout for an asset, if any.
///
/// The control plane stores layout bytes opaquely, so committed bytes that
/// fail [`kivi_redundancy::RedundancyLayout::decode`] are treated as absent
/// (`None`): readers never see a half-parsed layout, and operators observe
/// the missing publication as an unpublished asset rather than corrupt
/// state.
#[must_use]
pub fn published_layout(state: &ControlState, asset: &AssetId) -> Option<PublishedLayout> {
    let record = state.redundancy_layout(&layout_key(asset))?;
    let layout = kivi_redundancy::RedundancyLayout::decode(&record.layout).ok()?;
    Some(PublishedLayout {
        control_generation: record.control_generation,
        layout,
    })
}

#[cfg(test)]
mod tests {
    use kivi_types::NodeId;

    use super::*;
    use crate::NodeState;

    fn record(id: u64, state: NodeState, domain: &str) -> NodeRecord {
        NodeRecord {
            node: NodeId::from_u64(id),
            peer: "127.0.0.1:9144".parse().expect("peer"),
            native: "127.0.0.1:9044".parse().expect("native"),
            admin: "127.0.0.1:19444".parse().expect("admin"),
            cert_fingerprint: [7u8; 32],
            failure_domain: domain.to_owned(),
            weight: 1,
            state,
        }
    }

    #[test]
    fn descriptors_carry_domain_and_health() {
        let records = vec![
            record(1, NodeState::Active, "rack-a"),
            record(2, NodeState::Draining, "rack-b"),
            record(3, NodeState::Active, "rack-c"),
        ];
        let descriptors = descriptors_from_registry(&records);
        assert_eq!(descriptors.len(), 3);
        assert_eq!(descriptors[0].domain.rack, "rack-a");
        assert_eq!(descriptors[1].health, NodeHealth::Draining);
    }

    fn published_asset() -> kivi_redundancy::AssetId {
        kivi_redundancy::AssetId::new(kivi_redundancy::AssetKind::Chunk, 7, [11; 32])
            .expect("asset")
    }

    fn layout_bytes(asset: &kivi_redundancy::AssetId, generation: u64) -> Vec<u8> {
        let params =
            kivi_redundancy::SchemeParams::Replication(kivi_redundancy::ReplicationParams {
                copies: 2,
            });
        let fragments = (0..2u32)
            .map(|index| kivi_redundancy::FragmentRecord {
                id: kivi_redundancy::FragmentId::for_bytes(
                    *asset, generation, index, params, b"bytes",
                ),
                node: NodeId::from_u64(u64::from(index) + 1),
                stored_len: 5,
            })
            .collect();
        kivi_redundancy::RedundancyLayout::new(*asset, 5, generation, params, fragments)
            .expect("builds")
            .encode()
    }

    #[test]
    fn layout_key_generation_and_published_layout() {
        use crate::layouts::LayoutRecord;
        use crate::state::ControlState;

        let asset = published_asset();
        let other = kivi_redundancy::AssetId::new(kivi_redundancy::AssetKind::Chunk, 7, [12; 32])
            .expect("asset");
        assert_eq!(layout_key(&asset), layout_key(&asset));
        assert_ne!(layout_key(&asset), layout_key(&other));

        let mut state = ControlState::empty();
        assert!(published_layout(&state, &asset).is_none());
        state
            .apply(&crate::ControlMutation::PublishRedundancyLayout {
                key: layout_key(&asset),
                generation: 1,
                record: LayoutRecord::new(1, layout_bytes(&asset, 1)),
            })
            .expect("publishes");
        let published = published_layout(&state, &asset).expect("present");
        assert_eq!(published.control_generation, 1);
        assert_eq!(published.layout.generation, 1);
        assert_eq!(published.layout.asset, asset);
        assert!(published_layout(&state, &other).is_none());
    }

    #[test]
    fn corrupt_committed_bytes_read_as_absent() {
        use crate::layouts::LayoutRecord;
        use crate::state::ControlState;

        // The control plane stores layout bytes opaquely: garbage commits
        // (impossible through honest coordinators, possible through a bug)
        // surface as unpublished, never as corrupt layouts.
        let asset = published_asset();
        let mut state = ControlState::empty();
        state
            .apply(&crate::ControlMutation::PublishRedundancyLayout {
                key: layout_key(&asset),
                generation: 1,
                record: LayoutRecord::new(1, vec![0xFFu8; 32]),
            })
            .expect("opaque store accepts");
        assert!(published_layout(&state, &asset).is_none());
    }
}
