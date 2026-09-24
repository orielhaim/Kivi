//! Integration: real chunk/checkpoint durability paths use the fabric.
//!
//! The fabric never duplicates the Raft durability gate and never
//! erasure-codes mutable Raft state. It protects immutable durable assets —
//! chunks, chunk manifests, checkpoint bands/segments, dedup components —
//! whose bytes the owning stores already stage durably. Adapters here convert
//! those stores' identities into [`crate::AssetId`]s and drive
//! protect/read/transition through one [`crate::RedundancyFabric`], so the
//! same physical policy covers every immutable family.
//!
//! Node joins, drains, migrations, and failures integrate by refreshing the
//! fabric's cluster view ([`cluster_view_from_records`]) and calling
//! [`crate::RedundancyFabric::drain_node`] — no separate membership system.

use kivi_types::{ChunkId, ManifestId, NodeId, SecurityDomainId};

use crate::{AssetId, AssetKind, InformationAsset, NodeDescriptor, NodeHealth, RedundancyError};

/// Converts a chunk's identity into a fabric asset.
#[must_use]
pub fn chunk_asset(id: ChunkId, domain: SecurityDomainId, logical_len: u64) -> InformationAsset {
    InformationAsset::new(AssetId::chunk(id, domain), logical_len)
}

/// Converts a chunk manifest's identity into a fabric asset.
#[must_use]
pub fn manifest_asset(
    id: ManifestId,
    domain: SecurityDomainId,
    canonical_len: u64,
) -> InformationAsset {
    InformationAsset::new(AssetId::chunk_manifest(id, domain), canonical_len)
}

/// Converts a checkpoint band's content hash into a fabric asset.
#[must_use]
pub fn band_asset(hash: [u8; 32], stored_len: u64) -> InformationAsset {
    InformationAsset::new(
        AssetId::checkpoint(AssetKind::CheckpointBand, hash),
        stored_len,
    )
}

/// Converts a checkpoint manifest's content hash into a fabric asset.
#[must_use]
pub fn checkpoint_manifest_asset(hash: [u8; 32], stored_len: u64) -> InformationAsset {
    InformationAsset::new(
        AssetId::checkpoint(AssetKind::CheckpointManifest, hash),
        stored_len,
    )
}

/// Converts a checkpoint dedup component into a fabric asset.
#[must_use]
pub fn dedup_asset(hash: [u8; 32], stored_len: u64) -> InformationAsset {
    InformationAsset::new(
        AssetId::checkpoint(AssetKind::CheckpointDedup, hash),
        stored_len,
    )
}

/// Minimal node record view for cluster-view conversion (mirrors
/// `kivi-control`'s `NodeRecord` fields without depending on that crate).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterNodeRecord {
    /// Stable node identity.
    pub id: NodeId,
    /// Failure-domain label (`zone/rack/host-group`; empty when unknown).
    pub failure_domain: String,
    /// Whether the node accepts new placements.
    pub accepts_new: bool,
    /// Whether the node serves reads.
    pub serves_reads: bool,
    /// Whether the node's data needs replacement.
    pub needs_repair: bool,
}

impl ClusterNodeRecord {
    /// Builds an active record with no domain label.
    #[must_use]
    pub const fn active(id: NodeId) -> Self {
        Self {
            id,
            failure_domain: String::new(),
            accepts_new: true,
            serves_reads: true,
            needs_repair: false,
        }
    }
}

/// Converts control-plane node records into fabric descriptors.
///
/// Health mapping: `accepts_new → Active`; `needs_repair → Unavailable`
/// (drain vs outage is carried by the label where the caller knows it —
/// pass `draining_ids` to mark operator drains as [`NodeHealth::Draining`]
/// instead of `Unavailable`).
#[must_use]
pub fn cluster_view_from_records(
    records: &[ClusterNodeRecord],
    draining_ids: &[NodeId],
) -> Vec<NodeDescriptor> {
    records
        .iter()
        .map(|record| {
            let health = if draining_ids.contains(&record.id) || record.needs_repair {
                if draining_ids.contains(&record.id) {
                    NodeHealth::Draining
                } else {
                    NodeHealth::Unavailable
                }
            } else if record.accepts_new {
                NodeHealth::Active
            } else {
                NodeHealth::Joining
            };
            // The control plane's single failure-domain label maps onto the
            // rack slot (the fabric treats unknown zone/region as collapsed,
            // so this never over-claims independence).
            let domain = crate::FailureDomain::full(
                record.id,
                record.failure_domain.clone(),
                String::new(),
                String::new(),
            );
            NodeDescriptor {
                id: record.id,
                domain,
                health,
                weight: 1,
            }
        })
        .collect()
}

/// Verifies fabric bytes against a chunk id before the caller installs them
/// as authoritative (mirrors the sidecar gate's discipline: hash first).
///
/// # Errors
///
/// Returns [`RedundancyError::VerificationFailed`] on mismatch.
pub fn verify_chunk_bytes(
    id: ChunkId,
    domain: SecurityDomainId,
    bytes: &[u8],
) -> Result<(), RedundancyError> {
    let asset = chunk_asset(id, domain, bytes.len() as u64);
    asset.verify_bytes(bytes)
}

/// Verifies fabric bytes against a manifest id.
///
/// # Errors
///
/// Returns [`RedundancyError::VerificationFailed`] on mismatch.
pub fn verify_manifest_bytes(
    id: ManifestId,
    domain: SecurityDomainId,
    canonical: &[u8],
) -> Result<(), RedundancyError> {
    let asset = manifest_asset(id, domain, canonical.len() as u64);
    asset.verify_bytes(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_adapter_verifies_bytes() {
        let domain = SecurityDomainId::from_u64(4);
        let bytes = b"adapter bytes".to_vec();
        let id = kivi_codec::integrity::chunk_id(domain, &bytes);
        verify_chunk_bytes(id, domain, &bytes).expect("verifies");
        assert!(verify_chunk_bytes(id, domain, b"lies").is_err());
        let asset = chunk_asset(id, domain, bytes.len() as u64);
        assert_eq!(asset.id.as_chunk(), Some(id));
    }

    #[test]
    fn cluster_view_maps_health() {
        let records = vec![
            ClusterNodeRecord::active(NodeId::from_u64(1)),
            ClusterNodeRecord {
                id: NodeId::from_u64(2),
                failure_domain: "rack-a".to_owned(),
                accepts_new: false,
                serves_reads: false,
                needs_repair: true,
            },
        ];
        let view = cluster_view_from_records(&records, &[NodeId::from_u64(2)]);
        assert_eq!(view[0].health, NodeHealth::Active);
        assert_eq!(view[1].health, NodeHealth::Draining);
        assert_eq!(view[1].domain.rack, "rack-a");
    }
}
