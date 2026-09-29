//! Integration: real chunk/checkpoint durability paths use the fabric.
//!
//! The fabric never duplicates the Raft durability gate and never
//! erasure-codes mutable Raft state. It protects immutable durable assets -
//! chunks, chunk manifests, checkpoint bands/segments, dedup components -
//! whose bytes the owning stores already stage durably. Adapters here convert
//! those stores' identities into [`crate::AssetId`]s and drive
//! protect/read/transition through one [`crate::RedundancyFabric`], so the
//! same physical policy covers every immutable family.
//!
//! Node joins, drains, migrations, and failures integrate by refreshing the
//! fabric's cluster view and calling [`crate::RedundancyFabric::drain_node`] -
//! no separate membership system.

use kivi_types::{ChunkId, ManifestId, SecurityDomainId};

use crate::{AssetId, AssetKind, InformationAsset, RedundancyError};

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
}
