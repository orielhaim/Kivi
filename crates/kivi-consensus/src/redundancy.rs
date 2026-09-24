//! Sidecar redundancy: the fabric as an additional verified sidecar source.
//!
//! ## The gate is never weakened
//!
//! A replica must not contribute a durable Raft replication acknowledgement
//! for an entry referencing a [`kivi_types::ManifestId`] until that manifest
//! and every chunk it references are locally durable and verified, with all
//! chunk-store barriers complete (see [`crate::gate::SidecarGate`]). Only
//! then may the Raft append complete.
//!
//! The helpers here never substitute for that gate. They protect the same
//! immutable sidecar bytes (chunks, canonical manifests) in a
//! [`kivi_redundancy::RedundancyFabric`] and read them back content-verified,
//! so a follower missing sidecars gains one more verified source to fetch
//! from before its local durability barrier. Bytes that fail verification
//! are rejected before install, mirroring the gate's hash-first discipline:
//! no unverified fabric byte ever becomes durable local state and no fabric
//! read ever counts as a replication acknowledgement.

use kivi_redundancy::{
    InformationAsset, RedundancyError, RedundancyFabric, RedundancyIntent, RedundancyLayout,
};
use kivi_types::{ChunkId, ManifestId, SecurityDomainId};

/// Converts a sidecar manifest identity into a fabric asset.
///
/// Delegates to [`kivi_redundancy::integration::manifest_asset`]; the
/// [`kivi_types::ManifestId`] vocabulary is shared, never paralleled.
#[must_use]
pub fn sidecar_assets(
    manifest: ManifestId,
    domain: SecurityDomainId,
    len: u64,
) -> InformationAsset {
    kivi_redundancy::integration::manifest_asset(manifest, domain, len)
}

/// Converts a sidecar chunk identity into a fabric asset.
///
/// Delegates to [`kivi_redundancy::integration::chunk_asset`].
#[must_use]
pub fn sidecar_chunk_asset(chunk: ChunkId, domain: SecurityDomainId, len: u64) -> InformationAsset {
    kivi_redundancy::integration::chunk_asset(chunk, domain, len)
}

/// Protects canonical sidecar manifest bytes in the fabric.
///
/// The manifest must already be locally staged and synced; this only adds
/// physical protection for follower repair. Never a substitute for the
/// durability gate.
///
/// # Errors
///
/// Returns [`RedundancyError`] on verification, planning, placement, or
/// publication failure.
pub fn protect_sidecar(
    fabric: &mut RedundancyFabric,
    manifest: ManifestId,
    domain: SecurityDomainId,
    canonical: &[u8],
) -> Result<RedundancyLayout, RedundancyError> {
    let asset = sidecar_assets(manifest, domain, canonical.len() as u64);
    fabric.protect(asset, canonical, &RedundancyIntent::survive_one())
}

/// Protects sidecar chunk bytes in the fabric.
///
/// Same gate-preserving contract as [`protect_sidecar`].
///
/// # Errors
///
/// Returns [`RedundancyError`] on verification, planning, placement, or
/// publication failure.
pub fn protect_chunk_sidecar(
    fabric: &mut RedundancyFabric,
    chunk: ChunkId,
    domain: SecurityDomainId,
    bytes: &[u8],
) -> Result<RedundancyLayout, RedundancyError> {
    let asset = sidecar_chunk_asset(chunk, domain, bytes.len() as u64);
    fabric.protect(asset, bytes, &RedundancyIntent::survive_one())
}

/// Reads canonical sidecar manifest bytes back through the fabric.
///
/// Callers must verify (this already does) and then durably install through
/// the sidecar store barrier before the gate may acknowledge.
///
/// # Errors
///
/// Returns [`RedundancyError`] when no layout exists, reconstruction fails,
/// or the bytes do not verify.
pub fn read_sidecar_via_fabric(
    fabric: &mut RedundancyFabric,
    manifest: ManifestId,
    domain: SecurityDomainId,
    len: u64,
) -> Result<Vec<u8>, RedundancyError> {
    fabric.read(sidecar_assets(manifest, domain, len))
}

/// Reads sidecar chunk bytes back through the fabric.
///
/// Same verified-source contract as [`read_sidecar_via_fabric`].
///
/// # Errors
///
/// Returns [`RedundancyError`] when no layout exists, reconstruction fails,
/// or the bytes do not verify.
pub fn read_chunk_via_fabric(
    fabric: &mut RedundancyFabric,
    chunk: ChunkId,
    domain: SecurityDomainId,
    len: u64,
) -> Result<Vec<u8>, RedundancyError> {
    fabric.read(sidecar_chunk_asset(chunk, domain, len))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOMAIN: SecurityDomainId = SecurityDomainId::from_u64(1);

    #[test]
    fn sidecar_identity_maps_and_verifies() {
        // Chunk leg: domain-separated identity with hash-first discipline.
        let bytes = b"sidecar chunk bytes".to_vec();
        let chunk = kivi_codec::integrity::chunk_id(DOMAIN, &bytes);
        let chunk_asset = sidecar_chunk_asset(chunk, DOMAIN, bytes.len() as u64);
        assert_eq!(
            chunk_asset,
            kivi_redundancy::integration::chunk_asset(chunk, DOMAIN, bytes.len() as u64)
        );
        chunk_asset.verify_bytes(&bytes).expect("verifies");
        assert!(chunk_asset.verify_bytes(b"lies").is_err());
        let wrong_len =
            kivi_redundancy::integration::chunk_asset(chunk, DOMAIN, bytes.len() as u64 + 1);
        assert!(wrong_len.verify_bytes(&bytes).is_err());

        // Manifest leg: build one real manifest so the id is well-formed.
        let (manifest_obj, manifest_id) = kivi_chunk::build_manifest(
            DOMAIN,
            kivi_chunk::Chunking::DEFAULT,
            kivi_chunk::ChunkCodecId::NONE,
            bytes.len() as u64,
            vec![kivi_chunk::ChunkEntry {
                id: chunk,
                len: bytes.len() as u64,
            }],
        )
        .expect("builds manifest");
        let mut canonical = Vec::new();
        manifest_obj.encode_canonical(&mut canonical);
        assert_eq!(
            kivi_codec::integrity::manifest_id(DOMAIN, &canonical),
            manifest_id
        );
        let asset = sidecar_assets(manifest_id, DOMAIN, canonical.len() as u64);
        assert_eq!(
            asset,
            kivi_redundancy::integration::manifest_asset(
                manifest_id,
                DOMAIN,
                canonical.len() as u64
            )
        );
        asset.verify_bytes(&canonical).expect("verifies");
        // Tampered bytes never verify: the gate would refuse to install
        // them and never acknowledge the Raft append.
        let mut tampered = canonical.clone();
        if let Some(first) = tampered.first_mut() {
            *first ^= 0xFF;
        }
        assert!(asset.verify_bytes(&tampered).is_err());
    }
}
