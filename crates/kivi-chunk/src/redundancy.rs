//! Chunk-store redundancy adapter: immutable identities into the fabric.
//!
//! The chunk store stages packs locally and proves durability through its
//! own sync barrier first; the [`kivi_redundancy::RedundancyFabric`] is an
//! additional verified source for the same immutable bytes, never a
//! substitute for the local durability ordering. Identity is never
//! duplicated here: every asset constructor below delegates to
//! [`kivi_redundancy::integration`] so [`kivi_types::ChunkId`] and
//! [`kivi_types::ManifestId`] stay the single vocabulary.

use kivi_redundancy::{
    InformationAsset, RedundancyError, RedundancyFabric, RedundancyIntent, RedundancyLayout,
};
use kivi_types::{ChunkId, ManifestId, SecurityDomainId};

/// Converts a chunk identity into a fabric asset.
///
/// Delegates to [`kivi_redundancy::integration::chunk_asset`]; no identity
/// logic is duplicated here.
#[must_use]
pub fn chunk_asset(id: ChunkId, domain: SecurityDomainId, len: u64) -> InformationAsset {
    kivi_redundancy::integration::chunk_asset(id, domain, len)
}

/// Converts a chunk-manifest identity into a fabric asset.
///
/// Delegates to [`kivi_redundancy::integration::manifest_asset`].
#[must_use]
pub fn manifest_asset(id: ManifestId, domain: SecurityDomainId, len: u64) -> InformationAsset {
    kivi_redundancy::integration::manifest_asset(id, domain, len)
}

/// Protects staged chunk bytes in the fabric under the minimum durable
/// intent (survive one independent failure).
///
/// The caller stages and syncs through [`crate::ChunkStore`] first; this
/// only adds physical protection for the same verified bytes.
///
/// # Errors
///
/// Returns [`RedundancyError`] when the bytes do not verify under `id`, the
/// intent cannot be planned or placed, or the fabric cannot publish.
pub fn protect_staged(
    fabric: &mut RedundancyFabric,
    domain: SecurityDomainId,
    id: ChunkId,
    bytes: &[u8],
) -> Result<RedundancyLayout, RedundancyError> {
    let asset = chunk_asset(id, domain, bytes.len() as u64);
    fabric.protect(asset, bytes, &RedundancyIntent::survive_one())
}

/// Protects staged manifest bytes in the fabric.
///
/// Same contract as [`protect_staged`], for canonical manifest bytes.
///
/// # Errors
///
/// Returns [`RedundancyError`] on verification, planning, placement, or
/// publication failure.
pub fn protect_manifest_staged(
    fabric: &mut RedundancyFabric,
    domain: SecurityDomainId,
    id: ManifestId,
    canonical: &[u8],
) -> Result<RedundancyLayout, RedundancyError> {
    let asset = manifest_asset(id, domain, canonical.len() as u64);
    fabric.protect(asset, canonical, &RedundancyIntent::survive_one())
}

/// Reads chunk bytes back through the fabric, content-verified against `id`.
///
/// # Errors
///
/// Returns [`RedundancyError`] when no layout exists, reconstruction fails,
/// or the bytes do not verify.
pub fn read_via_fabric(
    fabric: &mut RedundancyFabric,
    domain: SecurityDomainId,
    id: ChunkId,
    len: u64,
) -> Result<Vec<u8>, RedundancyError> {
    let asset = chunk_asset(id, domain, len);
    fabric.read(asset)
}

/// Reads canonical manifest bytes back through the fabric.
///
/// # Errors
///
/// Returns [`RedundancyError`] when no layout exists, reconstruction fails,
/// or the bytes do not verify.
pub fn read_manifest_via_fabric(
    fabric: &mut RedundancyFabric,
    domain: SecurityDomainId,
    id: ManifestId,
    len: u64,
) -> Result<Vec<u8>, RedundancyError> {
    let asset = manifest_asset(id, domain, len);
    fabric.read(asset)
}

#[cfg(test)]
mod tests {
    use kivi_redundancy::{FabricConfig, NodeDescriptor};
    use kivi_types::{NodeId, WallTimestamp};

    use super::*;
    use crate::store::ChunkStore;

    const DOMAIN: SecurityDomainId = SecurityDomainId::from_u64(3);

    fn fabric_nodes() -> Vec<NodeDescriptor> {
        (1..=5)
            .map(|id| NodeDescriptor::active(NodeId::from_u64(id)))
            .collect()
    }

    fn open_store(root: &std::path::Path) -> ChunkStore {
        ChunkStore::open(
            root,
            0,
            DOMAIN,
            1024 * 1024,
            WallTimestamp::from_micros(1_000_000),
        )
        .expect("store opens")
        .0
    }

    #[test]
    fn fabric_recovers_chunk_after_pack_loss() {
        let chunk_root = tempfile::tempdir().expect("scratch");
        let fabric_root = tempfile::tempdir().expect("scratch");
        let bytes = vec![7u8; 4096];
        let id = kivi_codec::integrity::chunk_id(DOMAIN, &bytes);
        let (manifest_obj, manifest_id) = crate::manifest::build_manifest(
            DOMAIN,
            crate::policy::Chunking {
                version: crate::policy::CHUNKING_V1,
                chunk_size: 4096,
            },
            crate::codec::ChunkCodecId::NONE,
            bytes.len() as u64,
            vec![crate::manifest::ChunkEntry {
                id,
                len: bytes.len() as u64,
            }],
        )
        .expect("builds manifest");
        let mut canonical = Vec::new();
        manifest_obj.encode_canonical(&mut canonical);

        let mut store = open_store(chunk_root.path());
        store.stage_chunk(id, &bytes).expect("stages chunk");
        store
            .stage_manifest(manifest_id, &canonical)
            .expect("stages manifest");
        store.sync().expect("syncs");
        assert!(store.durable_chunk(id).is_some());
        assert!(store.durable_manifest(manifest_id).is_some());

        let (mut fabric, _) = RedundancyFabric::open(
            &fabric_root.path().join("redundancy"),
            FabricConfig::conservative(),
            fabric_nodes(),
        )
        .expect("fabric opens");
        protect_staged(&mut fabric, DOMAIN, id, &bytes).expect("protects chunk");
        protect_manifest_staged(&mut fabric, DOMAIN, manifest_id, &canonical)
            .expect("protects manifest");
        drop(store);

        // Simulate total loss of the primary: delete every pack file.
        let lane = chunk_root.path().join("chunks").join("lane-0000");
        let mut deleted = 0u32;
        for entry in std::fs::read_dir(&lane).expect("lists lane") {
            let entry = entry.expect("reads entry");
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "pack") {
                std::fs::remove_file(&path).expect("deletes pack");
                deleted += 1;
            }
        }
        assert!(deleted > 0, "a pack file must exist to delete");

        let back = read_via_fabric(&mut fabric, DOMAIN, id, bytes.len() as u64).expect("recovers");
        assert_eq!(back, bytes);
        let back_manifest =
            read_manifest_via_fabric(&mut fabric, DOMAIN, manifest_id, canonical.len() as u64)
                .expect("recovers manifest");
        assert_eq!(back_manifest, canonical);
    }
}
