//! Checkpoint redundancy adapter: bands, manifests, and dedup via the fabric.
//!
//! Checkpoint files are immutable and content-addressed; the fabric holds an
//! additional verified copy of the same file bytes under its own BLAKE3
//! identity. Two identities coexist by design: the checkpoint content
//! identity (BLAKE3 of header prefix plus body, minted by `build_*`) proves
//! checkpoint provenance via `load_*`, while the fabric asset below is the
//! BLAKE3 of the exact whole-file bytes protected (so `verify_bytes`
//! passes). Recovery returns byte-identical files, which then load under
//! their checkpoint identities. Identity constructors delegate to
//! [`kivi_redundancy::integration`]; no hashing logic is duplicated here.

use kivi_redundancy::{
    InformationAsset, RedundancyError, RedundancyFabric, RedundancyIntent, RedundancyLayout,
};

/// Converts a band file's fabric hash into a fabric asset.
///
/// `hash` is the BLAKE3 of the whole band file bytes (the fabric identity),
/// not the checkpoint content hash; `stored_len` is the file length.
/// Delegates to [`kivi_redundancy::integration::band_asset`].
#[must_use]
pub fn band_asset(hash: [u8; 32], stored_len: u64) -> InformationAsset {
    kivi_redundancy::integration::band_asset(hash, stored_len)
}

/// Converts a manifest file's fabric hash into a fabric asset.
///
/// Same whole-file identity discipline as [`band_asset`]; delegates to
/// [`kivi_redundancy::integration::checkpoint_manifest_asset`].
#[must_use]
pub fn manifest_asset(hash: [u8; 32], stored_len: u64) -> InformationAsset {
    kivi_redundancy::integration::checkpoint_manifest_asset(hash, stored_len)
}

/// Converts a dedup-component file's fabric hash into a fabric asset.
///
/// Same whole-file identity discipline as [`band_asset`]; delegates to
/// [`kivi_redundancy::integration::dedup_asset`].
#[must_use]
pub fn dedup_asset(hash: [u8; 32], stored_len: u64) -> InformationAsset {
    kivi_redundancy::integration::dedup_asset(hash, stored_len)
}

/// Protects whole band file bytes in the fabric (survive one failure).
///
/// # Errors
///
/// Returns [`RedundancyError`] on verification, planning, placement, or
/// publication failure.
pub fn protect_band(
    fabric: &mut RedundancyFabric,
    hash: [u8; 32],
    bytes: &[u8],
) -> Result<RedundancyLayout, RedundancyError> {
    let asset = band_asset(hash, bytes.len() as u64);
    fabric.protect(asset, bytes, &RedundancyIntent::survive_one())
}

/// Protects whole manifest file bytes in the fabric.
///
/// # Errors
///
/// Returns [`RedundancyError`] on verification, planning, placement, or
/// publication failure.
pub fn protect_manifest(
    fabric: &mut RedundancyFabric,
    hash: [u8; 32],
    bytes: &[u8],
) -> Result<RedundancyLayout, RedundancyError> {
    let asset = manifest_asset(hash, bytes.len() as u64);
    fabric.protect(asset, bytes, &RedundancyIntent::survive_one())
}

/// Protects whole dedup-component file bytes in the fabric.
///
/// # Errors
///
/// Returns [`RedundancyError`] on verification, planning, placement, or
/// publication failure.
pub fn protect_dedup(
    fabric: &mut RedundancyFabric,
    hash: [u8; 32],
    bytes: &[u8],
) -> Result<RedundancyLayout, RedundancyError> {
    let asset = dedup_asset(hash, bytes.len() as u64);
    fabric.protect(asset, bytes, &RedundancyIntent::survive_one())
}

/// Reads arbitrary checkpoint bytes back through the fabric.
///
/// The caller builds `asset` with [`band_asset`], [`manifest_asset`], or
/// [`dedup_asset`] using the whole-file BLAKE3 and length.
///
/// # Errors
///
/// Returns [`RedundancyError`] when no layout exists, reconstruction fails,
/// or the bytes do not verify.
pub fn read_via_fabric(
    fabric: &mut RedundancyFabric,
    asset: InformationAsset,
) -> Result<Vec<u8>, RedundancyError> {
    fabric.read(asset)
}

/// Reads band file bytes back through the fabric.
///
/// # Errors
///
/// Returns [`RedundancyError`] when no layout exists, reconstruction fails,
/// or the bytes do not verify.
pub fn read_band_via_fabric(
    fabric: &mut RedundancyFabric,
    hash: [u8; 32],
    len: u64,
) -> Result<Vec<u8>, RedundancyError> {
    fabric.read(band_asset(hash, len))
}

/// Reads manifest file bytes back through the fabric.
///
/// # Errors
///
/// Returns [`RedundancyError`] when no layout exists, reconstruction fails,
/// or the bytes do not verify.
pub fn read_manifest_via_fabric(
    fabric: &mut RedundancyFabric,
    hash: [u8; 32],
    len: u64,
) -> Result<Vec<u8>, RedundancyError> {
    fabric.read(manifest_asset(hash, len))
}

/// Reads dedup-component file bytes back through the fabric.
///
/// # Errors
///
/// Returns [`RedundancyError`] when no layout exists, reconstruction fails,
/// or the bytes do not verify.
pub fn read_dedup_via_fabric(
    fabric: &mut RedundancyFabric,
    hash: [u8; 32],
    len: u64,
) -> Result<Vec<u8>, RedundancyError> {
    fabric.read(dedup_asset(hash, len))
}

#[cfg(test)]
mod tests {
    use kivi_redundancy::{FabricConfig, NodeDescriptor};
    use kivi_state::{Key, LogicalValue, ObjectVersion, StoredObject};
    use kivi_types::{Expiry, NodeId, TabletId};

    use super::*;
    use crate::band::{BandRecord, CompressionCodecId, build_band, load_band};
    use crate::layout::BandLayoutId;

    fn fabric_nodes() -> Vec<NodeDescriptor> {
        (1..=5)
            .map(|id| NodeDescriptor::active(NodeId::from_u64(id)))
            .collect()
    }

    fn record(key: &str, value: i64) -> BandRecord {
        BandRecord {
            hash: 0xABCD,
            key: Key::from(key),
            object: StoredObject::restore(
                LogicalValue::StrictCounter(value),
                ObjectVersion::FIRST,
                Expiry::NEVER,
            ),
        }
    }

    #[test]
    fn fabric_recovers_band_after_file_loss() {
        let scratch = tempfile::tempdir().expect("scratch");
        let built = build_band(
            TabletId::from_u64(1),
            41,
            0,
            BandLayoutId::HASH_PREFIX_8,
            CompressionCodecId::NONE,
            vec![record("b", 2), record("a", 1)],
        )
        .expect("builds");
        // Fabric identity is the whole-file BLAKE3 (distinct from the
        // checkpoint content identity in `built.hash`).
        let file_hash = kivi_codec::integrity::blake3_256(&built.bytes);
        let file_len = built.bytes.len() as u64;

        let (mut fabric, _) = RedundancyFabric::open(
            &scratch.path().join("redundancy"),
            FabricConfig::conservative(),
            fabric_nodes(),
        )
        .expect("fabric opens");
        protect_band(&mut fabric, file_hash, &built.bytes).expect("protects");

        // Simulate loss of the primary: write the band file, then delete it.
        let band_path = scratch.path().join("band.band");
        std::fs::write(&band_path, &built.bytes).expect("writes band");
        std::fs::remove_file(&band_path).expect("deletes band");
        assert!(!band_path.exists());

        let back = read_band_via_fabric(&mut fabric, file_hash, file_len).expect("recovers band");
        assert_eq!(back, built.bytes);
        // Recovered bytes still load under the checkpoint content identity.
        let loaded = load_band(
            TabletId::from_u64(1),
            0,
            BandLayoutId::HASH_PREFIX_8,
            built.hash,
            &back,
        )
        .expect("recovered band loads");
        assert_eq!(loaded.records.len(), 2);
    }
}
