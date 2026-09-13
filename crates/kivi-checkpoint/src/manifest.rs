//! Checkpoint manifest format: the immutable index of one checkpoint.
//!
//! The manifest binds a checkpoint cut to its content-addressed artifacts.
//! It is itself content-addressed (`manifests/<blake3hex>.manifest`) and
//! referenced by the tablet's `CURRENT` pointer.
//!
//! ```text
//! header (88 bytes, fixed):
//!   0   4  magic "KVCM"
//!   4   2  major = 1
//!   6   2  minor = 0
//!   8  16  cluster (u128)
//!   24  8  node (u64)
//!   32  8  incarnation (u64, the writer's own)
//!   40  8  namespace (u64)
//!   48  8  tablet (u64)
//!   56  8  epoch (u64)
//!   64  8  guard (u64)
//!   72  8  cut commit position (u64)
//!   80  2  band-layout id (u16)
//!   82  2  band descriptor count (u16, must equal layout band count)
//!   84  4  CRC32C of bytes [0..84]
//! body:
//!   band descriptors sorted by band id, 56 bytes each:
//!     band u16, content hash [32], length u64, records u32,
//!     codec u8, reserved u8, band cut u64
//!   dedup reference (48 bytes):
//!     content hash [32], length u64, sessions u32, outcomes u32, reserved u32
//!   feature floor u64, has_prev u8, prev manifest hash [32], reserved [7]
//! footer (40 bytes): body CRC32C, BLAKE3 of header[0..84] + body
//! (content identity covers artifact identity), footer CRC32C
//! (uniform footer with bands and dedup components)
//! ```
//!
//! Band `cut` may lag the manifest cut for reused bands (their bytes were
//! sealed earlier and provably unchanged since); it must never lead it.
//! The manifest cut is the checkpoint's durable position for recovery and
//! WAL reclamation.

use kivi_codec::integrity::crc32c_checksum;
use kivi_types::{
    ClusterId, NamespaceId, NodeId, NodeIncarnation, TabletEpoch, TabletId, WriteGuardGeneration,
};

use crate::artifact::ArtifactHash;
use crate::band::CompressionCodecId;
use crate::error::CheckpointError;
use crate::layout::{BANDS_PER_TABLET, BandLayoutId};

/// Magic word: ASCII `"KVCM"` read as a little-endian `u32`.
pub const MANIFEST_MAGIC: u32 = 0x4D43_564B;
/// Current manifest major version.
pub const MANIFEST_MAJOR: u16 = 1;
/// Current manifest minor version.
pub const MANIFEST_MINOR: u16 = 0;
/// Encoded manifest header length in bytes.
pub const MANIFEST_HEADER_LEN: usize = 88;
/// Encoded band descriptor length in bytes.
pub const BAND_DESCRIPTOR_LEN: usize = 56;
/// Encoded dedup reference length in bytes.
pub const DEDUP_REF_LEN: usize = 48;
/// Feature floor this stage mints (capability floor for readers).
pub const FEATURE_FLOOR_V1: u64 = 1;

/// One band reference inside a manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BandDescriptor {
    /// Band id.
    pub band: u16,
    /// Content hash of the band file.
    pub hash: ArtifactHash,
    /// File length in bytes.
    pub len: u64,
    /// Records in the band.
    pub records: u32,
    /// Compression codec of the band body.
    pub codec: CompressionCodecId,
    /// Cut the band's bytes were sealed at (<= manifest cut).
    pub cut: u64,
}

/// The dedup-component reference inside a manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DedupRef {
    /// Content hash of the dedup file.
    pub hash: ArtifactHash,
    /// File length in bytes.
    pub len: u64,
    /// Sessions encoded.
    pub sessions: u32,
    /// Outcomes encoded.
    pub outcomes: u32,
}

/// A built manifest: content identity plus file bytes.
#[derive(Debug, Clone)]
pub struct BuiltManifest {
    /// BLAKE3 of the body (filename and CURRENT reference).
    pub hash: ArtifactHash,
    /// Complete file bytes.
    pub bytes: Vec<u8>,
}

/// A verified loaded manifest.
#[derive(Debug, Clone)]
pub struct LoadedManifest {
    /// Owning cluster.
    pub cluster: ClusterId,
    /// Owning node.
    pub node: NodeId,
    /// Writer incarnation (forensics).
    pub incarnation: NodeIncarnation,
    /// Namespace served.
    pub namespace: NamespaceId,
    /// Tablet checkpointed.
    pub tablet: TabletId,
    /// Tablet epoch at capture.
    pub epoch: TabletEpoch,
    /// Write-guard generation at capture.
    pub guard: WriteGuardGeneration,
    /// Checkpoint cut (durable position for recovery/reclamation).
    pub cut: u64,
    /// Band layout used.
    pub layout: BandLayoutId,
    /// Band descriptors sorted by band id (all bands, always).
    pub bands: Vec<BandDescriptor>,
    /// Dedup component reference.
    pub dedup: DedupRef,
    /// Feature/capability floor.
    pub feature_floor: u64,
    /// Previous manifest hash (retention chain), if any.
    pub previous: Option<ArtifactHash>,
}

/// Manifest build inputs: checkpoint identity plus artifact references.
#[derive(Debug, Clone)]
pub struct ManifestInput {
    /// Owning cluster.
    pub cluster: ClusterId,
    /// Owning node.
    pub node: NodeId,
    /// Writer incarnation.
    pub incarnation: NodeIncarnation,
    /// Namespace served.
    pub namespace: NamespaceId,
    /// Tablet checkpointed.
    pub tablet: TabletId,
    /// Tablet epoch at capture.
    pub epoch: TabletEpoch,
    /// Write-guard generation at capture.
    pub guard: WriteGuardGeneration,
    /// Checkpoint cut.
    pub cut: u64,
    /// Band layout used.
    pub layout: BandLayoutId,
    /// All band descriptors (sorted by the builder).
    pub bands: Vec<BandDescriptor>,
    /// Dedup component reference.
    pub dedup: DedupRef,
    /// Previous manifest hash (retention chain), if any.
    pub previous: Option<ArtifactHash>,
}

/// Builds one immutable manifest file.
///
/// # Errors
///
/// Returns [`CheckpointError`] when the band set is incomplete (every
/// band must be present -- absence is never an empty band) or encodings
/// overflow.
pub fn build_manifest(input: ManifestInput) -> Result<BuiltManifest, CheckpointError> {
    let mut bands = input.bands;
    bands.sort_by_key(|descriptor| descriptor.band);
    if bands.len() != BANDS_PER_TABLET {
        return Err(CheckpointError::Format {
            detail: format!(
                "manifest must reference all {BANDS_PER_TABLET} bands, got {}",
                bands.len()
            ),
        });
    }
    for (index, descriptor) in bands.iter().enumerate() {
        if usize::from(descriptor.band) != index {
            return Err(CheckpointError::Format {
                detail: format!("manifest band order broken at index {index}"),
            });
        }
        if descriptor.cut > input.cut {
            return Err(CheckpointError::Format {
                detail: format!(
                    "band {} cut {} leads manifest cut {}",
                    descriptor.band, descriptor.cut, input.cut
                ),
            });
        }
    }
    let mut header = [0u8; MANIFEST_HEADER_LEN];
    header[0..4].copy_from_slice(&MANIFEST_MAGIC.to_le_bytes());
    header[4..6].copy_from_slice(&MANIFEST_MAJOR.to_le_bytes());
    header[6..8].copy_from_slice(&MANIFEST_MINOR.to_le_bytes());
    header[8..24].copy_from_slice(&input.cluster.as_u128().to_le_bytes());
    header[24..32].copy_from_slice(&input.node.as_u64().to_le_bytes());
    header[32..40].copy_from_slice(&input.incarnation.as_u64().to_le_bytes());
    header[40..48].copy_from_slice(&input.namespace.as_u64().to_le_bytes());
    header[48..56].copy_from_slice(&input.tablet.as_u64().to_le_bytes());
    header[56..64].copy_from_slice(&input.epoch.as_u64().to_le_bytes());
    header[64..72].copy_from_slice(&input.guard.as_u64().to_le_bytes());
    header[72..80].copy_from_slice(&input.cut.to_le_bytes());
    header[80..82].copy_from_slice(&input.layout.as_u16().to_le_bytes());
    header[82..84].copy_from_slice(&crate::layout::BAND_COUNT_U16.to_le_bytes());
    let header_crc = crc32c_checksum(&header[..84]);
    header[84..88].copy_from_slice(&header_crc.to_le_bytes());
    let mut body = Vec::new();
    for descriptor in &bands {
        body.extend_from_slice(&descriptor.band.to_le_bytes());
        body.extend_from_slice(descriptor.hash.as_bytes());
        body.extend_from_slice(&descriptor.len.to_le_bytes());
        body.extend_from_slice(&descriptor.records.to_le_bytes());
        body.push(descriptor.codec.as_u8());
        body.push(0);
        body.extend_from_slice(&descriptor.cut.to_le_bytes());
    }
    body.extend_from_slice(input.dedup.hash.as_bytes());
    body.extend_from_slice(&input.dedup.len.to_le_bytes());
    body.extend_from_slice(&input.dedup.sessions.to_le_bytes());
    body.extend_from_slice(&input.dedup.outcomes.to_le_bytes());
    body.extend_from_slice(&FEATURE_FLOOR_V1.to_le_bytes());
    match input.previous {
        None => {
            body.push(0);
            body.extend_from_slice(&[0u8; 32]);
        }
        Some(hash) => {
            body.push(1);
            body.extend_from_slice(hash.as_bytes());
        }
    }
    body.extend_from_slice(&[0u8; 7]);
    let body_crc = crc32c_checksum(&body);
    let content_hash = crate::artifact::ArtifactHash::of_chunks(&[&header[..84], &body]);
    let mut footer = [0u8; 40];
    footer[0..4].copy_from_slice(&body_crc.to_le_bytes());
    footer[4..36].copy_from_slice(content_hash.as_bytes());
    let footer_crc = crc32c_checksum(&footer[..36]);
    footer[36..40].copy_from_slice(&footer_crc.to_le_bytes());
    let mut bytes = Vec::with_capacity(MANIFEST_HEADER_LEN + body.len() + footer.len());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&body);
    bytes.extend_from_slice(&footer);
    Ok(BuiltManifest {
        hash: content_hash,
        bytes,
    })
}

/// Loads and fully verifies one manifest file: filename identity, header
/// CRC and provenance, body CRC and content hash, band-set completeness
/// and order, per-band cut discipline, and codec/layout support.
///
/// # Errors
///
/// Returns [`CheckpointError`] on any integrity or format violation.
// Long linear field-by-field decoder (like the WAL scanner).
#[allow(clippy::too_many_lines)]
pub fn load_manifest(
    expected_hash: ArtifactHash,
    bytes: &[u8],
) -> Result<LoadedManifest, CheckpointError> {
    let corrupt = |detail: String| CheckpointError::Corrupt { detail };
    let format = |detail: String| CheckpointError::Format { detail };
    if bytes.len() < MANIFEST_HEADER_LEN + 40 {
        return Err(corrupt(format!(
            "manifest shorter than header + footer: {} bytes",
            bytes.len()
        )));
    }
    let (header, rest) = bytes.split_at(MANIFEST_HEADER_LEN);
    let (body, footer) = rest.split_at(rest.len() - 40);
    if u32::from_le_bytes(
        header[0..4]
            .try_into()
            .map_err(|_| corrupt("manifest magic unreadable".to_owned()))?,
    ) != MANIFEST_MAGIC
    {
        return Err(format("manifest magic mismatch".to_owned()));
    }
    if crc32c_checksum(&header[..84])
        != u32::from_le_bytes(
            header[84..88]
                .try_into()
                .map_err(|_| corrupt("manifest header CRC unreadable".to_owned()))?,
        )
    {
        return Err(corrupt("manifest header CRC mismatch".to_owned()));
    }
    let major = u16::from_le_bytes(
        header[4..6]
            .try_into()
            .map_err(|_| corrupt("manifest version unreadable".to_owned()))?,
    );
    let minor = u16::from_le_bytes(
        header[6..8]
            .try_into()
            .map_err(|_| corrupt("manifest version unreadable".to_owned()))?,
    );
    if major != MANIFEST_MAJOR || minor != MANIFEST_MINOR {
        return Err(CheckpointError::Unsupported {
            detail: format!("manifest version {major}.{minor}"),
        });
    }
    let cluster = ClusterId::from_u128(u128::from_le_bytes(
        header[8..24]
            .try_into()
            .map_err(|_| corrupt("manifest cluster unreadable".to_owned()))?,
    ));
    let node = NodeId::from_u64(u64::from_le_bytes(
        header[24..32]
            .try_into()
            .map_err(|_| corrupt("manifest node unreadable".to_owned()))?,
    ));
    let incarnation = NodeIncarnation::from_u64(u64::from_le_bytes(
        header[32..40]
            .try_into()
            .map_err(|_| corrupt("manifest incarnation unreadable".to_owned()))?,
    ));
    let namespace = NamespaceId::from_u64(u64::from_le_bytes(
        header[40..48]
            .try_into()
            .map_err(|_| corrupt("manifest namespace unreadable".to_owned()))?,
    ));
    let tablet = TabletId::from_u64(u64::from_le_bytes(
        header[48..56]
            .try_into()
            .map_err(|_| corrupt("manifest tablet unreadable".to_owned()))?,
    ));
    let epoch = TabletEpoch::from_u64(u64::from_le_bytes(
        header[56..64]
            .try_into()
            .map_err(|_| corrupt("manifest epoch unreadable".to_owned()))?,
    ));
    let guard = WriteGuardGeneration::from_u64(u64::from_le_bytes(
        header[64..72]
            .try_into()
            .map_err(|_| corrupt("manifest guard unreadable".to_owned()))?,
    ));
    let cut = u64::from_le_bytes(
        header[72..80]
            .try_into()
            .map_err(|_| corrupt("manifest cut unreadable".to_owned()))?,
    );
    let layout = BandLayoutId::from_u16(u16::from_le_bytes(
        header[80..82]
            .try_into()
            .map_err(|_| corrupt("manifest layout unreadable".to_owned()))?,
    ));
    if !layout.is_supported() {
        return Err(CheckpointError::Unsupported {
            detail: format!("manifest band layout {layout}"),
        });
    }
    let band_count = u16::from_le_bytes(
        header[82..84]
            .try_into()
            .map_err(|_| corrupt("manifest band count unreadable".to_owned()))?,
    ) as usize;
    if band_count != BANDS_PER_TABLET {
        return Err(format(format!(
            "manifest band count {band_count} is not {BANDS_PER_TABLET}"
        )));
    }
    let expect_body = BANDS_PER_TABLET * BAND_DESCRIPTOR_LEN + DEDUP_REF_LEN + 8 + 1 + 32 + 7;
    if body.len() != expect_body {
        return Err(corrupt(format!(
            "manifest body length {} disagrees with descriptor layout {expect_body}",
            body.len()
        )));
    }
    if u32::from_le_bytes(
        footer[0..4]
            .try_into()
            .map_err(|_| corrupt("manifest footer unreadable".to_owned()))?,
    ) != crc32c_checksum(body)
    {
        return Err(corrupt("manifest body CRC mismatch".to_owned()));
    }
    let content_hash: [u8; 32] = footer[4..36]
        .try_into()
        .map_err(|_| corrupt("manifest content hash unreadable".to_owned()))?;
    if crate::artifact::ArtifactHash::of_chunks(&[&header[..84], body]).as_bytes() != &content_hash
    {
        return Err(corrupt("manifest BLAKE3 content hash mismatch".to_owned()));
    }
    if ArtifactHash::from_bytes(content_hash) != expected_hash {
        return Err(corrupt(
            "manifest content hash does not match its catalog reference".to_owned(),
        ));
    }
    if crc32c_checksum(&footer[..36])
        != u32::from_le_bytes(
            footer[36..40]
                .try_into()
                .map_err(|_| corrupt("manifest footer CRC unreadable".to_owned()))?,
        )
    {
        return Err(corrupt("manifest footer CRC mismatch".to_owned()));
    }
    let mut bands = Vec::with_capacity(BANDS_PER_TABLET);
    for index in 0..BANDS_PER_TABLET {
        let base = index * BAND_DESCRIPTOR_LEN;
        let descriptor = &body[base..base + BAND_DESCRIPTOR_LEN];
        let band = u16::from_le_bytes(
            descriptor[0..2]
                .try_into()
                .map_err(|_| corrupt("band descriptor unreadable".to_owned()))?,
        );
        if usize::from(band) != index {
            return Err(corrupt(format!(
                "manifest band order broken at index {index}"
            )));
        }
        let hash: [u8; 32] = descriptor[2..34]
            .try_into()
            .map_err(|_| corrupt("band hash unreadable".to_owned()))?;
        let len = u64::from_le_bytes(
            descriptor[34..42]
                .try_into()
                .map_err(|_| corrupt("band length unreadable".to_owned()))?,
        );
        let records = u32::from_le_bytes(
            descriptor[42..46]
                .try_into()
                .map_err(|_| corrupt("band records unreadable".to_owned()))?,
        );
        let codec = CompressionCodecId::from_u8(descriptor[46]);
        if !codec.is_supported() {
            return Err(CheckpointError::Unsupported {
                detail: format!("band {band} compression codec {codec}"),
            });
        }
        if descriptor[47] != 0 {
            return Err(format(format!("band {band} reserved byte nonzero")));
        }
        let band_cut = u64::from_le_bytes(
            descriptor[48..56]
                .try_into()
                .map_err(|_| corrupt("band cut unreadable".to_owned()))?,
        );
        if band_cut > cut {
            return Err(corrupt(format!(
                "band {band} cut {band_cut} leads manifest cut {cut}"
            )));
        }
        bands.push(BandDescriptor {
            band,
            hash: ArtifactHash::from_bytes(hash),
            len,
            records,
            codec,
            cut: band_cut,
        });
    }
    let tail = &body[BANDS_PER_TABLET * BAND_DESCRIPTOR_LEN..];
    let dedup_hash: [u8; 32] = tail[0..32]
        .try_into()
        .map_err(|_| corrupt("dedup hash unreadable".to_owned()))?;
    let dedup_len = u64::from_le_bytes(
        tail[32..40]
            .try_into()
            .map_err(|_| corrupt("dedup length unreadable".to_owned()))?,
    );
    let dedup_sessions = u32::from_le_bytes(
        tail[40..44]
            .try_into()
            .map_err(|_| corrupt("dedup sessions unreadable".to_owned()))?,
    );
    let dedup_outcomes = u32::from_le_bytes(
        tail[44..48]
            .try_into()
            .map_err(|_| corrupt("dedup outcomes unreadable".to_owned()))?,
    );
    let feature_floor = u64::from_le_bytes(
        tail[48..56]
            .try_into()
            .map_err(|_| corrupt("feature floor unreadable".to_owned()))?,
    );
    if feature_floor != FEATURE_FLOOR_V1 {
        return Err(CheckpointError::Unsupported {
            detail: format!("checkpoint feature floor {feature_floor}"),
        });
    }
    let previous = match tail[56] {
        0 => {
            if tail[57..89] != [0u8; 32] {
                return Err(format("previous hash present without flag".to_owned()));
            }
            None
        }
        1 => {
            let hash: [u8; 32] = tail[57..89]
                .try_into()
                .map_err(|_| corrupt("previous hash unreadable".to_owned()))?;
            Some(ArtifactHash::from_bytes(hash))
        }
        flag => {
            return Err(format(format!("previous flag {flag} is not 0 or 1")));
        }
    };
    if tail[89..96] != [0u8; 7] {
        return Err(format("manifest reserved bytes nonzero".to_owned()));
    }
    Ok(LoadedManifest {
        cluster,
        node,
        incarnation,
        namespace,
        tablet,
        epoch,
        guard,
        cut,
        layout,
        bands,
        dedup: DedupRef {
            hash: ArtifactHash::from_bytes(dedup_hash),
            len: dedup_len,
            sessions: dedup_sessions,
            outcomes: dedup_outcomes,
        },
        feature_floor,
        previous,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::ArtifactHash;

    fn descriptor(band: u16, cut: u64) -> BandDescriptor {
        BandDescriptor {
            band,
            hash: ArtifactHash::of(format!("band-{band}").as_bytes()),
            len: 128,
            records: 2,
            codec: CompressionCodecId::NONE,
            cut,
        }
    }

    fn input(cut: u64) -> ManifestInput {
        ManifestInput {
            cluster: ClusterId::from_u128(1),
            node: NodeId::from_u64(2),
            incarnation: NodeIncarnation::from_u64(3),
            namespace: NamespaceId::from_u64(4),
            tablet: TabletId::from_u64(5),
            epoch: TabletEpoch::from_u64(6),
            guard: WriteGuardGeneration::from_u64(7),
            cut,
            layout: BandLayoutId::HASH_PREFIX_8,
            bands: (0..crate::layout::BAND_COUNT_U16)
                .map(|band| descriptor(band, cut))
                .collect(),
            dedup: DedupRef {
                hash: ArtifactHash::of(b"dedup"),
                len: 64,
                sessions: 1,
                outcomes: 2,
            },
            previous: None,
        }
    }

    #[test]
    fn manifest_round_trips_with_chain_link() {
        let first = build_manifest(input(10)).expect("builds");
        let second = build_manifest(ManifestInput {
            previous: Some(first.hash),
            ..input(20)
        })
        .expect("builds");
        let loaded = load_manifest(second.hash, &second.bytes).expect("loads");
        assert_eq!(loaded.cut, 20);
        assert_eq!(loaded.bands.len(), BANDS_PER_TABLET);
        assert_eq!(loaded.previous, Some(first.hash));
        assert_eq!(loaded.feature_floor, FEATURE_FLOOR_V1);
    }

    #[test]
    fn incomplete_band_sets_are_rejected() {
        let mut partial = input(10);
        partial.bands.pop();
        assert!(build_manifest(partial).is_err());
    }

    #[test]
    fn leading_band_cuts_are_rejected() {
        let mut bad = input(10);
        bad.bands[3].cut = 11;
        assert!(build_manifest(bad).is_err());
    }

    #[test]
    fn corruption_is_detected() {
        let built = build_manifest(input(10)).expect("builds");
        let mut damaged = built.bytes.clone();
        damaged[MANIFEST_HEADER_LEN + 5] ^= 0xFF;
        assert!(load_manifest(built.hash, &damaged).is_err());
    }
}
