//! Immutable checkpoint band format: canonical sorted records per band.
//!
//! One band file holds every object of a tablet whose key maps to that
//! physical band at the checkpoint cut. Files are content-addressed
//! (`bands/<blake3hex>.band`); the manifest references them by hash.
//!
//! ```text
//! header (60 bytes, fixed):
//!   0   4  magic "KVCB"
//!   4   2  major = 1
//!   6   2  minor = 0
//!   8   8  tablet (u64)
//!   16  8  cut commit position (u64)
//!   24  2  band id (u16)
//!   26  2  band-layout id (u16)
//!   28  4  record count (u32)
//!   32  8  stored body length in bytes (u64)
//!   40  8  uncompressed body length in bytes (u64)
//!   48  1  compression codec id (u8)
//!   49  7  reserved (0)
//!   56  4  CRC32C of bytes [0..56]
//! stored body (body_len bytes; decompress per codec for the raw body)
//! raw body: records sorted by (partition_hash, key bytes):
//!   partition_hash u128, key_len u32, key[..],
//!   repr u8 (0 = inline, 1 = chunked; others rejected),
//!   type u8 (0 = bytes, 1 = counter),
//!   version u64, expiry (Expiry codec),
//!   payload: inline bytes → len u32 + raw; inline counter → i64;
//!            chunked bytes → manifest [32] + logical_len u64
//!            (small roots only — bulk bytes stay in chunk packs)
//! footer (40 bytes, fixed):
//!   0   4  CRC32C of the stored body bytes
//!   4  32  BLAKE3 of header[0..56] + stored body (content identity:
//!          covers artifact identity, not just the body — empty bands of
//!          different ids share an empty body but never an identity)
//!   36  4  CRC32C of bytes [0..36]
//! ```
//!
//! Full key comparison is authoritative: the partition hash orders coarsely
//! (and lets recovery skip bands by hash range in the future), but records
//! with colliding hashes sort by full key bytes and duplicates are
//! rejected.

use kivi_codec::integrity::crc32c_checksum;
use kivi_codec::{CodecError, Decode, Encode};
use kivi_state::{Key, LogicalValue, StoredObject};
use kivi_types::TabletId;

use crate::artifact::ArtifactHash;
use crate::error::CheckpointError;
use crate::layout::{BANDS_PER_TABLET, BandLayoutId};

/// Magic word: ASCII `"KVCB"` read as a little-endian `u32`.
pub const BAND_MAGIC: u32 = 0x4243_564B;
/// Current band major version.
pub const BAND_MAJOR: u16 = 1;
/// Current band minor version.
pub const BAND_MINOR: u16 = 0;
/// Encoded band header length in bytes.
pub const BAND_HEADER_LEN: usize = 60;
/// Encoded band footer length in bytes.
pub const BAND_FOOTER_LEN: usize = 40;
/// Inline representation tag (the only one this stage mints).
pub const REPR_INLINE: u8 = 0;
/// Chunked-root representation tag: the record carries a manifest id plus
/// logical length, never bulk bytes. Fixed forever within band version 1.
pub const REPR_CHUNKED: u8 = 1;
/// Fabric-root representation tag: the record carries a fabric object id
/// plus logical length and published version, never bulk bytes. Fixed
/// forever within band version 1. The bytes stay materialized in the
/// fabric; recovery re-links by id and validates version + checksum.
pub const REPR_FABRIC: u8 = 2;
/// Bytes value tag.
pub const TYPE_BYTES: u8 = 0;
/// Counter value tag.
pub const TYPE_COUNTER: u8 = 1;
/// Commutative-counter value tag (order-free sum). Fixed forever.
pub const TYPE_COMMUTATIVE: u8 = 2;
/// Bounded-counter value tag (value plus escrow share). Fixed forever.
pub const TYPE_BOUNDED: u8 = 3;
/// Semaphore value tag (capacity plus live permits). Fixed forever.
pub const TYPE_SEMAPHORE: u8 = 4;
/// Lease value tag (holder plus fencing). Fixed forever.
pub const TYPE_LEASE: u8 = 5;
/// Stream-shard value tag (cursor plus retained entries). Fixed forever.
pub const TYPE_STREAM_SHARD: u8 = 6;

/// Compression codec for a band body. `NONE` is the production default;
/// `LZ4` is the evaluated optional experiment (see the checkpoint
/// benchmark) — never silent, always explicit in the header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompressionCodecId(u8);

impl CompressionCodecId {
    /// No compression (production default).
    pub const NONE: Self = Self(0);
    /// LZ4 block compression (optional experiment arm).
    pub const LZ4: Self = Self(1);

    /// Wraps a raw value, including unknown future assignments.
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self.0
    }

    /// Whether this build can encode and decode this codec.
    #[must_use]
    pub const fn is_supported(self) -> bool {
        matches!(self.0, 0 | 1)
    }
}

impl core::fmt::Display for CompressionCodecId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0 => write!(f, "none"),
            1 => write!(f, "lz4"),
            other => write!(f, "unknown-codec({other})"),
        }
    }
}

/// One object placed into a band: its partition hash (coarse order), key
/// (authoritative order), and stored state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BandRecord {
    /// Partition hash of the key under the checkpoint layout's hasher.
    pub hash: u128,
    /// Object key.
    pub key: Key,
    /// Stored state (value, version, expiry).
    pub object: StoredObject,
}

/// A built band: content identity plus the exact file bytes to publish.
#[derive(Debug, Clone)]
pub struct BuiltBand {
    /// BLAKE3 of the stored body (filename and manifest reference).
    pub hash: ArtifactHash,
    /// Complete file bytes (header + stored body + footer).
    pub bytes: Vec<u8>,
    /// Records encoded.
    pub records: u32,
    /// Uncompressed body bytes (before codec).
    pub raw_bytes: u64,
    /// Codec the stored body uses (recorded in the header and manifest).
    pub codec: CompressionCodecId,
}

/// A verified loaded band.
#[derive(Debug, Clone)]
pub struct LoadedBand {
    /// Tablet the band belongs to.
    pub tablet: TabletId,
    /// Checkpoint cut the band was taken at.
    pub cut: u64,
    /// Band id.
    pub band: u16,
    /// Records in (hash, key) order.
    pub records: Vec<BandRecord>,
}

/// Builds one immutable band file from unsorted records. Sorts by
/// (`hash`, `key`), rejects duplicate keys, encodes, hashes.
///
/// An empty record list still produces a valid empty band: deletion of a
/// band's final object is explicit state, never inferred absence (spec Q).
///
/// # Errors
///
/// Returns [`CheckpointError`] on duplicate keys, unsupported codecs, or
/// oversized encodings.
pub fn build_band(
    tablet: TabletId,
    cut: u64,
    band: u16,
    layout: BandLayoutId,
    codec: CompressionCodecId,
    mut records: Vec<BandRecord>,
) -> Result<BuiltBand, CheckpointError> {
    if usize::from(band) >= BANDS_PER_TABLET {
        return Err(CheckpointError::Format {
            detail: format!("band id {band} outside 0..{BANDS_PER_TABLET}"),
        });
    }
    if !codec.is_supported() {
        return Err(CheckpointError::Unsupported {
            detail: format!("band compression codec {codec} not enabled in this build"),
        });
    }
    records.sort_by(|left, right| {
        (left.hash, left.key.as_bytes()).cmp(&(right.hash, right.key.as_bytes()))
    });
    for pair in records.windows(2) {
        if pair[0].hash == pair[1].hash && pair[0].key == pair[1].key {
            return Err(CheckpointError::Format {
                detail: format!("duplicate key in band {band}: {:?}", pair[0].key),
            });
        }
    }
    let mut raw = Vec::new();
    for record in &records {
        encode_record(record, &mut raw)?;
    }
    // Codec branch (the only place compression lands): NONE stores the raw
    // body verbatim; LZ4 stores the deterministic block-compressed body
    // (empty bodies stay empty — no codec edge cases, and header identity
    // still separates the codecs). Content identity covers the stored body,
    // so identical logical content under the same codec has identical
    // identity; different codecs never share identity (headers differ).
    let stored = match codec.as_u8() {
        0 => raw.clone(),
        1 if raw.is_empty() => Vec::new(),
        1 => lz4_flex::block::compress(&raw),
        _ => {
            return Err(CheckpointError::Unsupported {
                detail: format!("band compression codec {codec} not enabled in this build"),
            });
        }
    };
    let records_u32 = u32::try_from(records.len()).map_err(|_| CheckpointError::Format {
        detail: format!("band {band} holds more than u32::MAX records"),
    })?;
    let mut header = [0u8; BAND_HEADER_LEN];
    header[0..4].copy_from_slice(&BAND_MAGIC.to_le_bytes());
    header[4..6].copy_from_slice(&BAND_MAJOR.to_le_bytes());
    header[6..8].copy_from_slice(&BAND_MINOR.to_le_bytes());
    header[8..16].copy_from_slice(&tablet.as_u64().to_le_bytes());
    header[16..24].copy_from_slice(&cut.to_le_bytes());
    header[24..26].copy_from_slice(&band.to_le_bytes());
    header[26..28].copy_from_slice(&layout.as_u16().to_le_bytes());
    header[28..32].copy_from_slice(&records_u32.to_le_bytes());
    header[32..40].copy_from_slice(&(stored.len() as u64).to_le_bytes());
    header[40..48].copy_from_slice(&(raw.len() as u64).to_le_bytes());
    header[48] = codec.as_u8();
    // bytes [49..56] stay zero (reserved).
    let header_crc = crc32c_checksum(&header[..56]);
    header[56..60].copy_from_slice(&header_crc.to_le_bytes());
    let body_crc = crc32c_checksum(&stored);
    let content_hash = crate::artifact::ArtifactHash::of_chunks(&[&header[..56], &stored]);
    let mut footer = [0u8; BAND_FOOTER_LEN];
    footer[0..4].copy_from_slice(&body_crc.to_le_bytes());
    footer[4..36].copy_from_slice(content_hash.as_bytes());
    let footer_crc = crc32c_checksum(&footer[..36]);
    footer[36..40].copy_from_slice(&footer_crc.to_le_bytes());
    let mut bytes = Vec::with_capacity(BAND_HEADER_LEN + stored.len() + BAND_FOOTER_LEN);
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&stored);
    bytes.extend_from_slice(&footer);
    Ok(BuiltBand {
        hash: content_hash,
        bytes,
        records: records_u32,
        raw_bytes: raw.len() as u64,
        codec,
    })
}

/// Loads and fully verifies one band file: filename identity, header CRC
/// and provenance, body CRC and content hash, record count, ordering,
/// duplicate keys, codec support, and object decoding.
///
/// # Errors
///
/// Returns [`CheckpointError`] on any integrity or format violation. A
/// corrupt immutable artifact is never silently accepted.
// Long linear field-by-field decoder (like the WAL scanner): split
// further only if fields grow.
#[allow(clippy::too_many_lines)]
pub fn load_band(
    expected_tablet: TabletId,
    expected_band: u16,
    expected_layout: BandLayoutId,
    expected_hash: ArtifactHash,
    bytes: &[u8],
) -> Result<LoadedBand, CheckpointError> {
    let corrupt = |detail: String| CheckpointError::Corrupt { detail };
    let format = |detail: String| CheckpointError::Format { detail };
    if bytes.len() < BAND_HEADER_LEN + BAND_FOOTER_LEN {
        return Err(corrupt(format!(
            "band file shorter than header + footer: {} bytes",
            bytes.len()
        )));
    }
    let (header, rest) = bytes.split_at(BAND_HEADER_LEN);
    let (stored, footer) = rest.split_at(rest.len() - BAND_FOOTER_LEN);
    if u32::from_le_bytes(
        header[0..4]
            .try_into()
            .map_err(|_| corrupt("band header unreadable".to_owned()))?,
    ) != BAND_MAGIC
    {
        return Err(format("band magic mismatch".to_owned()));
    }
    if crc32c_checksum(&header[..56])
        != u32::from_le_bytes(
            header[56..60]
                .try_into()
                .map_err(|_| corrupt("band header CRC unreadable".to_owned()))?,
        )
    {
        return Err(corrupt("band header CRC mismatch".to_owned()));
    }
    let major = u16::from_le_bytes(
        header[4..6]
            .try_into()
            .map_err(|_| corrupt("band version unreadable".to_owned()))?,
    );
    let minor = u16::from_le_bytes(
        header[6..8]
            .try_into()
            .map_err(|_| corrupt("band version unreadable".to_owned()))?,
    );
    if major != BAND_MAJOR || minor != BAND_MINOR {
        return Err(CheckpointError::Unsupported {
            detail: format!("band version {major}.{minor}"),
        });
    }
    let tablet = TabletId::from_u64(u64::from_le_bytes(
        header[8..16]
            .try_into()
            .map_err(|_| corrupt("band tablet unreadable".to_owned()))?,
    ));
    if tablet != expected_tablet {
        return Err(format(format!(
            "band tablet {tablet} does not match descriptor"
        )));
    }
    let cut = u64::from_le_bytes(
        header[16..24]
            .try_into()
            .map_err(|_| corrupt("band cut unreadable".to_owned()))?,
    );
    let band = u16::from_le_bytes(
        header[24..26]
            .try_into()
            .map_err(|_| corrupt("band id unreadable".to_owned()))?,
    );
    if band != expected_band {
        return Err(format(format!("band id {band} does not match descriptor")));
    }
    let layout = BandLayoutId::from_u16(u16::from_le_bytes(
        header[26..28]
            .try_into()
            .map_err(|_| corrupt("band layout unreadable".to_owned()))?,
    ));
    if layout != expected_layout {
        return Err(format(format!(
            "band layout {layout} does not match manifest"
        )));
    }
    if !layout.is_supported() {
        return Err(CheckpointError::Unsupported {
            detail: format!("band layout {layout}"),
        });
    }
    let record_count = u32::from_le_bytes(
        header[28..32]
            .try_into()
            .map_err(|_| corrupt("band count unreadable".to_owned()))?,
    ) as usize;
    let body_len = usize::try_from(u64::from_le_bytes(
        header[32..40]
            .try_into()
            .map_err(|_| corrupt("band length unreadable".to_owned()))?,
    ))
    .map_err(|_| corrupt("band length exceeds address space".to_owned()))?;
    let raw_len = u64::from_le_bytes(
        header[40..48]
            .try_into()
            .map_err(|_| corrupt("band length unreadable".to_owned()))?,
    );
    let codec = CompressionCodecId::from_u8(header[48]);
    if header[49..56] != [0u8; 7] {
        return Err(format("band reserved bytes nonzero".to_owned()));
    }
    if body_len != stored.len() {
        return Err(corrupt(format!(
            "band body length {body_len} disagrees with file {}",
            stored.len()
        )));
    }
    if !codec.is_supported() {
        return Err(CheckpointError::Unsupported {
            detail: format!("band compression codec {codec}"),
        });
    }
    // Codec-aware body recovery (integrity of the stored body is verified
    // below before this raw view is trusted): NONE views the stored body
    // directly; LZ4 decompresses it against the authoritative raw length.
    // Unknown codecs never reach here (rejected above).
    let raw_len_usize = usize::try_from(raw_len)
        .map_err(|_| corrupt("band length exceeds address space".to_owned()))?;
    let raw_owned: Vec<u8>;
    let raw_slice: &[u8] = match codec.as_u8() {
        0 => {
            if body_len != raw_len_usize {
                return Err(corrupt(format!(
                    "band stored length {body_len} disagrees with raw length {raw_len} for codec {codec}"
                )));
            }
            stored
        }
        1 => {
            if raw_len_usize == 0 {
                if !stored.is_empty() {
                    return Err(corrupt(format!(
                        "band stored length {body_len} disagrees with empty raw body for codec {codec}"
                    )));
                }
                stored
            } else {
                raw_owned = lz4_flex::block::decompress(stored, raw_len_usize)
                    .map_err(|_| corrupt("band LZ4 body fails decompression".to_owned()))?;
                if raw_owned.len() != raw_len_usize {
                    return Err(corrupt(format!(
                        "band LZ4 decoded length {} disagrees with raw length {raw_len}",
                        raw_owned.len()
                    )));
                }
                &raw_owned
            }
        }
        _ => {
            return Err(CheckpointError::Unsupported {
                detail: format!("band compression codec {codec}"),
            });
        }
    };
    if u32::from_le_bytes(
        footer[0..4]
            .try_into()
            .map_err(|_| corrupt("band footer unreadable".to_owned()))?,
    ) != crc32c_checksum(stored)
    {
        return Err(corrupt("band body CRC mismatch".to_owned()));
    }
    let content_hash: [u8; 32] = footer[4..36]
        .try_into()
        .map_err(|_| corrupt("band content hash unreadable".to_owned()))?;
    if crate::artifact::ArtifactHash::of_chunks(&[&header[..56], stored]).as_bytes()
        != &content_hash
    {
        return Err(corrupt("band BLAKE3 content hash mismatch".to_owned()));
    }
    if ArtifactHash::from_bytes(content_hash) != expected_hash {
        return Err(corrupt(
            "band content hash does not match its manifest reference".to_owned(),
        ));
    }
    if crc32c_checksum(&footer[..36])
        != u32::from_le_bytes(
            footer[36..40]
                .try_into()
                .map_err(|_| corrupt("band footer CRC unreadable".to_owned()))?,
        )
    {
        return Err(corrupt("band footer CRC mismatch".to_owned()));
    }
    let mut records = Vec::with_capacity(record_count.min(1_000_000));
    let mut cursor = raw_slice;
    let mut previous: Option<(u128, Vec<u8>)> = None;
    while !cursor.is_empty() {
        let (record, consumed) = decode_record(cursor)
            .map_err(|error| corrupt(format!("band record undecodable: {error:?}")))?;
        if let Some((prev_hash, prev_key)) = &previous
            && (*prev_hash, prev_key.as_slice()) >= (record.hash, record.key.as_bytes())
        {
            return Err(corrupt(
                "band records out of order or duplicated".to_owned(),
            ));
        }
        previous = Some((record.hash, record.key.as_bytes().to_vec()));
        records.push(record);
        cursor = &cursor[consumed..];
    }
    if records.len() != record_count {
        return Err(corrupt(format!(
            "band record count {} disagrees with decoded {}",
            record_count,
            records.len()
        )));
    }
    Ok(LoadedBand {
        tablet,
        cut,
        band,
        records,
    })
}

fn encode_record(record: &BandRecord, out: &mut Vec<u8>) -> Result<(), CheckpointError> {
    out.extend_from_slice(&record.hash.to_le_bytes());
    let key = record.key.as_bytes();
    let key_len = u32::try_from(key.len()).map_err(|_| CheckpointError::Format {
        detail: "band key exceeds u32 length".to_owned(),
    })?;
    out.extend_from_slice(&key_len.to_le_bytes());
    out.extend_from_slice(key);
    match record.object.value() {
        // Chunked roots serialize as their small root (manifest id plus
        // logical length), never the bulk bytes: checkpoints hold small
        // logical roots while chunk packs hold bulk information (§29).
        LogicalValue::Chunked(chunked) => {
            out.push(REPR_CHUNKED);
            out.push(TYPE_BYTES);
            out.extend_from_slice(&record.object.version().as_u64().to_le_bytes());
            record.object.expiry().encode(out);
            out.extend_from_slice(chunked.manifest.as_bytes());
            out.extend_from_slice(&chunked.logical_len.to_le_bytes());
        }
        // Fabric roots serialize as their small root (fabric id plus
        // logical length and published version), never the materialized
        // bytes: the fabric owns bulk information, the checkpoint owns
        // the logical reference. Recovery re-links and revalidates.
        LogicalValue::Fabric(fabric) => {
            out.push(REPR_FABRIC);
            out.push(TYPE_BYTES);
            out.extend_from_slice(&record.object.version().as_u64().to_le_bytes());
            record.object.expiry().encode(out);
            out.extend_from_slice(&fabric.id.to_le_bytes());
            out.extend_from_slice(&fabric.logical_len.to_le_bytes());
            out.extend_from_slice(&fabric.version.to_le_bytes());
        }
        LogicalValue::Bytes(value) => {
            out.push(REPR_INLINE);
            out.push(TYPE_BYTES);
            out.extend_from_slice(&record.object.version().as_u64().to_le_bytes());
            record.object.expiry().encode(out);
            let len = u32::try_from(value.len()).map_err(|_| CheckpointError::Format {
                detail: "band value exceeds u32 length".to_owned(),
            })?;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(value);
        }
        LogicalValue::StrictCounter(value) => {
            out.push(REPR_INLINE);
            out.push(TYPE_COUNTER);
            out.extend_from_slice(&record.object.version().as_u64().to_le_bytes());
            record.object.expiry().encode(out);
            out.extend_from_slice(&value.to_le_bytes());
        }
        LogicalValue::CommutativeCounter(value) => {
            out.push(REPR_INLINE);
            out.push(TYPE_COMMUTATIVE);
            out.extend_from_slice(&record.object.version().as_u64().to_le_bytes());
            record.object.expiry().encode(out);
            out.extend_from_slice(&value.to_le_bytes());
        }
        LogicalValue::BoundedCounter(state) => {
            out.push(REPR_INLINE);
            out.push(TYPE_BOUNDED);
            out.extend_from_slice(&record.object.version().as_u64().to_le_bytes());
            record.object.expiry().encode(out);
            state.encode(out);
        }
        LogicalValue::Semaphore(state) => {
            out.push(REPR_INLINE);
            out.push(TYPE_SEMAPHORE);
            out.extend_from_slice(&record.object.version().as_u64().to_le_bytes());
            record.object.expiry().encode(out);
            state.encode(out);
        }
        LogicalValue::Lease(state) => {
            out.push(REPR_INLINE);
            out.push(TYPE_LEASE);
            out.extend_from_slice(&record.object.version().as_u64().to_le_bytes());
            record.object.expiry().encode(out);
            state.encode(out);
        }
        LogicalValue::StreamShard(shard) => {
            out.push(REPR_INLINE);
            out.push(TYPE_STREAM_SHARD);
            out.extend_from_slice(&record.object.version().as_u64().to_le_bytes());
            record.object.expiry().encode(out);
            shard.encode(out);
        }
    }
    Ok(())
}

/// Takes exactly `N` bytes, advancing the cursor. Short input is the
/// only failure (callers size their reads from framing they already
/// validated). Shared with the dedup component decoder (same take rule,
/// one definition).
pub(crate) fn band_take<const N: usize>(
    input: &[u8],
    at: &mut usize,
) -> Result<[u8; N], CodecError> {
    if input.len() < *at + N {
        return Err(CodecError::Truncated {
            expected: *at + N,
            available: input.len(),
        });
    }
    let mut out = [0u8; N];
    out.copy_from_slice(&input[*at..*at + N]);
    *at += N;
    Ok(out)
}

fn decode_record(input: &[u8]) -> Result<(BandRecord, usize), CodecError> {
    let mut at = 0usize;
    let hash = u128::from_le_bytes(band_take(input, &mut at)?);
    let key_len = u32::from_le_bytes(band_take(input, &mut at)?) as usize;
    if input.len() < at + key_len {
        return Err(CodecError::Truncated {
            expected: at + key_len,
            available: input.len(),
        });
    }
    let key = Key::from(input[at..at + key_len].to_vec());
    at += key_len;
    let repr = band_take::<1>(input, &mut at)?[0];
    if repr != REPR_INLINE && repr != REPR_CHUNKED && repr != REPR_FABRIC {
        return Err(CodecError::InvalidTag {
            kind: "band representation",
            tag: repr,
        });
    }
    let value_tag = band_take::<1>(input, &mut at)?[0];
    let version =
        kivi_state::ObjectVersion::from_u64(u64::from_le_bytes(band_take(input, &mut at)?));
    let (expiry, used) = kivi_types::Expiry::decode(&input[at..])?;
    at += used;
    // Chunked roots decode to references: dependency validation proves
    // their manifests and chunks before recovery serves them (§30).
    if repr == REPR_CHUNKED {
        if value_tag != TYPE_BYTES {
            return Err(CodecError::InvalidTag {
                kind: "chunked band value type",
                tag: value_tag,
            });
        }
        let raw = band_take::<32>(input, &mut at)?;
        let logical_len = u64::from_le_bytes(band_take(input, &mut at)?);
        let value = LogicalValue::Chunked(kivi_state::ChunkedRef {
            manifest: kivi_types::ManifestId::from_bytes(raw),
            logical_len,
        });
        return Ok((
            BandRecord {
                hash,
                key,
                object: StoredObject::restore(value, version, expiry),
            },
            at,
        ));
    }
    // Fabric roots decode to references: recovery re-links the fabric id
    // and proves version plus integrity before serving.
    if repr == REPR_FABRIC {
        if value_tag != TYPE_BYTES {
            return Err(CodecError::InvalidTag {
                kind: "fabric band value type",
                tag: value_tag,
            });
        }
        let id = u64::from_le_bytes(band_take(input, &mut at)?);
        let logical_len = u64::from_le_bytes(band_take(input, &mut at)?);
        let fabric_version = u64::from_le_bytes(band_take(input, &mut at)?);
        let value = LogicalValue::Fabric(kivi_state::FabricRef {
            id,
            logical_len,
            version: fabric_version,
        });
        return Ok((
            BandRecord {
                hash,
                key,
                object: StoredObject::restore(value, version, expiry),
            },
            at,
        ));
    }
    let (value, used) = decode_band_value(input, at, value_tag)?;
    at = used;
    Ok((
        BandRecord {
            hash,
            key,
            object: StoredObject::restore(value, version, expiry),
        },
        at,
    ))
}

/// Decodes one inline band value by its type tag, verifying structural
/// invariants (bounded rights, semaphore capacity) at the boundary: a
/// checkpoint carrying a violated invariant fails closed here, never
/// after restore.
fn decode_band_value(
    input: &[u8],
    mut at: usize,
    value_tag: u8,
) -> Result<(LogicalValue, usize), CodecError> {
    let value = match value_tag {
        TYPE_BYTES => {
            let len = u32::from_le_bytes(band_take(input, &mut at)?) as usize;
            if input.len() < at + len {
                return Err(CodecError::Truncated {
                    expected: at + len,
                    available: input.len(),
                });
            }
            let value = LogicalValue::Bytes(bytes::Bytes::copy_from_slice(&input[at..at + len]));
            at += len;
            value
        }
        TYPE_COUNTER => LogicalValue::StrictCounter(i64::from_le_bytes(band_take(input, &mut at)?)),
        TYPE_COMMUTATIVE => {
            LogicalValue::CommutativeCounter(i64::from_le_bytes(band_take(input, &mut at)?))
        }
        TYPE_BOUNDED => {
            let (state, used) =
                kivi_state::BoundedCounterState::decode(&input[at..]).map_err(|_| {
                    CodecError::InvalidTag {
                        kind: "band bounded counter",
                        tag: TYPE_BOUNDED,
                    }
                })?;
            if !state.invariant_holds() {
                return Err(CodecError::InvalidTag {
                    kind: "band bounded counter",
                    tag: TYPE_BOUNDED,
                });
            }
            at += used;
            LogicalValue::BoundedCounter(state)
        }
        TYPE_SEMAPHORE => {
            let (state, used) = kivi_state::SemaphoreState::decode(&input[at..]).map_err(|_| {
                CodecError::InvalidTag {
                    kind: "band semaphore",
                    tag: TYPE_SEMAPHORE,
                }
            })?;
            if !state.invariant_holds() {
                return Err(CodecError::InvalidTag {
                    kind: "band semaphore",
                    tag: TYPE_SEMAPHORE,
                });
            }
            at += used;
            LogicalValue::Semaphore(state)
        }
        TYPE_LEASE => {
            let (state, used) = kivi_state::LeaseState::decode(&input[at..]).map_err(|_| {
                CodecError::InvalidTag {
                    kind: "band lease",
                    tag: TYPE_LEASE,
                }
            })?;
            at += used;
            LogicalValue::Lease(state)
        }
        TYPE_STREAM_SHARD => {
            let (shard, used) =
                kivi_state::StreamShardState::decode(&input[at..]).map_err(|_| {
                    CodecError::InvalidTag {
                        kind: "band stream shard",
                        tag: TYPE_STREAM_SHARD,
                    }
                })?;
            at += used;
            LogicalValue::StreamShard(shard)
        }
        other => {
            return Err(CodecError::InvalidTag {
                kind: "band value type",
                tag: other,
            });
        }
    };
    Ok((value, at))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_state::ObjectVersion;
    use kivi_types::{Expiry, NamespaceId};

    const NS: NamespaceId = NamespaceId::from_u64(3);

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
    fn band_round_trips_sorted_with_content_identity() {
        let built = build_band(
            TabletId::from_u64(1),
            41,
            7,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            CompressionCodecId::NONE,
            vec![record("b", 2), record("a", 1)],
        )
        .expect("builds");
        assert_eq!(built.records, 2);
        let loaded = load_band(
            TabletId::from_u64(1),
            7,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            built.hash,
            &built.bytes,
        )
        .expect("loads");
        assert_eq!(loaded.cut, 41);
        // Sorted by key regardless of input order.
        assert_eq!(loaded.records[0].key, Key::from("a"));
        assert_eq!(loaded.records[1].key, Key::from("b"));
        let _ = NS;
    }

    #[test]
    fn chunked_roots_round_trip_as_manifest_references() {
        // Chunked roots serialize as manifest id + length (§29): bulk
        // bytes never enter the band. The loaded root must equal the
        // stored one exactly (same manifest, same length, same version).
        let chunked = BandRecord {
            hash: 0xABCD,
            key: Key::from("big"),
            object: StoredObject::restore(
                LogicalValue::Chunked(kivi_state::ChunkedRef {
                    manifest: kivi_types::ManifestId::from_bytes([0xC4; 32]),
                    logical_len: 3_000_000,
                }),
                ObjectVersion::FIRST,
                Expiry::NEVER,
            ),
        };
        let built = build_band(
            TabletId::from_u64(1),
            12,
            2,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            CompressionCodecId::NONE,
            vec![chunked, record("a", 1)],
        )
        .expect("builds");
        assert_eq!(built.records, 2);
        let loaded = load_band(
            TabletId::from_u64(1),
            2,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            built.hash,
            &built.bytes,
        )
        .expect("loads");
        assert_eq!(loaded.records.len(), 2);
        let big = loaded
            .records
            .iter()
            .find(|r| r.key == Key::from("big"))
            .expect("big");
        assert_eq!(
            big.object.value(),
            &LogicalValue::Chunked(kivi_state::ChunkedRef {
                manifest: kivi_types::ManifestId::from_bytes([0xC4; 32]),
                logical_len: 3_000_000,
            })
        );
        // A chunked counter is nonsense: forging the type byte under a
        // chunked repr must fail decoding (repr and type must agree).
        // Layout is hash u128 + key_len u32 + key + repr + type, so the
        // type byte sits at a computable offset for a known key.
        let solo = BandRecord {
            hash: 0xABCD,
            key: Key::from("k"),
            object: StoredObject::restore(
                LogicalValue::Chunked(kivi_state::ChunkedRef {
                    manifest: kivi_types::ManifestId::from_bytes([0xC4; 32]),
                    logical_len: 7,
                }),
                ObjectVersion::FIRST,
                Expiry::NEVER,
            ),
        };
        let mut forged = Vec::new();
        encode_record(&solo, &mut forged).expect("encodes");
        let type_at = 16 + 4 + 1 + 1;
        assert_eq!(forged[type_at], TYPE_BYTES);
        forged[type_at] = TYPE_COUNTER;
        assert!(matches!(
            decode_record(&forged),
            Err(CodecError::InvalidTag {
                kind: "chunked band value type",
                ..
            })
        ));
    }

    #[test]
    fn empty_band_is_valid_explicit_state() {
        let built = build_band(
            TabletId::from_u64(1),
            9,
            0,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            CompressionCodecId::NONE,
            Vec::new(),
        )
        .expect("empty band builds");
        assert_eq!(built.records, 0);
        let loaded = load_band(
            TabletId::from_u64(1),
            0,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            built.hash,
            &built.bytes,
        )
        .expect("empty band loads");
        assert!(loaded.records.is_empty());
    }

    #[test]
    fn duplicate_keys_fail_the_build() {
        let err = build_band(
            TabletId::from_u64(1),
            1,
            0,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            CompressionCodecId::NONE,
            vec![record("a", 1), record("a", 2)],
        )
        .expect_err("duplicates rejected");
        assert!(matches!(err, CheckpointError::Format { .. }));
    }

    #[test]
    fn every_byte_flip_is_detected() {
        let built = build_band(
            TabletId::from_u64(1),
            1,
            3,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            CompressionCodecId::NONE,
            vec![record("k", 5)],
        )
        .expect("builds");
        // Flipping any single byte anywhere must fail loading (header
        // CRC, body CRC, content hash, footer CRC, or count/order checks).
        for index in 0..built.bytes.len() {
            let mut damaged = built.bytes.clone();
            damaged[index] ^= 0xFF;
            load_band(
                TabletId::from_u64(1),
                3,
                crate::layout::BandLayoutId::HASH_PREFIX_8,
                built.hash,
                &damaged,
            )
            .expect_err(&format!("byte {index} flip must be detected"));
        }
    }

    #[test]
    fn empty_bands_of_different_ids_have_distinct_identities() {
        // Regression: content identity covers header identity, not just
        // the body — empty bands share an empty body but must never share
        // a filename.
        let first = build_band(
            TabletId::from_u64(1),
            1,
            0,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            CompressionCodecId::NONE,
            Vec::new(),
        )
        .expect("builds");
        let second = build_band(
            TabletId::from_u64(1),
            1,
            1,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            CompressionCodecId::NONE,
            Vec::new(),
        )
        .expect("builds");
        assert_ne!(first.hash, second.hash);
        load_band(
            TabletId::from_u64(1),
            0,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            first.hash,
            &first.bytes,
        )
        .expect("first loads");
        load_band(
            TabletId::from_u64(1),
            1,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            second.hash,
            &second.bytes,
        )
        .expect("second loads");
    }

    #[test]
    fn wrong_tablet_or_hash_fails_loading() {
        let built = build_band(
            TabletId::from_u64(1),
            1,
            3,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            CompressionCodecId::NONE,
            vec![record("k", 5)],
        )
        .expect("builds");
        assert!(
            load_band(
                TabletId::from_u64(2),
                3,
                crate::layout::BandLayoutId::HASH_PREFIX_8,
                built.hash,
                &built.bytes,
            )
            .is_err()
        );
        let other = ArtifactHash::of(b"something else");
        assert!(
            load_band(
                TabletId::from_u64(1),
                3,
                crate::layout::BandLayoutId::HASH_PREFIX_8,
                other,
                &built.bytes,
            )
            .is_err()
        );
    }

    fn bytes_record(key: &str, value: &[u8]) -> BandRecord {
        BandRecord {
            hash: 0x1234,
            key: Key::from(key),
            object: StoredObject::restore(
                LogicalValue::Bytes(bytes::Bytes::copy_from_slice(value)),
                ObjectVersion::FIRST,
                Expiry::NEVER,
            ),
        }
    }

    #[test]
    fn lz4_round_trips_and_shrinks_repetitive_bodies() {
        // Representative compressible body (not zeros): repeated phrases.
        let value = b"the quick brown fox jumps over the lazy dog. ".repeat(64);
        let records = vec![
            bytes_record("a", &value),
            bytes_record("b", &value),
            record("n", 42),
        ];
        let none = build_band(
            TabletId::from_u64(1),
            11,
            5,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            CompressionCodecId::NONE,
            records.clone(),
        )
        .expect("none builds");
        let lz4 = build_band(
            TabletId::from_u64(1),
            11,
            5,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            CompressionCodecId::LZ4,
            records,
        )
        .expect("lz4 builds");
        assert_eq!(lz4.codec, CompressionCodecId::LZ4);
        assert!(
            lz4.bytes.len() < none.bytes.len(),
            "lz4 {} should shrink repetitive none {}",
            lz4.bytes.len(),
            none.bytes.len()
        );
        // Identities differ by codec (headers differ), both load to the
        // same logical records.
        assert_ne!(lz4.hash, none.hash);
        let loaded = load_band(
            TabletId::from_u64(1),
            5,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            lz4.hash,
            &lz4.bytes,
        )
        .expect("lz4 loads");
        assert_eq!(loaded.records.len(), 3);
        assert_eq!(loaded.records[0].key, Key::from("a"));
        // Codec byte is explicit in the header.
        assert_eq!(lz4.bytes[48], CompressionCodecId::LZ4.as_u8());
        assert_eq!(none.bytes[48], CompressionCodecId::NONE.as_u8());
    }

    #[test]
    fn lz4_empty_band_round_trips() {
        let built = build_band(
            TabletId::from_u64(1),
            3,
            2,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            CompressionCodecId::LZ4,
            Vec::new(),
        )
        .expect("empty lz4 builds");
        assert_eq!(built.codec, CompressionCodecId::LZ4);
        let loaded = load_band(
            TabletId::from_u64(1),
            2,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            built.hash,
            &built.bytes,
        )
        .expect("empty lz4 loads");
        assert!(loaded.records.is_empty());
    }

    #[test]
    fn lz4_corruption_fails_loudly() {
        let value = b"compressible compressible compressible ".repeat(32);
        let built = build_band(
            TabletId::from_u64(1),
            1,
            3,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            CompressionCodecId::LZ4,
            vec![bytes_record("k", &value)],
        )
        .expect("builds");
        // Flip bytes across the stored body (past the 60-byte header):
        // body CRC, content hash, or decompression must fail — never a
        // silent wrong record.
        let body_start = BAND_HEADER_LEN;
        let body_end = built.bytes.len() - BAND_FOOTER_LEN;
        assert!(body_end > body_start, "lz4 body exists");
        for index in [
            body_start,
            usize::midpoint(body_start, body_end),
            body_end - 1,
        ] {
            let mut damaged = built.bytes.clone();
            damaged[index] ^= 0xFF;
            load_band(
                TabletId::from_u64(1),
                3,
                crate::layout::BandLayoutId::HASH_PREFIX_8,
                built.hash,
                &damaged,
            )
            .expect_err(&format!("lz4 byte {index} flip must be detected"));
        }
    }

    #[test]
    fn unknown_codec_is_rejected_explicitly() {
        let built = build_band(
            TabletId::from_u64(1),
            1,
            3,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            CompressionCodecId::NONE,
            vec![record("k", 5)],
        )
        .expect("builds");
        // Rewrite the codec byte to an unknown assignment and fix the
        // header CRC so the loader reaches the codec gate (not a CRC fail).
        let mut forged = built.bytes.clone();
        forged[48] = 0x7F;
        let crc = kivi_codec::integrity::crc32c_checksum(&forged[..56]);
        forged[56..60].copy_from_slice(&crc.to_le_bytes());
        let err = load_band(
            TabletId::from_u64(1),
            3,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            built.hash,
            &forged,
        )
        .expect_err("unknown codec rejected");
        assert!(
            matches!(err, CheckpointError::Unsupported { .. }),
            "explicit unsupported, got {err:?}"
        );
        // Builds with unknown codecs fail the same way.
        let err = build_band(
            TabletId::from_u64(1),
            1,
            3,
            crate::layout::BandLayoutId::HASH_PREFIX_8,
            CompressionCodecId::from_u8(0x7F),
            vec![record("k", 5)],
        )
        .expect_err("unknown build codec rejected");
        assert!(matches!(err, CheckpointError::Unsupported { .. }));
    }
}
