//! Append-only immutable chunk packs: the production chunk container.
//!
//! One pack file holds back-to-back self-validating records (chunks and
//! manifests) under one lane directory:
//!
//! ```text
//! chunks/
//!   lane-0000/
//!     00000000000000000001.pack
//!     00000000000000000002.pack
//! ```
//!
//! Layout mirrors the WAL's conservative philosophy with pack-appropriate
//! rules: every record carries framing, lengths, CRC32C integrity, and a
//! content identity, all Kivi-owned little-endian encodings (no serde,
//! bincode, or rkyv anywhere near durable bytes). Differences from the WAL
//! are deliberate: files are self-describing (lane/sequence live in the
//! header, so GC-created gaps need no contiguity repair), and only the
//! *active* (highest-sequence) pack may end mid-record — a torn tail there
//! truncates to the last complete boundary, while any damage in a sealed
//! pack fails loudly.
//!
//! ```text
//! pack header (64 bytes, fixed):
//!   magic u32 "KVCP", major u16 = 1, minor u16 = 0,
//!   lane u16, reserved u16,
//!   seq u64, created_wall_micros u64, target_bytes u64,
//!   reserved [24],
//!   header_crc u32 over bytes [..60]
//!
//! chunk record:
//!   magic u32 "KVCC", version u16 = 1, codec u8, flags u8 (0),
//!   chunk_id [32], logical_len u64, stored_len u64,
//!   header_crc u32 over the preceding 56 bytes,
//!   body[stored_len],
//!   body_crc u32 over the body
//!
//! manifest record:
//!   magic u32 "KVMP", version u16 = 1, codec u8, flags u8 (0),
//!   manifest_id [32], body_len u64,
//!   header_crc u32 over the preceding 48 bytes,
//!   body[body_len] (canonical manifest bytes),
//!   body_crc u32 over the body
//! ```

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use kivi_types::{ChunkId, ManifestId, SecurityDomainId};

use crate::codec::ChunkCodecId;
use crate::error::ChunkError;

/// Magic word: ASCII `"KVCP"` read as a little-endian `u32`.
pub const PACK_MAGIC: u32 = 0x5043_564B;
/// Magic word: ASCII `"KVCC"` read as a little-endian `u32`.
pub const CHUNK_RECORD_MAGIC: u32 = 0x4343_564B;
/// Magic word: ASCII `"KVMP"` read as a little-endian `u32`.
pub const MANIFEST_RECORD_MAGIC: u32 = 0x504D_564B;
/// Pack major version.
pub const PACK_MAJOR: u16 = 1;
/// Pack minor version.
pub const PACK_MINOR: u16 = 0;
/// Record format version minted in this stage.
pub const RECORD_VERSION: u16 = 1;
/// Encoded pack header length in bytes.
pub const PACK_HEADER_LEN: usize = 64;
/// Encoded chunk record header length in bytes.
pub const CHUNK_HEADER_LEN: usize = 60;
/// Encoded manifest record header length in bytes.
pub const MANIFEST_HEADER_LEN: usize = 52;
/// Body CRC length in bytes.
pub const BODY_CRC_LEN: usize = 4;

/// Slack a pack file may exceed its rotation target by: one maximum record
/// plus framing. Larger files are a rotation bug, reported before reading
/// them wholesale.
pub const PACK_SIZE_SLACK: u64 = 16 * 1024 * 1024;

/// Formats a lane directory name (`lane-0007`).
#[must_use]
pub fn lane_dir_name(lane: u16) -> String {
    format!("lane-{lane:04}")
}

/// Formats a pack file name (`00000000000000000007.pack`).
#[must_use]
pub fn pack_file_name(seq: u64) -> String {
    format!("{seq:020}.pack")
}

/// Parses a pack file name back to its sequence, if well-formed.
#[must_use]
pub fn parse_pack_name(name: &str) -> Option<u64> {
    name.strip_suffix(".pack")?.parse::<u64>().ok()
}

/// Which immutable record a pack slot holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecordKind {
    /// Logical chunk bytes addressed by [`ChunkId`].
    Chunk,
    /// Canonical manifest bytes addressed by [`ManifestId`].
    Manifest,
}

/// Content address of one pack record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecordId {
    /// Chunk address.
    Chunk(ChunkId),
    /// Manifest address.
    Manifest(ManifestId),
}

impl RecordId {
    /// Which record holds this address.
    #[must_use]
    pub const fn kind(self) -> RecordKind {
        match self {
            Self::Chunk(_) => RecordKind::Chunk,
            Self::Manifest(_) => RecordKind::Manifest,
        }
    }
}

/// One verified pack record: identity, lengths, and physical position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScannedRecord {
    /// Content address.
    pub id: RecordId,
    /// Chunk logical bytes (0 for manifests).
    pub logical_len: u64,
    /// Stored body bytes (chunk body or canonical manifest).
    pub stored_len: u64,
    /// Byte offset of the record start within the file.
    pub offset: u64,
    /// Full record bytes (header + body + CRC).
    pub total_len: u64,
}

/// Decoded record header before body verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RecordHeader {
    kind: RecordKind,
    codec: ChunkCodecId,
    id: [u8; 32],
    logical_len: u64,
    stored_len: u64,
    header_len: usize,
}

/// Encodes one chunk record: framed header, body, trailing CRC.
/// With `NONE` the body is the logical bytes verbatim.
#[must_use]
pub fn encode_chunk_record(id: ChunkId, logical_len: u64, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(CHUNK_HEADER_LEN + body.len() + BODY_CRC_LEN);
    let mut header = [0u8; CHUNK_HEADER_LEN];
    header[0..4].copy_from_slice(&CHUNK_RECORD_MAGIC.to_le_bytes());
    header[4..6].copy_from_slice(&RECORD_VERSION.to_le_bytes());
    header[6] = ChunkCodecId::NONE.as_u8();
    header[7] = 0;
    header[8..40].copy_from_slice(id.as_bytes());
    header[40..48].copy_from_slice(&logical_len.to_le_bytes());
    header[48..56].copy_from_slice(&(body.len() as u64).to_le_bytes());
    let crc = kivi_codec::integrity::crc32c_checksum(&header[..56]);
    header[56..60].copy_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(body);
    let body_crc = kivi_codec::integrity::crc32c_checksum(body);
    out.extend_from_slice(&body_crc.to_le_bytes());
    out
}

/// Encodes one manifest record around canonical manifest bytes.
#[must_use]
pub fn encode_manifest_record(id: ManifestId, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(MANIFEST_HEADER_LEN + body.len() + BODY_CRC_LEN);
    let mut header = [0u8; MANIFEST_HEADER_LEN];
    header[0..4].copy_from_slice(&MANIFEST_RECORD_MAGIC.to_le_bytes());
    header[4..6].copy_from_slice(&RECORD_VERSION.to_le_bytes());
    header[6] = ChunkCodecId::NONE.as_u8();
    header[7] = 0;
    header[8..40].copy_from_slice(id.as_bytes());
    header[40..48].copy_from_slice(&(body.len() as u64).to_le_bytes());
    let crc = kivi_codec::integrity::crc32c_checksum(&header[..48]);
    header[48..52].copy_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(body);
    let body_crc = kivi_codec::integrity::crc32c_checksum(body);
    out.extend_from_slice(&body_crc.to_le_bytes());
    out
}

/// Encodes a pack header for (`lane`, `seq`).
#[must_use]
pub fn encode_pack_header(lane: u16, seq: u64, wall_micros: u64, target_bytes: u64) -> [u8; 64] {
    let mut header = [0u8; PACK_HEADER_LEN];
    header[0..4].copy_from_slice(&PACK_MAGIC.to_le_bytes());
    header[4..6].copy_from_slice(&PACK_MAJOR.to_le_bytes());
    header[6..8].copy_from_slice(&PACK_MINOR.to_le_bytes());
    header[8..10].copy_from_slice(&lane.to_le_bytes());
    header[10..12].copy_from_slice(&0u16.to_le_bytes());
    header[12..20].copy_from_slice(&seq.to_le_bytes());
    header[20..28].copy_from_slice(&wall_micros.to_le_bytes());
    header[28..36].copy_from_slice(&target_bytes.to_le_bytes());
    // bytes [36..60] stay zero (reserved).
    let crc = kivi_codec::integrity::crc32c_checksum(&header[..60]);
    header[60..64].copy_from_slice(&crc.to_le_bytes());
    header
}

/// Validates a pack header against its expected lane and sequence.
pub(crate) fn decode_pack_header(
    bytes: &[u8; PACK_HEADER_LEN],
    lane: u16,
    seq: u64,
) -> Result<(), ChunkError> {
    let corrupt = |detail: &str| ChunkError::CorruptChunk {
        id: ChunkId::ZERO,
        detail: format!("pack {seq} header: {detail}"),
    };
    if u32::from_le_bytes(bytes[0..4].try_into().expect("len 4")) != PACK_MAGIC {
        return Err(corrupt("magic mismatch"));
    }
    if u16::from_le_bytes(bytes[4..6].try_into().expect("len 2")) != PACK_MAJOR
        || u16::from_le_bytes(bytes[6..8].try_into().expect("len 2")) != PACK_MINOR
    {
        return Err(ChunkError::Unsupported {
            detail: "pack version".to_owned(),
        });
    }
    if u16::from_le_bytes(bytes[8..10].try_into().expect("len 2")) != lane
        || u64::from_le_bytes(bytes[12..20].try_into().expect("len 8")) != seq
    {
        return Err(corrupt("lane or sequence mismatch (misdirected file)"));
    }
    if bytes[10..12] != [0, 0] || bytes[36..60] != [0; 24] {
        return Err(corrupt("reserved bytes nonzero"));
    }
    let expect = u32::from_le_bytes(bytes[60..64].try_into().expect("len 4"));
    if kivi_codec::integrity::crc32c_checksum(&bytes[..60]) != expect {
        return Err(corrupt("header CRC mismatch"));
    }
    Ok(())
}

/// Decodes and validates one record header: framing, versions, codec
/// support, reserved bytes, header CRC, and allocation-guarding length
/// caps. The body itself is verified by the caller.
fn decode_record_header(input: &[u8]) -> Result<RecordHeader, ChunkError> {
    let corrupt = |detail: String| ChunkError::CorruptChunk {
        id: ChunkId::ZERO,
        detail,
    };
    if input.len() < 8 {
        return Err(corrupt("record header truncated".to_owned()));
    }
    let magic = u32::from_le_bytes(input[0..4].try_into().expect("len 4"));
    let (kind, header_len) = match magic {
        CHUNK_RECORD_MAGIC => (RecordKind::Chunk, CHUNK_HEADER_LEN),
        MANIFEST_RECORD_MAGIC => (RecordKind::Manifest, MANIFEST_HEADER_LEN),
        _ => return Err(corrupt("record magic mismatch".to_owned())),
    };
    if input.len() < header_len {
        return Err(corrupt("record header truncated".to_owned()));
    }
    let version = u16::from_le_bytes(input[4..6].try_into().expect("len 2"));
    if version != RECORD_VERSION {
        return Err(ChunkError::Unsupported {
            detail: format!("pack record version {version}"),
        });
    }
    let codec = ChunkCodecId::from_u8(input[6]);
    if !codec.is_supported() {
        return Err(ChunkError::Unsupported {
            detail: format!("pack record codec {codec}"),
        });
    }
    if input[7] != 0 {
        return Err(corrupt("record flags nonzero".to_owned()));
    }
    let mut id = [0u8; 32];
    id.copy_from_slice(&input[8..40]);
    let (logical_len, stored_len) = match kind {
        RecordKind::Chunk => (
            u64::from_le_bytes(input[40..48].try_into().expect("len 8")),
            u64::from_le_bytes(input[48..56].try_into().expect("len 8")),
        ),
        RecordKind::Manifest => (
            0,
            u64::from_le_bytes(input[40..48].try_into().expect("len 8")),
        ),
    };
    if stored_len > crate::policy::MAX_STORED_BODY {
        return Err(ChunkError::TooLarge {
            len: stored_len,
            max: crate::policy::MAX_STORED_BODY,
            context: "pack record body",
        });
    }
    let crc_at = header_len - 4;
    let expect = u32::from_le_bytes(input[crc_at..header_len].try_into().expect("len 4"));
    if kivi_codec::integrity::crc32c_checksum(&input[..crc_at]) != expect {
        return Err(corrupt("record header CRC mismatch".to_owned()));
    }
    Ok(RecordHeader {
        kind,
        codec,
        id,
        logical_len,
        stored_len,
        header_len,
    })
}

/// Reads exactly `buf.len()` bytes at `offset` (`pread` via seek+read).
fn read_at(file: &mut File, path: &Path, offset: u64, buf: &mut [u8]) -> Result<(), ChunkError> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| ChunkError::io("seek pack", path, &error))?;
    file.read_exact(buf)
        .map_err(|error| ChunkError::io("read pack", path, &error))?;
    Ok(())
}

/// Reads and fully verifies the record at `offset`: framing, CRCs, and the
/// content identity recomputed over the body under `domain`. Returns the
/// record summary plus the body bytes.
///
/// Short reads surface as [`ChunkError::Io`] (unexpected EOF mid-record);
/// callers that can legally end mid-record (active-tail scan) check
/// lengths first and never call this past the end.
fn read_verify_record(
    file: &mut File,
    path: &Path,
    offset: u64,
    domain: SecurityDomainId,
) -> Result<(ScannedRecord, Vec<u8>), ChunkError> {
    // Peek the magic to size the header read.
    let mut magic = [0u8; 8];
    read_at(file, path, offset, &mut magic)?;
    let (kind, header_len) = match u32::from_le_bytes(magic[0..4].try_into().expect("len 4")) {
        CHUNK_RECORD_MAGIC => (RecordKind::Chunk, CHUNK_HEADER_LEN),
        MANIFEST_RECORD_MAGIC => (RecordKind::Manifest, MANIFEST_HEADER_LEN),
        _ => {
            return Err(ChunkError::CorruptChunk {
                id: ChunkId::ZERO,
                detail: format!("record magic mismatch at offset {offset}"),
            });
        }
    };
    let mut header = vec![0u8; header_len];
    read_at(file, path, offset, &mut header)?;
    let decoded = decode_record_header(&header)?;
    debug_assert_eq!(decoded.kind, kind);
    let body_at = offset + header_len as u64;
    let body_len = usize::try_from(decoded.stored_len).map_err(|_| ChunkError::TooLarge {
        len: decoded.stored_len,
        max: u64::MAX,
        context: "pack record body",
    })?;
    let mut body = vec![0u8; body_len];
    read_at(file, path, body_at, &mut body)?;
    let mut crc = [0u8; 4];
    read_at(file, path, body_at + decoded.stored_len, &mut crc)?;
    if kivi_codec::integrity::crc32c_checksum(&body) != u32::from_le_bytes(crc) {
        return Err(ChunkError::CorruptChunk {
            id: ChunkId::ZERO,
            detail: format!("record body CRC mismatch at offset {offset}"),
        });
    }
    let id = match kind {
        RecordKind::Chunk => {
            let id = ChunkId::from_bytes(decoded.id);
            let recomputed = kivi_codec::integrity::chunk_id(domain, &body);
            if recomputed != id {
                return Err(ChunkError::CorruptChunk {
                    id,
                    detail: format!("chunk content hash mismatch at offset {offset}"),
                });
            }
            // With NONE the stored body is the logical body: lengths agree
            // by construction, and the manifest layer rechecks chunking.
            if body.len() as u64 != decoded.logical_len {
                return Err(ChunkError::CorruptChunk {
                    id,
                    detail: format!(
                        "chunk stored length disagrees with logical length at offset {offset}"
                    ),
                });
            }
            RecordId::Chunk(id)
        }
        RecordKind::Manifest => {
            let id = ManifestId::from_bytes(decoded.id);
            let manifest =
                crate::manifest::verify_manifest(id, &body).map_err(|error| match error {
                    ChunkError::CorruptManifest { detail, .. } => ChunkError::CorruptManifest {
                        id,
                        detail: format!("pack manifest at offset {offset}: {detail}"),
                    },
                    other => other,
                })?;
            if manifest.domain != domain {
                return Err(ChunkError::CorruptManifest {
                    id,
                    detail: format!(
                        "pack manifest names a foreign security domain at offset {offset}"
                    ),
                });
            }
            RecordId::Manifest(id)
        }
    };
    Ok((
        ScannedRecord {
            id,
            logical_len: decoded.logical_len,
            stored_len: decoded.stored_len,
            offset,
            total_len: header_len as u64 + decoded.stored_len + BODY_CRC_LEN as u64,
        },
        body,
    ))
}

/// One scanned pack: verified records plus torn-tail accounting.
#[derive(Debug)]
pub struct PackScan {
    /// Verified records in file order.
    pub records: Vec<ScannedRecord>,
    /// Bytes to keep (`file_len` when clean).
    pub kept_len: u64,
    /// Bytes truncated from a torn active tail (0 when clean).
    pub truncated_bytes: u64,
}

/// Scans one pack file: validates the header, then every record.
/// `active` selects torn-tail semantics: a file that ends mid-record
/// truncates to the last complete boundary (repaired by the store);
/// the same damage in a sealed pack is corruption.
///
/// `file_len` bounds all reads; short files fail before any record trust.
///
/// # Errors
///
/// Returns [`ChunkError`] on header mismatch, corrupt records (or torn
/// tails in sealed packs), oversize declarations, or I/O failure.
///
/// # Panics
///
/// Never panics on pack bytes: every length is gated before slicing.
pub fn scan_pack(
    path: &Path,
    lane: u16,
    seq: u64,
    domain: SecurityDomainId,
    file_len: u64,
    active: bool,
    file: &mut File,
) -> Result<PackScan, ChunkError> {
    let short = |what: &str| ChunkError::CorruptChunk {
        id: ChunkId::ZERO,
        detail: format!("pack {seq} {what}"),
    };
    if file_len < PACK_HEADER_LEN as u64 {
        if active {
            // A headerless active file proves nothing was published: the
            // store removes it and falls back, never truncating blindly.
            return Ok(PackScan {
                records: Vec::new(),
                kept_len: 0,
                truncated_bytes: file_len,
            });
        }
        return Err(short("shorter than its header"));
    }
    let mut header = [0u8; PACK_HEADER_LEN];
    read_at(file, path, 0, &mut header)?;
    decode_pack_header(&header, lane, seq)?;
    let mut records = Vec::new();
    let mut pos = PACK_HEADER_LEN as u64;
    while pos < file_len {
        // Length-gate every read: only complete records verify. An
        // incomplete one is a torn tail (active) or corruption (sealed).
        let remaining = file_len - pos;
        if remaining < 8 {
            return torn_or_corrupt(active, records, pos, file_len);
        }
        let mut magic = [0u8; 8];
        read_at(file, path, pos, &mut magic)?;
        // Unparseable tail bytes are a torn write when the pack is still
        // active (a torn sector leaves garbage, not a clean prefix — see
        // the module docs) and damage otherwise. Truncating a genuinely
        // corrupt (rather than torn) last record still fails loudly later:
        // the WAL-tail and checkpoint dependency gates prove every
        // committed root's chunks before serving.
        let header_len = match u32::from_le_bytes(magic[0..4].try_into().expect("len 4")) {
            CHUNK_RECORD_MAGIC => CHUNK_HEADER_LEN,
            MANIFEST_RECORD_MAGIC => MANIFEST_HEADER_LEN,
            _ => return torn_or_corrupt(active, records, pos, file_len),
        };
        if remaining < header_len as u64 {
            return torn_or_corrupt(active, records, pos, file_len);
        }
        let mut header = vec![0u8; header_len];
        read_at(file, path, pos, &mut header)?;
        // decode_record_header validates framing/CRC/caps; a failure here
        // on complete bytes is corruption either way (a torn write cannot
        // forge a valid header CRC... and an invalid one is damage).
        let decoded = match decode_record_header(&header) {
            Ok(decoded) => decoded,
            Err(error) => {
                // A torn tail can land mid-header with arbitrary bytes;
                // only the active pack may claim that reading.
                if active {
                    return Ok(PackScan {
                        records,
                        kept_len: pos,
                        truncated_bytes: file_len - pos,
                    });
                }
                return Err(error);
            }
        };
        let end = pos + header_len as u64 + decoded.stored_len + BODY_CRC_LEN as u64;
        if file_len < end {
            return torn_or_corrupt(active, records, pos, file_len);
        }
        let (record, _) = read_verify_record(file, path, pos, domain)?;
        records.push(record);
        pos = end;
    }
    Ok(PackScan {
        records,
        kept_len: file_len,
        truncated_bytes: 0,
    })
}

/// Torn-tail accounting for an incomplete record: truncate (active) or fail
/// (sealed). Already-verified records are preserved either way; only the
/// incomplete tail is dropped or condemned. Corruption elsewhere never
/// reaches here.
fn torn_or_corrupt(
    active: bool,
    records: Vec<ScannedRecord>,
    pos: u64,
    file_len: u64,
) -> Result<PackScan, ChunkError> {
    if active {
        return Ok(PackScan {
            records,
            kept_len: pos,
            truncated_bytes: file_len - pos,
        });
    }
    Err(ChunkError::CorruptChunk {
        id: ChunkId::ZERO,
        detail: format!("sealed pack ends mid-record at offset {pos}"),
    })
}

/// Appends one fully framed record to the active pack file.
///
/// # Errors
///
/// Returns [`ChunkError`] when the write fails; the file position past a
/// partial write is the caller's torn tail to repair, never trusted.
pub fn append_record(file: &mut File, path: &Path, record: &[u8]) -> Result<(), ChunkError> {
    file.write_all(record)
        .map_err(|error| ChunkError::io("append pack record", path, &error))
}

/// Lists pack sequences present in a lane directory, oldest first.
/// Non-pack files are ignored (foreign files are never touched).
///
/// # Errors
///
/// Returns [`ChunkError`] when the directory itself is unreadable.
pub fn list_packs(dir: &Path) -> Result<Vec<u64>, ChunkError> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(ChunkError::io("list chunk lane", dir, &error)),
    };
    let mut seqs = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(seq) = parse_pack_name(&name) {
            seqs.push(seq);
        }
    }
    seqs.sort_unstable();
    Ok(seqs)
}

/// Verifies one record in place without trusting the index: re-reads the
/// bytes at `location` and runs full verification for `expected`.
/// Cold-read path shared by the store and the engine's async lane.
/// Short reads surface as I/O errors; callers that can legally end
/// mid-record (active-tail scan) check lengths first and never call this
/// past the end.
///
/// # Errors
///
/// Returns [`ChunkError`] on short reads, framing/CRC/content-hash
/// mismatch, or index/location disagreement.
///
/// # Panics
///
/// Never panics on pack bytes: lengths are gated before slicing.
pub fn verify_record_at(
    file: &mut File,
    path: &Path,
    location: &ScannedRecord,
    domain: SecurityDomainId,
    expected: RecordId,
) -> Result<Vec<u8>, ChunkError> {
    if location.id != expected {
        return Err(ChunkError::CorruptChunk {
            id: ChunkId::ZERO,
            detail: "index location disagrees with requested address".to_owned(),
        });
    }
    let (record, body) = read_verify_record(file, path, location.offset, domain)?;
    if record.id != expected || record.total_len != location.total_len {
        return Err(ChunkError::CorruptChunk {
            id: ChunkId::ZERO,
            detail: "pack bytes changed under a verified location".to_owned(),
        });
    }
    Ok(body)
}

impl ScannedRecord {
    /// Byte range of the full record within its file.
    #[must_use]
    pub const fn file_range(self) -> (u64, u64) {
        (self.offset, self.offset + self.total_len)
    }
}

/// Opens a pack file path for a lane directory and sequence.
#[must_use]
pub fn pack_path(dir: &Path, seq: u64) -> PathBuf {
    dir.join(pack_file_name(seq))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_codec::integrity::chunk_id;
    use kivi_types::SecurityDomainId;

    const DOMAIN: SecurityDomainId = SecurityDomainId::from_u64(7);

    fn chunk_bytes(byte: u8, len: usize) -> Vec<u8> {
        vec![byte; len]
    }

    #[test]
    fn chunk_records_round_trip_with_identity() {
        let body = chunk_bytes(0xAB, 1024);
        let id = chunk_id(DOMAIN, &body);
        let record = encode_chunk_record(id, body.len() as u64, &body);
        // Decode the header straight from the bytes.
        let header = decode_record_header(&record).expect("header decodes");
        assert_eq!(header.kind, RecordKind::Chunk);
        assert_eq!(ChunkId::from_bytes(header.id), id);
        assert_eq!(header.stored_len, 1024);
        // Full verify through a real file.
        let dir = tempfile::tempdir().expect("scratch");
        let path = dir.path().join("t.pack");
        std::fs::write(&path, &record).expect("write");
        let mut file = File::options().read(true).open(&path).expect("open");
        let (scanned, back) = read_verify_record(&mut file, &path, 0, DOMAIN).expect("verifies");
        assert_eq!(scanned.id, RecordId::Chunk(id));
        assert_eq!(back, body);
    }

    #[test]
    fn every_byte_flip_is_detected() {
        let body = chunk_bytes(0xCD, 256);
        let id = chunk_id(DOMAIN, &body);
        let record = encode_chunk_record(id, body.len() as u64, &body);
        for index in 0..record.len() {
            let mut damaged = record.clone();
            damaged[index] ^= 0xFF;
            let header_ok = decode_record_header(&damaged).is_ok();
            // Header damage fails at the header; body damage fails at the
            // body CRC or the content hash. Either way the record never
            // verifies: emulate the read path end to end.
            let dir = tempfile::tempdir().expect("scratch");
            let path = dir.path().join("t.pack");
            std::fs::write(&path, &damaged).expect("write");
            let mut file = File::options().read(true).open(&path).expect("open");
            let verified = read_verify_record(&mut file, &path, 0, DOMAIN).is_ok();
            assert!(
                !verified,
                "byte {index} flip must be detected (header_ok={header_ok})"
            );
        }
    }

    #[test]
    fn wrong_domain_rejects_foreign_chunks() {
        let body = chunk_bytes(0xEF, 128);
        let id = chunk_id(DOMAIN, &body);
        let record = encode_chunk_record(id, body.len() as u64, &body);
        let dir = tempfile::tempdir().expect("scratch");
        let path = dir.path().join("t.pack");
        std::fs::write(&path, &record).expect("write");
        let mut file = File::options().read(true).open(&path).expect("open");
        let foreign = SecurityDomainId::from_u64(8);
        assert!(matches!(
            read_verify_record(&mut file, &path, 0, foreign),
            Err(ChunkError::CorruptChunk { .. })
        ));
    }

    #[test]
    fn oversize_declarations_fail_before_allocation() {
        let mut record = encode_chunk_record(chunk_id(DOMAIN, b"x"), 1, b"x");
        // Forge a huge stored length with a fixed header CRC.
        let huge = crate::policy::MAX_STORED_BODY + 1;
        record[48..56].copy_from_slice(&huge.to_le_bytes());
        let crc = kivi_codec::integrity::crc32c_checksum(&record[..56]);
        record[56..60].copy_from_slice(&crc.to_le_bytes());
        assert!(matches!(
            decode_record_header(&record),
            Err(ChunkError::TooLarge { .. })
        ));
    }

    #[test]
    fn unknown_record_kinds_and_codecs_fail_loudly() {
        let body = chunk_bytes(1, 16);
        let mut record = encode_chunk_record(chunk_id(DOMAIN, &body), 16, &body);
        record[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        assert!(decode_record_header(&record).is_err());
        let mut record = encode_chunk_record(chunk_id(DOMAIN, &body), 16, &body);
        record[6] = 0x7F;
        assert!(matches!(
            decode_record_header(&record),
            Err(ChunkError::Unsupported { .. })
        ));
    }

    #[test]
    fn pack_names_round_trip() {
        assert_eq!(pack_file_name(7), "00000000000000000007.pack");
        assert_eq!(parse_pack_name("00000000000000000007.pack"), Some(7));
        assert_eq!(parse_pack_name("00000000000000000007.wal"), None);
        assert_eq!(lane_dir_name(3), "lane-0003");
    }
}
