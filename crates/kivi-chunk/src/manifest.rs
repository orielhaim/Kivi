//! Canonical chunk manifests: versioned, content-addressed chunk lists.
//!
//! A manifest is the small authoritative root a tablet stores for a large
//! value: total length, chunking parameters, and the ordered chunk
//! sequence. Bulk bytes live in chunk packs (§6); the manifest is what the
//! WAL, checkpoints, and GC roots reference.
//!
//! Identity is computed over the canonical encoding below, so identical
//! chunk sequences under identical parameters share one [`ManifestId`]
//! (manifest dedup, §36) while any byte difference forks the identity.

use kivi_types::{ChunkId, ManifestId, SecurityDomainId};

use crate::codec::ChunkCodecId;
use crate::error::ChunkError;
use crate::policy::{CHUNKING_V1, Chunking, MAX_CHUNKS_PER_MANIFEST};

/// Manifest format version 1. Pinned alongside the canonical layout.
pub const MANIFEST_VERSION: u16 = 1;

/// Magic word: ASCII `"KVCH"` read as a little-endian `u32`. Distinct from
/// every WAL (`KVWL`/`KVWB`/`KVWE`), checkpoint (`KVCB`/`KVCM`), and pack
/// (`KVCP`/`KVCC`/`KVMP`) magic by construction.
pub const MANIFEST_MAGIC: u32 = 0x4843_564B;

/// Fixed canonical header length in bytes:
///
/// ```text
/// magic u32, version u16, chunking_version u16, codec u8, reserved u8,
/// domain u64, total_len u64, chunk_size u64, entry_count u32  (= 38)
/// ```
pub const MANIFEST_HEADER_LEN: usize = 38;

/// One ordered chunk reference: 32-byte id plus its logical length (40).
pub const MANIFEST_ENTRY_LEN: usize = 40;

/// One ordered chunk reference inside a manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkEntry {
    /// Content address of the immutable logical chunk bytes.
    pub id: ChunkId,
    /// Logical bytes this entry contributes.
    pub len: u64,
}

/// A validated chunk manifest: the canonical root of one chunked value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkManifest {
    /// Security domain the ids were derived under (dedup boundary).
    pub domain: SecurityDomainId,
    /// Total logical bytes across all entries.
    pub total_len: u64,
    /// Chunking algorithm version that produced the sequence.
    pub chunking_version: u16,
    /// Logical bytes per full chunk.
    pub chunk_size: u64,
    /// Body codec of the referenced chunks.
    pub codec: ChunkCodecId,
    /// Ordered chunk references covering `[0, total_len)`.
    pub entries: Vec<ChunkEntry>,
}

impl ChunkManifest {
    /// Exact canonical length in bytes for this entry count.
    #[must_use]
    pub const fn canonical_len(entry_count: usize) -> usize {
        MANIFEST_HEADER_LEN + MANIFEST_ENTRY_LEN * entry_count
    }

    /// Appends the canonical encoding to `out`.
    pub fn encode_canonical(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&MANIFEST_MAGIC.to_le_bytes());
        out.extend_from_slice(&MANIFEST_VERSION.to_le_bytes());
        out.extend_from_slice(&self.chunking_version.to_le_bytes());
        out.push(self.codec.as_u8());
        out.push(0);
        out.extend_from_slice(&self.domain.as_u64().to_le_bytes());
        out.extend_from_slice(&self.total_len.to_le_bytes());
        out.extend_from_slice(&self.chunk_size.to_le_bytes());
        let count = u32::try_from(self.entries.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&count.to_le_bytes());
        for entry in &self.entries {
            out.extend_from_slice(entry.id.as_bytes());
            out.extend_from_slice(&entry.len.to_le_bytes());
        }
    }

    /// Canonical bytes plus the [`ManifestId`] over them.
    #[must_use]
    pub fn canonical_with_id(&self, domain: SecurityDomainId) -> (Vec<u8>, ManifestId) {
        let mut bytes = Vec::with_capacity(Self::canonical_len(self.entries.len()));
        self.encode_canonical(&mut bytes);
        let id = kivi_codec::integrity::manifest_id(domain, &bytes);
        (bytes, id)
    }
}

/// Builds a manifest from chunk references, validating structure and
/// returning the canonical identity.
///
/// Validation (never trusted from input):
/// entry count within bound; lengths sum to `total_len`; every non-final
/// entry is exactly `chunk_size`; no zero-length entry (the empty value is
/// the only zero-entry manifest); chunking version and codec supported.
///
/// # Errors
///
/// Returns [`ChunkError`] on any structural or support violation.
pub fn build_manifest(
    domain: SecurityDomainId,
    chunking: Chunking,
    codec: ChunkCodecId,
    total_len: u64,
    entries: Vec<ChunkEntry>,
) -> Result<(ChunkManifest, ManifestId), ChunkError> {
    chunking.validate()?;
    if !codec.is_supported() {
        return Err(ChunkError::Unsupported {
            detail: format!("chunk codec {codec}"),
        });
    }
    if entries.len() > MAX_CHUNKS_PER_MANIFEST {
        return Err(ChunkError::TooLarge {
            len: entries.len() as u64,
            max: MAX_CHUNKS_PER_MANIFEST as u64,
            context: "manifest entries",
        });
    }
    validate_entries(chunking, total_len, &entries)?;
    let manifest = ChunkManifest {
        domain,
        total_len,
        chunking_version: chunking.version,
        chunk_size: chunking.chunk_size as u64,
        codec,
        entries,
    };
    let mut canonical = Vec::with_capacity(ChunkManifest::canonical_len(manifest.entries.len()));
    manifest.encode_canonical(&mut canonical);
    let id = kivi_codec::integrity::manifest_id(domain, &canonical);
    Ok((manifest, id))
}

/// Shared structural validation for built and decoded manifests.
fn validate_entries(
    chunking: Chunking,
    total_len: u64,
    entries: &[ChunkEntry],
) -> Result<(), ChunkError> {
    let corrupt = |detail: String| ChunkError::CorruptManifest {
        id: ManifestId::ZERO,
        detail,
    };
    if entries.is_empty() {
        return if total_len == 0 {
            Ok(())
        } else {
            Err(corrupt(format!(
                "empty entry list with nonzero total {total_len}"
            )))
        };
    }
    let mut sum = 0u64;
    for (index, entry) in entries.iter().enumerate() {
        let last = index + 1 == entries.len();
        if entry.len == 0 {
            return Err(corrupt("zero-length manifest entry".to_owned()));
        }
        if !entry.id.is_valid() {
            return Err(corrupt("manifest entry with invalid chunk id".to_owned()));
        }
        if !last && entry.len != chunking.chunk_size as u64 {
            return Err(corrupt(format!(
                "non-final entry {index} length {} disobeys chunk size {}",
                entry.len, chunking.chunk_size
            )));
        }
        sum = sum
            .checked_add(entry.len)
            .ok_or_else(|| corrupt("manifest entry lengths overflow u64".to_owned()))?;
    }
    if sum != total_len {
        return Err(corrupt(format!(
            "entry lengths sum {sum} disagrees with total {total_len}"
        )));
    }
    Ok(())
}

/// Decodes canonical bytes and verifies the expected identity: parse,
/// structural validation, then hash recomputation. Any mismatch is
/// corruption, never a fallback.
///
/// # Errors
///
/// Returns [`ChunkError`] on truncation, version/codec rejection,
/// structural violation, or identity mismatch. Never panics on untrusted
/// input.
pub fn verify_manifest(expected: ManifestId, bytes: &[u8]) -> Result<ChunkManifest, ChunkError> {
    let corrupt = |detail: String| ChunkError::CorruptManifest {
        id: expected,
        detail,
    };
    if !expected.is_valid() {
        return Err(corrupt(
            "expected manifest id is the zero sentinel".to_owned(),
        ));
    }
    let manifest = decode_manifest(bytes).map_err(corrupt)?;
    let recomputed = kivi_codec::integrity::manifest_id(manifest.domain, bytes);
    if recomputed != expected {
        return Err(ChunkError::CorruptManifest {
            id: expected,
            detail: "manifest content hash does not match its reference".to_owned(),
        });
    }
    Ok(manifest)
}

/// Decodes and structurally validates canonical bytes (identity checked
/// separately by [`verify_manifest`]).
fn decode_manifest(bytes: &[u8]) -> Result<ChunkManifest, String> {
    let mut at = 0usize;
    let take = |input: &[u8], at: &mut usize, n: usize| -> Result<Vec<u8>, String> {
        if input.len() < *at + n {
            return Err(format!(
                "manifest truncated: need {} have {}",
                *at + n,
                input.len()
            ));
        }
        let out = input[*at..*at + n].to_vec();
        *at += n;
        Ok(out)
    };
    let magic = u32::from_le_bytes(take(bytes, &mut at, 4)?[..4].try_into().expect("len 4"));
    if magic != MANIFEST_MAGIC {
        return Err("manifest magic mismatch".to_owned());
    }
    let version = u16::from_le_bytes(take(bytes, &mut at, 2)?[..2].try_into().expect("len 2"));
    if version != MANIFEST_VERSION {
        return Err(format!("manifest version {version} unsupported"));
    }
    let chunking_version =
        u16::from_le_bytes(take(bytes, &mut at, 2)?[..2].try_into().expect("len 2"));
    if chunking_version != CHUNKING_V1 {
        return Err(format!(
            "manifest chunking version {chunking_version} unsupported"
        ));
    }
    let codec = ChunkCodecId::from_u8(take(bytes, &mut at, 1)?[0]);
    if !codec.is_supported() {
        return Err(format!("manifest chunk codec {codec} unsupported"));
    }
    if take(bytes, &mut at, 1)?[0] != 0 {
        return Err("manifest reserved byte nonzero".to_owned());
    }
    let domain = SecurityDomainId::from_u64(u64::from_le_bytes(
        take(bytes, &mut at, 8)?[..8].try_into().expect("len 8"),
    ));
    let total_len = u64::from_le_bytes(take(bytes, &mut at, 8)?[..8].try_into().expect("len 8"));
    let chunk_size = u64::from_le_bytes(take(bytes, &mut at, 8)?[..8].try_into().expect("len 8"));
    if chunk_size == 0 || chunk_size > u64::from(u32::MAX) {
        return Err("manifest chunk size out of range".to_owned());
    }
    let count =
        u32::from_le_bytes(take(bytes, &mut at, 4)?[..4].try_into().expect("len 4")) as usize;
    if count > MAX_CHUNKS_PER_MANIFEST {
        return Err("manifest entry count exceeds bound".to_owned());
    }
    let mut entries = Vec::with_capacity(count.min(1024));
    for _ in 0..count {
        let id_raw = take(bytes, &mut at, 32)?;
        let mut id_bytes = [0u8; 32];
        id_bytes.copy_from_slice(&id_raw);
        let len = u64::from_le_bytes(take(bytes, &mut at, 8)?[..8].try_into().expect("len 8"));
        entries.push(ChunkEntry {
            id: ChunkId::from_bytes(id_bytes),
            len,
        });
    }
    if at != bytes.len() {
        return Err(format!(
            "manifest trailing bytes: {} of {}",
            bytes.len() - at,
            bytes.len()
        ));
    }
    let chunk_size_usize =
        usize::try_from(chunk_size).map_err(|_| "chunk size exceeds address space".to_owned())?;
    validate_entries(
        Chunking {
            version: chunking_version,
            chunk_size: chunk_size_usize,
        },
        total_len,
        &entries,
    )
    .map_err(|error| match error {
        ChunkError::CorruptManifest { detail, .. } => detail,
        other => format!("{other:?}"),
    })?;
    Ok(ChunkManifest {
        domain,
        total_len,
        chunking_version,
        chunk_size,
        codec,
        entries,
    })
}

/// Splices replacement entries over the logical range
/// `[offset, offset + replaced_len)` of an entry list: the partial-update
/// primitive (§27, §28).
///
/// Boundaries must align to existing chunk edges (the engine always
/// replaces whole affected chunks: it loads them, patches the byte range,
/// and re-chunks the span, so alignment holds by construction). The
/// replacement lengths must sum to exactly `replaced_len`; the total length
/// is unchanged. Untouched entries keep their [`ChunkId`]s byte-for-byte,
/// so only the replacement chunks plus a new manifest become newly durable.
///
/// # Errors
///
/// Returns [`ChunkError::Invalid`] on misaligned boundaries, length
/// mismatch, or out-of-range spans. Pure: no I/O, no hashing (the caller
/// builds the new manifest for the returned list).
pub fn splice_entries(
    entries: &[ChunkEntry],
    chunk_size: u64,
    offset: u64,
    replaced_len: u64,
    replacement: &[(ChunkId, u64)],
) -> Result<Vec<ChunkEntry>, ChunkError> {
    let invalid = |detail: String| ChunkError::Invalid { detail };
    if replaced_len == 0 {
        return Err(invalid("splice range must be nonempty".to_owned()));
    }
    let total: u64 = entries.iter().map(|entry| entry.len).sum();
    let end = offset
        .checked_add(replaced_len)
        .ok_or_else(|| invalid("splice range overflows u64".to_owned()))?;
    if end > total {
        return Err(invalid(format!(
            "splice range [{offset}, {end}) exceeds total {total}"
        )));
    }
    // Alignment: both edges must sit on cumulative chunk boundaries.
    let mut cursor = 0u64;
    let mut start_ok = offset == 0;
    let mut end_ok = end == total;
    for entry in entries {
        cursor += entry.len;
        if cursor == offset {
            start_ok = true;
        }
        if cursor == end {
            end_ok = true;
        }
    }
    if !start_ok || !end_ok {
        return Err(invalid(format!(
            "splice range [{offset}, {end}) is not chunk-aligned (chunk size {chunk_size})"
        )));
    }
    let replacement_len: u64 = replacement.iter().map(|(_, len)| len).sum();
    if replacement_len != replaced_len {
        return Err(invalid(format!(
            "replacement covers {replacement_len} bytes but the range holds {replaced_len}"
        )));
    }
    for (id, len) in replacement {
        if *len == 0 || !id.is_valid() {
            return Err(invalid(
                "replacement holds an empty or invalid entry".to_owned(),
            ));
        }
    }
    // Single pass: keep entries outside `[offset, end)`, plant the
    // replacement exactly at the `offset` boundary. Alignment was proven
    // above, so the insertion point always exists; its absence is an
    // internal error, reported instead of panicking.
    let mut positioned = Vec::with_capacity(entries.len() + replacement.len());
    let mut cursor = 0u64;
    let mut inserted = false;
    for entry in entries {
        if !inserted && cursor == offset {
            for (id, len) in replacement {
                positioned.push(ChunkEntry { id: *id, len: *len });
            }
            inserted = true;
        }
        let entry_end = cursor + entry.len;
        if entry_end <= offset || cursor >= end {
            positioned.push(*entry);
        }
        cursor = entry_end;
    }
    if !inserted {
        return Err(invalid(
            "splice insertion point missing despite aligned range".to_owned(),
        ));
    }
    Ok(positioned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_codec::integrity::chunk_id;

    const DOMAIN: SecurityDomainId = SecurityDomainId::from_u64(11);

    fn entry(byte: u8, len: u64) -> ChunkEntry {
        ChunkEntry {
            id: chunk_id(DOMAIN, &[byte; 64]),
            len,
        }
    }

    fn chunking() -> Chunking {
        Chunking {
            version: CHUNKING_V1,
            chunk_size: 100,
        }
    }

    #[test]
    fn build_validates_and_hashes_canonically() {
        let entries = vec![entry(1, 100), entry(2, 100), entry(3, 40)];
        let (manifest, id) =
            build_manifest(DOMAIN, chunking(), ChunkCodecId::NONE, 240, entries.clone())
                .expect("builds");
        assert_eq!(manifest.total_len, 240);
        assert_eq!(manifest.entries, entries);
        // Identity is stable and domain-separated.
        let (_, again) =
            build_manifest(DOMAIN, chunking(), ChunkCodecId::NONE, 240, entries.clone())
                .expect("rebuilds");
        assert_eq!(id, again);
        let (_, foreign) = build_manifest(
            SecurityDomainId::from_u64(12),
            chunking(),
            ChunkCodecId::NONE,
            240,
            entries,
        )
        .expect("rebuilds");
        assert_ne!(id, foreign);
    }

    #[test]
    fn build_rejects_ragged_and_lying_inputs() {
        // Non-final entry disobeys the chunk size.
        assert!(
            build_manifest(
                DOMAIN,
                chunking(),
                ChunkCodecId::NONE,
                200,
                vec![entry(1, 90), entry(2, 110)]
            )
            .is_err()
        );
        // Lengths sum past the total.
        assert!(
            build_manifest(
                DOMAIN,
                chunking(),
                ChunkCodecId::NONE,
                199,
                vec![entry(1, 100), entry(2, 100)]
            )
            .is_err()
        );
        // Zero-length entry.
        assert!(
            build_manifest(
                DOMAIN,
                chunking(),
                ChunkCodecId::NONE,
                100,
                vec![entry(1, 0), entry(2, 100)]
            )
            .is_err()
        );
        // Invalid chunk id.
        assert!(
            build_manifest(
                DOMAIN,
                chunking(),
                ChunkCodecId::NONE,
                100,
                vec![ChunkEntry {
                    id: ChunkId::ZERO,
                    len: 100
                }]
            )
            .is_err()
        );
        // Empty entries demand an empty value.
        assert!(build_manifest(DOMAIN, chunking(), ChunkCodecId::NONE, 7, Vec::new()).is_err());
        assert!(build_manifest(DOMAIN, chunking(), ChunkCodecId::NONE, 0, Vec::new()).is_ok());
        // Unsupported codec and chunking fail closed.
        assert!(build_manifest(DOMAIN, chunking(), ChunkCodecId::LZ4, 0, Vec::new()).is_err());
        assert!(
            build_manifest(
                DOMAIN,
                Chunking {
                    version: 0x7F,
                    ..chunking()
                },
                ChunkCodecId::NONE,
                0,
                Vec::new()
            )
            .is_err()
        );
    }

    #[test]
    fn verify_round_trips_and_rejects_tampering() {
        let entries = vec![entry(1, 100), entry(2, 50)];
        let (manifest, id) =
            build_manifest(DOMAIN, chunking(), ChunkCodecId::NONE, 150, entries).expect("builds");
        let (canonical, recomputed) = manifest.canonical_with_id(DOMAIN);
        assert_eq!(id, recomputed);
        let loaded = verify_manifest(id, &canonical).expect("verifies");
        assert_eq!(loaded, manifest);
        // Any byte flip fails: truncation, magic, body, or hash link.
        for index in [0, 10, canonical.len() / 2, canonical.len() - 1] {
            let mut damaged = canonical.clone();
            damaged[index] ^= 0xFF;
            assert!(
                verify_manifest(id, &damaged).is_err(),
                "byte {index} flip must fail"
            );
        }
        // Wrong expected id fails even on intact bytes.
        let other = ManifestId::from_bytes([0x77; 32]);
        assert!(verify_manifest(other, &canonical).is_err());
        // Truncation fails.
        assert!(verify_manifest(id, &canonical[..canonical.len() - 1]).is_err());
    }

    #[test]
    fn splice_reuses_untouched_chunks() {
        // Old: A B C D E (100 each); replace the bytes inside C only.
        let ids = [10u8, 20, 30, 40, 50].map(|byte| chunk_id(DOMAIN, &[byte; 64]));
        let entries: Vec<ChunkEntry> = ids
            .iter()
            .map(|id| ChunkEntry { id: *id, len: 100 })
            .collect();
        let replacement_id = chunk_id(DOMAIN, b"replacement-C");
        let spliced =
            splice_entries(&entries, 100, 200, 100, &[(replacement_id, 100)]).expect("splices");
        assert_eq!(spliced.len(), 5);
        assert_eq!(spliced[0].id, ids[0]);
        assert_eq!(spliced[1].id, ids[1]);
        assert_eq!(spliced[2].id, replacement_id);
        assert_eq!(spliced[3].id, ids[3]);
        assert_eq!(spliced[4].id, ids[4]);
        // The new manifest shares four of five chunk addresses: only the
        // replacement chunk plus the manifest itself are newly durable.
        let (_, old_id) =
            build_manifest(DOMAIN, chunking(), ChunkCodecId::NONE, 500, entries).expect("old");
        let (_, new_id) =
            build_manifest(DOMAIN, chunking(), ChunkCodecId::NONE, 500, spliced).expect("new");
        assert_ne!(old_id, new_id);
    }

    #[test]
    fn splice_rejects_misaligned_and_lying_ranges() {
        let entries = vec![entry(1, 100), entry(2, 100)];
        let good = (entry(9, 100).id, 100);
        // Mid-chunk edges.
        assert!(splice_entries(&entries, 100, 50, 100, &[good]).is_err());
        assert!(splice_entries(&entries, 100, 0, 150, &[good]).is_err());
        // Replacement sum mismatch.
        assert!(splice_entries(&entries, 100, 0, 100, &[(good.0, 90)]).is_err());
        // Past the end and empty ranges.
        assert!(splice_entries(&entries, 100, 100, 200, &[good]).is_err());
        assert!(splice_entries(&entries, 100, 0, 0, &[]).is_err());
        // Multi-chunk replacement spanning two chunks keeps the outer two.
        let chunks: Vec<ChunkEntry> = (1u8..=4).map(|byte| entry(byte, 100)).collect();
        let spliced = splice_entries(&chunks, 100, 100, 200, &[good, good]).expect("spans");
        assert_eq!(spliced.len(), 4);
        assert_eq!(spliced[0], chunks[0]);
        assert_eq!(spliced[3], chunks[3]);
    }
}
