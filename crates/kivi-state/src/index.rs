//! Secondary indexes as ordinary ordered state (RFC §9 + index direction).
//!
//! An index is another ordered representation of primary data: index
//! entries are ordinary Kivi keys in an ordered index namespace, so they
//! split, merge, migrate, repair, and scan with the same machinery. No
//! index-specific replication exists.
//!
//! ## Encoding (Kivi-owned format `KIVI-IDX1`, pinned)
//!
//! Bytes in an index namespace are canonical Kivi data-model bytes: the
//! format version is pinned here and covered by golden vectors, so a future
//! dependency or refactor can never silently change persistent order.
//!
//! ```text
//! non-unique: HEADER || index_id BE64 || 0x00 || esc(term) || 0x00 || esc(primary) || 0x00
//! unique:     HEADER || index_id BE64 || 0x00 || esc(term) || 0x00
//! ```
//!
//! `esc` escapes `0x00 → 0x00 0xFF` (FoundationDB-tuple-style); fields are
//! `0x00`-terminated. The encoding is order-preserving: all entries for one
//! term are contiguous, ordered by primary key; terms order
//! lexicographically. An equality query is one prefix range; a term range
//! is a wider prefix range.
//!
//! Entry values carry the primary reference + version so queries can skip
//! stale entries while distributed finalization converges:
//!
//! ```text
//! non-unique value: version BE64            (primary key is in the entry key)
//! unique value:     version BE64 || primary bytes
//! ```
//!
//! ## Caller-supplied terms
//!
//! Kivi stores opaque bytes and cannot infer index terms: the caller
//! supplies `index_terms` on every indexed write, and the primary tablet
//! retains the current `IndexId → terms` projection under a `\xff` system
//! key beside the primary (see [`projection_key`]) so updates/deletes can
//! remove stale entries without client memory.

use kivi_types::NamespaceId;

use crate::object::{Key, ObjectVersion};

/// Pinned index-key format tag (9 bytes).
pub const INDEX_FORMAT_TAG: &[u8; 9] = b"KIVI-IDX1";
/// Format version for projection-record and entry-value encodings.
pub const INDEX_ENCODING_VERSION: u16 = 1;

/// Secondary index identity (control-plane assigned).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IndexId(u64);

impl IndexId {
    /// Wraps a raw id.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw id.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl core::fmt::Display for IndexId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "idx{}", self.0)
    }
}

/// Index uniqueness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexKind {
    /// Many primaries per term; term range scans list them in primary order.
    NonUnique,
    /// One primary per term; concurrent claims conflict.
    Unique,
}

/// Lifecycle of an index definition (control-plane owned).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexState {
    /// Accepting maintenance writes, invisible to queries (backfill window).
    Building,
    /// Maintained and queryable.
    Ready,
    /// Maintenance stopped, entries being removed.
    Dropping,
}

/// Index definition: which primary namespace feeds which ordered index
/// namespace, and how.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexDefinition {
    /// Index identity.
    pub id: IndexId,
    /// Namespace holding primary objects.
    pub primary: NamespaceId,
    /// Ordered namespace holding index entries.
    pub index_ns: NamespaceId,
    /// Uniqueness contract.
    pub kind: IndexKind,
    /// Lifecycle state.
    pub state: IndexState,
}

/// Index key-codec failure (structural only).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum IndexCodecError {
    /// Input is not an index key of the expected shape.
    #[error("not an index key")]
    NotAnIndexKey,
    /// Truncated input where the format promised bytes.
    #[error("truncated index key")]
    Truncated,
}

/// Escapes one field: `0x00 → 0x00 0xFF`, then terminates with `0x00`.
fn push_escaped(out: &mut Vec<u8>, field: &[u8]) {
    for byte in field {
        if *byte == 0x00 {
            out.extend_from_slice(&[0x00, 0xFF]);
        } else {
            out.push(*byte);
        }
    }
    out.push(0x00);
}

/// Reads one escaped field starting at `pos`; returns the field and the
/// position just past its terminator.
fn take_escaped(input: &[u8], mut pos: usize) -> Result<(Vec<u8>, usize), IndexCodecError> {
    let mut field = Vec::new();
    loop {
        let byte = input.get(pos).copied().ok_or(IndexCodecError::Truncated)?;
        pos += 1;
        if byte == 0x00 {
            match input.get(pos).copied() {
                Some(0xFF) => {
                    field.push(0x00);
                    pos += 1;
                }
                _ => break,
            }
        } else {
            field.push(byte);
        }
    }
    Ok((field, pos))
}

/// Encodes a non-unique index entry key.
#[must_use]
pub fn encode_non_unique_key(index: IndexId, term: &[u8], primary: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(24 + term.len() + primary.len());
    out.extend_from_slice(INDEX_FORMAT_TAG);
    out.extend_from_slice(&index.as_u64().to_be_bytes());
    out.push(0x00);
    push_escaped(&mut out, term);
    push_escaped(&mut out, primary);
    out
}

/// Encodes a unique index entry key.
#[must_use]
pub fn encode_unique_key(index: IndexId, term: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(24 + term.len());
    out.extend_from_slice(INDEX_FORMAT_TAG);
    out.extend_from_slice(&index.as_u64().to_be_bytes());
    out.push(0x00);
    push_escaped(&mut out, term);
    out
}

/// Term prefix shared by every entry for `(index, term)`: equality scans
/// use `[prefix, prefix_successor)`; term ranges compose prefixes.
#[must_use]
pub fn term_prefix(index: IndexId, term: &[u8], unique: bool) -> Vec<u8> {
    if unique {
        encode_unique_key(index, term)
    } else {
        let mut out = Vec::with_capacity(24 + term.len());
        out.extend_from_slice(INDEX_FORMAT_TAG);
        out.extend_from_slice(&index.as_u64().to_be_bytes());
        out.push(0x00);
        push_escaped(&mut out, term);
        out
    }
}

/// Exclusive end of a prefix range: the lexicographic successor of
/// `prefix`, or `None` when `prefix` is all `0xFF` (unbounded above).
#[must_use]
pub fn prefix_successor(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut end = prefix.to_vec();
    while let Some(last) = end.pop() {
        if last != 0xFF {
            end.push(last + 1);
            return Some(end);
        }
    }
    None
}

/// Decodes a non-unique entry key into `(index, term, primary)`.
///
/// # Errors
///
/// Returns [`IndexCodecError`] when the input is not a well-formed
/// non-unique entry key.
pub fn decode_non_unique_key(input: &[u8]) -> Result<(IndexId, Vec<u8>, Vec<u8>), IndexCodecError> {
    use IndexCodecError as Fault;
    if input.len() < 18 || &input[..9] != INDEX_FORMAT_TAG || input[17] != 0x00 {
        return Err(Fault::NotAnIndexKey);
    }
    let index = IndexId::from_u64(u64::from_be_bytes(
        input[9..17].try_into().map_err(|_| Fault::NotAnIndexKey)?,
    ));
    let (term, at) = take_escaped(input, 18)?;
    let (primary, end) = take_escaped(input, at)?;
    if end != input.len() {
        return Err(Fault::NotAnIndexKey);
    }
    Ok((index, term, primary))
}

/// Decodes a unique entry key into `(index, term)`.
///
/// # Errors
///
/// Returns [`IndexCodecError`] when the input is not a well-formed unique
/// entry key.
pub fn decode_unique_key(input: &[u8]) -> Result<(IndexId, Vec<u8>), IndexCodecError> {
    use IndexCodecError as Fault;
    if input.len() < 18 || &input[..9] != INDEX_FORMAT_TAG || input[17] != 0x00 {
        return Err(Fault::NotAnIndexKey);
    }
    let index = IndexId::from_u64(u64::from_be_bytes(
        input[9..17].try_into().map_err(|_| Fault::NotAnIndexKey)?,
    ));
    let (term, end) = take_escaped(input, 18)?;
    if end != input.len() {
        return Err(Fault::NotAnIndexKey);
    }
    Ok((index, term))
}

/// Encodes a non-unique entry value: the referenced primary version.
#[must_use]
pub fn encode_non_unique_value(version: ObjectVersion) -> Vec<u8> {
    version.as_u64().to_be_bytes().to_vec()
}

/// Encodes a unique entry value: referenced primary version + primary key.
#[must_use]
pub fn encode_unique_value(version: ObjectVersion, primary: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + primary.len());
    out.extend_from_slice(&version.as_u64().to_be_bytes());
    out.extend_from_slice(primary);
    out
}

/// Decodes an entry value into `(version, primary_or_empty)`.
///
/// # Errors
///
/// Returns [`IndexCodecError::Truncated`] when fewer than 8 bytes remain.
pub fn decode_entry_value(input: &[u8]) -> Result<(ObjectVersion, &[u8]), IndexCodecError> {
    if input.len() < 8 {
        return Err(IndexCodecError::Truncated);
    }
    let version = ObjectVersion::from_u64(u64::from_be_bytes(
        input[..8]
            .try_into()
            .map_err(|_| IndexCodecError::Truncated)?,
    ));
    Ok((version, &input[8..]))
}

/// System key holding one primary's index projections:
/// `\xff kivi/proj/ <primary bytes>`.
///
/// Routing is by stripped remainder: routers route `<primary>` normally,
/// so the projection always lands on the primary's tablet (same routing
/// input modulo the prefix). Scans skip `\xff`-prefixed keys, so
/// projections never surface in range results.
#[must_use]
pub fn projection_key(primary: &[u8]) -> Key {
    let mut out = Vec::with_capacity(12 + primary.len());
    out.push(0xFF);
    out.extend_from_slice(b"kivi/proj/");
    out.extend_from_slice(primary);
    Key::from(out)
}

/// Projection-key prefix after the system byte: `kivi/proj/`.
pub const PROJECTION_PREFIX: &[u8; 10] = b"kivi/proj/";

/// Strips a projection key to its primary (`None` for ordinary keys).
/// Routers use this to co-locate projections with their primaries.
#[must_use]
pub fn parse_projection_primary(key: &[u8]) -> Option<&[u8]> {
    if key.len() <= 11 || key[0] != 0xFF || &key[1..11] != PROJECTION_PREFIX {
        return None;
    }
    Some(&key[11..])
}

/// Whether `key` names an index entry (reserved `KIVI-IDX1` prefix).
/// Direct single-key writes to index entries are rejected at admission
/// (only the transactional index-maintenance path may write them);
/// scans include them (index queries are scans).
#[must_use]
pub fn is_index_key(key: &[u8]) -> bool {
    key.len() >= INDEX_FORMAT_TAG.len() && &key[..INDEX_FORMAT_TAG.len()] == INDEX_FORMAT_TAG
}

/// Projection record: `IndexId → current terms` for one primary.
/// Sorted by index id (deterministic encoding).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexProjections {
    /// Current terms per index.
    pub terms: std::collections::BTreeMap<IndexId, Vec<Vec<u8>>>,
}

impl IndexProjections {
    /// Encodes the record (versioned, little-endian lengths).
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&INDEX_ENCODING_VERSION.to_le_bytes());
        out.extend_from_slice(
            &u32::try_from(self.terms.len())
                .unwrap_or(u32::MAX)
                .to_le_bytes(),
        );
        for (index, terms) in &self.terms {
            out.extend_from_slice(&index.as_u64().to_le_bytes());
            out.extend_from_slice(&u32::try_from(terms.len()).unwrap_or(u32::MAX).to_le_bytes());
            for term in terms {
                out.extend_from_slice(&u32::try_from(term.len()).unwrap_or(u32::MAX).to_le_bytes());
                out.extend_from_slice(term);
            }
        }
        out
    }

    /// Decodes a record, requiring exact consumption.
    ///
    /// # Errors
    ///
    /// Returns [`IndexCodecError`] on structural failure.
    pub fn decode(input: &[u8]) -> Result<Self, IndexCodecError> {
        let take_u32 = |pos: usize| -> Result<(u32, usize), IndexCodecError> {
            input
                .get(pos..pos + 4)
                .ok_or(IndexCodecError::Truncated)
                .and_then(|bytes| {
                    bytes
                        .try_into()
                        .map(u32::from_le_bytes)
                        .map_err(|_| IndexCodecError::Truncated)
                })
                .map(|value| (value, pos + 4))
        };
        if input.len() < 2 {
            return Err(IndexCodecError::Truncated);
        }
        if u16::from_le_bytes([input[0], input[1]]) != INDEX_ENCODING_VERSION {
            return Err(IndexCodecError::NotAnIndexKey);
        }
        let mut pos = 2;
        let (indexes, next) = take_u32(pos)?;
        pos = next;
        let mut terms = std::collections::BTreeMap::new();
        for _ in 0..indexes {
            if input.len() < pos + 8 {
                return Err(IndexCodecError::Truncated);
            }
            let id = IndexId::from_u64(u64::from_le_bytes(
                input[pos..pos + 8]
                    .try_into()
                    .map_err(|_| IndexCodecError::Truncated)?,
            ));
            pos += 8;
            let (count, next) = take_u32(pos)?;
            pos = next;
            let mut owned = Vec::with_capacity(count.min(64) as usize);
            for _ in 0..count {
                let (len, next) = take_u32(pos)?;
                pos = next;
                let len = len as usize;
                let bytes = input
                    .get(pos..pos + len)
                    .ok_or(IndexCodecError::Truncated)?;
                pos += len;
                owned.push(bytes.to_vec());
            }
            terms.insert(id, owned);
        }
        if pos != input.len() {
            return Err(IndexCodecError::NotAnIndexKey);
        }
        Ok(Self { terms })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_index_key_vectors() {
        // Pinned vectors: any format change must update these deliberately.
        let key = encode_non_unique_key(IndexId::from_u64(7), b"alice", b"user:1");
        let mut expected = Vec::new();
        expected.extend_from_slice(b"KIVI-IDX1");
        expected.extend_from_slice(&7u64.to_be_bytes());
        expected.push(0x00);
        expected.extend_from_slice(b"alice");
        expected.push(0x00);
        expected.extend_from_slice(b"user:1");
        expected.push(0x00);
        assert_eq!(key, expected);

        let unique = encode_unique_key(IndexId::from_u64(7), b"alice@example.com");
        let mut expected_unique = Vec::new();
        expected_unique.extend_from_slice(b"KIVI-IDX1");
        expected_unique.extend_from_slice(&7u64.to_be_bytes());
        expected_unique.push(0x00);
        expected_unique.extend_from_slice(b"alice@example.com");
        expected_unique.push(0x00);
        assert_eq!(unique, expected_unique);

        // NUL bytes escape and round-trip.
        let escaped = encode_non_unique_key(IndexId::from_u64(1), b"a\x00b", b"k\x00");
        let (index, term, primary) = decode_non_unique_key(&escaped).expect("round trip");
        assert_eq!(
            (index, term.as_slice(), primary.as_slice()),
            (
                IndexId::from_u64(1),
                b"a\x00b".as_slice(),
                b"k\x00".as_slice()
            )
        );
        let (uindex, uterm) = decode_unique_key(&unique).expect("round trip");
        assert_eq!(
            (uindex, uterm.as_slice()),
            (IndexId::from_u64(7), b"alice@example.com".as_slice())
        );
    }

    #[test]
    fn term_ordering_groups_entries() {
        // Same term, different primaries: contiguous and primary-ordered.
        let a = encode_non_unique_key(IndexId::from_u64(1), b"t", b"a");
        let b = encode_non_unique_key(IndexId::from_u64(1), b"t", b"b");
        let other = encode_non_unique_key(IndexId::from_u64(1), b"u", b"a");
        assert!(a < b);
        assert!(b < other);
        // Equality prefix covers exactly the term's entries.
        let prefix = term_prefix(IndexId::from_u64(1), b"t", false);
        assert!(a.starts_with(&prefix));
        assert!(b.starts_with(&prefix));
        assert!(!other.starts_with(&prefix));
        let end = prefix_successor(&prefix).expect("successor");
        assert!(a < end);
        assert!(b < end);
        assert!(other >= end);
        // Earlier terms sort earlier.
        let earlier = term_prefix(IndexId::from_u64(1), b"s", false);
        assert!(earlier < prefix);
    }

    #[test]
    fn entry_values_carry_primary_versions() {
        let encoded = encode_non_unique_value(ObjectVersion::from_u64(42));
        let (version, rest) = decode_entry_value(&encoded).expect("decode");
        assert_eq!(version, ObjectVersion::from_u64(42));
        assert!(rest.is_empty());
        let unique = encode_unique_value(ObjectVersion::from_u64(9), b"user:7");
        let (version, primary) = decode_entry_value(&unique).expect("decode");
        assert_eq!(version, ObjectVersion::from_u64(9));
        assert_eq!(primary, b"user:7");
    }

    #[test]
    fn projection_records_round_trip() {
        let mut projections = IndexProjections::default();
        projections.terms.insert(
            IndexId::from_u64(3),
            vec![b"alice".to_vec(), b"a\x00b".to_vec()],
        );
        projections.terms.insert(IndexId::from_u64(5), vec![]);
        let back = IndexProjections::decode(&projections.encode()).expect("round trip");
        assert_eq!(back, projections);
        assert!(IndexProjections::decode(b"short").is_err());
    }

    #[test]
    fn malformed_keys_rejected() {
        assert_eq!(
            decode_non_unique_key(b"garbage"),
            Err(IndexCodecError::NotAnIndexKey)
        );
        assert_eq!(
            decode_unique_key(&encode_non_unique_key(IndexId::from_u64(1), b"t", b"p")),
            Err(IndexCodecError::NotAnIndexKey)
        );
    }

    #[test]
    fn projection_and_index_key_classification() {
        let primary = b"user:1";
        let key = projection_key(primary);
        assert!(key.is_system());
        assert_eq!(
            parse_projection_primary(key.as_bytes()),
            Some(primary.as_slice())
        );
        assert_eq!(parse_projection_primary(b"user:1"), None);
        assert_eq!(parse_projection_primary(b"\xffkivi/proj/"), None);
        assert!(is_index_key(&encode_non_unique_key(
            IndexId::from_u64(1),
            b"t",
            b"p"
        )));
        assert!(is_index_key(&encode_unique_key(IndexId::from_u64(1), b"t")));
        assert!(!is_index_key(b"user:1"));
        assert!(!is_index_key(b"KIVI-IDX"));
    }
}
