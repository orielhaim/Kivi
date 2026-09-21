//! Bounded local range scans over the ordered key index (RFC §9).
//!
//! [`ObjectStore::scan_range`](crate::ObjectStore::scan_range) is the one
//! canonical local scan primitive. It seeks the ordered key index and emits
//! a bounded page; it never materializes an unbounded tablet range.
//!
//! ## Semantics
//!
//! * Range is `[start, end)`: `start` inclusive, `end` exclusive,
//!   `None` meaning unbounded on that side. Byte order is lexicographic
//!   (`memcmp`), matching the ordered-routing order.
//! * Keys under the reserved system prefix (`0xFF`, see
//!   [`Key::is_system`](crate::Key::is_system)) are skipped: transaction
//!   records and index-projection metadata live beside user data but never
//!   surface in scans.
//! * Expired objects behave as absent (reads never delete).
//! * `max_items` bounds entries; `max_bytes` bounds inline value bytes in
//!   the page. Chunked values never inline: they answer as
//!   [`ScannedValue::Chunked`] descriptors resolved through the normal
//!   `get_stream` path. An inline value that does not fit the remaining
//!   budget stops the page (it becomes the resume point), unless the page
//!   is still empty — then it answers once as [`ScannedValue::Oversize`]
//!   so the scan always makes progress.
//! * The page carries `exhausted` plus `last_key`: resume by scanning from
//!   just after `last_key` (the cross-tablet cursor owns that increment).
//!
//! Scans are reads: they never enter the WAL and never create mutations.
//! Cross-tablet scans compose these pages in key order; each tablet read
//! holds Kivi's strong `Latest` semantics, but the full multi-tablet scan
//! is NOT one global snapshot (see `ScanConsistency` in the client).

use bytes::Bytes;
use kivi_types::ManifestId;

use crate::index::IndexId;
use crate::object::Key;

/// Scan direction over `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScanDirection {
    /// Ascending key order.
    Forward,
    /// Descending key order.
    Reverse,
}

/// What a scan page carries per key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScanProjection {
    /// Keys only (no value bytes cross the wire).
    KeysOnly,
    /// Keys with bounded values (see [`ScannedValue`]).
    KeysAndValues,
}

/// Bounds and budgets for one local scan page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanSpec {
    /// Inclusive lower bound (`None` = from the first key).
    pub start: Option<Vec<u8>>,
    /// Exclusive upper bound (`None` = to the last key).
    pub end: Option<Vec<u8>>,
    /// Key order.
    pub direction: ScanDirection,
    /// Maximum entries in the page (must be `>= 1`).
    pub max_items: usize,
    /// Budget for inline value bytes in the page (must be `>= 1`).
    pub max_bytes: usize,
    /// Per-key payload shape.
    pub projection: ScanProjection,
}

impl ScanSpec {
    /// Builds a spec, rejecting empty bounds and zero budgets.
    ///
    /// # Errors
    ///
    /// Returns [`ScanError::EmptyRange`] when a bounded end does not exceed
    /// the start, or [`ScanError::InvalidLimit`] for a zero budget.
    pub fn new(
        start: Option<Vec<u8>>,
        end: Option<Vec<u8>>,
        direction: ScanDirection,
        max_items: usize,
        max_bytes: usize,
        projection: ScanProjection,
    ) -> Result<Self, ScanError> {
        if max_items == 0 || max_bytes == 0 {
            return Err(ScanError::InvalidLimit);
        }
        if let (Some(lower), Some(upper)) = (&start, &end)
            && lower >= upper
        {
            return Err(ScanError::EmptyRange);
        }
        Ok(Self {
            start,
            end,
            direction,
            max_items,
            max_bytes,
            projection,
        })
    }

    /// Whether `key` lies in `[start, end)`.
    #[must_use]
    pub fn contains(&self, key: &[u8]) -> bool {
        if let Some(lower) = &self.start
            && key < lower.as_slice()
        {
            return false;
        }
        self.end.as_ref().is_none_or(|upper| key < upper.as_slice())
    }

    /// Whether raw scan bounds name one non-unique `(index, term)` window
    /// (`[prefix, prefix_successor)`, forward only). Server fan-out uses
    /// this before tiling routing (the term prefix is not a user key);
    /// [`ScanSpec::index_term`](Self::index_term) is the typed twin.
    #[must_use]
    pub fn is_term_window(start: Option<&[u8]>, end: Option<&[u8]>, direction: u8) -> bool {
        use crate::index::{INDEX_FORMAT_TAG, term_prefix, term_prefix_parts};
        if direction != 0 {
            return false;
        }
        let Some(start) = start else { return false };
        let Some(end) = end else { return false };
        let Some((index, term)) = term_prefix_parts(start) else {
            return false;
        };
        if start != term_prefix(index, &term, false) {
            return false;
        }
        if start.len() < INDEX_FORMAT_TAG.len() {
            return false;
        }
        crate::index::prefix_successor(start).is_some_and(|next| next == end)
    }

    /// When the window is exactly one non-unique `(index, term)` prefix
    /// (`[prefix, prefix_successor)`), returns the term cursor. Term
    /// lookups fan out per tablet through
    /// [`ObjectStore::scan_index_term`](crate::ObjectStore::scan_index_term),
    /// so co-located entries answer wherever their primaries live.
    #[must_use]
    pub fn index_term(&self) -> Option<IndexTermCursor> {
        use crate::index::{INDEX_FORMAT_TAG, term_prefix};
        if self.direction != ScanDirection::Forward {
            return None;
        }
        let start = self.start.as_deref()?;
        // Prefix form only: the start must decode as a bare term prefix
        // (no primary suffix), so the window covers one term. A full
        // entry key never qualifies here.
        let (index, term) = crate::index::term_prefix_parts(start)?;
        if start != term_prefix(index, &term, false) {
            return None;
        }
        let end = self.end.as_deref()?;
        if end != crate::index::prefix_successor(start)?.as_slice() {
            return None;
        }
        // Confirm the tag shape (non-unique term prefix, not a full entry).
        if start.len() < INDEX_FORMAT_TAG.len() {
            return None;
        }
        Some(IndexTermCursor { index, term })
    }

    /// Whether raw scan bounds name one `(index, [start_term, end_term))`
    /// term range (`[prefix(start), prefix(end))`, forward only). Range
    /// lookups fan out per tablet through
    /// [`ObjectStore::scan_index_term_range`](crate::ObjectStore::scan_index_term_range).
    /// Disjoint from [`is_term_window`](Self::is_term_window): a single-term
    /// end is a byte successor (ends `0x01`), never a term prefix (ends
    /// `0x00`).
    #[must_use]
    pub fn is_term_range_window(start: Option<&[u8]>, end: Option<&[u8]>, direction: u8) -> bool {
        use crate::index::{term_prefix, term_prefix_parts};
        if direction != 0 {
            return false;
        }
        let (Some(start), Some(end)) = (start, end) else {
            return false;
        };
        let (Some((index, start_term)), Some((end_index, end_term))) =
            (term_prefix_parts(start), term_prefix_parts(end))
        else {
            return false;
        };
        if index != end_index {
            return false;
        }
        if start != term_prefix(index, &start_term, false) {
            return false;
        }
        if end != term_prefix(index, &end_term, false) {
            return false;
        }
        start_term.as_slice() < end_term.as_slice()
    }

    /// When the window is exactly one `(index, [start_term, end_term))`
    /// term range, returns the range cursor. See
    /// [`is_term_range_window`](Self::is_term_range_window).
    #[must_use]
    pub fn index_term_range(&self) -> Option<IndexTermRangeCursor> {
        use crate::index::term_prefix;
        if self.direction != ScanDirection::Forward {
            return None;
        }
        let start = self.start.as_deref()?;
        let end = self.end.as_deref()?;
        let (index, start_term) = crate::index::term_prefix_parts(start)?;
        let (end_index, end_term) = crate::index::term_prefix_parts(end)?;
        if index != end_index {
            return None;
        }
        if start != term_prefix(index, &start_term, false) {
            return None;
        }
        if end != term_prefix(index, &end_term, false) {
            return None;
        }
        if start_term.as_slice() >= end_term.as_slice() {
            return None;
        }
        Some(IndexTermRangeCursor {
            index,
            start_term,
            end_term,
        })
    }
}

/// One non-unique term cursor: the index and term whose co-located
/// entries a tablet lists from its own state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexTermCursor {
    /// Index identity.
    pub index: IndexId,
    /// Term bytes.
    pub term: Vec<u8>,
}

/// One non-unique term-range cursor: the index and `[start_term, end_term)`
/// whose co-located entries a tablet lists from its own state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexTermRangeCursor {
    /// Index identity.
    pub index: IndexId,
    /// Inclusive start term.
    pub start_term: Vec<u8>,
    /// Exclusive end term.
    pub end_term: Vec<u8>,
}

/// One scanned key and its projected payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanEntry {
    /// Scanned key bytes.
    pub key: Key,
    /// Projected value (`None` under [`KeysOnly`](ScanProjection::KeysOnly)).
    pub value: Option<ScannedValue>,
}

/// Value payload of a scan entry under
/// [`KeysAndValues`](ScanProjection::KeysAndValues).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScannedValue {
    /// Resident bytes (fit the page budget).
    Inline(Bytes),
    /// Exact counter value (always inline: 8 bytes).
    Counter(i64),
    /// Chunked bytes by reference: resolve through `get_stream`, never
    /// inlined no matter how small (large immutable values must never be
    /// copied into scan responses).
    Chunked {
        /// Manifest addressing the immutable chunk sequence.
        manifest: ManifestId,
        /// Total logical bytes across the manifest.
        logical_len: u64,
    },
    /// Fabric bytes by reference: resolve through the Memory Fabric, never
    /// inlined no matter how small (materialized values must never be
    /// copied into scan responses; resolution may suspend).
    Fabric {
        /// Fabric-scoped object id assigned at staging.
        fabric_id: u64,
        /// Total logical bytes of the materialized value.
        logical_len: u64,
        /// Logical version the bytes were published at (pins reads).
        version: u64,
    },
    /// Resident bytes that exceeded the page budget on an otherwise empty
    /// page: length only, fetch through a point `get`/`get_stream`.
    Oversize {
        /// Logical length of the value.
        logical_len: u64,
    },
    /// Typed semantic value by small descriptor (see
    /// [`LogicalValue::scan_descriptor`](crate::LogicalValue::scan_descriptor)):
    /// counters, escrow state, semaphore load, lease grants, and stream
    /// shard cursors project without inlining unbounded state (permit sets
    /// and retained entries never enter scan pages).
    Semantic {
        /// Object type of the scanned value.
        object: crate::ObjectType,
        /// Small canonical descriptor bytes for `object`.
        descriptor: Bytes,
    },
}

/// One bounded scan page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanPage {
    /// Entries in scan order (ascending for forward, descending for reverse).
    pub entries: Vec<ScanEntry>,
    /// Whether no further keys remain in the range in scan order.
    pub exhausted: bool,
    /// Last emitted key, if any (cursor resume point).
    pub last_key: Option<Key>,
}

/// Local scan failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ScanError {
    /// The tablet has no ordered index (hash-layout fast path keeps no
    /// ordered structure; enable it before scanning).
    #[error("ordered index is not enabled on this tablet")]
    OrderedIndexDisabled,
    /// A zero item or byte budget was supplied.
    #[error("scan budgets must be nonzero")]
    InvalidLimit,
    /// A bounded end does not exceed the start.
    #[error("scan end must exceed start")]
    EmptyRange,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_rejects_empty_ranges_and_zero_budgets() {
        assert_eq!(
            ScanSpec::new(
                Some(b"b".to_vec()),
                Some(b"a".to_vec()),
                ScanDirection::Forward,
                10,
                1024,
                ScanProjection::KeysOnly,
            ),
            Err(ScanError::EmptyRange)
        );
        assert_eq!(
            ScanSpec::new(
                None,
                None,
                ScanDirection::Forward,
                0,
                1024,
                ScanProjection::KeysOnly
            ),
            Err(ScanError::InvalidLimit)
        );
        assert_eq!(
            ScanSpec::new(
                None,
                None,
                ScanDirection::Forward,
                10,
                0,
                ScanProjection::KeysOnly
            ),
            Err(ScanError::InvalidLimit)
        );
    }

    #[test]
    fn spec_contains_follows_half_open_semantics() {
        let spec = ScanSpec::new(
            Some(b"b".to_vec()),
            Some(b"d".to_vec()),
            ScanDirection::Forward,
            10,
            1024,
            ScanProjection::KeysOnly,
        )
        .expect("spec");
        assert!(!spec.contains(b"a"));
        assert!(spec.contains(b"b"));
        assert!(spec.contains(b"c"));
        assert!(!spec.contains(b"d"));
    }

    #[test]
    fn term_windows_cover_exactly_one_term() {
        use crate::index::{IndexId, term_prefix};
        let prefix = term_prefix(IndexId::from_u64(9), b"red", false);
        let end = crate::index::prefix_successor(&prefix).expect("successor");
        let spec = ScanSpec::new(
            Some(prefix.clone()),
            Some(end.clone()),
            ScanDirection::Forward,
            10,
            1024,
            ScanProjection::KeysAndValues,
        )
        .expect("term spec");
        let cursor = spec.index_term().expect("term cursor");
        assert_eq!(cursor.index, IndexId::from_u64(9));
        assert_eq!(cursor.term, b"red");
        assert!(ScanSpec::is_term_window(Some(&prefix), Some(&end), 0));
        // A full entry key is not a term window.
        let entry = crate::encode_non_unique_key(IndexId::from_u64(9), b"red", b"a");
        assert!(ScanSpec::index_term(&spec).is_some());
        let entry_spec = ScanSpec::new(
            Some(entry.clone()),
            Some(end.clone()),
            ScanDirection::Forward,
            10,
            1024,
            ScanProjection::KeysAndValues,
        )
        .expect("entry spec");
        assert_eq!(entry_spec.index_term(), None);
    }
}
