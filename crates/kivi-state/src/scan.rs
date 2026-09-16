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
    /// Resident bytes that exceeded the page budget on an otherwise empty
    /// page: length only, fetch through a point `get`/`get_stream`.
    Oversize {
        /// Logical length of the value.
        logical_len: u64,
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
}
