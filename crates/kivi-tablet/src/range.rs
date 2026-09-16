//! Partition ranges for hash and ordered namespace layouts (RFC §9).
//!
//! A namespace uses one layout:
//!
//! * **Hash** (default, Redis-compatible state): `key bytes → stable
//!   128-bit partition hash → adaptive prefix tree → tablet`. A [`HashPrefix`]
//!   names one tree node, e.g. `0*` splitting later into `00*` and `01*`.
//!   The hash is 128 bits (not 64) so deep splits keep consuming fresh bits
//!   of the same hash instead of forcing a rehash.
//! * **Ordered** (range scans, ordered maps, time series): half-open key
//!   intervals such as `["a", "g")`, `["g", "p")`, `["p", ∞)`.
//!
//! The directory consumes already-computed [`PartitionHash`] values for hash
//! lookup, so this crate never chooses a hash algorithm; the stable
//! partition-hash function is a separate future decision. Ranges are
//! validated at construction and re-checked by directory validation.

use core::fmt;

use kivi_types::PartitionHash;

/// Maximum prefix length in bits (one full 128-bit partition hash).
pub const MAX_PREFIX_LEN: u8 = 128;

/// One node of the adaptive hash-partition tree: the set of hashes whose top
/// `len` bits equal the top `len` bits of `bits`.
///
/// Canonical form: the low `128 - len` bits of `bits` must be zero (a `len`
/// of `0` with zero bits names the root covering the whole hash space).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HashPrefix {
    bits: u128,
    len: u8,
}

impl HashPrefix {
    /// Builds a prefix, enforcing canonical form.
    ///
    /// # Errors
    ///
    /// Returns [`RangeError::PrefixTooLong`] if `len` exceeds 128, or
    /// [`RangeError::PrefixUncanonical`] if insignificant low bits are set.
    pub fn new(bits: u128, len: u8) -> Result<Self, RangeError> {
        if len > MAX_PREFIX_LEN {
            return Err(RangeError::PrefixTooLong { len });
        }
        if len == 0 {
            if bits != 0 {
                return Err(RangeError::PrefixUncanonical { bits, len });
            }
            return Ok(Self { bits, len });
        }
        let insignificant = if len == MAX_PREFIX_LEN {
            0
        } else {
            u128::MAX >> len
        };
        if bits & insignificant != 0 {
            return Err(RangeError::PrefixUncanonical { bits, len });
        }
        Ok(Self { bits, len })
    }

    /// Returns the significant high bits (low `128 - len` bits are zero).
    #[must_use]
    pub const fn bits(self) -> u128 {
        self.bits
    }

    /// Returns the prefix length in bits (`0` covers the whole space).
    ///
    /// Named `prefix_len` (not `len`) deliberately: a zero length here means
    /// maximal coverage, so `is_empty` semantics would be a lie.
    #[must_use]
    pub const fn prefix_len(self) -> u8 {
        self.len
    }

    /// Whether this is the root covering every hash.
    #[must_use]
    pub const fn is_root(self) -> bool {
        self.len == 0
    }

    /// Re-runs construction validation (used by directory validation).
    ///
    /// # Errors
    ///
    /// Returns the same [`RangeError`] [`new`](Self::new) would return.
    pub fn check(&self) -> Result<(), RangeError> {
        Self::new(self.bits, self.len).map(|_| ())
    }

    /// Whether `hash` falls under this prefix.
    #[must_use]
    pub fn contains(&self, hash: PartitionHash) -> bool {
        if self.len == 0 {
            return true;
        }
        let shift = MAX_PREFIX_LEN - self.len;
        (hash.as_u128() >> shift) == (self.bits >> shift)
    }

    /// Whether the two prefixes share at least one hash (symmetric).
    ///
    /// Two prefixes overlap exactly when their shorter common length agrees;
    /// identical and ancestor/descendant prefixes therefore overlap.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        let common = self.len.min(other.len);
        if common == 0 {
            return true;
        }
        let shift = MAX_PREFIX_LEN - common;
        (self.bits >> shift) == (other.bits >> shift)
    }

    /// Splits this prefix at its midpoint into `(left, right, split_hash)`:
    /// `left = [bits, split_hash)`, `right = [split_hash, bits+2*half)`.
    /// `split_hash` is the first hash of the right half (also
    /// `right.bits()`), the deterministic boundary persisted in
    /// control-plane split-plan flows. The architecture permits arbitrary
    /// future split points; this stage supports midpoint only.
    ///
    /// # Errors
    ///
    /// Returns [`RangeError::PrefixTooLong`] when the prefix is already a
    /// single address (`len == 128`, unsplittable).
    pub fn split_midpoint(self) -> Result<(Self, Self, u128), RangeError> {
        if self.len >= MAX_PREFIX_LEN {
            return Err(RangeError::PrefixTooLong { len: self.len });
        }
        let len = self.len + 1;
        let half = 1u128 << (MAX_PREFIX_LEN - len);
        let left = Self::new(self.bits, len)?;
        let right = Self::new(self.bits | half, len)?;
        Ok((left, right, self.bits | half))
    }

    /// Whether `left` and `right` are the midpoint children of `self`
    /// (adjacency for merge validation).
    #[must_use]
    pub fn is_midpoint_split_of(self, left: Self, right: Self) -> bool {
        self.split_midpoint()
            .is_ok_and(|(expect_left, expect_right, _)| {
                expect_left == left && expect_right == right
            })
    }
}

impl fmt::Display for HashPrefix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.len == 0 {
            return write!(f, "*");
        }
        for index in 0..self.len {
            let bit = (self.bits >> (MAX_PREFIX_LEN - 1 - index)) & 1;
            write!(f, "{bit}")?;
        }
        write!(f, "*")
    }
}

/// One ordered-layout interval `[start, end)`: `start` inclusive, `end`
/// exclusive, `None` meaning `+∞` (RFC §9 examples `["a", "g")`, `["p", ∞)`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OrderedRange {
    start: Vec<u8>,
    end: Option<Vec<u8>>,
}

impl OrderedRange {
    /// Builds a half-open interval, requiring `start < end` when bounded.
    ///
    /// # Errors
    ///
    /// Returns [`RangeError::EmptyOrderedRange`] if a bounded end does not
    /// strictly exceed the start.
    pub fn new(start: Vec<u8>, end: Option<Vec<u8>>) -> Result<Self, RangeError> {
        if let Some(bound) = &end
            && start >= *bound
        {
            return Err(RangeError::EmptyOrderedRange);
        }
        Ok(Self { start, end })
    }

    /// Returns the inclusive lower bound.
    #[must_use]
    pub fn start(&self) -> &[u8] {
        &self.start
    }

    /// Returns the exclusive upper bound, or `None` for `+∞`.
    #[must_use]
    pub fn end(&self) -> Option<&[u8]> {
        self.end.as_deref()
    }

    /// Re-runs construction validation (used by directory validation).
    ///
    /// # Errors
    ///
    /// Returns the same [`RangeError`] [`new`](Self::new) would return.
    pub fn check(&self) -> Result<(), RangeError> {
        Self::new(self.start.clone(), self.end.clone()).map(|_| ())
    }

    /// Whether `key` lies in `[start, end)`.
    #[must_use]
    pub fn contains(&self, key: &[u8]) -> bool {
        if key < self.start.as_slice() {
            return false;
        }
        self.end.as_ref().is_none_or(|bound| key < bound.as_slice())
    }

    /// Whether the two intervals share at least one key (symmetric).
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        below_end(&self.start, other.end.as_deref()) && below_end(&other.start, self.end.as_deref())
    }

    /// Splits `[start, end)` at `key` into `([start, key), [key, end))`.
    ///
    /// # Errors
    ///
    /// Returns [`RangeError::InvalidSplitKey`] when `key` is not strictly
    /// interior (`start < key < end`, with `+∞` treated as unbounded above).
    pub fn split_at(&self, key: &[u8]) -> Result<(Self, Self), RangeError> {
        if key <= self.start.as_slice() {
            return Err(RangeError::InvalidSplitKey);
        }
        if let Some(bound) = self.end.as_deref()
            && key >= bound
        {
            return Err(RangeError::InvalidSplitKey);
        }
        Ok((
            Self {
                start: self.start.clone(),
                end: Some(key.to_vec()),
            },
            Self {
                start: key.to_vec(),
                end: self.end.clone(),
            },
        ))
    }

    /// Whether `self` ends exactly where `other` starts (mergeable pair).
    #[must_use]
    pub fn is_adjacent_to(&self, other: &Self) -> bool {
        self.end.as_deref() == Some(other.start.as_slice())
    }

    /// Merges two adjacent intervals `[a, b) + [b, c) = [a, c)`.
    ///
    /// # Errors
    ///
    /// Returns [`RangeError::NotAdjacent`] unless `self` ends exactly where
    /// `other` starts.
    pub fn merge_with(&self, other: &Self) -> Result<Self, RangeError> {
        if !self.is_adjacent_to(other) {
            return Err(RangeError::NotAdjacent);
        }
        Ok(Self {
            start: self.start.clone(),
            end: other.end.clone(),
        })
    }
}

/// Whether `start` lies strictly below `end`, treating `None` as `+∞`.
fn below_end(start: &[u8], end: Option<&[u8]>) -> bool {
    end.is_none_or(|bound| start < bound)
}

impl fmt::Display for OrderedRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        write_key(f, &self.start)?;
        write!(f, ", ")?;
        match &self.end {
            Some(bound) => {
                write_key(f, bound)?;
            }
            None => write!(f, "∞")?,
        }
        write!(f, ")")
    }
}

/// Writes a key with non-printable bytes escaped.
fn write_key(f: &mut fmt::Formatter<'_>, key: &[u8]) -> fmt::Result {
    for byte in key {
        if byte.is_ascii_graphic() || *byte == b' ' {
            write!(f, "{}", *byte as char)?;
        } else {
            write!(f, "\\x{byte:02x}")?;
        }
    }
    Ok(())
}

/// Which namespace layout a range belongs to (RFC §9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartitionKind {
    /// Adaptive hash-prefix tree.
    Hash,
    /// Ordered key intervals.
    Ordered,
}

impl fmt::Display for PartitionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hash => write!(f, "hash"),
            Self::Ordered => write!(f, "ordered"),
        }
    }
}

/// Partition range in either namespace layout.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PartitionRange {
    /// Adaptive hash-prefix tree node.
    Hash(HashPrefix),
    /// Ordered key interval.
    Ordered(OrderedRange),
}

impl PartitionRange {
    /// Returns which layout this range belongs to.
    #[must_use]
    pub const fn kind(&self) -> PartitionKind {
        match self {
            Self::Hash(_) => PartitionKind::Hash,
            Self::Ordered(_) => PartitionKind::Ordered,
        }
    }

    /// Re-runs construction validation (used by directory validation).
    ///
    /// # Errors
    ///
    /// Returns the underlying [`RangeError`] on invalid ranges.
    pub fn check(&self) -> Result<(), RangeError> {
        match self {
            Self::Hash(prefix) => prefix.check(),
            Self::Ordered(range) => range.check(),
        }
    }

    /// Whether partition hash `hash` routes into this range.
    ///
    /// Always `false` for ordered ranges: those route by key, never by hash.
    #[must_use]
    pub fn contains_hash(&self, hash: PartitionHash) -> bool {
        match self {
            Self::Hash(prefix) => prefix.contains(hash),
            Self::Ordered(_) => false,
        }
    }

    /// Whether `key` routes into this range.
    ///
    /// Always `false` for hash ranges: those route by partition hash, which
    /// is computed by a future hashing layer, never by raw key here.
    #[must_use]
    pub fn contains_key(&self, key: &[u8]) -> bool {
        match self {
            Self::Hash(_) => false,
            Self::Ordered(range) => range.contains(key),
        }
    }

    /// Whether two ranges of the same layout share at least one point.
    ///
    /// Returns `false` for mixed layouts: hash prefixes and key intervals
    /// live in different spaces, and active tablets of mixed layouts are
    /// rejected by directory validation, so a mixed comparison never governs
    /// a routing decision.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Hash(left), Self::Hash(right)) => left.overlaps(right),
            (Self::Ordered(left), Self::Ordered(right)) => left.overlaps(right),
            _ => false,
        }
    }
}

impl fmt::Display for PartitionRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hash(prefix) => write!(f, "hash:{prefix}"),
            Self::Ordered(range) => write!(f, "ordered:{range}"),
        }
    }
}

/// Reason a partition range was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum RangeError {
    /// Hash prefix length exceeds 128 bits.
    #[error("hash prefix length {len} exceeds 128 bits")]
    PrefixTooLong {
        /// Observed length.
        len: u8,
    },
    /// Hash prefix has insignificant low bits set (non-canonical form).
    #[error("hash prefix {bits:#034X}/{len} has insignificant bits set")]
    PrefixUncanonical {
        /// Observed bits.
        bits: u128,
        /// Observed length.
        len: u8,
    },
    /// Ordered range end does not strictly exceed its start.
    #[error("ordered range end must strictly exceed start")]
    EmptyOrderedRange,
    /// Split key is not strictly interior to the ordered range.
    #[error("ordered split key must satisfy start < key < end")]
    InvalidSplitKey,
    /// Two ordered ranges are not adjacent and cannot merge.
    #[error("ordered ranges are not adjacent (left.end must equal right.start)")]
    NotAdjacent,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_prefix_root_covers_everything() {
        let root = HashPrefix::new(0, 0).expect("root");
        assert!(root.is_root());
        assert!(root.contains(PartitionHash::from_u128(0)));
        assert!(root.contains(PartitionHash::from_u128(u128::MAX)));
        assert_eq!(root.to_string(), "*");
    }

    #[test]
    fn hash_prefix_split_example_from_rfc() {
        // RFC §9: `0*` splits into `00*` and `01*` (top bits of the 128-bit hash).
        let parent = HashPrefix::new(0x0000_0000_0000_0000_0000_0000_0000_0000, 1).expect("0*");
        let left = HashPrefix::new(0x0000_0000_0000_0000_0000_0000_0000_0000, 2).expect("00*");
        let right = HashPrefix::new(0x4000_0000_0000_0000_0000_0000_0000_0000, 2).expect("01*");
        assert_eq!(parent.to_string(), "0*");
        assert_eq!(left.to_string(), "00*");
        assert_eq!(right.to_string(), "01*");
        // Children partition the parent: disjoint from each other, each
        // contained in the parent.
        assert!(!left.overlaps(&right));
        assert!(left.overlaps(&parent));
        assert!(right.overlaps(&parent));
        assert!(parent.overlaps(&left));
        // Point checks: top bits decide.
        let hash = PartitionHash::from_u128;
        assert!(left.contains(hash(0x0000_0000_0000_0000_0000_0000_0000_0000)));
        assert!(left.contains(hash(0x3FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF)));
        assert!(!left.contains(hash(0x4000_0000_0000_0000_0000_0000_0000_0000)));
        assert!(right.contains(hash(0x4000_0000_0000_0000_0000_0000_0000_0000)));
        assert!(right.contains(hash(0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF)));
        assert!(!right.contains(hash(0x8000_0000_0000_0000_0000_0000_0000_0000)));
    }

    #[test]
    fn hash_prefix_rejects_noncanonical_and_long() {
        assert_eq!(
            HashPrefix::new(0, 129),
            Err(RangeError::PrefixTooLong { len: 129 })
        );
        // Low bit set with len 127: insignificant.
        assert!(matches!(
            HashPrefix::new(0x0000_0000_0000_0000_0000_0000_0000_0001, 127),
            Err(RangeError::PrefixUncanonical { .. })
        ));
        // Root with nonzero bits is not canonical.
        assert!(matches!(
            HashPrefix::new(1, 0),
            Err(RangeError::PrefixUncanonical { .. })
        ));
        // Full-length prefixes accept any bits (all significant).
        assert!(HashPrefix::new(u128::MAX, 128).is_ok());
        assert!(HashPrefix::new(1, 128).is_ok());
    }

    #[test]
    fn prefix_lengths_at_64_and_128_bit_boundaries() {
        // Lengths straddling the old 64-bit boundary plus both extremes.
        for len in [0u8, 1, 63, 64, 65, 127, 128] {
            let zero = HashPrefix::new(0, len).expect("zero prefix canonical");
            assert_eq!(zero.prefix_len(), len);
            assert!(zero.contains(PartitionHash::from_u128(0)));
            assert_eq!(
                zero.contains(PartitionHash::from_u128(u128::MAX)),
                len == 0,
                "only the root covers the top hash (len {len})"
            );
        }
        // Interior boundary point at each width: the first hash outside `0*/len`.
        for len in [1u8, 63, 64, 65, 127] {
            let prefix = HashPrefix::new(0, len).expect("zero prefix");
            let first_outside = 1u128 << (128 - len);
            assert!(prefix.contains(PartitionHash::from_u128(first_outside - 1)));
            assert!(!prefix.contains(PartitionHash::from_u128(first_outside)));
        }
        // Top-bit prefix covers the upper half exactly.
        let upper = HashPrefix::new(1u128 << 127, 1).expect("1*");
        assert!(upper.contains(PartitionHash::from_u128(1u128 << 127)));
        assert!(upper.contains(PartitionHash::from_u128(u128::MAX)));
        assert!(!upper.contains(PartitionHash::from_u128(0)));
        // Length 127 distinguishes adjacent hashes.
        let low127 = HashPrefix::new(4, 127).expect("low 127 bits of 0b101");
        assert!(low127.contains(PartitionHash::from_u128(4)));
        assert!(low127.contains(PartitionHash::from_u128(5)));
        assert!(!low127.contains(PartitionHash::from_u128(6)));
        assert!(!low127.contains(PartitionHash::from_u128(3)));
        // Length 128 is an exact match.
        let exact = HashPrefix::new(0xDEAD_BEEF, 128).expect("exact");
        assert!(exact.contains(PartitionHash::from_u128(0xDEAD_BEEF)));
        assert!(!exact.contains(PartitionHash::from_u128(0xDEAD_BEF0)));
        assert_eq!(exact.prefix_len(), 128);
    }

    #[test]
    fn hash_overlap_is_symmetric_over_small_space() {
        let prefixes: Vec<HashPrefix> = (0..=2)
            .flat_map(|len| {
                (0..(1u128 << len)).map(move |cell| {
                    let bits = if len == 0 { 0 } else { cell << (128 - len) };
                    HashPrefix::new(bits, len).expect("canonical")
                })
            })
            .collect();
        assert_eq!(prefixes.len(), 1 + 2 + 4);
        for left in &prefixes {
            for right in &prefixes {
                assert_eq!(
                    left.overlaps(right),
                    right.overlaps(left),
                    "overlap must be symmetric"
                );
                if left.overlaps(right) {
                    // Witness: the longer prefix extended with zeros lies in both.
                    let (longer, shorter) = if left.prefix_len() >= right.prefix_len() {
                        (left, right)
                    } else {
                        (right, left)
                    };
                    assert!(shorter.contains(PartitionHash::from_u128(longer.bits())));
                } else {
                    // No 8-bit sample grid point lies in both.
                    for cell in 0..256u128 {
                        let hash = PartitionHash::from_u128(cell << 120);
                        assert!(
                            !(left.contains(hash) && right.contains(hash)),
                            "disjoint prefixes share {hash}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn ordered_ranges_follow_rfc_example() {
        // RFC §9: `["a", "g")`, `["g", "p")`, `["p", ∞)`.
        let a = OrderedRange::new(b"a".to_vec(), Some(b"g".to_vec())).expect("[a,g)");
        let b = OrderedRange::new(b"g".to_vec(), Some(b"p".to_vec())).expect("[g,p)");
        let c = OrderedRange::new(b"p".to_vec(), None).expect("[p,inf)");
        assert!(a.contains(b"a"));
        assert!(a.contains(b"f"));
        assert!(!a.contains(b"g"));
        assert!(b.contains(b"g"));
        assert!(!b.contains(b"p"));
        assert!(c.contains(b"p"));
        assert!(c.contains(b"zzz"));
        assert!(!a.overlaps(&b));
        assert!(!b.overlaps(&c));
        assert!(!a.overlaps(&c));
        assert!(a.overlaps(&a.clone()));
    }

    #[test]
    fn ordered_overlap_detects_partial_and_nested() {
        let outer = OrderedRange::new(b"a".to_vec(), Some(b"z".to_vec())).expect("outer");
        let inner = OrderedRange::new(b"m".to_vec(), Some(b"n".to_vec())).expect("inner");
        let tail = OrderedRange::new(b"y".to_vec(), None).expect("tail");
        assert!(outer.overlaps(&inner));
        assert!(inner.overlaps(&outer));
        assert!(outer.overlaps(&tail));
        // Adjacent half-open intervals do not overlap.
        let after = OrderedRange::new(b"z".to_vec(), None).expect("after");
        assert!(!outer.overlaps(&after));
    }

    #[test]
    fn ordered_rejects_empty_and_inverted() {
        assert_eq!(
            OrderedRange::new(b"g".to_vec(), Some(b"g".to_vec())),
            Err(RangeError::EmptyOrderedRange)
        );
        assert_eq!(
            OrderedRange::new(b"z".to_vec(), Some(b"a".to_vec())),
            Err(RangeError::EmptyOrderedRange)
        );
        // Empty end bound is below any nonempty start.
        assert_eq!(
            OrderedRange::new(b"a".to_vec(), Some(Vec::new())),
            Err(RangeError::EmptyOrderedRange)
        );
    }

    #[test]
    fn mixed_layouts_never_route_or_overlap() {
        let hash = PartitionRange::Hash(HashPrefix::new(0, 0).expect("root"));
        let ordered =
            PartitionRange::Ordered(OrderedRange::new(Vec::new(), None).expect("all keys"));
        assert!(!hash.contains_key(b"a"));
        assert!(!ordered.contains_hash(PartitionHash::from_u128(0)));
        assert!(!hash.overlaps(&ordered));
        assert_eq!(hash.kind(), PartitionKind::Hash);
        assert_eq!(ordered.kind(), PartitionKind::Ordered);
    }

    #[test]
    fn ordered_split_at_real_key_boundary() {
        let whole = OrderedRange::new(b"a".to_vec(), Some(b"z".to_vec())).expect("whole");
        let (left, right) = whole.split_at(b"m").expect("split at m");
        assert_eq!(left.start(), b"a");
        assert_eq!(left.end(), Some(b"m".as_slice()));
        assert_eq!(right.start(), b"m");
        assert_eq!(right.end(), Some(b"z".as_slice()));
        assert!(left.is_adjacent_to(&right));
        assert!(!left.overlaps(&right));
        assert_eq!(left.merge_with(&right).expect("merge"), whole);
        // Non-interior keys rejected: at/before start, at/past end.
        assert_eq!(whole.split_at(b"a"), Err(RangeError::InvalidSplitKey));
        assert_eq!(whole.split_at(b"z"), Err(RangeError::InvalidSplitKey));
        assert_eq!(whole.split_at(b"0"), Err(RangeError::InvalidSplitKey));
        assert_eq!(whole.split_at(b"zz"), Err(RangeError::InvalidSplitKey));
        // Unbounded tail splits at any key past start.
        let tail = OrderedRange::new(b"p".to_vec(), None).expect("tail");
        let (l, r) = tail.split_at(b"q").expect("split tail");
        assert_eq!(l.end(), Some(b"q".as_slice()));
        assert_eq!(r.end(), None);
        assert!(l.is_adjacent_to(&r));
        // Non-adjacent merge rejected.
        let far = OrderedRange::new(b"q".to_vec(), None).expect("far");
        assert_eq!(left.merge_with(&far), Err(RangeError::NotAdjacent));
    }
}
