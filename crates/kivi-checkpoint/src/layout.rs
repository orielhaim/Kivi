//! Physical checkpoint band layout: stable key-to-band mapping.
//!
//! A checkpoint divides each tablet into immutable bands so incremental
//! checkpoints rewrite only what changed. Band mapping is a PHYSICAL
//! format decision: it never affects logical routing, but it is versioned
//! and pinned like every other format choice — a layout change mints a
//! new [`BandLayoutId`], and manifests record which layout they use.
//!
//! Layout V1 (`hash-prefix-8`):
//!
//! ```text
//! band = partition_hash >> 120   (top 8 bits → 256 bands)
//! ```
//!
//! The partition hash already spreads keys uniformly (BLAKE3-128), so the
//! top byte spreads bands uniformly too. For hash tablets this derives
//! from the partition hash beneath the tablet prefix; ordered layouts
//! must supply their own explicitly versioned physical function rather
//! than coupling band boundaries to range semantics.

use kivi_state::PartitionHasher;
use kivi_types::{NamespaceId, PartitionHash};

/// Bands per tablet in layout V1. A `[u64; 4]` bitset tracks a tablet's
/// dirty bands without per-mutation allocation.
pub const BANDS_PER_TABLET: usize = 256;

/// Bands per tablet as `u16` (wire width for band ids).
pub const BAND_COUNT_U16: u16 = 256;

const _: () = assert!(BANDS_PER_TABLET == BAND_COUNT_U16 as usize);

/// Words in a 256-bit dirty-band bitset.
pub const DIRTY_WORDS: usize = BANDS_PER_TABLET / 64;

/// Identifies the physical band-mapping algorithm behind a checkpoint.
/// Stored in every manifest; unknown ids fail loading loudly (never guess
/// a physical layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BandLayoutId(u16);

impl BandLayoutId {
    /// No layout (unknown provenance).
    pub const NONE: Self = Self(0);
    /// V1: top 8 bits of the V1 partition hash (256 bands).
    pub const HASH_PREFIX_8: Self = Self(1);

    /// Wraps a raw value, including unknown future assignments.
    #[must_use]
    pub const fn from_u16(value: u16) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        self.0
    }

    /// Whether this build can map and verify this layout.
    #[must_use]
    pub const fn is_supported(self) -> bool {
        matches!(self.0, 1)
    }
}

impl core::fmt::Display for BandLayoutId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0 => write!(f, "none"),
            1 => write!(f, "hash-prefix-8-v1"),
            other => write!(f, "unknown-band-layout({other})"),
        }
    }
}

/// Versioned physical band mapper. The hasher is fixed to
/// [`PartitionHasher::V1`] in layout V1: band identity must be as stable
/// as the partition hash itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BandLayout {
    id: BandLayoutId,
}

impl BandLayout {
    /// The current production layout (V1).
    pub const V1: Self = Self {
        id: BandLayoutId::HASH_PREFIX_8,
    };

    /// Builds a mapper for an explicitly named layout.
    #[must_use]
    pub const fn new(id: BandLayoutId) -> Self {
        Self { id }
    }

    /// Returns the layout identity.
    #[must_use]
    pub const fn id(self) -> BandLayoutId {
        self.id
    }

    /// Maps `key` to its band (`0..256`). Returns `None` for layouts this
    /// build cannot produce rather than guessing a physical placement.
    #[must_use]
    pub fn band_of(&self, namespace: NamespaceId, key: &[u8]) -> Option<u16> {
        match self.id {
            BandLayoutId::HASH_PREFIX_8 => {
                let hash: PartitionHash = PartitionHasher::V1.hash(namespace, key)?;
                Some((hash.as_u128() >> 120) as u16)
            }
            _ => None,
        }
    }
}

/// Compact dirty-band state: one bit per band. Mutation apply sets bits;
/// checkpoint capture takes them; preparation of the next checkpoint
/// diffs against the last published set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BandDirty {
    words: [u64; DIRTY_WORDS],
}

impl BandDirty {
    /// No dirty bands.
    #[must_use]
    pub fn clean() -> Self {
        Self {
            words: [0; DIRTY_WORDS],
        }
    }

    /// Marks `band` dirty.
    pub fn mark(&mut self, band: u16) {
        let band = (band as usize) % BANDS_PER_TABLET;
        self.words[band / 64] |= 1u64 << (band % 64);
    }

    /// Whether `band` is dirty.
    #[must_use]
    pub fn is_dirty(&self, band: u16) -> bool {
        let band = (band as usize) % BANDS_PER_TABLET;
        self.words[band / 64] & (1u64 << (band % 64)) != 0
    }

    /// Whether no band is dirty.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }

    /// Counts dirty bands.
    #[must_use]
    pub fn dirty_count(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// Clears all bits (after a successful checkpoint covers them).
    pub fn clear(&mut self) {
        self.words = [0; DIRTY_WORDS];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NS: NamespaceId = NamespaceId::from_u64(9);

    #[test]
    fn v1_covers_all_256_bands() {
        let layout = BandLayout::V1;
        let mut seen = [false; BANDS_PER_TABLET];
        for index in 0..20_000u32 {
            let band = layout
                .band_of(NS, format!("key:{index}").as_bytes())
                .expect("supported") as usize;
            seen[band] = true;
        }
        assert!(
            seen.iter().all(|seen| *seen),
            "uniform hash must reach every band"
        );
    }

    #[test]
    fn band_mapping_is_deterministic_and_namespaced() {
        let layout = BandLayout::V1;
        let first = layout.band_of(NS, b"user:1").expect("supported");
        assert_eq!(layout.band_of(NS, b"user:1").expect("supported"), first);
        assert_ne!(
            layout
                .band_of(NamespaceId::from_u64(10), b"user:1")
                .expect("supported"),
            first,
            "namespace participates in band placement"
        );
    }

    #[test]
    fn unknown_layouts_fail_explicitly() {
        let layout = BandLayout::new(BandLayoutId::from_u16(0xBEEF));
        assert!(!layout.id().is_supported());
        assert_eq!(layout.band_of(NS, b"k"), None);
    }

    #[test]
    fn dirty_bitset_tracks_without_allocation() {
        let mut dirty = BandDirty::clean();
        assert!(dirty.is_clean());
        dirty.mark(0);
        dirty.mark(255);
        dirty.mark(255);
        assert!(dirty.is_dirty(0));
        assert!(dirty.is_dirty(255));
        assert!(!dirty.is_dirty(1));
        assert_eq!(dirty.dirty_count(), 2);
        dirty.clear();
        assert!(dirty.is_clean());
    }
}
