//! Hash-domain types: content identity and partition routing input.
//!
//! [`ChunkId`] is a content address with a fixed, pinned construction (RFC
//! §29). [`PartitionHash`] is the opposite: an opaque 128-bit routing input
//! whose producing algorithm is intentionally undecided. The directory routes
//! on these values without ever choosing - or depending on - a hash function,
//! so adopting a concrete stable partition hash later changes producers only,
//! never the routing model.

use core::fmt;

/// Domain-separation tag mixed into every chunk hash.
///
/// Versioned (`V1`) so a future change of construction produces disjoint IDs
/// instead of colliding with existing content addresses.
pub const CHUNK_DOMAIN_TAG: &[u8] = b"KIVI-CHUNK-V1";

/// Domain-separation tag mixed into every chunk-manifest hash.
///
/// A manifest names an ordered list of chunks, not bytes, so it hashes in a
/// disjoint domain from chunks: a chunk can never alias a manifest address
/// even if their canonical bytes happened to coincide.
pub const MANIFEST_DOMAIN_TAG: &[u8] = b"KIVI-MANIFEST-V1";

/// 256-bit content address of an immutable chunk (RFC §28, §29).
///
/// Construction (pinned - this formula is the durable contract):
///
/// ```text
/// ChunkId = BLAKE3("KIVI-CHUNK-V1" || le64(security_domain) || chunk_bytes)
/// ```
///
/// Displayed as 64 lowercase hexadecimal digits. The all-zero value can never
/// be produced by the hash construction, so it serves as the invalid sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkId([u8; 32]);

impl ChunkId {
    /// All-zero sentinel. Never a real content address.
    pub const ZERO: Self = Self([0; 32]);

    /// Wraps 32 digest bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Returns the raw digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Whether this could be a real content address (i.e. is non-zero).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.0 != [0; 32]
    }
}

impl From<[u8; 32]> for ChunkId {
    fn from(bytes: [u8; 32]) -> Self {
        Self::from_bytes(bytes)
    }
}

impl From<ChunkId> for [u8; 32] {
    fn from(id: ChunkId) -> Self {
        id.0
    }
}

impl fmt::Display for ChunkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// 256-bit content address of an immutable chunk manifest (ordered chunk
/// list plus chunking parameters).
///
/// Construction (pinned - this formula is the durable contract):
///
/// ```text
/// ManifestId = BLAKE3("KIVI-MANIFEST-V1" || le64(security_domain) || canonical_manifest_bytes)
/// ```
///
/// Same sentinel rule as [`ChunkId`]: all-zero is never a real address.
/// Manifests and chunks live in disjoint domains, so identical bytes hash
/// to different addresses across the two types by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ManifestId([u8; 32]);

impl ManifestId {
    /// All-zero sentinel. Never a real content address.
    pub const ZERO: Self = Self([0; 32]);

    /// Wraps 32 digest bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Returns the raw digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Whether this could be a real content address (i.e. is non-zero).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.0 != [0; 32]
    }
}

impl From<[u8; 32]> for ManifestId {
    fn from(bytes: [u8; 32]) -> Self {
        Self::from_bytes(bytes)
    }
}

impl From<ManifestId> for [u8; 32] {
    fn from(id: ManifestId) -> Self {
        id.0
    }
}

impl fmt::Display for ManifestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Stable partition hash routing input: the 128-bit value a key maps to before
/// the adaptive prefix tree routes it to a tablet (RFC §9).
///
/// Deliberately opaque - no algorithm is pinned here. Producers (a future
/// key-hash layer) compute these; the directory only compares prefixes of
/// them. Fixing 128 bits now (rather than 64) reserves the full prefix space
/// for tablet growth without ever rehashing: deeper splits consume more bits
/// of the same hash. Displayed as 32 lowercase hexadecimal digits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PartitionHash(u128);

impl PartitionHash {
    /// Wraps a raw 128-bit hash value.
    #[must_use]
    pub const fn from_u128(value: u128) -> Self {
        Self(value)
    }

    /// Returns the raw 128-bit hash value.
    #[must_use]
    pub const fn as_u128(self) -> u128 {
        self.0
    }
}

impl From<PartitionHash> for u128 {
    /// Returns the raw 128-bit hash value.
    fn from(hash: PartitionHash) -> Self {
        hash.0
    }
}

impl From<u128> for PartitionHash {
    /// Wraps a raw 128-bit hash value.
    fn from(value: u128) -> Self {
        PartitionHash(value)
    }
}

impl fmt::Display for PartitionHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All-zero is the invalid sentinel for both content addresses, and the
    /// rendered form is exactly 64 lowercase hex digits - the width matters,
    /// because a shorter rendering would collide two distinct addresses in a
    /// log or a filename.
    #[test]
    fn content_addresses_are_fixed_width_hex_with_a_zero_sentinel() {
        let chunk = ChunkId::from_bytes([0xABu8; 32]);
        assert!(!ChunkId::ZERO.is_valid());
        assert!(chunk.is_valid());
        assert_eq!(chunk.as_bytes(), &[0xABu8; 32]);
        assert_eq!(<[u8; 32]>::from(chunk), [0xABu8; 32]);
        assert_eq!(chunk.to_string(), "ab".repeat(32));

        let manifest = ManifestId::from_bytes([0xCDu8; 32]);
        assert!(!ManifestId::ZERO.is_valid());
        assert!(manifest.is_valid());
        assert_eq!(manifest.to_string(), "cd".repeat(32));

        // Ordering is byte-lexicographic, which is what manifest sorting relies on.
        assert!(ChunkId::from_bytes([0x00; 32]) < ChunkId::from_bytes([0xFF; 32]));
        assert!(ManifestId::ZERO < manifest);
    }

    /// The routing hash renders as 32 hex digits and round-trips as a `u128`.
    /// It is compared by prefix for routing, so a rendering that dropped
    /// leading zeros would make two distinct hashes print identically.
    #[test]
    fn partition_hash_renders_as_32_hex_digits() {
        let hash = PartitionHash::from_u128(0x1234_ABCD);
        assert_eq!(hash.as_u128(), 0x1234_ABCD);
        assert_eq!(u128::from(hash), 0x1234_ABCD);
        assert_eq!(hash.to_string(), "0000000000000000000000001234abcd");
        assert_eq!(
            PartitionHash::from_u128(u128::MAX).to_string(),
            "f".repeat(32)
        );
    }
}
