//! Project-owned partition hashing boundary (RFC §9).
//!
//! Routing needs a stable `key bytes → 128-bit hash` function that never
//! changes under a declared algorithm version. The directory consumes
//! [`PartitionHash`] values without knowing their producer; this module owns
//! the producers behind a versioned [`PartitionHasher`].
//!
//! V1 (`BLAKE3-128`, [`PartitionHashAlgorithmId::BLAKE3_128`]) derives from
//! BLAKE3, which the project already trusts for content identity:
//!
//! ```text
//! digest = BLAKE3("KIVI-PART-V1" || le64(namespace) || key_bytes)
//! hash   = le128(digest[0..16])
//! ```
//!
//! Design evaluation before locking V1 in: BLAKE3 costs on the order of
//! 100ns per small key — measurable but small next to channel hops and
//! tablet execution, and it buys platform-independent stability plus
//! preimage strength hostile keys cannot game into hotspots beyond mere
//! co-location (which degrades balance only — full key equality always
//! decides membership, so correctness never depends on hash quality).
//! A faster V2 (e.g. hardware CRC or short-input specialist) can be added
//! behind a new algorithm id with dual-hash migration; the directory and
//! prefix model already span the full 128 bits, so no rehash of stored
//! state is ever structural. Rust's randomized `Hash` output is never used
//! as routing identity anywhere in this path.

use core::fmt;

use kivi_codec::{CodecError, Decode, Encode};
use kivi_types::{NamespaceId, PartitionHash};

/// Domain-separation tag mixed into every V1 partition hash. Versioned so a
/// future construction change yields disjoint hashes by construction.
pub const PARTITION_DOMAIN_TAG: &[u8] = b"KIVI-PART-V1";

/// Identifies the partition-hash algorithm behind a routing decision.
/// Stored wherever routing provenance is recorded; unknown ids remain
/// representable for future capability negotiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PartitionHashAlgorithmId(u16);

impl PartitionHashAlgorithmId {
    /// No algorithm (hash provenance unknown).
    pub const NONE: Self = Self(0);
    /// V1: BLAKE3-derived 128-bit hash defined in this module.
    pub const BLAKE3_128: Self = Self(1);

    /// Wraps a raw wire value, including unknown future assignments.
    #[must_use]
    pub const fn from_u16(value: u16) -> Self {
        Self(value)
    }

    /// Returns the raw wire value.
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        self.0
    }

    /// Whether this implementation can produce and verify this algorithm.
    #[must_use]
    pub const fn is_supported(self) -> bool {
        matches!(self.0, 1)
    }
}

impl fmt::Display for PartitionHashAlgorithmId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            0 => write!(f, "none"),
            1 => write!(f, "blake3-128-v1"),
            other => write!(f, "unknown-partition-hash({other})"),
        }
    }
}

impl Encode for PartitionHashAlgorithmId {
    fn encoded_len(&self) -> usize {
        2
    }

    fn encode(&self, out: &mut Vec<u8>) {
        self.0.encode(out);
    }
}

impl Decode for PartitionHashAlgorithmId {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (value, consumed) = u16::decode(input)?;
        Ok((Self::from_u16(value), consumed))
    }
}

/// Versioned key-to-partition-hash function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PartitionHasher {
    algorithm: PartitionHashAlgorithmId,
}

impl PartitionHasher {
    /// The current production hasher (V1).
    pub const V1: Self = Self {
        algorithm: PartitionHashAlgorithmId::BLAKE3_128,
    };

    /// Builds a hasher for an explicitly named algorithm.
    #[must_use]
    pub const fn new(algorithm: PartitionHashAlgorithmId) -> Self {
        Self { algorithm }
    }

    /// Returns the algorithm this hasher implements.
    #[must_use]
    pub const fn algorithm(self) -> PartitionHashAlgorithmId {
        self.algorithm
    }

    /// Hashes `key` within `namespace` to a stable 128-bit partition hash.
    ///
    /// Deterministic across processes, platforms, and releases for a fixed
    /// algorithm version. Returns `None` for algorithms this build cannot
    /// produce rather than guessing with the wrong construction.
    #[must_use]
    pub fn hash(&self, namespace: NamespaceId, key: &[u8]) -> Option<PartitionHash> {
        match self.algorithm {
            PartitionHashAlgorithmId::BLAKE3_128 => {
                let mut hasher = blake3::Hasher::new();
                hasher.update(PARTITION_DOMAIN_TAG);
                hasher.update(&namespace.as_u64().to_le_bytes());
                hasher.update(key);
                let digest = hasher.finalize();
                let mut raw = [0u8; 16];
                raw.copy_from_slice(&digest.as_bytes()[..16]);
                Some(PartitionHash::from_u128(u128::from_le_bytes(raw)))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NS: NamespaceId = NamespaceId::from_u64(7);

    #[test]
    fn v1_is_deterministic_and_namespace_scoped() {
        let hasher = PartitionHasher::V1;
        let first = hasher.hash(NS, b"user:123").expect("supported");
        assert_eq!(hasher.hash(NS, b"user:123").expect("supported"), first);
        assert_ne!(hasher.hash(NS, b"user:124").expect("supported"), first);
        assert_ne!(
            hasher
                .hash(NamespaceId::from_u64(8), b"user:123")
                .expect("supported"),
            first,
            "namespace participates in the hash"
        );
    }

    #[test]
    fn frozen_vector_pins_v1_forever() {
        // Generated by this exact construction; any intentional algorithm
        // change mints a new algorithm id, never edits these bytes.
        assert_eq!(
            PartitionHasher::V1.hash(NS, b"hello").expect("supported"),
            PartitionHash::from_u128(0xfad1_5e9b_f890_834c_93fc_602e_ae3e_6c35)
        );
        assert_eq!(
            PartitionHasher::V1.hash(NS, b"").expect("supported"),
            PartitionHasher::V1.hash(NS, b"").expect("supported"),
            "empty keys hash deterministically"
        );
    }

    #[test]
    fn unknown_algorithms_fail_explicitly() {
        let hasher = PartitionHasher::new(PartitionHashAlgorithmId::from_u16(0xBEEF));
        assert!(!hasher.algorithm().is_supported());
        assert_eq!(hasher.hash(NS, b"hello"), None);
        assert_eq!(
            PartitionHashAlgorithmId::decode_exact(
                &PartitionHashAlgorithmId::BLAKE3_128.encode_to_vec()
            )
            .expect("round trip"),
            PartitionHashAlgorithmId::BLAKE3_128
        );
    }
}
