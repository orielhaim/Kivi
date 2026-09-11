//! Stored integrity-algorithm identifiers (RFC §54).
//!
//! The wire stores *which* algorithm produced each checksum or content hash;
//! no algorithm is assumed permanent for the project's lifetime. Unknown IDs
//! remain representable so capability negotiation (RFC §226) can name them;
//! use sites decide support via [`HashAlgorithmId::is_supported`] and friends.
//!
//! Canonical assignments:
//!
//! | ID | Hash algorithm | Checksum algorithm |
//! |----|----------------|--------------------|
//! | 0  | none           | none               |
//! | 1  | BLAKE3-256     | CRC32C (Castagnoli)|

use core::fmt;

use crate::error::CodecError;
use crate::traits::{Decode, Encode};

/// Identifies the strong-hash algorithm behind a content identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HashAlgorithmId(u16);

impl HashAlgorithmId {
    /// No strong hash present.
    pub const NONE: Self = Self(0);
    /// BLAKE3-256 (RFC §54 default for content identity).
    pub const BLAKE3_256: Self = Self(1);

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

    /// Whether this decoder can verify content hashed with this algorithm.
    #[must_use]
    pub const fn is_supported(self) -> bool {
        matches!(self.0, 1)
    }
}

impl fmt::Display for HashAlgorithmId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            0 => write!(f, "none"),
            1 => write!(f, "blake3-256"),
            other => write!(f, "unknown-hash({other})"),
        }
    }
}

impl Encode for HashAlgorithmId {
    fn encoded_len(&self) -> usize {
        2
    }

    fn encode(&self, out: &mut Vec<u8>) {
        self.0.encode(out);
    }
}

impl Decode for HashAlgorithmId {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (value, consumed) = u16::decode(input)?;
        Ok((Self::from_u16(value), consumed))
    }
}

/// Identifies the checksum algorithm behind a corruption-detection field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChecksumAlgorithmId(u16);

impl ChecksumAlgorithmId {
    /// No checksum present.
    pub const NONE: Self = Self(0);
    /// CRC32C Castagnoli (RFC §54 default for accidental-corruption detection).
    pub const CRC32C: Self = Self(1);

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

    /// Whether this decoder can verify checksums of this algorithm.
    #[must_use]
    pub const fn is_supported(self) -> bool {
        matches!(self.0, 1)
    }
}

impl fmt::Display for ChecksumAlgorithmId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            0 => write!(f, "none"),
            1 => write!(f, "crc32c"),
            other => write!(f, "unknown-checksum({other})"),
        }
    }
}

impl Encode for ChecksumAlgorithmId {
    fn encoded_len(&self) -> usize {
        2
    }

    fn encode(&self, out: &mut Vec<u8>) {
        self.0.encode(out);
    }
}

impl Decode for ChecksumAlgorithmId {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (value, consumed) = u16::decode(input)?;
        Ok((Self::from_u16(value), consumed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_algorithms_round_trip_and_report_supported() {
        assert!(HashAlgorithmId::BLAKE3_256.is_supported());
        assert!(!HashAlgorithmId::NONE.is_supported());
        assert!(ChecksumAlgorithmId::CRC32C.is_supported());
        let encoded = HashAlgorithmId::BLAKE3_256.encode_to_vec();
        assert_eq!(
            HashAlgorithmId::decode_exact(&encoded).expect("round trip"),
            HashAlgorithmId::BLAKE3_256
        );
    }

    #[test]
    fn unknown_ids_survive_the_wire_for_negotiation() {
        let unknown = HashAlgorithmId::from_u16(0xBEEF);
        assert!(!unknown.is_supported());
        assert_eq!(unknown.to_string(), "unknown-hash(48879)");
        assert_eq!(
            HashAlgorithmId::decode_exact(&unknown.encode_to_vec()).expect("opaque"),
            unknown
        );
        assert_eq!(
            ChecksumAlgorithmId::from_u16(7).to_string(),
            "unknown-checksum(7)"
        );
    }
}
