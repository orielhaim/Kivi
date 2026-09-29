//! Chunk codec identifiers.
//!
//! `ChunkId` always names canonical *logical* bytes: the address is computed
//! before any physical encoding, so a future compressed representation of
//! the same logical chunk keeps the same identity and existing manifests
//! keep resolving. The record carries both the logical and stored lengths
//! plus this codec so decoders know how to recover the logical bytes the id
//! names.
//!
//! `NONE` is the only codec wired here. Every other assignment fails
//! verification: this build never guesses past a format it does not own.

/// Opaque chunk-body codec assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkCodecId(u8);

impl ChunkCodecId {
    /// Stored bytes are the logical bytes verbatim. The only codec this
    /// mints or reads.
    pub const NONE: Self = Self(0);

    /// Wraps a raw assignment, including unknown future values.
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        Self(value)
    }

    /// Returns the raw assignment.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self.0
    }

    /// Whether this build encodes and decodes this codec.
    #[must_use]
    pub const fn is_supported(self) -> bool {
        matches!(self.0, 0)
    }
}

impl core::fmt::Display for ChunkCodecId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0 => write!(f, "none"),
            other => write!(f, "unknown-codec({other})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_zero_is_supported() {
        assert!(ChunkCodecId::NONE.is_supported());
        assert!(!ChunkCodecId::from_u8(1).is_supported());
        assert!(!ChunkCodecId::from_u8(0x7F).is_supported());
        assert_eq!(ChunkCodecId::NONE.as_u8(), 0);
    }
}
