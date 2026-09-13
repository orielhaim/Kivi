//! Chunk body codec identifiers.
//!
//! `ChunkId` always names canonical *logical* bytes (§2): the address is
//! computed before any physical encoding, so a future compressed
//! representation of the same logical chunk keeps the same identity and
//! existing manifests keep resolving. The record carries both the logical
//! and stored lengths plus this codec so decoders know how to recover the
//! logical bytes the id names.
//!
//! This stage wires `NONE` only. `LZ4` keeps its reserved assignment so a
//! future experiment cannot silently collide, but builds and loads reject
//! it explicitly until measured.

/// Opaque chunk-body codec assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkCodecId(u8);

impl ChunkCodecId {
    /// Stored bytes are the logical bytes verbatim. The only codec this
    /// stage mints or reads.
    pub const NONE: Self = Self(0);
    /// Reserved assignment for a future LZ4 experiment. Rejected by
    /// [`is_supported`](Self::is_supported) until then.
    pub const LZ4: Self = Self(1);

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
            1 => write!(f, "lz4(reserved)"),
            other => write!(f, "unknown-codec({other})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_none_is_supported_this_stage() {
        assert!(ChunkCodecId::NONE.is_supported());
        assert!(!ChunkCodecId::LZ4.is_supported());
        assert!(!ChunkCodecId::from_u8(0x7F).is_supported());
        assert_eq!(ChunkCodecId::NONE.as_u8(), 0);
    }
}
