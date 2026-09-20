//! Compressed DRAM as an explicit representation (not swap).
//!
//! `lz4_flex` is the baseline codec (borrowed mechanism; Kivi owns the
//! block framing). Compression is conditional: a candidate block is
//! compressed, its ratio and CPU cost measured, and the representation
//! adopted only when savings exceed [`COMPRESSED_MIN_SAVINGS_BPS`] and the
//! object is not write-hot. Decompression is verified by round-trip
//! equality before publication.

use bytes::Bytes;

use crate::error::MemoryError;

/// Minimum savings in basis points for adopting compression: the stored
/// block must be at least 15% smaller than the logical bytes.
pub const COMPRESSED_MIN_SAVINGS_BPS: u64 = 1_500;

/// Maximum logical bytes accepted for compression (the medium ceiling).
pub const COMPRESSED_MAX_LOGICAL: usize = crate::MEDIUM_MAX;

/// Framing magic for one compressed block: `KZ42`.
pub const COMPRESSED_MAGIC: u32 = 0x5A4B_3432;

/// Codec identifier for `lz4_flex` block compression, version 1.
pub const COMPRESSED_CODEC_LZ4: u8 = 1;

/// One compressed block: self-describing framing plus the codec payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedBlock {
    /// Logical (uncompressed) length.
    pub logical_len: u32,
    /// Stored framing + payload bytes.
    pub stored: Bytes,
    /// Measured compression ratio in basis points (stored/logical).
    pub ratio_bps: u64,
}

impl CompressedBlock {
    /// Compresses `logical` when worthwhile, returning `None` when the
    /// block is too small, incompressible, or below the savings floor.
    ///
    /// Small inputs (`<= 64` bytes) are rejected without measuring: the
    /// framing overhead alone exceeds any plausible saving, and the tiny
    /// fast path must never pay compression setup.
    #[must_use]
    pub fn compress_if_worthwhile(logical: &Bytes) -> Option<Self> {
        if logical.len() <= 64 || logical.len() > COMPRESSED_MAX_LOGICAL {
            return None;
        }
        let payload = lz4_flex::block::compress(logical.as_ref());
        let stored_len = 16 + payload.len();
        let logical_len = logical.len();
        // Basis points without floating point. Lengths are bounded by
        // COMPRESSED_MAX_LOGICAL (256 KiB), far below u64 precision
        // limits, so widening casts here are exact.
        #[allow(clippy::cast_possible_truncation)]
        let ratio_bps = (stored_len as u64 * 10_000) / (logical_len as u64);
        if ratio_bps > 10_000 - COMPRESSED_MIN_SAVINGS_BPS {
            return None;
        }
        let Ok(logical_u32) = u32::try_from(logical_len) else {
            return None;
        };
        let Ok(payload_u32) = u32::try_from(payload.len()) else {
            return None;
        };
        let mut stored = Vec::with_capacity(stored_len);
        stored.extend_from_slice(&COMPRESSED_MAGIC.to_le_bytes());
        stored.push(COMPRESSED_CODEC_LZ4);
        stored.push(0);
        stored.push(0);
        stored.push(0);
        stored.extend_from_slice(&logical_u32.to_le_bytes());
        stored.extend_from_slice(&payload_u32.to_le_bytes());
        stored.extend_from_slice(&payload);
        Some(Self {
            logical_len: logical_u32,
            stored: Bytes::from(stored),
            ratio_bps,
        })
    }

    /// Decompresses and verifies a block.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::CorruptRepresentation`] on framing, codec,
    /// or length mismatch, and [`MemoryError::TooLarge`] when the
    /// declared logical length exceeds the medium ceiling.
    pub fn decompress(&self) -> Result<Bytes, MemoryError> {
        decode_stored(&self.stored)
    }

    /// Stored length in bytes.
    #[must_use]
    pub const fn stored_len(&self) -> usize {
        self.stored.len()
    }
}

/// Decodes framed stored bytes produced by [`CompressedBlock`].
///
/// # Errors
///
/// Returns [`MemoryError::CorruptRepresentation`] on any framing or codec
/// mismatch.
pub fn decode_stored(stored: &Bytes) -> Result<Bytes, MemoryError> {
    if stored.len() < 16 {
        return Err(MemoryError::CorruptRepresentation {
            object: u64::MAX,
            detail: "compressed block shorter than framing".to_owned(),
        });
    }
    let magic = u32::from_le_bytes([stored[0], stored[1], stored[2], stored[3]]);
    if magic != COMPRESSED_MAGIC {
        return Err(MemoryError::CorruptRepresentation {
            object: u64::MAX,
            detail: "compressed block magic mismatch".to_owned(),
        });
    }
    if stored[4] != COMPRESSED_CODEC_LZ4 {
        return Err(MemoryError::Unsupported {
            detail: "unknown compressed block codec".to_owned(),
        });
    }
    let logical_len = u32::from_le_bytes([stored[8], stored[9], stored[10], stored[11]]) as usize;
    let payload_len = u32::from_le_bytes([stored[12], stored[13], stored[14], stored[15]]) as usize;
    if logical_len > COMPRESSED_MAX_LOGICAL {
        return Err(MemoryError::TooLarge {
            len: logical_len as u64,
            max: COMPRESSED_MAX_LOGICAL as u64,
            context: "compressed logical length",
        });
    }
    if stored.len() != 16 + payload_len {
        return Err(MemoryError::CorruptRepresentation {
            object: u64::MAX,
            detail: "compressed payload length mismatch".to_owned(),
        });
    }
    let payload = &stored[16..];
    let decoded = lz4_flex::block::decompress(payload, logical_len).map_err(|error| {
        MemoryError::CorruptRepresentation {
            object: u64::MAX,
            detail: format!("lz4 decompression failed: {error}"),
        }
    })?;
    if decoded.len() != logical_len {
        return Err(MemoryError::CorruptRepresentation {
            object: u64::MAX,
            detail: "decompressed length mismatch".to_owned(),
        });
    }
    Ok(Bytes::from(decoded))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_inputs_are_rejected_without_measuring() {
        let small = Bytes::from_static(b"hi");
        assert!(CompressedBlock::compress_if_worthwhile(&small).is_none());
    }

    #[test]
    fn compressible_text_roundtrips() {
        let logical = Bytes::from(vec![b'a'; 4096]);
        let block = CompressedBlock::compress_if_worthwhile(&logical).expect("compresses");
        assert!(block.ratio_bps < 8_500);
        let decoded = block.decompress().expect("decompresses");
        assert_eq!(decoded, logical);
    }

    #[test]
    fn incompressible_bytes_are_rejected() {
        // Deterministic pseudorandom bytes (xorshift, fixed seed).
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut bytes = vec![0u8; 4096];
        for byte in &mut bytes {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = u8::try_from((state >> 11) & 0xFF).expect("masked to one byte");
        }
        let logical = Bytes::from(bytes);
        assert!(CompressedBlock::compress_if_worthwhile(&logical).is_none());
    }

    #[test]
    fn framing_corruption_is_detected() {
        let logical = Bytes::from(vec![b'z'; 1024]);
        let block = CompressedBlock::compress_if_worthwhile(&logical).expect("compresses");
        let mut raw = block.stored.to_vec();
        raw[0] ^= 0xFF;
        let corrupt = Bytes::from(raw);
        assert!(decode_stored(&corrupt).is_err());
    }
}
