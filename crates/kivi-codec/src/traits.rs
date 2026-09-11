//! Canonical encode/decode traits and primitive implementations.
//!
//! All multi-byte integers are little-endian. Variable-length blobs are a
//! `u32` little-endian length prefix followed by raw bytes, bounded by
//! [`MAX_FRAME_BYTES`].

use crate::MAX_FRAME_BYTES;
use crate::error::CodecError;

/// Canonical binary encoding into a byte buffer.
pub trait Encode {
    /// Exact number of bytes [`encode`](Self::encode) will append.
    #[must_use]
    fn encoded_len(&self) -> usize;

    /// Appends the canonical encoding to `out`.
    fn encode(&self, out: &mut Vec<u8>);

    /// Encodes to a freshly allocated, exactly sized vector.
    #[must_use]
    fn encode_to_vec(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.encoded_len());
        self.encode(&mut out);
        out
    }
}

/// Canonical binary decoding from a byte slice.
///
/// Returns the value plus the number of bytes consumed, so frames can be
/// parsed sequentially without copying.
pub trait Decode: Sized {
    /// Decodes one value from the prefix of `input`.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`] if the input is truncated, out of bounds, or
    /// fails validation. Never panics on untrusted input.
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError>;

    /// Decodes one value requiring the input to be consumed exactly.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`] on any decode failure, or
    /// [`CodecError::TrailingBytes`] if bytes remain after the value.
    fn decode_exact(input: &[u8]) -> Result<Self, CodecError> {
        let (value, consumed) = Self::decode(input)?;
        if consumed == input.len() {
            Ok(value)
        } else {
            Err(CodecError::TrailingBytes {
                consumed,
                total: input.len(),
            })
        }
    }
}

impl Encode for u8 {
    fn encoded_len(&self) -> usize {
        1
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.push(*self);
    }
}

impl Decode for u8 {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let [byte] = input.first_chunk().ok_or(CodecError::Truncated {
            expected: 1,
            available: input.len(),
        })?;
        Ok((*byte, 1))
    }
}

impl Encode for u16 {
    fn encoded_len(&self) -> usize {
        2
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl Decode for u16 {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let bytes = input.first_chunk().ok_or(CodecError::Truncated {
            expected: 2,
            available: input.len(),
        })?;
        Ok((Self::from_le_bytes(*bytes), 2))
    }
}

impl Encode for u32 {
    fn encoded_len(&self) -> usize {
        4
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl Decode for u32 {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let bytes = input.first_chunk().ok_or(CodecError::Truncated {
            expected: 4,
            available: input.len(),
        })?;
        Ok((Self::from_le_bytes(*bytes), 4))
    }
}

impl Encode for u64 {
    fn encoded_len(&self) -> usize {
        8
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl Decode for u64 {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let bytes = input.first_chunk().ok_or(CodecError::Truncated {
            expected: 8,
            available: input.len(),
        })?;
        Ok((Self::from_le_bytes(*bytes), 8))
    }
}

impl Encode for u128 {
    fn encoded_len(&self) -> usize {
        16
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl Decode for u128 {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let bytes = input.first_chunk().ok_or(CodecError::Truncated {
            expected: 16,
            available: input.len(),
        })?;
        Ok((Self::from_le_bytes(*bytes), 16))
    }
}

impl Encode for bool {
    fn encoded_len(&self) -> usize {
        1
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.push(u8::from(*self));
    }
}

impl Decode for bool {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (byte, consumed) = u8::decode(input)?;
        match byte {
            0 => Ok((false, consumed)),
            1 => Ok((true, consumed)),
            other => Err(CodecError::InvalidBoolEncoding { byte: other }),
        }
    }
}

impl<const N: usize> Encode for [u8; N] {
    fn encoded_len(&self) -> usize {
        N
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self);
    }
}

impl<const N: usize> Decode for [u8; N] {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let bytes: &[u8; N] = input.first_chunk().ok_or(CodecError::Truncated {
            expected: N,
            available: input.len(),
        })?;
        Ok((*bytes, N))
    }
}

/// Appends a length-prefixed blob: `u32` little-endian length plus raw bytes.
///
/// # Panics
///
/// Panics if `bytes` exceeds [`MAX_FRAME_BYTES`].
/// Callers with untrusted or unbounded input must check the bound first and
/// return [`CodecError::TooLarge`] instead of calling this.
pub fn encode_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    assert!(
        bytes.len() <= MAX_FRAME_BYTES,
        "blob of {} bytes exceeds maximum {MAX_FRAME_BYTES}",
        bytes.len()
    );
    let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
    len.encode(out);
    out.extend_from_slice(bytes);
}

/// Decodes a length-prefixed blob as a borrowed slice.
///
/// Prefer this on read paths (zero copy); use [`decode_byte_vec`] when
/// ownership is required.
///
/// # Errors
///
/// Returns [`CodecError`] if the length prefix is truncated, declares more
/// than [`MAX_FRAME_BYTES`], or runs past the input.
pub fn decode_byte_slice(input: &[u8]) -> Result<(&[u8], usize), CodecError> {
    let (len, prefix) = u32::decode(input)?;
    let len = usize::try_from(len).unwrap_or(MAX_FRAME_BYTES);
    if len > MAX_FRAME_BYTES {
        return Err(CodecError::TooLarge {
            len,
            max: MAX_FRAME_BYTES,
        });
    }
    let end = prefix.checked_add(len).ok_or(CodecError::TooLarge {
        len,
        max: MAX_FRAME_BYTES,
    })?;
    if input.len() < end {
        return Err(CodecError::Truncated {
            expected: end,
            available: input.len(),
        });
    }
    Ok((&input[prefix..end], end))
}

/// Decodes a length-prefixed blob into an owned vector.
///
/// Same validation as [`decode_byte_slice`]; copies the payload.
///
/// # Errors
///
/// Returns [`CodecError`] on truncation, oversize length, or overrun.
pub fn decode_byte_vec(input: &[u8]) -> Result<(Vec<u8>, usize), CodecError> {
    let (slice, consumed) = decode_byte_slice(input)?;
    Ok((slice.to_vec(), consumed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_round_trip_little_endian() {
        for value in [0u64, 1, 918, u64::MAX] {
            let bytes = value.encode_to_vec();
            assert_eq!(bytes, value.to_le_bytes());
            assert_eq!(u64::decode_exact(&bytes).expect("round trip"), value);
        }
        let wide = u128::from(0xDEAD_BEEF_u32);
        assert_eq!(wide.encode_to_vec().len(), 16);
        assert_eq!(
            u128::decode_exact(&wide.encode_to_vec()).expect("u128"),
            wide
        );
        assert_eq!(
            u16::decode_exact(&0x0102u16.to_le_bytes()).expect("u16"),
            0x0102
        );
        assert_eq!(
            u32::decode_exact(&0x0102_0304u32.to_le_bytes()).expect("u32"),
            0x0102_0304
        );
    }

    #[test]
    fn bool_accepts_only_zero_and_one() {
        assert!(!bool::decode_exact(&[0]).expect("false"));
        assert!(bool::decode_exact(&[1]).expect("true"));
        assert_eq!(
            bool::decode(&[2]),
            Err(CodecError::InvalidBoolEncoding { byte: 2 })
        );
    }

    #[test]
    fn truncation_reports_expected_and_available() {
        assert_eq!(
            u64::decode(&[1, 2, 3]),
            Err(CodecError::Truncated {
                expected: 8,
                available: 3
            })
        );
        assert_eq!(
            u64::decode_exact(&[0u8; 9]),
            Err(CodecError::TrailingBytes {
                consumed: 8,
                total: 9
            })
        );
    }

    #[test]
    fn blobs_round_trip_with_length_prefix() {
        let payload = vec![7u8; 1024];
        let mut out = Vec::new();
        encode_bytes(&mut out, &payload);
        assert_eq!(out.len(), 4 + 1024);
        let (slice, consumed) = decode_byte_slice(&out).expect("slice");
        assert_eq!(slice, payload.as_slice());
        assert_eq!(consumed, out.len());
        let (owned, _) = decode_byte_vec(&out).expect("vec");
        assert_eq!(owned, payload);
    }

    #[test]
    fn blob_decode_rejects_lies_about_length() {
        let mut bad = 10_000u32.to_le_bytes().to_vec();
        bad.extend_from_slice(&[0u8; 8]);
        assert!(matches!(
            decode_byte_slice(&bad),
            Err(CodecError::Truncated { .. })
        ));
        let huge = u32::try_from(MAX_FRAME_BYTES + 1).expect("bound fits in u32");
        assert!(matches!(
            decode_byte_slice(&huge.to_le_bytes()),
            Err(CodecError::TooLarge { .. })
        ));
    }

    #[test]
    #[should_panic(expected = "exceeds maximum")]
    fn blob_encode_panics_over_bound() {
        encode_bytes(&mut Vec::new(), &vec![0u8; MAX_FRAME_BYTES + 1]);
    }
}
