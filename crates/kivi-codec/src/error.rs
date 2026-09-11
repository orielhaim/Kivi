//! Codec failure modes.
//!
//! Every decode failure names what was wrong, never panics on untrusted
//! input, and fails closed: truncated, corrupt, or version-unknown bytes are
//! rejected rather than guessed at.

use kivi_types::NamespaceNameError;

/// Failure decoding project-owned binary formats.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CodecError {
    /// Input ended before the value was complete.
    #[error("truncated value: needed {expected} bytes, had {available}")]
    Truncated {
        /// Bytes the value required.
        expected: usize,
        /// Bytes actually available.
        available: usize,
    },
    /// Strict decode consumed fewer bytes than supplied.
    #[error("trailing bytes: consumed {consumed} of {total}")]
    TrailingBytes {
        /// Bytes consumed by the value.
        consumed: usize,
        /// Total bytes supplied.
        total: usize,
    },
    /// Frame magic did not match [`KIVI_MAGIC`](crate::KIVI_MAGIC).
    #[error("invalid magic: found {found:#010X}")]
    InvalidMagic {
        /// Observed magic word.
        found: u32,
    },
    /// Structurally invalid header bytes for their claimed shape.
    #[error("malformed {context}")]
    Malformed {
        /// What was being decoded (static context, never untrusted bytes).
        context: &'static str,
    },
    /// Major/minor version is not understood by this decoder.
    #[error("unsupported version {major}.{minor}")]
    UnsupportedVersion {
        /// Observed major version.
        major: u16,
        /// Observed minor version.
        minor: u16,
    },
    /// Reserved flags were set; their meaning is unspecified, so the frame
    /// is rejected rather than interpreted.
    #[error("reserved flags set: {flags:#010X}")]
    NonZeroReservedFlags {
        /// Observed flags word.
        flags: u32,
    },
    /// A declared length exceeds [`MAX_FRAME_BYTES`](crate::MAX_FRAME_BYTES).
    #[error("length {len} exceeds maximum {max}")]
    TooLarge {
        /// Declared length in bytes.
        len: usize,
        /// Maximum accepted length in bytes.
        max: usize,
    },
    /// Header CRC32C did not match the header bytes.
    #[error("header checksum mismatch: stored {expected:#010X}, computed {computed:#010X}")]
    HeaderChecksumMismatch {
        /// CRC stored in the frame.
        expected: u32,
        /// CRC computed over the received header bytes.
        computed: u32,
    },
    /// Payload CRC32C did not match the payload bytes.
    #[error("payload checksum mismatch: stored {expected:#010X}, computed {computed:#010X}")]
    PayloadChecksumMismatch {
        /// CRC stored in the frame.
        expected: u32,
        /// CRC computed over the received payload bytes.
        computed: u32,
    },
    /// Boolean encoding was not `0` or `1`.
    #[error("invalid bool encoding: {byte:#04X}")]
    InvalidBoolEncoding {
        /// Observed byte.
        byte: u8,
    },
    /// Enumeration tag is unknown for the named value.
    #[error("unknown {kind} tag: {tag:#04X}")]
    InvalidTag {
        /// Which value was being decoded (e.g. `"read-contract"`).
        kind: &'static str,
        /// Observed tag byte.
        tag: u8,
    },
    /// Byte blob was not valid UTF-8 where a string was expected.
    #[error("invalid UTF-8: {0}")]
    InvalidUtf8(#[source] core::str::Utf8Error),
    /// Decoded bytes failed namespace-name validation.
    #[error("invalid namespace name: {0}")]
    InvalidNamespaceName(#[source] NamespaceNameError),
    /// Hash algorithm ID is not implemented by this decoder.
    #[error("unsupported hash algorithm id {id}")]
    UnsupportedHashAlgorithm {
        /// Observed algorithm ID.
        id: u16,
    },
    /// Checksum algorithm ID is not implemented by this decoder.
    #[error("unsupported checksum algorithm id {id}")]
    UnsupportedChecksumAlgorithm {
        /// Observed algorithm ID.
        id: u16,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn sources_chain_only_where_wrapping() {
        let bytes = [0xFFu8];
        let utf8 = core::str::from_utf8(bytes.as_slice()).expect_err("invalid");
        let wrapped = CodecError::InvalidUtf8(utf8);
        assert!(wrapped.source().is_some());
        assert_eq!(wrapped.to_string(), format!("invalid UTF-8: {utf8}"));
        assert!(
            CodecError::Truncated {
                expected: 9,
                available: 3
            }
            .source()
            .is_none()
        );
        assert_eq!(
            CodecError::TooLarge { len: 9, max: 8 }.to_string(),
            "length 9 exceeds maximum 8"
        );
    }
}
