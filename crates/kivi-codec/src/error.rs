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
    /// A declared length exceeds [`MAX_FRAME_BYTES`](crate::MAX_FRAME_BYTES).
    #[error("length {len} exceeds maximum {max}")]
    TooLarge {
        /// Declared length in bytes.
        len: usize,
        /// Maximum accepted length in bytes.
        max: usize,
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
}
