//! Human-readable namespace names (edge type, not the durable identity).
//!
//! [`NamespaceName`] is validated at the edge; the canonical durable identity
//! stored by replication, durability, and routing is [`crate::NamespaceId`].
//! Keeping the two separate means renames and validation-rule changes never
//! rewrite durable control state.

use core::fmt;

/// Maximum length of a namespace name, in bytes (names are ASCII-only, so
/// bytes and characters coincide).
pub const MAX_NAMESPACE_NAME_LEN: usize = 128;

/// Validated namespace name, e.g. `"sessions"`.
///
/// Rules: 1–128 bytes; ASCII lowercase alphanumeric plus `-` and `_`; must
/// start with an alphanumeric. The restricted charset keeps names safe to
/// embed in future log paths, object-store keys, and metric labels without
/// further escaping.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NamespaceName(String);

impl NamespaceName {
    /// Validates and wraps a name.
    ///
    /// # Errors
    ///
    /// Returns [`NamespaceNameError`] if the name is empty, too long, or
    /// contains a disallowed byte.
    pub fn new(name: &str) -> Result<Self, NamespaceNameError> {
        if name.is_empty() {
            return Err(NamespaceNameError::Empty);
        }
        if name.len() > MAX_NAMESPACE_NAME_LEN {
            return Err(NamespaceNameError::TooLong {
                len: name.len(),
                max: MAX_NAMESPACE_NAME_LEN,
            });
        }
        let bytes = name.as_bytes();
        for (index, byte) in bytes.iter().enumerate() {
            let valid = if index == 0 {
                byte.is_ascii_alphanumeric()
            } else {
                byte.is_ascii_alphanumeric() || *byte == b'-' || *byte == b'_'
            };
            if !valid {
                return Err(NamespaceNameError::InvalidByte { byte: *byte, index });
            }
            if !byte.is_ascii_lowercase() && byte.is_ascii_alphabetic() {
                return Err(NamespaceNameError::InvalidByte { byte: *byte, index });
            }
        }
        Ok(Self(name.to_owned()))
    }

    /// Returns the name as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns the length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the name is empty. Always `false`: validation rejects empty
    /// names, so this exists only to pair with [`len`](Self::len).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Consumes the wrapper and returns the inner string.
    #[must_use]
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for NamespaceName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<NamespaceName> for String {
    /// Consumes the wrapper and returns the inner string.
    fn from(name: NamespaceName) -> Self {
        name.0
    }
}

/// Reason a namespace name was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum NamespaceNameError {
    /// The name was empty.
    #[error("namespace name is empty")]
    Empty,
    /// The name exceeded [`MAX_NAMESPACE_NAME_LEN`].
    #[error("namespace name length {len} exceeds maximum {max}")]
    TooLong {
        /// Observed length in bytes.
        len: usize,
        /// Maximum allowed length in bytes.
        max: usize,
    },
    /// A byte outside the allowed charset, or an uppercase letter.
    #[error("namespace name has invalid byte {byte:#04X} at index {index}")]
    InvalidByte {
        /// The offending byte.
        byte: u8,
        /// Byte index within the name.
        index: usize,
    },
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// A namespace name is the main semantic and policy boundary, so the
    /// charset is enforced at construction and the rejection names the
    /// offending byte position: that index is what a caller needs to point at
    /// the input.
    #[rstest]
    #[case::leading_dash("-abc", 0)]
    #[case::uppercase("Abc", 0)]
    #[case::inner_space("a b", 1)]
    #[case::slash("a/b", 1)]
    fn a_bad_byte_is_reported_at_its_index(#[case] input: &str, #[case] index: usize) {
        assert!(matches!(
            NamespaceName::new(input),
            Err(NamespaceNameError::InvalidByte { index: found, .. }) if found == index
        ));
    }

    /// The accepted charset is lowercase, digits, dash and underscore, starting
    /// with a letter or digit. A name at exactly the length bound is accepted;
    /// one byte over is rejected, and the boundary matters because the length
    /// is persisted.
    #[test]
    fn the_charset_and_length_bound_are_inclusive_at_the_edges() {
        for input in ["sessions", "a-1_b2", "n9", "x"] {
            let name = NamespaceName::new(input).unwrap_or_else(|e| panic!("{input:?}: {e}"));
            assert_eq!(name.as_str(), input);
        }
        assert_eq!(
            NamespaceName::new("").unwrap_err(),
            NamespaceNameError::Empty
        );
        let max = "a".repeat(MAX_NAMESPACE_NAME_LEN);
        assert_eq!(
            NamespaceName::new(&max).expect("max length").len(),
            MAX_NAMESPACE_NAME_LEN
        );
        assert!(matches!(
            NamespaceName::new(&"a".repeat(MAX_NAMESPACE_NAME_LEN + 1)),
            Err(NamespaceNameError::TooLong { .. })
        ));
    }
}
