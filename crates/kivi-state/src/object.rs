//! Logical objects: keys, versions, types, values, and metadata.
//!
//! A [`StoredObject`] is a small authoritative root: a logical value, a
//! monotonically advanced [`ObjectVersion`], an [`Expiry`], and a physical
//! representation tag. The representation is private and currently always
//! inline; future representations change storage
//! mechanics, never the semantics defined here.

use core::fmt;

use bytes::Bytes;
use kivi_codec::{CodecError, Decode, Encode, decode_byte_vec, encode_bytes};
use kivi_types::Expiry;

/// Object key: an immutable shared byte string.
///
/// Backed by [`Bytes`] so clones are cheap and payloads can be shared with
/// the network layer later without copying. This is an in-memory handle,
/// not a durable format — future persistent layouts define their own key
/// encoding. Lookups construct a `Key` (one small copy for borrowed input);
/// a `Borrow<[u8]>` impl is deliberately absent so map hashing can never
/// silently diverge between owned and borrowed forms.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key(Bytes);

impl Key {
    /// Wraps bytes as a key.
    #[must_use]
    pub fn new(bytes: impl Into<Bytes>) -> Self {
        Self(bytes.into())
    }

    /// Returns the key bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Returns the key length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the key is empty (allowed, like Redis).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<Vec<u8>> for Key {
    /// Wraps owned bytes without copying.
    fn from(bytes: Vec<u8>) -> Self {
        Self(Bytes::from(bytes))
    }
}

impl From<&[u8]> for Key {
    /// Copies borrowed bytes into a shared handle.
    fn from(bytes: &[u8]) -> Self {
        Self(Bytes::copy_from_slice(bytes))
    }
}

impl From<Bytes> for Key {
    /// Wraps a shared handle without copying.
    fn from(bytes: Bytes) -> Self {
        Self(bytes)
    }
}

impl From<&str> for Key {
    /// Copies a string's bytes into a shared handle.
    fn from(text: &str) -> Self {
        Self(Bytes::copy_from_slice(text.as_bytes()))
    }
}

impl From<String> for Key {
    /// Wraps an owned string's bytes without copying.
    fn from(text: String) -> Self {
        Self(Bytes::from(text.into_bytes()))
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", String::from_utf8_lossy(&self.0))
    }
}

impl Encode for Key {
    fn encoded_len(&self) -> usize {
        4 + self.len()
    }

    fn encode(&self, out: &mut Vec<u8>) {
        encode_bytes(out, self.as_bytes());
    }
}

impl Decode for Key {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (bytes, consumed) = decode_byte_vec(input)?;
        Ok((Self::from(bytes), consumed))
    }
}

/// Logical version of one object: `1` at creation, `+1` per state-changing
/// mutation, checked (never wrapping or saturating).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObjectVersion(u64);

impl ObjectVersion {
    /// Version assigned at object creation.
    pub const FIRST: Self = Self(1);

    /// Wraps a raw version value.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Successor version for the next state-changing mutation.
    ///
    /// # Errors
    ///
    /// Returns [`VersionExhausted`] at `u64::MAX` instead of wrapping or
    /// saturating: reusing a version would alias two distinct object states.
    pub const fn next(self) -> Result<Self, VersionExhausted> {
        match self.0.checked_add(1) {
            Some(value) => Ok(Self(value)),
            None => Err(VersionExhausted),
        }
    }
}

impl From<ObjectVersion> for u64 {
    /// Returns the raw version value.
    fn from(version: ObjectVersion) -> Self {
        version.0
    }
}

impl From<u64> for ObjectVersion {
    /// Wraps a raw version value.
    fn from(value: u64) -> Self {
        ObjectVersion(value)
    }
}

impl fmt::Display for ObjectVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl Encode for ObjectVersion {
    fn encoded_len(&self) -> usize {
        8
    }

    fn encode(&self, out: &mut Vec<u8>) {
        self.0.encode(out);
    }
}

impl Decode for ObjectVersion {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (raw, consumed) = u64::decode(input)?;
        Ok((Self::from_u64(raw), consumed))
    }
}

/// Reported when an object's version counter cannot advance past `u64::MAX`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[error("object version space exhausted")]
pub struct VersionExhausted;

/// Logical type of a stored object. Types are strict: operations against the
/// wrong type fail without mutation (native semantics, not Redis command
/// strings).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectType {
    /// Opaque byte string.
    Bytes,
    /// Exact 64-bit counter with ordinal (non-commutative-result) semantics.
    StrictCounter,
}

impl fmt::Display for ObjectType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bytes => write!(f, "bytes"),
            Self::StrictCounter => write!(f, "strict-counter"),
        }
    }
}

/// Canonical wire tags. Fixed forever within framing version 1.
const TAG_OBJECT_BYTES: u8 = 1;
const TAG_OBJECT_COUNTER: u8 = 2;

impl Encode for ObjectType {
    fn encoded_len(&self) -> usize {
        1
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.push(match self {
            Self::Bytes => TAG_OBJECT_BYTES,
            Self::StrictCounter => TAG_OBJECT_COUNTER,
        });
    }
}

impl Decode for ObjectType {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (tag, consumed) = u8::decode(input)?;
        match tag {
            TAG_OBJECT_BYTES => Ok((Self::Bytes, consumed)),
            TAG_OBJECT_COUNTER => Ok((Self::StrictCounter, consumed)),
            other => Err(CodecError::InvalidTag {
                kind: "object-type",
                tag: other,
            }),
        }
    }
}

/// Logical value of a stored object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogicalValue {
    /// Opaque byte string.
    Bytes(Bytes),
    /// Exact counter value.
    StrictCounter(i64),
}

impl LogicalValue {
    /// Returns the value's logical type.
    #[must_use]
    pub const fn object_type(&self) -> ObjectType {
        match self {
            Self::Bytes(_) => ObjectType::Bytes,
            Self::StrictCounter(_) => ObjectType::StrictCounter,
        }
    }
}

/// Physical representation tag. Private and currently always inline; future
/// arena/chunked/compressed/cold representations extend this enum without
/// touching logical semantics or any public API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Representation {
    Inline,
}

/// A stored logical object: value, version, expiry, and representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredObject {
    value: LogicalValue,
    version: ObjectVersion,
    expiry: Expiry,
    representation: Representation,
}

impl StoredObject {
    /// Builds an object. Only the store constructs these; callers go through
    /// typed operations and mutations.
    pub(crate) fn new(value: LogicalValue, version: ObjectVersion, expiry: Expiry) -> Self {
        Self {
            value,
            version,
            expiry,
            representation: Representation::Inline,
        }
    }

    /// Returns the logical value.
    #[must_use]
    pub const fn value(&self) -> &LogicalValue {
        &self.value
    }

    /// Returns the logical version.
    #[must_use]
    pub const fn version(&self) -> ObjectVersion {
        self.version
    }

    /// Returns the logical expiry.
    #[must_use]
    pub const fn expiry(&self) -> Expiry {
        self.expiry
    }

    /// Returns the logical type.
    #[must_use]
    pub const fn object_type(&self) -> ObjectType {
        self.value.object_type()
    }

    /// Whether the object is logically absent at `now` (`now >= expires_at`).
    /// Pure comparison: the caller supplies time from the clock boundary.
    #[must_use]
    pub const fn is_expired(&self, now: kivi_types::UnixMicros) -> bool {
        self.expiry.is_expired(now)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_advance_and_fail_explicitly_at_the_top() {
        assert_eq!(ObjectVersion::FIRST.as_u64(), 1);
        assert_eq!(
            ObjectVersion::FIRST.next().expect("advances"),
            ObjectVersion::from_u64(2)
        );
        assert_eq!(
            ObjectVersion::from_u64(u64::MAX).next(),
            Err(VersionExhausted)
        );
    }

    #[test]
    fn keys_wrap_bytes_cheaply() {
        let key = Key::new(b"user:123".as_slice());
        assert_eq!(key.as_bytes(), b"user:123");
        assert_eq!(key.to_string(), "user:123");
        assert_eq!(Key::from("abc"), Key::new(b"abc".as_slice()));
    }
}
