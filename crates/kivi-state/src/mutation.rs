//! Typed Mutation IR: the canonical replicated/persisted semantics.
//!
//! User commands are never replicated as-is. [`prepare`](crate::ObjectStore::prepare)
//! normalizes each write into a [`Mutation`] carrying everything a replica
//! needs for deterministic replay: keys, values, deltas, and expiries —
//! never wall-clock reads, randomness, or local decisions. [`CounterAdd`](Mutation::CounterAdd)
//! deterministically derives its resulting value from prior ordered state on
//! every replica identically.
//!
//! The binary form below is project-owned (little-endian, versioned tags,
//! length-prefixed blobs) and part of the durable contract. No serde,
//! bincode, or rkyv type defines it.

use kivi_codec::{CodecError, Decode, Encode, decode_byte_vec, encode_bytes};
use kivi_types::Expiry;

use crate::object::{Key, ObjectType, ObjectVersion};

/// One deterministic state transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mutation {
    /// Store bytes under `key`, overwriting any type and clearing expiry.
    PutBytes {
        /// Target key.
        key: Key,
        /// Value to store.
        value: bytes::Bytes,
    },
    /// Remove `key`, reporting whether a live object existed.
    Delete {
        /// Target key.
        key: Key,
    },
    /// Add `delta` to a counter, creating it at `delta` when absent.
    CounterAdd {
        /// Target key.
        key: Key,
        /// Signed addend.
        delta: i64,
    },
    /// Attach (`Some`) or clear (`None`) an absolute expiry on a live key.
    SetExpiry {
        /// Target key.
        key: Key,
        /// New expiry (`NEVER` clears).
        expiry: Expiry,
    },
}

/// Canonical wire tags. Fixed forever within framing version 1; new
/// operations take new tags, never reuse.
const TAG_PUT_BYTES: u8 = 1;
const TAG_DELETE: u8 = 2;
const TAG_COUNTER_ADD: u8 = 3;
const TAG_SET_EXPIRY: u8 = 4;

impl Encode for Mutation {
    fn encoded_len(&self) -> usize {
        match self {
            Self::PutBytes { key, value } => 1 + (4 + key.len()) + (4 + value.len()),
            Self::Delete { key } => 1 + (4 + key.len()),
            Self::CounterAdd { key, .. } => 1 + (4 + key.len()) + 8,
            Self::SetExpiry { key, expiry } => 1 + (4 + key.len()) + expiry.encoded_len(),
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::PutBytes { key, value } => {
                out.push(TAG_PUT_BYTES);
                encode_bytes(out, key.as_bytes());
                encode_bytes(out, value);
            }
            Self::Delete { key } => {
                out.push(TAG_DELETE);
                encode_bytes(out, key.as_bytes());
            }
            Self::CounterAdd { key, delta } => {
                out.push(TAG_COUNTER_ADD);
                encode_bytes(out, key.as_bytes());
                out.extend_from_slice(&delta.to_le_bytes());
            }
            Self::SetExpiry { key, expiry } => {
                out.push(TAG_SET_EXPIRY);
                encode_bytes(out, key.as_bytes());
                expiry.encode(out);
            }
        }
    }
}

impl Decode for Mutation {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (tag, first) = u8::decode(input)?;
        match tag {
            TAG_PUT_BYTES => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (value, third) = decode_byte_vec(&input[first + second..])?;
                Ok((
                    Self::PutBytes {
                        key: Key::from(key),
                        value: bytes::Bytes::from(value),
                    },
                    first + second + third,
                ))
            }
            TAG_DELETE => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                Ok((
                    Self::Delete {
                        key: Key::from(key),
                    },
                    first + second,
                ))
            }
            TAG_COUNTER_ADD => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (raw, third) = <[u8; 8]>::decode(&input[first + second..])?;
                Ok((
                    Self::CounterAdd {
                        key: Key::from(key),
                        delta: i64::from_le_bytes(raw),
                    },
                    first + second + third,
                ))
            }
            TAG_SET_EXPIRY => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (expiry, third) = Expiry::decode(&input[first + second..])?;
                Ok((
                    Self::SetExpiry {
                        key: Key::from(key),
                        expiry,
                    },
                    first + second + third,
                ))
            }
            other => Err(CodecError::InvalidTag {
                kind: "mutation",
                tag: other,
            }),
        }
    }
}

/// Outcome of one applied mutation.
///
/// Exhaustive by design (see [`OperationResult`](crate::OperationResult)):
/// consumers handle every outcome explicitly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApplyOutcome {
    /// Bytes stored with the new version.
    Put {
        /// Version after the store.
        version: ObjectVersion,
    },
    /// Key removed; `existed` names a previously live object.
    Deleted {
        /// Whether a live object was removed.
        existed: bool,
    },
    /// Counter stored with the new value and version.
    Counter {
        /// Version after the addition.
        version: ObjectVersion,
        /// Value after the addition.
        value: i64,
    },
    /// Expiry attached or cleared.
    Expiry {
        /// Whether a live object took the change.
        applied: bool,
        /// Version after the change (`None` when untouched).
        version: Option<ObjectVersion>,
    },
}

/// Deterministic apply failure. Same state plus same mutation always agrees,
/// so replicas never diverge through these paths.
///
/// Exhaustive like [`ApplyOutcome`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ApplyError {
    /// An object version cannot advance past `u64::MAX`.
    #[error("object version space exhausted")]
    VersionExhausted,
    /// A counter addition overflows `i64`.
    #[error("counter addition overflows i64")]
    CounterOverflow,
    /// A counter mutation met a byte string (only reachable by applying
    /// mutations against divergent state).
    #[error("mutation needs {expected}, key holds {found}")]
    TypeMismatch {
        /// Type the mutation required.
        expected: ObjectType,
        /// Type actually stored.
        found: ObjectType,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_types::UnixMicros;

    #[test]
    fn mutation_codecs_round_trip() {
        let key = Key::from("user:1");
        let cases = vec![
            Mutation::PutBytes {
                key: key.clone(),
                value: bytes::Bytes::from_static(b"v"),
            },
            Mutation::Delete { key: key.clone() },
            Mutation::CounterAdd {
                key: key.clone(),
                delta: -41,
            },
            Mutation::SetExpiry {
                key: key.clone(),
                expiry: Expiry::at(UnixMicros::from_micros(99)),
            },
            Mutation::SetExpiry {
                key: key.clone(),
                expiry: Expiry::NEVER,
            },
        ];
        for mutation in cases {
            assert_eq!(mutation.encoded_len(), mutation.encode_to_vec().len());
            assert_eq!(
                Mutation::decode_exact(&mutation.encode_to_vec()).expect("round trip"),
                mutation
            );
        }
    }

    #[test]
    fn malformed_mutations_are_rejected() {
        assert_eq!(
            Mutation::decode(&[0xFF]),
            Err(CodecError::InvalidTag {
                kind: "mutation",
                tag: 0xFF
            })
        );
        let valid = Mutation::Delete {
            key: Key::from("k"),
        }
        .encode_to_vec();
        assert!(matches!(
            Mutation::decode(&valid[..valid.len() - 1]),
            Err(CodecError::Truncated { .. })
        ));
        assert!(matches!(
            Mutation::decode_exact(&[valid.clone(), vec![0]].concat()),
            Err(CodecError::TrailingBytes { .. })
        ));
    }
}
