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
use kivi_types::{Expiry, ManifestId};

use crate::object::{Key, ObjectType, ObjectVersion};
use crate::txn::{TxnExpect, TxnId, TxnWriteKind};

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
    /// Point `key` at a chunked value: the manifest addressing its
    /// immutable chunks plus the total logical length. The WAL carries
    /// this small root transition only — bulk bytes live in chunk packs
    /// whose durability precedes this record (§11, §12). Overwrites any
    /// type and clears expiry, exactly like [`PutBytes`](Self::PutBytes).
    ReplaceChunkedRoot {
        /// Target key.
        key: Key,
        /// Manifest addressing the immutable chunk sequence.
        manifest: ManifestId,
        /// Total logical bytes across the manifest.
        logical_len: u64,
    },
    /// Patch a byte range of `key`'s value at `offset` with `patch`,
    /// zero-padding past-the-end gaps (Redis `SETRANGE` semantics). Unlike
    /// [`PutBytes`](Self::PutBytes), a splice preserves the live expiry:
    /// partial writes touch bytes, never the TTL. The tablet layer only
    /// admits this against absent, expired, or inline bases whose patched
    /// result stays within the legacy inline bound; chunked bases are
    /// resolved and restaged by the engine first (with the preserved
    /// expiry), so apply meets no chunk reference it cannot read. Replay is
    /// deterministic: the same ordered state yields the same base, hence
    /// the same splice.
    SpliceBytes {
        /// Target key.
        key: Key,
        /// Logical byte index the patch overwrites from.
        offset: u64,
        /// Bytes written starting at `offset`.
        patch: bytes::Bytes,
    },
    /// Store bytes under `key` with an explicit expiry, produced only by
    /// conditional sets whose policy resolved to a concrete stamp at
    /// preparation time (`Clear` resolves to `NEVER`, `Keep` to the live
    /// expiry or `NEVER`, `ExpireAt` to its stamp). Overwrites any type.
    /// The expiry rides in the mutation so replay applies the same stamp
    /// without re-reading state.
    PutBytesWithExpiry {
        /// Target key.
        key: Key,
        /// Value to store.
        value: bytes::Bytes,
        /// Resolved expiry to attach.
        expiry: Expiry,
    },
    /// Point `key` at a chunked value with an explicit expiry: the chunked
    /// spelling of [`PutBytesWithExpiry`](Self::PutBytesWithExpiry) for
    /// conditional stores whose staged value exceeded the inline bound.
    ReplaceChunkedRootWithExpiry {
        /// Target key.
        key: Key,
        /// Manifest addressing the immutable chunk sequence.
        manifest: ManifestId,
        /// Total logical bytes across the manifest.
        logical_len: u64,
        /// Resolved expiry to attach.
        expiry: Expiry,
    },
    /// Reserve `key` for transaction `txn`: validates the OCC expectation
    /// and records a durable intent without touching user-visible state.
    /// Single-key like every other variant (`key` is the reserved key), so
    /// overlays, dirty bands, and sweeps need no multi-key contract.
    TxnPrepare {
        /// Transaction identity (idempotency key for the reservation).
        txn: TxnId,
        /// Coordinator tablet owning the decision record.
        coordinator: u64,
        /// Reserved key.
        key: Key,
        /// OCC version expectation.
        expect: TxnExpect,
        /// Prepared write (applied at finalize-commit).
        write: TxnWriteKind,
    },
    /// Resolve one key's intent for `txn`: commit applies the prepared
    /// write, abort discards it. Idempotent: a missing intent (already
    /// finalized or never prepared here) answers finalized-not-applied.
    TxnFinalize {
        /// Transaction identity.
        txn: TxnId,
        /// Key whose intent resolves.
        key: Key,
        /// Whether to apply (`true`) or discard (`false`) the intent.
        commit: bool,
    },
}

/// Canonical wire tags. Fixed forever within framing version 1; new
/// operations take new tags, never reuse.
const TAG_PUT_BYTES: u8 = 1;
const TAG_DELETE: u8 = 2;
const TAG_COUNTER_ADD: u8 = 3;
const TAG_SET_EXPIRY: u8 = 4;
/// Chunked-root mutation tag. New tags never reuse old ones.
const TAG_REPLACE_CHUNKED_ROOT: u8 = 5;
/// Byte-range splice mutation tag. New tags never reuse old ones.
const TAG_SPLICE_BYTES: u8 = 6;
/// Conditional inline-store mutation tag. New tags never reuse old ones.
const TAG_PUT_BYTES_WITH_EXPIRY: u8 = 7;
/// Conditional chunked-root mutation tag. New tags never reuse old ones.
const TAG_REPLACE_CHUNKED_ROOT_WITH_EXPIRY: u8 = 8;
/// Transaction prepare (durable intent reservation) tag.
const TAG_TXN_PREPARE: u8 = 9;
/// Transaction per-key finalize tag.
const TAG_TXN_FINALIZE: u8 = 10;

impl Mutation {
    /// Returns the single key this mutation touches. Every mutation in
    /// this stage is single-key; dirty-band tracking and overlays rely on
    /// that (a future multi-key mutation must extend this contract, not
    /// silently bypass it). Transaction variants reserve/resolve exactly
    /// one key each, so the contract holds for 2PC as well.
    #[must_use]
    pub fn key(&self) -> &Key {
        match self {
            Self::PutBytes { key, .. }
            | Self::Delete { key }
            | Self::CounterAdd { key, .. }
            | Self::SetExpiry { key, .. }
            | Self::ReplaceChunkedRoot { key, .. }
            | Self::SpliceBytes { key, .. }
            | Self::PutBytesWithExpiry { key, .. }
            | Self::ReplaceChunkedRootWithExpiry { key, .. }
            | Self::TxnPrepare { key, .. }
            | Self::TxnFinalize { key, .. } => key,
        }
    }

    /// Reconstructs the originating operation. Exact inverse of
    /// `prepare`'s normalization: `SetExpiry{NEVER}` always came from
    /// `PersistExpiry` (which is the only producer of `NEVER`), any other
    /// expiry came from `ExpireAt`. WAL replay uses this to share the single
    /// [`outcome_for`](crate::ops::outcome_for) mapping with the live path.
    #[must_use]
    pub fn as_operation(&self) -> crate::ops::Operation {
        use crate::ops::Operation;
        match self {
            Self::PutBytes { key, value } => Operation::Set {
                key: key.clone(),
                value: value.clone(),
            },
            Self::Delete { key } => Operation::Delete { key: key.clone() },
            Self::CounterAdd { key, delta } => Operation::CounterAdd {
                key: key.clone(),
                delta: *delta,
            },
            Self::SetExpiry { key, expiry } => {
                if *expiry == kivi_types::Expiry::NEVER {
                    Operation::PersistExpiry { key: key.clone() }
                } else {
                    Operation::ExpireAt {
                        key: key.clone(),
                        expires_at: expiry
                            .as_stamp()
                            .unwrap_or(kivi_types::UnixMicros::from_micros(0)),
                    }
                }
            }
            Self::ReplaceChunkedRoot {
                key,
                manifest,
                logical_len,
            } => Operation::SetChunked {
                key: key.clone(),
                manifest: *manifest,
                logical_len: *logical_len,
            },
            Self::SpliceBytes { key, offset, patch } => Operation::SetRange {
                key: key.clone(),
                offset: *offset,
                patch: patch.clone(),
            },
            Self::PutBytesWithExpiry { key, value, expiry } => {
                let policy = if *expiry == kivi_types::Expiry::NEVER {
                    crate::ops::ExpiryPolicy::Clear
                } else {
                    crate::ops::ExpiryPolicy::ExpireAt(
                        expiry
                            .as_stamp()
                            .unwrap_or(kivi_types::UnixMicros::from_micros(0)),
                    )
                };
                Operation::SetConditional {
                    key: key.clone(),
                    value: value.clone(),
                    condition: crate::ops::SetCondition::Always,
                    expiry: policy,
                }
            }
            Self::ReplaceChunkedRootWithExpiry {
                key,
                manifest: _,
                logical_len: _,
                expiry,
            } => {
                let policy = if *expiry == kivi_types::Expiry::NEVER {
                    crate::ops::ExpiryPolicy::Clear
                } else {
                    crate::ops::ExpiryPolicy::ExpireAt(
                        expiry
                            .as_stamp()
                            .unwrap_or(kivi_types::UnixMicros::from_micros(0)),
                    )
                };
                // Chunked conditional roots replay through the same
                // conditional shape; the manifest itself is the mutation's
                // truth, the operation is only the replay-mapping key.
                Operation::SetConditional {
                    key: key.clone(),
                    value: bytes::Bytes::new(),
                    condition: crate::ops::SetCondition::Always,
                    expiry: policy,
                }
            }
            Self::TxnPrepare {
                txn,
                coordinator,
                key,
                expect,
                write,
            } => Operation::TxnPrepare {
                txn: *txn,
                coordinator: kivi_types::TabletId::from_u64(*coordinator),
                write: crate::txn::TxnWrite {
                    key: key.clone(),
                    kind: write.clone(),
                    expect: *expect,
                },
            },
            Self::TxnFinalize { txn, key, commit } => Operation::TxnFinalize {
                txn: *txn,
                key: key.clone(),
                commit: *commit,
            },
        }
    }
}

impl Encode for Mutation {
    fn encoded_len(&self) -> usize {
        match self {
            Self::PutBytes { key, value } => 1 + (4 + key.len()) + (4 + value.len()),
            Self::Delete { key } => 1 + (4 + key.len()),
            Self::CounterAdd { key, .. } => 1 + (4 + key.len()) + 8,
            Self::SetExpiry { key, expiry } => 1 + (4 + key.len()) + expiry.encoded_len(),
            Self::ReplaceChunkedRoot { key, .. } => 1 + (4 + key.len()) + 32 + 8,
            Self::SpliceBytes { key, patch, .. } => 1 + (4 + key.len()) + 8 + (4 + patch.len()),
            Self::PutBytesWithExpiry { key, value, expiry } => {
                1 + (4 + key.len()) + (4 + value.len()) + expiry.encoded_len()
            }
            Self::ReplaceChunkedRootWithExpiry { key, expiry, .. } => {
                1 + (4 + key.len()) + 32 + 8 + expiry.encoded_len()
            }
            Self::TxnPrepare {
                key, expect, write, ..
            } => 1 + 16 + 8 + (4 + key.len()) + expect.encoded_len() + write.encoded_len(),
            Self::TxnFinalize { key, .. } => 1 + 16 + (4 + key.len()) + 1,
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
            Self::ReplaceChunkedRoot {
                key,
                manifest,
                logical_len,
            } => {
                out.push(TAG_REPLACE_CHUNKED_ROOT);
                encode_bytes(out, key.as_bytes());
                out.extend_from_slice(manifest.as_bytes());
                logical_len.encode(out);
            }
            Self::SpliceBytes { key, offset, patch } => {
                out.push(TAG_SPLICE_BYTES);
                encode_bytes(out, key.as_bytes());
                offset.encode(out);
                encode_bytes(out, patch);
            }
            Self::PutBytesWithExpiry { key, value, expiry } => {
                out.push(TAG_PUT_BYTES_WITH_EXPIRY);
                encode_bytes(out, key.as_bytes());
                encode_bytes(out, value);
                expiry.encode(out);
            }
            Self::ReplaceChunkedRootWithExpiry {
                key,
                manifest,
                logical_len,
                expiry,
            } => {
                out.push(TAG_REPLACE_CHUNKED_ROOT_WITH_EXPIRY);
                encode_bytes(out, key.as_bytes());
                out.extend_from_slice(manifest.as_bytes());
                logical_len.encode(out);
                expiry.encode(out);
            }
            Self::TxnPrepare {
                txn,
                coordinator,
                key,
                expect,
                write,
            } => {
                out.push(TAG_TXN_PREPARE);
                txn.encode(out);
                coordinator.encode(out);
                encode_bytes(out, key.as_bytes());
                expect.encode(out);
                write.encode(out);
            }
            Self::TxnFinalize { txn, key, commit } => {
                out.push(TAG_TXN_FINALIZE);
                txn.encode(out);
                encode_bytes(out, key.as_bytes());
                commit.encode(out);
            }
        }
    }
}

impl Decode for Mutation {
    #[allow(clippy::too_many_lines)]
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
            TAG_REPLACE_CHUNKED_ROOT => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (raw, third) = <[u8; 32]>::decode(&input[first + second..])?;
                let (logical_len, fourth) = u64::decode(&input[first + second + third..])?;
                Ok((
                    Self::ReplaceChunkedRoot {
                        key: Key::from(key),
                        manifest: ManifestId::from_bytes(raw),
                        logical_len,
                    },
                    first + second + third + fourth,
                ))
            }
            TAG_SPLICE_BYTES => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (offset, third) = u64::decode(&input[first + second..])?;
                let (patch, fourth) = decode_byte_vec(&input[first + second + third..])?;
                Ok((
                    Self::SpliceBytes {
                        key: Key::from(key),
                        offset,
                        patch: bytes::Bytes::from(patch),
                    },
                    first + second + third + fourth,
                ))
            }
            TAG_PUT_BYTES_WITH_EXPIRY => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (value, third) = decode_byte_vec(&input[first + second..])?;
                let (expiry, fourth) = Expiry::decode(&input[first + second + third..])?;
                Ok((
                    Self::PutBytesWithExpiry {
                        key: Key::from(key),
                        value: bytes::Bytes::from(value),
                        expiry,
                    },
                    first + second + third + fourth,
                ))
            }
            TAG_REPLACE_CHUNKED_ROOT_WITH_EXPIRY => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (raw, third) = <[u8; 32]>::decode(&input[first + second..])?;
                let (logical_len, fourth) = u64::decode(&input[first + second + third..])?;
                let (expiry, fifth) = Expiry::decode(&input[first + second + third + fourth..])?;
                Ok((
                    Self::ReplaceChunkedRootWithExpiry {
                        key: Key::from(key),
                        manifest: ManifestId::from_bytes(raw),
                        logical_len,
                        expiry,
                    },
                    first + second + third + fourth + fifth,
                ))
            }
            TAG_TXN_PREPARE => {
                let (txn, second) = TxnId::decode(&input[first..])?;
                let (coordinator, third) = u64::decode(&input[first + second..])?;
                let (key, fourth) = decode_byte_vec(&input[first + second + third..])?;
                let (expect, fifth) = TxnExpect::decode(&input[first + second + third + fourth..])?;
                let (write, sixth) =
                    TxnWriteKind::decode(&input[first + second + third + fourth + fifth..])?;
                Ok((
                    Self::TxnPrepare {
                        txn,
                        coordinator,
                        key: Key::from(key),
                        expect,
                        write,
                    },
                    first + second + third + fourth + fifth + sixth,
                ))
            }
            TAG_TXN_FINALIZE => {
                let (txn, second) = TxnId::decode(&input[first..])?;
                let (key, third) = decode_byte_vec(&input[first + second..])?;
                let (commit, fourth) = bool::decode(&input[first + second + third..])?;
                Ok((
                    Self::TxnFinalize {
                        txn,
                        key: Key::from(key),
                        commit,
                    },
                    first + second + third + fourth,
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
    /// Transaction intent reserved (prepare validated and recorded).
    TxnPrepared,
    /// Transaction prepare evaluated to a conflict (OCC mismatch or a live
    /// intent from another transaction blocks the key). Deterministic: same
    /// log on same state always agrees, so this is a normal outcome, never
    /// divergence.
    TxnConflict,
    /// One key's intent resolved: commit applied the prepared write (or the
    /// intent was already gone), abort discarded it.
    TxnFinalized {
        /// Whether a prepared write was applied (`false` for aborts and for
        /// idempotent replays with no intent left).
        applied: bool,
        /// Version after the applied write (`None` when nothing applied).
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
    /// A splice met a chunked base the engine must have restaged first,
    /// or an offset that overflows `u64`. Only reachable by applying
    /// mutations against divergent state: admission resolves chunked
    /// bases through the lane and validates arithmetic before the WAL.
    #[error("byte-range splice cannot apply to stored state")]
    UnresolvableSplice,
    /// A transactional prepare or finalize diverged from its validated
    /// prediction (same log on same state never produces this): either a
    /// driver bug replaying a different write under one `TxnId`, or state
    /// divergence. Fail closed, never partial intent state.
    #[error("transaction mutation diverged from its prepared prediction")]
    TxnDiverged,
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
            Mutation::ReplaceChunkedRoot {
                key: key.clone(),
                manifest: ManifestId::from_bytes([0x11; 32]),
                logical_len: 3_000_000,
            },
            Mutation::SpliceBytes {
                key: key.clone(),
                offset: 7,
                patch: bytes::Bytes::from_static(b"patch"),
            },
            Mutation::SpliceBytes {
                key: key.clone(),
                offset: u64::MAX,
                patch: bytes::Bytes::new(),
            },
            Mutation::PutBytesWithExpiry {
                key: key.clone(),
                value: bytes::Bytes::from_static(b"v"),
                expiry: Expiry::NEVER,
            },
            Mutation::PutBytesWithExpiry {
                key: key.clone(),
                value: bytes::Bytes::from_static(b"v"),
                expiry: Expiry::at(UnixMicros::from_micros(7)),
            },
            Mutation::ReplaceChunkedRootWithExpiry {
                key: key.clone(),
                manifest: ManifestId::from_bytes([0x33; 32]),
                logical_len: 11,
                expiry: Expiry::NEVER,
            },
            Mutation::TxnPrepare {
                txn: crate::txn::TxnId::derive(3, 4, 0),
                coordinator: 9,
                key: key.clone(),
                expect: crate::txn::TxnExpect::Version(ObjectVersion::from_u64(2)),
                write: crate::txn::TxnWriteKind::Put(bytes::Bytes::from_static(b"v")),
            },
            Mutation::TxnPrepare {
                txn: crate::txn::TxnId::derive(3, 5, 0),
                coordinator: 9,
                key: key.clone(),
                expect: crate::txn::TxnExpect::Absent,
                write: crate::txn::TxnWriteKind::CounterAdd(-3),
            },
            Mutation::TxnFinalize {
                txn: crate::txn::TxnId::derive(3, 4, 0),
                key: key.clone(),
                commit: true,
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
