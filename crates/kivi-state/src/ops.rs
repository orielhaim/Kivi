//! Typed native operations, results, and failures.
//!
//! The core never sees Redis command strings: callers express intent with
//! [`Operation`], whose semantics are explicit and stable. Wrong-type access
//! fails without mutation; counter overflow fails without mutation; expired
//! objects behave as absent (a read never deletes).

use bytes::Bytes;
use kivi_codec::{CodecError, Decode, Encode, decode_byte_vec, encode_bytes};
use kivi_types::{Expiry, ManifestId, UnixMicros};

use crate::mutation::{ApplyOutcome, Mutation};
use crate::object::{Key, ObjectType, ObjectVersion};

/// One typed native operation against a single key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operation {
    /// Fetch bytes; fails with [`WrongType`](OpError::WrongType) on counters.
    Get {
        /// Key to read.
        key: Key,
    },
    /// Store bytes, overwriting any type and clearing any expiry (SET semantics).
    Set {
        /// Key to write.
        key: Key,
        /// Value to store.
        value: Bytes,
    },
    /// Store a large byte string by chunk reference, overwriting any type
    /// and clearing any expiry — the chunked spelling of [`Set`](Self::Set).
    /// The engine stages every referenced chunk and proves pack durability
    /// *before* admitting this; the store trusts the reference exactly as
    /// it trusts an inline value. Produced by the streaming commit path and
    /// by large legacy `Set`s the engine converts at admission, never by
    /// end users directly.
    SetChunked {
        /// Key to write.
        key: Key,
        /// Manifest addressing the immutable chunk sequence.
        manifest: ManifestId,
        /// Total logical bytes across the manifest.
        logical_len: u64,
    },
    /// Patch a byte range of a value (partial update). Against absent or
    /// expired state the base is the empty string; past-the-end gaps
    /// zero-pad (Redis `SETRANGE` semantics). Against byte strings the
    /// untouched prefix and suffix survive; against counters this fails with
    /// [`WrongType`](OpError::WrongType) without mutation. Unlike
    /// [`Set`](Self::Set), a patch preserves the live expiry: it touches
    /// bytes, never the TTL. Large results the engine converts to chunked
    /// roots at admission (carrying the preserved expiry), so the store
    /// spells the outcome as whatever representation fits.
    SetRange {
        /// Key to patch.
        key: Key,
        /// Logical byte index the patch overwrites from.
        offset: u64,
        /// Bytes written starting at `offset`.
        patch: Bytes,
    },
    /// Remove the key (including expired-but-unreclaimed state).
    Delete {
        /// Key to remove.
        key: Key,
    },
    /// Whether a live (present and unexpired) value exists, of any type.
    Exists {
        /// Key to probe.
        key: Key,
    },
    /// Fetch a counter value; fails with `WrongType` on byte strings.
    CounterGet {
        /// Key to read.
        key: Key,
    },
    /// Add `delta` to a counter, creating it at `delta` when absent.
    /// Negative deltas decrement. Overflow fails without mutation.
    CounterAdd {
        /// Key to update.
        key: Key,
        /// Signed addend.
        delta: i64,
    },
    /// Attach an absolute expiry; missing keys report `applied: false`.
    ExpireAt {
        /// Key to expire.
        key: Key,
        /// Logical expiration timestamp.
        expires_at: UnixMicros,
    },
    /// Remove any expiry; keys without one report `removed: false`.
    PersistExpiry {
        /// Key to persist.
        key: Key,
    },
    /// Read the expiry of a live key (`None` when the key is absent).
    GetExpiry {
        /// Key to inspect.
        key: Key,
    },
    /// Read a byte slice of a value without materializing the whole object
    /// on the caller's terms: `[offset, offset + len)` clamped to the
    /// logical length (empty when `offset` is past the end). Against absent
    /// or expired state answers `None`; against counters fails with
    /// [`WrongType`](OpError::WrongType) without mutation. The engine may
    /// satisfy this from only the required immutable chunks instead of the
    /// full payload.
    GetRange {
        /// Key to slice.
        key: Key,
        /// Logical byte index the slice starts at.
        offset: u64,
        /// Maximum bytes to return from `offset`.
        len: u64,
    },
    /// Report the logical byte length of a value (`None` when absent).
    /// Uses logical metadata only: chunked roots answer from the root
    /// without reading any chunk payload. Counters fail with
    /// [`WrongType`](OpError::WrongType) without mutation.
    BytesLength {
        /// Key to measure.
        key: Key,
    },
    /// Conditionally store bytes, atomically at the owning tablet. The
    /// condition is evaluated against live state (present and unexpired)
    /// and the write applies only when it holds; the expiry policy selects
    /// the stored expiry (`Clear` clears, `Keep` preserves the live expiry
    /// or stays immortal when absent, `ExpireAt` attaches a stamp). There
    /// is deliberately no `ReturnPrevious` in this stage: returning the
    /// previous value would force materialization of arbitrarily large
    /// chunked payloads into a bounded response.
    SetConditional {
        /// Key to write.
        key: Key,
        /// Value to store when the condition holds.
        value: Bytes,
        /// Presence condition evaluated atomically.
        condition: SetCondition,
        /// Expiry policy for the stored object.
        expiry: ExpiryPolicy,
    },
    /// Conditionally store a large byte string by chunk reference: the
    /// chunked spelling of [`SetConditional`](Self::SetConditional). The
    /// engine stages every referenced chunk and proves pack durability
    /// *before* admitting this; the store trusts the reference exactly as
    /// it trusts an inline value. Produced by large conditional stores the
    /// engine converts at admission, never by end users directly.
    SetConditionalChunked {
        /// Key to write.
        key: Key,
        /// Manifest addressing the immutable chunk sequence.
        manifest: ManifestId,
        /// Total logical bytes across the manifest.
        logical_len: u64,
        /// Presence condition evaluated atomically.
        condition: SetCondition,
        /// Expiry policy for the stored object.
        expiry: ExpiryPolicy,
    },
}

/// Presence condition for [`Operation::SetConditional`]: evaluated against
/// live state (present and unexpired) atomically at the owning tablet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SetCondition {
    /// Always store (same admission as [`Operation::Set`] plus policy).
    Always,
    /// Store only when no live value exists.
    IfAbsent,
    /// Store only when a live value exists.
    IfPresent,
}

/// Expiry policy for [`Operation::SetConditional`]: a Kivi-owned model,
/// not Redis `EX`/`PX`/`KEEPTTL` flags (the RESP adapter translates those
/// onto this).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpiryPolicy {
    /// Clear any expiry (same as [`Operation::Set`]).
    Clear,
    /// Preserve the live expiry, or stay immortal when creating.
    Keep,
    /// Attach this absolute logical timestamp on store.
    ExpireAt(UnixMicros),
}

impl Operation {
    /// Returns the single key this operation touches. Every operation in
    /// this stage is single-key, which is what makes batch-local overlays
    /// sound: preparation reads no state beyond this key.
    #[must_use]
    pub fn key(&self) -> &Key {
        match self {
            Self::Get { key }
            | Self::Set { key, .. }
            | Self::SetChunked { key, .. }
            | Self::SetRange { key, .. }
            | Self::Delete { key }
            | Self::Exists { key }
            | Self::CounterGet { key }
            | Self::CounterAdd { key, .. }
            | Self::ExpireAt { key, .. }
            | Self::PersistExpiry { key }
            | Self::GetExpiry { key }
            | Self::GetRange { key, .. }
            | Self::BytesLength { key }
            | Self::SetConditional { key, .. }
            | Self::SetConditionalChunked { key, .. } => key,
        }
    }

    /// Whether this operation can change logical state (and therefore enters
    /// the WAL in durable mode). Reads never do, whatever their outcome.
    /// This is the semantic boundary both frontends share: the native wire
    /// decoder and the RESP adapter meet here, never in opcode bytes.
    #[must_use]
    pub const fn is_mutating(&self) -> bool {
        match self {
            Self::Set { .. }
            | Self::SetChunked { .. }
            | Self::SetRange { .. }
            | Self::Delete { .. }
            | Self::CounterAdd { .. }
            | Self::ExpireAt { .. }
            | Self::PersistExpiry { .. }
            | Self::SetConditional { .. }
            | Self::SetConditionalChunked { .. } => true,
            Self::Get { .. }
            | Self::Exists { .. }
            | Self::CounterGet { .. }
            | Self::GetExpiry { .. }
            | Self::GetRange { .. }
            | Self::BytesLength { .. } => false,
        }
    }
}

/// Outcome of one prepared operation.
///
/// Exhaustive by design: every consumer in the workspace must handle every
/// outcome, so adding an operation breaks matches at compile time instead of
/// silently falling into a wildcard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationResult {
    /// `Get` payload (`None` when absent, expired, or — via error — mistyped).
    Value(Option<Bytes>),
    /// `Get` hit a chunked root: the engine must resolve these bytes
    /// through the chunk lane before replying. Never crosses the wire and
    /// never persists: reads answer inline, so no WAL record, dedup entry,
    /// or checkpoint band ever names this variant. It exists so the
    /// deterministic store can name "bytes live elsewhere" without
    /// performing I/O itself.
    ChunkedValue {
        /// Manifest addressing the immutable chunk sequence.
        manifest: ManifestId,
        /// Total logical bytes across the manifest.
        logical_len: u64,
    },
    /// `Set` completed with the new version.
    Stored {
        /// Version after the store.
        version: crate::object::ObjectVersion,
    },
    /// `Delete` completed; `existed` names a previously live object.
    Deleted {
        /// Whether a live object was removed.
        existed: bool,
    },
    /// `Exists` answer.
    Exists(bool),
    /// `CounterGet` value (`None` when absent).
    Counter(Option<i64>),
    /// `CounterAdd` completed with the new value and version.
    CounterUpdated {
        /// Value after the addition.
        value: i64,
        /// Version after the addition.
        version: crate::object::ObjectVersion,
    },
    /// `ExpireAt` outcome.
    ExpirySet {
        /// Whether the key was live and took the expiry.
        applied: bool,
    },
    /// `PersistExpiry` outcome.
    ExpiryPersisted {
        /// Whether an expiry was actually removed.
        removed: bool,
    },
    /// `GetExpiry` answer (`None` when the key is absent).
    Expiry(Option<Expiry>),
    /// `BytesLength` answer (`None` when the key is absent).
    Length(Option<u64>),
    /// `SetConditional` outcome: whether the condition held and the store
    /// applied. `applied: false` carries no version and mutated nothing.
    ConditionalSet {
        /// Whether the condition held and the value was stored.
        applied: bool,
        /// Version after the store (`None` when not applied).
        version: Option<crate::object::ObjectVersion>,
    },
}

/// Native operation failure. Reads and validations fail; validated writes
/// apply deterministically.
///
/// Exhaustive like [`OperationResult`]: new failure modes must be handled
/// explicitly everywhere they surface.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OpError {
    /// The key holds a different logical type; nothing was mutated.
    #[error("wrong type: operation needs {expected}, key holds {found}")]
    WrongType {
        /// Type the operation required.
        expected: ObjectType,
        /// Type actually stored.
        found: ObjectType,
    },
    /// A counter addition overflowed `i64`; nothing was mutated.
    #[error("counter addition overflows i64")]
    CounterOverflow,
    /// A `SetRange` met a chunked root its admission plan did not expect:
    /// the root changed representation between the engine's peek and
    /// preparation. Nothing was mutated; the client retries and the retry
    /// re-plans against the new root. Unreachable on the current
    /// single-owner paths (peek and prepare share one synchronous tablet
    /// turn), kept total for future replica proposals.
    #[error("range base changed representation during admission; retry")]
    StaleRangeBase,
}

/// Canonical wire tags. Fixed forever within framing version 1.
const TAG_OP_WRONG_TYPE: u8 = 1;
const TAG_OP_OVERFLOW: u8 = 2;
/// Stale range-base tag. New tags never reuse old ones.
const TAG_OP_STALE_RANGE_BASE: u8 = 3;

impl Encode for OpError {
    fn encoded_len(&self) -> usize {
        match self {
            Self::WrongType { .. } => 1 + 1 + 1,
            Self::CounterOverflow | Self::StaleRangeBase => 1,
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::WrongType { expected, found } => {
                out.push(TAG_OP_WRONG_TYPE);
                expected.encode(out);
                found.encode(out);
            }
            Self::CounterOverflow => out.push(TAG_OP_OVERFLOW),
            Self::StaleRangeBase => out.push(TAG_OP_STALE_RANGE_BASE),
        }
    }
}

impl Decode for OpError {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (tag, first) = u8::decode(input)?;
        match tag {
            TAG_OP_WRONG_TYPE => {
                let (expected, second) = ObjectType::decode(&input[first..])?;
                let (found, third) = ObjectType::decode(&input[first + second..])?;
                Ok((Self::WrongType { expected, found }, first + second + third))
            }
            TAG_OP_OVERFLOW => Ok((Self::CounterOverflow, first)),
            TAG_OP_STALE_RANGE_BASE => Ok((Self::StaleRangeBase, first)),
            other => Err(CodecError::InvalidTag {
                kind: "op-error",
                tag: other,
            }),
        }
    }
}

/// Canonical wire tags for [`OperationResult`]. Fixed forever within
/// framing version 1.
const TAG_RES_VALUE: u8 = 1;
const TAG_RES_STORED: u8 = 2;
const TAG_RES_DELETED: u8 = 3;
const TAG_RES_EXISTS: u8 = 4;
const TAG_RES_COUNTER: u8 = 5;
const TAG_RES_COUNTER_UPDATED: u8 = 6;
const TAG_RES_EXPIRY_SET: u8 = 7;
const TAG_RES_EXPIRY_PERSISTED: u8 = 8;
const TAG_RES_EXPIRY: u8 = 9;
/// Engine-internal chunked-read marker. New tag, never reused; see the
/// variant docs for why it never persists.
const TAG_RES_CHUNKED: u8 = 10;
/// Logical byte-length answer tag. New tags never reuse old ones.
const TAG_RES_LENGTH: u8 = 11;
/// Conditional-set outcome tag. New tags never reuse old ones.
const TAG_RES_CONDITIONAL_SET: u8 = 12;

impl Encode for OperationResult {
    fn encoded_len(&self) -> usize {
        match self {
            Self::Value(value) => 1 + 1 + value.as_ref().map_or(0, |bytes| 4 + bytes.len()),
            Self::ChunkedValue { .. } => 1 + 32 + 8,
            Self::Stored { .. } => 1 + 8,
            Self::Deleted { .. }
            | Self::Exists(_)
            | Self::ExpirySet { .. }
            | Self::ExpiryPersisted { .. } => 1 + 1,
            Self::Counter(value) => 1 + 1 + usize::from(value.is_some()) * 8,
            Self::CounterUpdated { .. } => 1 + 8 + 8,
            Self::Expiry(value) => 1 + 1 + value.as_ref().map_or(0, Expiry::encoded_len),
            Self::Length(value) => 1 + 1 + usize::from(value.is_some()) * 8,
            Self::ConditionalSet { version, .. } => 1 + 1 + 1 + version.as_ref().map_or(0, |_| 8),
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::Value(value) => {
                out.push(TAG_RES_VALUE);
                match value {
                    None => out.push(0),
                    Some(bytes) => {
                        out.push(1);
                        encode_bytes(out, bytes);
                    }
                }
            }
            Self::Stored { version } => {
                out.push(TAG_RES_STORED);
                version.encode(out);
            }
            Self::Deleted { existed } => {
                out.push(TAG_RES_DELETED);
                existed.encode(out);
            }
            Self::Exists(present) => {
                out.push(TAG_RES_EXISTS);
                present.encode(out);
            }
            Self::Counter(value) => {
                out.push(TAG_RES_COUNTER);
                match value {
                    None => out.push(0),
                    Some(counter) => {
                        out.push(1);
                        out.extend_from_slice(&counter.to_le_bytes());
                    }
                }
            }
            Self::CounterUpdated { value, version } => {
                out.push(TAG_RES_COUNTER_UPDATED);
                out.extend_from_slice(&value.to_le_bytes());
                version.encode(out);
            }
            Self::ExpirySet { applied } => {
                out.push(TAG_RES_EXPIRY_SET);
                applied.encode(out);
            }
            Self::ExpiryPersisted { removed } => {
                out.push(TAG_RES_EXPIRY_PERSISTED);
                removed.encode(out);
            }
            Self::Expiry(value) => {
                out.push(TAG_RES_EXPIRY);
                match value {
                    None => out.push(0),
                    Some(expiry) => {
                        out.push(1);
                        expiry.encode(out);
                    }
                }
            }
            Self::ChunkedValue {
                manifest,
                logical_len,
            } => {
                out.push(TAG_RES_CHUNKED);
                out.extend_from_slice(manifest.as_bytes());
                logical_len.encode(out);
            }
            Self::Length(value) => {
                out.push(TAG_RES_LENGTH);
                match value {
                    None => out.push(0),
                    Some(len) => {
                        out.push(1);
                        len.encode(out);
                    }
                }
            }
            Self::ConditionalSet { applied, version } => {
                out.push(TAG_RES_CONDITIONAL_SET);
                applied.encode(out);
                match version {
                    None => out.push(0),
                    Some(version) => {
                        out.push(1);
                        version.encode(out);
                    }
                }
            }
        }
    }
}

impl Decode for OperationResult {
    #[allow(clippy::too_many_lines)]
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (tag, first) = u8::decode(input)?;
        match tag {
            TAG_RES_VALUE => {
                let (present, second) = bool::decode(&input[first..])?;
                if !present {
                    return Ok((Self::Value(None), first + second));
                }
                let (bytes, third) = decode_byte_vec(&input[first + second..])?;
                Ok((
                    Self::Value(Some(Bytes::from(bytes))),
                    first + second + third,
                ))
            }
            TAG_RES_STORED => {
                let (version, second) = ObjectVersion::decode(&input[first..])?;
                Ok((Self::Stored { version }, first + second))
            }
            TAG_RES_DELETED => {
                let (existed, second) = bool::decode(&input[first..])?;
                Ok((Self::Deleted { existed }, first + second))
            }
            TAG_RES_EXISTS => {
                let (present, second) = bool::decode(&input[first..])?;
                Ok((Self::Exists(present), first + second))
            }
            TAG_RES_COUNTER => {
                let (present, second) = bool::decode(&input[first..])?;
                if !present {
                    return Ok((Self::Counter(None), first + second));
                }
                let (raw, third) = <[u8; 8]>::decode(&input[first + second..])?;
                Ok((
                    Self::Counter(Some(i64::from_le_bytes(raw))),
                    first + second + third,
                ))
            }
            TAG_RES_COUNTER_UPDATED => {
                let (raw_value, second) = <[u8; 8]>::decode(&input[first..])?;
                let (version, third) = ObjectVersion::decode(&input[first + second..])?;
                Ok((
                    Self::CounterUpdated {
                        value: i64::from_le_bytes(raw_value),
                        version,
                    },
                    first + second + third,
                ))
            }
            TAG_RES_EXPIRY_SET => {
                let (applied, second) = bool::decode(&input[first..])?;
                Ok((Self::ExpirySet { applied }, first + second))
            }
            TAG_RES_EXPIRY_PERSISTED => {
                let (removed, second) = bool::decode(&input[first..])?;
                Ok((Self::ExpiryPersisted { removed }, first + second))
            }
            TAG_RES_EXPIRY => {
                let (present, second) = bool::decode(&input[first..])?;
                if !present {
                    return Ok((Self::Expiry(None), first + second));
                }
                let (expiry, third) = Expiry::decode(&input[first + second..])?;
                Ok((Self::Expiry(Some(expiry)), first + second + third))
            }
            TAG_RES_CHUNKED => {
                let (raw, second) = <[u8; 32]>::decode(&input[first..])?;
                let (logical_len, third) = u64::decode(&input[first + second..])?;
                Ok((
                    Self::ChunkedValue {
                        manifest: ManifestId::from_bytes(raw),
                        logical_len,
                    },
                    first + second + third,
                ))
            }
            TAG_RES_LENGTH => {
                let (present, second) = bool::decode(&input[first..])?;
                if !present {
                    return Ok((Self::Length(None), first + second));
                }
                let (len, third) = u64::decode(&input[first + second..])?;
                Ok((Self::Length(Some(len)), first + second + third))
            }
            TAG_RES_CONDITIONAL_SET => {
                let (applied, second) = bool::decode(&input[first..])?;
                let (present, third) = bool::decode(&input[first + second..])?;
                if !present {
                    return Ok((
                        Self::ConditionalSet {
                            applied,
                            version: None,
                        },
                        first + second + third,
                    ));
                }
                let (version, fourth) = ObjectVersion::decode(&input[first + second + third..])?;
                Ok((
                    Self::ConditionalSet {
                        applied,
                        version: Some(version),
                    },
                    first + second + third + fourth,
                ))
            }
            other => Err(CodecError::InvalidTag {
                kind: "operation-result",
                tag: other,
            }),
        }
    }
}

/// Terminal outcome of one durable mutating request: either the completed
/// result or the terminal rejection that must be replayed to a retrying
/// client instead of re-executing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DurableOutcome {
    /// The request completed with this result.
    Completed(OperationResult),
    /// The request was terminally rejected without mutating.
    Rejected(OpError),
    /// An object version could not advance; nothing mutated.
    VersionExhausted,
}

/// Canonical wire tags. Fixed forever within framing version 1.
const TAG_OUTCOME_COMPLETED: u8 = 1;
const TAG_OUTCOME_REJECTED: u8 = 2;
const TAG_OUTCOME_VERSION_EXHAUSTED: u8 = 3;

impl Encode for DurableOutcome {
    fn encoded_len(&self) -> usize {
        match self {
            Self::Completed(result) => 1 + result.encoded_len(),
            Self::Rejected(error) => 1 + error.encoded_len(),
            Self::VersionExhausted => 1,
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::Completed(result) => {
                out.push(TAG_OUTCOME_COMPLETED);
                result.encode(out);
            }
            Self::Rejected(error) => {
                out.push(TAG_OUTCOME_REJECTED);
                error.encode(out);
            }
            Self::VersionExhausted => out.push(TAG_OUTCOME_VERSION_EXHAUSTED),
        }
    }
}

impl Decode for DurableOutcome {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (tag, first) = u8::decode(input)?;
        match tag {
            TAG_OUTCOME_COMPLETED => {
                let (result, second) = OperationResult::decode(&input[first..])?;
                Ok((Self::Completed(result), first + second))
            }
            TAG_OUTCOME_REJECTED => {
                let (error, second) = OpError::decode(&input[first..])?;
                Ok((Self::Rejected(error), first + second))
            }
            TAG_OUTCOME_VERSION_EXHAUSTED => Ok((Self::VersionExhausted, first)),
            other => Err(CodecError::InvalidTag {
                kind: "durable-outcome",
                tag: other,
            }),
        }
    }
}

/// Maps one applied mutation to its operation result. This is the single
/// canonical `(Mutation, ApplyOutcome)` mapping: the live path and WAL
/// replay verification share it, so any divergence fails loudly instead of
/// forking semantics.
///
/// `is_persist_expiry` disambiguates `SetExpiry` outcomes (`ExpirySet` for
/// `ExpireAt`, `ExpiryPersisted` for `PersistExpiry`); the WAL record and
/// dedup entry always know the originating opcode.
///
/// # Panics
///
/// Panics on impossible mutation/outcome pairings (e.g. a `Set` yielding a
/// counter outcome): preparation and application run back-to-back on the
/// same state, so disagreement is an internal logic bug, never a runtime
/// condition.
#[must_use]
pub fn outcome_for(
    mutation: &Mutation,
    outcome: &ApplyOutcome,
    is_persist_expiry: bool,
) -> OperationResult {
    match (mutation, outcome) {
        // Inline, chunked, and spliced stores answer identically: the
        // result names the new version either way, never the representation.
        (
            Mutation::PutBytes { .. }
            | Mutation::ReplaceChunkedRoot { .. }
            | Mutation::SpliceBytes { .. },
            ApplyOutcome::Put { version },
        ) => OperationResult::Stored { version: *version },
        // Conditional stores answer with their applied flag: the mutation
        // only exists when the condition held, so reaching here means
        // applied with the new version.
        (
            Mutation::PutBytesWithExpiry { .. } | Mutation::ReplaceChunkedRootWithExpiry { .. },
            ApplyOutcome::Put { version },
        ) => OperationResult::ConditionalSet {
            applied: true,
            version: Some(*version),
        },
        (Mutation::Delete { .. }, ApplyOutcome::Deleted { existed }) => {
            OperationResult::Deleted { existed: *existed }
        }
        (Mutation::CounterAdd { .. }, ApplyOutcome::Counter { value, version }) => {
            OperationResult::CounterUpdated {
                value: *value,
                version: *version,
            }
        }
        (Mutation::SetExpiry { .. }, ApplyOutcome::Expiry { applied, .. }) => {
            if is_persist_expiry {
                OperationResult::ExpiryPersisted { removed: *applied }
            } else {
                OperationResult::ExpirySet { applied: *applied }
            }
        }
        (mutation, outcome) => {
            panic!("prepare/apply contract violated: {mutation:?} -> {outcome:?}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_codec::{Decode, Encode};
    use kivi_types::UnixMicros;

    fn round_trip<T>(value: &T) -> T
    where
        T: Encode + Decode + PartialEq + core::fmt::Debug,
    {
        let mut bytes = Vec::new();
        value.encode(&mut bytes);
        assert_eq!(bytes.len(), value.encoded_len(), "length prefix honest");
        let (back, consumed) = T::decode(&bytes).expect("decode");
        assert_eq!(consumed, bytes.len(), "exact consumption");
        assert_eq!(&back, value);
        back
    }

    #[test]
    fn operation_results_round_trip_every_variant() {
        let version = ObjectVersion::from_u64(7);
        round_trip(&OperationResult::Value(None));
        round_trip(&OperationResult::Value(Some(Bytes::from_static(b"v"))));
        round_trip(&OperationResult::ChunkedValue {
            manifest: ManifestId::from_bytes([0x22; 32]),
            logical_len: 3_000_000,
        });
        round_trip(&OperationResult::Stored { version });
        round_trip(&OperationResult::Deleted { existed: true });
        round_trip(&OperationResult::Exists(false));
        round_trip(&OperationResult::Counter(None));
        round_trip(&OperationResult::Counter(Some(-41)));
        round_trip(&OperationResult::CounterUpdated { value: 9, version });
        round_trip(&OperationResult::ExpirySet { applied: true });
        round_trip(&OperationResult::ExpiryPersisted { removed: false });
        round_trip(&OperationResult::Expiry(None));
        round_trip(&OperationResult::Expiry(Some(Expiry::at(
            UnixMicros::from_micros(99),
        ))));
        round_trip(&OperationResult::Expiry(Some(Expiry::NEVER)));
        round_trip(&OperationResult::Length(None));
        round_trip(&OperationResult::Length(Some(41)));
        round_trip(&OperationResult::ConditionalSet {
            applied: false,
            version: None,
        });
        round_trip(&OperationResult::ConditionalSet {
            applied: true,
            version: Some(version),
        });
    }

    #[test]
    fn op_errors_and_durable_outcomes_round_trip() {
        round_trip(&OpError::WrongType {
            expected: ObjectType::Bytes,
            found: ObjectType::StrictCounter,
        });
        round_trip(&OpError::CounterOverflow);
        round_trip(&DurableOutcome::Completed(OperationResult::Stored {
            version: ObjectVersion::FIRST,
        }));
        round_trip(&DurableOutcome::Rejected(OpError::CounterOverflow));
        round_trip(&DurableOutcome::VersionExhausted);
    }

    #[test]
    fn stored_wire_bytes_are_stable() {
        // Golden bytes: tag 2 then version 7 LE. Pin the layout deliberately.
        let mut bytes = Vec::new();
        OperationResult::Stored {
            version: ObjectVersion::from_u64(7),
        }
        .encode(&mut bytes);
        assert_eq!(bytes, vec![2, 7, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn unknown_result_tags_rejected() {
        assert!(OperationResult::decode(&[0xFF]).is_err());
        assert!(OpError::decode(&[0xFF]).is_err());
        assert!(DurableOutcome::decode(&[0xFF]).is_err());
    }

    #[test]
    fn outcome_for_covers_the_matrix() {
        let version = ObjectVersion::from_u64(3);
        let key = Key::from("k");
        assert_eq!(
            outcome_for(
                &Mutation::PutBytes {
                    key: key.clone(),
                    value: Bytes::from_static(b"v"),
                },
                &ApplyOutcome::Put { version },
                false,
            ),
            OperationResult::Stored { version }
        );
        assert_eq!(
            outcome_for(
                &Mutation::SetExpiry {
                    key: key.clone(),
                    expiry: Expiry::NEVER,
                },
                &ApplyOutcome::Expiry {
                    applied: true,
                    version: Some(version),
                },
                true,
            ),
            OperationResult::ExpiryPersisted { removed: true }
        );
        assert_eq!(
            outcome_for(
                &Mutation::SetExpiry {
                    key: key.clone(),
                    expiry: Expiry::at(UnixMicros::from_micros(5)),
                },
                &ApplyOutcome::Expiry {
                    applied: true,
                    version: Some(version),
                },
                false,
            ),
            OperationResult::ExpirySet { applied: true }
        );
        assert_eq!(
            outcome_for(
                &Mutation::PutBytesWithExpiry {
                    key: key.clone(),
                    value: Bytes::from_static(b"v"),
                    expiry: Expiry::NEVER,
                },
                &ApplyOutcome::Put { version },
                false,
            ),
            OperationResult::ConditionalSet {
                applied: true,
                version: Some(version),
            }
        );
        assert_eq!(
            outcome_for(
                &Mutation::ReplaceChunkedRootWithExpiry {
                    key,
                    manifest: ManifestId::from_bytes([0x11; 32]),
                    logical_len: 9,
                    expiry: Expiry::NEVER,
                },
                &ApplyOutcome::Put { version },
                false,
            ),
            OperationResult::ConditionalSet {
                applied: true,
                version: Some(version),
            }
        );
    }

    #[test]
    fn conditional_sets_evaluate_atomically_with_policies() {
        use crate::store::ObjectStore;
        let now = UnixMicros::from_micros(1_000_000);
        let mut store = ObjectStore::new();
        let key = Key::from("k");
        // IfAbsent stores on missing.
        match store
            .prepare(
                &Operation::SetConditional {
                    key: key.clone(),
                    value: Bytes::from_static(b"v1"),
                    condition: SetCondition::IfAbsent,
                    expiry: ExpiryPolicy::Clear,
                },
                now,
            )
            .expect("prepare")
        {
            crate::store::Prepared::Write(mutation) => {
                store.apply(&mutation, now).expect("apply");
            }
            crate::store::Prepared::Read(_) => panic!("IfAbsent on missing must write"),
        }
        // IfAbsent refuses when present without mutating the version.
        match store
            .prepare(
                &Operation::SetConditional {
                    key: key.clone(),
                    value: Bytes::from_static(b"v2"),
                    condition: SetCondition::IfAbsent,
                    expiry: ExpiryPolicy::Clear,
                },
                now,
            )
            .expect("prepare")
        {
            crate::store::Prepared::Read(OperationResult::ConditionalSet { applied, .. }) => {
                assert!(!applied);
            }
            _ => panic!("IfAbsent on present must not apply"),
        }
        // IfPresent stores and Keep preserves a dated expiry.
        let deadline = UnixMicros::from_micros(9_000_000);
        match store
            .prepare(
                &Operation::ExpireAt {
                    key: key.clone(),
                    expires_at: deadline,
                },
                now,
            )
            .expect("prepare")
        {
            crate::store::Prepared::Write(mutation) => {
                store.apply(&mutation, now).expect("apply");
            }
            crate::store::Prepared::Read(_) => panic!("expire must write"),
        }
        match store
            .prepare(
                &Operation::SetConditional {
                    key: key.clone(),
                    value: Bytes::from_static(b"v3"),
                    condition: SetCondition::IfPresent,
                    expiry: ExpiryPolicy::Keep,
                },
                now,
            )
            .expect("prepare")
        {
            crate::store::Prepared::Write(mutation) => {
                store.apply(&mutation, now).expect("apply");
            }
            crate::store::Prepared::Read(_) => panic!("IfPresent on present must write"),
        }
        match store
            .prepare(&Operation::GetExpiry { key: key.clone() }, now)
            .expect("prepare")
        {
            crate::store::Prepared::Read(OperationResult::Expiry(Some(expiry))) => {
                assert_eq!(expiry, Expiry::at(deadline));
            }
            _ => panic!("Keep must preserve the deadline"),
        }
    }

    #[test]
    fn ranges_slice_and_measure_without_payload_reads() {
        use crate::store::ObjectStore;
        let now = UnixMicros::from_micros(1_000_000);
        let mut store = ObjectStore::new();
        let key = Key::from("k");
        // Missing reads answer absent.
        match store
            .prepare(
                &Operation::GetRange {
                    key: key.clone(),
                    offset: 0,
                    len: 4,
                },
                now,
            )
            .expect("prepare")
        {
            crate::store::Prepared::Read(OperationResult::Value(None)) => {}
            _ => panic!("missing range must be None"),
        }
        match store
            .prepare(&Operation::BytesLength { key: key.clone() }, now)
            .expect("prepare")
        {
            crate::store::Prepared::Read(OperationResult::Length(None)) => {}
            _ => panic!("missing length must be None"),
        }
        match store
            .prepare(
                &Operation::Set {
                    key: key.clone(),
                    value: Bytes::from_static(b"hello"),
                },
                now,
            )
            .expect("prepare")
        {
            crate::store::Prepared::Write(mutation) => {
                store.apply(&mutation, now).expect("apply");
            }
            crate::store::Prepared::Read(_) => panic!("set must write"),
        }
        match store
            .prepare(
                &Operation::GetRange {
                    key: key.clone(),
                    offset: 1,
                    len: 3,
                },
                now,
            )
            .expect("prepare")
        {
            crate::store::Prepared::Read(OperationResult::Value(Some(bytes))) => {
                assert_eq!(&bytes[..], b"ell");
            }
            _ => panic!("range must slice"),
        }
        match store
            .prepare(
                &Operation::GetRange {
                    key: key.clone(),
                    offset: 99,
                    len: 4,
                },
                now,
            )
            .expect("prepare")
        {
            crate::store::Prepared::Read(OperationResult::Value(Some(bytes))) => {
                assert!(bytes.is_empty());
            }
            _ => panic!("past-the-end range must be empty"),
        }
        match store
            .prepare(&Operation::BytesLength { key }, now)
            .expect("prepare")
        {
            crate::store::Prepared::Read(OperationResult::Length(Some(len))) => {
                assert_eq!(len, 5);
            }
            _ => panic!("length must measure"),
        }
    }
}
