//! Kivi-owned replicated command (stage G, framing v2 since stage M).
//!
//! The consensus application data is [`ReplicatedMutation`]: a deterministic
//! [`MutationEnvelope`] (tablet authority,
//! request identity, typed [`Mutation`](kivi_state::Mutation) IR) plus the
//! deterministic [`expected`](kivi_state::OperationResult) outcome the
//! proposer computed and every replica's apply must reproduce, plus the two
//! proposer-materialized inputs deterministic replay needs beyond the
//! envelope:
//!
//! * `now`: the leader's client-boundary wall time. [`ObjectStore::apply`](kivi_state::ObjectStore::apply)
//!   takes `now` for expiry comparison, so replicas must apply at the
//!   leader's timestamp, never their own wall clock — otherwise an object
//!   expiring between propose and apply would fork outcomes (RFC §50:
//!   nondeterminism is materialized by the proposer before enveloping).
//! * `ack_floor`: the client's acknowledgement watermark. Session floor
//!   advances drop retained outcomes, so the advance must replicate in log
//!   order or replicas would retain different outcome sets.
//!
//! The encoding is canonical little-endian Kivi bytes — never JSON, never
//! `OpenRaft`'s memory representation, never serde:
//!
//! ```text
//! version u16 (= 2), now u64 micros, ack_floor u64,
//! envelope_len u32, envelope[…], expected_len u32, expected[…]
//! ```
//!
//! Framing v1 (no `now`/`ack_floor`) is rejected: no v1 command was ever
//! committed by a production group (v1 only served the density spike and
//! unit tests), so the breaking internal change aliases no durable history.
//!
//! Replicas never re-execute proposer choices: the envelope plus `now`
//! materialize nondeterminism before replication, and `expected` lets apply
//! verify it reproduced the identical outcome (divergence is
//! corruption-worthy).

use kivi_codec::{Decode, Encode};
use kivi_state::{MutationEnvelope, OperationResult};
use kivi_types::{RequestSeq, UnixMicros};
/// Framing version of [`ReplicatedMutation`].
pub const REPLICATED_MUTATION_VERSION: u16 = 2;

/// What gets proposed, committed, and applied: one deterministic command
/// plus the outcome it must produce everywhere and the materialized inputs
/// replay needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicatedMutation {
    envelope: MutationEnvelope,
    expected: OperationResult,
    now: UnixMicros,
    ack_floor: RequestSeq,
}

/// Outcome installed for one applied entry: the deterministic result
/// returned to the original caller and to every same-identity retry
/// (lost-response failover). Command entries always carry one; leader
/// barriers (blank) and membership entries carry none — no client waits
/// on those positions, and `None` keeps that fact explicit instead of
/// inventing an outcome no one asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicatedOutcome {
    outcome: Option<OperationResult>,
}

/// Replicated-command decode failure (structural only; semantic checks
/// like authority fencing happen at apply time, never here).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ReplicatedMutationError {
    /// Framing version this binary cannot speak.
    #[error("unsupported replicated-mutation version {found}")]
    UnsupportedVersion {
        /// Observed version.
        found: u16,
    },
    /// Truncated input where framing promised bytes.
    #[error("truncated replicated mutation")]
    Truncated,
    /// A length prefix exceeds the input (never allocated blindly).
    #[error("declared length {len} exceeds input {available}")]
    Oversized {
        /// Declared length.
        len: usize,
        /// Bytes actually available.
        available: usize,
    },
    /// The envelope bytes do not decode.
    #[error("undecodable mutation envelope")]
    BadEnvelope,
    /// The expected-outcome bytes do not decode.
    #[error("undecodable expected outcome")]
    BadExpected,
    /// Trailing bytes after the framed command.
    #[error("trailing bytes in replicated mutation")]
    TrailingBytes,
}

impl ReplicatedMutation {
    /// Builds the command the leader proposes: the validated envelope
    /// plus the deterministic outcome its apply produced (and every
    /// replica's apply must reproduce), the client-boundary wall time the
    /// leader materialized, and the client's acknowledgement watermark.
    #[must_use]
    pub const fn new(
        envelope: MutationEnvelope,
        expected: OperationResult,
        now: UnixMicros,
        ack_floor: RequestSeq,
    ) -> Self {
        Self {
            envelope,
            expected,
            now,
            ack_floor,
        }
    }

    /// Returns the carried envelope (authority, identity, mutation IR).
    #[must_use]
    pub const fn envelope(&self) -> &MutationEnvelope {
        &self.envelope
    }

    /// Returns the deterministic outcome apply must reproduce.
    #[must_use]
    pub const fn expected(&self) -> &OperationResult {
        &self.expected
    }

    /// Returns the leader-materialized wall time replicas apply at.
    #[must_use]
    pub const fn now(&self) -> UnixMicros {
        self.now
    }

    /// Returns the client acknowledgement watermark replicated with this
    /// command.
    #[must_use]
    pub const fn ack_floor(&self) -> RequestSeq {
        self.ack_floor
    }

    /// Encodes the canonical bytes.
    #[must_use]
    pub fn encode_to_vec(&self) -> Vec<u8> {
        let envelope = self.envelope.encode_to_vec();
        let expected = self.expected.encode_to_vec();
        let mut out = Vec::with_capacity(2 + 8 + 8 + 4 + envelope.len() + 4 + expected.len());
        out.extend_from_slice(&REPLICATED_MUTATION_VERSION.to_le_bytes());
        out.extend_from_slice(&self.now.as_micros().to_le_bytes());
        out.extend_from_slice(&self.ack_floor.as_u64().to_le_bytes());
        push_blob(&mut out, &envelope);
        push_blob(&mut out, &expected);
        out
    }

    /// Decodes exactly one command: structural validation only.
    ///
    /// # Errors
    ///
    /// Returns [`ReplicatedMutationError`] on version mismatch,
    /// truncation, oversize declarations, undecodable parts, or trailing
    /// bytes. Framing v1 is rejected as
    /// [`UnsupportedVersion`](ReplicatedMutationError::UnsupportedVersion):
    /// it never committed in production and carries no materialized time.
    pub fn decode_exact(input: &[u8]) -> Result<Self, ReplicatedMutationError> {
        use ReplicatedMutationError as Fault;
        if input.len() < 2 + 8 + 8 {
            return Err(Fault::Truncated);
        }
        let version = u16::from_le_bytes(input[..2].try_into().unwrap_or([0; 2]));
        if version != REPLICATED_MUTATION_VERSION {
            return Err(Fault::UnsupportedVersion { found: version });
        }
        let now = UnixMicros::from_micros(u64::from_le_bytes(
            input[2..10].try_into().unwrap_or([0; 8]),
        ));
        let ack_floor = RequestSeq::from_u64(u64::from_le_bytes(
            input[10..18].try_into().unwrap_or([0; 8]),
        ));
        let (envelope_bytes, rest) = take_blob(&input[18..])?;
        let (expected_bytes, rest) = take_blob(rest)?;
        if !rest.is_empty() {
            return Err(Fault::TrailingBytes);
        }
        let envelope =
            MutationEnvelope::decode_exact(envelope_bytes).map_err(|_| Fault::BadEnvelope)?;
        let expected =
            OperationResult::decode_exact(expected_bytes).map_err(|_| Fault::BadExpected)?;
        Ok(Self {
            envelope,
            expected,
            now,
            ack_floor,
        })
    }
}

impl ReplicatedOutcome {
    /// Builds the installed outcome for one committed command.
    #[must_use]
    pub const fn new(outcome: OperationResult) -> Self {
        Self {
            outcome: Some(outcome),
        }
    }

    /// Builds the empty outcome for barrier/membership positions.
    #[must_use]
    pub const fn none() -> Self {
        Self { outcome: None }
    }

    /// Returns the deterministic outcome (`None` on barrier/membership
    /// positions).
    #[must_use]
    pub const fn outcome(&self) -> Option<&OperationResult> {
        self.outcome.as_ref()
    }
}

impl core::fmt::Display for ReplicatedMutation {
    /// Short identity summary (session/sequence plus opcode family).
    /// `OpenRaft`'s `AppData` bound requires `Display`; the canonical
    /// encoding stays the byte format above, never this string.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "mutation({}#{})",
            self.envelope.client().session(),
            self.envelope.client().seq(),
        )
    }
}

impl core::fmt::Display for ReplicatedOutcome {
    /// Installed-outcome summary. Same `Display`-for-`AppDataResponse`
    /// role as above; replies on the wire stay canonical bytes.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.outcome {
            Some(outcome) => write!(f, "outcome({outcome:?})"),
            None => write!(f, "outcome(none)"),
        }
    }
}

fn push_blob(out: &mut Vec<u8>, bytes: &[u8]) {
    // Both blobs are bounded upstream by envelope validation
    // (`MAX_FRAME_BYTES`); the length prefix stays exact-width.
    let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
}

fn take_blob(input: &[u8]) -> Result<(&[u8], &[u8]), ReplicatedMutationError> {
    use ReplicatedMutationError as Fault;
    if input.len() < 4 {
        return Err(Fault::Truncated);
    }
    let len = u32::from_le_bytes(input[..4].try_into().unwrap_or([0; 4])) as usize;
    if input.len() < 4 + len {
        return Err(Fault::Oversized {
            len,
            available: input.len(),
        });
    }
    Ok((&input[4..4 + len], &input[4 + len..]))
}

#[cfg(test)]
mod tests {
    use kivi_state::{Key, Mutation, ObjectVersion};
    use kivi_types::{
        NamespaceId, RequestIdentity, RequestSeq, SessionId, TabletAuthority, TabletEpoch,
        TabletId, UnixMicros, WriteGuardGeneration,
    };

    use super::*;

    fn sample() -> ReplicatedMutation {
        let authority = TabletAuthority::new(
            TabletId::from_u64(9),
            TabletEpoch::INITIAL,
            WriteGuardGeneration::INITIAL,
        );
        let envelope = MutationEnvelope::new(
            NamespaceId::from_u64(1),
            authority,
            RequestIdentity::new(SessionId::from_u128(0x5E55), RequestSeq::from_u64(1)),
            None,
            Mutation::CounterAdd {
                key: Key::from("n"),
                delta: 41,
            },
        );
        ReplicatedMutation::new(
            envelope,
            OperationResult::CounterUpdated {
                value: 41,
                version: ObjectVersion::from_u64(7),
            },
            UnixMicros::from_micros(1_000_000),
            RequestSeq::from_u64(0),
        )
    }

    #[test]
    fn command_round_trips() {
        let command = sample();
        let back = ReplicatedMutation::decode_exact(&command.encode_to_vec()).expect("decodes");
        assert_eq!(back, command);
        assert_eq!(
            back.envelope().operation(),
            &Mutation::CounterAdd {
                key: Key::from("n"),
                delta: 41,
            }
        );
        assert_eq!(
            back.expected(),
            &OperationResult::CounterUpdated {
                value: 41,
                version: ObjectVersion::from_u64(7),
            }
        );
    }

    #[test]
    fn corrupt_framing_fails_loudly() {
        let good = sample().encode_to_vec();
        // Truncations at every prefix point fail (header is now
        // 2 + 8 + 8 bytes before the blobs).
        for cut in [0, 1, 2, 3, 6, 10, 17, 18, 22] {
            assert!(
                ReplicatedMutation::decode_exact(&good[..cut.min(good.len())]).is_err(),
                "cut at {cut} must fail"
            );
        }
        // Materialized inputs survive the round trip.
        let back = ReplicatedMutation::decode_exact(&good).expect("decodes");
        assert_eq!(back.now(), UnixMicros::from_micros(1_000_000));
        assert_eq!(back.ack_floor(), RequestSeq::from_u64(0));
        // Version bump fails.
        let mut versioned = good.clone();
        versioned[0] = versioned[0].wrapping_add(1);
        assert!(matches!(
            ReplicatedMutation::decode_exact(&versioned),
            Err(ReplicatedMutationError::UnsupportedVersion { .. })
        ));
        // Framing v1 (no materialized time) is rejected, never misread.
        let mut v1 = good.clone();
        v1[0] = 1;
        v1[1] = 0;
        assert!(matches!(
            ReplicatedMutation::decode_exact(&v1),
            Err(ReplicatedMutationError::UnsupportedVersion { found: 1 })
        ));
        // Trailing garbage fails.
        let mut trailed = good.clone();
        trailed.push(0xFF);
        assert!(matches!(
            ReplicatedMutation::decode_exact(&trailed),
            Err(ReplicatedMutationError::TrailingBytes)
        ));
        // A flipped idempotency tag breaks the envelope decode (bad
        // envelope, not a wrong value). Tag offset inside the envelope is
        // 2 + 8*4 + 16 + 8; the envelope blob starts at framing offset 22.
        let mut damaged = good;
        damaged[22 + 2 + 8 * 4 + 16 + 8] = 0x7F;
        assert!(matches!(
            ReplicatedMutation::decode_exact(&damaged),
            Err(ReplicatedMutationError::BadEnvelope)
        ));
    }
}
