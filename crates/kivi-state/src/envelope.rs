//! Mutation envelope: the versioned carrier binding a typed mutation to its
//! routing, fencing, and deduplication identity (RFC §48, §49).
//!
//! Every application mutation travels as a [`MutationEnvelope`]: namespace,
//! tablet authority triple, client request identity, optional durable
//! idempotency key, and one typed [`Mutation`]. Replicas replay the envelope
//! bytes; they never re-execute proposer choices (RFC §50: nondeterminism is
//! materialized by the proposer before enveloping).
//!
//! Framing version is 1. The pre-release opaque-bytes payload was replaced by
//! typed Mutation IR before anything durable existed anywhere, so no
//! compatibility burden carries over — v1 simply always meant this layout.

use kivi_codec::{CodecError, Decode, Encode, MAX_FRAME_BYTES};
use kivi_types::{
    AuthorityMismatch, IdempotencyKey, NamespaceId, RequestIdentity, RequestSeq, SessionId,
    TabletAuthority, TabletEpoch, TabletId, WriteGuardGeneration,
};

use crate::mutation::Mutation;

/// Version of the envelope framing defined here.
pub const MUTATION_ENVELOPE_VERSION: u16 = 1;

/// Canonical field order on the wire: version `u16`, namespace `u64`,
/// tablet `u64`, epoch `u64`, guard `u64`, session `u128`, sequence `u64`,
/// idempotency tag (`0` absent, `1` present + `u128` key), then the typed
/// mutation in its canonical encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutationEnvelope {
    version: u16,
    namespace: NamespaceId,
    tablet: TabletId,
    epoch: TabletEpoch,
    guard: WriteGuardGeneration,
    client: RequestIdentity,
    idempotency: Option<IdempotencyKey>,
    operation: Mutation,
}

impl MutationEnvelope {
    /// Builds an envelope for one mutation attempt.
    ///
    /// The `authority` triple bundles the tablet, epoch, and write-guard
    /// generation the writer claims; it is checked against live state via
    /// [`check_against`](Self::check_against), not trusted on arrival.
    /// `operation` is the already-validated deterministic mutation.
    ///
    /// Bounds are enforced by [`validate`](Self::validate), not here, so
    /// construction itself cannot fail and invalid envelopes are representable
    /// for negative-path testing.
    #[must_use]
    pub fn new(
        namespace: NamespaceId,
        authority: TabletAuthority,
        client: RequestIdentity,
        idempotency: Option<IdempotencyKey>,
        operation: Mutation,
    ) -> Self {
        Self {
            version: MUTATION_ENVELOPE_VERSION,
            namespace,
            tablet: authority.tablet(),
            epoch: authority.epoch(),
            guard: authority.guard(),
            client,
            idempotency,
            operation,
        }
    }

    /// Returns the envelope framing version.
    #[must_use]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// Returns the target namespace.
    #[must_use]
    pub const fn namespace(&self) -> NamespaceId {
        self.namespace
    }

    /// Returns the target tablet.
    #[must_use]
    pub const fn tablet(&self) -> TabletId {
        self.tablet
    }

    /// Returns the claimed tablet epoch.
    #[must_use]
    pub const fn epoch(&self) -> TabletEpoch {
        self.epoch
    }

    /// Returns the claimed write-guard generation.
    #[must_use]
    pub const fn guard(&self) -> WriteGuardGeneration {
        self.guard
    }

    /// Reconstructs the claimed authority triple.
    #[must_use]
    pub const fn authority(&self) -> TabletAuthority {
        TabletAuthority::new(self.tablet, self.epoch, self.guard)
    }

    /// Returns the client request identity for lost-reply deduplication.
    #[must_use]
    pub const fn client(&self) -> RequestIdentity {
        self.client
    }

    /// Returns the optional durable idempotency key.
    #[must_use]
    pub const fn idempotency(&self) -> Option<IdempotencyKey> {
        self.idempotency
    }

    /// Returns the typed mutation carried by this envelope.
    #[must_use]
    pub fn operation(&self) -> &Mutation {
        &self.operation
    }

    /// Validates the envelope against the `current` tablet authority.
    ///
    /// A stale owner may still execute code, but its mutations commit nothing
    /// (RFC §194, §251 fencing invariant).
    ///
    /// # Errors
    ///
    /// Returns [`AuthorityMismatch`] if the claimed triple is stale,
    /// misrouted, or otherwise not current.
    pub fn check_against(&self, current: &TabletAuthority) -> Result<(), AuthorityMismatch> {
        self.authority().check_against(current)
    }

    /// Validates self-contained bounds: supported version and encoded size.
    ///
    /// # Errors
    ///
    /// Returns [`EnvelopeError::UnsupportedVersion`] for envelopes from a
    /// newer framing, or [`EnvelopeError::OperationTooLarge`] when the
    /// encoded mutation exceeds `MAX_FRAME_BYTES`.
    pub fn validate(&self) -> Result<(), EnvelopeError> {
        if self.version != MUTATION_ENVELOPE_VERSION {
            return Err(EnvelopeError::UnsupportedVersion {
                found: self.version,
            });
        }
        let len = self.operation.encoded_len();
        if len > MAX_FRAME_BYTES {
            return Err(EnvelopeError::OperationTooLarge {
                len,
                max: MAX_FRAME_BYTES,
            });
        }
        Ok(())
    }
}

impl Encode for MutationEnvelope {
    fn encoded_len(&self) -> usize {
        2 + 8
            + 8
            + 8
            + 8
            + 16
            + 8
            + 1
            + self.idempotency.map_or(0, |_| 16)
            + self.operation.encoded_len()
    }

    fn encode(&self, out: &mut Vec<u8>) {
        self.version.encode(out);
        self.namespace.as_u64().encode(out);
        self.tablet.as_u64().encode(out);
        self.epoch.as_u64().encode(out);
        self.guard.as_u64().encode(out);
        self.client.session().as_u128().encode(out);
        self.client.seq().as_u64().encode(out);
        match self.idempotency {
            None => out.push(0),
            Some(key) => {
                out.push(1);
                key.as_u128().encode(out);
            }
        }
        self.operation.encode(out);
    }
}

impl Decode for MutationEnvelope {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (version, first) = u16::decode(input)?;
        let (namespace, second) = u64::decode(&input[first..])?;
        let (tablet, third) = u64::decode(&input[first + second..])?;
        let (epoch, fourth) = u64::decode(&input[first + second + third..])?;
        let (guard, fifth) = u64::decode(&input[first + second + third + fourth..])?;
        let prefix = first + second + third + fourth + fifth;
        let (session, sixth) = u128::decode(&input[prefix..])?;
        let (seq, seventh) = u64::decode(&input[prefix + sixth..])?;
        let tag_at = prefix + sixth + seventh;
        let (tag, eighth) = u8::decode(&input[tag_at..])?;
        let (idempotency, ninth) = match tag {
            0 => (None, 0),
            1 => {
                let (key, used) = u128::decode(&input[tag_at + eighth..])?;
                (Some(IdempotencyKey::from_u128(key)), used)
            }
            other => {
                return Err(CodecError::InvalidTag {
                    kind: "idempotency",
                    tag: other,
                });
            }
        };
        let payload_at = tag_at + eighth + ninth;
        let (operation, tenth) = Mutation::decode(&input[payload_at..])?;
        let envelope = Self {
            version,
            namespace: NamespaceId::from_u64(namespace),
            tablet: TabletId::from_u64(tablet),
            epoch: TabletEpoch::from_u64(epoch),
            guard: WriteGuardGeneration::from_u64(guard),
            client: RequestIdentity::new(SessionId::from_u128(session), RequestSeq::from_u64(seq)),
            idempotency,
            operation,
        };
        Ok((envelope, payload_at + tenth))
    }
}

/// Self-contained envelope admission failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum EnvelopeError {
    /// Envelope framing version is not supported by this binary.
    #[error("unsupported envelope version {found}")]
    UnsupportedVersion {
        /// Observed version.
        found: u16,
    },
    /// Encoded mutation exceeds the admission bound.
    #[error("operation of {len} bytes exceeds maximum {max}")]
    OperationTooLarge {
        /// Observed encoded length in bytes.
        len: usize,
        /// Maximum accepted length in bytes.
        max: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::Key;

    fn sample() -> (MutationEnvelope, TabletAuthority) {
        let authority = TabletAuthority::new(
            TabletId::from_u64(918),
            TabletEpoch::from_u64(381),
            WriteGuardGeneration::from_u64(12),
        );
        let envelope = MutationEnvelope::new(
            NamespaceId::from_u64(3),
            authority,
            RequestIdentity::new(SessionId::from_u128(0x00C0_FFEE), RequestSeq::from_u64(7)),
            Some(IdempotencyKey::from_u128(0x000A_11CE)),
            Mutation::CounterAdd {
                key: Key::from("sessions:user:123"),
                delta: 1,
            },
        );
        (envelope, authority)
    }

    #[test]
    fn envelope_round_trips_with_and_without_idempotency() {
        let (envelope, _) = sample();
        assert_eq!(envelope.encoded_len(), envelope.encode_to_vec().len());
        let decoded =
            MutationEnvelope::decode_exact(&envelope.encode_to_vec()).expect("round trip");
        assert_eq!(decoded, envelope);
        assert_eq!(
            decoded.operation(),
            &Mutation::CounterAdd {
                key: Key::from("sessions:user:123"),
                delta: 1,
            }
        );

        let (mut bare, _) = sample();
        bare.idempotency = None;
        bare.operation = Mutation::Delete {
            key: Key::from("gone"),
        };
        let decoded = MutationEnvelope::decode_exact(&bare.encode_to_vec()).expect("bare");
        assert_eq!(decoded, bare);
        assert_eq!(decoded.idempotency(), None);
    }

    #[test]
    fn fresh_envelope_validates_and_matches_its_authority() {
        let (envelope, authority) = sample();
        assert_eq!(envelope.version(), MUTATION_ENVELOPE_VERSION);
        assert_eq!(envelope.authority(), authority);
        assert_eq!(envelope.validate(), Ok(()));
        assert_eq!(envelope.check_against(&authority), Ok(()));
    }

    #[test]
    fn stale_claim_is_rejected_against_newer_authority() {
        let (envelope, _) = sample();
        let newer = TabletAuthority::new(
            TabletId::from_u64(918),
            TabletEpoch::from_u64(382),
            WriteGuardGeneration::from_u64(1),
        );
        assert!(envelope.check_against(&newer).is_err());
    }

    #[test]
    fn unknown_tags_are_rejected() {
        let (envelope, _) = sample();
        let mut encoded = envelope.encode_to_vec();
        // Flip the idempotency tag byte (offset 2+8*4+16+8 = 58).
        let tag_at = 2 + 8 + 8 + 8 + 8 + 16 + 8;
        encoded[tag_at] = 0x7F;
        assert!(MutationEnvelope::decode(&encoded).is_err());
        // Truncate inside the typed mutation.
        let mut short = envelope.encode_to_vec();
        short.pop();
        assert!(MutationEnvelope::decode(&short).is_err());
    }
}
