//! Request identity for safe retries (RFC §63–§65).
//!
//! Native mutations carry a ([`SessionId`], [`RequestSeq`]) pair. A client
//! retry with the same pair after a lost reply returns the recorded outcome
//! instead of executing a second time. Optional durable idempotency across
//! client restarts uses [`IdempotencyKey`]; reusing a key for a *different*
//! logical mutation is a conflict, never a second execution.

use core::fmt;

use crate::ids::{RequestSeq, SessionId};

/// Unique address of one mutation attempt within a client session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestIdentity {
    session: SessionId,
    seq: RequestSeq,
}

impl RequestIdentity {
    /// Builds an identity from its two halves.
    #[must_use]
    pub const fn new(session: SessionId, seq: RequestSeq) -> Self {
        Self { session, seq }
    }

    /// Returns the session half.
    #[must_use]
    pub const fn session(self) -> SessionId {
        self.session
    }

    /// Returns the sequence half.
    #[must_use]
    pub const fn seq(self) -> RequestSeq {
        self.seq
    }
}

impl fmt::Display for RequestIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.session, self.seq)
    }
}

/// One mutating request's full retry identity: who issued it, which sequence
/// number it carries, and how far the client has acknowledged.
///
/// The server uses `client` for exactly-once lookup and `ack_floor` to
/// advance the session's durable floor (dropping outcomes the client has
/// explicitly acknowledged and will never retry). The floor only moves
/// forward over consecutively completed sequences, so an in-flight sequence
/// is never acknowledged beneath itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MutationIdentity {
    /// Session + sequence uniquely naming this mutation attempt.
    pub client: RequestIdentity,
    /// Client acknowledgement watermark: every sequence `<= ack_floor` in
    /// this session is complete from the client's perspective.
    pub ack_floor: RequestSeq,
}

impl MutationIdentity {
    /// Builds a retry identity from its parts.
    #[must_use]
    pub const fn new(client: RequestIdentity, ack_floor: RequestSeq) -> Self {
        Self { client, ack_floor }
    }
}

/// Durable idempotency key surviving client restarts (RFC §65).
///
/// The canonical durable form is a fixed 128-bit value. Human-facing clients
/// that prefer variable-length tokens (opaque strings) map them through a
/// client-side adapter as the low 128 bits of `BLAKE3(token)`, interpreted
/// little-endian; the adapter lives outside the correctness core, so the
/// canonical form here never changes. The stored record additionally binds
/// the key to a request fingerprint, so key reuse for a different mutation
/// yields `IDEMPOTENCY_CONFLICT` rather than a wrong replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IdempotencyKey(u128);

impl IdempotencyKey {
    /// Wraps a raw 128-bit value.
    #[must_use]
    pub const fn from_u128(value: u128) -> Self {
        Self(value)
    }

    /// Returns the raw 128-bit value.
    #[must_use]
    pub const fn as_u128(self) -> u128 {
        self.0
    }
}

impl fmt::Display for IdempotencyKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

impl From<IdempotencyKey> for u128 {
    /// Returns the raw 128-bit value.
    fn from(key: IdempotencyKey) -> Self {
        key.0
    }
}

impl From<u128> for IdempotencyKey {
    /// Wraps a raw 128-bit value.
    fn from(value: u128) -> Self {
        Self(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_exposes_both_halves() {
        let session = SessionId::from_u128(0x00C0_FFEE);
        let seq = RequestSeq::from_u64(41);
        let id = RequestIdentity::new(session, seq);
        assert_eq!(id.session(), session);
        assert_eq!(id.seq(), seq);
        assert_eq!(id.to_string(), format!("{session}:{seq}"));
    }

    #[test]
    fn identities_order_by_session_then_sequence() {
        let a = RequestIdentity::new(SessionId::from_u128(1), RequestSeq::from_u64(9));
        let b = RequestIdentity::new(SessionId::from_u128(1), RequestSeq::from_u64(10));
        let c = RequestIdentity::new(SessionId::from_u128(2), RequestSeq::from_u64(0));
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn idempotency_key_round_trips() {
        let key = IdempotencyKey::from_u128(0xDEAD_BEEF);
        assert_eq!(key.as_u128(), 0xDEAD_BEEF);
        assert_eq!(key.to_string().len(), 32);
    }
}
