//! Canonical encodings for every `kivi-types` value.
//!
//! Fixed-width little-endian integers, raw 32-byte chunk IDs,
//! length-prefixed names, and tagged composites. Field order below is the
//! wire contract; changing it requires a major version bump.

use core::time::Duration;

use kivi_types::{
    ChunkId, ClusterId, CommitPosition, CommitToken, Expiry, IdempotencyKey, NamespaceId,
    NamespaceName, NodeId, NodeIncarnation, PartitionHash, ReadContract, RequestIdentity,
    RequestSeq, SecurityDomainId, SessionId, TabletAuthority, TabletEpoch, TabletId, Ticks,
    WallTimestamp, WorkerId, WriteGuardGeneration,
};

use crate::error::CodecError;
use crate::traits::{Decode, Encode, decode_byte_slice, encode_bytes};

/// Implements fixed-width integer-backed encoding for one newtype.
macro_rules! impl_int_encoding {
    ($type:ty, $raw:ty, $width:expr) => {
        impl Encode for $type {
            fn encoded_len(&self) -> usize {
                $width
            }

            fn encode(&self, out: &mut Vec<u8>) {
                <$raw>::from(*self).encode(out);
            }
        }

        impl Decode for $type {
            fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
                let (raw, consumed) = <$raw>::decode(input)?;
                Ok((Self::from(raw), consumed))
            }
        }
    };
}

// `u64`-backed identifiers are 8 bytes little-endian on the wire.
impl_int_encoding!(NodeId, u64, 8);
impl_int_encoding!(WorkerId, u64, 8);
impl_int_encoding!(NamespaceId, u64, 8);
impl_int_encoding!(TabletId, u64, 8);
impl_int_encoding!(TabletEpoch, u64, 8);
impl_int_encoding!(WriteGuardGeneration, u64, 8);
impl_int_encoding!(NodeIncarnation, u64, 8);
impl_int_encoding!(CommitPosition, u64, 8);
impl_int_encoding!(RequestSeq, u64, 8);
impl_int_encoding!(SecurityDomainId, u64, 8);
impl_int_encoding!(Ticks, u64, 8);

// Wall-clock stamps encode as 8-byte little-endian _signed_ Unix micros -
// the Kivi-owned durable representation. Written by hand (not the macro)
// because `WallTimestamp` deliberately has no `From<i64>`: raw integers
// cross the boundary only here and at clock edges.
impl Encode for WallTimestamp {
    fn encoded_len(&self) -> usize {
        8
    }

    fn encode(&self, out: &mut Vec<u8>) {
        self.as_micros().encode(out);
    }
}

impl Decode for WallTimestamp {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (raw, consumed) = i64::decode(input)?;
        Ok((Self::from_micros(raw), consumed))
    }
}

// `u128`-backed identifiers are 16 bytes little-endian on the wire.
impl_int_encoding!(ClusterId, u128, 16);
impl_int_encoding!(SessionId, u128, 16);
impl_int_encoding!(IdempotencyKey, u128, 16);
impl_int_encoding!(PartitionHash, u128, 16);

// `u32`-backed identifiers are 4 bytes little-endian on the wire.

impl Encode for ChunkId {
    fn encoded_len(&self) -> usize {
        32
    }

    fn encode(&self, out: &mut Vec<u8>) {
        self.as_bytes().encode(out);
    }
}

impl Decode for ChunkId {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (bytes, consumed) = <[u8; 32]>::decode(input)?;
        Ok((Self::from_bytes(bytes), consumed))
    }
}

impl Encode for NamespaceName {
    fn encoded_len(&self) -> usize {
        4 + self.as_str().len()
    }

    fn encode(&self, out: &mut Vec<u8>) {
        encode_bytes(out, self.as_str().as_bytes());
    }
}

impl Decode for NamespaceName {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (bytes, consumed) = decode_byte_slice(input)?;
        let text = core::str::from_utf8(bytes).map_err(CodecError::InvalidUtf8)?;
        let name = Self::new(text).map_err(CodecError::InvalidNamespaceName)?;
        Ok((name, consumed))
    }
}

impl Encode for RequestIdentity {
    fn encoded_len(&self) -> usize {
        24
    }

    fn encode(&self, out: &mut Vec<u8>) {
        self.session().as_u128().encode(out);
        self.seq().as_u64().encode(out);
    }
}

impl Decode for RequestIdentity {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (session, first) = u128::decode(input)?;
        let (seq, second) = u64::decode(&input[first..])?;
        Ok((
            Self::new(SessionId::from_u128(session), RequestSeq::from_u64(seq)),
            first + second,
        ))
    }
}

impl Encode for TabletAuthority {
    fn encoded_len(&self) -> usize {
        24
    }

    fn encode(&self, out: &mut Vec<u8>) {
        self.tablet().as_u64().encode(out);
        self.epoch().as_u64().encode(out);
        self.guard().as_u64().encode(out);
    }
}

impl Decode for TabletAuthority {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (tablet, first) = u64::decode(input)?;
        let (epoch, second) = u64::decode(&input[first..])?;
        let (guard, third) = u64::decode(&input[first + second..])?;
        Ok((
            Self::new(
                TabletId::from_u64(tablet),
                TabletEpoch::from_u64(epoch),
                WriteGuardGeneration::from_u64(guard),
            ),
            first + second + third,
        ))
    }
}

impl Encode for CommitToken {
    fn encoded_len(&self) -> usize {
        24
    }

    fn encode(&self, out: &mut Vec<u8>) {
        self.tablet().as_u64().encode(out);
        self.epoch().as_u64().encode(out);
        self.position().as_u64().encode(out);
    }
}

impl Decode for CommitToken {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (tablet, first) = u64::decode(input)?;
        let (epoch, second) = u64::decode(&input[first..])?;
        let (position, third) = u64::decode(&input[first + second..])?;
        Ok((
            Self::new(
                TabletId::from_u64(tablet),
                TabletEpoch::from_u64(epoch),
                CommitPosition::from_u64(position),
            ),
            first + second + third,
        ))
    }
}

impl Encode for Expiry {
    fn encoded_len(&self) -> usize {
        match self.as_stamp() {
            None => 1,
            Some(_) => 9,
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self.as_stamp() {
            None => out.push(0),
            Some(when) => {
                out.push(1);
                when.encode(out);
            }
        }
    }
}

impl Decode for Expiry {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (tag, first) = u8::decode(input)?;
        match tag {
            0 => Ok((Self::NEVER, first)),
            1 => {
                let (when, second) = WallTimestamp::decode(&input[first..])?;
                Ok((Self::at(when), first + second))
            }
            other => Err(CodecError::InvalidTag {
                kind: "expiry",
                tag: other,
            }),
        }
    }
}

/// Canonical tag bytes for [`ReadContract`].
const READ_LATEST_TAG: u8 = 0;
/// Canonical tag bytes for [`ReadContract`].
const READ_AT_LEAST_TAG: u8 = 1;
/// Canonical tag bytes for [`ReadContract`].
const READ_BOUNDED_STALE_TAG: u8 = 2;
/// Canonical tag bytes for [`ReadContract`].
const READ_ANY_TAG: u8 = 3;

impl Encode for ReadContract {
    fn encoded_len(&self) -> usize {
        match self {
            Self::Latest | Self::Any => 1,
            Self::AtLeast(_) => 1 + 24,
            Self::BoundedStale { .. } => 1 + 8,
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::Latest => out.push(READ_LATEST_TAG),
            Self::AtLeast(token) => {
                out.push(READ_AT_LEAST_TAG);
                token.encode(out);
            }
            Self::BoundedStale { max_staleness } => {
                out.push(READ_BOUNDED_STALE_TAG);
                duration_micros(*max_staleness).encode(out);
            }
            Self::Any => out.push(READ_ANY_TAG),
        }
    }
}

impl Decode for ReadContract {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (tag, first) = u8::decode(input)?;
        match tag {
            READ_LATEST_TAG => Ok((Self::Latest, first)),
            READ_AT_LEAST_TAG => {
                let (token, second) = CommitToken::decode(&input[first..])?;
                Ok((Self::AtLeast(token), first + second))
            }
            READ_BOUNDED_STALE_TAG => {
                let (micros, second) = u64::decode(&input[first..])?;
                Ok((
                    Self::BoundedStale {
                        max_staleness: Duration::from_micros(micros),
                    },
                    first + second,
                ))
            }
            READ_ANY_TAG => Ok((Self::Any, first)),
            other => Err(CodecError::InvalidTag {
                kind: "read-contract",
                tag: other,
            }),
        }
    }
}

/// Saturating conversion of a [`Duration`] to whole microseconds for the wire.
fn duration_micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Field order and width are the wire contract: a `TabletAuthority` that
    /// encoded its three `u64`s in a different order, or dropped one, would
    /// silently mis-fence on a peer running the same build. `encoded_len` is
    /// included because callers pre-allocate with it.
    #[test]
    fn every_composite_encodes_to_its_declared_width() {
        let authority = TabletAuthority::new(
            TabletId::from_u64(918),
            TabletEpoch::from_u64(381),
            WriteGuardGeneration::from_u64(12),
        );
        let token = CommitToken::new(
            TabletId::from_u64(5),
            TabletEpoch::from_u64(2),
            CommitPosition::from_u64(100),
        );
        let identity =
            RequestIdentity::new(SessionId::from_u128(0x00C0_FFEE), RequestSeq::from_u64(41));
        let dated = Expiry::at(WallTimestamp::from_micros(500));

        // (label, encoded bytes, declared width)
        let cases: [(&str, Vec<u8>, usize); 14] = [
            ("tablet-id", TabletId::from_u64(918).encode_to_vec(), 8),
            ("session-id", SessionId::from_u128(1).encode_to_vec(), 16),
            (
                "partition-hash",
                PartitionHash::from_u128(u128::MAX).encode_to_vec(),
                16,
            ),
            ("chunk-id", ChunkId::ZERO.encode_to_vec(), 32),
            (
                "wall-timestamp",
                WallTimestamp::from_micros(-1).encode_to_vec(),
                8,
            ),
            ("authority", authority.encode_to_vec(), 24),
            ("commit-token", token.encode_to_vec(), 24),
            ("request-identity", identity.encode_to_vec(), 24),
            ("expiry-never", Expiry::NEVER.encode_to_vec(), 1),
            ("expiry-dated", dated.encode_to_vec(), 9),
            ("contract-latest", ReadContract::Latest.encode_to_vec(), 1),
            ("contract-any", ReadContract::Any.encode_to_vec(), 1),
            (
                "contract-at-least",
                ReadContract::AtLeast(token).encode_to_vec(),
                25,
            ),
            (
                "contract-bounded",
                ReadContract::BoundedStale {
                    max_staleness: Duration::from_millis(100),
                }
                .encode_to_vec(),
                9,
            ),
        ];
        for (label, bytes, width) in cases {
            assert_eq!(bytes.len(), width, "{label} changed width");
        }
    }

    /// Every composite must survive a full encode/decode cycle: an identity
    /// that decodes to a *different* value is silent mis-fencing.
    #[test]
    fn every_composite_round_trips() {
        let token = CommitToken::new(
            TabletId::from_u64(5),
            TabletEpoch::INITIAL,
            CommitPosition::FIRST,
        );
        let authority = TabletAuthority::new(
            TabletId::from_u64(918),
            TabletEpoch::from_u64(381),
            WriteGuardGeneration::from_u64(12),
        );
        let identity =
            RequestIdentity::new(SessionId::from_u128(0x00C0_FFEE), RequestSeq::from_u64(41));
        assert_eq!(
            TabletAuthority::decode_exact(&authority.encode_to_vec()).expect("authority"),
            authority
        );
        assert_eq!(
            CommitToken::decode_exact(&token.encode_to_vec()).expect("token"),
            token
        );
        assert_eq!(
            RequestIdentity::decode_exact(&identity.encode_to_vec()).expect("identity"),
            identity
        );
        assert_eq!(
            Expiry::decode_exact(&Expiry::NEVER.encode_to_vec()).expect("never"),
            Expiry::NEVER
        );
        assert_eq!(
            Expiry::decode_exact(&Expiry::at(WallTimestamp::from_micros(500)).encode_to_vec())
                .expect("dated"),
            Expiry::at(WallTimestamp::from_micros(500))
        );
        for contract in [
            ReadContract::Latest,
            ReadContract::AtLeast(token),
            ReadContract::BoundedStale {
                max_staleness: Duration::from_millis(100),
            },
            ReadContract::Any,
        ] {
            assert_eq!(
                ReadContract::decode_exact(&contract.encode_to_vec()).expect("contract"),
                contract
            );
        }
    }

    /// Signed microseconds: a pre-epoch stamp must survive the wire and must
    /// not be confused with the unsigned reading of the same bytes.
    #[test]
    fn wall_timestamps_round_trip_signed() {
        for micros in [
            i64::MIN,
            -1_000_001,
            -1,
            0,
            1,
            500,
            9_000_000_000_000_000,
            i64::MAX,
        ] {
            let stamp = WallTimestamp::from_micros(micros);
            assert_eq!(stamp.encoded_len(), 8);
            assert_eq!(
                WallTimestamp::decode_exact(&stamp.encode_to_vec()).expect("round trip"),
                stamp,
                "micros {micros} must round-trip"
            );
        }
        // `-1` is all ones, observably distinct from an unsigned zero.
        assert_eq!(
            WallTimestamp::from_micros(-1).encode_to_vec(),
            [0xFF; 8].to_vec()
        );
    }

    /// Namespace names cross a security-relevant boundary, so a length prefix
    /// that lies, or a charset violation, must be rejected rather than
    /// normalised into something valid.
    #[test]
    fn namespace_names_reject_a_lying_prefix_and_a_bad_charset() {
        let name = NamespaceName::new("sessions").expect("valid");
        assert_eq!(name.encode_to_vec().len(), 4 + 8);

        let mut short = 8u32.to_le_bytes().to_vec();
        short.extend_from_slice(b"se");
        assert!(NamespaceName::decode(&short).is_err());

        let mut bad = Vec::new();
        encode_bytes(&mut bad, "UPPER".as_bytes());
        assert!(matches!(
            NamespaceName::decode(&bad),
            Err(CodecError::InvalidNamespaceName(_))
        ));
    }

    /// Every tag byte outside the assigned set must be refused: decoding one
    /// as a known value would fabricate a read contract or an expiry.
    #[test]
    fn unknown_tags_and_truncation_are_rejected() {
        assert_eq!(
            ReadContract::decode(&[0xFF]),
            Err(CodecError::InvalidTag {
                kind: "read-contract",
                tag: 0xFF
            })
        );
        assert_eq!(
            Expiry::decode(&[0xFF]),
            Err(CodecError::InvalidTag {
                kind: "expiry",
                tag: 0xFF
            })
        );
        let authority = TabletAuthority::new(
            TabletId::from_u64(1),
            TabletEpoch::INITIAL,
            WriteGuardGeneration::INITIAL,
        );
        assert!(matches!(
            TabletAuthority::decode(&authority.encode_to_vec()[..10]),
            Err(CodecError::Truncated { .. })
        ));
    }
}
