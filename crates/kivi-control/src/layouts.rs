//! Cluster-wide redundancy layout authority: map keys and opaque records.
//!
//! The control Raft group is the authority for which redundancy generation
//! is published per asset. One published generation exists per asset; stale
//! workers are fenced at [`ControlState`](crate::state::ControlState)
//! apply time; restart discovers state via control log replay plus
//! snapshot; coordinator failure survives via Raft. No new consensus is
//! introduced: publication is an ordinary [`ControlMutation`](crate::mutation::ControlMutation).
//!
//! The control plane stores layout bytes **opaquely**: it never parses
//! [`RedundancyLayout`](kivi_redundancy::RedundancyLayout) bytes. The parsed
//! asset fields ride alongside as a [`LayoutKey`] for map indexing, and the
//! fencing generation rides both in the mutation body and in the
//! [`LayoutRecord`] so apply never decodes the opaque payload.

use kivi_redundancy::{AssetId, AssetKind};

/// Encoded [`LayoutKey`] length: kind `u8`, domain `u64` LE, hash `[u8; 32]`.
pub const LAYOUT_KEY_LEN: usize = 1 + 8 + 32;

/// Map key for one asset's published redundancy layout.
///
/// Parsed asset identity riding alongside the opaque layout bytes: the
/// control plane indexes on this without ever parsing the layout payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LayoutKey {
    /// Asset family discriminant ([`AssetKind`] frozen values).
    pub kind: u8,
    /// Raw security-domain value.
    pub domain: u64,
    /// Content hash minted by the owning fabric (never the zero sentinel).
    pub hash: [u8; 32],
}

impl LayoutKey {
    /// Derives the map key from an asset identity.
    #[must_use]
    pub const fn from_asset(asset: &AssetId) -> Self {
        Self {
            kind: asset.kind.as_u8(),
            domain: asset.domain,
            hash: asset.hash,
        }
    }

    /// Rebuilds the asset identity.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError`] on unknown kind discriminants or the zero
    /// hash sentinel.
    pub fn to_asset(&self) -> Result<AssetId, LayoutError> {
        let kind =
            AssetKind::from_u8(self.kind).map_err(|_| LayoutError::BadKind { found: self.kind })?;
        AssetId::new(kind, self.domain, self.hash).map_err(|_| LayoutError::ZeroHash)
    }

    /// Encodes the canonical bytes (`kind u8, domain u64 LE, hash[32]`).
    #[must_use]
    pub fn encode_to_vec(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(LAYOUT_KEY_LEN);
        out.push(self.kind);
        out.extend_from_slice(&self.domain.to_le_bytes());
        out.extend_from_slice(&self.hash);
        out
    }

    /// Decodes exactly one key.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError`] on truncation, trailing bytes, unknown kind
    /// discriminants, or the zero hash sentinel.
    pub fn decode_exact(input: &[u8]) -> Result<Self, LayoutError> {
        use LayoutError as Fault;
        if input.len() < LAYOUT_KEY_LEN {
            return Err(Fault::Truncated);
        }
        if input.len() > LAYOUT_KEY_LEN {
            return Err(Fault::TrailingBytes);
        }
        let kind = input[0];
        if AssetKind::from_u8(kind).is_err() {
            return Err(Fault::BadKind { found: kind });
        }
        let domain = u64::from_le_bytes(input[1..9].try_into().unwrap_or([0; 8]));
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&input[9..41]);
        if hash == [0; 32] {
            return Err(Fault::ZeroHash);
        }
        Ok(Self { kind, domain, hash })
    }
}

/// One asset's published redundancy record: fencing generation plus opaque
/// canonical [`RedundancyLayout`](kivi_redundancy::RedundancyLayout) bytes
/// (exactly what `RedundancyLayout::encode` minted).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutRecord {
    /// Control-plane generation that published this layout (fencing; never
    /// zero).
    pub control_generation: u64,
    /// Opaque canonical layout bytes; the control plane never parses these.
    pub layout: Vec<u8>,
}

impl LayoutRecord {
    /// Builds a record from its generation and opaque layout bytes.
    #[must_use]
    pub fn new(control_generation: u64, layout: Vec<u8>) -> Self {
        Self {
            control_generation,
            layout,
        }
    }

    /// Encodes the canonical bytes (`generation u64 LE, len u32 LE,
    /// layout[...]`).
    #[must_use]
    pub fn encode_to_vec(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + 4 + self.layout.len());
        out.extend_from_slice(&self.control_generation.to_le_bytes());
        let len = u32::try_from(self.layout.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&self.layout);
        out
    }

    /// Decodes exactly one record.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError`] on truncation, declared lengths that exceed
    /// the wire cap or the input, trailing bytes, the zero generation
    /// sentinel, or empty layout bytes.
    pub fn decode_exact(input: &[u8]) -> Result<Self, LayoutError> {
        use LayoutError as Fault;
        if input.len() < 8 + 4 {
            return Err(Fault::Truncated);
        }
        let control_generation = u64::from_le_bytes(input[..8].try_into().unwrap_or([0; 8]));
        if control_generation == 0 {
            return Err(Fault::ZeroGeneration);
        }
        let len = u32::from_le_bytes(input[8..12].try_into().unwrap_or([0; 4])) as usize;
        if len > kivi_redundancy::MAX_LAYOUT_WIRE_BYTES {
            return Err(Fault::Oversized {
                len,
                available: input.len(),
            });
        }
        if input.len() < 12 + len {
            return Err(Fault::Oversized {
                len,
                available: input.len(),
            });
        }
        if input.len() > 12 + len {
            return Err(Fault::TrailingBytes);
        }
        if len == 0 {
            return Err(Fault::EmptyLayout);
        }
        Ok(Self {
            control_generation,
            layout: input[12..12 + len].to_vec(),
        })
    }
}

/// Why a layout key or record was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum LayoutError {
    /// Truncated input where framing promised bytes.
    #[error("truncated layout record")]
    Truncated,
    /// A length prefix exceeds the wire cap or the available input.
    #[error("declared length {len} exceeds limit or input {available}")]
    Oversized {
        /// Declared length.
        len: usize,
        /// Bytes actually available.
        available: usize,
    },
    /// Trailing bytes after the framed record.
    #[error("trailing bytes in layout record")]
    TrailingBytes,
    /// Unknown asset-kind discriminant in canonical bytes.
    #[error("unknown asset kind {found}")]
    BadKind {
        /// Observed discriminant.
        found: u8,
    },
    /// Asset hash is the all-zero sentinel.
    #[error("asset hash is the zero sentinel")]
    ZeroHash,
    /// Control generation 0 is the invalid sentinel.
    #[error("control generation 0 is the invalid sentinel")]
    ZeroGeneration,
    /// Layout payload is empty (never a valid encoded layout).
    #[error("empty layout bytes")]
    EmptyLayout,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset() -> AssetId {
        AssetId::new(AssetKind::Chunk, 7, [11; 32]).expect("asset")
    }

    #[test]
    fn key_round_trips_through_asset() {
        let key = LayoutKey::from_asset(&asset());
        assert_eq!(key.encode_to_vec().len(), LAYOUT_KEY_LEN);
        let back = LayoutKey::decode_exact(&key.encode_to_vec()).expect("decodes");
        assert_eq!(back, key);
        assert_eq!(back.to_asset().expect("asset"), asset());
    }

    #[test]
    fn key_rejects_zero_hash_bad_kind_and_framing() {
        let mut bad_hash = LayoutKey::from_asset(&asset()).encode_to_vec();
        bad_hash[9..41].copy_from_slice(&[0; 32]);
        assert!(matches!(
            LayoutKey::decode_exact(&bad_hash),
            Err(LayoutError::ZeroHash)
        ));
        let mut bad_kind = LayoutKey::from_asset(&asset()).encode_to_vec();
        bad_kind[0] = 9;
        assert!(matches!(
            LayoutKey::decode_exact(&bad_kind),
            Err(LayoutError::BadKind { .. })
        ));
        let good = LayoutKey::from_asset(&asset()).encode_to_vec();
        assert!(matches!(
            LayoutKey::decode_exact(&good[..good.len() - 1]),
            Err(LayoutError::Truncated)
        ));
        let mut trailed = good.clone();
        trailed.push(0xFF);
        assert!(matches!(
            LayoutKey::decode_exact(&trailed),
            Err(LayoutError::TrailingBytes)
        ));
    }

    #[test]
    fn record_round_trips_and_rejects_sentinels() {
        let record = LayoutRecord::new(3, vec![7u8; 16]);
        let back = LayoutRecord::decode_exact(&record.encode_to_vec()).expect("decodes");
        assert_eq!(back, record);
        let zero_gen = LayoutRecord::new(0, vec![7u8; 16]);
        assert!(matches!(
            LayoutRecord::decode_exact(&zero_gen.encode_to_vec()),
            Err(LayoutError::ZeroGeneration)
        ));
        let empty = LayoutRecord::new(1, Vec::new());
        assert!(matches!(
            LayoutRecord::decode_exact(&empty.encode_to_vec()),
            Err(LayoutError::EmptyLayout)
        ));
        let good = record.encode_to_vec();
        assert!(matches!(
            LayoutRecord::decode_exact(&good[..4]),
            Err(LayoutError::Truncated)
        ));
        let mut trailed = good.clone();
        trailed.push(0xFF);
        assert!(matches!(
            LayoutRecord::decode_exact(&trailed),
            Err(LayoutError::TrailingBytes)
        ));
        let mut oversized = good;
        oversized[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            LayoutRecord::decode_exact(&oversized),
            Err(LayoutError::Oversized { .. })
        ));
    }
}
