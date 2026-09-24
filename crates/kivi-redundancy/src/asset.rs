//! Immutable information assets: the unit of redundancy protection.
//!
//! An [`InformationAsset`] names *what information this is* independently of
//! how it is physically protected. Its [`AssetId`] never depends on replica
//! count, erasure-code parameters, node placement, physical offsets, or
//! current repair state: the same logical bytes keep the same id across
//! replication, `RS(k,m)` layouts, scheme transitions, and repairs.
//!
//! The vocabulary reuses existing immutable identity instead of inventing a
//! parallel object system: [`AssetId`] wraps the canonical content hashes
//! already minted by the chunk fabric (`ChunkId`, `ManifestId`) and the
//! checkpoint fabric (`ArtifactHash` raw digests) behind one [`AssetKind`]
//! tag. No filesystem path or storage offset ever enters the identity.

use kivi_types::{ChunkId, ManifestId, SecurityDomainId};

/// Which immutable family an asset belongs to.
///
/// The discriminant travels durably in fragment metadata and layout files,
/// so assignments are frozen: `0 = Chunk`, `1 = ChunkManifest`,
/// `2 = CheckpointBand`, `3 = CheckpointManifest`, `4 = CheckpointDedup`,
/// `5 = GenericImmutable` (future immutable durability artifacts).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AssetKind {
    /// One content-addressed logical chunk (`ChunkId`).
    Chunk,
    /// One canonical chunk manifest (`ManifestId`).
    ChunkManifest,
    /// One immutable checkpoint band (content hash).
    CheckpointBand,
    /// One checkpoint manifest (content hash).
    CheckpointManifest,
    /// One checkpoint dedup component (content hash).
    CheckpointDedup,
    /// Any other immutable durability artifact with a 32-byte identity.
    GenericImmutable,
}

impl AssetKind {
    /// Wraps a raw discriminant, rejecting unknown values.
    ///
    /// # Errors
    ///
    /// Returns [`crate::RedundancyError::BadLayout`] for unknown discriminants.
    pub fn from_u8(value: u8) -> Result<Self, crate::RedundancyError> {
        match value {
            0 => Ok(Self::Chunk),
            1 => Ok(Self::ChunkManifest),
            2 => Ok(Self::CheckpointBand),
            3 => Ok(Self::CheckpointManifest),
            4 => Ok(Self::CheckpointDedup),
            5 => Ok(Self::GenericImmutable),
            other => Err(crate::RedundancyError::BadLayout {
                detail: format!("unknown asset kind {other}"),
            }),
        }
    }

    /// Returns the frozen discriminant.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::Chunk => 0,
            Self::ChunkManifest => 1,
            Self::CheckpointBand => 2,
            Self::CheckpointManifest => 3,
            Self::CheckpointDedup => 4,
            Self::GenericImmutable => 5,
        }
    }

    /// Human-readable family name for operators.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Chunk => "chunk",
            Self::ChunkManifest => "chunk-manifest",
            Self::CheckpointBand => "checkpoint-band",
            Self::CheckpointManifest => "checkpoint-manifest",
            Self::CheckpointDedup => "checkpoint-dedup",
            Self::GenericImmutable => "generic-immutable",
        }
    }
}

/// Logical identity of one immutable asset.
///
/// Binds the family ([`AssetKind`]), the 32-byte content hash minted by the
/// owning fabric, and the deduplication/verification domain (`u64`; the
/// [`SecurityDomainId`] raw value for chunk families, `0` for checkpoint
/// artifacts whose identity already binds cluster/tablet provenance).
/// Physical facts — replica count, `(k,m)`, placement, offsets, repair
/// state — never participate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AssetId {
    /// Immutable family.
    pub kind: AssetKind,
    /// Raw security-domain value (`0` when the family carries none).
    pub domain: u64,
    /// Content hash minted by the owning fabric.
    pub hash: [u8; 32],
}

impl AssetId {
    /// Builds an asset id from its parts, rejecting the zero sentinel.
    ///
    /// # Errors
    ///
    /// Returns [`crate::RedundancyError::InvalidAsset`] for the all-zero hash.
    pub fn new(
        kind: AssetKind,
        domain: u64,
        hash: [u8; 32],
    ) -> Result<Self, crate::RedundancyError> {
        if hash == [0; 32] {
            return Err(crate::RedundancyError::invalid_asset(
                "asset hash is the zero sentinel",
            ));
        }
        Ok(Self { kind, domain, hash })
    }

    /// Names one chunk's bytes under a security domain.
    #[must_use]
    pub fn chunk(id: ChunkId, domain: SecurityDomainId) -> Self {
        Self {
            kind: AssetKind::Chunk,
            domain: domain.as_u64(),
            hash: *id.as_bytes(),
        }
    }

    /// Names one chunk manifest under a security domain.
    #[must_use]
    pub fn chunk_manifest(id: ManifestId, domain: SecurityDomainId) -> Self {
        Self {
            kind: AssetKind::ChunkManifest,
            domain: domain.as_u64(),
            hash: *id.as_bytes(),
        }
    }

    /// Names one checkpoint artifact by its raw content hash.
    #[must_use]
    pub fn checkpoint(kind: AssetKind, hash: [u8; 32]) -> Self {
        debug_assert!(
            matches!(
                kind,
                AssetKind::CheckpointBand
                    | AssetKind::CheckpointManifest
                    | AssetKind::CheckpointDedup
            ),
            "checkpoint constructor needs a checkpoint kind"
        );
        Self {
            kind,
            domain: 0,
            hash,
        }
    }

    /// Names any other immutable artifact with an explicit 32-byte identity.
    ///
    /// # Errors
    ///
    /// Returns [`crate::RedundancyError::InvalidAsset`] for the zero sentinel.
    pub fn generic(hash: [u8; 32], domain: u64) -> Result<Self, crate::RedundancyError> {
        Self::new(AssetKind::GenericImmutable, domain, hash)
    }

    /// Returns the wrapped chunk id when this names a chunk.
    #[must_use]
    pub fn as_chunk(self) -> Option<ChunkId> {
        if self.kind == AssetKind::Chunk {
            Some(ChunkId::from_bytes(self.hash))
        } else {
            None
        }
    }

    /// Returns the wrapped manifest id when this names a chunk manifest.
    #[must_use]
    pub fn as_manifest(self) -> Option<ManifestId> {
        if self.kind == AssetKind::ChunkManifest {
            Some(ManifestId::from_bytes(self.hash))
        } else {
            None
        }
    }

    /// Returns the security domain when the family carries one.
    #[must_use]
    pub fn security_domain(self) -> Option<SecurityDomainId> {
        match self.kind {
            AssetKind::Chunk | AssetKind::ChunkManifest => {
                Some(SecurityDomainId::from_u64(self.domain))
            }
            AssetKind::CheckpointBand
            | AssetKind::CheckpointManifest
            | AssetKind::CheckpointDedup
            | AssetKind::GenericImmutable => None,
        }
    }

    /// Lowercase hex of the content hash (file names, diagnostics).
    #[must_use]
    pub fn hex(self) -> String {
        use core::fmt::Write as _;
        let mut out = String::with_capacity(64);
        for byte in self.hash {
            let _ = write!(out, "{byte:02x}");
        }
        out
    }
}

impl core::fmt::Display for AssetId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}:{}", self.kind.name(), self.hex())
    }
}

/// One immutable asset requiring protection: its logical identity plus the
/// facts verification needs.
///
/// `logical_len` is the exact byte length of the immutable information;
/// reconstruction verifies both length and content hash before publication.
/// It is *expected size*, never part of the identity: the same [`AssetId`]
/// with a lying length fails verification instead of addressing new data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InformationAsset {
    /// Logical identity (family + content hash + domain).
    pub id: AssetId,
    /// Exact logical bytes.
    pub logical_len: u64,
}

impl InformationAsset {
    /// Builds an asset descriptor.
    #[must_use]
    pub const fn new(id: AssetId, logical_len: u64) -> Self {
        Self { id, logical_len }
    }

    /// What information is this (family + hash, no physical facts).
    #[must_use]
    pub const fn id(self) -> AssetId {
        self.id
    }

    /// Expected logical bytes (verification input, not identity).
    #[must_use]
    pub const fn logical_len(self) -> u64 {
        self.logical_len
    }

    /// Verifies candidate bytes against the logical identity: length
    /// agreement plus content-hash recomputation under the owning fabric's
    /// construction.
    ///
    /// # Errors
    ///
    /// Returns [`crate::RedundancyError::VerificationFailed`] on length
    /// disagreement or hash mismatch. Never panics on untrusted bytes.
    pub fn verify_bytes(&self, bytes: &[u8]) -> Result<(), crate::RedundancyError> {
        if bytes.len() as u64 != self.logical_len {
            return Err(crate::RedundancyError::VerificationFailed {
                detail: format!(
                    "asset {} length {} disagrees with expected {}",
                    self.id,
                    bytes.len(),
                    self.logical_len
                ),
            });
        }
        let recomputed = match self.id.kind {
            AssetKind::Chunk => {
                let domain = SecurityDomainId::from_u64(self.id.domain);
                *kivi_codec::integrity::chunk_id(domain, bytes).as_bytes()
            }
            AssetKind::ChunkManifest => {
                let domain = SecurityDomainId::from_u64(self.id.domain);
                *kivi_codec::integrity::manifest_id(domain, bytes).as_bytes()
            }
            AssetKind::CheckpointBand
            | AssetKind::CheckpointManifest
            | AssetKind::CheckpointDedup
            | AssetKind::GenericImmutable => kivi_codec::integrity::blake3_256(bytes),
        };
        if recomputed != self.id.hash {
            return Err(crate::RedundancyError::VerificationFailed {
                detail: format!("asset {} content hash mismatch", self.id),
            });
        }
        Ok(())
    }

    /// Computes the asset id for raw bytes under the chunk construction.
    #[must_use]
    pub fn chunk_for_bytes(bytes: &[u8], domain: SecurityDomainId) -> Self {
        let id = kivi_codec::integrity::chunk_id(domain, bytes);
        Self {
            id: AssetId::chunk(id, domain),
            logical_len: bytes.len() as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOMAIN: SecurityDomainId = SecurityDomainId::from_u64(7);

    #[test]
    fn chunk_asset_round_trips_identity() {
        let bytes = b"immutable information";
        let asset = InformationAsset::chunk_for_bytes(bytes, DOMAIN);
        assert_eq!(asset.logical_len(), bytes.len() as u64);
        asset.verify_bytes(bytes).expect("verifies");
        assert!(asset.verify_bytes(b"other").is_err());
        let mut lying = *bytes;
        let _ = &mut lying;
        let wrong_len = InformationAsset::new(asset.id, 1);
        assert!(wrong_len.verify_bytes(bytes).is_err());
    }

    #[test]
    fn zero_hash_is_rejected() {
        assert!(AssetId::new(AssetKind::Chunk, 1, [0; 32]).is_err());
        assert!(AssetId::generic([0; 32], 0).is_err());
        assert!(AssetId::generic([9; 32], 3).is_ok());
    }

    #[test]
    fn kind_discriminants_are_frozen() {
        assert_eq!(AssetKind::from_u8(0).expect("chunk"), AssetKind::Chunk);
        assert_eq!(
            AssetKind::from_u8(1).expect("manifest"),
            AssetKind::ChunkManifest
        );
        assert!(AssetKind::from_u8(9).is_err());
        assert_eq!(AssetKind::Chunk.as_u8(), 0);
    }

    #[test]
    fn logical_identity_ignores_physical_facts() {
        let a = InformationAsset::chunk_for_bytes(b"same", DOMAIN);
        let b = InformationAsset::chunk_for_bytes(b"same", DOMAIN);
        assert_eq!(a.id, b.id);
        // Different domain forks the identity (dedup boundary travels).
        let c = InformationAsset::chunk_for_bytes(b"same", SecurityDomainId::from_u64(8));
        assert_ne!(a.id, c.id);
    }
}
