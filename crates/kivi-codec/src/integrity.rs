//! Integrity primitives: CRC32C checksums and BLAKE3 content identity.
//!
//! Thin, byte-exact wrappers over the borrowed mechanism crates:
//! `crc32c` for cheap accidental-corruption detection, `blake3` for content
//! identity. The project owns the constructions built on top - in particular
//! [`chunk_id`], whose exact byte layout is part of the durable contract.

use kivi_types::{
    ChunkId, ManifestId, SecurityDomainId, hash::CHUNK_DOMAIN_TAG, hash::MANIFEST_DOMAIN_TAG,
};

/// CRC32C (Castagnoli) checksum of `data`, hardware-accelerated when
/// available with a software fallback.
#[must_use]
pub fn crc32c_checksum(data: &[u8]) -> u32 {
    crc32c::crc32c(data)
}

/// BLAKE3-256 digest of `data`.
#[must_use]
pub fn blake3_256(data: &[u8]) -> [u8; 32] {
    blake3::hash(data).into()
}

/// Derives the content address of an immutable chunk.
///
/// Byte-exact construction (durable - must never change for `V1`):
///
/// ```text
/// ChunkId = BLAKE3(CHUNK_DOMAIN_TAG || le64(security_domain) || chunk_bytes)
/// ```
///
/// where `CHUNK_DOMAIN_TAG` is [`CHUNK_DOMAIN_TAG`]. Equal bytes under
/// different security domains therefore address different chunks, which is
/// what constrains deduplication to one domain.
#[must_use]
pub fn chunk_id(domain: SecurityDomainId, canonical_bytes: &[u8]) -> ChunkId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CHUNK_DOMAIN_TAG);
    hasher.update(&domain.as_u64().to_le_bytes());
    hasher.update(canonical_bytes);
    ChunkId::from_bytes(hasher.finalize().into())
}

/// Derives the content address of an immutable chunk manifest.
///
/// Byte-exact construction (durable - must never change for `V1`):
///
/// ```text
/// ManifestId = BLAKE3(MANIFEST_DOMAIN_TAG || le64(security_domain) || canonical_manifest_bytes)
/// ```
///
/// The disjoint tag is what keeps manifest addresses from ever aliasing
/// chunk addresses: [`chunk_id`] and [`manifest_id`] agree on neither
/// inputs nor outputs for the same payload.
#[must_use]
pub fn manifest_id(domain: SecurityDomainId, canonical_bytes: &[u8]) -> ManifestId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(MANIFEST_DOMAIN_TAG);
    hasher.update(&domain.as_u64().to_le_bytes());
    hasher.update(canonical_bytes);
    ManifestId::from_bytes(hasher.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The domain tag is a security boundary: equal bytes under different
    /// security domains must address different chunks, or deduplication would
    /// cross the domain it is constrained to. The same holds for the disjoint
    /// chunk/manifest tags, which keep two address spaces from ever aliasing.
    #[test]
    fn content_addresses_are_domain_and_space_separated() {
        let bytes = b"same canonical bytes";
        let one = SecurityDomainId::from_u64(1);
        let two = SecurityDomainId::from_u64(2);

        let chunk_one = chunk_id(one, bytes);
        let chunk_two = chunk_id(two, bytes);
        assert!(chunk_one.is_valid());
        assert_ne!(chunk_one, chunk_two, "equal bytes aliased across domains");
        // The tag participates: a bare hash of the bytes is a different address.
        assert_ne!(chunk_one, ChunkId::from_bytes(blake3_256(bytes)));
        // Different bytes differ within one domain.
        assert_ne!(chunk_one, chunk_id(one, b"other bytes"));

        let manifest_one = manifest_id(one, bytes);
        assert!(manifest_one.is_valid());
        assert_ne!(manifest_one.as_bytes(), chunk_one.as_bytes());
        assert_ne!(manifest_one, manifest_id(two, bytes));
    }
}
