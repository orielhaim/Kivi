//! Integrity primitives: CRC32C checksums and BLAKE3 content identity.
//!
//! Thin, byte-exact wrappers over the borrowed mechanism crates (RFC §209):
//! `crc32c` for cheap accidental-corruption detection, `blake3` for content
//! identity. The project owns the constructions built on top — in particular
//! [`chunk_id`], whose exact byte layout is part of the durable contract.

use kivi_types::{ChunkId, SecurityDomainId, hash::CHUNK_DOMAIN_TAG};

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

/// Derives the content address of an immutable chunk (RFC §29).
///
/// Byte-exact construction (durable — must never change for `V1`):
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksums_and_hashes_are_deterministic() {
        let data = b"tablet-918 band-C";
        assert_eq!(crc32c_checksum(data), crc32c_checksum(data));
        assert_eq!(blake3_256(data), blake3_256(data));
    }

    #[test]
    fn avalanche_single_bit_changes_everything() {
        let (a, b) = (b"chunk-payload-A", b"chunk-payload-B");
        assert_ne!(crc32c_checksum(a), crc32c_checksum(b));
        assert_ne!(blake3_256(a), blake3_256(b));
    }

    #[test]
    fn chunk_ids_are_domain_separated() {
        let bytes = b"same canonical bytes";
        let left = chunk_id(SecurityDomainId::from_u64(1), bytes);
        let right = chunk_id(SecurityDomainId::from_u64(2), bytes);
        assert_ne!(left, right);
        assert!(left.is_valid() && right.is_valid());
        // The domain tag participates: a bare hash of the bytes differs.
        assert_ne!(left, ChunkId::from_bytes(blake3_256(bytes)));
        // Same domain and bytes are stable.
        assert_eq!(left, chunk_id(SecurityDomainId::from_u64(1), bytes));
        // Different bytes differ within one domain.
        assert_ne!(
            left,
            chunk_id(SecurityDomainId::from_u64(1), b"other bytes")
        );
    }

    #[test]
    fn empty_inputs_have_defined_identities() {
        let empty_chunk = chunk_id(SecurityDomainId::from_u64(9), &[]);
        assert!(empty_chunk.is_valid());
        assert_eq!(crc32c_checksum(&[]), crc32c_checksum(&[]));
    }
}
