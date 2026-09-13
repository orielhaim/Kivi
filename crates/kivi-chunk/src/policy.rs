//! Chunking policy: deterministic fixed-size logical chunking.
//!
//! Chunk boundaries are a pure function of the logical byte length (RFC
//! §27): they never depend on network frames, upload packet sizes, or
//! timing. The same value under the same [`Chunking`] always yields the
//! same chunk sequence, which is what makes cross-writer dedup and
//! manifest reuse sound.
//!
//! The inline/chunk representation threshold lives here as the versioned
//! default, but the *decision* belongs to the engine: crossing it changes
//! physical representation only, never logical `Bytes` semantics.

use crate::error::ChunkError;

/// Chunking format version 1: contiguous fixed-size pieces, final piece is
/// the remainder. Pinned: manifests record this version, so a future
/// content-defined chunker mints `V2` instead of redefining `V1`.
pub const CHUNKING_V1: u16 = 1;

/// Default logical chunk size: 1 MiB. Large enough to amortize per-record
/// framing and pack syncs, small enough to bound partial-update and repair
/// amplification.
pub const DEFAULT_CHUNK_SIZE: usize = 1_048_576;

/// Default inline/chunk representation threshold: 256 KiB. Values at or
/// below it stay single-resident `Inline` roots; larger values become
/// `Chunked` roots. Physical only: `GET` cannot tell the difference.
/// Chosen by the threshold benchmark (see the stage report); overrideable
/// per engine configuration, recorded nowhere durable (the manifest
/// carries its own chunk size, so the threshold leaves no trace).
pub const DEFAULT_INLINE_THRESHOLD: u64 = 262_144;

/// Hard cap on entries per manifest. Bounds manifest bodies (~40 bytes per
/// entry, so ~2.6 MiB worst case) and every allocation decoded from one.
pub const MAX_CHUNKS_PER_MANIFEST: usize = 65_535;

/// Hard cap on one record's stored body. A well-formed chunk body never
/// exceeds the chunk size; manifests never approach this. Anything larger
/// is corruption, rejected before allocation.
pub const MAX_STORED_BODY: u64 = 8 * 1024 * 1024;

/// Largest value the legacy (non-streaming) `GET` resolves inline: 64 MiB,
/// matching the protocol frame ceiling philosophy. Larger chunked values
/// require the streaming read path, which never builds one giant response
/// frame — or giant resident allocation — for anyone.
pub const MAX_LEGACY_VALUE_BYTES: u64 = 64 * 1024 * 1024;

/// Versioned chunking parameters. Stored in every manifest, so old data
/// stays readable when the defaults evolve: the policy that *wrote* a
/// manifest travels with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Chunking {
    /// Chunking algorithm version ([`CHUNKING_V1`]).
    pub version: u16,
    /// Logical bytes per full chunk.
    pub chunk_size: usize,
}

impl Chunking {
    /// Production chunking: V1 at 1 MiB.
    pub const DEFAULT: Self = Self {
        version: CHUNKING_V1,
        chunk_size: DEFAULT_CHUNK_SIZE,
    };

    /// Validates that this build can chunk and de-chunk under these
    /// parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Unsupported`] for unknown versions and
    /// [`ChunkError::Invalid`] for a zero chunk size.
    pub fn validate(self) -> Result<(), ChunkError> {
        if self.version != CHUNKING_V1 {
            return Err(ChunkError::Unsupported {
                detail: "unknown chunking version".to_owned(),
            });
        }
        if self.chunk_size == 0 {
            return Err(ChunkError::Invalid {
                detail: "chunk size must be nonzero".to_owned(),
            });
        }
        Ok(())
    }

    /// Splits a logical length into per-chunk lengths: as many full
    /// `chunk_size` pieces as fit, then the nonzero remainder. Empty input
    /// yields no chunks (a zero-length value is an empty manifest).
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::TooLarge`] when the piece count exceeds the
    /// address space (only reachable for absurd lengths on narrow
    /// targets — staged values always fit memory first).
    pub fn split_lengths(self, total_len: u64) -> Result<Vec<u64>, ChunkError> {
        let size = self.chunk_size as u64;
        if size == 0 || total_len == 0 {
            return Ok(Vec::new());
        }
        let full = total_len / size;
        let rest = total_len % size;
        let count =
            usize::try_from(full + u64::from(rest > 0)).map_err(|_| ChunkError::TooLarge {
                len: full,
                max: u64::MAX,
                context: "chunk count",
            })?;
        let mut out = Vec::with_capacity(count);
        out.extend(std::iter::repeat_n(size, count - usize::from(rest > 0)));
        if rest > 0 {
            out.push(rest);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_lengths_follow_the_spec_example() {
        // 2_400_000 bytes at 1 MiB chunks: 1 MiB, 1 MiB, 302_848 rest.
        let lens = Chunking::DEFAULT.split_lengths(2_400_000).expect("splits");
        assert_eq!(
            lens,
            vec![
                DEFAULT_CHUNK_SIZE as u64,
                DEFAULT_CHUNK_SIZE as u64,
                302_848
            ]
        );
    }

    #[test]
    fn exact_multiples_have_no_remainder() {
        let lens = Chunking::DEFAULT
            .split_lengths(DEFAULT_CHUNK_SIZE as u64 * 2)
            .expect("splits");
        assert_eq!(lens, vec![DEFAULT_CHUNK_SIZE as u64; 2]);
    }

    #[test]
    fn empty_splits_to_no_chunks() {
        assert!(
            Chunking::DEFAULT
                .split_lengths(0)
                .expect("splits")
                .is_empty()
        );
    }

    #[test]
    fn validation_rejects_unknown_versions_and_zero_sizes() {
        assert!(Chunking::DEFAULT.validate().is_ok());
        assert!(
            Chunking {
                version: 0x7F,
                ..Chunking::DEFAULT
            }
            .validate()
            .is_err()
        );
        assert!(
            Chunking {
                chunk_size: 0,
                ..Chunking::DEFAULT
            }
            .validate()
            .is_err()
        );
    }
}
