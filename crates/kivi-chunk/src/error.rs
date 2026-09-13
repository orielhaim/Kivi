//! Chunk fabric failure modes.
//!
//! The taxonomy keeps recovery-relevant cases distinct (RFC §251 payload
//! rule, task §43): a missing or corrupt chunk/manfiest must read as
//! *repair me later*, never as generic `Internal`. Anything else is a
//! local bug, a bound, or an I/O failure.

use kivi_types::{ChunkId, ManifestId};

/// Every way the immutable chunk substrate can refuse.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ChunkError {
    /// Filesystem I/O failure with the operation, path, and OS detail.
    #[error("chunk I/O during {op} on {}: {message}", path.display())]
    Io {
        /// What was being attempted (`open pack`, `sync active pack`, ...).
        op: &'static str,
        /// File or directory involved.
        path: std::path::PathBuf,
        /// OS error rendered at the failure site (never reconstructed).
        message: String,
    },
    /// A required chunk has no valid representation anywhere locally.
    /// Later multi-source recovery repairs this from a replica, archive,
    /// or coded information; locally it is corruption, never absence.
    #[error("required chunk {id} has no valid local representation")]
    MissingChunk {
        /// The chunk that cannot be served.
        id: ChunkId,
    },
    /// A required manifest has no valid representation anywhere locally.
    #[error("required manifest {id} has no valid local representation")]
    MissingManifest {
        /// The manifest that cannot be served.
        id: ManifestId,
    },
    /// A chunk record failed verification (framing, CRC, length, or
    /// content-hash mismatch). Quarantine the representation; do not serve
    /// its bytes.
    #[error("chunk {id} failed verification: {detail}")]
    CorruptChunk {
        /// The chunk that failed (or `ZERO` when the id itself was unreadable).
        id: ChunkId,
        /// Which check failed.
        detail: String,
    },
    /// A manifest record failed verification or structural validation
    /// (length-sum mismatch, illegal chunking, bad hash link).
    #[error("manifest {id} failed verification: {detail}")]
    CorruptManifest {
        /// The manifest that failed (or `ZERO` when unreadable).
        id: ManifestId,
        /// Which check failed.
        detail: String,
    },
    /// A versioned format or codec this build does not implement. Never
    /// guessed past: old binaries fail loudly on new formats instead of
    /// misreading them.
    #[error("unsupported chunk format: {detail}")]
    Unsupported {
        /// What was encountered (codec id, format version, ...).
        detail: String,
    },
    /// A declared length exceeds the allocation guard. Checked before any
    /// allocation, so hostile pack bytes cannot force huge allocations.
    #[error("chunk length {len} exceeds bound {max}: {context}")]
    TooLarge {
        /// Declared length.
        len: u64,
        /// Enforced bound.
        max: u64,
        /// What was being decoded.
        context: &'static str,
    },
    /// Invalid engine-level use (empty chunk list with nonzero length,
    /// misaligned splice range, ...). A caller bug, never storage damage.
    #[error("invalid chunk request: {detail}")]
    Invalid {
        /// What was wrong.
        detail: String,
    },
    /// The lane's bounded job queue is full. Fast backpressure, never a
    /// wait: the caller answers `Overloaded` (ideally with a backoff) like
    /// every other saturated Kivi queue.
    #[error("chunk lane saturated")]
    Overloaded,
}

impl ChunkError {
    /// Builds the [`ChunkError::Io`] variant from an [`std::io::Error`]
    /// at the failure site.
    #[must_use]
    pub fn io(op: &'static str, path: &std::path::Path, error: &std::io::Error) -> Self {
        Self::Io {
            op,
            path: path.to_owned(),
            message: error.to_string(),
        }
    }
}
