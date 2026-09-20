//! Memory Fabric failure modes (RFC §251 hardware rule).
//!
//! A failed optimization degrades performance or cost; it never invents,
//! loses, or corrupts logical state. These errors therefore read as *retry,
//! fall back, or repair me*, never as silent data loss.

use crate::handle::ObjectHandle;

/// Every way the Memory Fabric can refuse an optimization step.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum MemoryError {
    /// A stale generational handle was used after relocation, compaction,
    /// or reclamation. The caller must re-resolve the object root; the
    /// bytes it wanted are still owned by the fabric under the new handle.
    #[error("stale handle {handle:?}: expected generation {expected}, found {found}")]
    StaleHandle {
        /// The handle the caller presented.
        handle: ObjectHandle,
        /// Generation the caller expected.
        expected: u32,
        /// Generation currently stored in the slot.
        found: u32,
    },
    /// No valid representation exists for a live object. This is a bug or
    /// an incomplete transition, never an eviction outcome: eviction always
    /// proves a reconstruction source first.
    #[error("object {object} has no valid representation")]
    NoRepresentation {
        /// Internal object index.
        object: u64,
    },
    /// A representation failed verification (CRC, length, or hash
    /// mismatch). The representation is quarantined; another valid
    /// representation or reconstruction source still exists by invariant.
    #[error("representation of object {object} failed verification: {detail}")]
    CorruptRepresentation {
        /// Internal object index.
        object: u64,
        /// Which check failed.
        detail: String,
    },
    /// The backing provider failed (capacity, device error, injected
    /// fault). Fall back to another representation; do not serve the
    /// failed bytes.
    #[error("provider {provider} failed for object {object}: {detail}")]
    ProviderFailed {
        /// Provider identifier.
        provider: &'static str,
        /// Internal object index.
        object: u64,
        /// What failed.
        detail: String,
    },
    /// A bounded queue is full. Fast backpressure like every other Kivi
    /// queue: answer `Overloaded` with a backoff, never wait inline.
    #[error("memory fabric queue {queue} saturated")]
    Overloaded {
        /// Which queue refused.
        queue: &'static str,
    },
    /// The movement budget for this tick is exhausted. Retry after the
    /// next budget window; foreground latency is more important than
    /// finishing migration now.
    #[error("movement budget exhausted: {detail}")]
    BudgetExhausted {
        /// What ran out.
        detail: &'static str,
    },
    /// A transition was cancelled (object mutated, tablet moved, or queue
    /// shed load). No state changed; the old representation is still valid.
    #[error("transition for object {object} cancelled: {reason}")]
    Cancelled {
        /// Internal object index.
        object: u64,
        /// Why it was cancelled.
        reason: &'static str,
    },
    /// A mutation arrived while a new representation was being built. The
    /// in-flight build is discarded; the authoritative state already
    /// reflects the mutation.
    #[error("object {object} mutated during transition (build v{build} vs current v{current})")]
    MutatedDuringBuild {
        /// Internal object index.
        object: u64,
        /// Version the build started from.
        build: u64,
        /// Current version.
        current: u64,
    },
    /// A versioned durable/physical format this build does not implement.
    /// Old binaries fail loudly on new formats instead of misreading them.
    #[error("unsupported memory format: {detail}")]
    Unsupported {
        /// What was encountered.
        detail: String,
    },
    /// A declared length exceeds the allocation guard. Checked before any
    /// allocation, so hostile bytes cannot force huge allocations.
    #[error("memory length {len} exceeds bound {max}: {context}")]
    TooLarge {
        /// Declared length.
        len: u64,
        /// Enforced bound.
        max: u64,
        /// What was being decoded.
        context: &'static str,
    },
    /// Filesystem I/O failure with the operation, path, and OS detail.
    #[error("memory I/O during {op} on {}: {message}", path.display())]
    Io {
        /// What was being attempted.
        op: &'static str,
        /// File or directory involved.
        path: std::path::PathBuf,
        /// OS error rendered at the failure site.
        message: String,
    },
    /// Invalid caller use (empty bytes where nonempty required, unknown
    /// object, ...). A caller bug, never storage damage.
    #[error("invalid memory request: {detail}")]
    Invalid {
        /// What was wrong.
        detail: String,
    },
}

impl MemoryError {
    /// Builds the [`MemoryError::Io`] variant from an [`std::io::Error`]
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
