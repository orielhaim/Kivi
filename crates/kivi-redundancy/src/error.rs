//! Redundancy Fabric errors: typed, fail-closed, never silent data loss.

/// Every failure mode the fabric distinguishes.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum RedundancyError {
    /// Asset identity is the zero sentinel or otherwise malformed.
    #[error("invalid asset: {detail}")]
    InvalidAsset {
        /// Human-readable cause.
        detail: String,
    },
    /// Redundancy intent violates hard safety policy.
    #[error("invalid redundancy intent: {detail}")]
    InvalidIntent {
        /// Human-readable cause.
        detail: String,
    },
    /// Scheme parameters violate bounded limits.
    #[error("invalid scheme parameters: {detail}")]
    InvalidParams {
        /// Human-readable cause.
        detail: String,
    },
    /// Placement cannot satisfy failure-independence requirements.
    #[error("placement unsatisfiable: {detail}")]
    Placement {
        /// Human-readable cause.
        detail: String,
    },
    /// Fragment is addressed but no representation exists.
    #[error("fragment missing: asset {asset} generation {generation} index {index}")]
    MissingFragment {
        /// Asset addressed.
        asset: String,
        /// Generation addressed.
        generation: u64,
        /// Fragment index addressed.
        index: u32,
    },
    /// Fragment bytes could not be read (I/O, not corruption).
    #[error("fragment unreadable: {detail}")]
    Unreadable {
        /// Human-readable cause.
        detail: String,
    },
    /// Fragment failed integrity verification.
    #[error("fragment corrupt: {detail}")]
    CorruptFragment {
        /// Human-readable cause.
        detail: String,
    },
    /// Reconstructed bytes failed content-address verification.
    #[error("reconstruction failed verification: {detail}")]
    VerificationFailed {
        /// Human-readable cause.
        detail: String,
    },
    /// Not enough healthy information to reconstruct.
    #[error("unrecoverable: {detail}")]
    Unrecoverable {
        /// Human-readable cause.
        detail: String,
    },
    /// Bounded resources exhausted; caller retries later.
    #[error("redundancy overloaded: {detail}")]
    Overloaded {
        /// Human-readable cause.
        detail: String,
    },
    /// Stale generation work was fenced and rejected.
    #[error("stale generation {generation} rejected (current {current})")]
    StaleGeneration {
        /// Generation carried by the stale work.
        generation: u64,
        /// Current published generation.
        current: u64,
    },
    /// Stale incarnation work was fenced and rejected: the target node
    /// restarted (or the caller holds a superseded epoch) and the request
    /// carried the wrong incarnation. `target == 0` is the bootstrap probe
    /// and is always allowed; any other mismatch fails closed here so a
    /// caller can refresh and retry exactly once.
    #[error("stale incarnation {current} rejected (expected {expected})")]
    StaleIncarnation {
        /// Incarnation the receiver currently holds.
        expected: u64,
        /// Incarnation the request carried.
        current: u64,
    },
    /// A remote fragment holder could not be reached (transport failure,
    /// corrupt bytes on the holder, or missing representation treated as a
    /// dead end for this attempt — the caller tries alternates).
    #[error("fragment holder unreachable: {detail}")]
    Unreachable {
        /// Human-readable cause.
        detail: String,
    },
    /// A remote fragment operation exceeded its deadline; the caller retries
    /// or falls back to alternates (never served partial bytes).
    #[error("fragment operation timed out: {op}")]
    Timeout {
        /// Operation that timed out (`put`, `get`, `has`, `drop`, `layout`).
        op: String,
    },
    /// Durable layout metadata is malformed or unsupported.
    #[error("layout metadata invalid: {detail}")]
    BadLayout {
        /// Human-readable cause.
        detail: String,
    },
    /// Filesystem/storage failure.
    #[error("redundancy I/O failed: {detail}")]
    Io {
        /// Human-readable cause.
        detail: String,
    },
    /// Chunk-layer failure (integration adapter).
    #[error("chunk backend failed: {detail}")]
    Chunk {
        /// Human-readable cause.
        detail: String,
    },
    /// Checkpoint-layer failure (integration adapter).
    #[error("checkpoint backend failed: {detail}")]
    Checkpoint {
        /// Human-readable cause.
        detail: String,
    },
}

impl RedundancyError {
    /// Convenience for invalid-asset failures.
    #[must_use]
    pub fn invalid_asset(detail: impl Into<String>) -> Self {
        Self::InvalidAsset {
            detail: detail.into(),
        }
    }

    /// Convenience for invalid-intent failures.
    #[must_use]
    pub fn invalid_intent(detail: impl Into<String>) -> Self {
        Self::InvalidIntent {
            detail: detail.into(),
        }
    }

    /// Convenience for invalid-params failures.
    #[must_use]
    pub fn invalid_params(detail: impl Into<String>) -> Self {
        Self::InvalidParams {
            detail: detail.into(),
        }
    }

    /// Convenience for I/O failures with path context.
    #[must_use]
    pub fn io(context: &str, path: &std::path::Path, error: &std::io::Error) -> Self {
        Self::Io {
            detail: format!("{context} {}: {error}", path.display()),
        }
    }

    /// Whether this error means the asset cannot be recovered right now.
    #[must_use]
    pub const fn is_unrecoverable(&self) -> bool {
        matches!(self, Self::Unrecoverable { .. })
    }

    /// Whether the caller should retry later (bounded overload, not loss).
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self, Self::Overloaded { .. } | Self::Unreadable { .. })
    }
}
