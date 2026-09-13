//! Checkpoint failure modes: loud, typed, never silent.
//!
//! Every variant names what broke and where. A corrupt immutable artifact
//! is never accepted, and recovery distinguishes *interrupted publication*
//! (invisible by construction — retry or use the previous checkpoint)
//! from *real corruption* (fail the database rather than serve a lie).

/// Checkpoint failure modes.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CheckpointError {
    /// Filesystem I/O failure with operational context.
    #[error("checkpoint I/O during {op}: {message}")]
    Io {
        /// What was being attempted (static context).
        op: &'static str,
        /// The underlying failure.
        message: String,
    },
    /// An immutable artifact's content hash does not match its bytes (or
    /// its content-addressed filename): real corruption, never truncation.
    #[error("checkpoint artifact corrupt: {detail}")]
    Corrupt {
        /// What failed validation.
        detail: String,
    },
    /// A structural/versioned format expectation broke (magic, version,
    /// layout, codec, counts, ordering, tablet identity, cut regression).
    #[error("checkpoint format violation: {detail}")]
    Format {
        /// What broke.
        detail: String,
    },
    /// A band references an unsupported layout or codec. New formats need
    /// new code — never guess a physical layout.
    #[error("unsupported checkpoint feature: {detail}")]
    Unsupported {
        /// What is not implemented by this build.
        detail: String,
    },
    /// No installed checkpoint exists (fresh database or nothing published
    /// yet). Not an error for recovery — it means WAL replay from genesis.
    #[error("no installed checkpoint")]
    Absent,
}

impl CheckpointError {
    /// Formats an I/O failure with path context.
    #[must_use]
    pub fn io(op: &'static str, path: &std::path::Path, error: &std::io::Error) -> Self {
        Self::Io {
            op,
            message: format!("{}: {error}", path.display()),
        }
    }
}
