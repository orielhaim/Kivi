//! RESP error taxonomy: parse failures map to Redis errors, never panics.
//!
//! Kivi engine failures map onto the small Redis error surface this profile
//! speaks (`WRONGTYPE`, `ERR`, `BUSY`). Rust internals never cross the wire.

/// Failures produced while decoding or validating one RESP request.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum RespError {
    /// Input is truncated so far; the caller must read more bytes.
    #[error("incomplete frame")]
    Incomplete,
    /// Input is structurally invalid (bad shape, bad lengths, bad pairing).
    #[error("malformed request")]
    Malformed,
    /// A declared length exceeds the configured bound.
    #[error("request exceeds bound")]
    TooLarge,
    /// Too many pipelined requests are already queued.
    #[error("pipeline depth exceeded")]
    PipelineFull,
    /// The command is unknown or explicitly unsupported in this profile.
    #[error("unsupported command")]
    Unsupported {
        /// Command name as sent (uppercased ASCII, possibly empty).
        command: String,
    },
    /// Arguments fail validation (arity, integers, conflicting options).
    #[error("invalid arguments")]
    InvalidArguments {
        /// Human-readable reason (ASCII, no Rust internals).
        reason: String,
    },
}

/// How a Kivi engine outcome surfaces as a Redis error prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KiviErrorKind {
    /// Key holds a different logical type; nothing mutated.
    WrongType,
    /// Counter overflow or version exhaustion; nothing mutated.
    Rejected,
    /// Invalid request that reached execution (defensive; usually caught
    /// at parse time).
    Invalid,
    /// Bounded queues full or lane saturated; safe to retry with backoff.
    Overloaded,
    /// Unexpected failure; safe to retry, likely pointless.
    Internal,
}

impl KiviErrorKind {
    /// Renders the Redis error prefix plus message (without the leading
    /// `-`; framing adds it).
    #[must_use]
    pub const fn message(self) -> &'static str {
        match self {
            Self::WrongType => "WRONGTYPE Operation against a key holding the wrong kind of value",
            Self::Rejected => "ERR operation rejected",
            Self::Invalid => "ERR invalid request",
            Self::Overloaded => "BUSY Kivi overloaded, retry with backoff",
            Self::Internal => "ERR internal error",
        }
    }
}
