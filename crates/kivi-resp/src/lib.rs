//! `kivi-resp`: optional Redis/RESP compatibility edge for Kivi.
//!
//! Architecture: RESP is an edge protocol, never Kivi's semantic
//! foundation. This crate owns RESP parsing/encoding, per-connection RESP
//! version state, Redis command recognition, argument parsing, Redis-to-Kivi
//! translation, Kivi-to-Redis result translation, and compatibility
//! metadata. It never defines database semantics: every command maps onto
//! the same typed [`Operation`](kivi_state::Operation) the native frontend
//! executes, and conditional writes stay atomic at the owning tablet (no
//! `GET`-then-`SET` in the adapter).
//!
//! ```text
//! RESP2/RESP3
//!    ↓
//! kivi-resp (here)
//!    ↓
//! Kivi semantic request (`Operation`)
//!    ↓
//! same Kivi engine
//! ```
//!
//! Wire parsing uses the stable
//! [`redis-protocol`](https://docs.rs/redis-protocol) crate's direct
//! decode/encode interfaces (never its `codec` feature, so no `tokio-util`
//! enters this path).
//!
//! # Compatibility profile
//!
//! Kivi is not a Redis drop-in replacement. The supported subset is
//! [`Kivi RESP Compatibility Profile v1`](self::PROFILE_VERSION): exact
//! mappings where Redis semantics hold, explicit errors everywhere else.
//! The single source of truth is [`command::REGISTRY`]; dispatch,
//! `COMMAND` output, and tests all read from it.

pub mod command;
pub mod connection;
pub mod error;
pub mod translate;

pub use command::{CommandKind, CommandSpec, CompatClass, REGISTRY, arity_ok, lookup};
pub use connection::{ConnConfig, ConnMetrics, ConnState, RespConnection, RespVersion};
pub use error::{KiviErrorKind, RespError};
pub use translate::{Action, ExecuteError, Executor, Immediate, RedisOp, Reply};

/// Compatibility profile version string (pinned; future Redis releases do
/// not change Kivi behavior automatically).
pub const PROFILE_VERSION: &str = "Kivi RESP Compatibility Profile v1";

/// One row of the machine-testable support table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompatEntry {
    /// Uppercase command name.
    pub command: &'static str,
    /// Compatibility class.
    pub status: CompatClass,
    /// One-line note.
    pub notes: &'static str,
}

/// Returns the full support table, in registry order.
#[must_use]
pub fn compatibility_table() -> Vec<CompatEntry> {
    REGISTRY
        .iter()
        .map(|spec| CompatEntry {
            command: spec.name,
            status: spec.compat,
            notes: spec.notes,
        })
        .collect()
}
