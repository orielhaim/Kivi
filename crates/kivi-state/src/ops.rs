//! Typed native operations, results, and failures.
//!
//! The core never sees Redis command strings: callers express intent with
//! [`Operation`], whose semantics are explicit and stable. Wrong-type access
//! fails without mutation; counter overflow fails without mutation; expired
//! objects behave as absent (a read never deletes).

use bytes::Bytes;
use kivi_types::{Expiry, UnixMicros};

use crate::object::{Key, ObjectType};

/// One typed native operation against a single key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operation {
    /// Fetch bytes; fails with [`WrongType`](OpError::WrongType) on counters.
    Get {
        /// Key to read.
        key: Key,
    },
    /// Store bytes, overwriting any type and clearing any expiry (SET semantics).
    Set {
        /// Key to write.
        key: Key,
        /// Value to store.
        value: Bytes,
    },
    /// Remove the key (including expired-but-unreclaimed state).
    Delete {
        /// Key to remove.
        key: Key,
    },
    /// Whether a live (present and unexpired) value exists, of any type.
    Exists {
        /// Key to probe.
        key: Key,
    },
    /// Fetch a counter value; fails with `WrongType` on byte strings.
    CounterGet {
        /// Key to read.
        key: Key,
    },
    /// Add `delta` to a counter, creating it at `delta` when absent.
    /// Negative deltas decrement. Overflow fails without mutation.
    CounterAdd {
        /// Key to update.
        key: Key,
        /// Signed addend.
        delta: i64,
    },
    /// Attach an absolute expiry; missing keys report `applied: false`.
    ExpireAt {
        /// Key to expire.
        key: Key,
        /// Logical expiration timestamp.
        expires_at: UnixMicros,
    },
    /// Remove any expiry; keys without one report `removed: false`.
    PersistExpiry {
        /// Key to persist.
        key: Key,
    },
    /// Read the expiry of a live key (`None` when the key is absent).
    GetExpiry {
        /// Key to inspect.
        key: Key,
    },
}

/// Outcome of one prepared operation.
///
/// Exhaustive by design: every consumer in the workspace must handle every
/// outcome, so adding an operation breaks matches at compile time instead of
/// silently falling into a wildcard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationResult {
    /// `Get` payload (`None` when absent, expired, or — via error — mistyped).
    Value(Option<Bytes>),
    /// `Set` completed with the new version.
    Stored {
        /// Version after the store.
        version: crate::object::ObjectVersion,
    },
    /// `Delete` completed; `existed` names a previously live object.
    Deleted {
        /// Whether a live object was removed.
        existed: bool,
    },
    /// `Exists` answer.
    Exists(bool),
    /// `CounterGet` value (`None` when absent).
    Counter(Option<i64>),
    /// `CounterAdd` completed with the new value and version.
    CounterUpdated {
        /// Value after the addition.
        value: i64,
        /// Version after the addition.
        version: crate::object::ObjectVersion,
    },
    /// `ExpireAt` outcome.
    ExpirySet {
        /// Whether the key was live and took the expiry.
        applied: bool,
    },
    /// `PersistExpiry` outcome.
    ExpiryPersisted {
        /// Whether an expiry was actually removed.
        removed: bool,
    },
    /// `GetExpiry` answer (`None` when the key is absent).
    Expiry(Option<Expiry>),
}

/// Native operation failure. Reads and validations fail; validated writes
/// apply deterministically.
///
/// Exhaustive like [`OperationResult`]: new failure modes must be handled
/// explicitly everywhere they surface.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OpError {
    /// The key holds a different logical type; nothing was mutated.
    #[error("wrong type: operation needs {expected}, key holds {found}")]
    WrongType {
        /// Type the operation required.
        expected: ObjectType,
        /// Type actually stored.
        found: ObjectType,
    },
    /// A counter addition overflowed `i64`; nothing was mutated.
    #[error("counter addition overflows i64")]
    CounterOverflow,
}
