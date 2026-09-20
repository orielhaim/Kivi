//! Semantic Fabric: project-owned operation semantics (RFC §73–§79).
//!
//! Kivi knows what its built-ins mean. Every native operation maps to one
//! [`OperationSemantics`] descriptor below; no user or WASM input can mint
//! one. Capabilities ([`Caps::COMMUTATIVE_RESULT`], [`Caps::PARTITIONABLE`],
//! …) are granted only by matching a built-in operation here, never by
//! caller assertion.
//!
//! The descriptors drive three decisions without owning execution:
//!
//! ```text
//! without REQUIRES_TOTAL_ORDER + with COMMUTATIVE_RESULT
//!     → eligible for a future unordered fast path (CURP; not implemented)
//! with BOUNDED
//!     → escrow accounting applies, never a global check-then-act
//! with PARTITIONABLE
//!     → per-shard ordering suffices, no global total order is claimed
//! ```
//!
//! Descriptors are `const`: semantics are a compile-time property of the
//! operation, not runtime input.

use bitflags::bitflags;

use crate::ops::Operation;

bitflags! {
    /// Capability flags for one built-in operation class. Flags compose:
    /// each named descriptor below is the exact capability set its class
    /// carries, and future execution paths may only rely on flags granted
    /// here.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Caps: u16 {
        /// Observes state.
        const READS = 1 << 0;
        /// Changes logical state.
        const WRITES = 1 << 1;
        /// Re-execution with the same identity yields the same outcome.
        const IDEMPOTENT = 1 << 2;
        /// Reordering concurrent instances converges to the same state.
        const COMMUTATIVE_STATE = 1 << 3;
        /// Reordering concurrent instances yields the same observable results.
        const COMMUTATIVE_RESULT = 1 << 4;
        /// Grouping/nesting instances preserves meaning.
        const ASSOCIATIVE = 1 << 5;
        /// Work splits across shards without a global order.
        const PARTITIONABLE = 1 << 6;
        /// State moves monotonically (offsets, fencing tokens, versions).
        const MONOTONIC = 1 << 7;
        /// A global bound constrains the value (capacity, outstanding).
        const BOUNDED = 1 << 8;
        /// Requires the tablet's total order (ordered consensus path).
        const REQUIRES_TOTAL_ORDER = 1 << 9;
    }
}

/// Where an operation's conflicts live: what must serialize against what.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConflictScope {
    /// Conflicts are confined to one key (point OCC + intent reservation).
    Key,
    /// Conflicts are confined to one stream shard (per-shard offsets).
    Shard,
    /// Conflicts draw on a shared escrow pool (rights transfer coordinates).
    EscrowPool,
    /// No shared mutable state (pure reads of one key's metadata).
    None,
}

/// Project-owned semantic description of one built-in operation class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationSemantics {
    /// Capability flags carried by the class.
    pub caps: Caps,
    /// What must serialize against what.
    pub conflict_scope: ConflictScope,
}

/// Plain point read: no order requirement beyond the read contract.
pub const READ: OperationSemantics = OperationSemantics {
    caps: Caps::READS
        .union(Caps::IDEMPOTENT)
        .union(Caps::COMMUTATIVE_STATE)
        .union(Caps::COMMUTATIVE_RESULT),
    conflict_scope: ConflictScope::None,
};

/// Strict byte/counter write: totally ordered within its tablet.
pub const STRICT_WRITE: OperationSemantics = OperationSemantics {
    caps: Caps::WRITES.union(Caps::REQUIRES_TOTAL_ORDER),
    conflict_scope: ConflictScope::Key,
};

/// Strict counter addition: the final state commutes but the returned
/// ordinal does not — structurally barred from unordered paths.
pub const STRICT_COUNTER_ADD: OperationSemantics = OperationSemantics {
    caps: Caps::READS
        .union(Caps::WRITES)
        .union(Caps::COMMUTATIVE_STATE)
        .union(Caps::ASSOCIATIVE)
        .union(Caps::REQUIRES_TOTAL_ORDER),
    conflict_scope: ConflictScope::Key,
};

/// Commutative addition: neither state nor result reveals order.
pub const COMMUTATIVE_ADD: OperationSemantics = OperationSemantics {
    caps: Caps::WRITES
        .union(Caps::COMMUTATIVE_STATE)
        .union(Caps::COMMUTATIVE_RESULT)
        .union(Caps::ASSOCIATIVE),
    conflict_scope: ConflictScope::Key,
};

/// Escrow-bounded mutation: local while rights cover it, coordinated
/// transfer when they do not. Never a global check-then-act.
pub const BOUNDED_MUTATION: OperationSemantics = OperationSemantics {
    caps: Caps::READS
        .union(Caps::WRITES)
        .union(Caps::BOUNDED)
        .union(Caps::REQUIRES_TOTAL_ORDER),
    conflict_scope: ConflictScope::EscrowPool,
};

/// Semaphore acquire/release: bounded like escrow, keyed by permit id.
pub const SEMAPHORE_MUTATION: OperationSemantics = OperationSemantics {
    caps: Caps::READS
        .union(Caps::WRITES)
        .union(Caps::IDEMPOTENT)
        .union(Caps::COMMUTATIVE_RESULT)
        .union(Caps::BOUNDED)
        .union(Caps::REQUIRES_TOTAL_ORDER),
    conflict_scope: ConflictScope::Key,
};

/// Lease grant/renew/release: monotonic fencing, totally ordered grants.
pub const LEASE_MUTATION: OperationSemantics = OperationSemantics {
    caps: Caps::READS
        .union(Caps::WRITES)
        .union(Caps::IDEMPOTENT)
        .union(Caps::MONOTONIC)
        .union(Caps::REQUIRES_TOTAL_ORDER),
    conflict_scope: ConflictScope::Key,
};

/// Stream append: ordered within one shard, partitionable across shards.
pub const STREAM_APPEND: OperationSemantics = OperationSemantics {
    caps: Caps::WRITES
        .union(Caps::IDEMPOTENT)
        .union(Caps::COMMUTATIVE_RESULT)
        .union(Caps::PARTITIONABLE)
        .union(Caps::MONOTONIC)
        .union(Caps::REQUIRES_TOTAL_ORDER),
    conflict_scope: ConflictScope::Shard,
};

/// Stream read/trim: shard-local metadata, no global order implied.
pub const STREAM_READ: OperationSemantics = OperationSemantics {
    caps: Caps::READS
        .union(Caps::IDEMPOTENT)
        .union(Caps::COMMUTATIVE_STATE)
        .union(Caps::COMMUTATIVE_RESULT)
        .union(Caps::PARTITIONABLE),
    conflict_scope: ConflictScope::Shard,
};

/// Transaction step: OCC validation plus a durable intent; the decision is
/// a separate totally ordered record.
pub const TRANSACTION_STEP: OperationSemantics = OperationSemantics {
    caps: Caps::READS
        .union(Caps::WRITES)
        .union(Caps::IDEMPOTENT)
        .union(Caps::REQUIRES_TOTAL_ORDER),
    conflict_scope: ConflictScope::Key,
};

/// Returns the project-owned semantics of one operation. Total: every
/// operation maps to exactly one descriptor, so a new operation without an
/// entry fails to compile at its call site's exhaustiveness check.
#[must_use]
pub const fn semantics_of(operation: &Operation) -> OperationSemantics {
    match operation {
        Operation::Get { .. }
        | Operation::Exists { .. }
        | Operation::CounterGet { .. }
        | Operation::GetExpiry { .. }
        | Operation::GetRange { .. }
        | Operation::BytesLength { .. }
        | Operation::GetVersion { .. }
        | Operation::CommutativeGet { .. }
        | Operation::BoundedCounterGet { .. }
        | Operation::SemaphoreInspect { .. } => READ,
        Operation::Set { .. }
        | Operation::SetChunked { .. }
        | Operation::SetFabric { .. }
        | Operation::SetRange { .. }
        | Operation::Delete { .. }
        | Operation::ExpireAt { .. }
        | Operation::PersistExpiry { .. }
        | Operation::SetConditional { .. }
        | Operation::SetConditionalChunked { .. }
        | Operation::SetConditionalFabric { .. } => STRICT_WRITE,
        Operation::CounterAdd { .. } => STRICT_COUNTER_ADD,
        Operation::CommutativeAdd { .. } => COMMUTATIVE_ADD,
        Operation::BoundedCounterCreate { .. }
        | Operation::BoundedCounterAdd { .. }
        | Operation::EscrowTransfer { .. } => BOUNDED_MUTATION,
        Operation::SemaphoreCreate { .. }
        | Operation::SemaphoreAcquire { .. }
        | Operation::SemaphoreRelease { .. } => SEMAPHORE_MUTATION,
        Operation::LeaseAcquire { .. }
        | Operation::LeaseRenew { .. }
        | Operation::LeaseRelease { .. }
        | Operation::LeaseInspect { .. } => LEASE_MUTATION,
        Operation::StreamCreate { .. } | Operation::StreamAppend { .. } => STREAM_APPEND,
        Operation::StreamRead { .. } | Operation::StreamTrim { .. } => STREAM_READ,
        Operation::TxnPrepare { .. }
        | Operation::TxnFinalize { .. }
        | Operation::TxnCommitLocal { .. } => TRANSACTION_STEP,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_counter_result_does_not_commute() {
        assert!(STRICT_COUNTER_ADD.caps.contains(Caps::COMMUTATIVE_STATE));
        assert!(!STRICT_COUNTER_ADD.caps.contains(Caps::COMMUTATIVE_RESULT));
        assert!(STRICT_COUNTER_ADD.caps.contains(Caps::REQUIRES_TOTAL_ORDER));
    }

    #[test]
    fn commutative_counter_commutes_observably() {
        assert!(COMMUTATIVE_ADD.caps.contains(Caps::COMMUTATIVE_STATE));
        assert!(COMMUTATIVE_ADD.caps.contains(Caps::COMMUTATIVE_RESULT));
        assert!(!COMMUTATIVE_ADD.caps.contains(Caps::REQUIRES_TOTAL_ORDER));
    }

    #[test]
    fn stream_is_partitionable_without_global_order() {
        assert!(STREAM_APPEND.caps.contains(Caps::PARTITIONABLE));
        assert_eq!(STREAM_APPEND.conflict_scope, ConflictScope::Shard);
    }
}
