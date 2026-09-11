//! Durability provider concepts: what the rest of Kivi depends on.
//!
//! Tablets and workers talk about [`PersistIntent`] (what must survive),
//! [`CommitProof`] (what did survive), and [`RecoveryRecord`] (what came
//! back) through the [`DurabilityProvider`] trait. No `File`, `rustix`,
//! `io_uring`, or platform type crosses this boundary: future providers
//! (`QuorumLog`, `PersistentMemory`, `CXL`) implement the same trait.

use crate::wal::WalRecord;

/// How strongly one append must be barriered before the caller replies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DurabilityLevel {
    /// The batch is file-synced before acknowledgement: a later crash
    /// still replays it. The only level this stage implements.
    Sync,
}

/// One append unit: an ordered list of WAL records plus its barrier.
#[derive(Debug)]
pub struct PersistIntent<'a> {
    /// Records to make durable, in order, as one physical batch.
    pub records: &'a [WalRecord],
    /// Barrier the acknowledgement waits for.
    pub level: DurabilityLevel,
}

/// Proof that an append survived: the batch identity plus what it cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitProof {
    /// Per-lane batch sequence number assigned to this batch.
    pub batch_seq: u64,
    /// Bytes made durable by this append (header + body + footer).
    pub bytes_durable: u64,
    /// Sync operations issued for this append (one in the baseline).
    pub fsyncs: u64,
}

/// One recovered WAL record with its physical position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryRecord {
    /// Per-lane batch sequence the record arrived in.
    pub batch_seq: u64,
    /// The logical record.
    pub record: WalRecord,
}

/// Coarse provider state for operators (see the storage-health model).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageHealth {
    /// Appends succeed; the lane is fully writable.
    Healthy,
    /// The filesystem reports no space: mutating requests fail explicitly
    /// while reads continue where safe. Clears on the next successful
    /// append (space was freed); nothing is half-recorded either way.
    ReadOnly,
    /// A non-capacity persistence failure: the lane is terminal until
    /// restart. Mutating requests fail; already-durable history is intact.
    Failed,
}

impl core::fmt::Display for StorageHealth {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::ReadOnly => write!(f, "read-only"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

/// Cumulative per-lane counters for operators and benchmarks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LaneStats {
    /// Batches appended (one fsync each in the baseline).
    pub batches: u64,
    /// Logical records appended.
    pub records: u64,
    /// Payload + framing bytes made durable.
    pub bytes: u64,
    /// Sync operations issued.
    pub fsyncs: u64,
}

/// The durability boundary. Implementations are `Send` (they move into
/// worker threads); the single-owner discipline stays with the caller.
pub trait DurabilityProvider: Send {
    /// Makes one intent durable per its level and returns the proof.
    /// Reads never call this; mutating requests call it exactly once per
    /// accepted logical request, before any reply.
    ///
    /// # Errors
    ///
    /// Returns [`crate::DurabilityError`] when the batch cannot be made
    /// durable. The caller must not mutate logical state on failure.
    fn append_batch(
        &mut self,
        intent: &PersistIntent<'_>,
    ) -> Result<CommitProof, crate::DurabilityError>;

    /// Current coarse health (see [`StorageHealth`]).
    fn health(&self) -> StorageHealth;

    /// The WAL lane this provider owns.
    fn lane(&self) -> u16;

    /// Cumulative counters.
    fn stats(&self) -> LaneStats;
}
