//! Durability provider concepts: what the rest of Kivi depends on.
//!
//! Tablets and workers talk about [`PersistIntent`] (what must survive),
//! [`CommitProof`] (what did survive), and [`StorageHealth`] through the
//! [`DurabilityProvider`] trait. No `File`, `rustix`, `io_uring`, or platform
//! type crosses this boundary.

use crate::wal::WalRecord;

/// How strongly one append must be barriered before the caller replies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DurabilityLevel {
    /// The batch is file-synced before acknowledgement: a later crash
    /// still replays it. The only level implemented here.
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

/// Where one durability barrier's wall time actually goes, split into the
/// three phases a batch passes through on the lane thread: turning records
/// into a body, handing bytes to the OS, and forcing them to stable storage.
///
/// The split exists because the three have completely different remedies.
/// `encode` scales with bytes and shrinks by not encoding twice; `write` is
/// syscall-bound and shrinks by fewer, larger writes; `fsync` is a fixed
/// platform cost that shrinks *only* by amortizing more mutations behind
/// each one. An operator who can read all three can tell which of those is
/// worth doing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BarrierCost {
    /// Batches measured (the denominator for every mean below).
    pub batches: u64,
    /// Cumulative record-encode time.
    pub encode_ns: u64,
    /// Cumulative file-write time (the `write` syscalls, pre-flush).
    pub write_ns: u64,
    /// Cumulative flush-to-stable-storage time.
    pub fsync_ns: u64,
    /// Worst single flush.
    pub fsync_max_ns: u64,
    /// Cumulative records per batch (for the per-mutation view).
    pub records: u64,
    /// Cumulative time in segment rotation (create, header flush, directory
    /// sync). A rotation pays a *second* platform flush, so it lands here
    /// rather than inside `fsync_ns`.
    pub rotate_ns: u64,
    /// Rotations performed.
    pub rotations: u64,
    /// Cumulative framing time (batch header CRC, body CRC, footer): the
    /// work between having encoded records and handing bytes to the OS.
    pub frame_ns: u64,
}

impl BarrierCost {
    /// Nanoseconds of a duration, saturating rather than wrapping so a
    /// duration beyond ~584 years cannot make a running total go backwards.
    fn ns(duration: core::time::Duration) -> u64 {
        u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
    }

    /// Folds one measured batch into the running totals.
    pub fn record(
        &mut self,
        records: usize,
        encode: core::time::Duration,
        write: core::time::Duration,
        fsync: core::time::Duration,
    ) {
        self.batches += 1;
        self.records += records as u64;
        self.encode_ns += Self::ns(encode);
        self.write_ns += Self::ns(write);
        self.fsync_ns += Self::ns(fsync);
        self.fsync_max_ns = self.fsync_max_ns.max(Self::ns(fsync));
    }
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
