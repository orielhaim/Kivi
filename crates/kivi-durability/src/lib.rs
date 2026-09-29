//! Production durability for Kivi: data directory, segmented WAL, recovery.
//!
//! This crate owns crash semantics; the rest of Kivi depends on its
//! concepts ([`PersistIntent`], [`CommitProof`], [`DurabilityProvider`]),
//! never on filesystem handles. Tablet and state code stay free of files,
//! `rustix`, `io_uring`, and platform calls.
//!
//! Data-directory layout:
//!
//! ```text
//! <data-dir>/
//!     LOCK                exclusive process lock (std file locking)
//!     node.meta           crash-safe node identity + incarnation (44 bytes)
//!     wal/
//!         lane-0000/
//!             00000000000000000001.wal
//!             ...
//! ```
//!
//! WAL binary formats are explicit little-endian with CRC32C integrity,
//! versioned tags, and length prefixes (no serde/bincode/rkyv anywhere).
//! See [`wal`] for the byte layouts and [`fs`] for the platform durability
//! semantics (Linux and Windows differ; the difference is modeled, not
//! hidden).

pub mod error;
pub mod fs;
pub mod node;
pub mod provider;
pub mod raft;
pub mod wal;

pub use error::{DurabilityError, RecoveryError};
pub use node::{NodeSeed, OpenDir, open_data_dir};
pub use provider::{
    BarrierCost, CommitProof, DurabilityLevel, DurabilityProvider, LaneStats, PersistIntent,
    StorageHealth,
};
pub use raft::{
    RaftCommitted, RaftEntry, RaftEntryPayload, RaftMembership, RaftPurge, RaftRecord,
    RaftTruncate, RaftVote,
};
pub use wal::{
    DEFAULT_SEGMENT_TARGET_BYTES, LaneFloor, LaneIdentity, LocalWalLane, MutationRecord,
    OutcomeRecord, RecoverySummary, SealedSegmentSummary, WalEntry, WalRecord, WorkerLaneStats,
};
