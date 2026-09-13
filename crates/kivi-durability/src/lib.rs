//! Production durability for Kivi: data directory, segmented WAL, recovery.
//!
//! This crate owns crash semantics; the rest of Kivi depends on its
//! concepts ([`PersistIntent`], [`CommitProof`], [`RecoveryRecord`],
//! [`DurabilityProvider`]), never on filesystem handles. Tablet and state
//! code stay free of files, `rustix`, `io_uring`, and platform calls, so
//! future providers (`QuorumLog`, `PersistentMemory`, `CXL`) slot in
//! without rewriting tablet semantics. Only the local WAL provider ships
//! in this stage.
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
pub mod wal;

pub use error::{DurabilityError, RecoveryError};
pub use node::{NodeMeta, OpenDir, open_data_dir};
pub use provider::{
    CommitProof, DurabilityLevel, DurabilityProvider, LaneStats, PersistIntent, RecoveryRecord,
    StorageHealth,
};
pub use wal::{
    DEFAULT_SEGMENT_TARGET_BYTES, LaneFloor, LaneIdentity, LaneRecovery, LocalWalLane,
    MutationRecord, OutcomeRecord, RecoveredRecord, RecoverySummary, SealedSegmentSummary,
    WAL_MAJOR, WAL_MAX_BODY_BYTES, WalRecord, WorkerLaneStats,
};
