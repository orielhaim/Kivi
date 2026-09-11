//! Durability failure modes: operational errors and recovery rejections.
//!
//! [`DurabilityError`] covers things that go wrong while running (I/O,
//! capacity, locks, misconfiguration). [`RecoveryError`] covers a database
//! that refuses to open (corruption outside the torn tail, structural
//! violations, semantic divergence). Both are `Clone + Eq` so they travel
//! through engine error enums without losing exhaustiveness.

use std::io;
use std::path::PathBuf;

/// Operational durability failure. Reads are unaffected by every variant
/// here (they never touch the provider); mutating requests fail instead of
/// recording memory-only writes.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DurabilityError {
    /// Another process holds the data-directory lock.
    #[error("data directory locked (held by another process): {path:?}")]
    Locked {
        /// The contested data directory.
        path: PathBuf,
    },
    /// The filesystem reports no space during a durable write. The lane
    /// goes read-only (explicit, recoverable); nothing is half-recorded.
    #[error("storage full during {op} at {path:?}")]
    NoSpace {
        /// What was being attempted (static context).
        op: &'static str,
        /// File or directory involved.
        path: PathBuf,
    },
    /// Filesystem I/O failure. The lane is failed (terminal until restart).
    #[error("filesystem error during {op}: {message}")]
    Io {
        /// What was being attempted (static context).
        op: &'static str,
        /// The underlying error message (OS code preserved separately).
        message: String,
        /// Raw OS error code, when the platform supplied one.
        code: Option<i32>,
    },
    /// Invalid durability configuration (zero segment target, no workers).
    #[error("invalid durability configuration: {reason}")]
    InvalidConfig {
        /// What was wrong (static context).
        reason: &'static str,
    },
    /// The node incarnation reached `u64::MAX` and cannot advance: the data
    /// directory can never start again.
    #[error("node incarnation space exhausted")]
    IncarnationExhausted,
}

impl DurabilityError {
    /// Classifies a filesystem failure at one call site: full disks become
    /// [`NoSpace`](Self::NoSpace) (read-only lane), everything else becomes
    /// [`Io`](Self::Io) (failed lane). The raw OS code is preserved for
    /// operators; the chain stays typed (never a bare string).
    #[must_use]
    pub fn io(op: &'static str, path: &std::path::Path, error: &io::Error) -> Self {
        if error.kind() == io::ErrorKind::StorageFull {
            Self::NoSpace {
                op,
                path: path.to_owned(),
            }
        } else {
            Self::Io {
                op,
                message: error.to_string(),
                code: error.raw_os_error(),
            }
        }
    }
}

/// A database that refuses to open. Recovery never silently truncates
/// committed history and never converts corruption into acknowledged data
/// loss: anything outside a torn final tail fails here, loudly.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum RecoveryError {
    /// A lane directory contains an unparsable file name.
    #[error("lane {lane:04} holds an unexpected file: {name:?}")]
    UnexpectedFile {
        /// The lane being scanned.
        lane: u16,
        /// The offending file name.
        name: String,
    },
    /// Segment sequence numbers skip (a missing segment file).
    #[error("lane {lane:04} skips segment {expected} (found {found})")]
    SegmentGap {
        /// The lane being scanned.
        lane: u16,
        /// Expected sequence number.
        expected: u64,
        /// Sequence number actually found.
        found: u64,
    },
    /// A segment file fails structural validation (magic, version,
    /// identity, checksums, framing) outside the torn tail.
    #[error("lane {lane:04} segment {segment} is corrupt: {reason}")]
    CorruptSegment {
        /// The lane being scanned.
        lane: u16,
        /// The offending segment.
        segment: u64,
        /// What was wrong (static context).
        reason: &'static str,
    },
    /// A segment claims a different cluster, node, lane, or segment number
    /// than the file and directory that hold it.
    #[error("misdirected segment file: {reason}")]
    Misdirected {
        /// What did not match (static context).
        reason: &'static str,
    },
    /// Unsupported segment, batch, or record format version.
    #[error("unsupported format version {major}.{minor} for {context}")]
    UnsupportedVersion {
        /// Observed major version.
        major: u16,
        /// Observed minor version.
        minor: u16,
        /// What was being decoded (static context).
        context: &'static str,
    },
    /// Unknown state-changing record kind or version. Recovery fails rather
    /// than skipping: skipping could resurrect the wrong state.
    #[error("unknown record kind {kind} version {version}")]
    UnknownRecord {
        /// Observed record kind tag.
        kind: u16,
        /// Observed record version.
        version: u16,
    },
    /// A record names a tablet absent from (or retired in) the startup
    /// directory. The configuration forgot a tablet; discarding its history
    /// would be silent data loss.
    #[error("record names unknown tablet {tablet}")]
    UnknownTablet {
        /// The tablet no directory entry covers.
        tablet: u64,
    },
    /// A record's namespace differs from the engine's: the data directory
    /// belongs to a different deployment than the startup configuration.
    #[error("record for tablet {tablet} belongs to namespace {found}, engine serves {expected}")]
    NamespaceMismatch {
        /// The affected tablet.
        tablet: u64,
        /// The engine's namespace.
        expected: u64,
        /// The record's namespace.
        found: u64,
    },
    /// A record's fencing (epoch/guard) disagrees with the startup
    /// directory's entry for its tablet.
    #[error("record fencing disagrees with directory for tablet {tablet}")]
    FencingMismatch {
        /// The affected tablet.
        tablet: u64,
    },
    /// A tablet's commit positions skip or repeat.
    #[error("tablet {tablet} breaks commit order (expected {expected}, found {found})")]
    ChainBreak {
        /// The affected tablet.
        tablet: u64,
        /// Expected next commit position.
        expected: u64,
        /// Observed commit position.
        found: u64,
    },
    /// Re-applying a recorded mutation produced a different outcome than
    /// the recorded expected outcome: corruption, semantic incompatibility,
    /// or a deterministic-apply bug. Never accepted silently.
    #[error("replay diverged on tablet {tablet} commit {commit}")]
    OutcomeMismatch {
        /// The affected tablet.
        tablet: u64,
        /// The diverging commit position.
        commit: u64,
    },
    /// A declared length exceeds the allocation guard.
    #[error("declared length {len} exceeds maximum {max}")]
    Oversized {
        /// Declared length.
        len: usize,
        /// Configured maximum.
        max: usize,
    },
    /// Truncated input where a complete structure was required (not a torn
    /// tail position).
    #[error("truncated {context}")]
    Truncated {
        /// What was being decoded (static context).
        context: &'static str,
    },
}
