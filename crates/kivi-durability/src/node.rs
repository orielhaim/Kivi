//! Persistent node identity and the data-directory lifecycle.
//!
//! Layout owned here:
//!
//! ```text
//! <data-dir>/
//!     LOCK         zero-length coordination file, exclusively locked
//!     node.meta    crash-safe identity record (44 bytes, see below)
//!     wal/         one lane directory per worker (see [`crate::wal`])
//! ```
//!
//! `node.meta` binary format (all little-endian):
//!
//! ```text
//! 0   4  magic "KVNM" (0x4D4E564B)
//! 4   2  major = 1
//! 6   2  minor = 0
//! 8   16 cluster (u128)
//! 24  8  node (u64)
//! 32  8  incarnation (u64)
//! 40  4  CRC32C of bytes [0..40]
//! total 44 = NODE_META_LEN
//! ```
//!
//! Startup protocol: acquire the exclusive `LOCK` first (a second process
//! fails clearly with [`DurabilityError::Locked`]); then either create
//! identity (CSPRNG cluster + node, incarnation `INITIAL`, durably
//! published) or load, validate, advance the incarnation by exactly one,
//! and durably republish — all before any network listener binds. Restart
//! preserves cluster/node and strictly advances incarnation; exhaustion
//! fails startup instead of wrapping.

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use kivi_codec::integrity::crc32c_checksum;
use kivi_types::{ClusterId, NodeId, NodeIncarnation};

use crate::error::DurabilityError;
use crate::fs::{create_dir_all_sync, remove_stale_tmp, write_atomic_sync};

/// Magic word: ASCII `"KVNM"` read as a little-endian `u32`.
pub const NODE_META_MAGIC: u32 = 0x4D4E_564B;
/// Current metadata major version.
pub const NODE_META_MAJOR: u16 = 1;
/// Current metadata minor version.
pub const NODE_META_MINOR: u16 = 0;
/// Encoded metadata length in bytes.
pub const NODE_META_LEN: usize = 44;
/// Coordination file name inside the data directory.
pub const LOCK_FILE_NAME: &str = "LOCK";
/// Identity file name inside the data directory.
pub const NODE_META_FILE_NAME: &str = "node.meta";
/// WAL root directory name inside the data directory.
pub const WAL_DIR_NAME: &str = "wal";

/// Persistent node identity: stable across restarts, advanced incarnation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeMeta {
    /// Stable cluster identity (CSPRNG at first startup).
    pub cluster: ClusterId,
    /// Stable node identity (CSPRNG at first startup, nonzero).
    pub node: NodeId,
    /// This process's incarnation (strictly increasing per startup).
    pub incarnation: NodeIncarnation,
}

impl NodeMeta {
    /// Encodes the 44 canonical bytes.
    #[must_use]
    pub fn encode(&self) -> [u8; NODE_META_LEN] {
        let mut out = [0u8; NODE_META_LEN];
        out[0..4].copy_from_slice(&NODE_META_MAGIC.to_le_bytes());
        out[4..6].copy_from_slice(&NODE_META_MAJOR.to_le_bytes());
        out[6..8].copy_from_slice(&NODE_META_MINOR.to_le_bytes());
        out[8..24].copy_from_slice(&self.cluster.as_u128().to_le_bytes());
        out[24..32].copy_from_slice(&self.node.as_u64().to_le_bytes());
        out[32..40].copy_from_slice(&self.incarnation.as_u64().to_le_bytes());
        let crc = crc32c_checksum(&out[..40]);
        out[40..44].copy_from_slice(&crc.to_le_bytes());
        out
    }

    /// Decodes and fully validates 44 bytes (magic, version, checksum).
    ///
    /// # Errors
    ///
    /// Returns [`DurabilityError`] on wrong length, bad magic, unknown
    /// version, or checksum mismatch. Never panics on untrusted bytes.
    pub fn decode(input: &[u8]) -> Result<Self, DurabilityError> {
        if input.len() != NODE_META_LEN {
            return Err(DurabilityError::InvalidConfig {
                reason: "node.meta has wrong length",
            });
        }
        let magic = u32::from_le_bytes(input[0..4].try_into().unwrap_or([0; 4]));
        if magic != NODE_META_MAGIC {
            return Err(DurabilityError::InvalidConfig {
                reason: "node.meta has bad magic (not a Kivi data directory?)",
            });
        }
        let major = u16::from_le_bytes(input[4..6].try_into().unwrap_or([0; 2]));
        let minor = u16::from_le_bytes(input[6..8].try_into().unwrap_or([0; 2]));
        if major != NODE_META_MAJOR || minor != NODE_META_MINOR {
            return Err(DurabilityError::InvalidConfig {
                reason: "node.meta has unsupported version",
            });
        }
        let stored = u32::from_le_bytes(input[40..44].try_into().unwrap_or([0; 4]));
        if crc32c_checksum(&input[..40]) != stored {
            return Err(DurabilityError::InvalidConfig {
                reason: "node.meta checksum mismatch (corrupt identity?)",
            });
        }
        Ok(Self {
            cluster: ClusterId::from_u128(u128::from_le_bytes(
                input[8..24].try_into().unwrap_or([0; 16]),
            )),
            node: NodeId::from_u64(u64::from_le_bytes(
                input[24..32].try_into().unwrap_or([0; 8]),
            )),
            incarnation: NodeIncarnation::from_u64(u64::from_le_bytes(
                input[32..40].try_into().unwrap_or([0; 8]),
            )),
        })
    }
}

/// An opened data directory: lock held, identity established, WAL root
/// ready. Dropping this releases the process lock (after which another
/// process may open the directory).
#[derive(Debug)]
pub struct OpenDir {
    /// The data-directory root.
    pub dir: PathBuf,
    /// This process's identity (fresh or loaded + advanced).
    pub meta: NodeMeta,
    /// Whether this startup created the identity (first startup).
    pub fresh: bool,
    /// The held exclusive lock; dropping releases it.
    _lock: File,
}

impl OpenDir {
    /// Returns the WAL root directory.
    #[must_use]
    pub fn wal_dir(&self) -> PathBuf {
        self.dir.join(WAL_DIR_NAME)
    }
}

/// Draws 128 CSPRNG bits from the OS. Only randomness source for
/// persistent and session identities in the workspace.
fn random_u128() -> Result<u128, DurabilityError> {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).map_err(|error| DurabilityError::Io {
        op: "draw OS randomness",
        message: error.to_string(),
        code: None,
    })?;
    Ok(u128::from_le_bytes(bytes))
}

/// Draws a nonzero node identity (zero stays reserved for sentinels).
fn random_node_id() -> Result<NodeId, DurabilityError> {
    loop {
        let mut bytes = [0u8; 8];
        getrandom::fill(&mut bytes).map_err(|error| DurabilityError::Io {
            op: "draw OS randomness",
            message: error.to_string(),
            code: None,
        })?;
        // Retry the 2^-64 zero draw rather than biasing any value.
        let candidate = u64::from_le_bytes(bytes);
        if candidate != 0 {
            return Ok(NodeId::from_u64(candidate));
        }
    }
}

/// Opens (creating if needed) a Kivi data directory: lock, identity,
/// incarnation advance, WAL root — in that order, before any listener.
///
/// # Errors
///
/// Returns [`DurabilityError::Locked`] when another process holds the
/// directory, [`DurabilityError::IncarnationExhausted`] when the
/// incarnation cannot advance, or filesystem/validation failures.
/// Nothing is published until every step succeeds.
pub fn open_data_dir(path: &Path) -> Result<OpenDir, DurabilityError> {
    let dir_sync = create_dir_all_sync(path)
        .map_err(|error| DurabilityError::io("create data directory", path, &error))?;
    tracing::debug!(dir = %path.display(), ?dir_sync, "data directory ready");
    // The lock guards every mutation below against concurrent processes.
    // `File::try_lock` (std, stable since 1.89): non-blocking and
    // unambiguous — a held lock is an immediate clear error, no waiting.
    let lock_path = path.join(LOCK_FILE_NAME);
    let lock = File::create(&lock_path)
        .map_err(|error| DurabilityError::io("create data-directory lock", &lock_path, &error))?;
    lock.try_lock().map_err(|_| DurabilityError::Locked {
        path: path.to_owned(),
    })?;
    let meta_path = path.join(NODE_META_FILE_NAME);
    remove_stale_tmp(&meta_path);
    let (meta, fresh) = match fs::read(&meta_path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            // First startup: mint stable identities from the OS CSPRNG.
            let fresh = NodeMeta {
                cluster: ClusterId::from_u128(random_u128()?),
                node: random_node_id()?,
                incarnation: NodeIncarnation::INITIAL,
            };
            publish_meta(&meta_path, &fresh)?;
            (fresh, true)
        }
        Err(error) => {
            return Err(DurabilityError::io(
                "read node metadata",
                &meta_path,
                &error,
            ));
        }
        Ok(bytes) => {
            let loaded = NodeMeta::decode(&bytes)?;
            if !loaded.incarnation.is_valid() {
                return Err(DurabilityError::InvalidConfig {
                    reason: "node.meta carries an invalid incarnation",
                });
            }
            // Strictly advance before serving anything: a restarted process
            // must never present a recycled incarnation.
            let advanced = loaded
                .incarnation
                .next()
                .map_err(|_| DurabilityError::IncarnationExhausted)?;
            let next = NodeMeta {
                incarnation: advanced,
                ..loaded
            };
            publish_meta(&meta_path, &next)?;
            (next, false)
        }
    };
    let wal_dir = path.join(WAL_DIR_NAME);
    let wal_sync = create_dir_all_sync(&wal_dir)
        .map_err(|error| DurabilityError::io("create WAL root", &wal_dir, &error))?;
    tracing::debug!(dir = %wal_dir.display(), ?wal_sync, "WAL root ready");
    tracing::info!(
        cluster = meta.cluster.as_u128(),
        node = meta.node.as_u64(),
        incarnation = meta.incarnation.as_u64(),
        fresh,
        dir = %path.display(),
        "data directory open",
    );
    Ok(OpenDir {
        dir: path.to_owned(),
        meta,
        fresh,
        _lock: lock,
    })
}

/// Crash-safe identity publication (tmp + sync + rename + dir sync).
fn publish_meta(path: &Path, meta: &NodeMeta) -> Result<(), DurabilityError> {
    write_atomic_sync(path, &meta.encode())
        .map_err(|error| DurabilityError::io("publish node metadata", path, &error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_round_trips_and_rejects_damage() {
        let meta = NodeMeta {
            cluster: ClusterId::from_u128(0xC10C),
            node: NodeId::from_u64(7),
            incarnation: NodeIncarnation::from_u64(41),
        };
        let bytes = meta.encode();
        assert_eq!(bytes.len(), NODE_META_LEN);
        assert_eq!(NodeMeta::decode(&bytes).expect("decode"), meta);
        // Golden prefix: magic + version are fixed bytes.
        assert_eq!(&bytes[0..4], b"KVNM");
        assert_eq!(&bytes[4..8], &[1, 0, 0, 0]);
        let mut bad = bytes;
        bad[10] ^= 0xFF;
        assert!(NodeMeta::decode(&bad).is_err(), "bit flip detected");
        let mut bad = bytes;
        bad[0] ^= 0xFF;
        assert!(NodeMeta::decode(&bad).is_err(), "bad magic rejected");
        assert!(NodeMeta::decode(&bytes[..20]).is_err(), "short rejected");
    }

    #[test]
    fn first_startup_mints_identity_and_restart_advances() {
        let scratch = tempfile::tempdir().expect("scratch");
        let first = open_data_dir(scratch.path()).expect("first open");
        assert!(first.fresh);
        assert_eq!(first.meta.incarnation, NodeIncarnation::INITIAL);
        assert!(first.meta.node.as_u64() != 0);
        let (cluster, node) = (first.meta.cluster, first.meta.node);
        drop(first);
        let second = open_data_dir(scratch.path()).expect("second open");
        assert!(!second.fresh);
        assert_eq!(second.meta.cluster, cluster);
        assert_eq!(second.meta.node, node);
        assert_eq!(
            second.meta.incarnation,
            NodeIncarnation::from_u64(NodeIncarnation::INITIAL.as_u64() + 1)
        );
        drop(second);
        let third = open_data_dir(scratch.path()).expect("third open");
        assert!(third.meta.incarnation.supersedes(NodeIncarnation::from_u64(
            NodeIncarnation::INITIAL.as_u64() + 1
        )));
    }

    #[test]
    fn second_process_fails_clearly_on_lock() {
        // Separate `open` calls own separate lock descriptions on both
        // backends (Unix `flock`, Windows `LockFileEx`), so contention
        // inside one test process proves exactly what a second process
        // would hit: the lock is held, not merely referenced.
        let scratch = tempfile::tempdir().expect("scratch");
        let _first = open_data_dir(scratch.path()).expect("first open");
        let err = open_data_dir(scratch.path()).expect_err("second open fails");
        assert!(
            matches!(err, DurabilityError::Locked { .. }),
            "clear lock error, got {err:?}"
        );
    }

    #[test]
    fn corrupt_meta_fails_startup() {
        let scratch = tempfile::tempdir().expect("scratch");
        let opened = open_data_dir(scratch.path()).expect("first open");
        drop(opened);
        let meta_path = scratch.path().join(NODE_META_FILE_NAME);
        let mut bytes = fs::read(&meta_path).expect("read");
        bytes[30] ^= 0x01;
        fs::write(&meta_path, &bytes).expect("corrupt");
        let err = open_data_dir(scratch.path()).expect_err("corrupt meta fails");
        assert!(
            matches!(err, DurabilityError::InvalidConfig { .. }),
            "clear config error, got {err:?}"
        );
    }
}
