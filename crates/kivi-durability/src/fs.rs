//! Crash-safe filesystem primitives and their platform semantics.
//!
//! The project owns the crash protocol; the OS owns the mechanism. Every
//! helper here documents exactly what it guarantees where.
//!
//! ## Linux
//!
//! * File data + size are durable after [`sync_file`] (`fdatasync`: data
//!   plus the metadata required to access it, which includes the size).
//! * `rename` is atomic; the new name is durable after [`sync_dir`] on the
//!   containing directory.
//! * Directory `fsync` is a real, required durability primitive.
//!
//! ## Windows
//!
//! * File data is durable after [`sync_file`] (`FlushFileBuffers`; Windows
//!   has no data-only flush — full flush every time).
//! * `rename` (`MoveFile`) is atomic via NTFS journaling.
//! * **There is no directory-sync API on Windows.** Namespace operations
//!   cannot be force-flushed by userspace; their durability timing is
//!   OS/journal policy. [`sync_dir`] therefore reports
//!   [`DirSync::UnsupportedPlatform`] instead of pretending. Atomicity of
//!   the rename itself still holds (journaled), so crash recovery sees
//!   either the old or the new name — never a half rename.
//!
//! Consequence for the WAL: file content durability is strict on both
//! platforms (every batch is file-synced before acknowledgement); only the
//! *timing* of directory-entry durability differs, and segment files are
//! never renamed (created once, appended, only truncated at the tail), so
//! the difference has no observable effect on WAL recovery. It matters for
//! `node.meta` publication and segment creation, where the documented
//! protocol (sync file before rename) is still the best available on both.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

/// Outcome of syncing a directory: actually flushed, or unsupported by the
/// platform (Windows). Callers must not ignore the difference silently —
/// this type is `#[must_use]` and both arms are logged at the call sites
/// that matter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum DirSync {
    /// The directory entry state was flushed to stable storage.
    Synced,
    /// The platform has no directory-sync API; namespace durability timing
    /// is OS policy (see module docs). Atomicity still holds.
    UnsupportedPlatform,
}

/// Flushes one file's content (and, where the OS requires it, metadata) to
/// stable storage: `fdatasync` on Unix, full flush on Windows.
///
/// # Errors
///
/// Returns [`io::Error`] when the flush fails; the data is not durable.
pub fn sync_file(file: &File) -> io::Result<()> {
    #[cfg(unix)]
    {
        file.sync_data()
    }
    #[cfg(windows)]
    {
        file.sync_all()
    }
    #[cfg(not(any(unix, windows)))]
    {
        file.sync_all()
    }
}

/// Flushes a directory's namespace state. See [`DirSync`] and the module
/// docs for the Windows difference.
///
/// # Errors
///
/// Returns [`io::Error`] when opening or syncing the directory fails.
pub fn sync_dir(path: &Path) -> io::Result<DirSync> {
    #[cfg(unix)]
    {
        let dir = File::open(path)?;
        dir.sync_all()?;
        Ok(DirSync::Synced)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(DirSync::UnsupportedPlatform)
    }
}

/// Creates a directory (and parents), then syncs it and its parent so the
/// new entries are durable where the platform allows.
///
/// # Errors
///
/// Returns [`io::Error`] when creation or syncing fails.
pub fn create_dir_all_sync(path: &Path) -> io::Result<DirSync> {
    fs::create_dir_all(path)?;
    let outcome = sync_dir(path)?;
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        let _ = sync_dir(parent)?;
    }
    Ok(outcome)
}

/// Crash-safe file publication: write a sibling temporary file, sync it,
/// atomically rename over the target, then sync the containing directory
/// where supported. A crash leaves either the old or the new content —
/// never a half file. Stale temporary files from a previous crash are
/// removed first (their content was never published).
///
/// # Errors
///
/// Returns [`io::Error`] on any filesystem failure; nothing is published.
pub fn write_atomic_sync(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = tmp_path(path);
    // A previous crash may have left an unpublished temporary behind.
    let _ = fs::remove_file(&tmp);
    {
        let mut file = File::create_new(&tmp)?;
        file.write_all(bytes)?;
        sync_file(&file)?;
    }
    fs::rename(&tmp, path)?;
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        let _ = sync_dir(parent)?;
    }
    Ok(())
}

/// Sibling temporary path used by [`write_atomic_sync`].
#[must_use]
pub fn tmp_path(path: &Path) -> std::path::PathBuf {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    std::path::PathBuf::from(tmp)
}

/// Removes a stale unpublished temporary file, ignoring all failures
/// (best effort: leftovers are harmless, publication never reads them).
pub fn remove_stale_tmp(path: &Path) {
    let _ = fs::remove_file(tmp_path(path));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_write_round_trips_and_cleans_tmp() {
        let scratch = tempfile::tempdir().expect("scratch dir");
        let target = scratch.path().join("node.meta");
        write_atomic_sync(&target, b"hello").expect("publish");
        assert_eq!(fs::read(&target).expect("read"), b"hello");
        assert!(!tmp_path(&target).exists(), "no tmp left behind");
        write_atomic_sync(&target, b"hello world").expect("republish");
        assert_eq!(fs::read(&target).expect("read"), b"hello world");
    }

    #[test]
    fn dir_sync_reports_honestly() {
        let scratch = tempfile::tempdir().expect("scratch dir");
        let outcome = sync_dir(scratch.path()).expect("sync_dir works on existing dir");
        #[cfg(unix)]
        assert_eq!(outcome, DirSync::Synced);
        #[cfg(windows)]
        assert_eq!(outcome, DirSync::UnsupportedPlatform);
    }
}
