//! Deterministic virtual persistent storage.
//!
//! A conservative persistence model for crash-safety testing — not an ext4
//! emulation. The single load-bearing rule is `write != durable`: every file
//! carries a process-visible (pending) image and a crash-surviving (durable)
//! image, and only explicit [`StorageOp`] barrier operations (`SyncFile`,
//! `SyncDir`) move state from the former to the latter. A crash reverts all
//! pending images to their durable counterparts.
//!
//! Directory entries are versioned independently from file data, so the
//! classic power-loss-safe publication pattern is testable exactly:
//!
//! ```text
//! write temp → sync temp → rename temp final → sync parent dir
//! ```
//!
//! Skip the parent sync and a crash loses the rename (entry revert) while
//! the synced temp data survives orphaned — the precise bug class this model
//! exists to expose. Reads observe pending images (what the process sees);
//! `durable_image` / `durable_entries` inspect what a crash would keep.
//!
//! Failures arrive as explicit [`StorageFault`] parameters (recorded in the
//! trace by the driver), plus natural [`StorageError::NoSpace`] from the
//! capacity bound and a device on/offline flag. No failure is silent and
//! none depends on ambient randomness.

use std::collections::{BTreeMap, BTreeSet};

/// A persistence operation on the virtual disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageOp {
    /// Create an empty file (plus parent entry). Fails if present.
    Create {
        /// File path.
        path: String,
    },
    /// Write `data` at `offset`; the file must already exist (see
    /// [`StorageOp::Create`]). Offset must not lie past end of file.
    Write {
        /// File path.
        path: String,
        /// Byte offset to write at.
        offset: u64,
        /// Bytes to store.
        data: Vec<u8>,
    },
    /// Resize the pending image (zero-filling on growth).
    Truncate {
        /// File path.
        path: String,
        /// New length in bytes.
        len: u64,
    },
    /// Move a file, updating both parent directories. Destination must not exist.
    Rename {
        /// Source path.
        from: String,
        /// Destination path.
        to: String,
    },
    /// Unlink a file (pending-side; persists only via parent `SyncDir`).
    Remove {
        /// File path.
        path: String,
    },
    /// Persist one file's pending image (data barrier only).
    SyncFile {
        /// File path.
        path: String,
    },
    /// Persist one directory's pending entries (namespace barrier only).
    SyncDir {
        /// Directory path (`""` is the root).
        dir: String,
    },
}

/// Deterministic injected device/file fault for one operation.
///
/// Every variant is an explicit, reproducible behavior — the driver records
/// the fault alongside the outcome, so faulted runs replay exactly.
/// Bit rot uses the dedicated [`VirtualDisk::corrupt`] method instead of a
/// variant here: corruption strikes durable state independently of any
/// operation in flight.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum StorageFault {
    /// No fault injected.
    None,
    /// Fail the operation with [`StorageError::EIO`]; nothing changes.
    EIO,
    /// Fail the operation with [`StorageError::NoSpace`]; nothing changes.
    NoSpace,
    /// Apply only the first `applied_bytes` of a write (clamped to the
    /// write length); the operation still reports success.
    TornWrite {
        /// Bytes of the write to apply.
        applied_bytes: usize,
    },
    /// Acknowledge a write without changing anything (phantom success).
    LostWrite,
    /// Apply a write to `victim` instead of its target (which is untouched).
    Misdirect {
        /// Path receiving the write.
        victim: String,
    },
    /// Fail the operation with [`StorageError::Unavailable`]; nothing changes.
    Unavailable,
}

impl Default for StorageFault {
    /// No fault injected.
    fn default() -> Self {
        Self::None
    }
}

/// Outcome of a successfully applied operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageResult {
    /// Operation completed with no payload.
    Done,
    /// Bytes read (pending image).
    Read(Vec<u8>),
    /// Bytes the caller believes were written (may exceed what stuck when
    /// a write fault was injected — that deception is the fault).
    Written {
        /// Acknowledged byte count.
        applied: usize,
    },
}

/// Storage operation failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum StorageError {
    /// Path is empty or contains NUL.
    #[error("invalid path {path:?}")]
    InvalidPath {
        /// Offending path.
        path: String,
    },
    /// No such file (or no directory entry pointing at it).
    #[error("not found: {path:?}")]
    NotFound {
        /// Missing path.
        path: String,
    },
    /// Create/rename target already exists.
    #[error("already exists: {path:?}")]
    AlreadyExists {
        /// Conflicting path.
        path: String,
    },
    /// Write offset lies past end of file.
    #[error("offset {offset} past end of file ({len} bytes)")]
    InvalidOffset {
        /// Requested offset.
        offset: u64,
        /// Current length.
        len: u64,
    },
    /// Corruption offset lies past end of the durable image.
    #[error("corrupt offset {offset} past durable length {len}")]
    CorruptOutOfRange {
        /// Requested offset.
        offset: u64,
        /// Durable length.
        len: u64,
    },
    /// The fault does not apply to this operation kind.
    #[error("fault does not apply to this operation")]
    InapplicableFault,
    /// Injected or device I/O error.
    #[error("injected I/O error")]
    EIO,
    /// Capacity exceeded (natural or injected).
    #[error("no space left on device")]
    NoSpace,
    /// Device flagged offline (or per-operation unavailability injected).
    #[error("device unavailable")]
    Unavailable,
}

/// One file: independent pending (process-visible) and durable images.
/// `None` means absent on that side (never created, or removed pre-sync).
#[derive(Debug, Clone, PartialEq, Eq)]
struct SimFile {
    pending: Option<Vec<u8>>,
    durable: Option<Vec<u8>>,
}

/// One directory: independent pending and durable child-name sets.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct SimDir {
    pending: BTreeSet<String>,
    durable: BTreeSet<String>,
}

/// Deterministic virtual disk: explicit volatile/durable split per file and
/// per directory, capacity bound, and on/offline flag.
#[derive(Debug, Clone)]
pub struct VirtualDisk {
    files: BTreeMap<String, SimFile>,
    dirs: BTreeMap<String, SimDir>,
    capacity_bytes: u64,
    online: bool,
}

impl VirtualDisk {
    /// Creates an empty online disk with `capacity_bytes` of space and a
    /// synced empty root directory.
    #[must_use]
    pub fn new(capacity_bytes: u64) -> Self {
        let mut dirs = BTreeMap::new();
        dirs.insert(String::new(), SimDir::default());
        Self {
            files: BTreeMap::new(),
            dirs,
            capacity_bytes,
            online: true,
        }
    }

    /// Sets device availability. While offline every operation fails with
    /// [`StorageError::Unavailable`]; the flag survives crashes (it models
    /// the device, not the process).
    pub fn set_online(&mut self, online: bool) {
        self.online = online;
    }

    /// Whether the device currently serves operations.
    #[must_use]
    pub const fn is_online(&self) -> bool {
        self.online
    }

    /// Applies one operation with an explicit injected fault.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] on missing paths, collisions, capacity or
    /// device failures, and inapplicable fault/operation combinations.
    /// Failed operations change nothing.
    pub fn apply(
        &mut self,
        op: &StorageOp,
        fault: &StorageFault,
    ) -> Result<StorageResult, StorageError> {
        if !self.online || *fault == StorageFault::Unavailable {
            return Err(StorageError::Unavailable);
        }
        if *fault == StorageFault::EIO {
            return Err(StorageError::EIO);
        }
        if *fault == StorageFault::NoSpace {
            return Err(StorageError::NoSpace);
        }
        match op {
            StorageOp::Create { path } => self.apply_create(path, fault),
            StorageOp::Write { path, offset, data } => self.apply_write(path, *offset, data, fault),
            StorageOp::Truncate { path, len } => self.apply_truncate(path, *len, fault),
            StorageOp::Rename { from, to } => self.apply_rename(from, to, fault),
            StorageOp::Remove { path } => self.apply_remove(path, fault),
            StorageOp::SyncFile { path } => self.apply_sync_file(path, fault),
            StorageOp::SyncDir { dir } => self.apply_sync_dir(dir, fault),
        }
    }

    /// Length of a file's pending image, or `NotFound` when absent.
    fn pending_len(&self, path: &str) -> Result<u64, StorageError> {
        let file = self.files.get(path).ok_or_else(|| StorageError::NotFound {
            path: path.to_owned(),
        })?;
        let current = file
            .pending
            .as_ref()
            .ok_or_else(|| StorageError::NotFound {
                path: path.to_owned(),
            })?;
        Ok(u64::try_from(current.len()).unwrap_or(u64::MAX))
    }

    /// Mutable access to a file's pending image, or `NotFound` when absent.
    fn pending_image_mut(&mut self, path: &str) -> Result<&mut Vec<u8>, StorageError> {
        self.files
            .get_mut(path)
            .ok_or_else(|| StorageError::NotFound {
                path: path.to_owned(),
            })?
            .pending
            .as_mut()
            .ok_or_else(|| StorageError::NotFound {
                path: path.to_owned(),
            })
    }

    /// Rejects writes starting past end of file.
    fn check_offset(&self, path: &str, offset: u64) -> Result<(), StorageError> {
        let len = self.pending_len(path)?;
        if offset > len {
            return Err(StorageError::InvalidOffset { offset, len });
        }
        Ok(())
    }

    /// Stores bytes at an offset, growing with zero fill after a capacity check.
    fn store_bytes(&mut self, path: &str, offset: u64, data: &[u8]) -> Result<(), StorageError> {
        let end = offset
            .checked_add(data_len(data))
            .ok_or(StorageError::NoSpace)?;
        self.grow_if_needed(path, end)?;
        let image = self.pending_image_mut(path)?;
        if image.len() < end_usize(end) {
            image.resize(end_usize(end), 0);
        }
        image[offset_usize(offset)..end_usize(end)].copy_from_slice(data);
        Ok(())
    }

    fn apply_create(
        &mut self,
        path: &str,
        fault: &StorageFault,
    ) -> Result<StorageResult, StorageError> {
        Self::reject_fault(fault, &[])?;
        let path = Self::checked_path(path)?;
        if self.visible(&path).is_some() {
            return Err(StorageError::AlreadyExists { path });
        }
        self.files.insert(
            path.clone(),
            SimFile {
                pending: Some(Vec::new()),
                durable: None,
            },
        );
        self.add_entry(&path);
        Ok(StorageResult::Done)
    }

    fn apply_write(
        &mut self,
        path: &str,
        offset: u64,
        data: &[u8],
        fault: &StorageFault,
    ) -> Result<StorageResult, StorageError> {
        let path = Self::checked_path(path)?;
        self.check_offset(&path, offset)?;
        match fault {
            StorageFault::None => {
                self.store_bytes(&path, offset, data)?;
                Ok(StorageResult::Written {
                    applied: data.len(),
                })
            }
            StorageFault::TornWrite { applied_bytes } => {
                let applied = (*applied_bytes).min(data.len());
                self.store_bytes(&path, offset, &data[..applied])?;
                Ok(StorageResult::Written {
                    applied: data.len(),
                })
            }
            StorageFault::LostWrite => Ok(StorageResult::Written {
                applied: data.len(),
            }),
            StorageFault::Misdirect { victim } => {
                let victim = Self::checked_path(victim)?;
                self.check_offset(&victim, offset)?;
                self.store_bytes(&victim, offset, data)?;
                Ok(StorageResult::Written {
                    applied: data.len(),
                })
            }
            // EIO / NoSpace / Unavailable return before dispatch; the wildcard
            // covers them (and any future variants).
            _ => Err(StorageError::InapplicableFault),
        }
    }

    fn apply_truncate(
        &mut self,
        path: &str,
        len: u64,
        fault: &StorageFault,
    ) -> Result<StorageResult, StorageError> {
        Self::reject_fault(fault, &[])?;
        let path = Self::checked_path(path)?;
        self.pending_len(&path)?;
        self.grow_if_needed(&path, len)?;
        self.pending_image_mut(&path)?.resize(end_usize(len), 0);
        Ok(StorageResult::Done)
    }

    fn apply_rename(
        &mut self,
        from: &str,
        to: &str,
        fault: &StorageFault,
    ) -> Result<StorageResult, StorageError> {
        Self::reject_fault(fault, &[])?;
        let from = Self::checked_path(from)?;
        let to = Self::checked_path(to)?;
        self.pending_len(&from)?;
        if self.visible(&to).is_some() {
            return Err(StorageError::AlreadyExists { path: to });
        }
        let file = self
            .files
            .remove(&from)
            .ok_or_else(|| StorageError::NotFound { path: from.clone() })?;
        self.remove_entry(&from);
        self.files.insert(to.clone(), file);
        self.add_entry(&to);
        Ok(StorageResult::Done)
    }

    fn apply_remove(
        &mut self,
        path: &str,
        fault: &StorageFault,
    ) -> Result<StorageResult, StorageError> {
        Self::reject_fault(fault, &[])?;
        let path = Self::checked_path(path)?;
        let file = self
            .files
            .get_mut(&path)
            .ok_or_else(|| StorageError::NotFound { path: path.clone() })?;
        if file.pending.is_none() {
            return Err(StorageError::NotFound { path: path.clone() });
        }
        file.pending = None;
        self.remove_entry(&path);
        Ok(StorageResult::Done)
    }

    fn apply_sync_file(
        &mut self,
        path: &str,
        fault: &StorageFault,
    ) -> Result<StorageResult, StorageError> {
        Self::reject_fault(fault, &[])?;
        let path = Self::checked_path(path)?;
        let file = self
            .files
            .get_mut(&path)
            .ok_or_else(|| StorageError::NotFound { path: path.clone() })?;
        if file.pending.is_none() && file.durable.is_none() {
            return Err(StorageError::NotFound { path });
        }
        file.durable.clone_from(&file.pending);
        Ok(StorageResult::Done)
    }

    fn apply_sync_dir(
        &mut self,
        dir: &str,
        fault: &StorageFault,
    ) -> Result<StorageResult, StorageError> {
        Self::reject_fault(fault, &[])?;
        if dir_contains_nul(dir) {
            return Err(StorageError::InvalidPath {
                path: dir.to_owned(),
            });
        }
        // Unlike file paths, the root directory `""` is valid here.
        if !self.dirs.contains_key(dir) {
            return Err(StorageError::NotFound {
                path: dir.to_owned(),
            });
        }
        let entry = self
            .dirs
            .get_mut(dir)
            .ok_or_else(|| StorageError::NotFound {
                path: dir.to_owned(),
            })?;
        entry.durable.clone_from(&entry.pending);
        self.collect_garbage(dir);
        Ok(StorageResult::Done)
    }

    /// Flips `mask` into the durable image at `offset`, modeling latent bit
    /// rot (or a misdirected device write) independent of any operation in
    /// flight. The pending image is untouched: a process re-reading its own
    /// recent write still sees clean data until a crash reverts to rot.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] if the device is offline, the file has no
    /// durable image, or `offset` lies past its end.
    pub fn corrupt(&mut self, path: &str, offset: u64, mask: u8) -> Result<(), StorageError> {
        if !self.online {
            return Err(StorageError::Unavailable);
        }
        let path = Self::checked_path(path)?;
        let file = self
            .files
            .get_mut(&path)
            .ok_or_else(|| StorageError::NotFound { path: path.clone() })?;
        let durable = file
            .durable
            .as_mut()
            .ok_or_else(|| StorageError::NotFound { path: path.clone() })?;
        let len = u64::try_from(durable.len()).unwrap_or(u64::MAX);
        if offset >= len {
            return Err(StorageError::CorruptOutOfRange { offset, len });
        }
        durable[offset_usize(offset)] ^= mask;
        Ok(())
    }

    /// Reads the process-visible (pending) image. Helpers for writes that
    /// must observe their own unstaged data use this, not `durable_image`.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] if the device is offline or the file (or its
    /// directory entry) is absent on the pending side.
    pub fn read(&self, path: &str) -> Result<Vec<u8>, StorageError> {
        if !self.online {
            return Err(StorageError::Unavailable);
        }
        let path = Self::checked_path(path)?;
        let (parent, name) = split(&path);
        let listed = self
            .dirs
            .get(&parent)
            .is_some_and(|dir| dir.pending.contains(&name));
        if !listed {
            return Err(StorageError::NotFound { path });
        }
        self.files
            .get(&path)
            .and_then(|file| file.pending.clone())
            .ok_or(StorageError::NotFound { path })
    }

    /// Inspects the crash-surviving (durable) image, if any. Test/driver
    /// inspection only — never process I/O.
    #[must_use]
    pub fn durable_image(&self, path: &str) -> Option<Vec<u8>> {
        self.files.get(path).and_then(|file| file.durable.clone())
    }

    /// Inspects crash-surviving directory entries. Test/driver inspection only.
    #[must_use]
    pub fn durable_entries(&self, dir: &str) -> Vec<String> {
        self.dirs
            .get(dir)
            .map_or_else(Vec::new, |entry| entry.durable.iter().cloned().collect())
    }

    /// Discards all unsynchronized state, modeling a power loss / process
    /// crash: every pending image and entry set reverts to its durable
    /// counterpart. Durable state, capacity, and the online flag survive.
    pub fn crash(&mut self) {
        for file in self.files.values_mut() {
            file.pending.clone_from(&file.durable);
        }
        for dir in self.dirs.values_mut() {
            dir.pending.clone_from(&dir.durable);
        }
    }

    /// Drops file structs whose removal fully persisted: name absent from the
    /// directory's durable entries and no pending image remains. Unsynced
    /// removals (resurrectable by crash) and unlinked-but-present data are
    /// deliberately kept.
    fn collect_garbage(&mut self, dir: &str) {
        let live: BTreeSet<String> = self
            .dirs
            .get(dir)
            .map_or_else(BTreeSet::new, |entry| entry.durable.clone());
        let victims: Vec<String> = self
            .files
            .iter()
            .filter(|(path, file)| {
                let (parent, name) = split(path);
                parent == dir && !live.contains(&name) && file.pending.is_none()
            })
            .map(|(path, _)| (*path).clone())
            .collect();
        for victim in victims {
            self.files.remove(&victim);
        }
    }
    /// Total bytes currently held in pending images (the capacity account).
    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        self.files
            .values()
            .filter_map(|file| file.pending.as_ref())
            .map(|data| u64::try_from(data.len()).unwrap_or(u64::MAX))
            .fold(0u64, u64::saturating_add)
    }

    /// Configured capacity in bytes.
    #[must_use]
    pub const fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    /// The process-visible presence of a path (pending entry plus pending data).
    fn visible(&self, path: &str) -> Option<()> {
        let (parent, name) = split(path);
        let listed = self
            .dirs
            .get(&parent)
            .is_some_and(|dir| dir.pending.contains(&name));
        let present = self
            .files
            .get(path)
            .is_some_and(|file| file.pending.is_some());
        (listed && present).then_some(())
    }

    /// Adds a directory entry on the pending side, creating parents as needed.
    fn add_entry(&mut self, path: &str) {
        let (parent, name) = split(path);
        self.dirs.entry(parent).or_default().pending.insert(name);
    }

    /// Removes a directory entry on the pending side, if present.
    fn remove_entry(&mut self, path: &str) {
        let (parent, name) = split(path);
        if let Some(dir) = self.dirs.get_mut(&parent) {
            dir.pending.remove(&name);
        }
    }

    /// Reserves capacity for a pending image reaching `want` bytes.
    fn grow_if_needed(&self, path: &str, want: u64) -> Result<(), StorageError> {
        let current = self
            .files
            .get(path)
            .and_then(|file| file.pending.as_ref())
            .map_or(0, |data| u64::try_from(data.len()).unwrap_or(u64::MAX));
        if want <= current {
            return Ok(());
        }
        let growth = want - current;
        let used = self.used_bytes();
        if used.saturating_add(growth) > self.capacity_bytes {
            return Err(StorageError::NoSpace);
        }
        Ok(())
    }

    /// Rejects any fault outside the applicable set for simple metadata ops.
    fn reject_fault(fault: &StorageFault, allowed: &[StorageFault]) -> Result<(), StorageError> {
        if *fault == StorageFault::None || allowed.contains(fault) {
            Ok(())
        } else {
            Err(StorageError::InapplicableFault)
        }
    }

    /// Validates path shape (non-empty, no NUL).
    fn checked_path(path: &str) -> Result<String, StorageError> {
        if path.is_empty() || dir_contains_nul(path) {
            return Err(StorageError::InvalidPath {
                path: path.to_owned(),
            });
        }
        Ok(path.to_owned())
    }
}

/// Splits a path into parent directory and final name (`""` parents the root).
fn split(path: &str) -> (String, String) {
    match path.rsplit_once('/') {
        Some((parent, name)) => (parent.to_owned(), name.to_owned()),
        None => (String::new(), path.to_owned()),
    }
}

/// Whether the string contains a NUL byte.
fn dir_contains_nul(text: &str) -> bool {
    text.bytes().any(|byte| byte == 0)
}

/// Length of a byte buffer as `u64` (saturating only past addressable memory,
/// which cannot happen for a live allocation).
fn data_len(data: &[u8]) -> u64 {
    u64::try_from(data.len()).unwrap_or(u64::MAX)
}

/// Converts a validated in-range `u64` length to `usize` for indexing.
fn end_usize(len: u64) -> usize {
    usize::try_from(len).unwrap_or(usize::MAX)
}

/// Converts a validated in-range `u64` offset to `usize` for indexing.
fn offset_usize(offset: u64) -> usize {
    usize::try_from(offset).unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::{fixture, rstest};

    #[fixture]
    fn disk() -> VirtualDisk {
        VirtualDisk::new(1 << 20)
    }

    fn create_write(disk: &mut VirtualDisk, path: &str, data: &[u8]) {
        disk.apply(
            &StorageOp::Create {
                path: path.to_owned(),
            },
            &StorageFault::None,
        )
        .expect("create");
        disk.apply(
            &StorageOp::Write {
                path: path.to_owned(),
                offset: 0,
                data: data.to_vec(),
            },
            &StorageFault::None,
        )
        .expect("write");
    }

    fn sync_file_dir(disk: &mut VirtualDisk, path: &str) {
        let (parent, _) = split(path);
        disk.apply(
            &StorageOp::SyncFile {
                path: path.to_owned(),
            },
            &StorageFault::None,
        )
        .expect("sync file");
        disk.apply(&StorageOp::SyncDir { dir: parent }, &StorageFault::None)
            .expect("sync dir");
    }

    #[test]
    fn write_is_visible_but_not_durable_until_sync() {
        let mut disk = disk();
        create_write(&mut disk, "w/seg", b"data");
        assert_eq!(disk.read("w/seg").expect("pending visible"), b"data");
        assert_eq!(disk.durable_image("w/seg"), None);
        disk.crash();
        assert_eq!(
            disk.read("w/seg"),
            Err(StorageError::NotFound {
                path: "w/seg".to_owned()
            })
        );
        assert_eq!(disk.durable_image("w/seg"), None);
    }

    #[rstest]
    #[case(false, false, false, false)]
    #[case(true, false, false, true)]
    #[case(false, true, false, false)]
    #[case(true, true, true, true)]
    fn crash_boundaries(
        mut disk: VirtualDisk,
        #[case] sync_data: bool,
        #[case] sync_dir: bool,
        #[case] expect_readable: bool,
        #[case] expect_durable_data: bool,
    ) {
        create_write(&mut disk, "w/seg", b"payload");
        if sync_data {
            disk.apply(
                &StorageOp::SyncFile {
                    path: "w/seg".to_owned(),
                },
                &StorageFault::None,
            )
            .expect("sync file");
        }
        if sync_dir {
            disk.apply(
                &StorageOp::SyncDir {
                    dir: "w".to_owned(),
                },
                &StorageFault::None,
            )
            .expect("sync dir");
        }
        disk.crash();
        assert_eq!(disk.read("w/seg").is_ok(), expect_readable);
        assert_eq!(disk.durable_image("w/seg").is_some(), expect_durable_data);
    }

    #[test]
    fn safe_publication_pattern_survives_crash() {
        let mut disk = disk();
        // write temp → sync temp → rename → sync parent dir.
        create_write(&mut disk, "w/tmp", b"checkpoint");
        disk.apply(
            &StorageOp::SyncFile {
                path: "w/tmp".to_owned(),
            },
            &StorageFault::None,
        )
        .expect("sync temp");
        disk.apply(
            &StorageOp::Rename {
                from: "w/tmp".to_owned(),
                to: "w/final".to_owned(),
            },
            &StorageFault::None,
        )
        .expect("rename");
        disk.apply(
            &StorageOp::SyncDir {
                dir: "w".to_owned(),
            },
            &StorageFault::None,
        )
        .expect("sync parent");
        disk.crash();
        assert_eq!(disk.read("w/final").expect("published"), b"checkpoint");
    }

    #[test]
    fn rename_without_parent_sync_loses_the_name() {
        let mut disk = disk();
        create_write(&mut disk, "w/tmp", b"checkpoint");
        disk.apply(
            &StorageOp::SyncFile {
                path: "w/tmp".to_owned(),
            },
            &StorageFault::None,
        )
        .expect("sync temp");
        disk.apply(
            &StorageOp::Rename {
                from: "w/tmp".to_owned(),
                to: "w/final".to_owned(),
            },
            &StorageFault::None,
        )
        .expect("rename");
        disk.crash();
        // The namespace change never persisted: the classic missing-fsync bug.
        assert!(disk.read("w/final").is_err());
    }

    #[test]
    fn synced_removal_frees_while_unsynced_removal_resurrects() {
        let mut disk = disk();
        create_write(&mut disk, "f", b"v1");
        sync_file_dir(&mut disk, "f");
        disk.apply(
            &StorageOp::Remove {
                path: "f".to_owned(),
            },
            &StorageFault::None,
        )
        .expect("remove");
        disk.crash();
        assert_eq!(disk.read("f").expect("resurrected"), b"v1");

        disk.apply(
            &StorageOp::Remove {
                path: "f".to_owned(),
            },
            &StorageFault::None,
        )
        .expect("remove again");
        disk.apply(
            &StorageOp::SyncDir { dir: String::new() },
            &StorageFault::None,
        )
        .expect("sync root");
        disk.crash();
        assert!(disk.read("f").is_err());
        assert_eq!(disk.durable_image("f"), None);
    }

    #[rstest]
    #[case(StorageFault::EIO, StorageError::EIO)]
    #[case(StorageFault::NoSpace, StorageError::NoSpace)]
    #[case(StorageFault::Unavailable, StorageError::Unavailable)]
    fn terminal_faults_fail_without_change(
        mut disk: VirtualDisk,
        #[case] fault: StorageFault,
        #[case] error: StorageError,
    ) {
        create_write(&mut disk, "f", b"v1");
        let before = disk.read("f").expect("baseline");
        let result = disk.apply(
            &StorageOp::Write {
                path: "f".to_owned(),
                offset: 0,
                data: b"v2".to_vec(),
            },
            &fault,
        );
        assert_eq!(result, Err(error));
        assert_eq!(disk.read("f").expect("unchanged"), before);
    }

    #[test]
    fn torn_write_applies_a_prefix_but_reports_full() {
        let mut disk = disk();
        create_write(&mut disk, "f", b"..........");
        let result = disk
            .apply(
                &StorageOp::Write {
                    path: "f".to_owned(),
                    offset: 0,
                    data: b"0123456789".to_vec(),
                },
                &StorageFault::TornWrite { applied_bytes: 4 },
            )
            .expect("torn write succeeds");
        assert_eq!(result, StorageResult::Written { applied: 10 });
        assert_eq!(disk.read("f").expect("read"), b"0123......");
    }

    #[test]
    fn lost_write_changes_nothing_but_reports_full() {
        let mut disk = disk();
        create_write(&mut disk, "f", b"v1");
        let result = disk
            .apply(
                &StorageOp::Write {
                    path: "f".to_owned(),
                    offset: 0,
                    data: b"v2".to_vec(),
                },
                &StorageFault::LostWrite,
            )
            .expect("lost write succeeds");
        assert_eq!(result, StorageResult::Written { applied: 2 });
        assert_eq!(disk.read("f").expect("read"), b"v1");
    }

    #[test]
    fn misdirected_write_hits_victim_only() {
        let mut disk = disk();
        create_write(&mut disk, "a", b"aaaa");
        create_write(&mut disk, "b", b"bbbb");
        disk.apply(
            &StorageOp::Write {
                path: "a".to_owned(),
                offset: 0,
                data: b"XX".to_vec(),
            },
            &StorageFault::Misdirect {
                victim: "b".to_owned(),
            },
        )
        .expect("misdirect succeeds");
        assert_eq!(disk.read("a").expect("target untouched"), b"aaaa");
        assert_eq!(disk.read("b").expect("victim hit"), b"XXbb");
    }

    #[test]
    fn corruption_hits_durable_only_until_crash_exposes_it() {
        let mut disk = disk();
        create_write(&mut disk, "f", b"clean");
        sync_file_dir(&mut disk, "f");
        disk.corrupt("f", 0, 0xFF).expect("corrupt");
        // Process-visible image is untouched; durable image rotted.
        assert_eq!(disk.read("f").expect("pending clean"), b"clean");
        assert_ne!(disk.durable_image("f").expect("durable"), b"clean");
        disk.crash();
        assert_ne!(disk.read("f").expect("rot exposed"), b"clean");
        assert_eq!(
            disk.corrupt("f", 99, 0xFF),
            Err(StorageError::CorruptOutOfRange { offset: 99, len: 5 })
        );
    }

    #[test]
    fn natural_enospc_tracks_pending_bytes() {
        let mut disk = VirtualDisk::new(10);
        create_write(&mut disk, "f", b"123456");
        assert_eq!(disk.used_bytes(), 6);
        assert_eq!(
            disk.apply(
                &StorageOp::Write {
                    path: "f".to_owned(),
                    offset: 6,
                    data: b"123456".to_vec(),
                },
                &StorageFault::None,
            ),
            Err(StorageError::NoSpace)
        );
        disk.apply(
            &StorageOp::Truncate {
                path: "f".to_owned(),
                len: 2,
            },
            &StorageFault::None,
        )
        .expect("truncate frees");
        assert_eq!(disk.used_bytes(), 2);
    }

    #[test]
    fn offline_device_rejects_everything_but_survives_crash() {
        let mut disk = disk();
        create_write(&mut disk, "f", b"v1");
        sync_file_dir(&mut disk, "f");
        disk.set_online(false);
        assert!(!disk.is_online());
        assert_eq!(disk.read("f"), Err(StorageError::Unavailable));
        assert!(matches!(
            disk.apply(
                &StorageOp::SyncFile {
                    path: "f".to_owned()
                },
                &StorageFault::None,
            ),
            Err(StorageError::Unavailable)
        ));
        disk.crash();
        assert!(!disk.is_online());
        disk.set_online(true);
        // Synced state survives the crash even while offline; unsynced would not.
        assert_eq!(disk.read("f").expect("durable survives"), b"v1");
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let mut disk = disk();
        assert!(matches!(
            disk.apply(
                &StorageOp::Create {
                    path: String::new()
                },
                &StorageFault::None,
            ),
            Err(StorageError::InvalidPath { .. })
        ));
        create_write(&mut disk, "f", b"v1");
        assert!(matches!(
            disk.apply(
                &StorageOp::Write {
                    path: "f".to_owned(),
                    offset: 99,
                    data: b"x".to_vec(),
                },
                &StorageFault::None,
            ),
            Err(StorageError::InvalidOffset { .. })
        ));
        assert!(matches!(
            disk.apply(
                &StorageOp::SyncFile {
                    path: "f".to_owned(),
                },
                &StorageFault::TornWrite { applied_bytes: 1 },
            ),
            Err(StorageError::InapplicableFault)
        ));
        assert!(matches!(
            disk.apply(
                &StorageOp::SyncDir {
                    dir: "missing".to_owned(),
                },
                &StorageFault::None,
            ),
            Err(StorageError::NotFound { .. })
        ));
    }
}
