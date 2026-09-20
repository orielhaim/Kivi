//! Fabric journal: the durable id→record map.
//!
//! The journal is the recovery and GC truth for where a fabric id's
//! bytes can be rebuilt from. The durability lane appends one entry per
//! sealed payload under the same barrier as the record bytes
//! (durable-before-reference); loads verify framing plus CRC32C and
//! truncate torn tails, failing loudly on non-tail corruption. Journal
//! files are worker-local (single writer); ids never cross workers.

use std::path::Path;

use kivi_types::TabletId;

use crate::error::MemoryError;
use crate::nvme::{NvmeOptions, NvmeProvider};
use crate::{BehaviorClass, OffcoreImport};

/// Journal magic: `KVFJ` (Kivi fabric journal).
const JOURNAL_MAGIC: u32 = 0x4A_46_56_4B;
/// Journal format version. Old versions fail loudly on load.
const JOURNAL_VERSION: u16 = 1;

/// Which device file a journal record names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JournalFile {
    /// Durable admission record (durability-lane single writer).
    Material,
    /// Optimization demotion copy (offcore-lane single writer).
    Demotion,
}

/// One durable id→record mapping: the recovery and GC truth for where a
/// fabric id's bytes can be rebuilt from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournalEntry {
    /// Fabric-scoped object id.
    pub fabric_id: u64,
    /// Owning tablet (observability, per-tablet footprint).
    pub tablet: TabletId,
    /// Logical key (observability: why is this in `NVMe`?).
    pub key: Vec<u8>,
    /// Logical version sealed at.
    pub version: u64,
    /// Which file holds the record.
    pub file: JournalFile,
    /// Device offset of the record.
    pub offset: u64,
    /// Stored payload length.
    pub len: u64,
    /// CRC32C of the payload.
    pub checksum: u32,
}

/// Encodes one journal entry: magic + version + framing + CRC32C.
fn encode_entry(entry: &JournalEntry) -> Vec<u8> {
    let mut body = Vec::with_capacity(8 + 8 + 4 + entry.key.len() + 8 + 1 + 8 + 8 + 4);
    body.extend_from_slice(&entry.fabric_id.to_le_bytes());
    body.extend_from_slice(&entry.tablet.as_u64().to_le_bytes());
    let key_len = u32::try_from(entry.key.len()).unwrap_or(u32::MAX);
    body.extend_from_slice(&key_len.to_le_bytes());
    body.extend_from_slice(&entry.key);
    body.extend_from_slice(&entry.version.to_le_bytes());
    body.push(match entry.file {
        JournalFile::Material => 0,
        JournalFile::Demotion => 1,
    });
    body.extend_from_slice(&entry.offset.to_le_bytes());
    body.extend_from_slice(&entry.len.to_le_bytes());
    body.extend_from_slice(&entry.checksum.to_le_bytes());
    let mut out = Vec::with_capacity(4 + 2 + 4 + body.len() + 4);
    out.extend_from_slice(&JOURNAL_MAGIC.to_le_bytes());
    out.extend_from_slice(&JOURNAL_VERSION.to_le_bytes());
    out.extend_from_slice(&(body.len() as u64).to_le_bytes());
    out.extend_from_slice(&body);
    out.extend_from_slice(&crc32c::crc32c(&body).to_le_bytes());
    out
}

/// Decodes one journal entry, verifying framing and CRC.
fn decode_entry(bytes: &[u8]) -> Option<(JournalEntry, usize)> {
    if bytes.len() < 14 {
        return None;
    }
    if u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) != JOURNAL_MAGIC {
        return None;
    }
    if u16::from_le_bytes([bytes[4], bytes[5]]) != JOURNAL_VERSION {
        return None;
    }
    let frame_len = usize::try_from(u64::from_le_bytes([
        bytes[6], bytes[7], bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13],
    ]))
    .unwrap_or(usize::MAX);
    if bytes.len() < 14 + frame_len + 4 {
        return None;
    }
    let body = &bytes[14..14 + frame_len];
    if crc32c::crc32c(body)
        != u32::from_le_bytes([
            bytes[14 + frame_len],
            bytes[14 + frame_len + 1],
            bytes[14 + frame_len + 2],
            bytes[14 + frame_len + 3],
        ])
    {
        return None;
    }
    let mut at = 0usize;
    let take = |at: &mut usize, n: usize| -> Option<Vec<u8>> {
        if body.len() < *at + n {
            return None;
        }
        let out = body[*at..*at + n].to_vec();
        *at += n;
        Some(out)
    };
    let fabric_id = u64::from_le_bytes(take(&mut at, 8)?.try_into().ok()?);
    let tablet = TabletId::from_u64(u64::from_le_bytes(take(&mut at, 8)?.try_into().ok()?));
    let key_len = u32::from_le_bytes(take(&mut at, 4)?.try_into().ok()?) as usize;
    let key = take(&mut at, key_len)?;
    let version = u64::from_le_bytes(take(&mut at, 8)?.try_into().ok()?);
    let file = match take(&mut at, 1)?[0] {
        0 => JournalFile::Material,
        1 => JournalFile::Demotion,
        _ => return None,
    };
    let offset = u64::from_le_bytes(take(&mut at, 8)?.try_into().ok()?);
    let payload_len = u64::from_le_bytes(take(&mut at, 8)?.try_into().ok()?);
    let checksum = u32::from_le_bytes(take(&mut at, 4)?.try_into().ok()?);
    if at != body.len() {
        return None;
    }
    Some((
        JournalEntry {
            fabric_id,
            tablet,
            key,
            version,
            file,
            offset,
            len: payload_len,
            checksum,
        },
        14 + frame_len + 4,
    ))
}

/// Appends journal entries with one barrier (durability-lane seal path).
/// Torn tails truncate on load; corrupt non-tail records fail the load
/// loudly (never partial trust).
///
/// # Errors
///
/// Returns [`std::io::Error`] on filesystem failure.
pub fn journal_append_batch(path: &Path, entries: &[JournalEntry]) -> std::io::Result<()> {
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    for entry in entries {
        file.write_all(&encode_entry(entry))?;
    }
    file.sync_all()?;
    if let Some(parent) = path.parent() {
        // Best-effort directory fsync: opening a directory as a file
        // fails on some platforms (notably Windows), and durability
        // here rests on the file sync above, never on this.
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

/// Loads and verifies the journal, truncating a torn tail. Corrupt
/// non-tail records fail the load loudly (never partial trust).
/// A missing journal is not an error (fresh worker).
///
/// # Errors
///
/// Returns [`std::io::Error`] on filesystem failure; corruption
/// surfaces as `InvalidData`.
pub fn journal_load(path: &Path) -> std::io::Result<Vec<JournalEntry>> {
    use std::io::{Error, ErrorKind};
    let bytes = match std::fs::read(path) {
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
        Ok(bytes) => bytes,
    };
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < bytes.len() {
        if let Some((entry, used)) = decode_entry(&bytes[at..]) {
            out.push(entry);
            at += used;
            continue;
        }
        // Torn tail only when the remainder cannot even frame a
        // record; anything else is corruption, fail loudly.
        if bytes.len() - at < 14 {
            if let Ok(file) = std::fs::OpenOptions::new().write(true).open(path) {
                let _ = file.set_len(at as u64);
            }
            break;
        }
        return Err(Error::new(ErrorKind::InvalidData, "fabric journal corrupt"));
    }
    Ok(out)
}

/// Rewrites the journal with exactly `live` entries (checkpoint-time
/// compaction), atomically replacing the old file.
///
/// # Errors
///
/// Returns a description on filesystem failure.
pub fn journal_compact(path: &Path, live: &[JournalEntry]) -> Result<(), String> {
    let tmp = path.with_extension("journal.tmp");
    {
        use std::io::Write as _;
        let mut file = std::fs::File::create(&tmp).map_err(|error| error.to_string())?;
        for entry in live {
            file.write_all(&encode_entry(entry))
                .map_err(|error| error.to_string())?;
        }
        file.sync_all().map_err(|error| error.to_string())?;
    }
    std::fs::rename(&tmp, path).map_err(|error| error.to_string())?;
    if let Some(parent) = path.parent()
        && let Ok(dir) = std::fs::File::open(parent)
    {
        let _ = dir.sync_all();
    }
    Ok(())
}

/// Converts a seal locator into a recovery import (cold: off-core
/// residence, hydrate lazily on first touch).
#[must_use]
pub fn import_from_entry(entry: &JournalEntry) -> OffcoreImport {
    OffcoreImport {
        id: entry.fabric_id,
        version: entry.version,
        offset: entry.offset,
        len: entry.len,
        checksum: entry.checksum,
        intent: crate::MaterializationIntent::cold(),
        class: BehaviorClass::ColdCandidate,
    }
}

/// Opens (creating) the durable record provider for a worker.
///
/// # Errors
///
/// Returns [`MemoryError::Io`] when the directory or data file cannot be
/// opened, and [`MemoryError::Unsupported`] on newer formats.
pub fn open_material_provider(dir: &Path) -> Result<NvmeProvider, MemoryError> {
    NvmeProvider::open(dir, NvmeOptions::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn journal_roundtrip_with_torn_tail() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("fabric.journal");
        let entries = vec![
            JournalEntry {
                fabric_id: 7,
                tablet: TabletId::from_u64(3),
                key: b"user:1".to_vec(),
                version: 2,
                file: JournalFile::Material,
                offset: 0,
                len: 100,
                checksum: 0xDEAD_BEEF,
            },
            JournalEntry {
                fabric_id: 8,
                tablet: TabletId::from_u64(3),
                key: b"user:2".to_vec(),
                version: 1,
                file: JournalFile::Demotion,
                offset: 128,
                len: 50,
                checksum: 0x1234_5678,
            },
        ];
        journal_append_batch(&path, &entries).expect("append");
        let loaded = journal_load(&path).expect("load");
        assert_eq!(loaded, entries);
        // Torn tail truncates instead of failing.
        {
            use std::io::Write as _;
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .expect("open");
            file.write_all(b"partial").expect("write");
        }
        let loaded = journal_load(&path).expect("torn tail truncates");
        assert_eq!(loaded, entries);
    }

    #[test]
    fn journal_compact_keeps_only_live() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("fabric.journal");
        let live = JournalEntry {
            fabric_id: 1,
            tablet: TabletId::from_u64(1),
            key: b"k".to_vec(),
            version: 1,
            file: JournalFile::Material,
            offset: 0,
            len: 10,
            checksum: 1,
        };
        let dead = JournalEntry {
            fabric_id: 2,
            ..live.clone()
        };
        journal_append_batch(&path, &[live.clone(), dead]).expect("append");
        journal_compact(&path, std::slice::from_ref(&live)).expect("compact");
        assert_eq!(journal_load(&path).expect("load"), vec![live]);
    }

    #[test]
    fn corrupt_non_tail_record_fails_loudly() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("fabric.journal");
        let live = JournalEntry {
            fabric_id: 1,
            tablet: TabletId::from_u64(1),
            key: b"k".to_vec(),
            version: 1,
            file: JournalFile::Material,
            offset: 0,
            len: 10,
            checksum: 1,
        };
        journal_append_batch(&path, &[live]).expect("append");
        // Corrupt the first record body (not a torn tail).
        let mut raw = std::fs::read(&path).expect("read");
        raw[20] ^= 0xFF;
        std::fs::write(&path, raw).expect("clobber");
        let error = journal_load(&path).expect_err("corrupt record fails");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }
}
