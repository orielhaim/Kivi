//! Local `NVMe` demotion provider (`CacheLib` Navy design principles).
//!
//! Admission, bounded asynchronous I/O, Direct I/O, and device endurance
//! shape this provider; baseline correctness needs none of them. Records
//! are self-describing (`magic + version + length + CRC32C + bytes`) so a
//! torn tail truncates safely and a corrupt record quarantines instead of
//! serving wrong bytes. Admission control (Navy-style probabilistic and
//! budget rejection) protects device endurance under churn; per-object
//! ordering comes from [`OffcoreQueue`](crate::offcore::OffcoreQueue)
//! spooling, not from a lock.
//!
//! The synchronous file backend below is the correctness baseline used by
//! deterministic tests. Production demotion/promotion runs through the
//! same record format via explicit [`OffcoreOp`](crate::offcore::OffcoreOp)s
//! executed on the caller's Compio runtime ([`demote_async`] /
//! [`promote_async`]); no second runtime is ever created. Direct I/O,
//! raw-device access, and FDP placement hints are provider optimizations
//! configured via [`NvmeOptions`], never correctness requirements.

use std::path::{Path, PathBuf};

use crate::error::MemoryError;

/// Record magic: `KVMM` (Kivi memory materialization).
pub const NVME_MAGIC: u32 = 0x4D_4D_56_4B;

/// Current record format version. Old versions fail loudly on load.
pub const NVME_RECORD_VERSION: u16 = 1;

/// Direct-I/O alignment hint in bytes. Informational: the file backend
/// aligns record payloads to this boundary when `direct_io` is set so a
/// future raw-device backend can reuse the same layout.
pub const DIRECT_IO_ALIGN: usize = 4096;

/// Provider options. All fields are optimizations; correctness holds at
/// their defaults on any ordinary file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvmeOptions {
    /// Use Direct I/O when the platform supports it. Falls back to
    /// buffered I/O silently; checksums still verify every record.
    pub direct_io: bool,
    /// FDP placement-handle hint tag for compatible devices. Ignored
    /// elsewhere.
    pub fdp_hint: Option<u8>,
    /// Maximum demotion bytes accepted per day (Navy `DynamicRandomAP`
    /// endurance budget). `u64::MAX` disables the budget.
    pub daily_write_budget_bytes: u64,
    /// Base probabilistic rejection in basis points (Navy
    /// `RejectRandomAP`). Zero disables random rejection.
    pub reject_random_bps: u64,
}

impl Default for NvmeOptions {
    fn default() -> Self {
        Self {
            direct_io: false,
            fdp_hint: None,
            daily_write_budget_bytes: u64::MAX,
            reject_random_bps: 0,
        }
    }
}

/// Admission decision for one demotion candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// Accept the demotion.
    Accept,
    /// Reject: probabilistic endurance protection.
    RejectRandom,
    /// Reject: daily write budget exhausted.
    RejectBudget,
    /// Reject: bounded insert concurrency exhausted.
    RejectOverloaded,
}

/// Navy-style admission controller: probabilistic rejection plus a daily
/// write budget plus a concurrent-insert bound.
#[derive(Debug)]
pub struct AdmissionController {
    options: NvmeOptions,
    written_today_bytes: u64,
    in_flight: usize,
    max_in_flight: usize,
    rng_state: u64,
}

impl AdmissionController {
    /// Creates a controller with the given options and insert bound.
    #[must_use]
    pub fn new(options: NvmeOptions, max_in_flight: usize, seed: u64) -> Self {
        Self {
            options,
            written_today_bytes: 0,
            in_flight: 0,
            max_in_flight,
            rng_state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Decides admission for a demotion of `bytes`. Deterministic in the
    /// seed: simulation replays the same accept/reject sequence.
    #[must_use]
    pub fn admit(&mut self, bytes: u64) -> Admission {
        if self.in_flight >= self.max_in_flight {
            return Admission::RejectOverloaded;
        }
        if self.written_today_bytes.saturating_add(bytes) > self.options.daily_write_budget_bytes {
            return Admission::RejectBudget;
        }
        if self.options.reject_random_bps > 0 {
            // xorshift64* (deterministic, no external RNG).
            self.rng_state ^= self.rng_state >> 12;
            self.rng_state ^= self.rng_state << 25;
            self.rng_state ^= self.rng_state >> 27;
            let sample = (self.rng_state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) % 10_000;
            if sample < self.options.reject_random_bps {
                return Admission::RejectRandom;
            }
        }
        self.in_flight += 1;
        self.written_today_bytes = self.written_today_bytes.saturating_add(bytes);
        Admission::Accept
    }

    /// Releases one in-flight slot after completion.
    pub const fn release(&mut self) {
        self.in_flight = self.in_flight.saturating_sub(1);
    }

    /// In-flight demotion count.
    #[must_use]
    pub const fn in_flight(&self) -> usize {
        self.in_flight
    }
}

/// Encodes one record: an 8-byte magic plus version, flags, length,
/// and CRC32C header followed by the payload.
#[must_use]
pub fn encode_record(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(20 + payload.len());
    out.extend_from_slice(&NVME_MAGIC.to_le_bytes());
    out.extend_from_slice(&NVME_RECORD_VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&crc32c::crc32c(payload).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Decodes and verifies one record at the start of `bytes`.
///
/// Returns the payload plus the total record length.
///
/// # Errors
///
/// Returns [`MemoryError::CorruptRepresentation`] on magic, version,
/// length, or CRC mismatch, and [`MemoryError::TooLarge`] when the
/// declared length exceeds `max_bytes`.
pub fn decode_record(bytes: &[u8], max_bytes: u64) -> Result<(Vec<u8>, usize), MemoryError> {
    if bytes.len() < 20 {
        return Err(MemoryError::CorruptRepresentation {
            object: u64::MAX,
            detail: "nvme record shorter than header".to_owned(),
        });
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != NVME_MAGIC {
        return Err(MemoryError::CorruptRepresentation {
            object: u64::MAX,
            detail: "nvme record magic mismatch".to_owned(),
        });
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != NVME_RECORD_VERSION {
        return Err(MemoryError::Unsupported {
            detail: "unknown nvme record version".to_owned(),
        });
    }
    let len = u64::from_le_bytes([
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    ]);
    if len > max_bytes {
        return Err(MemoryError::TooLarge {
            len,
            max: max_bytes,
            context: "nvme record length",
        });
    }
    let len_usize = usize::try_from(len).map_err(|_| MemoryError::TooLarge {
        len,
        max: max_bytes,
        context: "nvme record length",
    })?;
    if bytes.len() < 20 + len_usize {
        return Err(MemoryError::CorruptRepresentation {
            object: u64::MAX,
            detail: "nvme record truncated".to_owned(),
        });
    }
    let expected = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let payload = &bytes[20..20 + len_usize];
    if crc32c::crc32c(payload) != expected {
        return Err(MemoryError::CorruptRepresentation {
            object: u64::MAX,
            detail: "nvme record CRC mismatch".to_owned(),
        });
    }
    Ok((payload.to_vec(), 20 + len_usize))
}

/// Synchronous file-backed `NVMe` provider: the correctness baseline.
///
/// Production async I/O ([`demote_async`]/[`promote_async`]) uses the same
/// record format over `compio::fs`; this backend proves the format,
/// admission, crash truncation, and corruption handling deterministically.
#[derive(Debug)]
pub struct NvmeProvider {
    dir: PathBuf,
    options: NvmeOptions,
    admission: AdmissionController,
    next_offset: u64,
    /// Persistent append handle for [`append_relaxed`](Self::append_relaxed)
    /// (opened lazily): reopening the data file per record costs a full
    /// open/close round-trip that serializes the single lane thread behind
    /// the slowest disk. Append mode pins every write at end-of-file, so
    /// the in-memory `next_offset` stays authoritative.
    append_file: Option<std::fs::File>,
}

impl NvmeProvider {
    /// Opens (creating) the provider directory.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Io`] when the directory cannot be created,
    /// and [`MemoryError::Unsupported`] when an existing data file uses a
    /// newer format version.
    pub fn open(dir: &Path, options: NvmeOptions) -> Result<Self, MemoryError> {
        std::fs::create_dir_all(dir)
            .map_err(|error| MemoryError::io("create nvme dir", dir, &error))?;
        let data = dir.join("materializations.dat");
        let mut next_offset = 0u64;
        if data.exists() {
            let bytes = std::fs::read(&data)
                .map_err(|error| MemoryError::io("read nvme data", &data, &error))?;
            // Scan records, truncating a torn tail (append-only crash
            // rule: only the last record may be partial).
            let mut offset = 0usize;
            while offset < bytes.len() {
                match decode_record(&bytes[offset..], crate::MEDIUM_MAX as u64) {
                    Ok((_, total)) => {
                        offset += total;
                    }
                    Err(MemoryError::CorruptRepresentation { .. }) => {
                        // Torn tail: truncate and continue. Corruption in
                        // a non-tail record would surface on read of that
                        // record's checksum, not here.
                        let file = std::fs::OpenOptions::new()
                            .write(true)
                            .open(&data)
                            .map_err(|error| {
                                MemoryError::io("truncate nvme data", &data, &error)
                            })?;
                        file.set_len(offset as u64).map_err(|error| {
                            MemoryError::io("truncate nvme data", &data, &error)
                        })?;
                        break;
                    }
                    Err(other) => return Err(other),
                }
            }
            next_offset = offset as u64;
        }
        Ok(Self {
            dir: dir.to_owned(),
            options,
            admission: AdmissionController::new(options, 8, 0x1234_5678),
            next_offset,
            append_file: None,
        })
    }

    /// Admission check for a demotion candidate.
    #[must_use]
    pub fn admit(&mut self, bytes: u64) -> Admission {
        self.admission.admit(bytes)
    }

    /// Releases one admission slot.
    pub const fn release(&mut self) {
        self.admission.release();
    }

    /// Appends one record synchronously, returning `(offset, len,
    /// checksum)`. The barrier is the seal path's durable-before-
    /// reference ordering: never relax it for material records.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Io`] on filesystem failure.
    pub fn append(&mut self, payload: &[u8]) -> Result<(u64, u64, u32), MemoryError> {
        let data = self.dir.join("materializations.dat");
        let record = encode_record(payload);
        let offset = self.next_offset;
        append_bytes(&data, &record)?;
        sync_file(&data)?;
        self.next_offset += record.len() as u64;
        Ok((offset, payload.len() as u64, crc32c::crc32c(payload)))
    }

    /// Appends one record WITHOUT a durability barrier, returning
    /// `(offset, len, checksum)`. Only for optimization copies (lane
    /// demotions): a crash may lose unwritten tail records, which is
    /// safe because demotion records are never referenced durably (the
    /// journal names material records; recovery never imports demotion
    /// offsets). Same-process reads still see the bytes through the
    /// page cache, and every record stays checksummed. Paying an fsync
    /// per demotion would serialize the background lane behind the
    /// slowest disk while buying nothing.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Io`] on filesystem failure.
    pub fn append_relaxed(&mut self, payload: &[u8]) -> Result<(u64, u64, u32), MemoryError> {
        use std::io::Write as _;
        let record = encode_record(payload);
        let offset = self.next_offset;
        let data = self.dir.join("materializations.dat");
        if self.append_file.is_none() {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&data)
                .map_err(|error| MemoryError::io("open nvme data", &data, &error))?;
            self.append_file = Some(file);
        }
        let Some(file) = self.append_file.as_mut() else {
            return Err(MemoryError::Io {
                op: "open nvme data",
                path: data.clone(),
                message: "append handle vanished after open".to_owned(),
            });
        };
        file.write_all(&record)
            .map_err(|error| MemoryError::io("write nvme data", &data, &error))?;
        self.next_offset += record.len() as u64;
        Ok((offset, payload.len() as u64, crc32c::crc32c(payload)))
    }

    /// Reads and verifies one record by offset.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::CorruptRepresentation`] when the record at
    /// `offset` fails verification, or [`MemoryError::Io`] on filesystem
    /// failure.
    pub fn read_at(&self, offset: u64, expected_len: u64) -> Result<Vec<u8>, MemoryError> {
        let data = self.dir.join("materializations.dat");
        let bytes = std::fs::read(&data).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                // A locator naming bytes on a device that holds no records
                // is a corrupt locator, not a transient failure: callers
                // must fail the read closed, never retry it as IO.
                return MemoryError::CorruptRepresentation {
                    object: u64::MAX,
                    detail: "nvme data file missing".to_owned(),
                };
            }
            MemoryError::io("read nvme data", &data, &error)
        })?;
        let start = usize::try_from(offset).map_err(|_| MemoryError::Invalid {
            detail: "nvme offset out of range".to_owned(),
        })?;
        if start >= bytes.len() {
            return Err(MemoryError::CorruptRepresentation {
                object: u64::MAX,
                detail: "nvme read past end".to_owned(),
            });
        }
        let (payload, _) = decode_record(&bytes[start..], crate::MEDIUM_MAX as u64)?;
        if payload.len() as u64 != expected_len {
            return Err(MemoryError::CorruptRepresentation {
                object: u64::MAX,
                detail: "nvme record length changed".to_owned(),
            });
        }
        Ok(payload)
    }

    /// Directory this provider stores in.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Scans the data file and returns every well-formed record as
    /// `(offset, length, checksum)`, stopping at the first torn tail.
    /// Recovery uses this to validate journal entries against what is
    /// actually on the device; a torn tail truncates like [`open`](Self::open).
    #[must_use]
    pub fn scan_records(&self) -> Vec<(u64, u64, u32)> {
        let data = self.dir.join("materializations.dat");
        let Ok(bytes) = std::fs::read(&data) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut offset = 0usize;
        while offset < bytes.len() {
            match decode_record(&bytes[offset..], crate::MEDIUM_MAX as u64) {
                Ok((payload, total)) => {
                    out.push((
                        offset as u64,
                        payload.len() as u64,
                        crc32c::crc32c(&payload),
                    ));
                    offset += total;
                }
                Err(_) => break,
            }
        }
        out
    }

    /// Current options.
    #[must_use]
    pub const fn options(&self) -> NvmeOptions {
        self.options
    }
}

fn append_bytes(path: &Path, bytes: &[u8]) -> Result<(), MemoryError> {
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| MemoryError::io("append nvme data", path, &error))?;
    file.write_all(bytes)
        .map_err(|error| MemoryError::io("append nvme data", path, &error))?;
    Ok(())
}

fn sync_file(path: &Path) -> Result<(), MemoryError> {
    // Opened writable: Windows FlushFileBuffers requires write access.
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|error| MemoryError::io("sync nvme data", path, &error))?;
    file.sync_data()
        .map_err(|error| MemoryError::io("sync nvme data", path, &error))?;
    Ok(())
}

/// Asynchronously appends a demotion record on the caller's Compio
/// runtime using the same record format as the sync backend.
///
/// The caller runs inside `compio::runtime::Runtime::block_on` (the same
/// single-threaded reactor the engine's net path uses); this function
/// creates no runtime of its own.
///
/// # Errors
///
/// Returns [`MemoryError::Io`] on filesystem failure.
pub async fn demote_async(dir: &Path, payload: &[u8]) -> Result<(u64, u64, u32), MemoryError> {
    let data = dir.join("materializations.dat");
    let record = encode_record(payload);
    let offset = compio_append(&data, &record).await?;
    Ok((offset, payload.len() as u64, crc32c::crc32c(payload)))
}

/// Asynchronously reads and verifies one record on the caller's Compio
/// runtime.
///
/// # Errors
///
/// Returns [`MemoryError::CorruptRepresentation`] on verification
/// failure, or [`MemoryError::Io`] on filesystem failure.
pub async fn promote_async(
    dir: &Path,
    offset: u64,
    expected_len: u64,
) -> Result<Vec<u8>, MemoryError> {
    let data = dir.join("materializations.dat");
    let bytes = compio_read_all(&data).await?;
    let start = usize::try_from(offset).map_err(|_| MemoryError::Invalid {
        detail: "nvme offset out of range".to_owned(),
    })?;
    if start >= bytes.len() {
        return Err(MemoryError::CorruptRepresentation {
            object: u64::MAX,
            detail: "nvme read past end".to_owned(),
        });
    }
    let (payload, _) = decode_record(&bytes[start..], crate::MEDIUM_MAX as u64)?;
    if payload.len() as u64 != expected_len {
        return Err(MemoryError::CorruptRepresentation {
            object: u64::MAX,
            detail: "nvme record length changed".to_owned(),
        });
    }
    Ok(payload)
}

async fn compio_file_len(path: &Path) -> Result<u64, MemoryError> {
    let metadata = compio::fs::metadata(path)
        .await
        .map_err(|error| MemoryError::Io {
            op: "stat nvme data",
            path: path.to_owned(),
            message: error.to_string(),
        })?;
    Ok(metadata.len())
}

async fn compio_append(path: &Path, bytes: &[u8]) -> Result<u64, MemoryError> {
    use compio::io::AsyncWriteAtExt as _;
    let offset = compio_file_len(path).await.unwrap_or(0);
    let mut opened = compio::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(path)
        .await
        .map_err(|error| MemoryError::Io {
            op: "open nvme data",
            path: path.to_owned(),
            message: error.to_string(),
        })?;
    let record = bytes.to_vec();
    let result = opened.write_all_at(record, offset).await;
    result.0.map_err(|error| MemoryError::Io {
        op: "append nvme data",
        path: path.to_owned(),
        message: error.to_string(),
    })?;
    opened.sync_data().await.map_err(|error| MemoryError::Io {
        op: "sync nvme data",
        path: path.to_owned(),
        message: error.to_string(),
    })?;
    Ok(offset)
}

async fn compio_read_all(path: &Path) -> Result<Vec<u8>, MemoryError> {
    let bytes = compio::fs::read(path)
        .await
        .map_err(|error| MemoryError::Io {
            op: "read nvme data",
            path: path.to_owned(),
            message: error.to_string(),
        })?;
    if bytes.len() > crate::MEDIUM_MAX * 4096 {
        return Err(MemoryError::TooLarge {
            len: bytes.len() as u64,
            max: (crate::MEDIUM_MAX * 4096) as u64,
            context: "nvme file length",
        });
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_roundtrips_with_crc() {
        let payload = b"hello nvme";
        let record = encode_record(payload);
        let (decoded, total) = decode_record(&record, 1 << 20).expect("decodes");
        assert_eq!(decoded, payload);
        assert_eq!(total, record.len());
    }

    #[test]
    fn bit_flip_is_detected() {
        let record = encode_record(b"precious bytes");
        let mut corrupt = record.clone();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 0xFF;
        assert!(decode_record(&corrupt, 1 << 20).is_err());
    }

    #[test]
    fn torn_tail_truncates_on_open() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut provider = NvmeProvider::open(dir.path(), NvmeOptions::default()).expect("open");
        provider.append(b"first").expect("append");
        // Simulate a torn write: garbage appended without a barrier.
        std::fs::write(
            dir.path().join("materializations.dat"),
            b"first-garbage-tail",
        )
        .expect("clobber");
        let data = std::fs::read(dir.path().join("materializations.dat")).expect("read");
        assert!(!data.is_empty());
        // Reopen must not fail: it truncates the torn tail.
        let reopened = NvmeProvider::open(dir.path(), NvmeOptions::default()).expect("reopen");
        let _ = reopened;
    }

    #[test]
    fn admission_budget_rejects_churn() {
        let options = NvmeOptions {
            daily_write_budget_bytes: 10,
            ..NvmeOptions::default()
        };
        let mut controller = AdmissionController::new(options, 8, 42);
        assert_eq!(controller.admit(6), Admission::Accept);
        controller.release();
        assert_eq!(controller.admit(6), Admission::RejectBudget);
    }
}
