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

use bytes::BytesMut;
use compio::BufResult;
use kivi_hardware::io::{AlignedTransfer, DirectFile, DirectIoPolicy, DirectMode};

use crate::error::MemoryError;

/// Record magic: `KVMM` (Kivi memory materialization).
pub const NVME_MAGIC: u32 = 0x4D_4D_56_4B;

/// Current record format version. Old versions fail loudly on load.
pub const NVME_RECORD_VERSION: u16 = 1;

/// Bytes of fixed record header: magic, version, flags, payload length, CRC.
/// The header is read on its own so a promotion reads the record it wants rather
/// than the file it lives in.
pub const RECORD_HEADER: usize = 20;

/// One positioned read from the Compio file, with a short read reported as the
/// torn tail it is.
///
/// A short read at the end of the file is a corrupt representation rather than a
/// transient I/O failure: retrying cannot produce bytes that were never
/// written, so a caller that retried would loop forever on a truncated tail.
async fn read_exact_at_async(
    file: &compio::fs::File,
    path: &Path,
    offset: u64,
    buf: &mut [u8],
) -> Result<(), MemoryError> {
    use compio::io::AsyncReadAt as _;

    let wanted = buf.len();
    // `AsyncReadAt` takes the buffer by value and requires it to be `'static`,
    // because the operation is handed to the runtime and the kernel may
    // complete it after the future is suspended. A caller-owned borrow cannot
    // cross that boundary, so the record is read into an owned buffer here and
    // copied out.
    //
    // The copy is bounded by one record and it replaces a copy of the whole
    // demotion store, which is what this function did before the promotion path
    // was made positioned. `BufResult` hands the buffer back on both arms, so
    // the bytes are recovered without a second read.
    let mut owned = BytesMut::new();
    owned.resize(wanted, 0);
    let BufResult(result, owned) = file.read_at(owned, offset).await;
    let read = result.map_err(|error| read_error(path, &error))?;
    if read != wanted {
        return Err(MemoryError::CorruptRepresentation {
            object: u64::MAX,
            detail: "nvme read past end".to_owned(),
        });
    }
    buf.copy_from_slice(&owned);
    Ok(())
}

/// One read failure, classified.
///
/// A short read or an `UnexpectedEof` at the end of the demotion store is a
/// torn tail, which is a corrupt representation rather than a transient
/// failure: retrying cannot produce bytes that were never written.
fn read_error(path: &Path, error: &std::io::Error) -> MemoryError {
    if error.kind() == std::io::ErrorKind::NotFound {
        // A locator naming bytes on a device that holds no records is a corrupt
        // locator, not a transient failure: callers must fail the read closed,
        // never retry it as I/O.
        return MemoryError::CorruptRepresentation {
            object: u64::MAX,
            detail: "nvme data file missing".to_owned(),
        };
    }
    if error.kind() == std::io::ErrorKind::UnexpectedEof {
        return MemoryError::CorruptRepresentation {
            object: u64::MAX,
            detail: "nvme read past end".to_owned(),
        };
    }
    MemoryError::Io {
        op: "read nvme data",
        path: path.to_owned(),
        message: error.to_string(),
    }
}

/// Bytes the record scan moves per unbuffered transfer.
///
/// Large enough that the per-request overhead is amortised over many records,
/// small enough to stay inside every L2 on every machine. A demotion store is
/// read in full only by recovery and by `scan_records`, both of which are
/// background work, so the transfer size is chosen for throughput rather than
/// for latency.
const SCAN_CHUNK: u64 = 1 << 20;

/// Provider options. All fields are optimizations; correctness holds at
/// their defaults on any ordinary file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvmeOptions {
    /// When to bypass the page cache for large transfers. The default decides
    /// per transfer from the transfer size and the volume's sector size, which
    /// is the only defensible answer: unbuffered I/O is faster above a
    /// device-dependent threshold and slower below it, and the threshold is not
    /// a compile-time constant.
    pub direct: DirectIoPolicy,
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
            direct: DirectIoPolicy::Adaptive,
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
/// Reads a record's payload length out of its header, without verifying the
/// record. The caller still runs [`decode_record`] over the assembled record, so
/// this is a sizing step and never a validation step.
fn declared_len(header: &[u8; RECORD_HEADER]) -> u64 {
    u64::from_le_bytes([
        header[8], header[9], header[10], header[11], header[12], header[13], header[14],
        header[15],
    ])
}

/// Reads exactly `buf.len()` bytes at `offset`, failing closed on a short read.
///
/// A short read at the end of the file is a torn tail, which is a corrupt
/// representation rather than a transient I/O failure: retrying cannot produce
/// bytes that were never written.
fn read_exact_at(
    file: &mut std::fs::File,
    path: &Path,
    offset: u64,
    buf: &mut [u8],
) -> Result<(), MemoryError> {
    use std::io::{Read, Seek, SeekFrom};
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| MemoryError::io("seek nvme data", path, &error))?;
    file.read_exact(buf).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            return MemoryError::CorruptRepresentation {
                object: u64::MAX,
                detail: "nvme read past end".to_owned(),
            };
        }
        MemoryError::io("read nvme data", path, &error)
    })
}

/// The volume's logical sector size, which bounds every unbuffered transfer.
///
/// Read from the device, not assumed. A wrong value does not fall back - it
/// fails the transfer - so a provider that cannot learn the answer stays
/// buffered rather than guessing.
fn sector_size(path: &Path) -> Option<u32> {
    kivi_hardware::io::logical_block_size(path)
}

/// Appends one window's worth of bytes to `carry` from a buffered handle, and
/// returns how many were read.
///
/// The buffered path has no alignment constraint, so it reads straight into
/// the carry buffer rather than through the aligned-transfer machinery. The
/// unbuffered path cannot do that, which is why [`AlignedTransfer`] exists and
/// why the two are separate functions rather than one function with a branch:
/// the branch would be inside the loop that the scan runs millions of times.
fn read_window_buffered(file: &mut std::fs::File, offset: u64, carry: &mut Vec<u8>) -> usize {
    use std::io::{Read as _, Seek as _, SeekFrom};

    let start = carry.len();
    let Ok(window) = usize::try_from(SCAN_CHUNK) else {
        return 0;
    };
    carry.resize(start + window, 0);
    if file.seek(SeekFrom::Start(offset)).is_err() {
        carry.truncate(start);
        return 0;
    }
    // A zero-byte read and a read error are the same thing to this caller: the
    // scan has no more records to find, and neither is a corrupt record the
    // caller could do anything about.
    if let Ok(read) = file.read(&mut carry[start..]) {
        carry.truncate(start + read);
        return read;
    }
    carry.truncate(start);
    0
}

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
    /// Direct-I/O mode, resolved once from the policy and the volume's logical
    /// sector size. Buffered unless the policy asked otherwise *and* the sector
    /// size is known, because an unaligned unbuffered read fails outright
    /// rather than falling back.
    direct: DirectMode,
    /// Alignment every unbuffered read and write must satisfy.
    alignment: u32,
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
        // The volume's logical sector size bounds every unbuffered transfer.
        // It is read from the device rather than assumed, because it differs
        // between NVMe, SATA and network-backed volumes and a wrong value does
        // not fall back - it fails.
        //
        // The data file may not exist yet on a fresh provider, and a directory
        // has no block size to report. Creating it up front is not a
        // behavioural change: the very first append would create it, and an
        // empty store is exactly what the file represents. It also makes the
        // mode resolvable at open, so `direct_mode` reports what the scan will
        // actually do rather than what a policy asked for.
        if !data.exists() {
            std::fs::write(&data, [])
                .map_err(|error| MemoryError::io("create nvme data", &data, &error))?;
        }
        let known = sector_size(&data);
        let alignment = known.unwrap_or(4096);
        // The mode is resolved once, from the policy and the volume's own
        // alignment. The per-record read path stays buffered regardless - see
        // [`NvmeProvider::read_at`] - and the mode resolved here is the one the
        // bulk scan uses.
        let direct = match known {
            Some(align) => options.direct.mode_for(SCAN_CHUNK, align),
            None => DirectMode::Buffered,
        };
        Ok(Self {
            dir: dir.to_owned(),
            options,
            admission: AdmissionController::new(options, 8, 0x1234_5678),
            next_offset,
            append_file: None,
            direct,
            alignment,
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
    /// This path is deliberately buffered whatever the direct-I/O policy says.
    /// A promotion reads one record of at most [`crate::MEDIUM_MAX`] bytes, and
    /// an unbuffered read of a small record is a full device round trip that the
    /// page cache would have served from memory. The measurements behind that
    /// decision are in `docs/hardware.md`: unbuffered I/O wins on the bulk scan
    /// and loses on the single-record read, so the policy selects a mode per
    /// path rather than per provider.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::CorruptRepresentation`] when the record at
    /// `offset` fails verification, or [`MemoryError::Io`] on filesystem
    /// failure.
    pub fn read_at(&self, offset: u64, expected_len: u64) -> Result<Vec<u8>, MemoryError> {
        let data = self.dir.join("materializations.dat");
        // Two positioned reads, sized from the record's own header: one for the
        // 20-byte header, one for the payload. The previous whole-file read
        // made a promotion cost a copy of the entire demotion store, so the
        // fifth promotion of a 1 KiB object copied as much as the first
        // promotion of a 1 GiB one.
        //
        // The handle is opened per call rather than cached in a `RefCell`:
        // caching needed interior mutability behind a `&self` method, and that
        // cost the provider its `Sync`, which is worth more than the open it
        // saved. The *append* handle is cached, because it is free of the
        // problem - it sits behind `&mut self` on the single-writer path that
        // owns every device append.
        let mut file =
            std::fs::File::open(&data).map_err(|error| Self::read_error(&data, &error))?;
        Self::read_record(&mut file, &data, offset, expected_len)
    }

    /// Assembles one record through `file` and verifies it.
    ///
    /// Split out of [`NvmeProvider::read_at`] so the read path and its test
    /// share one assembly.
    fn read_record(
        file: &mut std::fs::File,
        data: &Path,
        offset: u64,
        expected_len: u64,
    ) -> Result<Vec<u8>, MemoryError> {
        let mut header = [0_u8; RECORD_HEADER];
        read_exact_at(file, data, offset, &mut header)?;
        let len = declared_len(&header);
        if len > crate::MEDIUM_MAX as u64 {
            return Err(MemoryError::TooLarge {
                len,
                max: crate::MEDIUM_MAX as u64,
                context: "nvme record length",
            });
        }
        let len_usize = usize::try_from(len).map_err(|_| MemoryError::TooLarge {
            len,
            max: crate::MEDIUM_MAX as u64,
            context: "nvme record length",
        })?;
        let mut record = vec![0_u8; RECORD_HEADER + len_usize];
        record[..RECORD_HEADER].copy_from_slice(&header);
        read_exact_at(
            file,
            data,
            offset + RECORD_HEADER as u64,
            &mut record[RECORD_HEADER..],
        )?;
        let (payload, _) = decode_record(&record, crate::MEDIUM_MAX as u64)?;
        if payload.len() as u64 != expected_len {
            return Err(MemoryError::CorruptRepresentation {
                object: u64::MAX,
                detail: "nvme record length changed".to_owned(),
            });
        }
        Ok(payload)
    }

    fn read_error(data: &Path, error: &std::io::Error) -> MemoryError {
        if error.kind() == std::io::ErrorKind::NotFound {
            // A locator naming bytes on a device that holds no records is a
            // corrupt locator, not a transient failure: callers must fail the
            // read closed, never retry it as I/O.
            return MemoryError::CorruptRepresentation {
                object: u64::MAX,
                detail: "nvme data file missing".to_owned(),
            };
        }
        MemoryError::io("read nvme data", data, error)
    }

    /// The direct-I/O mode this provider resolved to, and the alignment every
    /// unbuffered transfer must satisfy. Diagnostics and tests; the answer never
    /// changes after [`NvmeProvider::open`].
    #[must_use]
    pub const fn direct_mode(&self) -> DirectMode {
        self.direct
    }

    /// The alignment unbuffered transfers must satisfy.
    #[must_use]
    pub const fn alignment(&self) -> u32 {
        self.alignment
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
    ///
    /// The scan is the one path in this provider that reads a large contiguous
    /// range, and it is therefore the one path where bypassing the page cache
    /// can pay: the bytes are read once, hashed, and discarded, so filling the
    /// cache with them costs memory and evicts something useful. The policy
    /// decides, and the transfer uses a reusable aligned buffer rather than a
    /// fresh allocation per block, which is what a direct read requires.
    ///
    /// A store that cannot be opened at all yields an empty scan rather than an
    /// error: a scan is a *report* about what is on the device, and "nothing
    /// could be read" is a report. A caller that needs to distinguish an empty
    /// store from an unreadable one is validating locators, which
    /// [`NvmeProvider::read_at`] already fails closed on.
    ///
    /// # Panics
    ///
    /// Never. A read failure, a short file and a corrupt record all produce a
    /// shorter vector rather than a panic, because the scan runs inside
    /// recovery and a panic there would turn a degraded store into an
    /// unavailable node.
    #[must_use]
    pub fn scan_records(&self) -> Vec<(u64, u64, u32)> {
        let data = self.dir.join("materializations.dat");
        let Ok(window_bytes) = usize::try_from(SCAN_CHUNK) else {
            return Vec::new();
        };
        let Ok(mut file) = DirectFile::open(&data, self.options.direct, SCAN_CHUNK)
            .or_else(|_| DirectFile::open_buffered(&data))
        else {
            return Vec::new();
        };
        let mut out = Vec::new();
        // Absolute file offset of the first byte in `carry`, which is how a
        // record's offset is recovered after a window boundary moves it.
        let mut carry_origin = 0_u64;
        // Bytes read but not yet consumed: at most one record, because a record
        // is consumed as soon as all of it is present.
        let mut carry: Vec<u8> = Vec::new();
        let mut window_offset = 0_u64;
        // One aligned buffer for the whole scan, reused per window. A fresh
        // aligned allocation per window is exactly the cost unbuffered I/O is
        // supposed to avoid, so it happens once.
        let mut window = AlignedTransfer::new(window_bytes, file.alignment());
        loop {
            let read = if file.is_unbuffered() {
                window
                    .read_at(&mut file, window_offset, window_bytes)
                    .unwrap_or(0)
            } else {
                read_window_buffered(file.file_mut(), window_offset, &mut carry)
            };
            if read == 0 {
                break;
            }
            window_offset += read as u64;

            let mut consumed = 0_usize;
            while carry.len() - consumed >= RECORD_HEADER {
                let header: [u8; RECORD_HEADER] = carry[consumed..consumed + RECORD_HEADER]
                    .try_into()
                    .expect("slice is exactly the header length");
                let Ok(len) = usize::try_from(declared_len(&header)) else {
                    break;
                };
                let total = RECORD_HEADER + len;
                if len > crate::MEDIUM_MAX || carry.len() - consumed < total {
                    break;
                }
                match decode_record(&carry[consumed..consumed + total], crate::MEDIUM_MAX as u64) {
                    Ok((payload, _)) => {
                        out.push((
                            carry_origin + consumed as u64,
                            payload.len() as u64,
                            crc32c::crc32c(&payload),
                        ));
                        consumed += total;
                    }
                    Err(_) => {
                        // A record that does not verify ends the scan: the file
                        // is append-only, so nothing after a bad record is
                        // trustworthy.
                        return out;
                    }
                }
            }
            if consumed > 0 {
                carry.drain(..consumed);
                carry_origin += consumed as u64;
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
    // Two positioned reads sized from the record's own header, matching the
    // synchronous path: a promotion costs one record, not a copy of the store.
    let file = compio::fs::OpenOptions::new()
        .read(true)
        .open(&data)
        .await
        .map_err(|error| MemoryError::Io {
            op: "open nvme data",
            path: data.clone(),
            message: error.to_string(),
        })?;
    let mut header = [0_u8; RECORD_HEADER];
    read_exact_at_async(&file, &data, offset, &mut header).await?;
    let len = declared_len(&header);
    if len > crate::MEDIUM_MAX as u64 {
        return Err(MemoryError::TooLarge {
            len,
            max: crate::MEDIUM_MAX as u64,
            context: "nvme record length",
        });
    }
    let len_usize = usize::try_from(len).map_err(|_| MemoryError::TooLarge {
        len,
        max: crate::MEDIUM_MAX as u64,
        context: "nvme record length",
    })?;
    let mut record = vec![0_u8; RECORD_HEADER + len_usize];
    record[..RECORD_HEADER].copy_from_slice(&header);
    read_exact_at_async(
        &file,
        &data,
        offset + RECORD_HEADER as u64,
        &mut record[RECORD_HEADER..],
    )
    .await?;
    let (payload, _) = decode_record(&record, crate::MEDIUM_MAX as u64)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scan_finds_every_record_it_wrote() {
        // The scan was rewritten from a whole-file read to a windowed one, and
        // the rewrite is only correct if it still finds every record across a
        // window boundary. Records are written small enough that many straddle
        // the 1 MiB window, so this exercises the carry path.
        let dir = tempfile::tempdir().expect("tempdir");
        let mut provider = NvmeProvider::open(dir.path(), NvmeOptions::default()).expect("open");
        let mut expected = Vec::new();
        for index in 0..4_000_usize {
            // Vary the length so records do not align with the window.
            let payload = vec![u8::try_from(index % 251).expect("masked"); 900 + index % 700];
            let (offset, len, crc) = provider.append_relaxed(&payload).expect("append");
            expected.push((offset, len, crc));
        }
        let scanned = provider.scan_records();
        assert_eq!(scanned.len(), expected.len(), "record count");
        for (found, want) in scanned.iter().zip(&expected) {
            assert_eq!(*found, *want, "offset, length and checksum");
        }
    }

    #[test]
    fn a_scan_of_an_empty_store_is_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let provider = NvmeProvider::open(dir.path(), NvmeOptions::default()).expect("open");
        assert!(provider.scan_records().is_empty());
    }

    #[test]
    fn a_scan_stops_at_a_torn_tail() {
        // The same append-only crash rule the reopen path follows: only the
        // last record may be partial, and a scan must not report it.
        let dir = tempfile::tempdir().expect("tempdir");
        let mut provider = NvmeProvider::open(dir.path(), NvmeOptions::default()).expect("open");
        provider.append_relaxed(b"first").expect("append");
        provider.append_relaxed(b"second").expect("append");
        let good = provider.scan_records();
        assert_eq!(good.len(), 2);
        // Append garbage after the last complete record.
        let data = dir.path().join("materializations.dat");
        let mut bytes = std::fs::read(&data).expect("read");
        bytes.extend_from_slice(b"torn-tail");
        std::fs::write(&data, &bytes).expect("clobber");
        assert_eq!(
            provider.scan_records().len(),
            2,
            "a torn tail is not a record"
        );
    }

    #[test]
    fn the_direct_policy_is_answered_from_the_device_not_a_constant() {
        let dir = tempfile::tempdir().expect("tempdir");
        let options = NvmeOptions {
            direct: DirectIoPolicy::Always,
            ..NvmeOptions::default()
        };
        let provider = NvmeProvider::open(dir.path(), options).expect("open");
        let align = provider.alignment();
        assert!(align >= 512 && align.is_power_of_two(), "align {align}");
        // Whether an unbuffered handle is obtainable is a property of the volume
        // the temporary directory happens to be on: `tmpfs` and `overlayfs`
        // refuse it everywhere, and a Windows volume that will not report its
        // alignment cannot be asked to open unbuffered. The assertion is that
        // the provider's answer matches the platform's, not that it is always
        // unbuffered.
        // The provider does not create the data file until its first append, so
        // the probe writes one. `DirectFile::open` is read-only, which is why it
        // needs the file to exist before it can answer.
        let data = dir.path().join("materializations.dat");
        std::fs::write(&data, vec![0_u8; 64 * 1024]).expect("write");
        let obtainable = DirectFile::open(&data, DirectIoPolicy::Always, 4096)
            .is_ok_and(|file| file.is_unbuffered());
        assert_eq!(
            provider.direct_mode().is_unbuffered(),
            obtainable,
            "the provider's mode must match what a real unbuffered open produces"
        );
    }

    #[test]
    fn a_never_policy_is_buffered_whatever_the_device_says() {
        let dir = tempfile::tempdir().expect("tempdir");
        let options = NvmeOptions {
            direct: DirectIoPolicy::Never,
            ..NvmeOptions::default()
        };
        let provider = NvmeProvider::open(dir.path(), options).expect("open");
        assert!(!provider.direct_mode().is_unbuffered());
    }

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
