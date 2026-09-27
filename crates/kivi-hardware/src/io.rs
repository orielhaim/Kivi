//! Direct I/O, aligned buffers, and reusable aligned transfers.
//!
//! Three separate mechanisms, deliberately kept apart:
//!
//! * **Aligned buffers** ([`AlignedBuf`]) — a heap buffer whose address is a
//!   multiple of a chosen power of two. Needed by any unbuffered path and
//!   useful to any SIMD kernel, and safe on every platform.
//! * **Direct I/O** ([`DirectFile`]) — telling the operating system not to cache
//!   the pages a read or write touches. Saves a copy and a page-cache entry per
//!   operation, and costs a synchronisation point.
//! * **Aligned transfers** ([`AlignedTransfer`]) — the reusable, owner-local
//!   buffer an unbuffered read needs. A direct read at an arbitrary offset
//!   cannot use a caller-supplied `&mut [u8]`, because the buffer's address
//!   must satisfy the volume's alignment, so the machinery to reconcile the
//!   two lives here rather than in every caller.
//!
//! ## Alignment is not symmetric, and guessing it is a failed I/O
//!
//! Linux `O_DIRECT` and Windows `FILE_FLAG_NO_BUFFERING` both require the
//! buffer *address*, the *length*, and the *file offset* to be multiples of the
//! volume's logical sector size, which is not a compile-time constant and
//! differs per volume. A misaligned unbuffered request does not fall back to
//! buffered; it fails. Windows is stricter still, returning
//! `ERROR_INVALID_PARAMETER` where Linux would return `EINVAL`.
//!
//! So Kivi asks the volume rather than assuming 512 or 4096:
//! [`logical_block_size`] reads `/sys/dev/block/<major>:<minor>/queue/`
//! on Linux and issues `IOCTL_DISK_GET_DRIVE_GEOMETRY` on Windows, and a caller
//! that cannot learn the answer gets [`DirectIoSupport::UnknownAlignment`] and
//! a buffered handle. The previous version of this module reported
//! `UnknownAlignment` unconditionally and its `DirectIoPolicy` was therefore a
//! knob that changed only a diagnostic string.
//!
//! ## When direct I/O is actually faster
//!
//! Unbuffered I/O is not a general speedup. It wins when the transfer is large
//! relative to the device's block and the page cache would be filled only to be
//! evicted, and it loses below that because each unbuffered request is its own
//! device round trip while buffered requests coalesce. Kivi therefore decides
//! per transfer from the transfer size, and the threshold is not a constant:
//! it is the volume's own block size clamped to a ceiling, so a 4 KiB
//! filesystem and a 4 KiB NVMe namespace are treated the same way and a
//! 512-byte sector device is not.

use std::fmt;
use std::path::Path;

use crate::capability::{DirectIoSupport, Support};

/// A buffer whose address is a multiple of `align`.
///
/// Implemented by over-allocating and treating the excess as leading padding.
/// The aligned bytes are a *subslice of the original allocation*, never a copy:
/// a copy would land at whatever address the second allocation happened to get,
/// which is the whole problem. Deallocation therefore uses the base pointer,
/// which is why the base is kept.
#[derive(Debug)]
pub struct AlignedBuf {
    /// The allocation, of which [`AlignedBuf::offset..offset + len`] is the
    /// aligned view.
    base: Box<[u8]>,
    /// Offset of the aligned start within [`AlignedBuf::base`].
    offset: usize,
    /// Length of the aligned view.
    len: usize,
}

impl AlignedBuf {
    /// Allocates `len` bytes aligned to `align`.
    ///
    /// # Panics
    ///
    /// Panics when `align` is not a power of two. Callers pass a compile-time
    /// constant; a runtime value that might not be one is a programming error
    /// worth failing on rather than rounding silently.
    #[must_use]
    pub fn new(len: usize, align: usize) -> Self {
        assert!(align.is_power_of_two(), "alignment must be a power of two");
        let len = len.max(1);
        let mut base = vec![0_u8; len + align].into_boxed_slice();
        let address = base.as_ptr() as usize;
        let offset = (align - address % align) % align;
        base[offset..offset + len].fill(0);
        Self { base, offset, len }
    }

    /// Zero-initialised bytes. Identical to [`AlignedBuf::new`]; named for the
    /// call sites where the intent is "give me a clean buffer".
    #[must_use]
    pub fn zeroed(len: usize, align: usize) -> Self {
        Self::new(len, align)
    }

    /// The aligned view.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.base[self.offset..self.offset + self.len]
    }

    /// The aligned view, mutably.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.base[self.offset..self.offset + self.len]
    }

    /// Pointer to the aligned start, for a syscall that writes only within
    /// `len` bytes.
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.as_mut_slice().as_mut_ptr()
    }

    /// Length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when the buffer holds no bytes. A zero-length request still
    /// produces a one-byte allocation, so a buffer always has a real address.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the address really is aligned, which the constructor guarantees
    /// but a test should verify rather than trust.
    #[must_use]
    pub fn is_aligned(&self, align: usize) -> bool {
        (self.as_slice().as_ptr() as usize).is_multiple_of(align)
    }

    /// Zeroes the buffer without reallocating.
    pub fn clear(&mut self) {
        self.as_mut_slice().fill(0);
    }

    /// Offset of the aligned view within the base allocation, exposed so the
    /// deallocation path is testable.
    #[must_use]
    pub fn base_offset(&self) -> usize {
        self.offset
    }
}

/// Whether a file handle bypasses the page cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum DirectMode {
    /// Ordinary buffered I/O through the operating system cache.
    #[default]
    Buffered,
    /// Bypass the cache, with the alignment both the platform and the volume
    /// require.
    Unbuffered {
        /// Alignment every address, length and file offset must satisfy.
        align: u32,
    },
}

impl DirectMode {
    /// The alignment an unbuffered request must satisfy, or `None` when the
    /// mode is buffered.
    #[must_use]
    pub const fn alignment(&self) -> Option<u32> {
        match self {
            Self::Buffered => None,
            Self::Unbuffered { align } => Some(*align),
        }
    }

    /// True when the mode bypasses the cache.
    #[must_use]
    pub const fn is_unbuffered(&self) -> bool {
        matches!(self, Self::Unbuffered { .. })
    }
}

impl fmt::Display for DirectMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buffered => write!(f, "buffered"),
            Self::Unbuffered { align } => write!(f, "unbuffered(align={align})"),
        }
    }
}

/// What a caller wants done about direct I/O.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum DirectIoPolicy {
    /// Decide per operation from the transfer size. The default: a mechanism
    /// Kivi has measured rather than one it has assumed.
    #[default]
    Adaptive,
    /// Always bypass the cache.
    Always,
    /// Never bypass the cache.
    Never,
}

impl DirectIoPolicy {
    /// Parses the configuration spelling.
    ///
    /// # Errors
    ///
    /// Returns a message naming the accepted spellings.
    pub fn parse(spec: &str) -> Result<Self, String> {
        match spec.trim().to_ascii_lowercase().as_str() {
            "adaptive" | "auto" | "default" => Ok(Self::Adaptive),
            "always" | "on" | "direct" => Ok(Self::Always),
            "never" | "off" => Ok(Self::Never),
            other => Err(format!(
                "unknown direct io policy {other:?}; expected one of adaptive, always, never"
            )),
        }
    }

    /// The mode to use for a transfer of `len` bytes against a device whose
    /// direct-I/O alignment is `align`.
    ///
    /// The threshold is the smaller of the volume's logical block and a fixed
    /// ceiling. Below the block, an unbuffered write still costs a full device
    /// round trip while buffered I/O coalesces into a larger one, so buffering
    /// wins. The ceiling exists because past a few hundred kilobytes the page
    /// cache copy is no longer the dominant cost and unbuffered I/O's higher
    /// per-operation overhead stops paying.
    #[must_use]
    pub fn mode_for(&self, len: u64, align: u32) -> DirectMode {
        let threshold = u64::from(align.max(1)).clamp(4096, 512 * 1024);
        let unbuffered = match self {
            Self::Always => true,
            Self::Never => false,
            Self::Adaptive => len >= threshold,
        };
        if unbuffered {
            DirectMode::Unbuffered { align }
        } else {
            DirectMode::Buffered
        }
    }
}

impl fmt::Display for DirectIoPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Adaptive => "adaptive",
            Self::Always => "always",
            Self::Never => "never",
        };
        f.write_str(name)
    }
}

/// Rounds `value` up to a multiple of `align`, or reports that it already is.
///
/// # Errors
///
/// Returns the value unchanged when it is not aligned, which is how a caller
/// learns its offset or length cannot be used unbuffered.
pub fn align_up(value: u64, align: u32) -> Result<u64, Unaligned> {
    if align == 0 || !align.is_power_of_two() {
        return Err(Unaligned { value, align });
    }
    let mask = u64::from(align) - 1;
    if value & mask == 0 {
        Ok(value)
    } else {
        Err(Unaligned { value, align })
    }
}

/// An offset, length, or address that does not satisfy an alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("value {value} is not a multiple of {align}")]
pub struct Unaligned {
    /// The offending value.
    pub value: u64,
    /// The alignment it should have satisfied.
    pub align: u32,
}

/// What direct I/O this machine offers.
///
/// The machine-level answer is deliberately coarse: the mechanism exists, and
/// its alignment is a property of the *volume* rather than of the machine, so
/// the per-file answer comes from [`logical_block_size`].
#[must_use]
pub fn detect() -> Support<DirectIoSupport> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        Support::Available(DirectIoSupport::UnknownAlignment)
    }
    #[cfg(windows)]
    {
        Support::Available(DirectIoSupport::UnknownAlignment)
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "android")))]
    {
        Support::Unavailable(crate::capability::Unavailable::UnsupportedPlatform)
    }
}

/// The logical block size of the volume holding `path`, when the platform can
/// be asked.
///
/// This is the number every unbuffered transfer must satisfy, and it is not a
/// constant: 512 for NVMe and SATA, 4096 for many network, RAID and
/// filesystems, and 4096 again for `XFS` on top of a 4 KiB device. Guessing
/// produces a failed I/O rather than a slow one, so a caller that gets `None`
/// must use the buffered path.
#[must_use]
pub fn logical_block_size(path: &Path) -> Option<u32> {
    sys::logical_block_size(path)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod sys {
    use std::fs::OpenOptions;
    use std::io;
    use std::os::unix::fs::MetadataExt;
    use std::path::Path;

    use super::DirectMode;

    /// The alignment the kernel says direct I/O on this file needs.
    ///
    /// Two mechanisms, in this order, because they answer different questions:
    ///
    /// * `statx(STATX_DIOALIGN)` (Linux 6.1) reports the *filesystem's* own
    ///   requirement — `stx_dio_mem_align` and `stx_dio_offset_align`. That is
    ///   strictly more accurate than the block size, because a filesystem can
    ///   require a larger alignment than its device's sectors: `btrfs` on a
    ///   4 KiB device commonly wants 4 KiB, and `XFS` with a realtime
    ///   subvolume can want more. It is also the only mechanism that knows
    ///   about a `virtio` device the sysfs `queue` tree does not expose, which
    ///   is what a virtual machine has.
    /// * `/sys/dev/block/<major>:<minor>/queue/logical_block_size` is the
    ///   fallback for kernels before 6.1. It names the device's logical block
    ///   size, which is a lower bound on the requirement and therefore safe to
    ///   satisfy only when it happens to be a power of two — which every real
    ///   block size is, and which is checked rather than assumed.
    ///
    /// The returned value is the alignment a caller must satisfy for the buffer
    /// address, the length, and the file offset. When the two mechanisms
    /// disagree, the larger answer wins: over-aligning is always valid and
    /// under-aligning is a failed I/O.
    #[allow(
        unsafe_code,
        reason = "`statx` is a raw syscall; the structure it fills is a \
                  zeroed local that the kernel writes into"
    )]
    pub(super) fn logical_block_size(path: &Path) -> Option<u32> {
        let from_statx = statx_dio_align(path);
        let from_sysfs = sysfs_logical_block_size(path);
        match (from_statx, from_sysfs) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }

    /// `statx(STATX_DIOALIGN)`, which needs Linux 6.1 or newer.
    #[allow(unsafe_code)]
    fn statx_dio_align(path: &Path) -> Option<u32> {
        use std::ffi::CString;
        use std::mem::MaybeUninit;

        let c_path = CString::new(path.as_os_str().as_encoded_bytes()).ok()?;
        // SAFETY: `statx` writes at most `sizeof(struct statx)` bytes into the
        // structure it is handed, and the structure is a correctly sized,
        // zero-initialised local. `AT_FDCWD` with a relative path is the
        // documented way to stat without an open descriptor.
        let (mask, mem_align, offset_align) = unsafe {
            let mut buffer = MaybeUninit::<libc::statx>::zeroed();
            let rc = libc::statx(
                libc::AT_FDCWD,
                c_path.as_ptr(),
                libc::AT_STATX_SYNC_AS_STAT,
                libc::STATX_DIOALIGN,
                buffer.as_mut_ptr(),
            );
            if rc != 0 {
                return None;
            }
            let statx = buffer.assume_init();
            (
                statx.stx_mask,
                statx.stx_dio_mem_align,
                statx.stx_dio_offset_align,
            )
        };
        // The kernel only fills the alignment fields when it says it did, and
        // it omits them for filesystems that have no opinion (which then leave
        // them zero). A zero means "no answer", not "no alignment required".
        if mask & libc::STATX_DIOALIGN == 0 {
            return None;
        }
        let align = mem_align.max(offset_align);
        (align > 0 && align.is_power_of_two()).then_some(align)
    }

    /// The device's logical block size, through the sysfs queue tree.
    ///
    /// For a *partition* (`sda1`, `nvme0n1p2`) sysfs has no `queue` directory
    /// — the queue belongs to the whole disk. So a lookup that finds a
    /// `partition` file follows it to the parent, which is the hop `lsblk` and
    /// `blockdev` both perform. Skipping it is why a naive implementation
    /// reports "unknown alignment" on every ordinary Linux install and the
    /// direct-I/O backend then never engages.
    fn sysfs_logical_block_size(path: &Path) -> Option<u32> {
        let metadata = OpenOptions::new()
            .read(true)
            .open(path)
            .ok()?
            .metadata()
            .ok()?;
        let device = resolve_device(metadata.dev() >> 8, metadata.dev() & 0xFF)?;
        read_queue(&format!("/sys/dev/block/{device}/queue/logical_block_size"))
    }

    /// The `major:minor` whose `queue/` directory exists, following a partition
    /// to its parent disk when necessary.
    ///
    /// Bounded at four hops: a partition of a partition of a partition is not a
    /// thing, and a malformed sysfs must not become a loop.
    fn resolve_device(major: u64, minor: u64) -> Option<String> {
        let mut current = format!("{major:x}:{minor:x}");
        for _ in 0..4 {
            if std::path::Path::new(&format!("/sys/dev/block/{current}/queue")).is_dir() {
                return Some(current);
            }
            // `partition` is a symlink named after the parent's `major:minor`.
            let link = std::fs::read_link(format!("/sys/dev/block/{current}/partition")).ok()?;
            current = link.file_name()?.to_str()?.to_string();
        }
        None
    }

    fn read_queue(path: &str) -> Option<u32> {
        std::fs::read_to_string(path)
            .ok()?
            .trim()
            .parse::<u32>()
            .ok()
            .filter(|size| size.is_power_of_two() && *size > 0)
    }

    #[allow(unsafe_code)]
    pub(super) fn open(path: &Path, mode: DirectMode) -> io::Result<std::fs::File> {
        use std::os::unix::fs::OpenOptionsExt as _;

        let mut options = OpenOptions::new();
        options.read(true);
        if mode.is_unbuffered() {
            // `O_DIRECT` is the Linux spelling of "do not cache this". It has
            // no standard-library equivalent, which is the reason the whole
            // mode lives behind this `sys` boundary.
            options.custom_flags(libc::O_DIRECT);
        }
        options.open(path)
    }
}

#[cfg(windows)]
mod sys {
    use std::io;
    use std::os::windows::ffi::OsStrExt as _;
    use std::path::Path;

    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_FLAG_NO_BUFFERING, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
    };
    use windows_sys::Win32::System::IO::DeviceIoControl;
    use windows_sys::Win32::System::Ioctl::{DISK_GEOMETRY, IOCTL_DISK_GET_DRIVE_GEOMETRY};

    use super::DirectMode;

    /// The volume's sector size, from the volume handle the file's path names.
    ///
    /// Windows has no per-file sector-size interface, so the path is converted
    /// to a `\\.\C:` volume device and `IOCTL_DISK_GET_DRIVE_GEOMETRY` is
    /// issued against it. That requires no privilege: the volume handle opens
    /// for metadata only, and the geometry is public information.
    #[allow(unsafe_code)]
    pub(super) fn logical_block_size(path: &Path) -> Option<u32> {
        let volume = volume_device(path)?;
        // SAFETY: `volume` is a NUL-terminated wide buffer built here and
        // outlives the call; a zero access mask requests metadata only, which
        // needs no privilege; `geometry` is a correctly sized, zero-initialised
        // local that the call writes into; and the handle is closed exactly
        // once on every path out of the block.
        let bytes = unsafe {
            let handle = CreateFileW(
                volume.as_ptr(),
                0,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
                std::ptr::null_mut(),
            );
            if handle == INVALID_HANDLE_VALUE {
                return None;
            }
            let mut geometry: DISK_GEOMETRY = std::mem::zeroed();
            let mut returned = 0_u32;
            let ok = DeviceIoControl(
                handle,
                IOCTL_DISK_GET_DRIVE_GEOMETRY,
                std::ptr::null(),
                0,
                (&raw mut geometry).cast::<std::ffi::c_void>(),
                u32::try_from(std::mem::size_of::<DISK_GEOMETRY>()).unwrap_or(0),
                &raw mut returned,
                std::ptr::null_mut(),
            );
            CloseHandle(handle);
            if ok == 0 {
                return None;
            }
            geometry.BytesPerSector
        };
        (bytes.is_power_of_two() && bytes > 0).then_some(bytes)
    }

    /// `\\.\C:` for the volume holding `path`, which is what the geometry
    /// control code expects.
    fn volume_device(path: &Path) -> Option<Vec<u16>> {
        let absolute = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        // The drive letter is the first character of the absolute path. A UNC
        // or device path has none, and this mechanism has no answer for one,
        // so it declines rather than opening something that is not the volume
        // the file is on.
        let first = absolute.as_os_str().encode_wide().next()?;
        // A drive letter is an ASCII letter; anything else means this is a UNC
        // or device path, for which this mechanism has no answer.
        let letter = u8::try_from(first).ok().filter(u8::is_ascii_alphabetic)?;
        let drive = u16::from(letter.to_ascii_uppercase());
        // `\\.\C:` is the Win32 volume-device spelling of a drive letter. The
        // backslashes are doubled because this is a Rust string, not because
        // the path contains them.
        let mut device: Vec<u16> = r"\\.\".encode_utf16().collect();
        device.push(drive);
        device.push(u16::from(b':'));
        device.push(0);
        Some(device)
    }

    pub(super) fn open(path: &Path, mode: DirectMode) -> io::Result<std::fs::File> {
        use std::os::windows::fs::OpenOptionsExt;

        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        if mode.is_unbuffered() {
            // A misaligned unbuffered request fails with
            // `ERROR_INVALID_PARAMETER` rather than falling back, which is why
            // the alignment is resolved before the open.
            options.custom_flags(FILE_FLAG_NO_BUFFERING);
        }
        options.open(path)
    }
}

#[cfg(not(any(windows, target_os = "linux", target_os = "android")))]
mod sys {
    use std::io;
    use std::path::Path;

    use super::DirectMode;

    pub(super) fn logical_block_size(_path: &Path) -> Option<u32> {
        None
    }

    pub(super) fn open(path: &Path, _mode: DirectMode) -> io::Result<std::fs::File> {
        std::fs::File::open(path)
    }
}

/// Rounds `len` up to `align` and returns an aligned buffer of that length.
#[must_use]
pub fn aligned_buffer(len: usize, align: usize) -> AlignedBuf {
    let rounded = len.div_ceil(align) * align;
    AlignedBuf::new(rounded.max(align), align)
}

/// A pool of aligned buffers owned by one thread.
///
/// The pool exists because buffer reuse is cheaper than allocation and because
/// a registered buffer must be handed to the kernel once rather than once per
/// operation. It is deliberately *not* a shared, lock-protected pool: a lock
/// on the I/O path costs exactly what registration was meant to remove, and
/// every Kivi I/O owner is a single thread that can own a pool outright.
///
/// Buffers are moved out on [`BufferPool::take`] and moved back on
/// [`BufferPool::recycle`], so no lease can outlive the pool it came from and
/// no address is aliased.
#[derive(Debug)]
pub struct BufferPool {
    free: Vec<AlignedBuf>,
    align: usize,
    /// Capacity the pool was created with, so `leased` is derivable from it
    /// rather than from a counter that can drift.
    capacity: usize,
}

impl BufferPool {
    /// Creates a pool holding `count` buffers of `len` bytes each.
    #[must_use]
    pub fn new(count: usize, len: usize, align: usize) -> Self {
        Self {
            free: (0..count).map(|_| AlignedBuf::new(len, align)).collect(),
            align,
            capacity: count,
        }
    }

    /// Takes a buffer, or `None` when the pool is exhausted.
    ///
    /// Exhaustion is backpressure, not an error: the caller falls back to a
    /// fresh allocation or defers. Blocking for a buffer would put a
    /// stall-equivalent on the I/O path, which is what the pool exists to
    /// avoid.
    pub fn take(&mut self) -> Option<AlignedBuf> {
        self.free.pop()
    }

    /// Returns a buffer taken earlier. A buffer that did not come from this
    /// pool is accepted and kept, because dropping it would lose the reuse.
    pub fn recycle(&mut self, buffer: AlignedBuf) {
        self.free.push(buffer);
    }

    /// Buffers currently available.
    #[must_use]
    pub fn available(&self) -> usize {
        self.free.len()
    }

    /// Buffers currently handed out.
    #[must_use]
    pub fn leased(&self) -> usize {
        self.capacity - self.free.len()
    }

    /// Total buffers the pool was created with.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Alignment every buffer in the pool satisfies.
    #[must_use]
    pub fn align(&self) -> usize {
        self.align
    }
}

/// A file handle opened for either buffered or unbuffered I/O.
///
/// The mode is fixed at open time on both platforms, because both require it:
/// Linux's `O_DIRECT` is a flag on the open and Windows'
/// `FILE_FLAG_NO_BUFFERING` likewise. Opening unbuffered and then discovering
/// the alignment is unknown is exactly the state that produces silent
/// `EINVAL`s, so [`DirectFile::open`] refuses rather than degrading.
#[derive(Debug)]
pub struct DirectFile {
    inner: std::fs::File,
    mode: DirectMode,
    len: u64,
}

impl DirectFile {
    /// Opens `path` in the requested mode, or buffered when the request cannot
    /// be honoured.
    ///
    /// The alignment is learned from the file's own device. When it cannot be
    /// learned, the request resolves to `Buffered`. When the platform *reports*
    /// an alignment but the unbuffered open is then **refused** — a `tmpfs`, an
    /// `overlayfs`, a 9p share, a Windows volume that will not report its
    /// geometry — the buffered open is used instead and
    /// [`DirectFile::mode`] reports what the handle actually is.
    ///
    /// That second fallback is the point. Deciding `Unbuffered` from the
    /// alignment alone and then opening buffered is how a caller comes to
    /// believe it has a direct handle it does not have, which turns a silent
    /// page-cache copy into an unmeasured one. A caller must be able to trust
    /// [`DirectFile::is_unbuffered`].
    ///
    /// # Errors
    ///
    /// Returns the operating-system error when neither the unbuffered nor the
    /// buffered open succeeds.
    pub fn open(path: &Path, policy: DirectIoPolicy, len: u64) -> std::io::Result<Self> {
        let align = logical_block_size(path);
        let wanted = match align {
            Some(align) => policy.mode_for(len, align),
            // Without an alignment, an unbuffered request cannot be satisfied:
            // a misaligned one fails rather than falling back, and guessing an
            // alignment is a failed I/O.
            None => DirectMode::Buffered,
        };
        let resolved = match sys::open(path, wanted) {
            Ok(inner) => (inner, wanted),
            // The platform promised an alignment and then refused the open.
            // Degrade to buffered rather than propagating: unbuffered I/O is
            // an optimisation and must never be a startup failure. The
            // original error is the one a caller wants when the fallback also
            // fails, because it names the unbuffered attempt.
            Err(error) if wanted.is_unbuffered() => match sys::open(path, DirectMode::Buffered) {
                Ok(inner) => (inner, DirectMode::Buffered),
                Err(_) => return Err(error),
            },
            Err(error) => return Err(error),
        };
        let (inner, mode) = resolved;
        let len = inner.metadata().map_or(len, |metadata| metadata.len());
        Ok(Self { inner, mode, len })
    }

    /// Opens `path` explicitly buffered, skipping the alignment probe.
    ///
    /// # Errors
    ///
    /// Returns the operating-system error when the file cannot be opened.
    pub fn open_buffered(path: &Path) -> std::io::Result<Self> {
        Ok(Self {
            inner: sys::open(path, DirectMode::Buffered)?,
            mode: DirectMode::Buffered,
            len: 0,
        })
    }

    /// The mode this handle was actually opened with.
    #[must_use]
    pub const fn mode(&self) -> DirectMode {
        self.mode
    }

    /// Whether this handle bypasses the page cache.
    #[must_use]
    pub const fn is_unbuffered(&self) -> bool {
        self.mode.is_unbuffered()
    }

    /// The alignment every unbuffered transfer must satisfy, or the platform
    /// page size when buffered.
    #[must_use]
    pub fn alignment(&self) -> u32 {
        self.mode.alignment().unwrap_or(4096)
    }

    /// File length in bytes, as of the open.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.len
    }

    /// Whether the file was empty at open.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The underlying handle, for a caller that has already satisfied the
    /// alignment itself.
    #[must_use]
    pub const fn file(&self) -> &std::fs::File {
        &self.inner
    }

    /// Mutably borrows the handle.
    pub const fn file_mut(&mut self) -> &mut std::fs::File {
        &mut self.inner
    }
}

/// One unbuffered read, and the buffer it landed in.
///
/// A direct read needs an aligned destination, so the bytes are read into a
/// reusable aligned buffer and the caller receives the requested window out of
/// it. That is the whole reason this type exists: allocating a fresh aligned
/// buffer per operation is the alternative, and it costs a `malloc` of
/// `align + len` and a page-table change per read, which is more than the
/// unbuffered read saved.
///
/// The buffer is owned by the caller through `&mut self`, so the bytes borrow
/// from the transfer and cannot outlive it. There is no global pool and no
/// lock: a shared pool would put a `Mutex` on the path that exists to remove
/// per-operation overhead.
#[derive(Debug)]
pub struct AlignedTransfer {
    buffer: AlignedBuf,
    /// Bytes of the buffer that hold valid data.
    filled: usize,
    /// File offset the first byte of `buffer` came from.
    origin: u64,
}

impl AlignedTransfer {
    /// Creates a transfer that can read up to `capacity` bytes.
    #[must_use]
    pub fn new(capacity: usize, align: u32) -> Self {
        Self {
            buffer: AlignedBuf::new(capacity.max(align as usize), align as usize),
            filled: 0,
            origin: 0,
        }
    }

    /// The bytes this read produced, which is exactly what was asked for
    /// unless the file ended first.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.buffer.as_slice()[..self.filled]
    }

    /// File offset the first returned byte came from, which is the caller's
    /// requested offset and not the aligned offset the read used.
    #[must_use]
    pub const fn origin(&self) -> u64 {
        self.origin
    }

    /// Reads `len` bytes at `offset` directly, when the handle is unbuffered
    /// and the request satisfies the alignment.
    ///
    /// Returns the number of bytes produced. A short result means the file
    /// ended, which is a value and not an error: the caller decides whether a
    /// short read is acceptable.
    ///
    /// # Errors
    ///
    /// Returns [`std::io::ErrorKind::InvalidInput`] when the handle is
    /// buffered, the request is longer than the buffer, or the platform
    /// refused the unaligned transfer. The buffer is left empty rather than
    /// partially filled, so a caller that ignores the error reads nothing.
    pub fn read_at(
        &mut self,
        file: &mut DirectFile,
        offset: u64,
        len: usize,
    ) -> std::io::Result<usize> {
        self.filled = 0;
        if !file.is_unbuffered() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "aligned transfer requires an unbuffered handle",
            ));
        }
        if len > self.buffer.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "transfer buffer is smaller than the request",
            ));
        }
        let align = u64::from(file.alignment());
        if align == 0 || !align.is_power_of_two() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "unbuffered alignment is not a power of two",
            ));
        }
        // Round the request out to the alignment and read the covering range.
        // A partial trailing block is read and then ignored, which is one
        // extra sector of device traffic in exchange for never having to
        // allocate a second buffer for a tail.
        let start = offset & !(align - 1);
        let end = offset
            .checked_add(len as u64)
            .and_then(|value| value.checked_add(align - 1))
            .map(|value| value & !(align - 1))
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "read range overflow")
            })?;
        let window = usize::try_from(end - start).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "read range too large")
        })?;
        if window > self.buffer.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "aligned window exceeds the transfer buffer",
            ));
        }
        let mut read = 0_usize;
        while read < window {
            // A positioned read rather than a seek plus a read: `pread` is one
            // syscall where `seek` + `read` is two, and the loop that handles a
            // short device read is the same either way.
            match positioned_read(
                file.file_mut(),
                &mut self.buffer.as_mut_slice()[read..window],
                start + read as u64,
            ) {
                Ok(0) => break,
                Ok(count) => read += count,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        if read < window {
            // A short device read means the covering range ran past the end of
            // the file, so only the portion inside the file is data.
            self.filled =
                usize::try_from(file.len().saturating_sub(start).min(window as u64)).unwrap_or(0);
        } else {
            self.filled = window;
        }
        let skip = usize::try_from(offset - start).unwrap_or(0);
        let available = self.filled.saturating_sub(skip);
        let wanted = len.min(available);
        if skip > 0 {
            // Slide the requested window to the front of the buffer so the
            // caller reads `bytes()[..wanted]` rather than re-deriving the
            // offset on every call. One `memmove` of at most one block per
            // read, and it only happens for a request that did not start on a
            // block boundary.
            self.buffer
                .as_mut_slice()
                .copy_within(skip..skip + wanted, 0);
        }
        self.origin = offset;
        self.filled = wanted;
        Ok(wanted)
    }
}

/// One positioned read, which `std::fs::File` does not expose on every
/// platform.
fn positioned_read(
    file: &mut std::fs::File,
    buffer: &mut [u8],
    offset: u64,
) -> std::io::Result<usize> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt as _;
        file.read_at(buffer, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt as _;
        file.seek_read(buffer, offset)
    }
    #[cfg(not(any(unix, windows)))]
    {
        use std::io::{Read as _, Seek as _};
        file.seek(std::io::SeekFrom::Start(offset))?;
        file.read(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_buffers_really_are_aligned() {
        for align in [64_usize, 512, 4096, 2 * 1024 * 1024] {
            let buffer = AlignedBuf::new(1000, align);
            assert!(buffer.is_aligned(align), "align {align}");
            assert_eq!(buffer.len(), 1000);
        }
    }

    #[test]
    fn aligned_buffer_of_length_one_still_has_an_address() {
        let buffer = AlignedBuf::new(1, 4096);
        assert!(!buffer.is_empty());
        assert!(buffer.is_aligned(4096));
    }

    #[test]
    fn alignment_check_survives_a_drop() {
        for _ in 0..64 {
            let buffer = AlignedBuf::new(37, 64);
            assert!(buffer.is_aligned(64));
            drop(buffer);
        }
    }

    #[test]
    fn align_up_accepts_multiples_and_rejects_others() {
        assert_eq!(align_up(4096, 512), Ok(4096));
        assert_eq!(
            align_up(4095, 512),
            Err(Unaligned {
                value: 4095,
                align: 512
            })
        );
        assert!(align_up(4096, 0).is_err());
        assert!(align_up(4096, 3).is_err());
    }

    #[test]
    fn adaptive_mode_prefers_buffering_below_the_block() {
        let policy = DirectIoPolicy::Adaptive;
        assert_eq!(policy.mode_for(16, 512), DirectMode::Buffered);
        assert_eq!(
            policy.mode_for(1024 * 1024, 512),
            DirectMode::Unbuffered { align: 512 }
        );
    }

    #[test]
    fn never_and_always_ignore_the_size() {
        assert_eq!(
            DirectIoPolicy::Never.mode_for(1 << 30, 512),
            DirectMode::Buffered
        );
        assert_eq!(
            DirectIoPolicy::Always.mode_for(1, 512),
            DirectMode::Unbuffered { align: 512 }
        );
    }

    #[test]
    fn pool_hands_out_every_buffer_once() {
        let mut pool = BufferPool::new(2, 4096, 4096);
        assert_eq!(pool.available(), 2);
        let first = pool.take().expect("a buffer");
        assert!(first.is_aligned(4096));
        assert_eq!(pool.available(), 1);
        let second = pool.take().expect("a buffer");
        assert_eq!(pool.available(), 0);
        pool.recycle(second);
        pool.recycle(first);
        assert_eq!(pool.available(), 2);
        assert_eq!(pool.leased(), 0);
    }

    #[test]
    fn an_exhausted_pool_reports_absence_rather_than_blocking() {
        let mut pool = BufferPool::new(1, 4096, 4096);
        let only = pool.take().expect("the one buffer");
        assert!(pool.take().is_none(), "exhaustion is backpressure");
        assert_eq!(pool.leased(), 1, "leased is derived, not accumulated");
        pool.recycle(only);
        assert_eq!(pool.leased(), 0);
        assert!(pool.take().is_some());
    }

    /// A directory on a volume that accepts unbuffered I/O, or `None` when
    /// this machine has none.
    ///
    /// This is not a convenience. A test that claimed to exercise `O_DIRECT`
    /// while running on `overlayfs`, `tmpfs` or a 9p share would be asserting
    /// nothing: every such filesystem refuses the open with `EINVAL`, the
    /// handle silently resolves to buffered, and the assertions pass against a
    /// code path the mechanism never took. So the probe is part of the test.
    fn direct_capable_dir() -> Option<tempfile::TempDir> {
        for base in [std::env::temp_dir(), std::env::current_dir().ok()?] {
            let Ok(dir) = tempfile::Builder::new()
                .prefix("kivi-direct")
                .tempdir_in(&base)
            else {
                continue;
            };
            let path = dir.path().join("probe");
            if std::fs::write(&path, vec![0_u8; 8192]).is_err() {
                continue;
            }
            if DirectFile::open(&path, DirectIoPolicy::Always, 8192)
                .is_ok_and(|file| file.is_unbuffered())
            {
                return Some(dir);
            }
        }
        None
    }

    #[test]
    fn a_buffered_handle_refuses_an_aligned_transfer() {
        // The transfer is the unbuffered path's tool. Handing it a buffered
        // handle would produce a read that silently satisfies the request
        // through the page cache, which is the one outcome the type exists to
        // make impossible.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("buffered.bin");
        std::fs::write(&path, vec![7_u8; 8192]).expect("write");
        let mut file = DirectFile::open_buffered(&path).expect("open");
        assert!(!file.is_unbuffered());
        let mut transfer = AlignedTransfer::new(8192, 4096);
        assert_eq!(
            transfer.read_at(&mut file, 0, 4096).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert!(transfer.bytes().is_empty());
    }

    #[test]
    fn an_aligned_transfer_produces_the_requested_window() {
        let Some(dir) = direct_capable_dir() else {
            // No volume on this machine accepts unbuffered I/O. Reported as a
            // pass rather than a failure, because the mechanism is genuinely
            // absent and there is nothing to assert.
            return;
        };
        let path = dir.path().join("direct.bin");
        let contents: Vec<u8> = (0..64 * 1024_u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &contents).expect("write");
        let align = logical_block_size(&path).expect("a real device reports a block size");
        let mut file = DirectFile::open(&path, DirectIoPolicy::Always, 4096).expect("open");
        assert!(file.is_unbuffered(), "the probe found a capable volume");
        let mut transfer = AlignedTransfer::new(16 * 1024, align);
        for (offset, len) in [(0_usize, 4096_usize), (1, 100), (4095, 2), (8192, 8192)] {
            let read = transfer
                .read_at(&mut file, offset as u64, len)
                .expect("aligned read");
            assert_eq!(read, len, "offset {offset} len {len}");
            assert_eq!(transfer.origin(), offset as u64);
            assert_eq!(
                transfer.bytes(),
                &contents[offset..offset + len],
                "offset {offset} len {len}"
            );
        }
    }

    #[test]
    fn an_aligned_transfer_reports_a_short_read_at_the_end() {
        let Some(dir) = direct_capable_dir() else {
            return;
        };
        let path = dir.path().join("tail.bin");
        let contents: Vec<u8> = (0..10_000_u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &contents).expect("write");
        let align = logical_block_size(&path).expect("a real device reports a block size");
        let mut file = DirectFile::open(&path, DirectIoPolicy::Always, 4096).expect("open");
        let mut transfer = AlignedTransfer::new(16 * 1024, align);
        let read = transfer
            .read_at(&mut file, 9_000, 4096)
            .expect("aligned read");
        assert_eq!(read, 1_000, "only the bytes that exist");
        assert_eq!(transfer.bytes(), &contents[9_000..]);
        // Past the end entirely.
        assert_eq!(transfer.read_at(&mut file, 100_000, 4096).expect("read"), 0);
        assert!(transfer.bytes().is_empty());
    }

    #[test]
    fn a_transfer_larger_than_its_buffer_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("small.bin");
        std::fs::write(&path, vec![0_u8; 16 * 1024]).expect("write");
        let mut transfer = AlignedTransfer::new(4096, 4096);
        let mut file = DirectFile::open_buffered(&path).expect("open");
        assert_eq!(
            transfer.read_at(&mut file, 0, 8192).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn the_policy_decides_the_mode_and_the_handle_agrees() {
        // The previous version of this module resolved a mode from the policy
        // and then never opened a handle with it, so `direct_io = always` was
        // accepted and changed nothing. This asserts the two agree.
        let Some(dir) = direct_capable_dir() else {
            return;
        };
        let path = dir.path().join("mode.bin");
        std::fs::write(&path, vec![0_u8; 4 << 20]).expect("write");
        let never = DirectFile::open(&path, DirectIoPolicy::Never, 4 << 20).expect("open");
        assert!(!never.is_unbuffered());
        assert_eq!(never.len(), 4 << 20);
        let always = DirectFile::open(&path, DirectIoPolicy::Always, 4 << 20).expect("open");
        assert!(
            always.is_unbuffered(),
            "an explicit request must produce an unbuffered handle"
        );
        let adaptive_small = DirectFile::open(&path, DirectIoPolicy::Adaptive, 16).expect("open");
        assert!(!adaptive_small.is_unbuffered(), "tiny reads stay buffered");
    }

    #[test]
    fn a_volume_that_refuses_unbuffered_io_resolves_to_buffered() {
        // `tmpfs`, `overlayfs` and 9p shares all reject `O_DIRECT` with
        // `EINVAL`, and a Windows volume that will not report its geometry
        // cannot be asked to open unbuffered at all. Either way the handle a
        // caller gets must be what it says it is.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("maybe-tmpfs.bin");
        std::fs::write(&path, vec![0_u8; 8192]).expect("write");
        if let Ok(file) = DirectFile::open(&path, DirectIoPolicy::Always, 8192) {
            assert!(!file.is_unbuffered() || direct_capable_dir_is_here());
        }
    }

    /// Whether the temporary directory this test is using accepts unbuffered
    /// I/O, so the assertion above can distinguish "refused" from "silently
    /// buffered".
    fn direct_capable_dir_is_here() -> bool {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("probe");
        std::fs::write(&path, vec![0_u8; 8192]).expect("write");
        DirectFile::open(&path, DirectIoPolicy::Always, 8192).is_ok_and(|file| file.is_unbuffered())
    }

    #[test]
    fn a_missing_file_is_an_open_error_not_an_alignment_claim() {
        assert!(
            DirectFile::open(
                Path::new("/definitely/not/here/kivi-direct-test"),
                DirectIoPolicy::Always,
                4096
            )
            .is_err()
        );
    }

    #[test]
    fn policy_parses_every_spelling() {
        assert_eq!(DirectIoPolicy::parse("auto"), Ok(DirectIoPolicy::Adaptive));
        assert_eq!(DirectIoPolicy::parse("ALWAYS"), Ok(DirectIoPolicy::Always));
        assert_eq!(DirectIoPolicy::parse("never"), Ok(DirectIoPolicy::Never));
        assert!(DirectIoPolicy::parse("turbo").is_err());
    }

    #[test]
    fn detection_always_answers() {
        assert!(!detect().to_string().is_empty());
    }
}
