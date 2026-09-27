//! CXL and pmem memory regions, exposed as a capacity map rather than as a
//! correctness-bearing store.
//!
//! Linux surfaces CXL capacity in two shapes, and Kivi treats them
//! differently:
//!
//! * **Device DAX** — `/dev/daxN.Y`, a character device that can be mapped
//!   directly. Detected in [`discover_regions`] and used by [`DaxRegion`].
//! * **kmem hotplug** — the same capacity hotplugged into the system, appearing
//!   as an ordinary NUMA node whose tier is `Cxl` or `Pmem`. Nothing here maps
//!   it: the memory fabric sees it as a memory node and treats it as any other
//!   node, which is correct because it is one.
//!
//! ## Persistence is never assumed
//!
//! A CXL Type-3 device may be volatile (cache- or memory-mode without
//! persistence) or persistent, depending on the device and on how firmware and
//! the kernel were configured. CXL 3.1 added explicit persistence signalling
//! precisely because the distinction is not inferable from the hardware shape.
//! So [`MemoryRegion::persistent`] is `false` unless a backend proved it, and
//! no durability contract in Kivi may name a region that did not.
//!
//! ## Disappearing is normal
//!
//! A DAX region can be removed while the process runs, and a region present at
//! startup can be gone after a firmware event. Every [`DaxRegion`] operation
//! therefore validates its range against the mapping and reports a clean
//! failure, and the memory fabric treats that failure as "this provider is
//! unavailable" rather than as a statement about the data.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use crate::capability::MemoryRegion;
use crate::topology::{IoDevice, MemoryTier, NumaNodeId, Topology};

/// Default mapping granularity. The kernel maps a DAX region in 2 MiB chunks
/// and will not fault a smaller one, so a smaller alignment is not a tuning
/// choice, it is a refusal.
pub const DAX_ALIGNMENT_BYTES: u64 = 2 * 1024 * 1024;

/// Bytes in one allocator unit, which is the mapping granularity.
const UNIT_BYTES: u64 = DAX_ALIGNMENT_BYTES;

/// Regions the machine offers.
#[must_use]
pub fn discover_regions(topology: &Topology) -> Vec<MemoryRegion> {
    let mut out: Vec<MemoryRegion> = crate::device::dax_regions()
        .iter()
        .filter_map(MemoryRegion::from_device)
        .collect();
    // A hotplugged CXL or pmem node is a region the fabric can use without
    // mapping anything. It is named here so the capability snapshot is honest,
    // and the memory fabric reaches it through the topology's memory nodes
    // rather than through a mapping.
    for node in &topology.memory_nodes {
        if !matches!(node.tier, MemoryTier::Cxl | MemoryTier::Pmem) {
            continue;
        }
        if out.iter().any(|region| region.numa_node == Some(node.id)) {
            continue;
        }
        out.push(MemoryRegion {
            path: format!("/sys/devices/system/node/{}", node.id),
            capacity_bytes: node.total_bytes.unwrap_or(0),
            align_bytes: DAX_ALIGNMENT_BYTES,
            numa_node: Some(node.id),
            persistent: false,
        });
    }
    out
}

impl MemoryRegion {
    /// Builds a region description from a discovered DAX character device.
    ///
    /// Capacity and alignment come from the region's sysfs directory. A region
    /// whose size cannot be read is not a region: without a capacity the
    /// allocator cannot bound itself, and an unbounded allocator over a device
    /// that can be removed at any moment is a way to lose writes.
    #[must_use]
    pub fn from_device(device: &IoDevice) -> Option<Self> {
        let sysfs = PathBuf::from(&device.path);
        let size = read_u64(&sysfs.join("size"))?;
        let mapping_aligned = read_u64(&sysfs.join("mapping_aligned_address")).unwrap_or(0);
        Some(Self {
            path: device.name.clone(),
            capacity_bytes: size,
            align_bytes: mapping_aligned.max(DAX_ALIGNMENT_BYTES),
            numa_node: device.numa_node,
            persistent: false,
        })
    }
}

fn read_u64(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
}

/// A mapped CXL or pmem region.
///
/// The mapping is process memory backed by a device. Reads and writes are
/// ordinary memory operations, so the region's contents survive everything
/// except the device itself leaving, which is why no durability contract names
/// the region without proof.
#[derive(Debug)]
pub struct DaxRegion {
    /// Character device or filesystem path the mapping came from.
    path: PathBuf,
    /// Bytes mapped.
    len: usize,
    /// The mapping.
    base: *mut u8,
    /// Memory node the region belongs to.
    numa_node: Option<NumaNodeId>,
    /// Extent accounting over the mapping.
    allocator: BoundedAllocator,
}

impl DaxRegion {
    /// Maps a region for reading and writing.
    ///
    /// # Errors
    ///
    /// Returns the operating-system error when the path cannot be opened or
    /// mapped. A CXL region that is absent produces the same error as any other
    /// missing file, and the caller treats it as an unavailable provider.
    pub fn map(region: &MemoryRegion) -> io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&region.path)?;
        let len = usize::try_from(region.capacity_bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "region too large"))?;
        if len == 0 {
            return Err(Self::no_capacity(&region.path));
        }
        let base = sys::map_shared(&file, len)?;
        Ok(Self {
            path: PathBuf::from(&region.path),
            len,
            base,
            numa_node: region.numa_node,
            allocator: BoundedAllocator::new(len as u64 / UNIT_BYTES),
        })
    }

    /// Wraps an anonymous mapping so that the region lifecycle and allocator
    /// can be exercised on a machine with no CXL hardware.
    ///
    /// This is how Kivi tests that a provider can disappear, that alignment is
    /// enforced, and that the allocator refuses to overcommit — without ever
    /// claiming the numbers describe CXL. A benchmark against an emulated
    /// region is reported as emulated.
    ///
    /// # Errors
    ///
    /// Returns the operating-system error when the mapping cannot be created.
    pub fn map_anonymous(len: usize, numa_node: Option<NumaNodeId>) -> io::Result<Self> {
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "region has no capacity",
            ));
        }
        let base = sys::map_anonymous(len)?;
        Ok(Self {
            path: PathBuf::from("<anonymous>"),
            len,
            base,
            numa_node,
            allocator: BoundedAllocator::new(len as u64 / UNIT_BYTES),
        })
    }

    fn no_capacity(path: &str) -> io::Error {
        io::Error::other(format!("cxl region {path} reports no capacity"))
    }

    /// Bytes mapped.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when the mapping holds no bytes, which never happens for a
    /// successfully mapped region.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Memory node the region belongs to.
    #[must_use]
    pub fn numa_node(&self) -> Option<NumaNodeId> {
        self.numa_node
    }

    /// The path the mapping came from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Bytes the allocator has handed out.
    #[must_use]
    pub fn allocated_bytes(&self) -> u64 {
        self.allocator.allocated_bytes()
    }

    /// Total capacity in allocator units.
    #[must_use]
    pub fn total_units(&self) -> u64 {
        self.allocator.total_units()
    }

    /// Takes `bytes` from the region, rounded up to the mapping granularity, or
    /// `None` when the region is exhausted.
    pub fn allocate(&mut self, bytes: u64) -> Option<u64> {
        self.allocator.allocate(units_for(bytes))
    }

    /// Returns a range to the region. A range that was never handed out is
    /// refused rather than ignored, because a double release would let two
    /// objects share one extent.
    pub fn release(&mut self, offset: u64, bytes: u64) -> bool {
        self.allocator.release(offset, units_for(bytes))
    }

    /// Reads a range of the region.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidInput`] when the range is not entirely
    /// inside the mapping.
    pub fn read(&self, offset: u64, len: usize) -> io::Result<&[u8]> {
        let start = usize::try_from(offset)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "offset overflow"))?;
        self.as_slice()
            .get(start..start + len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range outside region"))
    }

    /// Writes a range of the region.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidInput`] when the range is not entirely
    /// inside the mapping.
    pub fn write(&mut self, offset: u64, bytes: &[u8]) -> io::Result<()> {
        let start = usize::try_from(offset)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "offset overflow"))?;
        let end = start
            .checked_add(bytes.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range overflow"))?;
        let slot = self
            .as_mut_slice()
            .get_mut(start..end)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range outside region"))?;
        slot.copy_from_slice(bytes);
        Ok(())
    }

    #[allow(unsafe_code)]
    fn as_slice(&self) -> &[u8] {
        // SAFETY: `base` points at a live mapping of `len` bytes owned by this
        // struct, and this shared borrow proves no mutable borrow is live.
        unsafe { std::slice::from_raw_parts(self.base, self.len) }
    }

    #[allow(unsafe_code)]
    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as `as_slice`, and `&mut self` proves no other reference
        // into the mapping is alive.
        unsafe { std::slice::from_raw_parts_mut(self.base, self.len) }
    }
}

impl Drop for DaxRegion {
    fn drop(&mut self) {
        // SAFETY: `base` and `len` are exactly what the mapping call returned
        // and accepted, and this is the only unmapping, running with no other
        // reference into the mapping alive.
        sys::unmap(self.base, self.len);
    }
}

impl fmt::Display for DaxRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "dax(path={}, {mi}MiB, node={node:?})",
            self.path.display(),
            mi = self.len / (1024 * 1024),
            node = self.numa_node.map(|id| id.0),
        )
    }
}

/// Why a DAX operation failed. `io::Error` is not `Clone`, so this type is not.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DaxError {
    /// The region could not be opened or mapped.
    #[error("cxl region {path} could not be mapped: {source}")]
    Unmappable {
        /// Region path.
        path: String,
        /// Operating-system error.
        source: io::Error,
    },
    /// The region has no capacity, so nothing can be bounded.
    #[error("cxl region {path} reports no capacity")]
    NoCapacity {
        /// Region path.
        path: String,
    },
    /// The region is out of extents.
    #[error("cxl region is exhausted")]
    Exhausted,
}

fn units_for(bytes: u64) -> u64 {
    bytes.div_ceil(UNIT_BYTES).max(1)
}

/// A first-fit extent allocator over a region, in the region's own
/// granularity.
///
/// A bump allocator would be simpler and would be wrong: a region that can be
/// unmapped needs its extents reclaimable, and an allocator that hands the same
/// bytes to two objects gives its owner no way to retire either one. Bounded
/// extent tracking is the smallest thing that models the real lifecycle.
#[derive(Debug)]
struct BoundedAllocator {
    /// Total extents.
    total_units: u64,
    /// Free extents, sorted by offset and merged.
    free: Vec<(u64, u64)>,
    /// Units handed out.
    allocated: u64,
}

impl BoundedAllocator {
    fn new(total_units: u64) -> Self {
        Self {
            total_units,
            free: vec![(0, total_units)],
            allocated: 0,
        }
    }

    const fn total_units(&self) -> u64 {
        self.total_units
    }

    const fn allocated_bytes(&self) -> u64 {
        self.allocated * UNIT_BYTES
    }

    fn allocate(&mut self, units: u64) -> Option<u64> {
        if units == 0 {
            return None;
        }
        let index = self.free.iter().position(|(_, len)| *len >= units)?;
        let (start, len) = self.free[index];
        match len - units {
            0 => {
                self.free.remove(index);
            }
            rest => self.free[index] = (start + units, rest),
        }
        self.allocated += units;
        Some(start)
    }

    fn release(&mut self, start: u64, units: u64) -> bool {
        if units == 0 || start >= self.total_units || units > self.total_units - start {
            return false;
        }
        // A range that still lies inside a free extent was never handed out.
        // Refusing keeps a double release from letting two objects share one
        // extent, which is the failure this allocator exists to prevent.
        if self
            .free
            .iter()
            .any(|(offset, len)| start >= *offset && start + units <= offset + *len)
        {
            return false;
        }
        self.free.push((start, units));
        self.free.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(self.free.len());
        for (offset, len) in self.free.drain(..) {
            match merged.last_mut() {
                Some(last) if last.0 + last.1 == offset => last.1 += len,
                _ => merged.push((offset, len)),
            }
        }
        self.free = merged;
        self.allocated = self.allocated.saturating_sub(units);
        true
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod sys {
    use std::fs::File;
    use std::io;
    use std::os::fd::AsRawFd;
    use std::ptr;

    #[allow(unsafe_code)]
    pub(super) fn map_shared(file: &File, len: usize) -> io::Result<*mut u8> {
        // SAFETY: the descriptor is open for the duration of the call, `len` is
        // non-zero and was validated by the caller, and `MAP_FAILED` is turned
        // into an `io::Error` here. `MAP_SHARED` is required: writes through a
        // private mapping of a device would never reach the device.
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            Err(io::Error::last_os_error())
        } else {
            Ok(ptr.cast::<u8>())
        }
    }

    #[allow(unsafe_code)]
    pub(super) fn map_anonymous(len: usize) -> io::Result<*mut u8> {
        // SAFETY: a null hint lets the kernel choose, `len` is non-zero and
        // validated by the caller, and `MAP_FAILED` is turned into an
        // `io::Error` here. The mapping is shared so a second mapping of the
        // same region observes the same bytes.
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            Err(io::Error::last_os_error())
        } else {
            Ok(ptr.cast::<u8>())
        }
    }

    #[allow(unsafe_code)]
    pub(super) fn unmap(base: *mut u8, len: usize) {
        // SAFETY: `base` and `len` are what the mapping call returned and
        // accepted, and this is the only unmapping, running with no other
        // reference into the mapping alive.
        unsafe {
            libc::munmap(base.cast::<libc::c_void>(), len);
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
mod sys {
    use std::io;

    #[allow(unsafe_code)]
    pub(super) fn map_shared(_file: &std::fs::File, _len: usize) -> io::Result<*mut u8> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "device dax mapping is a linux facility",
        ))
    }

    #[allow(unsafe_code)]
    pub(super) fn map_anonymous(len: usize) -> io::Result<*mut u8> {
        // SAFETY: a read/write reserved-and-committed mapping of `len` bytes at
        // an address the kernel chooses. A null return is turned into an
        // `io::Error` here. The region is decommitted and released exactly
        // once, in `DaxRegion::drop`.
        let ptr = unsafe {
            windows_sys::Win32::System::Memory::VirtualAlloc(
                std::ptr::null_mut(),
                len,
                windows_sys::Win32::System::Memory::MEM_COMMIT
                    | windows_sys::Win32::System::Memory::MEM_RESERVE,
                windows_sys::Win32::System::Memory::PAGE_READWRITE,
            )
        };
        if ptr.is_null() {
            Err(io::Error::last_os_error())
        } else {
            Ok(ptr.cast::<u8>())
        }
    }

    #[allow(unsafe_code)]
    pub(super) fn unmap(base: *mut u8, _len: usize) {
        // SAFETY: `base` is what `VirtualAlloc` returned and accepted, and
        // this is the only release, running with no other reference into the
        // mapping alive. `MEM_RELEASE` requires the size to be zero, which the
        // platform contract specifies.
        unsafe {
            windows_sys::Win32::System::Memory::VirtualFree(
                base.cast(),
                0,
                windows_sys::Win32::System::Memory::MEM_RELEASE,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNITS: u64 = 16;

    fn allocator() -> BoundedAllocator {
        BoundedAllocator::new(UNITS)
    }

    #[test]
    fn allocation_is_bounded() {
        let mut pool = allocator();
        assert_eq!(pool.allocate(UNITS), Some(0));
        assert_eq!(pool.allocate(1), None, "a full region hands out nothing");
        assert_eq!(pool.total_units(), UNITS);
    }

    #[test]
    fn first_fit_reuses_the_lowest_free_extent() {
        let mut pool = allocator();
        assert_eq!(pool.allocate(4), Some(0));
        assert_eq!(pool.allocate(4), Some(4));
        assert_eq!(pool.allocate(4), Some(8));
        assert!(pool.release(4, 4));
        assert_eq!(pool.allocate(4), Some(4));
    }

    #[test]
    fn zero_sized_allocation_is_refused() {
        let mut pool = allocator();
        assert_eq!(pool.allocate(0), None);
        assert!(!pool.release(0, 0));
    }

    #[test]
    fn releasing_something_never_handed_out_is_refused() {
        let mut pool = allocator();
        assert!(!pool.release(UNITS - 1, 2));
        assert!(!pool.release(0, 1));
    }

    #[test]
    fn a_double_release_is_refused() {
        let mut pool = allocator();
        let start = pool.allocate(2).expect("room");
        assert!(pool.release(start, 2));
        assert!(!pool.release(start, 2), "a released extent is free again");
        assert_eq!(pool.allocated_bytes(), 0);
    }

    #[test]
    fn allocation_accounting_returns_to_zero() {
        let mut pool = allocator();
        for _ in 0..8 {
            let start = pool.allocate(2).expect("room");
            assert!(pool.release(start, 2));
        }
        assert_eq!(pool.allocated_bytes(), 0);
    }

    #[test]
    fn an_emulated_region_behaves_like_a_device() {
        // 64 MiB is 32 units at the 2 MiB granularity a real DAX mapping
        // requires. This is the machine-without-CXL test: the provider's
        // lifecycle and bounds are exercised and no claim is made about CXL
        // performance.
        let mut region = DaxRegion::map_anonymous(64 * 1024 * 1024, Some(NumaNodeId(1)))
            .expect("anonymous mapping");
        assert_eq!(region.len(), 64 * 1024 * 1024);
        assert_eq!(region.numa_node(), Some(NumaNodeId(1)));
        let start = region.allocate(2 * 1024 * 1024).expect("capacity");
        region.write(start, b"cxl").expect("inside the region");
        assert_eq!(region.read(start, 3).expect("inside"), b"cxl");
        assert!(region.write(u64::MAX, b"x").is_err());
        assert!(region.release(start, 2 * 1024 * 1024));
    }

    #[test]
    fn an_exhausted_region_reports_exhaustion_not_corruption() {
        let mut region = DaxRegion::map_anonymous(4 * 1024 * 1024, None).expect("mapping");
        assert!(region.allocate(2 * 1024 * 1024).is_some());
        assert!(region.allocate(4 * 1024 * 1024).is_none());
        // The mapping is still writable where it exists: bounds are checked
        // against the mapping, not against the allocator, because a Memory
        // Fabric provider is what turns "allocated" into "mine to touch".
        // Exposing a whole mapping for read and write is what a mapped device
        // is; refusing an unallocated range here would be a second, different
        // allocator hidden inside a mapping guard.
        assert!(region.write(0, &[0; 16]).is_ok());
        assert!(region.write(u64::MAX, b"x").is_err());
    }

    #[test]
    fn a_removed_region_is_absent_rather_than_fatal() {
        let region = MemoryRegion {
            path: "/dev/dax-does-not-exist".to_string(),
            capacity_bytes: DAX_ALIGNMENT_BYTES,
            align_bytes: DAX_ALIGNMENT_BYTES,
            numa_node: None,
            persistent: false,
        };
        assert!(DaxRegion::map(&region).is_err());
    }

    #[test]
    fn discovery_names_nothing_on_a_machine_without_tiered_memory() {
        assert!(discover_regions(&Topology::single_core()).is_empty());
    }
}
