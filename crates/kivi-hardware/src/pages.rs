//! Huge-page policy for long-lived arenas.
//!
//! The friction ladder, in the order Kivi climbs it:
//!
//! 1. **Do nothing.** The default allocator already gets huge pages for free
//!    where transparent huge pages are enabled, because the kernel promotes a
//!    region once it is backed by enough 4 KiB pages. Kivi's arenas are
//!    contiguous `Vec`s, so this already happens for large ones.
//! 2. **Advise.** Ask for huge pages on one arena explicitly. This costs
//!    nothing, needs no privilege, and degrades silently to 4 KiB pages when
//!    the region does not qualify. On Linux that is
//!    `madvise(MADV_HUGEPAGE)`; on Windows it is
//!    `VirtualAlloc(MEM_LARGE_PAGES)` which needs a privilege, so it is used
//!    only when the process holds it.
//! 3. **Reserve.** Pre-commit explicit huge pages. This is a global machine
//!    setting an operator must arrange, it pins address space, and Kivi never
//!    arranges it for itself.
//!
//! Kivi never applies this to every allocation. Huge pages trade TLB pressure
//! for fragmentation and for a slower fault, and a 200-byte object gains
//! nothing from either. The policy is therefore expressed per arena, by a
//! threshold the caller chooses, and measured rather than assumed.

use std::fmt;

use crate::capability::{HugePageSupport, Support};

/// How much memory before an arena is worth advising huge pages for.
///
/// Below this, TLB pressure is not the bottleneck and the promotion cost plus
/// the fragmentation risk are not repaid.
pub const HUGE_PAGE_THRESHOLD_BYTES: u64 = 4 * 1024 * 1024;

/// What a caller wants done about a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum HugePagePolicy {
    /// Leave the mapping alone. The default, and correct for small arenas.
    #[default]
    Inherit,
    /// Ask the kernel to back this region with huge pages where it can.
    Advice,
    /// Refuse huge pages for this region, for arenas whose footprint must not
    /// perturb the system's page pool.
    Never,
}

impl HugePagePolicy {
    /// Parses the configuration spelling.
    ///
    /// # Errors
    ///
    /// Returns a message naming the accepted spellings.
    pub fn parse(spec: &str) -> Result<Self, String> {
        match spec.trim().to_ascii_lowercase().as_str() {
            "inherit" | "default" | "off" => Ok(Self::Inherit),
            "advice" | "madvise" | "thp" | "on" => Ok(Self::Advice),
            "never" | "deny" => Ok(Self::Never),
            other => Err(format!(
                "unknown huge page policy {other:?}; expected one of inherit, advice, never"
            )),
        }
    }
}

impl fmt::Display for HugePagePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Inherit => "inherit",
            Self::Advice => "advice",
            Self::Never => "never",
        };
        f.write_str(name)
    }
}

/// What the platform can do about huge pages.
#[must_use]
pub fn detect() -> Support<HugePageSupport> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        // The kernel exposes its policy in these two files. A container that
        // mounts a partial sysfs gets the conservative answer.
        let enabled = std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled")
            .ok()
            .and_then(|raw| {
                raw.split_whitespace()
                    .find(|mode| mode.starts_with('[') && mode.ends_with(']'))
                    .map(|mode| mode.trim_matches(['[', ']']).to_string())
            });
        match enabled.as_deref() {
            Some("never") | None => Support::Available(HugePageSupport::None),
            Some(_) => Support::Available(HugePageSupport::Transparent),
        }
    }
    #[cfg(windows)]
    {
        // Windows has no transparent huge pages. `MEM_LARGE_PAGES` requires
        // SeLockMemoryPrivilege and a non-empty machine-level large-page pool,
        // so Kivi checks the privilege it can check and reports the mechanism
        // rather than the pool it cannot see.
        if large_pages_available() {
            Support::Available(HugePageSupport::Reserved {
                page_bytes: 2 * 1024 * 1024,
                count: 0,
            })
        } else {
            Support::Available(HugePageSupport::None)
        }
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "android")))]
    {
        Support::Unavailable(Unavailable::UnsupportedPlatform)
    }
}

/// Asks the kernel to back a live region with huge pages.
///
/// Returns whether the advice was delivered. A `false` result is normal, not an
/// error: the region may be too small, the system may be out of huge pages, or
/// the platform may not implement advice. Kivi never treats a missing hint as a
/// failure, because the region's contents are identical either way.
///
/// Taking a slice rather than a raw pointer keeps the mapping's liveness the
/// borrow checker's problem: the caller cannot pass a region that has already
/// been freed, which is the only way this advice could be unsound.
pub fn advise_region(region: &mut [u8]) -> bool {
    if region.is_empty() {
        return false;
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[allow(unsafe_code)]
    {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        // `sysconf` returns -1 on an unknown name and a positive `long`
        // otherwise, so a conversion failure and a zero are both "no page
        // size", and neither should produce an address arithmetic bug.
        let Ok(page) = usize::try_from(page) else {
            return false;
        };
        if page == 0 {
            return false;
        }
        let address = region.as_mut_ptr() as usize;
        let len = region.len();
        // The kernel rounds the range down to a page boundary, so an unaligned
        // request is not an error, but an entirely unaligned one would be a
        // no-op and is not worth issuing.
        let start = address & !(page - 1);
        let end = (address + len + page - 1) & !(page - 1);
        if end <= start {
            return false;
        }
        // SAFETY: the range is inside `region`, which the borrow checker keeps
        // live for the duration of the call, so it is a live mapping. The
        // range is page aligned, and `MADV_HUGEPAGE` has no other
        // preconditions. A failure is reported, never asserted: the mapping
        // stays valid and simply keeps 4 KiB pages.
        let rc =
            unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE) };
        rc == 0
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        false
    }
}

/// What the kernel actually backed a live region with.
///
/// Read from `/proc/self/smaps`, because a claim about huge pages that is not
/// read back from the kernel is a claim about a `madvise` return value. A
/// `true` from [`advise_region`] means the hint was delivered; only this means
/// it was taken.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RegionBacking {
    /// Size of the mapping containing the region.
    pub virtual_bytes: u64,
    /// Resident bytes in that mapping, from `Rss`.
    pub resident_bytes: u64,
    /// Anonymous resident bytes backed by a huge page, from `AnonHugePages`.
    pub anon_huge_bytes: u64,
    /// File-backed resident bytes, from `FilePmdMapped` and `Shared_Hugetlb`.
    pub file_backed_bytes: u64,
    /// Swapped-out bytes, from `Swap`.
    pub swap_bytes: u64,
    /// The kernel's base page size for this mapping, from `KernelPageSize`.
    pub kernel_page_bytes: u64,
    /// The page size the mapping is actually presented to the MMU with, from
    /// `MMUPageSize`. A region that was promoted reports 2 MiB here and 4 KiB
    /// in `kernel_page_bytes`, which is the most direct statement the kernel
    /// makes about what happened.
    pub mmu_page_bytes: u64,
}

impl RegionBacking {
    /// Fraction of the resident set backed by a huge page, in `[0, 1]`.
    ///
    /// Zero when nothing is resident, because a ratio of huge pages to nothing
    /// is not a small ratio, it is an unmeasured one.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn huge_fraction(&self) -> f64 {
        if self.resident_bytes == 0 {
            return 0.0;
        }
        let resident = self.resident_bytes as f64;
        let huge = f64::from(u32::try_from(self.anon_huge_bytes).unwrap_or(u32::MAX)).min(resident);
        huge / resident
    }

    /// Whether the kernel backed any part of the region with a huge page.
    #[must_use]
    pub const fn promoted(&self) -> bool {
        self.anon_huge_bytes > 0
    }
}

/// Reads the kernel's accounting for the mapping containing `region`.
///
/// `None` when the platform has no such interface, or when the region does not
/// lie inside any current mapping — which on a freed or unmapped region is the
/// only answer available, and is deliberately not zero.
#[must_use]
pub fn read_backing(region: &[u8]) -> Option<RegionBacking> {
    if region.is_empty() {
        return None;
    }
    let address = region.as_ptr() as usize;
    parse_smaps(&std::fs::read_to_string("/proc/self/smaps").ok()?, address)
}

/// Parses `smaps` for the mapping containing `address`.
///
/// Blocks are delimited by their *header* line, not by a blank line. Kernels
/// differ: Linux prints a blank line between mappings, WSL2's does not, and a
/// parser that splits on `\n\n` silently returns "no mapping" for every address
/// on the second. A header is the only line whose first field is a
/// `low-high` hexadecimal range, which is a rule that holds for both layouts.
///
/// Pure, so the accounting can be tested against the format the kernel prints
/// without needing a live mapping of a particular size or a promoted one.
#[must_use]
pub fn parse_smaps(text: &str, address: usize) -> Option<RegionBacking> {
    let mut block: Vec<&str> = Vec::new();
    for line in text.lines() {
        if header_range(line).is_some() {
            if let Some(backing) = block_backing(&block, address) {
                return Some(backing);
            }
            block.clear();
        }
        block.push(line);
    }
    block_backing(&block, address)
}

/// The address range of a `smaps` header line, or `None` for a data line.
fn header_range(line: &str) -> Option<(usize, usize)> {
    let range = line.split_whitespace().next()?;
    let (from, to) = range.split_once('-')?;
    Some((
        usize::from_str_radix(from, 16).ok()?,
        usize::from_str_radix(to, 16).ok()?,
    ))
}

/// The accounting for one block, when it is the block holding `address`.
fn block_backing(block: &[&str], address: usize) -> Option<RegionBacking> {
    let (from, to) = header_range(block.first()?)?;
    if address < from || address >= to {
        return None;
    }
    let mut backing = RegionBacking {
        virtual_bytes: u64::try_from(to.saturating_sub(from)).unwrap_or(u64::MAX),
        ..RegionBacking::default()
    };
    for line in &block[1..] {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        // `Rss: 1234 kB` reports kilobytes; `MMUPageSize: 4 kB` and
        // `MMUPageSize: 2 MB` both appear, so the unit is read, not assumed.
        let mut parts = value.split_whitespace();
        let raw: u64 = parts
            .next()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        let bytes = match parts.next() {
            Some("MB") => raw.saturating_mul(1024 * 1024),
            _ => raw.saturating_mul(1024),
        };
        match key.trim() {
            "Rss" => backing.resident_bytes = bytes,
            "AnonHugePages" => backing.anon_huge_bytes = bytes,
            "FilePmdMapped" | "Shared_Hugetlb" => {
                backing.file_backed_bytes = backing.file_backed_bytes.saturating_add(bytes);
            }
            "Swap" => backing.swap_bytes = bytes,
            "KernelPageSize" => backing.kernel_page_bytes = bytes,
            "MMUPageSize" => backing.mmu_page_bytes = bytes,
            _ => {}
        }
    }
    Some(backing)
}

/// Whether the calling process may map reserved large pages.
///
/// A Windows service can hold `SeLockMemoryPrivilege`; a desktop process
/// usually cannot. Kivi checks rather than assumes, because the failure mode of
/// a wrong assumption is an allocation error on a hot path.
#[must_use]
pub fn large_pages_available() -> bool {
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{GetLastError, LUID};
        use windows_sys::Win32::Security::{
            AdjustTokenPrivileges, LUID_AND_ATTRIBUTES, LookupPrivilegeValueW,
            TOKEN_ADJUST_PRIVILEGES, TOKEN_PRIVILEGES, TOKEN_QUERY,
        };
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

        // SAFETY: every structure below is a correctly sized, zero-initialised
        // stack local. `OpenProcessToken` writes a real token handle into
        // `token`, which is only read after a non-zero return; the handle is
        // the process token, which the process owns for its whole life and
        // must not close. The privilege name is a wide string literal with a
        // terminating null. `AdjustTokenPrivileges` is given a null output
        // buffer because only the return value and `GetLastError` are read.
        #[allow(unsafe_code)]
        unsafe {
            let mut token = std::ptr::null_mut();
            if OpenProcessToken(
                GetCurrentProcess(),
                TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                &raw mut token,
            ) == 0
            {
                return false;
            }
            let name: Vec<u16> = "SeLockMemoryPrivilege\0".encode_utf16().collect();
            let mut luid = LUID {
                LowPart: 0,
                HighPart: 0,
            };
            if LookupPrivilegeValueW(std::ptr::null(), name.as_ptr(), &raw mut luid) == 0 {
                return false;
            }
            let privileges = TOKEN_PRIVILEGES {
                PrivilegeCount: 1,
                Privileges: [LUID_AND_ATTRIBUTES {
                    Luid: luid,
                    Attributes: 0x2, // SE_PRIVILEGE_ENABLED
                }],
            };
            let mut returned = 0_u32;
            let ok = AdjustTokenPrivileges(
                token,
                0,
                &raw const privileges,
                0,
                std::ptr::null_mut(),
                &raw mut returned,
            );
            if ok == 0 {
                return false;
            }
            // `AdjustTokenPrivileges` reports success even when it granted
            // nothing; `GetLastError` then carries ERROR_NOT_ALL_ASSIGNED.
            GetLastError() == 0
        }
    }
    #[cfg(not(windows))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_parses_every_spelling() {
        assert_eq!(
            HugePagePolicy::parse("inherit"),
            Ok(HugePagePolicy::Inherit)
        );
        assert_eq!(
            HugePagePolicy::parse(" MADVISE "),
            Ok(HugePagePolicy::Advice)
        );
        assert_eq!(HugePagePolicy::parse("never"), Ok(HugePagePolicy::Never));
        assert!(HugePagePolicy::parse("always").is_err());
    }

    #[test]
    fn detection_always_answers() {
        let support = detect();
        assert!(!support.to_string().is_empty());
    }

    #[test]
    fn empty_region_is_never_advised() {
        assert!(!advise_region(&mut []));
    }

    #[test]
    fn advising_a_live_region_does_not_abort() {
        // The kernel refuses a mapping it does not recognise. The contract is
        // that advice is best-effort and never fatal, so this must simply
        // return whatever the platform says.
        let mut memory = vec![0_u8; 8192];
        let _ = advise_region(&mut memory);
    }

    /// A `smaps` excerpt in the kernel's own format: one 4 KiB-backed
    /// anonymous mapping, one promoted to 2 MiB, and a file mapping between
    /// them so a positional parser would attribute fields to the wrong block.
    const SMAPS: &str = "\
00400000-00401000 r--p 00000000 08:01 1
Size:                  4 kB
Rss:                   4 kB
AnonHugePages:         0 kB
Swap:                  0 kB
KernelPageSize:        4 kB
MMUPageSize:           4 kB

7f0000000000-7f0040000000 rw-p 00000000 00:00 0
Size:              65536 kB
Rss:               65536 kB
AnonHugePages:     65536 kB
FilePmdMapped:         0 kB
Swap:                  0 kB
KernelPageSize:        4 kB
MMUPageSize:           2 MB

7f0100000000-7f0110000000 rw-s 00000000 08:02 99
Size:             262144 kB
Rss:              131072 kB
FilePmdMapped:    131072 kB
Shared_Hugetlb:        0 kB
Swap:                  0 kB
KernelPageSize:        4 kB
MMUPageSize:           4 kB
";

    #[test]
    fn a_base_page_mapping_reports_no_promotion() {
        let backing = parse_smaps(SMAPS, 0x0040_0800).expect("a live mapping");
        assert_eq!(backing.virtual_bytes, 0x1000);
        assert_eq!(backing.resident_bytes, 4096);
        assert!(!backing.promoted());
        assert_eq!(backing.kernel_page_bytes, 4096);
        assert_eq!(backing.mmu_page_bytes, 4096);
        assert!((backing.huge_fraction() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn a_promoted_mapping_reports_its_huge_pages_in_bytes_not_kilobytes() {
        let backing = parse_smaps(SMAPS, 0x7f00_2000_0000).expect("a live mapping");
        assert!(backing.promoted());
        assert_eq!(
            backing.anon_huge_bytes,
            64 * 1024 * 1024,
            "65536 kB is 64 MiB"
        );
        assert_eq!(backing.resident_bytes, 64 * 1024 * 1024);
        assert_eq!(backing.mmu_page_bytes, 2 * 1024 * 1024, "the unit is read");
        assert_eq!(backing.kernel_page_bytes, 4096, "still a 4 KiB base page");
        assert!((backing.huge_fraction() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_file_mapping_reports_its_own_fields_and_none_of_the_anonymous_ones() {
        let backing = parse_smaps(SMAPS, 0x7f01_0500_0000).expect("a live mapping");
        assert!(
            !backing.promoted(),
            "a file mapping has no AnonHugePages line"
        );
        assert_eq!(backing.anon_huge_bytes, 0);
        assert_eq!(backing.file_backed_bytes, 128 * 1024 * 1024);
        assert_eq!(backing.resident_bytes, 128 * 1024 * 1024);
        assert_eq!(backing.virtual_bytes, 256 * 1024 * 1024);
    }

    #[test]
    fn an_address_outside_every_mapping_is_absent_not_zero() {
        // A freed or unmapped region has no accounting. Reporting zero would
        // make "the kernel gave me nothing" and "the kernel told me nothing"
        // the same answer.
        assert!(parse_smaps(SMAPS, 0xDEAD_0000).is_none());
    }

    /// The same content with the blank lines removed, which is what WSL2's
    /// kernel prints. A parser that delimits blocks on `\n\n` finds nothing
    /// here, and the honest-looking answer it returns — no accounting — hides
    /// a mapping that plainly exists.
    const SMAPS_NO_BLANK_LINES: &str = "\
00400000-00401000 r--p 00000000 08:01 1
Size:                  4 kB
Rss:                   4 kB
AnonHugePages:         0 kB
MMUPageSize:           4 kB
7f0000000000-7f0040000000 rw-p 00000000 00:00 0
Size:              65536 kB
Rss:               65536 kB
AnonHugePages:     65536 kB
MMUPageSize:           2 MB
7f0100000000-7f0110000000 rw-s 00000000 08:02 99
Size:             262144 kB
Rss:              131072 kB
FilePmdMapped:    131072 kB
";

    #[test]
    fn blocks_are_delimited_by_their_header_not_by_a_blank_line() {
        let promoted = parse_smaps(SMAPS_NO_BLANK_LINES, 0x7f00_2000_0000)
            .expect("a live mapping without blank-line separators");
        assert!(promoted.promoted());
        assert_eq!(promoted.anon_huge_bytes, 64 * 1024 * 1024);
        assert_eq!(promoted.mmu_page_bytes, 2 * 1024 * 1024);
        // The last block must not inherit the previous block's huge pages.
        let file_backed = parse_smaps(SMAPS_NO_BLANK_LINES, 0x7f01_0500_0000)
            .expect("a live mapping without blank-line separators");
        assert!(!file_backed.promoted());
        assert_eq!(file_backed.anon_huge_bytes, 0);
        assert_eq!(file_backed.file_backed_bytes, 128 * 1024 * 1024);
        // And the first, which shares a block boundary with a promoted one.
        let first = parse_smaps(SMAPS_NO_BLANK_LINES, 0x0040_0800).expect("the first block");
        assert!(!first.promoted());
        assert_eq!(first.resident_bytes, 4096);
    }

    #[test]
    fn a_path_containing_a_colon_does_not_confuse_the_block_delimiter() {
        // A header's device field is itself `major:minor`, and a mapped path can
        // contain a colon. Neither may make a data line look like a header.
        let with_path = "\
7f0000000000-7f0000100000 r--p 00000000 08:01 42 /mnt/we:ird/dir
Size:                  64 kB
Rss:                   64 kB
AnonHugePages:          0 kB
MMUPageSize:           4 kB
";
        let backing = parse_smaps(with_path, 0x7f00_0000_8000).expect("a mapped file");
        assert_eq!(backing.resident_bytes, 64 * 1024);
        assert!(!backing.promoted());
    }

    #[test]
    fn a_region_with_nothing_resident_has_an_unmeasured_ratio() {
        let backing = RegionBacking {
            resident_bytes: 0,
            anon_huge_bytes: 0,
            ..RegionBacking::default()
        };
        assert!((backing.huge_fraction() - 0.0).abs() < 1e-9);
        assert!(!backing.promoted());
    }

    #[test]
    fn huge_bytes_never_exceed_the_resident_set() {
        // A kernel that reported more huge bytes than resident bytes would make
        // the ratio exceed one; the reader clamps rather than claiming a
        // fraction of a set that does not exist.
        let backing = RegionBacking {
            resident_bytes: 4096,
            anon_huge_bytes: 8 * 1024 * 1024,
            ..RegionBacking::default()
        };
        assert!(backing.huge_fraction() <= 1.0);
    }

    #[test]
    fn reading_a_live_region_backing_answers_or_declines() {
        let mut memory = vec![0_u8; 1 << 20];
        // Whether the platform answers at all is a machine fact. What must hold
        // either way is that an answer describes *some* mapping containing this
        // region, and that a decline is not a zero.
        if let Some(backing) = read_backing(&memory) {
            assert!(backing.virtual_bytes >= memory.len() as u64);
            assert!(backing.resident_bytes <= backing.virtual_bytes);
        }
        assert!(read_backing(&[]).is_none());
        memory.clear();
    }
}
