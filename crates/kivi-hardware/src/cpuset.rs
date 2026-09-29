//! Cross-platform CPU sets, current-CPU queries, and thread binding.
//!
//! This is the only module in the crate that issues raw operating-system
//! calls for processor control. Everything above it works with [`CpuSet`],
//! [`CpuId`], and [`Binding`], which are project-owned and cannot express a
//! half-applied affinity.
//!
//! ## Why no affinity crate
//!
//! `core_affinity2` - the crate Kivi used before - offers exactly one
//! operation, "bind this thread to core N", and pulls `libafl_core` in to do
//! it. It cannot answer which hardware threads share a core, cannot bind a
//! thread to a set of cores, and cannot report where the thread actually
//! landed. Kivi needs a *set* (a core whose SMT siblings it may spill onto)
//! and needs the resulting CPU to be observable. The platform calls are about
//! sixty lines. `hwlocality` was evaluated and rejected: it is pre-1.0 and
//! needs a C toolchain (pkg-config, or cmake plus automake for the vendored
//! build) on a Windows support story upstream does not get right.
//!
//! ## Safety
//!
//! Every call below passes pointers to stack-allocated, correctly sized,
//! zero-initialised structures and copies the result out immediately. No
//! pointer outlives its call, no structure is read past the length the
//! platform reported, and no platform handle is retained.

use std::collections::BTreeSet;
use std::fmt;

use crate::topology::CpuId;

/// Processors per Windows processor group. A thread is bound to a
/// (group, mask) pair, and a mask is 64 bits wide, so this is the largest
/// span a single binding can express.
#[cfg(windows)]
pub const CPUS_PER_GROUP: u32 = 64;

/// A set of logical processors, ascending.
///
/// The empty set means "the platform's own choice" and is accepted everywhere
/// a set is taken, so a failed discovery never prevents startup.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CpuSet {
    cpus: BTreeSet<u32>,
}

impl CpuSet {
    /// An empty set: no restriction.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            cpus: BTreeSet::new(),
        }
    }

    /// A set containing exactly one processor.
    #[must_use]
    pub fn single(cpu: CpuId) -> Self {
        let mut cpus = BTreeSet::new();
        cpus.insert(cpu.0);
        Self { cpus }
    }

    /// A set of the given processors.
    #[must_use]
    pub fn of(cpus: impl IntoIterator<Item = u32>) -> Self {
        Self {
            cpus: cpus.into_iter().collect(),
        }
    }

    /// Every processor in `0..count`.
    #[must_use]
    pub fn range(count: u32) -> Self {
        Self {
            cpus: (0..count).collect(),
        }
    }

    /// The processors this process may currently run on.
    ///
    /// Returns `None` when the platform refuses to answer; callers treat that
    /// as unrestricted.
    #[must_use]
    pub fn current() -> Option<Self> {
        platform::allowed_cpus()
    }

    /// True when `cpu` is a member. An empty set contains nothing, so callers
    /// that mean "unrestricted" must test [`CpuSet::is_empty`] first.
    #[must_use]
    pub fn contains(&self, cpu: CpuId) -> bool {
        self.cpus.contains(&cpu.0)
    }

    /// Number of members.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cpus.len()
    }

    /// True when the set places no restriction.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cpus.is_empty()
    }

    /// Members, ascending.
    pub fn iter(&self) -> impl Iterator<Item = CpuId> + '_ {
        self.cpus.iter().map(|cpu| CpuId(*cpu))
    }

    /// Intersects with `other`.
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self {
        Self {
            cpus: self.cpus.intersection(&other.cpus).copied().collect(),
        }
    }

    /// Adds a processor.
    pub fn insert(&mut self, cpu: CpuId) -> bool {
        self.cpus.insert(cpu.0)
    }
}

impl fmt::Display for CpuSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.cpus.is_empty() {
            return write!(f, "any");
        }
        let mut iter = self.cpus.iter().peekable();
        while let Some(cpu) = iter.next() {
            let start = *cpu;
            let mut end = *cpu;
            while let Some(next) = iter.peek() {
                if **next == end + 1 {
                    end += 1;
                    iter.next();
                } else {
                    break;
                }
            }
            if start == end {
                write!(f, "{start}")?;
            } else {
                write!(f, "{start}-{end}")?;
            }
            if iter.peek().is_some() {
                write!(f, ",")?;
            }
        }
        Ok(())
    }
}

/// Parses a kernel CPU list such as `0-3,8,10-11` into its members.
///
/// The kernel's textual CPU lists are the only interface that describes an
/// arbitrary processor count without a size argument, and three callers need
/// it: the affinity mask, a memory node's processor list, and a cache
/// instance's sharing list. A single parser keeps them from disagreeing about
/// what `10-2` means.
#[must_use]
pub fn parse_cpu_list(raw: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for part in raw.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            Some((low, high)) => {
                if let (Ok(low), Ok(high)) = (low.trim().parse::<u32>(), high.trim().parse::<u32>())
                    && low <= high
                {
                    out.extend(low..=high);
                }
            }
            None => {
                if let Ok(cpu) = part.parse::<u32>() {
                    out.push(cpu);
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Where a bound thread actually runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Binding {
    /// The processor the thread was on when the binding completed.
    pub cpu: CpuId,
}

/// Why a binding could not be established.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum BindError {
    /// The set was empty, so there was nothing to bind to.
    #[error("cpu set is empty: nothing to bind")]
    EmptySet,
    /// The operating system rejected the request.
    #[error("operating system refused the cpu set {cpus}: {reason}")]
    Refused {
        /// Requested set.
        cpus: String,
        /// Platform-reported reason.
        reason: String,
    },
}

/// Binds the calling thread to `cpus` and reports where it landed.
///
/// Returns `Ok(None)` when `cpus` is empty, which is the documented
/// unrestricted case and is not an error. A non-empty set the platform
/// refuses is an error: silently running unpinned would make placement
/// diagnostics lie.
///
/// # Errors
///
/// Returns [`BindError::EmptySet`] never - an empty set returns `Ok(None)` -
/// and [`BindError::Refused`] when the platform rejects a non-empty set.
pub fn bind_current_thread(cpus: &CpuSet) -> Result<Option<Binding>, BindError> {
    if cpus.is_empty() {
        return Ok(None);
    }
    platform::bind_current_thread(cpus)?;
    let cpu = platform::current_cpu().ok_or(BindError::Refused {
        cpus: cpus.to_string(),
        reason: "thread bound but current cpu unreadable".to_string(),
    })?;
    Ok(Some(Binding { cpu }))
}

/// The processor the calling thread is running on right now.
#[must_use]
pub fn current_cpu() -> Option<CpuId> {
    platform::current_cpu()
}

#[cfg(windows)]
mod platform {
    use windows_sys::Win32::Foundation::GetLastError;
    use windows_sys::Win32::System::Kernel::PROCESSOR_NUMBER;
    use windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY;
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, GetCurrentProcessorNumberEx, GetCurrentThread, GetProcessAffinityMask,
        SetThreadGroupAffinity,
    };

    use super::{BindError, CPUS_PER_GROUP, CpuId, CpuSet};

    #[allow(unsafe_code)]
    pub(super) fn allowed_cpus() -> Option<CpuSet> {
        // The machine's processors are not the processors *this process* may
        // run on. A job object, a container CPU pin, or `start /affinity` all
        // restrict the mask, and handing a worker a CPU outside it makes the
        // bind fail on every launch. So ask for the process mask.
        let mut process_mask: usize = 0;
        let mut system_mask: usize = 0;
        // SAFETY: both out-params are correctly sized, zero-initialised locals
        // of the exact type the call writes, and the handle is a pseudo-handle
        // that needs no lifetime management. The call writes both masks or
        // fails without touching them.
        let ok = unsafe {
            GetProcessAffinityMask(
                GetCurrentProcess(),
                &raw mut process_mask,
                &raw mut system_mask,
            )
        };
        if ok != 0 {
            // `GetProcessAffinityMask` is a single-group API: the two `usize`
            // masks only describe machines of at most 64 logical processors.
            // A zero system mask means the machine is larger than the API can
            // describe, so any bits we read would be a partial view.
            if system_mask != 0 {
                return Some(CpuSet {
                    cpus: (0..usize::BITS)
                        .filter(|cpu| process_mask & (1 << cpu) != 0)
                        .collect(),
                });
            }
            None
        } else {
            None
        }
    }

    // The signature is uniform across platforms even where the platform cannot
    // fail, because the caller must be written once.
    #[allow(unsafe_code, clippy::unnecessary_wraps)]
    pub(super) fn current_cpu() -> Option<CpuId> {
        let mut number = PROCESSOR_NUMBER {
            Group: 0,
            Number: 0,
            Reserved: 0,
        };
        // SAFETY: `number` is a correctly sized, zero-initialised local of the
        // exact type the call writes, and the function has no other
        // preconditions. The binding reports no failure: it writes the current
        // processor or leaves the caller's initialisation in place, and either
        // way the value is a processor index the process may run on.
        unsafe { GetCurrentProcessorNumberEx(&raw mut number) };
        Some(CpuId(flat(
            u32::from(number.Group),
            u32::from(number.Number),
        )))
    }

    #[allow(unsafe_code)]
    pub(super) fn bind_current_thread(cpus: &CpuSet) -> Result<(), BindError> {
        let anchor = cpus.iter().next().map_or(0, |cpu| cpu.0);
        // A Windows thread is bound to a (group, mask) pair. A Kivi set that
        // straddles a group boundary binds to the group holding its anchor;
        // the remaining members stay reachable to the platform scheduler.
        let affinity = GROUP_AFFINITY {
            Mask: 1_usize << (anchor % CPUS_PER_GROUP),
            Group: u16::try_from(anchor / CPUS_PER_GROUP).unwrap_or(0),
            Reserved: [0; 3],
        };
        let mut previous = GROUP_AFFINITY::default();
        // SAFETY: `GetCurrentThread` returns a pseudo-handle valid for the
        // calling thread for the process's lifetime; both affinity structures
        // are correctly sized, zero-initialised stack locals that outlive the
        // call and are not aliased.
        let ok = unsafe {
            SetThreadGroupAffinity(GetCurrentThread(), &raw const affinity, &raw mut previous)
        };
        if ok == 0 {
            Err(BindError::Refused {
                cpus: cpus.to_string(),
                reason: last_error(),
            })
        } else {
            Ok(())
        }
    }

    #[allow(unsafe_code)]
    pub(super) fn flat(group: u32, index: u32) -> u32 {
        group * CPUS_PER_GROUP + index
    }

    #[allow(unsafe_code)]
    fn last_error() -> String {
        // SAFETY: `GetLastError` takes no arguments and has no preconditions.
        let code = unsafe { GetLastError() };
        format!("windows error {code}")
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod platform {
    use std::collections::BTreeSet;
    use std::mem::size_of;

    use super::{BindError, CpuId, CpuSet};

    #[allow(unsafe_code)]
    pub(super) fn allowed_cpus() -> Option<CpuSet> {
        // The kernel is authoritative about the mask, and `/proc/self/status`
        // is only a rendering of the same call. The fixed `cpu_set_t` covers
        // `CPU_SETSIZE` (1024) processors; a larger machine reports `EINVAL`
        // for this buffer, which is treated as "unrestricted" - the same answer
        // the platform gives when the mask is unreadable, and the one that
        // never hands a worker a processor it cannot be scheduled on.
        let mut set: libc::cpu_set_t = // SAFETY: `cpu_set_t` is a plain
            // bitmask with no invalid bit patterns, so zero is a valid value.
            unsafe { std::mem::zeroed() };
        // SAFETY: the buffer is a correctly typed, zero-initialised local, and
        // the kernel writes at most the length it is given.
        let rc = unsafe { libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &raw mut set) };
        if rc != 0 {
            return None;
        }
        let cpus: BTreeSet<u32> = (0..libc::CPU_SETSIZE as u32)
            .filter(|cpu| {
                // SAFETY: `cpu` is below `CPU_SETSIZE`, which is the bound
                // `CPU_ISSET` itself checks, and `set` is a live local.
                unsafe { libc::CPU_ISSET(*cpu as usize, &set) }
            })
            .collect();
        (!cpus.is_empty()).then_some(CpuSet { cpus })
    }

    // The signature is uniform across platforms even where the platform cannot
    // fail, because the caller must be written once.
    #[allow(unsafe_code, clippy::unnecessary_wraps)]
    pub(super) fn current_cpu() -> Option<CpuId> {
        // SAFETY: `sched_getcpu` takes no arguments, has no preconditions and
        // cannot leave memory inconsistent on failure; it returns a negative
        // value on error, which is checked below.
        let cpu = unsafe { libc::sched_getcpu() };
        // `cast_unsigned` rather than `as u32` because the sign is already
        // known to be non-negative here, and saying so is what the lint wants.
        (cpu >= 0).then_some(CpuId(cpu.cast_unsigned()))
    }

    #[allow(unsafe_code)]
    pub(super) fn bind_current_thread(cpus: &CpuSet) -> Result<(), BindError> {
        // `CPU_SET` writes one bit per processor with no bounds check of its
        // own beyond the buffer it is handed, so the index is bounded here
        // rather than trusted. A set naming a processor this platform cannot
        // represent is a refusal, not a truncated mask: a partially applied
        // affinity is exactly the half-applied state this module's types
        // exist to make unrepresentable.
        if let Some(cpu) = cpus.iter().find(|cpu| cpu.0 >= libc::CPU_SETSIZE as u32) {
            return Err(BindError::Refused {
                cpus: cpus.to_string(),
                reason: format!(
                    "cpu {} is beyond the {}-processor set this platform can express",
                    cpu.0,
                    libc::CPU_SETSIZE
                ),
            });
        }
        // SAFETY: `cpu_set_t` is a plain bitmask with no invalid bit patterns.
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        for cpu in cpus.iter() {
            // SAFETY: every index was just bounded below `CPU_SETSIZE`, and
            // `set` is a live local of exactly the type the macro writes.
            unsafe { libc::CPU_SET(cpu.0 as usize, &mut set) };
        }
        // SAFETY: only the members Kivi chose are enabled and the buffer is
        // handed to a syscall that reads exactly `size_of::<cpu_set_t>()`
        // bytes from it.
        let ok =
            unsafe { libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &raw const set) };
        if ok == 0 {
            Ok(())
        } else {
            Err(BindError::Refused {
                cpus: cpus.to_string(),
                reason: std::io::Error::last_os_error().to_string(),
            })
        }
    }
}

#[cfg(not(any(windows, target_os = "linux", target_os = "android")))]
mod platform {
    use super::{BindError, CpuId, CpuSet};

    #[allow(unsafe_code)]
    pub(super) fn allowed_cpus() -> Option<CpuSet> {
        None
    }

    // The signature is uniform across platforms even where the platform cannot
    // fail, because the caller must be written once.
    #[allow(unsafe_code, clippy::unnecessary_wraps)]
    pub(super) fn current_cpu() -> Option<CpuId> {
        None
    }

    #[allow(unsafe_code)]
    pub(super) fn bind_current_thread(cpus: &CpuSet) -> Result<(), BindError> {
        let _ = cpus;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An empty set means "unrestricted", and binding to it must be a no-op
    /// rather than a refusal: a process with no cpuset configured must still
    /// be schedulable everywhere. Run compression is what an operator reads
    /// from `/proc/self/status`, so it has to match that format.
    #[test]
    fn an_empty_set_is_unrestricted_and_runs_compress() {
        let set = CpuSet::empty();
        assert!(set.is_empty());
        assert!(!set.contains(CpuId(0)));
        assert_eq!(set.to_string(), "any");
        assert_eq!(bind_current_thread(&set), Ok(None));
        assert_eq!(CpuSet::of([0, 1, 2, 5, 9, 10]).to_string(), "0-2,5,9-10");
    }

    /// Two sets intersect to the processors both describe, which is how a
    /// placement plan intersects a role's affinity with a cgroup's.
    #[test]
    fn intersection_keeps_only_shared_processors() {
        assert_eq!(
            CpuSet::of([1, 2, 3])
                .intersect(&CpuSet::of([2, 3, 4]))
                .to_string(),
            "2-3"
        );
        assert_eq!(CpuSet::range(4).len(), 4);
        assert!(CpuSet::range(4).contains(CpuId(3)));
    }
}
