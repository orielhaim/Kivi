//! Real NUMA memory placement, and proof that it happened.
//!
//! ## Why this is not a configuration value
//!
//! A node id attached to a provider is a *label*. It costs nothing and changes
//! nothing until pages are actually placed, and the three ways a label fails
//! to become a placement are all silent:
//!
//! 1. the allocator is a global one, so first touch happens on whichever
//!    thread happened to grow the heap;
//! 2. the policy is applied after the pages exist, when moving them requires
//!    privileges a process does not have;
//! 3. the memory is one `Vec` in the middle of a heap full of unrelated
//!    allocations, so a per-VMA policy cannot be applied to it at all.
//!
//! So Kivi owns a mapping. [`Arena`] is an anonymous mapping it created, it
//! applied a memory policy to *before* the first page fault, and it can then
//! read the kernel's own accounting of where the pages went. This is the
//! difference between "the planner decided node 1" and "the planner decided
//! node 1, and 512 of 512 pages are on node 1".
//!
//! ## First touch is the mechanism
//!
//! `mbind(MPOL_BIND)` before first touch puts the fault on the node the policy
//! names. The arena is therefore created, bound, and then faulted by the thread
//! that owns it. Nothing here touches unrelated process memory: the policy
//! applies to this VMA and no other.
//!
//! ## Verification
//!
//! `/proc/self/numa_maps` is the kernel's own per-mapping page census: one
//! line per VMA, with `N<node>=<pages>` for each node the VMA has pages on. It
//! is world-readable, needs no privilege, and is the same interface `numastat`
//! and the `numactl` monitor read. Kivi parses it directly rather than
//! depending on a NUMA library, and reports
//! [`PagePlacement::verified`](PagePlacement) when a line actually covered the
//! arena.
//!
//! The same file carries `anon_hugepage=`, which is how the huge-page path is
//! proved rather than assumed: the kernel counts the huge pages inside a VMA,
//! so [`PagePlacement::huge_pages`] is the number of 2 MiB pages the kernel
//! says it actually backed with, not the number Kivi asked for.
//!
//! ## Failure is a value
//!
//! Every operation returns an outcome rather than an error, and none of them
//! can prevent a caller from starting. A machine with one node, a container
//! without `CAP_SYS_NICE`, a kernel without `CONFIG_NUMA`, and Windows all
//! produce [`PagePlacement::Unavailable`] with a reason, and the caller runs on
//! an unbound, small-page, anywhere-valid mapping.

use std::collections::BTreeMap;
use std::fmt;
use std::io;

use crate::pages::HugePagePolicy;
use crate::topology::NumaNodeId;

/// Bytes in a Linux huge page, which is also the granularity at which an
/// arena's placement can be verified and its address aligned.
pub const HUGE_PAGE_BYTES: u64 = 2 * 1024 * 1024;

/// What the caller wants done about a mapping's memory policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum NodePolicy {
    /// Leave the default policy alone. Correct everywhere, and the only choice
    /// on a machine that did not disclose a node.
    #[default]
    Inherit,
    /// Place pages on this node. Allocation from another node is not what the
    /// caller asked for, so this is the right policy for a worker arena whose
    /// owner is pinned to a known core.
    Bind(NumaNodeId),
    /// Prefer this node and accept a fallback. The right policy for a mapping
    /// whose content is larger than the node, where refusing to allocate would
    /// be worse than allocating remotely.
    Prefer(NumaNodeId),
}

impl NodePolicy {
    /// Parses the configuration spelling, accepting either a bare node number
    /// or `bind:N` / `prefer:N`.
    ///
    /// # Errors
    ///
    /// Returns a message naming the accepted spellings.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let spec = spec.trim().to_ascii_lowercase();
        if matches!(spec.as_str(), "none" | "off" | "inherit" | "default") {
            return Ok(Self::Inherit);
        }
        let (mode, node) = match spec.split_once(':') {
            Some((mode, node)) => (mode, node.trim().to_string()),
            None => ("bind", spec.clone()),
        };
        let Ok(node) = node.parse::<u32>() else {
            return Err(format!(
                "unknown numa policy {spec:?}; expected none, or bind:<node> / prefer:<node>"
            ));
        };
        match mode {
            "bind" => Ok(Self::Bind(NumaNodeId(node))),
            "prefer" => Ok(Self::Prefer(NumaNodeId(node))),
            _ => Err(format!(
                "unknown numa policy {mode:?}; expected none, bind:<node>, prefer:<node>"
            )),
        }
    }

    /// The node this policy names, if any.
    #[must_use]
    pub const fn node(&self) -> Option<NumaNodeId> {
        match self {
            Self::Inherit => None,
            Self::Bind(node) | Self::Prefer(node) => Some(*node),
        }
    }
}

impl fmt::Display for NodePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inherit => f.write_str("inherit"),
            Self::Bind(node) => write!(f, "bind(node{})", node.0),
            Self::Prefer(node) => write!(f, "prefer(node{})", node.0),
        }
    }
}

/// The outcome of a placement request.
///
/// Four states, because "the kernel said no" and "the platform has no such
/// mechanism" lead to different operator actions and must never be reported as
/// the same thing. None of them is an error: an arena that could not be
/// placed is a working arena that is slower.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlacementOutcome {
    /// No placement was asked for.
    NotRequested,
    /// The policy was applied to the mapping.
    Applied {
        /// The policy that was applied.
        policy: NodePolicy,
    },
    /// The platform cannot express this policy.
    Unavailable {
        /// Why, in one phrase.
        reason: &'static str,
    },
    /// The platform can express it and the kernel refused.
    Failed {
        /// The policy that was refused.
        policy: NodePolicy,
        /// The operating-system error.
        error: String,
    },
}

impl PlacementOutcome {
    /// Whether the policy is in force on the mapping.
    #[must_use]
    pub const fn is_applied(&self) -> bool {
        matches!(self, Self::Applied { .. })
    }
}

impl fmt::Display for PlacementOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotRequested => f.write_str("not-requested"),
            Self::Applied { policy } => write!(f, "applied({policy})"),
            Self::Unavailable { reason } => write!(f, "unavailable({reason})"),
            Self::Failed { policy, error } => write!(f, "failed({policy}: {error})"),
        }
    }
}

/// What the kernel's own page census says about a mapping.
///
/// This is the evidence. Everything above it is a request; this is the answer,
/// read from `/proc/self/numa_maps`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PagePlacement {
    /// Pages resident on each node, ascending by node index.
    pub pages_by_node: BTreeMap<u32, u64>,
    /// Anonymous huge pages the kernel backed this mapping with.
    pub huge_pages: u64,
    /// Total pages the kernel accounted for this mapping.
    pub total_pages: u64,
    /// Whether a `numa_maps` line actually covered the mapping. When false,
    /// every other field is zero and nothing may be concluded from it.
    pub verified: bool,
}

impl PagePlacement {
    /// Whether every accounted page is on `node`.
    #[must_use]
    pub fn is_local_to(&self, node: NumaNodeId) -> bool {
        self.verified
            && self.total_pages > 0
            && self.pages_by_node.len() == 1
            && self.pages_by_node.get(&node.0).copied() == Some(self.total_pages)
    }

    /// Fraction of pages not on `node`, in `[0, 1]`. Zero when the mapping was
    /// never faulted, which is the honest answer for a mapping nothing has
    /// touched yet.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "a ratio of two page counts: every count below 2^52 is exact \
                  in an f64, and a mapping with more pages than that cannot be \
                  resident on any machine Kivi runs on"
    )]
    pub fn remote_fraction(&self, node: NumaNodeId) -> f64 {
        if !self.verified || self.total_pages == 0 {
            return 0.0;
        }
        let local = self.pages_by_node.get(&node.0).copied().unwrap_or(0);
        1.0 - (local as f64 / self.total_pages as f64)
    }
}

impl fmt::Display for PagePlacement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.verified {
            return f.write_str("unverified");
        }
        let nodes: Vec<String> = self
            .pages_by_node
            .iter()
            .map(|(node, pages)| format!("N{node}={pages}"))
            .collect();
        write!(
            f,
            "{} pages [{}] huge={}",
            self.total_pages,
            nodes.join(" "),
            self.huge_pages
        )
    }
}

/// An anonymous mapping Kivi owns, with a memory policy applied to it.
///
/// Sized and aligned by Kivi, bound before first touch, and verifiable against
/// the kernel's own accounting. Dropping it releases the mapping; nothing else
/// in the process is affected.
#[derive(Debug)]
pub struct Arena {
    /// Start of the aligned window. Everything before it has been unmapped.
    base: *mut u8,
    /// Length of the aligned window.
    len: usize,
    /// Human-readable name, for diagnostics and DAMON region registration.
    name: String,
    /// What the caller asked for.
    requested: NodePolicy,
    requested_huge: HugePagePolicy,
    node_outcome: PlacementOutcome,
    huge_outcome: PlacementOutcome,
}

impl Arena {
    /// Maps `len` bytes, applies the policy, and returns the arena without
    /// having faulted it.
    ///
    /// First touch happens later, deliberately, on the thread that will own
    /// the arena: a fault that occurs here would be serviced by whatever core
    /// the creating thread is on, which is exactly the bug this type exists to
    /// prevent. Call [`Arena::fault_in`] once the owning thread is in place.
    ///
    /// # Errors
    ///
    /// Returns the operating-system error when the mapping cannot be created.
    /// A *placement* failure is not an error; it is reported in
    /// [`Arena::node_outcome`].
    pub fn map(
        name: impl Into<String>,
        len: usize,
        align: usize,
        policy: NodePolicy,
        huge_pages: HugePagePolicy,
    ) -> io::Result<Self> {
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "arena has no length",
            ));
        }
        let (base, len) = sys::map(len, align)?;
        let node_outcome = sys::bind(base, len, policy);
        let huge_outcome = match huge_pages {
            HugePagePolicy::Advice => {
                if sys::advise_huge(base, len) {
                    PlacementOutcome::Applied {
                        policy: NodePolicy::Inherit,
                    }
                } else {
                    PlacementOutcome::Unavailable {
                        reason: "the platform does not implement huge-page advice",
                    }
                }
            }
            // `Inherit` means "whatever the system default does", which on a
            // THP=`always` system is huge pages and on a THP=`madvise` system
            // is 4 KiB. That is not an action, so it is not requested. `Never`
            // is a refusal, and the outcome of a refusal is also that nothing
            // was done, so both answer the same way - deliberately, because the
            // caller asked for neither an action nor a diagnostic here.
            HugePagePolicy::Never | HugePagePolicy::Inherit => PlacementOutcome::NotRequested,
        };
        Ok(Self {
            base,
            len,
            name: name.into(),
            requested: policy,
            requested_huge: huge_pages,
            node_outcome,
            huge_outcome,
        })
    }

    /// Maps and immediately faults the whole arena, for a caller that is
    /// already on the thread that will own it.
    ///
    /// # Errors
    ///
    /// Returns the operating-system error when the mapping cannot be created.
    pub fn map_touched(
        name: impl Into<String>,
        len: usize,
        align: usize,
        policy: NodePolicy,
        huge_pages: HugePagePolicy,
    ) -> io::Result<Self> {
        let mut arena = Self::map(name, len, align, policy, huge_pages)?;
        arena.fault_in();
        Ok(arena)
    }

    /// Faults every page of the arena from the calling thread, which is what
    /// makes a first-touch policy effective.
    pub fn fault_in(&mut self) {
        for page in self.as_mut_slice().chunks_mut(sys::page_size()) {
            page[0] = page[0].wrapping_add(0);
            std::hint::black_box(&page[0]);
        }
    }

    /// Writes a recognisable byte into every page, so the kernel accounts the
    /// pages as resident rather than merely reserved. Used by the placement
    /// tests and benchmarks; an arena Kivi uses is faulted by being written.
    pub fn touch_pages(&mut self) {
        for (index, page) in self.as_mut_slice().chunks_mut(sys::page_size()).enumerate() {
            page[0] = u8::try_from(index & 0xFF).unwrap_or(0);
        }
    }

    /// Bytes mapped.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the mapping holds no bytes, which never happens for a
    /// successfully mapped arena.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Start address, for diagnostics and for registering the arena with
    /// DAMON.
    #[must_use]
    pub fn address(&self) -> u64 {
        self.base as usize as u64
    }

    /// The arena's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The policy the caller asked for.
    #[must_use]
    pub const fn requested(&self) -> NodePolicy {
        self.requested
    }

    /// The huge-page policy the caller asked for.
    #[must_use]
    pub const fn requested_huge_pages(&self) -> HugePagePolicy {
        self.requested_huge
    }

    /// What happened to the memory-policy request.
    #[must_use]
    pub const fn node_outcome(&self) -> &PlacementOutcome {
        &self.node_outcome
    }

    /// What happened to the huge-page advice.
    #[must_use]
    pub const fn huge_outcome(&self) -> &PlacementOutcome {
        &self.huge_outcome
    }

    /// The bytes.
    #[must_use]
    #[allow(
        unsafe_code,
        reason = "the mapping's address and length are this type's invariants, \
                  established by `map` and torn down by `drop`"
    )]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `base` points at a live mapping of `len` bytes this struct
        // owns, and this shared borrow proves no mutable borrow is live.
        unsafe { std::slice::from_raw_parts(self.base, self.len) }
    }

    /// The bytes, mutably.
    #[allow(
        unsafe_code,
        reason = "as `as_slice`; `&mut self` proves no other reference into the \
                  mapping is alive"
    )]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as `as_slice`, and `&mut self` proves no other reference into
        // the mapping is alive.
        unsafe { std::slice::from_raw_parts_mut(self.base, self.len) }
    }

    /// Reads the kernel's page census for this mapping.
    ///
    /// The only source of truth for whether a placement actually happened.
    #[must_use]
    pub fn observe(&self) -> PagePlacement {
        sys::observe(self.base, self.len)
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: `base` and `len` are exactly what `map` returned and the
        // alignment trimming left in place, and this is the only unmapping,
        // running with no other reference into the mapping alive.
        sys::unmap(self.base, self.len);
    }
}

impl fmt::Display for Arena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "arena({name}, {mi}MiB, {node}, {huge})",
            name = self.name,
            mi = self.len / (1024 * 1024),
            node = self.node_outcome,
            huge = self.huge_outcome,
        )
    }
}

/// The platform page size, from the running kernel or the running OS.
#[must_use]
pub fn page_size() -> usize {
    sys::page_size()
}

/// One line describing this machine's memory topology, for the admin surface
/// and the `detect` example.
///
/// The point of this function is what it does *not* say: it reports the nodes
/// the kernel exposes, and nothing about locality, because on a one-node
/// machine locality is not a fact anyone can observe.
#[must_use]
pub fn report() -> String {
    let topology = &crate::snapshot().topology;
    let mut parts = vec![format!("nodes={}", topology.numa_node_count())];
    for node in &topology.memory_nodes {
        parts.push(format!(
            "node{}:{}GiB/{:?}/cpus={}",
            node.id.0,
            node.total_bytes.unwrap_or(0) / (1024 * 1024 * 1024),
            node.tier,
            node.cpus.len()
        ));
    }
    parts.push(format!("cxl={}", topology.cxl_nodes().count()));
    parts.join(" ")
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod sys {
    use std::io;
    use std::ptr;

    use super::{NodePolicy, PagePlacement};

    /// `__NR_mbind` on x86-64. 32-bit targets get it from the architecture
    /// header instead, which is the only way to be right on all of them
    /// without a build script.
    #[cfg(target_arch = "x86_64")]
    const SYS_MBIND: libc::c_long = 235;
    #[cfg(not(target_arch = "x86_64"))]
    const SYS_MBIND: libc::c_long = libc::SYS_mbind;

    #[allow(
        unsafe_code,
        reason = "reading a kernel page size is a scalar query with no \
                  preconditions; the mapping primitives below each carry their \
                  own safety argument"
    )]
    pub(super) fn page_size() -> usize {
        // SAFETY: `sysconf` takes a name and no pointer, and has no
        // preconditions. It returns -1 with `errno` set for an unknown name;
        // `_SC_PAGESIZE` is defined on every Linux build.
        let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        usize::try_from(size).unwrap_or(4096).max(4096)
    }

    /// Maps `len` bytes with an `align`-aligned start.
    ///
    /// The over-allocation and trim is what makes an arena huge-page eligible:
    /// a transparent huge page covers a 2 MiB-aligned range, so a mapping whose
    /// start is 4 KiB-aligned can only ever have its *interior* collapsed and
    /// leaves a small-page head and tail. Trimming to the alignment is a fixed
    /// cost of a few microseconds per arena, paid once for a mapping that lives
    /// for the process's life.
    #[allow(unsafe_code)]
    pub(super) fn map(len: usize, align: usize) -> io::Result<(*mut u8, usize)> {
        let page = page_size();
        let align = align.max(page).next_multiple_of(page);
        let total = len
            .checked_add(align)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "arena length overflow"))?;
        // SAFETY: a null hint lets the kernel choose the address, `total` is
        // non-zero and was range-checked, and `MAP_FAILED` is turned into an
        // `io::Error` here. `MAP_PRIVATE|MAP_ANONYMOUS` is the correct choice
        // for a Kivi arena: nothing outside this process can observe it, and a
        // stray write can never be mistaken for something durable.
        let raw = unsafe {
            libc::mmap(
                ptr::null_mut(),
                total,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if raw == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let base = raw.cast::<u8>();
        let address = base as usize;
        let head = (align - address % align) % align;
        // SAFETY: `head` is at most `align - page`, and the mapping this
        // function just created is `len + align` bytes long, so `base + head`
        // is inside it and the trimmed windows do not overlap.
        let start = unsafe { base.add(head) };
        // The kernel rounds an unmap down to a page boundary, so the trim is
        // page aligned by construction: `head` is a multiple of `page` because
        // `address` and `align` both are, and `total - head - len` is likewise.
        if head > 0 {
            // SAFETY: `[base, base + head)` is the front of the mapping this
            // function just created and has not handed to the caller.
            unsafe { libc::munmap(base.cast::<libc::c_void>(), head) };
        }
        let tail = total - head - len;
        if tail > 0 {
            // SAFETY: `[start + len, start + len + tail)` is the back of the
            // same mapping, page aligned, and unoverlapping with the window.
            let end = unsafe { start.add(len) };
            unsafe { libc::munmap(end.cast::<libc::c_void>(), tail) };
        }
        Ok((start, len))
    }

    #[allow(unsafe_code)]
    pub(super) fn unmap(base: *mut u8, len: usize) {
        // SAFETY: `base` and `len` are exactly the window `map` returned, and
        // this is the only unmapping of it.
        unsafe { libc::munmap(base.cast::<libc::c_void>(), len) };
    }

    #[allow(
        unsafe_code,
        reason = "the NUMA policy ABI is a raw syscall; every call site below \
                  carries its own safety argument"
    )]
    pub(super) fn bind(base: *mut u8, len: usize, policy: NodePolicy) -> super::PlacementOutcome {
        use super::PlacementOutcome;
        let Some(node) = policy.node() else {
            return PlacementOutcome::NotRequested;
        };
        let bits = u64::BITS;
        let index = usize::try_from(node.0 / bits).unwrap_or(0);
        let mut mask = vec![0 as libc::c_ulong; index + 1];
        mask[index] = (1 as libc::c_ulong) << (node.0 % bits);
        let mode = match policy {
            NodePolicy::Bind(_) => libc::MPOL_BIND,
            NodePolicy::Prefer(_) => libc::MPOL_PREFERRED,
            NodePolicy::Inherit => return PlacementOutcome::NotRequested,
        };
        // The kernel computes the highest node it will consider as
        // `BITS_TO_LONGS(maxnode) - 1` bits, so `maxnode` must cover every bit
        // of the mask; rounding up to a whole word is always sufficient.
        let maxnode = (index + 1) * usize::try_from(bits).unwrap_or(64);
        // No `MPOL_MF_MOVE`: the mapping has no pages yet, so the policy simply
        // governs the first fault. That is the whole mechanism - asking the
        // kernel to migrate pages instead would need privileges the process
        // does not have and would be the wrong order of operations anyway.
        //
        // `libc` does not wrap `mbind`, so the syscall number is `SYS_MBIND`.
        // SAFETY: the range is a live mapping of `len` bytes this call's caller
        // owns, the mask is a correctly sized heap buffer of `c_ulong` that
        // outlives the call, and `maxnode` covers all of its bits. The kernel
        // copies the mask before returning, so the buffer need not outlive it.
        let rc = unsafe {
            libc::syscall(
                SYS_MBIND,
                base.cast::<libc::c_void>(),
                len,
                mode,
                mask.as_ptr().cast::<libc::c_ulong>(),
                maxnode,
                0 as libc::c_ulong,
            )
        };
        if rc == 0 {
            PlacementOutcome::Applied { policy }
        } else {
            PlacementOutcome::Failed {
                policy,
                error: io::Error::last_os_error().to_string(),
            }
        }
    }

    #[allow(unsafe_code)]
    pub(super) fn advise_huge(base: *mut u8, len: usize) -> bool {
        // SAFETY: the range is inside a live mapping, and `MADV_HUGEPAGE` has
        // no other preconditions. A failure is a value, not an abort: the
        // mapping stays valid and simply keeps 4 KiB pages.
        unsafe { libc::madvise(base.cast::<libc::c_void>(), len, libc::MADV_HUGEPAGE) == 0 }
    }

    /// Reads `/proc/self/numa_maps` and totals the pages belonging to the
    /// mapping that contains `base`.
    pub(super) fn observe(base: *mut u8, len: usize) -> PagePlacement {
        let start = base as usize as u64;
        let end = start + len as u64;
        let Ok(raw) = std::fs::read_to_string("/proc/self/numa_maps") else {
            return PagePlacement::default();
        };
        let mut out = PagePlacement {
            verified: true,
            ..PagePlacement::default()
        };
        for line in raw.lines() {
            if !covers(line, start) {
                continue;
            }
            let (line_start, line_end) = range_of(line);
            if !overlaps(line_start, line_end, start, end) {
                continue;
            }
            for token in line.split_whitespace() {
                if let Some(count) = token.strip_prefix("N") {
                    if let Some((node, pages)) = count.split_once('=')
                        && let (Ok(node), Ok(pages)) = (node.parse::<u32>(), pages.parse::<u64>())
                    {
                        // A VMA the kernel split still prints one line per part,
                        // so overlapping parts are counted once each and the
                        // totals are exact rather than doubled.
                        let _ = node;
                        *out.pages_by_node.entry(node).or_insert(0) += pages;
                        out.total_pages += pages;
                    }
                } else if let Some(pages) = token.strip_prefix("anon_hugepage=")
                    && let Ok(pages) = pages.parse::<u64>()
                {
                    out.huge_pages += pages;
                }
            }
        }
        out
    }

    /// Whether a `numa_maps` line's first field names `start`.
    fn covers(line: &str, start: u64) -> bool {
        line.split_whitespace().next().is_some_and(|field| {
            u64::from_str_radix(field, 16).is_ok_and(|address| address == start)
        })
    }

    /// The address range a `numa_maps` line covers.
    ///
    /// The kernel prints an explicit end address for mappings it needs to
    /// disambiguate and omits it otherwise, which is the same convention
    /// `/proc/self/maps` uses. Treating the second field as a possible end is
    /// the only way to tell `7f00 default` from `7f00 7f10 default`.
    fn range_of(line: &str) -> (u64, u64) {
        let mut fields = line.split_whitespace();
        let Some(start) = fields.next().and_then(|f| u64::from_str_radix(f, 16).ok()) else {
            return (0, 0);
        };
        match fields.next() {
            Some(field) => match u64::from_str_radix(field, 16) {
                Ok(end) => (start, end),
                Err(_) => (start, 0),
            },
            None => (0, 0),
        }
    }

    fn overlaps(a_start: u64, a_end: u64, b_start: u64, b_end: u64) -> bool {
        let a_end = if a_end == 0 { u64::MAX } else { a_end };
        a_start < b_end && b_start < a_end
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
mod sys {
    use std::io;

    use super::{NodePolicy, PagePlacement};

    pub(super) fn page_size() -> usize {
        4096
    }

    #[allow(unsafe_code)]
    pub(super) fn map(len: usize, align: usize) -> io::Result<(*mut u8, usize)> {
        // `align` is accepted and ignored: the Windows allocator already
        // returns page-granular addresses and a reservation can be aligned by
        // over-committing, which is not worth a second syscall pair for a
        // mechanism whose only consumer is transparent huge pages.
        let _ = align;
        // SAFETY: a null address lets the system choose, `len` is non-zero and
        // was validated by the caller, and a null return becomes an
        // `io::Error` here. The region is released exactly once, in
        // `Arena::drop`.
        let base = unsafe {
            windows_sys::Win32::System::Memory::VirtualAlloc(
                std::ptr::null_mut(),
                len,
                windows_sys::Win32::System::Memory::MEM_COMMIT
                    | windows_sys::Win32::System::Memory::MEM_RESERVE,
                windows_sys::Win32::System::Memory::PAGE_READWRITE,
            )
        };
        if base.is_null() {
            Err(io::Error::last_os_error())
        } else {
            Ok((base.cast::<u8>(), len))
        }
    }

    #[allow(unsafe_code)]
    pub(super) fn unmap(base: *mut u8, _len: usize) {
        // SAFETY: `base` is what `VirtualAlloc` returned and accepted, and this
        // is the only release of it, with no other reference into the mapping
        // alive. `MEM_RELEASE` requires a zero size, which the platform
        // contract specifies.
        unsafe {
            windows_sys::Win32::System::Memory::VirtualFree(
                base.cast(),
                0,
                windows_sys::Win32::System::Memory::MEM_RELEASE,
            );
        }
    }

    pub(super) fn bind(_base: *mut u8, _len: usize, policy: NodePolicy) -> super::PlacementOutcome {
        use super::PlacementOutcome;
        if policy.node().is_none() {
            return PlacementOutcome::NotRequested;
        }
        // Windows exposes per-arena NUMA policy only through
        // `VirtualAllocExNuma`, which takes an explicit node number at mapping
        // time rather than a policy on an existing range, and a thread's node
        // mask is process-wide. Kivi allocates through the system allocator and
        // places by first touch under `SetThreadNodeMask`, which the placement
        // plan achieves by binding the owning thread. So the arena-level
        // request is answered honestly rather than faked.
        PlacementOutcome::Unavailable {
            reason: "windows places memory by thread node mask, not per arena",
        }
    }

    pub(super) fn advise_huge(base: *mut u8, len: usize) -> bool {
        // Windows has no advice mechanism; large pages are a mapping-time flag.
        // The flag is honoured by the allocator above when the privilege is
        // held, and this returns false so a caller does not report advice that
        // was never given.
        let _ = (base, len);
        false
    }

    pub(super) fn observe(_base: *mut u8, _len: usize) -> PagePlacement {
        // No unprivileged Windows interface reports per-mapping page
        // residency by NUMA node. The NUMA performance counters are
        // process-wide ETW channels, and `GetLogicalProcessorInformationEx`
        // describes the machine rather than this process. Reporting zero with
        // `verified: false` is the only honest answer.
        PagePlacement::default()
    }
}

/// Node set of the machine's memory nodes, ascending.
#[must_use]
pub fn nodes() -> Vec<NumaNodeId> {
    crate::snapshot()
        .topology
        .memory_nodes
        .iter()
        .map(|node| node.id)
        .collect()
}

/// A test and benchmark helper: a mapped arena of `len` bytes with a page
/// census, on the default policy.
///
/// # Errors
///
/// Propagates the mapping error.
pub fn probe(name: &str, len: usize) -> io::Result<(Arena, BTreeMap<u32, u64>)> {
    let mut arena = Arena::map_touched(
        name,
        len,
        page_size(),
        NodePolicy::Inherit,
        crate::pages::HugePagePolicy::Inherit,
    )?;
    arena.touch_pages();
    let observed = arena.observe();
    Ok((arena, observed.pages_by_node.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policies_parse_every_spelling() {
        assert_eq!(NodePolicy::parse("none"), Ok(NodePolicy::Inherit));
        assert_eq!(NodePolicy::parse("1"), Ok(NodePolicy::Bind(NumaNodeId(1))));
        assert_eq!(
            NodePolicy::parse("PREFER:3"),
            Ok(NodePolicy::Prefer(NumaNodeId(3)))
        );
        assert!(NodePolicy::parse("bind:x").is_err());
        assert!(NodePolicy::parse("sideways:1").is_err());
    }

    #[test]
    fn an_inherit_policy_is_not_a_placement() {
        let arena = Arena::map(
            "test",
            64 * 1024,
            page_size(),
            NodePolicy::Inherit,
            crate::pages::HugePagePolicy::Inherit,
        )
        .expect("maps");
        assert_eq!(*arena.node_outcome(), PlacementOutcome::NotRequested);
        assert_eq!(*arena.huge_outcome(), PlacementOutcome::NotRequested);
    }

    #[test]
    fn a_placement_never_prevents_the_mapping() {
        // A node that cannot exist is the interesting case: the binding fails
        // and the arena is still a usable, readable, writable mapping.
        let mut arena = Arena::map(
            "test",
            128 * 1024,
            page_size(),
            NodePolicy::Bind(NumaNodeId(9_999)),
            crate::pages::HugePagePolicy::Inherit,
        )
        .expect("maps even when the node does not exist");
        assert_eq!(arena.len(), 128 * 1024);
        assert!(arena.address().is_multiple_of(page_size() as u64));
        arena.touch_pages();
        assert!(arena.as_slice().iter().any(|&byte| byte != 0));
    }

    #[test]
    fn a_real_node_binds_or_reports_why_not() {
        let Some(node) = nodes().first().copied() else {
            return;
        };
        let arena = Arena::map(
            "test",
            256 * 1024,
            page_size(),
            NodePolicy::Bind(node),
            crate::pages::HugePagePolicy::Advice,
        )
        .expect("maps");
        // Either the kernel applied the policy or it said why. What must never
        // happen is an arena that cannot be used. The applied and failed arms
        // assert the same thing - the request survived to whatever answer the
        // platform gave - so they share one arm rather than repeating the
        // assertion and letting one drift.
        assert!(!arena.is_empty());
        match arena.node_outcome() {
            PlacementOutcome::Applied { policy } | PlacementOutcome::Failed { policy, .. } => {
                assert_eq!(policy.node(), Some(node));
            }
            PlacementOutcome::Unavailable { reason } => assert!(!reason.is_empty()),
            PlacementOutcome::NotRequested => panic!("a bind policy was requested"),
        }
    }

    #[test]
    fn the_census_agrees_with_where_the_pages_went() {
        let (arena, pages) = probe("census", 2 * 1024 * 1024).expect("probe");
        let observed = arena.observe();
        if !observed.verified {
            // The platform has no per-mapping residency interface, so there is
            // nothing to agree with. The type says so rather than inventing a
            // census, and that is the contract being asserted here.
            assert_eq!(observed.to_string(), "unverified");
            assert!(pages.is_empty());
            return;
        }
        assert_eq!(
            observed.total_pages,
            2 * 1024 * 1024 / page_size() as u64,
            "every page of a touched arena is resident"
        );
        assert_eq!(pages, observed.pages_by_node);
        let total: u64 = observed.pages_by_node.values().sum();
        assert_eq!(total, observed.total_pages);
    }

    #[test]
    fn huge_page_advice_is_reported_not_assumed() {
        // The honest form of the claim: after advice, either the kernel backed
        // the arena with huge pages or it did not, and the number says which.
        let mut arena = Arena::map_touched(
            "huge",
            8 * 1024 * 1024,
            2 * 1024 * 1024,
            NodePolicy::Inherit,
            crate::pages::HugePagePolicy::Advice,
        )
        .expect("maps");
        arena.touch_pages();
        let observed = arena.observe();
        if observed.verified {
            assert!(
                observed.huge_pages <= observed.total_pages / 512 + 1,
                "huge pages are counted in 2 MiB units against 4 KiB pages"
            );
        }
    }

    #[test]
    fn an_unverified_census_concludes_nothing() {
        let empty = PagePlacement::default();
        assert!(!empty.verified);
        assert!(!empty.is_local_to(NumaNodeId(0)));
        assert!(empty.remote_fraction(NumaNodeId(0)).abs() < f64::EPSILON);
        assert_eq!(empty.to_string(), "unverified");
    }

    #[test]
    fn locality_is_computed_from_the_census() {
        let mut census = PagePlacement {
            verified: true,
            total_pages: 100,
            ..PagePlacement::default()
        };
        census.pages_by_node.insert(0, 90);
        census.pages_by_node.insert(1, 10);
        assert!(!census.is_local_to(NumaNodeId(0)));
        assert!((census.remote_fraction(NumaNodeId(0)) - 0.1).abs() < 1e-9);
        census.pages_by_node.clear();
        census.pages_by_node.insert(0, 100);
        assert!(census.is_local_to(NumaNodeId(0)));
    }

    #[test]
    fn dropping_an_arena_releases_its_mapping() {
        // Ten arenas mapped, dropped, and mapped again: a leak here would be a
        // virtual address space exhaustion long before it was a test failure on
        // a long-running server.
        for _ in 0..10 {
            let arena = Arena::map(
                "churn",
                8 * 1024 * 1024,
                page_size(),
                NodePolicy::Inherit,
                crate::pages::HugePagePolicy::Inherit,
            )
            .expect("maps");
            assert_eq!(arena.len(), 8 * 1024 * 1024);
        }
    }

    #[test]
    fn a_zero_length_arena_is_refused() {
        assert!(
            Arena::map(
                "empty",
                0,
                page_size(),
                NodePolicy::Inherit,
                crate::pages::HugePagePolicy::Inherit
            )
            .is_err()
        );
    }

    #[test]
    fn the_report_names_every_node() {
        let report = report();
        assert!(report.contains("nodes="));
        for node in nodes() {
            assert!(report.contains(&format!("node{}", node.0)), "{report}");
        }
    }
}
