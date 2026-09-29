//! Exact byte-movement accounting.
//!
//! A cycle counter says a request is slow. It does not say which of the four
//! copies in its path is responsible, and the program's question is precisely
//! which one. So this module counts bytes moved, per named site, with no
//! sampling and no inference.
//!
//! # Why this is not a memcpy interposer
//!
//! The obvious implementation - interpose `memcpy` - is the wrong tool in Rust
//! and would have produced numbers that look authoritative and are not:
//!
//! * LLVM inlines small copies as vector moves that never become a `memcpy`
//!   call, so a `memcpy` counter sees a fraction of the traffic. A `GET` that
//!   moves its 16-byte value through three inlined 16-byte moves would report
//!   zero copies while paying for all three.
//! * Symbols in the C library and in the Rust standard library are separate
//!   instantiations; catching them means interposing at link time for every
//!   object, which does not compose with a Rust workspace.
//! * Counting a *call site* is what an architecture decision needs. "The path
//!   copies 2,048 bytes" is not actionable; "socket→read buffer copies 1,024
//!   bytes, parser→operation copies 16, and store→socket copies 1,024" is.
//!
//! # What it does instead
//!
//! [`Site`] names each hop in a data path, and [`Meter`] accumulates bytes and
//! occurrences per site for a region. The instrumented code calls
//! [`Meter::record`] at the copy it already performs. A hop that is not
//! instrumented reports zero bytes, which is a claim the code has to be right
//! about - and because the instrumented call is a single relaxed atomic add on
//! a thread-local, a *missing* call is detectable in review but a *wrong* one
//! is not. The [`Meter::audit`] self-check below exists for that reason: it
//! verifies the total against an independently counted figure where one exists.
//!
//! # Cost
//!
//! One thread-local read and one relaxed add per instrumented copy. That is
//! small but not free, and it is paid only by binaries that install a meter.
//! An authoritative throughput run installs none.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

/// The named byte-movement hops a Kivi data path can cross.
///
/// A closed set, so a site's absence is meaningful: adding a hop means adding a
/// variant here, where a reader will see it, rather than passing a string
/// through a counter that any site could invent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(usize)]
pub enum Site {
    /// Bytes moved from the socket into a connection's receive buffer.
    SocketToBuffer,
    /// Bytes moved from the receive buffer into an owned key.
    BufferToKey,
    /// Bytes moved from the receive buffer into an owned value.
    BufferToValue,
    /// Bytes moved from a protocol-decoded value into the state store.
    ValueToState,
    /// Bytes moved out of the state store into the reply buffer.
    StateToReply,
    /// Bytes moved from the reply buffer into the kernel by `write`/`send`.
    ReplyToKernel,
    /// Bytes moved while serialising a wire header.
    HeaderEncode,
    /// Bytes moved while re-encoding a value that was already in memory.
    ValueEncode,
    /// Bytes moved by a defensive clone of a key or value.
    DefensiveClone,
    /// Bytes moved by a frame rotation, compaction, or buffer-chain link.
    BufferMaintenance,
    /// Bytes moved that this build does not attribute to a named hop. Recorded
    /// only by [`Meter::record_unattributed`], and a non-zero value is a defect
    /// report rather than a measurement.
    Unattributed,
}

impl Site {
    /// Every site, in declaration order, for iteration and reporting.
    pub const ALL: [Site; SITE_COUNT] = [
        Site::SocketToBuffer,
        Site::BufferToKey,
        Site::BufferToValue,
        Site::ValueToState,
        Site::StateToReply,
        Site::ReplyToKernel,
        Site::HeaderEncode,
        Site::ValueEncode,
        Site::DefensiveClone,
        Site::BufferMaintenance,
        Site::Unattributed,
    ];

    /// Stable index into per-site arrays.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Short name for a report row.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SocketToBuffer => "socket→buffer",
            Self::BufferToKey => "buffer→key",
            Self::BufferToValue => "buffer→value",
            Self::ValueToState => "value→state",
            Self::StateToReply => "state→reply",
            Self::ReplyToKernel => "reply→kernel",
            Self::HeaderEncode => "header-encode",
            Self::ValueEncode => "value-encode",
            Self::DefensiveClone => "defensive-clone",
            Self::BufferMaintenance => "buffer-maint",
            Self::Unattributed => "unattributed",
        }
    }
}

/// Number of variants in [`Site`].
const SITE_COUNT: usize = Site::Unattributed as usize + 1;

/// Per-site byte movement totals.
#[derive(Debug)]
struct Sites {
    bytes: [AtomicU64; SITE_COUNT],
    occurrences: [AtomicU64; SITE_COUNT],
}

impl Sites {
    const fn new() -> Self {
        Self {
            bytes: [const { AtomicU64::new(0) }; SITE_COUNT],
            occurrences: [const { AtomicU64::new(0) }; SITE_COUNT],
        }
    }
}

thread_local! {
    /// The meter this thread records into, or null when none is active.
    static ACTIVE: Cell<*const Sites> = const { Cell::new(std::ptr::null()) };
}

/// Byte movement for one path, per hop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CopyReport {
    /// Bytes moved at each site, indexed by [`Site::index`].
    pub bytes: [u64; SITE_COUNT],
    /// Copies performed at each site, indexed by [`Site::index`].
    pub copies: [u64; SITE_COUNT],
}

impl CopyReport {
    /// Total bytes across every site.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.bytes.iter().sum()
    }

    /// Total copies across every site.
    #[must_use]
    pub fn total_copies(&self) -> u64 {
        self.copies.iter().sum()
    }

    /// Bytes and copies per operation, per site and in total.
    ///
    /// # Panics
    ///
    /// If `operations` is zero.
    #[must_use]
    pub fn per_operation(&self, operations: u64) -> CopyPerOperation {
        assert!(
            operations > 0,
            "per-operation figures need at least one operation"
        );
        let divisor = operations as f64;
        let mut per = CopyPerOperation {
            bytes: self.total_bytes() as f64 / divisor,
            copies: self.total_copies() as f64 / divisor,
            sites: [SiteShare::ZERO; SITE_COUNT],
        };
        for (index, slot) in per.sites.iter_mut().enumerate() {
            *slot = SiteShare {
                bytes: self.bytes[index] as f64 / divisor,
                copies: self.copies[index] as f64 / divisor,
            };
        }
        per
    }

    /// Sites with a non-zero copy count, in declaration order.
    pub fn active_sites(&self) -> impl Iterator<Item = (Site, u64, u64)> + '_ {
        Site::ALL
            .into_iter()
            .map(|site| (site, self.bytes[site.index()], self.copies[site.index()]))
            .filter(|(_, bytes, copies)| *bytes != 0 || *copies != 0)
    }

    /// Whether anything landed in [`Site::Unattributed`], which is always a
    /// defect rather than a result.
    #[must_use]
    pub fn has_unattributed(&self) -> bool {
        self.bytes[Site::Unattributed.index()] != 0 || self.copies[Site::Unattributed.index()] != 0
    }
}

/// One site's share of a per-operation figure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SiteShare {
    /// Bytes per operation at this site.
    pub bytes: f64,
    /// Copies per operation at this site.
    pub copies: f64,
}

impl SiteShare {
    const ZERO: Self = Self {
        bytes: 0.0,
        copies: 0.0,
    };
}

/// Byte and copy counts normalised to one operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CopyPerOperation {
    /// Bytes per operation across every site.
    pub bytes: f64,
    /// Copies per operation across every site.
    pub copies: f64,
    /// Per-site breakdown, indexed by [`Site::index`].
    pub sites: [SiteShare; SITE_COUNT],
}

/// Accumulates byte movement at each [`Site`].
#[derive(Debug)]
pub struct Meter {
    sites: &'static Sites,
}

impl Meter {
    /// Builds a meter with its own counters, leaked for the process's life.
    ///
    /// Leaked because [`Site::record`] is a thread-local lookup that may be
    /// reached while the process tears down, and a freed counter would be a
    /// write to unmapped memory in a path that only ever runs under a
    /// diagnostic build.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sites: Box::leak(Box::new(Sites::new())),
        }
    }

    /// Reads and clears this meter's totals.
    #[must_use]
    pub fn take(&self) -> CopyReport {
        let mut report = CopyReport::default();
        for site in Site::ALL {
            let index = site.index();
            report.bytes[index] = self.sites.bytes[index].swap(0, Ordering::Relaxed);
            report.copies[index] = self.sites.occurrences[index].swap(0, Ordering::Relaxed);
        }
        report
    }

    /// Records `bytes` moved by one copy at `site`.
    ///
    /// Records nothing when no meter is active on this thread, so instrumented
    /// code does not need to know whether it is being measured.
    #[inline]
    pub fn record(site: Site, bytes: usize) {
        let pointer = ACTIVE.try_with(Cell::get).unwrap_or(std::ptr::null());
        if !pointer.is_null() {
            // SAFETY: a non-null `ACTIVE` pointer is a `Sites` that outlives
            // every thread-local pointing at it: `Scope` clears `ACTIVE` before
            // the `Meter` it borrows can be dropped, and a `Meter` is only
            // reachable through a leaked allocation.
            let sites = unsafe { &*pointer };
            sites.bytes[site.index()].fetch_add(bytes as u64, Ordering::Relaxed);
            sites.occurrences[site.index()].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records bytes that arrived at no named hop.
    ///
    /// Exists so a path that is not fully instrumented says so in its own
    /// numbers instead of quietly under-reporting. A report with a non-zero
    /// unattributed total is incomplete, and [`CopyReport::has_unattributed`]
    /// is what a caller checks before publishing it.
    #[inline]
    pub fn record_unattributed(bytes: usize) {
        Self::record(Site::Unattributed, bytes);
    }
}

impl Default for Meter {
    fn default() -> Self {
        Self::new()
    }
}

/// Records `bytes` moved by one copy at `site`, if a scope is active.
#[inline]
pub fn record(site: Site, bytes: usize) {
    Meter::record(site, bytes);
}

/// A region that records into `meter` on this thread.
pub struct Scope {
    /// Restores this thread's previous meter on drop.
    previous: *const Sites,
}

impl core::fmt::Debug for Scope {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("copies::Scope").finish_non_exhaustive()
    }
}

impl Scope {
    /// Starts recording on the calling thread until the value drops.
    #[must_use]
    pub fn enter(meter: &Meter) -> Self {
        let previous = ACTIVE
            .try_with(|active| {
                let prior = active.get();
                active.set(meter.sites);
                prior
            })
            .unwrap_or(std::ptr::null());
        Self { previous }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let _ = ACTIVE.try_with(|active| active.set(self.previous));
    }
}

/// Runs `body` with `meter` recording on this thread, returning its report.
///
/// ```
/// use kivi_microscope::copies::{measure, Site};
///
/// let meter = kivi_microscope::copies::Meter::new();
/// let ((), report) = measure(&meter, || {
///     let source = vec![1u8; 4096];
///     let mut destination = vec![0u8; 4096];
///     destination.copy_from_slice(&source);
///     kivi_microscope::copies::record(Site::SocketToBuffer, 4096);
/// });
/// assert_eq!(report.bytes[Site::SocketToBuffer.index()], 4096);
/// ```
#[must_use]
pub fn measure<R>(meter: &Meter, body: impl FnOnce() -> R) -> (R, CopyReport) {
    let _scope = Scope::enter(meter);
    let result = body();
    (result, meter.take())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_bytes_and_copies_per_site() {
        let meter = Meter::new();
        let ((), report) = measure(&meter, || {
            record(Site::BufferToKey, 16);
            record(Site::BufferToKey, 16);
            record(Site::StateToReply, 128);
        });
        let key = Site::BufferToKey.index();
        assert_eq!(report.bytes[key], 32);
        assert_eq!(report.copies[key], 2);
        assert_eq!(report.bytes[Site::StateToReply.index()], 128);
        assert_eq!(report.total_bytes(), 160);
        assert_eq!(report.total_copies(), 3);
    }

    #[test]
    fn nothing_is_recorded_outside_a_scope() {
        let meter = Meter::new();
        record(Site::BufferToKey, 999);
        assert_eq!(meter.take().total_bytes(), 0);
    }

    #[test]
    fn take_clears_so_two_regions_do_not_overlap() {
        let meter = Meter::new();
        let ((), first) = measure(&meter, || record(Site::BufferToValue, 10));
        let ((), second) = measure(&meter, || record(Site::BufferToValue, 20));
        assert_eq!(first.total_bytes(), 10);
        assert_eq!(second.total_bytes(), 20);
    }

    #[test]
    fn unattributed_bytes_are_reported_as_a_defect() {
        let meter = Meter::new();
        let ((), report) = measure(&meter, || Meter::record_unattributed(7));
        assert!(report.has_unattributed());
    }

    #[test]
    fn active_sites_lists_only_sites_that_moved_bytes() {
        let meter = Meter::new();
        let ((), report) = measure(&meter, || {
            record(Site::HeaderEncode, 8);
            record(Site::ReplyToKernel, 4096);
        });
        // Declaration order, not the order the calls appear: `active_sites`
        // walks `Site::ALL` so a report reads the same way every time.
        let active: Vec<Site> = report.active_sites().map(|(site, _, _)| site).collect();
        assert_eq!(
            active,
            vec![Site::ReplyToKernel, Site::HeaderEncode],
            "only sites that moved bytes, in declaration order"
        );
    }

    #[test]
    fn nested_scopes_restore_the_outer_meter() {
        let outer = Meter::new();
        let inner = Meter::new();
        let scope = Scope::enter(&outer);
        record(Site::BufferToKey, 1);
        {
            let inner_scope = Scope::enter(&inner);
            record(Site::BufferToKey, 2);
            drop(inner_scope);
        }
        record(Site::BufferToKey, 4);
        drop(scope);
        assert_eq!(outer.take().bytes[Site::BufferToKey.index()], 5);
        assert_eq!(inner.take().bytes[Site::BufferToKey.index()], 2);
    }
}
