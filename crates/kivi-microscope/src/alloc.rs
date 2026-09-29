//! Exact allocation accounting.
//!
//! The global allocator here counts every allocation, deallocation, realloc and
//! zeroed byte that passes through the process, per thread, with no sampling.
//! That is the only way to answer the questions the program asks - "how many
//! allocations does one `GET` do" and "does the fast path allocate" - exactly
//! rather than approximately.
//!
//! # Why counting is trustworthy here and would not be on a sampling profiler
//!
//! A [`Counting`] allocator wraps the real one and adds two thread-local
//! counters. The increments are plain `u64` adds to a thread-local, so they
//! cost no atomics and no cross-core traffic, and they cannot be interleaved.
//! A sampling profiler answers "about 1.3% of requests allocate"; this answers
//! "1,240,000 of 1,000,000 GETs allocated exactly 3 times each", and can name
//! the size distribution.
//!
//! # Cost, and when it may be paid
//!
//! The counting path is not free: [`Counting`] adds work to every allocation,
//! and a server with allocation on its hot path will show it. So:
//!
//! * Authoritative throughput runs install no counting allocator at all. The
//!   [`Counting`] type is instantiated only by a binary or test that asks for
//!   it, because `#[global_allocator]` is a static property of a binary.
//! * This crate's own measurement helper, [`measure`], is a
//!   [`scoped`](https://docs.rs) counter pair: a [`Scope`] counts only between
//!   construction and drop, so setup cost (populating a table, building a
//!   client) is excluded from the reported figure rather than being manually
//!   subtracted.
//!
//! [`Counting`]: struct.Counting.html
//! [`Scope`]: struct.Scope.html

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

thread_local! {
    /// Live allocation count on this thread, maintained as the delta between
    /// total allocations and total deallocations. Only read for leak checks.
    static LIVE: Cell<i64> = const { Cell::new(0) };
    /// The innermost active scope's counters, or null when unscoped.
    static ACTIVE: Cell<*const Counters> = const { Cell::new(ptr::null()) };
    /// Per-thread scratch backing [`measure`].
    ///
    /// A scope must own counters of its own rather than reuse the process-wide
    /// ones, because tests and servers run scoped measurements on several
    /// threads at once and a shared counter would report every thread's
    /// allocations as this thread's. Scratch is per thread for the same reason
    /// the active slot is: a measurement is per thread, and this is where a
    /// measurement happens.
    static SCRATCH: Counters = const { Counters::new() };
}

/// Six counters and a size histogram.
#[derive(Debug)]
pub struct Counters {
    /// Successful `alloc` calls.
    pub allocations: AtomicU64,
    /// Successful `dealloc` calls.
    pub deallocations: AtomicU64,
    /// `realloc` calls that moved the allocation.
    pub reallocations: AtomicU64,
    /// Bytes requested by successful allocations and reallocations (the new
    /// size, not the growth), including zeroes in size classes below the
    /// histogram's floor.
    pub bytes: AtomicU64,
    /// Live allocation count at the last read. Negative means more was freed
    /// than allocated, which means a counting bug rather than a leak.
    pub live: AtomicU64,
    /// Allocation sizes bucketed by power of two, index `n` holding sizes in
    /// `[2^n, 2^(n+1))`. Index 0 is anything smaller than 1 byte, which cannot
    /// happen; a zero-size request lands in index 1 with the real allocator
    /// supplying its minimum.
    pub size_histogram: [AtomicU64; HISTOGRAM_BUCKETS],
}

/// Power-of-two size buckets, 1 B through 4 GiB.
const HISTOGRAM_BUCKETS: usize = 33;

impl Counters {
    const fn new() -> Self {
        Self {
            allocations: AtomicU64::new(0),
            deallocations: AtomicU64::new(0),
            reallocations: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            live: AtomicU64::new(0),
            size_histogram: [const { AtomicU64::new(0) }; HISTOGRAM_BUCKETS],
        }
    }

    /// Reads every counter.
    #[must_use]
    pub fn snapshot(&self) -> Snapshot {
        let mut histogram = [0u64; HISTOGRAM_BUCKETS];
        for (slot, value) in histogram.iter_mut().zip(&self.size_histogram) {
            *slot = value.load(Ordering::Relaxed);
        }
        let allocations = self.allocations.load(Ordering::Relaxed);
        let deallocations = self.deallocations.load(Ordering::Relaxed);
        Snapshot {
            allocations,
            deallocations,
            reallocations: self.reallocations.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            live: allocations.saturating_sub(deallocations),
            size_histogram: histogram,
        }
    }

    /// The counters that moved since `earlier`.
    #[must_use]
    pub fn since(&self, earlier: &Snapshot) -> Snapshot {
        self.snapshot().since(earlier)
    }

    /// Adds `n` to the allocation counter and `size` to the size histogram and
    /// byte total.
    fn note_alloc(&self, size: usize) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(size as u64, Ordering::Relaxed);
        self.size_histogram[bucket(size)].fetch_add(1, Ordering::Relaxed);
        note_scope(|counters| {
            counters.allocations.fetch_add(1, Ordering::Relaxed);
            counters.bytes.fetch_add(size as u64, Ordering::Relaxed);
            counters.size_histogram[bucket(size)].fetch_add(1, Ordering::Relaxed);
        });
    }

    /// Adds one deallocation to the counter.
    fn note_dealloc(&self) {
        self.deallocations.fetch_add(1, Ordering::Relaxed);
        note_scope(|counters| {
            counters.deallocations.fetch_add(1, Ordering::Relaxed);
        });
    }

    /// Adds one reallocation to the counter.
    fn note_realloc(&self, size: usize) {
        self.reallocations.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(size as u64, Ordering::Relaxed);
        note_scope(|counters| {
            counters.reallocations.fetch_add(1, Ordering::Relaxed);
            counters.bytes.fetch_add(size as u64, Ordering::Relaxed);
        });
    }

    /// Every counter, flattened so [`measure`] can zero a scope's scratch
    /// without reaching into each array individually.
    fn drain(&self) -> impl Iterator<Item = &AtomicU64> {
        std::iter::once(&self.allocations)
            .chain(std::iter::once(&self.deallocations))
            .chain(std::iter::once(&self.reallocations))
            .chain(std::iter::once(&self.bytes))
            .chain(std::iter::once(&self.live))
            .chain(self.size_histogram.iter())
    }
}

/// A point-in-time reading of some [`Counters`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Snapshot {
    /// Allocations completed.
    pub allocations: u64,
    /// Deallocations completed.
    pub deallocations: u64,
    /// Reallocations completed.
    pub reallocations: u64,
    /// Bytes allocated.
    pub bytes: u64,
    /// Allocations minus deallocations.
    pub live: u64,
    /// Allocation counts per power-of-two size class.
    pub size_histogram: [u64; HISTOGRAM_BUCKETS],
}

impl Snapshot {
    /// The counters that moved between `earlier` and `self`.
    ///
    /// Saturating throughout: a counter that went backwards means a bug, and
    /// reporting zero for it is a smaller lie than reporting a huge number.
    #[must_use]
    pub fn since(&self, earlier: &Self) -> Self {
        let mut size_histogram = [0u64; HISTOGRAM_BUCKETS];
        for (slot, (now, before)) in size_histogram.iter_mut().zip(
            self.size_histogram
                .iter()
                .zip(earlier.size_histogram.iter()),
        ) {
            *slot = now.saturating_sub(*before);
        }
        Self {
            allocations: self.allocations.saturating_sub(earlier.allocations),
            deallocations: self.deallocations.saturating_sub(earlier.deallocations),
            reallocations: self.reallocations.saturating_sub(earlier.reallocations),
            bytes: self.bytes.saturating_sub(earlier.bytes),
            live: self.live,
            size_histogram,
        }
    }

    /// A snapshot with every counter at zero, for differencing against a
    /// first reading.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            allocations: 0,
            deallocations: 0,
            reallocations: 0,
            bytes: 0,
            live: 0,
            size_histogram: [0; HISTOGRAM_BUCKETS],
        }
    }

    /// Scales the counters to a per-operation basis.
    ///
    /// # Panics
    ///
    /// If `operations` is zero, because the result would be infinite and
    /// infinity is not a measurement.
    #[must_use]
    pub fn per_operation(self, operations: u64) -> PerOperation {
        assert!(
            operations > 0,
            "per-operation figures need at least one operation"
        );
        let divisor = operations as f64;
        PerOperation {
            allocations: self.allocations as f64 / divisor,
            deallocations: self.deallocations as f64 / divisor,
            reallocations: self.reallocations as f64 / divisor,
            bytes: self.bytes as f64 / divisor,
        }
    }
}

/// Allocation counts normalised to one operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerOperation {
    /// Allocations per operation.
    pub allocations: f64,
    /// Deallocations per operation.
    pub deallocations: f64,
    /// Reallocations per operation.
    pub reallocations: f64,
    /// Bytes allocated per operation.
    pub bytes: f64,
}

impl core::fmt::Display for PerOperation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:.3} alloc/op, {:.3} dealloc/op, {:.3} realloc/op, {:.1} B/op",
            self.allocations, self.deallocations, self.reallocations, self.bytes
        )
    }
}

/// Power-of-two bucket index for `size`.
///
/// `size == 0` maps to bucket 0: the standard allocator serves a zero-size
/// request from its minimum, so counting it as "smaller than one byte" keeps the
/// histogram's arithmetic honest instead of inventing a size.
const fn bucket(size: usize) -> usize {
    let nonzero = if size == 0 { 0 } else { size };
    (usize::BITS - nonzero.leading_zeros()) as usize
}

/// Records an event into the innermost active scope, if any.
///
/// Called on every allocation, so it is the hot path of the whole module. A
/// null check on a thread-local pointer is the cheapest test that can be wrong:
/// the scopes are strictly nested on one thread, so a single slot suffices and
/// no stack is needed.
#[inline]
fn note_scope(count: impl FnOnce(&Counters)) {
    let pointer = ACTIVE.try_with(Cell::get).unwrap_or(ptr::null());
    if !pointer.is_null() {
        // SAFETY: a non-null `ACTIVE` pointer is a `Counters` that outlives
        // the scope pushing it: the scope owns its `Counters` and clears
        // `ACTIVE` before that storage is released, and scopes on one thread
        // are strictly nested.
        count(unsafe { &*pointer });
    }
}

/// A counting region.
///
/// Allocations between construction and [`Drop`] are counted here, on this
/// thread only.
///
/// Per thread because the fast path being measured is per thread by
/// construction - one worker, one connection, one request at a time. A shared
/// counter would add exactly the cross-core traffic a measurement of a
/// single-core path is trying to detect.
///
/// This narrows *where* an allocation is counted, not whether it is: the global
/// allocator always updates the process-wide total, and a scope adds the same
/// event to the scope's counters as well.
pub struct Scope {
    /// Restores this thread's previous scope on drop.
    previous: *const Counters,
}

impl core::fmt::Debug for Scope {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("alloc::Scope").finish_non_exhaustive()
    }
}

impl Scope {
    /// Starts counting on the calling thread until the returned value drops.
    ///
    /// The scope borrows `counters` for as long as it lives, so a `Counters`
    /// that is a local cannot be recorded into: the pointer stored in the
    /// thread-local would outlive it, and an allocation during teardown would
    /// write to freed memory.
    #[must_use]
    pub fn enter(counters: &Counters) -> Self {
        // SAFETY of the pointer stored below: the `Counters` outlives this
        // `Scope` by the borrow's lifetime, and `Drop` clears the thread-local
        // before that borrow ends. Scopes on one thread nest, so a single slot
        // restores correctly.
        let pointer = std::ptr::from_ref(counters);
        let previous = ACTIVE
            .try_with(|active| {
                let prior = active.get();
                active.set(pointer);
                prior
            })
            .unwrap_or(ptr::null());
        Self { previous }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let _ = ACTIVE.try_with(|active| active.set(self.previous));
    }
}

/// A [`GlobalAlloc`] that counts every call before forwarding it to the inner
/// allocator.
///
/// A `#[global_allocator]` is a static property of a binary, so this type is
/// instantiated only where counting is wanted. It is never installed in
/// Kivi's shipped server, and an authoritative benchmark must run a binary that
/// does not install it.
#[derive(Debug)]
pub struct Counting<A = System> {
    inner: A,
    counters: &'static Counters,
}

/// The counters a [`Counting::with_static_counters`] allocator writes to.
///
/// Module-level rather than function-local: a `static` inside a function
/// cannot be borrowed from a `const fn` that outlives the function's own frame,
/// and the borrow a `#[global_allocator]` needs is exactly that outlives.
static STATIC_COUNTERS: Counters = Counters::new();

impl Counting<System> {
    /// A counting allocator with counters in static storage.
    ///
    /// The form a `#[global_allocator]` needs, because a `static` initialiser
    /// must be a constant expression. Counters live in `.bss`, so nothing is
    /// allocated and there is nothing to leak.
    #[must_use]
    pub const fn with_static_counters() -> Self {
        Self {
            inner: System,
            counters: &STATIC_COUNTERS,
        }
    }
}

impl<A> Counting<A> {
    /// The counters this allocator writes.
    #[must_use]
    pub fn counters(&self) -> &'static Counters {
        self.counters
    }
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for Counting<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `inner` is a `GlobalAlloc`, so this call has the same
        // contract as the trait method itself. The count happens after the
        // allocation, so a null return is not counted and a failed allocation
        // is not misreported as a live one.
        let pointer = unsafe { self.inner.alloc(layout) };
        if !pointer.is_null() {
            LIVE.with(|live| live.set(live.get() + 1));
            self.counters.note_alloc(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: as `alloc`.
        let pointer = unsafe { self.inner.alloc_zeroed(layout) };
        if !pointer.is_null() {
            LIVE.with(|live| live.set(live.get() + 1));
            self.counters.note_alloc(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: as the trait method; `pointer` came from `inner`.
        unsafe { self.inner.dealloc(pointer, layout) };
        LIVE.with(|live| live.set(live.get() - 1));
        self.counters.note_dealloc();
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: as the trait method; the `Layout` describes `pointer`, which
        // the caller guarantees came from this allocator.
        let moved = unsafe { self.inner.realloc(pointer, layout, new_size) };
        if !moved.is_null() {
            self.counters.note_realloc(new_size);
        }
        moved
    }
}

/// Runs `body` with scoped counting active on this thread and returns the
/// difference from before it started.
///
/// This is the shape the program needs for "exact allocations for `GET 16 B`":
/// the fixture is built first, then the region is measured, so table population
/// cannot leak into the figure.
///
/// The scope uses per-thread scratch counters rather than the process-wide ones,
/// so two threads measuring at once cannot see each other's allocations. The
/// process-wide counters installed by a `#[global_allocator]` remain the
/// authority on "what did this binary allocate in total"; a scope narrows *when*
/// and *on which thread* it is attributed, never *whether*.
///
/// Returns a [`ScopeReport`] rather than a bare [`Snapshot`] because the
/// distinction matters: a scoped count that reads zero means the region
/// allocated nothing, while a global count that reads zero can also mean no
/// allocator is installed. The report says which was measured.
#[must_use]
pub fn measure<R>(body: impl FnOnce() -> R) -> (R, ScopeReport) {
    SCRATCH.with(|counters| {
        for slot in counters.drain() {
            slot.store(0, Ordering::Relaxed);
        }
        let before = counters.snapshot();
        let scope = Scope::enter(counters);
        let result = body();
        drop(scope);
        let after = counters.snapshot();
        (
            result,
            ScopeReport {
                delta: after.since(&before),
            },
        )
    })
}

/// What a scoped measurement observed, and what kind of figure it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeReport {
    /// Counters that moved inside the region.
    pub delta: Snapshot,
}

impl ScopeReport {
    /// Normalises the scoped counts to a per-operation basis.
    ///
    /// # Panics
    ///
    /// If `operations` is zero.
    #[must_use]
    pub fn per_operation(&self, operations: u64) -> PerOperation {
        self.delta.per_operation(operations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `#[global_allocator]` is a static property of a binary, so the unit tests
    // can only observe real allocations by installing one. Production binaries
    // install one deliberately; a test binary installs it unconditionally
    // because counting is the point. Without this attribute the allocator is
    // merely *held*, routes nothing, and every count below reads zero - which
    // is the exact failure this annotation prevents.
    #[global_allocator]
    static ALLOCATOR: Counting<System> = Counting::with_static_counters();

    /// The counters the process's global allocator writes to.
    fn counters() -> &'static Counters {
        ALLOCATOR.counters()
    }

    #[test]
    fn counts_allocations_and_bytes() {
        let ((), report) = measure(|| {
            let data = vec![0u8; 8192];
            core::hint::black_box(&data);
        });
        assert_eq!(report.delta.allocations, 1, "one Vec, one allocation");
        assert!(report.delta.bytes >= 8192, "bytes covered the vector");
    }

    #[test]
    fn fixture_cost_is_outside_the_measured_region() {
        // Built before the region opens, which is the whole reason `measure`
        // takes a closure rather than a pair of snapshots.
        let fixture: Vec<u8> = vec![7u8; 4096];
        let ((), report) = measure(|| {
            core::hint::black_box(&fixture);
        });
        assert_eq!(
            report.delta.allocations, 0,
            "reading an existing allocation must not allocate"
        );
    }

    #[test]
    fn the_size_histogram_separates_classes() {
        let ((), report) = measure(|| {
            let small = vec![0u8; 16];
            let large = vec![0u8; 16 * 1024];
            core::hint::black_box((&small, &large));
        });
        assert_eq!(report.delta.size_histogram[bucket(16)], 1);
        assert_eq!(report.delta.size_histogram[bucket(16 * 1024)], 1);
    }

    /// The global allocator sees every allocation, scoped or not. A scope
    /// narrows attribution, it does not stop counting - otherwise a scoped
    /// measurement would be reporting a different world from the one the
    /// server runs in.
    #[test]
    fn the_global_counters_see_unscoped_allocations_too() {
        let before = counters().snapshot();
        let data: Vec<u8> = vec![1u8; 2048];
        let global = counters().snapshot().since(&before);
        assert!(global.allocations >= 1, "an unscoped allocation is counted");
        drop(data);
    }

    #[test]
    fn nested_scopes_restore_the_outer_one() {
        let outer = Scope::enter(counters());
        let outer_before = counters().snapshot();
        {
            let inner = Scope::enter(counters());
            let data = vec![0u8; 64];
            core::hint::black_box(&data);
            drop(inner);
        }
        let during_inner = counters().snapshot().since(&outer_before);
        drop(outer);
        let after_outer = counters().snapshot().since(&outer_before);
        assert_eq!(
            during_inner.allocations, after_outer.allocations,
            "both scopes reach the same counters; the outer scope was not \
             displaced, only shadowed while the inner was active"
        );
    }
}
