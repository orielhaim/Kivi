//! One runtime snapshot of what this machine can do.
//!
//! Every accelerator in Kivi is optional and is discovered exactly once, at
//! startup, into this snapshot. No request path probes hardware, so no decision
//! can be taken against a machine fact that changed without anybody noticing.
//!
//! Absence is a value carrying a reason, never a `bool`: "this kernel forbids
//! `perf_event_open`" and "this CPU has no PMU" call for different operator
//! action, and [`Unavailable`] is where that difference lives.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::OnceLock;

use crate::placement::PlacementStrategy;
use crate::topology::Topology;

/// Whether an optional capability is usable, and why not when it is not.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Support<T> {
    /// A probe opened the mechanism.
    Available(T),
    /// The mechanism is absent or refused, with the reason an operator can act
    /// on.
    Unavailable(Unavailable),
}

impl<T> Support<T> {
    /// True when a probe opened the mechanism.
    #[must_use]
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available(_))
    }

    /// The absence reason, when absent.
    #[must_use]
    pub fn unavailable(&self) -> Option<&Unavailable> {
        match self {
            Self::Available(_) => None,
            Self::Unavailable(reason) => Some(reason),
        }
    }

    /// The capability, when present.
    #[must_use]
    pub fn get(&self) -> Option<&T> {
        match self {
            Self::Available(value) => Some(value),
            Self::Unavailable(_) => None,
        }
    }
}

impl<T: fmt::Display> fmt::Display for Support<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Available(value) => write!(f, "available ({value})"),
            Self::Unavailable(reason) => write!(f, "unavailable ({reason})"),
        }
    }
}

/// Why an optional capability is missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Unavailable {
    /// This operating system has no such interface at all.
    UnsupportedPlatform,
    /// The kernel or firmware does not offer it on this machine.
    UnsupportedHardware,
    /// It exists but the process is not permitted to use it.
    PermissionDenied,
    /// The feature exists but was not compiled in.
    NotCompiledIn,
    /// The device or file the capability needs is absent.
    NotPresent,
}

impl fmt::Display for Unavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::UnsupportedPlatform => "unsupported-platform",
            Self::UnsupportedHardware => "unsupported-hardware",
            Self::PermissionDenied => "permission-denied",
            Self::NotCompiledIn => "not-compiled-in",
            Self::NotPresent => "not-present",
        };
        f.write_str(text)
    }
}

/// One instruction-set extension the running binary can use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum IsaFeature {
    /// x86 SSE4.2. Kivi's CRC32C path uses the `crc32` instruction when it is
    /// present, and a table implementation when it is not.
    Sse42,
    /// x86 AVX2, used by runtime-multiversioned Kivi kernels and by BLAKE3.
    Avx2,
    /// x86 AVX-512, usable without the pre-Ice-Lake frequency penalty.
    Avx512,
    /// AArch64 `NEON`, mandatory in the aarch64 baseline, so Kivi's
    /// runtime-multiversioned kernels always take the vector path there.
    Neon,
    /// AArch64 CRC32 instructions, mandatory in the same baseline.
    Crc32,
}

impl fmt::Display for IsaFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Sse42 => "sse4.2",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Neon => "neon",
            Self::Crc32 => "crc32",
        };
        f.write_str(name)
    }
}

/// The instruction sets the running binary can use right now.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IsaSupport {
    features: BTreeSet<IsaFeature>,
}

impl IsaSupport {
    /// Reads the running binary's usable instruction sets.
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            let mut features = BTreeSet::new();
            // SAFETY: each macro expands to a cached CPUID or XGETBV probe with
            // no arguments and no preconditions.
            #[allow(unsafe_code)]
            {
                if std::arch::is_x86_feature_detected!("sse4.2") {
                    features.insert(IsaFeature::Sse42);
                }
                if std::arch::is_x86_feature_detected!("avx2") {
                    features.insert(IsaFeature::Avx2);
                }
                // "Usable" means the full Ice Lake set: on Skylake-SP and
                // Cascade Lake the features exist but running them drops the
                // frequency of every core on the socket.
                if std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
                    && std::arch::is_x86_feature_detected!("avx512vl")
                    && std::arch::is_x86_feature_detected!("avx512dq")
                    && std::arch::is_x86_feature_detected!("avx512cd")
                    && std::arch::is_x86_feature_detected!("gfni")
                    && std::arch::is_x86_feature_detected!("vpclmulqdq")
                {
                    features.insert(IsaFeature::Avx512);
                }
            }
            Self { features }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // NEON and CRC32 are mandatory in the aarch64 baseline, so there
            // is nothing to probe and no path that could ever be missing.
            Self {
                features: BTreeSet::from([IsaFeature::Neon, IsaFeature::Crc32]),
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
        {
            Self::default()
        }
    }

    /// True when the feature is usable.
    #[must_use]
    pub fn has(&self, feature: IsaFeature) -> bool {
        self.features.contains(&feature)
    }

    /// Features, ascending.
    pub fn iter(&self) -> impl Iterator<Item = IsaFeature> + '_ {
        self.features.iter().copied()
    }
}

/// Huge-page backing, when the platform offers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HugePageSupport {
    /// Transparent huge pages can be advised on an ordinary mapping. No
    /// reservation, no privilege, no pre-commitment, and a mis-sized region
    /// simply stays 4 KiB.
    Transparent,
    /// Explicit huge pages are reserved and can be mapped directly, which needs
    /// a privileged reservation and a page count committed up front.
    Reserved {
        /// Bytes in one reserved page.
        page_bytes: u64,
        /// Pages the pool currently holds, zero when the pool is not visible.
        count: u64,
    },
    /// The platform offers neither.
    None,
}

impl fmt::Display for HugePageSupport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transparent => write!(f, "transparent"),
            Self::Reserved { page_bytes, count } => {
                write!(f, "reserved(page={page_bytes},count={count})")
            }
            Self::None => write!(f, "none"),
        }
    }
}

/// What a provider can do with direct I/O.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DirectIoSupport {
    /// Direct I/O is available. Aligning the buffer address, the length and the
    /// file offset to `sector_bytes` is the caller's obligation.
    Unbuffered {
        /// The volume's logical sector size, which every address, length and
        /// offset must be a multiple of.
        sector_bytes: u32,
    },
    /// A direct I/O mode exists but the volume's sector size could not be
    /// determined, so Kivi must not guess an alignment.
    UnknownAlignment,
    /// No direct I/O mode.
    None,
}

impl fmt::Display for DirectIoSupport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unbuffered { sector_bytes } => write!(f, "unbuffered(sector={sector_bytes})"),
            Self::UnknownAlignment => write!(f, "mode-present/alignment-unknown"),
            Self::None => write!(f, "none"),
        }
    }
}

/// One hardware counter the performance monitoring unit can provide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum PmuCounter {
    /// Hardware cycles.
    Cycles,
    /// Instructions retired.
    Instructions,
    /// Last-level cache references.
    CacheReferences,
    /// Last-level cache misses.
    CacheMisses,
    /// Stalled cycles, which is what separates a memory-bound worker from a
    /// compute-bound one.
    StalledCycles,
    /// Branch instructions.
    Branches,
    /// Branch mispredictions.
    BranchMisses,
    /// Page faults, including the minor faults a worker takes when it touches
    /// an arena for the first time.
    PageFaults,
    /// Context switches, which is how often a worker was descheduled.
    ContextSwitches,
    /// CPU migrations of the calling thread. Unambiguous where a context switch
    /// is not: a migration means the worker is running on a different core than
    /// the one its placement plan chose.
    Migrations,
    /// Nanoseconds the calling thread was scheduled for.
    ///
    /// Not a hardware counter — the kernel synthesises it — but it is the only
    /// interval denominator a machine with no virtualised PMU has, and naming it
    /// separately is what lets a consumer tell "no denominator" from "hardware
    /// cycles".
    TaskClock,
}

impl fmt::Display for PmuCounter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Cycles => "cycles",
            Self::Instructions => "instructions",
            Self::CacheReferences => "cache-references",
            Self::CacheMisses => "cache-misses",
            Self::StalledCycles => "stalled-cycles",
            Self::Branches => "branches",
            Self::BranchMisses => "branch-misses",
            Self::PageFaults => "page-faults",
            Self::ContextSwitches => "context-switches",
            Self::Migrations => "migrations",
            Self::TaskClock => "task-clock",
        };
        f.write_str(name)
    }
}

/// Counters available from the performance monitoring unit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PmuSupport {
    /// Counters this process may read. A counter the kernel refused must not
    /// appear: a consumer would otherwise divide by a count nothing records.
    counters: BTreeSet<PmuCounter>,
}

impl PmuSupport {
    /// Builds a support description from the counters a backend could open.
    #[must_use]
    pub fn new(counters: impl IntoIterator<Item = PmuCounter>) -> Self {
        Self {
            counters: counters.into_iter().collect(),
        }
    }

    /// Records a counter the backend opened.
    pub fn insert(&mut self, counter: PmuCounter) {
        self.counters.insert(counter);
    }

    /// True when the counter may be read.
    #[must_use]
    pub fn has(&self, counter: PmuCounter) -> bool {
        self.counters.contains(&counter)
    }

    /// Counters, ascending.
    pub fn iter(&self) -> impl Iterator<Item = PmuCounter> + '_ {
        self.counters.iter().copied()
    }
}

impl fmt::Display for PmuSupport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<String> = self.iter().map(|counter| counter.to_string()).collect();
        if names.is_empty() {
            return f.write_str("none");
        }
        write!(f, "{}", names.join(","))
    }
}

/// The knobs the kernel's DAMON interface offers.
///
/// `free_us` is the region-scan interval in microseconds, the smallest unit
/// DAMON aggregates over: a Kivi arena smaller than that cannot be resolved by
/// DAMON however the sampling is configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DamonParams {
    /// Region-scan interval in microseconds.
    pub free_us: u64,
    /// Whether the kernel's aggressive aggregation scheme is available.
    pub aggressive: bool,
}

impl fmt::Display for DamonParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "free={}us,aggressive={}", self.free_us, self.aggressive)
    }
}

/// One advanced `io_uring` facility the running kernel offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum IoUringFeature {
    /// Registered files, which remove a per-operation path lookup.
    RegisteredFiles,
    /// Registered buffers, which remove a per-operation mapping lookup and are
    /// the prerequisite for the fastest direct I/O.
    RegisteredBuffers,
    /// Buffer rings, which make buffer selection a shared-memory operation
    /// instead of one in the submission queue.
    BufferRing,
    /// Completion queue events, letting Kivi ask for the exact number of
    /// completions it intends to wait for.
    CompletionEvents,
    /// Kernel submission polling, which trades a dedicated core for lower
    /// submission latency. Off by default: the kernel thread lives as long as
    /// the ring.
    SubmissionPoll,
    /// Completion polling, which removes the interrupt entirely and requires
    /// direct I/O on a device that supports it. Off by default: only a thread
    /// that spins can drain such a queue.
    CompletionPoll,
}

impl fmt::Display for IoUringFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::RegisteredFiles => "registered-files",
            Self::RegisteredBuffers => "registered-buffers",
            Self::BufferRing => "buffer-ring",
            Self::CompletionEvents => "completion-events",
            Self::SubmissionPoll => "sqpoll",
            Self::CompletionPoll => "iopoll",
        };
        f.write_str(name)
    }
}

/// Advanced `io_uring` features the running kernel offers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IoUringSupport {
    features: BTreeSet<IoUringFeature>,
}

impl IoUringSupport {
    /// Builds a support description from the features a backend negotiated.
    #[must_use]
    pub fn new(features: impl IntoIterator<Item = IoUringFeature>) -> Self {
        Self {
            features: features.into_iter().collect(),
        }
    }

    /// True when the facility is available.
    #[must_use]
    pub fn has(&self, feature: IoUringFeature) -> bool {
        self.features.contains(&feature)
    }

    /// Features, ascending.
    pub fn iter(&self) -> impl Iterator<Item = IoUringFeature> + '_ {
        self.features.iter().copied()
    }
}

impl fmt::Display for IoUringSupport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<String> = self.iter().map(|feature| feature.to_string()).collect();
        if names.is_empty() {
            return f.write_str("none");
        }
        write!(f, "{}", names.join(","))
    }
}

/// One CXL or pmem region the memory fabric can use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRegion {
    /// Path the region is reached through: `/dev/daxN.Y`, a DAX filesystem
    /// path, or a hotplugged kmem node.
    pub path: String,
    /// Bytes the region can hold.
    pub capacity_bytes: u64,
    /// Mapping granularity the region requires.
    pub align_bytes: u64,
    /// Memory node the region's pages belong to, when the kernel says.
    pub numa_node: Option<crate::topology::NumaNodeId>,
    /// Whether the backend proved the region survives power loss. CXL memory may
    /// be volatile or persistent depending on the device and how it was
    /// configured, so a durability contract may only ever name a region that
    /// proved it.
    pub persistent: bool,
}

/// The whole machine, once.
#[derive(Debug, Clone)]
pub struct Capabilities {
    /// The physical topology the snapshot was derived from.
    pub topology: Topology,
    /// Instruction sets the running binary can use.
    pub isa: IsaSupport,
    /// Huge-page backing.
    pub huge_pages: Support<HugePageSupport>,
    /// Whether direct I/O is available, and against which alignment.
    pub direct_io: Support<DirectIoSupport>,
    /// Performance counters, when the platform and privileges allow.
    pub pmu: Support<PmuSupport>,
    /// Kernel DAMON access, the memory-access monitor.
    pub damon: Support<DamonParams>,
    /// Advanced `io_uring` features the running kernel offers.
    pub io_ring: Support<IoUringSupport>,
    /// CXL and pmem regions discovered on this machine.
    pub memory_regions: Vec<MemoryRegion>,
    /// RDMA character devices present.
    pub rdma_devices: Vec<String>,
    /// Intel Data Streaming Accelerator engines present.
    pub dsa_engines: Vec<String>,
    /// The placement strategy that will be used.
    pub placement: PlacementStrategy,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            topology: Topology::single_core(),
            isa: IsaSupport::default(),
            huge_pages: Support::Unavailable(Unavailable::NotPresent),
            direct_io: Support::Unavailable(Unavailable::NotPresent),
            pmu: Support::Unavailable(Unavailable::NotPresent),
            damon: Support::Unavailable(Unavailable::NotPresent),
            io_ring: Support::Unavailable(Unavailable::NotPresent),
            memory_regions: Vec::new(),
            rdma_devices: Vec::new(),
            dsa_engines: Vec::new(),
            placement: PlacementStrategy::Unbound,
        }
    }
}

impl PartialEq for Capabilities {
    fn eq(&self, other: &Self) -> bool {
        // Topology carries cache and device descriptions that are not
        // comparable without ordering the maps behind them; its summary is a
        // deterministic stand-in for the descriptive half.
        self.topology.summary() == other.topology.summary()
            && self.isa == other.isa
            && self.huge_pages == other.huge_pages
            && self.direct_io == other.direct_io
            && self.pmu == other.pmu
            && self.damon == other.damon
            && self.io_ring == other.io_ring
            && self.memory_regions == other.memory_regions
            && self.rdma_devices == other.rdma_devices
            && self.dsa_engines == other.dsa_engines
            && self.placement == other.placement
    }
}

impl Eq for Capabilities {}

/// One optional mechanism, and whether a probe opened it.
///
/// Enumerated devices (`memory_regions`, `rdma_devices`, `dsa_engines`) report
/// as unavailable with [`Unavailable::NotPresent`]: enumeration proves hardware
/// exists, and nothing claims more than that for a device no backend has opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mechanism {
    /// Stable name for logs and the admin surface.
    pub name: &'static str,
    /// Available when a probe opened the mechanism.
    pub support: Support<()>,
}

impl Capabilities {
    /// Reads the whole machine once.
    #[must_use]
    pub fn detect() -> Self {
        Self::detect_with(PlacementStrategy::default())
    }

    /// Reads the whole machine once, with a chosen placement strategy.
    #[must_use]
    pub fn detect_with(placement: PlacementStrategy) -> Self {
        Self {
            isa: IsaSupport::detect(),
            huge_pages: crate::pages::detect(),
            direct_io: crate::io::detect(),
            pmu: crate::telemetry::pmu_support(),
            damon: crate::telemetry::damon_support(),
            io_ring: crate::telemetry::io_uring_support(),
            memory_regions: crate::dax::discover_regions(&Topology::discover()),
            rdma_devices: crate::transport::discover_rdma(),
            dsa_engines: crate::accelerator::discover_dsa(),
            topology: Topology::discover(),
            placement,
        }
    }

    /// Every optional mechanism with whether a probe opened it.
    #[must_use]
    pub fn mechanisms(&self) -> Vec<Mechanism> {
        let mut out = vec![
            Mechanism {
                name: "huge-pages",
                support: erased(&self.huge_pages),
            },
            Mechanism {
                name: "direct-io",
                support: erased(&self.direct_io),
            },
            Mechanism {
                name: "pmu",
                support: erased(&self.pmu),
            },
            Mechanism {
                name: "damon",
                support: erased(&self.damon),
            },
            Mechanism {
                name: "io-uring",
                support: erased(&self.io_ring),
            },
        ];
        for (name, present) in [
            ("cxl-memory", !self.memory_regions.is_empty()),
            ("rdma", !self.rdma_devices.is_empty()),
            ("dsa", !self.dsa_engines.is_empty()),
        ] {
            out.push(Mechanism {
                name,
                support: if present {
                    Support::Available(())
                } else {
                    Support::Unavailable(Unavailable::NotPresent)
                },
            });
        }
        out
    }
}

/// Erases a detection result's value, keeping only whether it succeeded.
fn erased<T>(support: &Support<T>) -> Support<()> {
    match support {
        Support::Available(_) => Support::Available(()),
        Support::Unavailable(reason) => Support::Unavailable(*reason),
    }
}

static SNAPSHOT: OnceLock<Capabilities> = OnceLock::new();

/// The process-wide capability snapshot, detected on first use.
///
/// Detection reads sysfs, device-control codes and a handful of CPUID bits. It
/// runs exactly once so no request path pays for it, and the result is shared
/// so two subsystems cannot disagree about the machine.
#[must_use]
pub fn snapshot() -> &'static Capabilities {
    SNAPSHOT.get_or_init(Capabilities::detect)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_never_fails_and_always_has_a_cpu() {
        let capabilities = Capabilities::detect();
        assert!(capabilities.topology.logical_cpu_count() >= 1);
        assert!(capabilities.topology.physical_core_count() >= 1);
    }

    #[test]
    fn detection_is_deterministic() {
        assert_eq!(Capabilities::detect(), Capabilities::detect());
    }

    #[test]
    fn snapshot_is_shared() {
        assert!(std::ptr::eq(snapshot(), snapshot()));
    }

    #[test]
    fn unavailable_carries_a_reason() {
        let support: Support<u8> = Support::Unavailable(Unavailable::PermissionDenied);
        assert_eq!(support.get(), None);
        assert_eq!(support.unavailable(), Some(&Unavailable::PermissionDenied));
        assert!(!support.is_available());
    }

    #[test]
    fn every_mechanism_is_reported_exactly_once() {
        let capabilities = Capabilities::detect();
        let mechanisms = capabilities.mechanisms();
        let mut names: Vec<&str> = mechanisms.iter().map(|m| m.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), mechanisms.len());
        assert!(names.contains(&"pmu"));
    }

    #[test]
    fn an_enumerated_device_is_not_a_backend() {
        // Enumeration proves hardware exists. It does not prove a Kivi backend
        // opened it, so these are reported by presence, not as usable.
        let capabilities = Capabilities::detect();
        for mechanism in capabilities.mechanisms() {
            if matches!(mechanism.name, "cxl-memory" | "rdma" | "dsa") {
                assert_eq!(
                    mechanism.support.unavailable(),
                    (!mechanism.support.is_available()).then_some(&Unavailable::NotPresent)
                );
            }
        }
    }
}
