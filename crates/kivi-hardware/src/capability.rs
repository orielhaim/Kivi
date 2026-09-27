//! One runtime snapshot of what this machine can do.
//!
//! Every accelerator in Kivi is *optional* and every one of them is discovered
//! exactly once, at startup, into this snapshot. Nothing above this module
//! probes hardware, so no request path can be delayed by a device check, and
//! no decision can be taken against a machine fact that has since changed
//! without noticing the change.
//!
//! Every dimension is an enum with an `Unavailable` variant carrying a reason.
//! That is deliberate: a `bool` cannot say *why* something is missing, and the
//! difference between "this kernel forbids `perf_event_open`" and "this CPU has
//! no PMU" changes what an operator should do about it.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fmt;
use std::sync::OnceLock;

use crate::placement::PlacementStrategy;
use crate::topology::Topology;

/// Whether an optional capability is usable, and why not when it is not.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Support<T> {
    /// The capability is present and usable.
    Available(T),
    /// The capability is absent or refused. Carries the reason so an operator
    /// can act on it.
    Unavailable(Unavailable),
}

impl<T> Support<T> {
    /// The capability, when present.
    pub fn get(&self) -> Option<&T> {
        match self {
            Self::Available(value) => Some(value),
            Self::Unavailable(_) => None,
        }
    }

    /// True when present.
    #[must_use]
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available(_))
    }

    /// True when absent.
    #[must_use]
    pub fn is_unavailable(&self) -> bool {
        matches!(self, Self::Unavailable(_))
    }

    /// The absence reason, when absent.
    #[must_use]
    pub fn unavailable(&self) -> Option<&Unavailable> {
        match self {
            Self::Available(_) => None,
            Self::Unavailable(reason) => Some(reason),
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
    /// The capability was switched off by configuration.
    DisabledByConfig,
}

impl fmt::Display for Unavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::UnsupportedPlatform => "unsupported-platform",
            Self::UnsupportedHardware => "unsupported-hardware",
            Self::PermissionDenied => "permission-denied",
            Self::NotCompiledIn => "not-compiled-in",
            Self::NotPresent => "not-present",
            Self::DisabledByConfig => "disabled-by-config",
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
    /// AArch64 `NEON`, mandatory on every 64-bit ARM core, so Kivi's
    /// runtime-multiversioned kernels always take the vector path there.
    Neon,
    /// AArch64 CRC32 instructions, which `CRC32C` is mandatory to have.
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
            // no arguments and no preconditions. They are the standard way to
            // ask the running CPU what it supports.
            #[allow(unsafe_code)]
            {
                if std::arch::is_x86_feature_detected!("sse4.2") {
                    features.insert(IsaFeature::Sse42);
                }
                if std::arch::is_x86_feature_detected!("avx2") {
                    features.insert(IsaFeature::Avx2);
                }
                // "usable" means the full Ice Lake feature set. On Skylake-SP
                // and Cascade Lake the features exist but running them drops
                // the frequency of every core on the socket, which costs more
                // than the wider vectors gain.
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

impl fmt::Display for IsaSupport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<String> = self.iter().map(|feature| feature.to_string()).collect();
        f.write_str(&if names.is_empty() {
            "none".to_string()
        } else {
            names.join(",")
        })
    }
}

/// Huge-page backing, when the platform offers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HugePageSupport {
    /// Transparent huge pages can be advised on an ordinary mapping. This is
    /// the low-friction mechanism: no reservation, no privilege, no
    /// pre-commitment, and a mis-sized region simply stays 4 KiB.
    Transparent,
    /// Explicit huge pages are reserved and can be mapped directly, which
    /// needs a privileged reservation and a page count committed up front.
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
    /// Direct I/O is available and the volume's logical sector size is known.
    /// Alignment of buffer address, length and file offset to that sector size
    /// is the caller's obligation.
    Unbuffered {
        /// The volume's logical sector size, which every address, length and
        /// offset must be a multiple of.
        sector_bytes: u32,
    },
    /// The platform has a direct I/O mode but the volume's sector size could
    /// not be determined, so Kivi must not guess an alignment.
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
    /// CPU migrations of the calling thread. Unlike a context switch this is
    /// unambiguous: a migration means the worker was moved to a different core,
    /// so the cache and the memory node the placement plan chose for it are not
    /// the ones it is using.
    Migrations,
    /// Nanoseconds the calling thread was scheduled for.
    ///
    /// Not a *hardware* counter — the kernel synthesises it — but it is the only
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
    /// Counters this process may read.
    counters: BTreeSet<PmuCounter>,
    /// True when the kernel multiplexes counters across several events, meaning
    /// each count is scaled rather than exact.
    multiplexed: bool,
}

impl PmuSupport {
    /// Builds a support description from the counters a backend could open.
    #[must_use]
    pub fn new(counters: impl IntoIterator<Item = PmuCounter>, multiplexed: bool) -> Self {
        Self {
            counters: counters.into_iter().collect(),
            multiplexed,
        }
    }

    /// Records that a counter was opened, or dropped because the kernel
    /// refused it. A support set is written by the backend as it opens counters
    /// and read by everything above it, so a counter the kernel declined must
    /// not appear here: a consumer would otherwise divide by a count nothing is
    /// recording.
    pub fn insert(&mut self, counter: PmuCounter) {
        self.counters.insert(counter);
    }

    /// Records that the kernel could not schedule the whole counter group for
    /// the full interval. Sticky: once the kernel has demonstrated that this
    /// workload outruns the available counters, every later sample is treated
    /// as an estimate.
    pub const fn set_multiplexed(&mut self) {
        self.multiplexed = true;
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

    /// Whether counts are scaled estimates rather than exact.
    #[must_use]
    pub const fn multiplexed(&self) -> bool {
        self.multiplexed
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
    /// Whether the backend proved the region survives power loss. CXL memory
    /// may be volatile or persistent depending on the device and how it was
    /// configured, so Kivi never assumes and the durability contract only ever
    /// depends on a region that proved it.
    pub persistent: bool,
}

/// Where a capability stands, as five distinct answers rather than one.
///
/// The distinction exists because a single boolean forces an operator to guess.
/// "The PMU is off" could mean the CPU has no performance counters (buy
/// different hardware), the kernel does not implement them (use a different
/// kernel), this process is not permitted (change `perf_event_paranoid`), the
/// build left it out (change a feature flag), or a backend failed to initialise
/// (a bug). Those are five different conversations, and collapsing them into
/// `false` is what made the previous phase's `DirectIoPolicy` and `huge_pages`
/// settings decorative: the snapshot said "available" for a mechanism nothing
/// ever opened, and "unavailable" for one that worked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CapabilityStage {
    /// The hardware exists. Detected by enumeration.
    HardwarePresent,
    /// The operating system offers an interface to it. The backend's probe
    /// succeeded, so the mechanism is implemented in the running kernel.
    OsSupported,
    /// This process is permitted to use it. A separate answer from OS support
    /// because most hardware facilities are root-only by default and Kivi must
    /// never require elevation.
    Permitted,
    /// Kivi's code path for it is compiled in. A build-time fact, distinct
    /// from a runtime one.
    BackendBuilt,
    /// A Kivi backend opened it successfully. The only stage that proves the
    /// mechanism is actually in use.
    BackendInitialised,
}

impl CapabilityStage {
    /// Every stage in order, weakest first.
    pub const ORDER: [Self; 5] = [
        Self::HardwarePresent,
        Self::OsSupported,
        Self::Permitted,
        Self::BackendBuilt,
        Self::BackendInitialised,
    ];

    /// How far this stage is from usable, as an index into
    /// [`CapabilityStage::ORDER`]. `0` means the hardware is not there at all.
    #[must_use]
    pub const fn depth(self) -> usize {
        match self {
            Self::HardwarePresent => 0,
            Self::OsSupported => 1,
            Self::Permitted => 2,
            Self::BackendBuilt => 3,
            Self::BackendInitialised => 4,
        }
    }

    /// Whether the capability reached the final stage: a Kivi backend has it
    /// open right now.
    #[must_use]
    pub const fn is_usable(self) -> bool {
        matches!(self, Self::BackendInitialised)
    }

    /// The first stage this capability did not reach, phrased as what an
    /// operator would have to fix.
    #[must_use]
    pub const fn blocked_at(self) -> Option<&'static str> {
        match self {
            Self::HardwarePresent => Some("hardware is absent"),
            Self::OsSupported => Some("the operating system does not offer the interface"),
            Self::Permitted => Some("this process is not permitted to use it"),
            Self::BackendBuilt => Some("kivi's backend is not compiled in"),
            Self::BackendInitialised => None,
        }
    }
}

impl fmt::Display for CapabilityStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::HardwarePresent => "hardware-present",
            Self::OsSupported => "os-supported",
            Self::Permitted => "permitted",
            Self::BackendBuilt => "backend-built",
            Self::BackendInitialised => "backend-initialised",
        };
        f.write_str(name)
    }
}

/// How far one mechanism got, and what stopped it.
///
/// Both halves are needed and neither is derivable from the other. The stage
/// without the reason says "the PMU is unavailable" and leaves an operator
/// guessing between a machine change, a kernel change and a build flag; the
/// reason without the stage says "permission-denied" on a machine that has no
/// PMU at all, which is a confusing way to be told the truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MechanismStatus {
    /// The furthest stage reached.
    pub stage: CapabilityStage,
    /// Why the mechanism is not usable. `None` when it is.
    pub reason: Option<Unavailable>,
}

impl MechanismStatus {
    /// A usable mechanism: a Kivi backend has it open.
    #[must_use]
    pub const fn usable() -> Self {
        Self {
            stage: CapabilityStage::BackendInitialised,
            reason: None,
        }
    }

    /// A mechanism that stopped at `stage` for `reason`.
    #[must_use]
    pub const fn stopped(stage: CapabilityStage, reason: Unavailable) -> Self {
        Self {
            stage,
            reason: Some(reason),
        }
    }

    /// Whether a Kivi backend has this open right now.
    #[must_use]
    pub const fn is_usable(&self) -> bool {
        self.stage.is_usable()
    }

    /// What an operator would have to change, when there is something to
    /// change.
    #[must_use]
    pub const fn blocked_at(&self) -> Option<&'static str> {
        if self.is_usable() {
            None
        } else {
            self.stage.blocked_at()
        }
    }
}

impl fmt::Display for MechanismStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.reason {
            None => write!(f, "{}", self.stage),
            Some(reason) => write!(f, "{}[{reason}]", self.stage),
        }
    }
}

/// One optional mechanism Kivi knows how to use.
///
/// The set is closed and small. A mechanism is not a knob: every entry here
/// either has a backend that opens it or is deliberately absent from the tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Mechanism {
    /// Hardware performance counters.
    Pmu,
    /// The kernel's memory access monitor.
    Damon,
    /// Transparent or reserved huge pages.
    HugePages,
    /// Bypassing the page cache for I/O.
    DirectIo,
    /// Advanced `io_uring` features.
    IoUring,
    /// CXL and pmem regions.
    CxlMemory,
    /// RDMA character devices.
    Rdma,
    /// Intel Data Streaming Accelerator engines.
    Dsa,
}

impl fmt::Display for Mechanism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Pmu => "pmu",
            Self::Damon => "damon",
            Self::HugePages => "huge-pages",
            Self::DirectIo => "direct-io",
            Self::IoUring => "io-uring",
            Self::CxlMemory => "cxl-memory",
            Self::Rdma => "rdma",
            Self::Dsa => "dsa",
        };
        f.write_str(name)
    }
}

/// Resolves each mechanism to the furthest stage it actually reached.
///
/// The stage is derived from the detection result, not asserted alongside it,
/// so a mechanism cannot claim a stage its own probe did not demonstrate.
fn resolve_stages(
    huge_pages: &Support<HugePageSupport>,
    direct_io: &Support<DirectIoSupport>,
    pmu: &Support<PmuSupport>,
    damon: &Support<DamonSupport>,
    io_uring: &Support<IoUringSupport>,
) -> BTreeMap<Mechanism, MechanismStatus> {
    let mut stages = BTreeMap::new();
    stages.insert(Mechanism::Pmu, stage_of(&discard(pmu)));
    stages.insert(Mechanism::Damon, stage_of(&discard(damon)));
    stages.insert(Mechanism::HugePages, stage_of(&discard(huge_pages)));
    stages.insert(Mechanism::DirectIo, stage_of(&discard(direct_io)));
    stages.insert(Mechanism::IoUring, stage_of(&discard(io_uring)));
    stages
}

/// Erases a detection result's value, keeping only whether it succeeded.
fn discard<T>(support: &Support<T>) -> Support<()> {
    match support {
        Support::Available(_) => Support::Available(()),
        Support::Unavailable(reason) => Support::Unavailable(*reason),
    }
}

/// The status a two-state detection result corresponds to.
///
/// `available` means a probe succeeded, which is the strongest thing a probe
/// can establish: the mechanism is detected, the OS supports it, this process
/// is permitted, the backend is compiled in, and the backend initialised. A
/// caller that needs the finer distinction must open the backend, which is
/// exactly the right place to pay for it.
fn stage_of(support: &Support<()>) -> MechanismStatus {
    match support {
        Support::Available(()) => MechanismStatus::usable(),
        Support::Unavailable(reason) => {
            let stage = match reason {
                Unavailable::UnsupportedPlatform => CapabilityStage::HardwarePresent,
                Unavailable::UnsupportedHardware => CapabilityStage::OsSupported,
                Unavailable::PermissionDenied => CapabilityStage::Permitted,
                Unavailable::NotCompiledIn => CapabilityStage::BackendBuilt,
                Unavailable::NotPresent | Unavailable::DisabledByConfig => {
                    CapabilityStage::HardwarePresent
                }
            };
            MechanismStatus::stopped(stage, *reason)
        }
    }
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
    pub damon: Support<DamonSupport>,
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
    /// How far each optional mechanism actually got, and what stopped it.
    /// This is the answer to "is this knob real", and it is what the admin
    /// surface reports.
    pub stages: BTreeMap<Mechanism, MechanismStatus>,
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
            stages: BTreeMap::new(),
        }
    }
}

impl PartialEq for Capabilities {
    fn eq(&self, other: &Self) -> bool {
        // Topology carries cache and device descriptions that are not
        // comparable without ordering the maps behind them. Every field that a
        // decision depends on is compared; the descriptive rest is compared
        // through its own summary, which is deterministic.
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

impl Capabilities {
    /// Reads the whole machine once.
    #[must_use]
    pub fn detect() -> Self {
        Self::detect_with(PlacementStrategy::default())
    }

    /// Reads the whole machine once, with a chosen placement strategy.
    #[must_use]
    pub fn detect_with(placement: PlacementStrategy) -> Self {
        let topology = Topology::discover();
        let huge_pages = crate::pages::detect();
        let direct_io = crate::io::detect();
        let pmu = crate::telemetry::pmu_support();
        let damon = crate::telemetry::damon_support();
        let io_ring = crate::telemetry::io_uring_support();
        let memory_regions = crate::dax::discover_regions(&topology);
        let rdma_devices = crate::transport::discover_rdma();
        let dsa_engines = crate::accelerator::discover_dsa();
        let mut stages = resolve_stages(&huge_pages, &direct_io, &pmu, &damon, &io_ring);
        // These three are discovered by enumeration rather than by opening
        // something, so their stage is stated by what the enumeration proved:
        // hardware present. Nothing claims `BackendInitialised` for a device
        // Kivi has not mapped, and nothing claims `OsSupported` for a device
        // the kernel did not expose.
        for (mechanism, count) in [
            (Mechanism::CxlMemory, memory_regions.len()),
            (Mechanism::Rdma, rdma_devices.len()),
            (Mechanism::Dsa, dsa_engines.len()),
        ] {
            stages.insert(
                mechanism,
                if count == 0 {
                    MechanismStatus::stopped(
                        CapabilityStage::HardwarePresent,
                        Unavailable::NotPresent,
                    )
                } else {
                    MechanismStatus::stopped(
                        CapabilityStage::OsSupported,
                        Unavailable::NotCompiledIn,
                    )
                },
            );
        }
        Self {
            isa: IsaSupport::detect(),
            huge_pages,
            direct_io,
            pmu,
            damon,
            io_ring,
            memory_regions,
            rdma_devices,
            dsa_engines,
            topology,
            placement,
            stages,
        }
    }

    /// How far one mechanism actually got, and what stopped it.
    #[must_use]
    pub fn mechanism(&self, mechanism: Mechanism) -> MechanismStatus {
        self.stages
            .get(&mechanism)
            .copied()
            .unwrap_or(MechanismStatus::stopped(
                CapabilityStage::HardwarePresent,
                Unavailable::NotPresent,
            ))
    }

    /// One line per mechanism, `name=status`, for the admin surface.
    #[must_use]
    pub fn stage_report(&self) -> String {
        self.stages
            .iter()
            .map(|(mechanism, status)| format!("{mechanism}={status}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The machine's topology, borrowed from this snapshot.
    #[must_use]
    pub fn topology_logical(&self) -> &Topology {
        &self.topology
    }

    /// True when the process may use hardware performance counters.
    #[must_use]
    pub fn has_pmu(&self) -> bool {
        self.pmu.is_available()
    }

    /// Memory regions that proved persistence, which are the only ones a
    /// durability contract may name.
    pub fn persistent_regions(&self) -> impl Iterator<Item = &MemoryRegion> {
        self.memory_regions
            .iter()
            .filter(|region| region.persistent)
    }

    /// Memory regions that did not prove persistence. Safe to cache, never
    /// safe to promise durability on.
    pub fn volatile_regions(&self) -> impl Iterator<Item = &MemoryRegion> {
        self.memory_regions
            .iter()
            .filter(|region| !region.persistent)
    }

    /// Whether CXL or pmem memory exists on this machine at all.
    #[must_use]
    pub fn has_tiered_memory(&self) -> bool {
        !self.memory_regions.is_empty() || self.topology.cxl_nodes().next().is_some()
    }

    /// One line per dimension, for logs and the admin surface.
    #[must_use]
    pub fn report(&self) -> String {
        let regions = if self.memory_regions.is_empty() {
            "none".to_string()
        } else {
            self.memory_regions
                .iter()
                .map(|region| {
                    format!(
                        "{}({}MiB,persist={})",
                        region.path,
                        region.capacity_bytes / (1024 * 1024),
                        region.persistent
                    )
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        format!(
            "topology[{topology}] isa[{isa}] \
             huge_pages[{huge_pages}] direct_io[{direct_io}] pmu[{pmu}] damon[{damon}] \
             io_uring[{io_uring}] regions[{regions}] rdma[{rdma}] dsa[{dsa}] \
             placement[{placement}] stages[{stages}]",
            topology = self.topology.summary(),
            isa = self.isa,
            huge_pages = self.huge_pages,
            direct_io = self.direct_io,
            pmu = self.pmu,
            damon = self.damon,
            io_uring = self.io_ring,
            regions = regions,
            rdma = if self.rdma_devices.is_empty() {
                "none".to_string()
            } else {
                self.rdma_devices.join(",")
            },
            dsa = if self.dsa_engines.is_empty() {
                "none".to_string()
            } else {
                self.dsa_engines.join(",")
            },
            placement = self.placement,
            stages = self.stage_report(),
        )
    }
}

/// DAMON, the kernel's memory access monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DamonSupport {
    /// The kernel control interface is present and writable by this process.
    /// `free_us` is the region-scan interval in microseconds, which is the
    /// smallest unit DAMON aggregates over: a Kivi arena smaller than that
    /// cannot be resolved by DAMON no matter how the sampling is configured.
    Monitoring {
        /// Region-scan interval in microseconds, the smallest unit DAMON
        /// aggregates over.
        free_us: u64,
        /// Whether the kernel's aggressive aggregation scheme is available.
        aggressive: bool,
    },
}

impl fmt::Display for DamonSupport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Monitoring {
                free_us,
                aggressive,
            } => {
                write!(f, "monitoring(free={free_us}us,aggressive={aggressive})")
            }
        }
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
        f.write_str(&if names.is_empty() {
            "none".to_string()
        } else {
            names.join(",")
        })
    }
}

static SNAPSHOT: OnceLock<Capabilities> = OnceLock::new();

/// The process-wide capability snapshot, detected on first use.
///
/// Detection reads sysfs, device-control codes and a handful of CPUID bits. It is
/// done exactly once so that no request path can ever pay for it, and the
/// result is shared so that two subsystems cannot disagree about the machine.
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
        let first = Capabilities::detect();
        let second = Capabilities::detect();
        assert_eq!(first, second);
    }

    #[test]
    fn snapshot_is_shared() {
        assert!(std::ptr::eq(snapshot(), snapshot()));
    }

    #[test]
    fn unavailable_carries_a_reason() {
        let support: Support<u8> = Support::Unavailable(Unavailable::PermissionDenied);
        assert!(support.is_unavailable());
        assert_eq!(support.unavailable(), Some(&Unavailable::PermissionDenied));
        assert_eq!(support.to_string(), "unavailable (permission-denied)");
    }

    #[test]
    fn report_mentions_every_dimension() {
        let report = Capabilities::detect().report();
        for dimension in [
            "topology",
            "isa",
            "huge_pages",
            "direct_io",
            "pmu",
            "damon",
            "io_uring",
            "regions",
            "rdma",
            "dsa",
            "placement",
            "stages",
        ] {
            assert!(
                report.contains(dimension),
                "missing {dimension} in {report}"
            );
        }
    }

    #[test]
    fn every_mechanism_has_a_stage() {
        // A mechanism with no stage is a mechanism whose status nobody can
        // report, which is how a knob becomes decorative.
        let capabilities = Capabilities::detect();
        for mechanism in [
            Mechanism::Pmu,
            Mechanism::Damon,
            Mechanism::HugePages,
            Mechanism::DirectIo,
            Mechanism::IoUring,
            Mechanism::CxlMemory,
            Mechanism::Rdma,
            Mechanism::Dsa,
        ] {
            assert!(
                capabilities.stages.contains_key(&mechanism),
                "{mechanism} has no stage"
            );
        }
        assert!(capabilities.stage_report().contains("pmu="));
    }

    #[test]
    fn an_available_probe_reaches_backend_initialised() {
        // A probe that opened the mechanism is the strongest evidence a probe
        // can produce, so it maps to the final stage rather than to
        // `OsSupported` — which would understate what was demonstrated.
        let status = stage_of(&Support::Available(()));
        assert_eq!(status.stage, CapabilityStage::BackendInitialised);
        assert!(status.is_usable());
        assert_eq!(status.blocked_at(), None);
        assert!(CapabilityStage::BackendInitialised.is_usable());
        assert_eq!(CapabilityStage::BackendInitialised.blocked_at(), None);
    }

    #[test]
    fn each_refusal_names_a_different_stage() {
        // The whole reason for the five states: an operator's next action
        // differs for each.
        for (reason, stage) in [
            (
                Unavailable::UnsupportedPlatform,
                CapabilityStage::HardwarePresent,
            ),
            (
                Unavailable::UnsupportedHardware,
                CapabilityStage::OsSupported,
            ),
            (Unavailable::PermissionDenied, CapabilityStage::Permitted),
            (Unavailable::NotCompiledIn, CapabilityStage::BackendBuilt),
            (Unavailable::NotPresent, CapabilityStage::HardwarePresent),
        ] {
            let status = stage_of(&Support::Unavailable(reason));
            assert_eq!(status.stage, stage, "{reason}");
            assert!(!status.is_usable());
            assert!(status.blocked_at().is_some_and(|text| !text.is_empty()));
            assert_eq!(status.reason, Some(reason), "the reason survives");
        }
    }

    #[test]
    fn a_status_names_both_the_stage_and_the_reason() {
        let stopped =
            MechanismStatus::stopped(CapabilityStage::Permitted, Unavailable::PermissionDenied);
        assert!(!stopped.is_usable());
        assert_eq!(stopped.reason, Some(Unavailable::PermissionDenied));
        assert!(stopped.to_string().contains("permission-denied"));
        assert!(stopped.to_string().contains("permitted"));
        let usable = MechanismStatus::usable();
        assert!(usable.is_usable());
        assert_eq!(usable.reason, None);
        assert!(usable.to_string().contains("backend-initialised"));
    }

    #[test]
    fn a_discovered_device_does_not_claim_a_backend() {
        // Enumeration proves hardware exists. It does not prove a Kivi backend
        // opened it, and a snapshot that confuses the two is a snapshot that
        // says a feature works because a device was plugged in.
        let capabilities = Capabilities::detect();
        for mechanism in [Mechanism::CxlMemory, Mechanism::Rdma, Mechanism::Dsa] {
            let status = capabilities.mechanism(mechanism);
            assert_ne!(
                status.stage,
                CapabilityStage::BackendInitialised,
                "{mechanism}"
            );
        }
    }
}
