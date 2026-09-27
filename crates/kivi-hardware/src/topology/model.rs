use std::collections::BTreeMap;
use std::fmt;

use crate::cpuset::CpuSet;

/// A logical processor, as the operating system enumerates it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CpuId(pub u32);

impl fmt::Display for CpuId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A physical core, identified across packages and dies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CoreKey {
    /// Socket the core is soldered to.
    pub package: u32,
    /// Die within the socket. Servers expose this; desktops report die 0.
    pub die: u32,
    /// Core index within the die. This is a topology identity, not a CPU id.
    pub core: u32,
}

impl fmt::Display for CoreKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pkg{}/die{}/core{}", self.package, self.die, self.core)
    }
}

/// A memory (NUMA) node. Node 0 always exists; a uniform machine has only it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NumaNodeId(pub u32);

impl fmt::Display for NumaNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node{}", self.0)
    }
}

/// A CPU package, that is a socket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PackageId(pub u32);

impl fmt::Display for PackageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pkg{}", self.0)
    }
}

/// Relative performance class of a logical CPU.
///
/// Hybrid parts (Intel Core Ultra, Apple silicon, server Intel with E-cores)
/// place performance and efficiency cores behind the same OS scheduler with no
/// way to ask which is which through a CPU id. A storage engine must ask:
/// pinning a latency-critical durability lane to an efficiency core costs more
/// than the placement is worth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CpuClass {
    /// Fastest class the machine reports.
    Performance,
    /// Slowest class the machine reports.
    Efficiency,
    /// Every CPU is the same. The overwhelming majority of machines.
    Uniform,
    /// The platform does not expose a per-CPU class.
    Unknown,
}

/// Cache geometry, when the platform discloses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CacheKind {
    /// Shared data and instruction cache.
    Unified,
    /// Data cache.
    Data,
    /// Instruction cache.
    Instruction,
}

/// One logical processor.
#[derive(Debug, Clone)]
pub struct LogicalCpu {
    /// Operating-system CPU index. On machines with more than 64 processors
    /// this is a (group, index) pair flattened by [`Topology::cpu_of`]; the
    /// raw index is not addressable on its own.
    pub id: CpuId,
    /// Physical core this CPU is a hardware thread of.
    pub core: CoreKey,
    /// Every hardware thread of [`LogicalCpu::core`], ascending, this CPU
    /// included. A single-element slice means SMT is off or unavailable.
    pub smt_siblings: Vec<CpuId>,
    /// Memory node whose DRAM is closest to this CPU.
    pub numa_node: NumaNodeId,
    /// Efficiency class as reported by the platform, if any.
    pub efficiency_class: Option<u8>,
    /// Performance class derived from [`LogicalCpu::efficiency_class`] and the
    /// machine's class spread.
    pub class: CpuClass,
}

impl LogicalCpu {
    /// True when the CPU shares its physical core with at least one other
    /// hardware thread.
    #[must_use]
    pub fn shares_core(&self) -> bool {
        self.smt_siblings.len() > 1
    }
}

/// One physical core with its hardware threads.
#[derive(Debug, Clone)]
pub struct PhysicalCore {
    /// Identity across packages and dies.
    pub key: CoreKey,
    /// Hardware threads, ascending.
    pub logical_cpus: Vec<CpuId>,
    /// NUMA node of the first hardware thread. A core spans at most one node.
    pub numa_node: NumaNodeId,
    /// Performance class shared by every hardware thread.
    pub class: CpuClass,
    /// The last-level cache instance serving this core, if known.
    pub last_level_cache: Option<CacheLevel>,
    /// Whether the machine's last-level cache is shared with another core.
    /// Such a core is LLC-coupled and must not host two owners whose working
    /// sets both matter.
    pub llc_shared: bool,
}

impl PhysicalCore {
    /// The hardware thread used as the placement unit. Ascending CPU id keeps
    /// the choice stable across restarts and across discovery order.
    #[must_use]
    pub fn anchor_cpu(&self) -> CpuId {
        self.logical_cpus
            .first()
            .copied()
            .unwrap_or(CpuId(u32::MAX))
    }
}

/// One cache instance shared by a set of cores.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheLevel {
    /// 1 for L1, 2 for L2, 3 for L3.
    pub level: u8,
    /// Whether the cache holds data, instructions, or both.
    pub kind: CacheKind,
    /// Cache line size in bytes.
    pub line_bytes: u32,
    /// Total instance size in bytes.
    pub size_bytes: u64,
    /// Cores served by this instance.
    pub shared_by: Vec<CoreKey>,
}

/// Memory technology of a node, when the platform discloses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryTier {
    /// Ordinary DRAM.
    Ddr,
    /// CXL-attached memory, exposed either as a memory node or as a DAX
    /// device. Volatile or persistent depending on the device; Kivi never
    /// assumes.
    Cxl,
    /// Persistent memory: Intel Optane class, exposed as pmem.
    Pmem,
    /// The platform does not say.
    Unknown,
}

/// Explicit huge-page pools configured on a node.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HugePagePools {
    /// 2 MiB pages.
    pub pages_2mib: u64,
    /// 1 GiB pages.
    pub pages_1gib: u64,
}

/// One memory node.
#[derive(Debug, Clone)]
pub struct MemoryNode {
    /// Node index.
    pub id: NumaNodeId,
    /// Hardware threads local to this node.
    pub cpus: Vec<CpuId>,
    /// Physical memory backed by this node, when readable.
    pub total_bytes: Option<u64>,
    /// Technology class, for the memory fabric planner.
    pub tier: MemoryTier,
    /// Explicit huge-page pools configured on this node.
    pub huge_pages: HugePagePools,
}

impl MemoryNode {
    /// True when this node is ordinary DRAM, which is the only tier Kivi
    /// treats as the default resident tier.
    #[must_use]
    pub fn is_dram(&self) -> bool {
        matches!(self.tier, MemoryTier::Ddr)
    }
}

/// Category of a locality-attributed I/O device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceClass {
    /// Block device: `NVMe`, SATA, virtio.
    Block,
    /// RDMA-capable network adapter.
    RdmaNic,
    /// CXL memory region exposed to the kernel.
    CxlRegion,
    /// Intel Data Streaming Accelerator engine.
    DsaEngine,
}

/// An I/O device whose NUMA node the platform discloses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IoDevice {
    /// What kind of device this is.
    pub class: DeviceClass,
    /// Stable name, for example `nvme0` or `0000:01:00.0`.
    pub name: String,
    /// Node whose CPUs and DRAM are closest to the device.
    pub numa_node: Option<NumaNodeId>,
    /// Sysfs or device path the observation came from.
    pub path: String,
}

/// Which operating-system interface produced a topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TopologySource {
    /// `GetLogicalProcessorInformationEx` on Windows.
    WindowsTopologyApi,
    /// `/sys/devices/system` on Linux.
    LinuxSysfs,
    /// Nothing readable. One core, one node, one cache level.
    FallbackSingleCore,
}

impl fmt::Display for TopologySource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WindowsTopologyApi => write!(f, "windows-topology-api"),
            Self::LinuxSysfs => write!(f, "linux-sysfs"),
            Self::FallbackSingleCore => write!(f, "fallback-single-core"),
        }
    }
}

/// A whole-machine physical topology snapshot.
///
/// Built once at startup and then read only. Nothing here probes hardware, so
/// no hot path can be delayed by topology discovery.
#[derive(Debug, Clone)]
pub struct Topology {
    /// Operating-system interface the snapshot came from.
    pub source: TopologySource,
    /// Every logical processor, ascending by [`LogicalCpu::id`].
    pub logical_cpus: Vec<LogicalCpu>,
    /// Every physical core, ascending by [`PhysicalCore::key`].
    pub physical_cores: Vec<PhysicalCore>,
    /// Cache instances, ascending by `(level, size, shared_by)`.
    pub cache: Vec<CacheLevel>,
    /// Memory nodes, ascending by [`MemoryNode::id`].
    pub memory_nodes: Vec<MemoryNode>,
    /// I/O devices with disclosed locality, ascending by name.
    pub devices: Vec<IoDevice>,
    /// CPUs this process may run on, ascending.
    pub allowed_cpus: Vec<CpuId>,
    /// True when more than one [`CpuClass`] is present.
    pub heterogeneous: bool,
}

impl Topology {
    /// The smallest topology that is still a real machine: one core, one
    /// thread, one node, one 64 KiB unified L2, no devices.
    #[must_use]
    pub fn single_core() -> Self {
        let key = CoreKey {
            package: 0,
            die: 0,
            core: 0,
        };
        let cpu = CpuId(0);
        let node = NumaNodeId(0);
        let llc = CacheLevel {
            level: 2,
            kind: CacheKind::Unified,
            line_bytes: 64,
            size_bytes: 64 * 1024,
            shared_by: vec![key],
        };
        Self {
            source: TopologySource::FallbackSingleCore,
            logical_cpus: vec![LogicalCpu {
                id: cpu,
                core: key,
                smt_siblings: vec![cpu],
                numa_node: node,
                efficiency_class: None,
                class: CpuClass::Uniform,
            }],
            physical_cores: vec![PhysicalCore {
                key,
                logical_cpus: vec![cpu],
                numa_node: node,
                class: CpuClass::Uniform,
                last_level_cache: Some(llc.clone()),
                llc_shared: false,
            }],
            cache: vec![llc],
            memory_nodes: vec![MemoryNode {
                id: node,
                cpus: vec![cpu],
                total_bytes: None,
                tier: MemoryTier::Ddr,
                huge_pages: HugePagePools::default(),
            }],
            devices: Vec::new(),
            allowed_cpus: vec![cpu],
            heterogeneous: false,
        }
    }

    /// Builds a snapshot from raw backend observations and derives the
    /// relationships a backend does not report directly.
    #[must_use]
    pub fn from_parts(
        source: TopologySource,
        logical_cpus: Vec<LogicalCpu>,
        cache: Vec<CacheLevel>,
        memory_nodes: Vec<MemoryNode>,
        devices: Vec<IoDevice>,
        allowed: &CpuSet,
    ) -> Self {
        let mut logical_cpus = logical_cpus;
        logical_cpus.sort_by_key(|cpu| cpu.id);
        let class_spread: Vec<CpuClass> = {
            let mut classes: Vec<CpuClass> = logical_cpus.iter().map(|cpu| cpu.class).collect();
            classes.sort_unstable();
            classes.dedup();
            classes
        };
        let heterogeneous = class_spread.len() > 1;
        for cpu in &mut logical_cpus {
            if !heterogeneous {
                cpu.class = CpuClass::Uniform;
            }
            cpu.smt_siblings.sort_unstable();
        }

        let mut physical_cores: BTreeMap<CoreKey, PhysicalCore> = BTreeMap::new();
        for cpu in &logical_cpus {
            physical_cpus_insert(&mut physical_cores, cpu);
        }
        let last_level = cache.iter().map(|level| level.level).max().unwrap_or(0);
        let mut cores: Vec<PhysicalCore> = physical_cores.into_values().collect();
        for core in &mut cores {
            core.logical_cpus.sort_unstable();
            core.last_level_cache = cache
                .iter()
                .filter(|level| level.level == last_level && level.shared_by.contains(&core.key))
                .max_by_key(|level| (level.size_bytes, level.line_bytes))
                .cloned();
            core.llc_shared = core
                .last_level_cache
                .as_ref()
                .is_some_and(|llc| llc.shared_by.len() > 1);
        }

        let mut memory_nodes = memory_nodes;
        memory_nodes.sort_by_key(|node| node.id);
        for node in &mut memory_nodes {
            node.cpus.sort_unstable();
        }
        let mut cache = cache;
        cache.sort_by(|a, b| {
            a.level
                .cmp(&b.level)
                .then(a.size_bytes.cmp(&b.size_bytes))
                .then(a.shared_by.cmp(&b.shared_by))
        });
        let mut devices = devices;
        devices.sort_by(|a, b| a.name.cmp(&b.name));

        Self {
            source,
            logical_cpus,
            physical_cores: cores,
            cache,
            memory_nodes,
            devices,
            allowed_cpus: allowed.iter().collect(),
            heterogeneous,
        }
    }

    /// Drops every CPU outside `allowed`, and the cores, cache instances and
    /// memory nodes that end up with none.
    pub(crate) fn retain(&mut self, allowed: &CpuSet) {
        self.logical_cpus.retain(|cpu| allowed.contains(cpu.id));
        self.physical_cores
            .retain(|core| core.logical_cpus.iter().any(|id| allowed.contains(*id)));
        let live: Vec<CoreKey> = self.physical_cores.iter().map(|core| core.key).collect();
        for core in &mut self.physical_cores {
            core.logical_cpus.retain(|id| allowed.contains(*id));
        }
        for level in &mut self.cache {
            level.shared_by.retain(|key| live.contains(key));
        }
        self.cache.retain(|level| !level.shared_by.is_empty());
        for node in &mut self.memory_nodes {
            node.cpus.retain(|id| allowed.contains(*id));
        }
        self.memory_nodes.retain(|node| !node.cpus.is_empty());
        self.allowed_cpus = self.logical_cpus.iter().map(|cpu| cpu.id).collect();
    }

    /// Number of logical processors.
    #[must_use]
    pub fn logical_cpu_count(&self) -> usize {
        self.logical_cpus.len()
    }

    /// Number of physical cores.
    #[must_use]
    pub fn physical_core_count(&self) -> usize {
        self.physical_cores.len()
    }

    /// Number of memory nodes.
    #[must_use]
    pub fn numa_node_count(&self) -> usize {
        self.memory_nodes.len()
    }

    /// Number of sockets.
    #[must_use]
    pub fn package_count(&self) -> usize {
        self.physical_cores
            .iter()
            .map(|core| core.key.package)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    }

    /// The logical processor `ordinal` counting from zero, if it exists.
    #[must_use]
    pub fn cpu_of(&self, ordinal: usize) -> Option<CpuId> {
        self.logical_cpus.get(ordinal).map(|cpu| cpu.id)
    }

    /// The physical core `ordinal` counting from zero, if it exists.
    #[must_use]
    pub fn core_of(&self, ordinal: usize) -> Option<CoreKey> {
        self.physical_cores.get(ordinal).map(|core| core.key)
    }

    /// Looks up a logical processor.
    #[must_use]
    pub fn cpu(&self, id: CpuId) -> Option<&LogicalCpu> {
        self.logical_cpus
            .binary_search_by_key(&id, |cpu| cpu.id)
            .ok()
            .map(|index| &self.logical_cpus[index])
    }

    /// Looks up a physical core.
    #[must_use]
    pub fn core(&self, key: CoreKey) -> Option<&PhysicalCore> {
        self.physical_cores.iter().find(|core| core.key == key)
    }

    /// Looks up a memory node.
    #[must_use]
    pub fn memory_node(&self, id: NumaNodeId) -> Option<&MemoryNode> {
        self.memory_nodes.iter().find(|node| node.id == id)
    }

    /// Memory nodes holding ordinary DRAM.
    pub fn dram_nodes(&self) -> impl Iterator<Item = &MemoryNode> {
        self.memory_nodes.iter().filter(|node| node.is_dram())
    }

    /// Memory nodes backed by CXL memory.
    pub fn cxl_nodes(&self) -> impl Iterator<Item = &MemoryNode> {
        self.memory_nodes
            .iter()
            .filter(|node| node.tier == MemoryTier::Cxl)
    }

    /// Physical cores on a memory node, ascending.
    #[must_use]
    pub fn cores_on_node(&self, id: NumaNodeId) -> Vec<CoreKey> {
        self.physical_cores
            .iter()
            .filter(|core| core.numa_node == id)
            .map(|core| core.key)
            .collect()
    }

    /// The highest cache level present, or zero when none is known.
    #[must_use]
    pub fn last_level_cache(&self) -> u8 {
        self.cache
            .iter()
            .map(|level| level.level)
            .max()
            .unwrap_or(0)
    }

    /// Devices of one class with disclosed locality.
    pub fn devices_of(&self, class: DeviceClass) -> impl Iterator<Item = &IoDevice> {
        self.devices
            .iter()
            .filter(move |device| device.class == class)
    }

    /// Cores whose last-level cache is shared with at least one other core.
    /// A latency-sensitive owner placed on such a core contends for LLC with
    /// its neighbour regardless of which hardware thread it runs on.
    #[must_use]
    pub fn llc_coupled_cores(&self) -> Vec<CoreKey> {
        self.physical_cores
            .iter()
            .filter(|core| core.llc_shared)
            .map(|core| core.key)
            .collect()
    }

    /// Performance-class cores, falling back to every core when the machine
    /// is homogeneous or reports nothing.
    #[must_use]
    pub fn performance_cores(&self) -> Vec<CoreKey> {
        let cores: Vec<CoreKey> = self
            .physical_cores
            .iter()
            .filter(|core| core.class != CpuClass::Efficiency)
            .map(|core| core.key)
            .collect();
        if cores.is_empty() {
            self.physical_cores.iter().map(|core| core.key).collect()
        } else {
            cores
        }
    }

    /// Concise one-line description for logs and admin surfaces.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut roles: Vec<(DeviceClass, usize)> = Vec::new();
        for class in [
            DeviceClass::Block,
            DeviceClass::RdmaNic,
            DeviceClass::CxlRegion,
            DeviceClass::DsaEngine,
        ] {
            let count = self.devices_of(class).count();
            if count > 0 {
                roles.push((class, count));
            }
        }
        let devices = roles
            .iter()
            .map(|(class, count)| format!("{class:?}={count}"))
            .collect::<Vec<_>>()
            .join(" ");
        format!(
            "{source} logical={logical} cores={cores} packages={packages} nodes={nodes} \
             llc={llc} smt={smt} heterogeneous={heterogeneous}{devices}",
            source = self.source,
            logical = self.logical_cpu_count(),
            cores = self.physical_core_count(),
            packages = self.package_count(),
            nodes = self.numa_node_count(),
            llc = self.cache_summary(),
            smt = self.smt_enabled(),
            heterogeneous = self.heterogeneous,
            devices = if devices.is_empty() {
                String::new()
            } else {
                format!(" devices[{devices}]")
            },
        )
    }

    /// The cache hierarchy as distinct shapes, so a machine with one L1 per core
    /// reports `L1d:32K L1i:64K L2:3072K L3:36864K` rather than one line per
    /// core.
    #[must_use]
    pub fn cache_summary(&self) -> String {
        let mut shapes: Vec<(u8, CacheKind, u64, u32, usize)> = Vec::new();
        for level in &self.cache {
            let shape = (
                level.level,
                level.kind,
                level.size_bytes,
                level.line_bytes,
                0,
            );
            match shapes.iter_mut().find(|existing| {
                existing.0 == shape.0 && existing.1 == shape.1 && existing.2 == shape.2
            }) {
                Some(existing) => existing.4 += 1,
                None => shapes.push(shape),
            }
        }
        shapes.sort_unstable();
        if shapes.is_empty() {
            return "unknown".to_string();
        }
        shapes
            .into_iter()
            .map(|(level, kind, size, _, instances)| {
                let kind = match kind {
                    CacheKind::Unified => "",
                    CacheKind::Data => "d",
                    CacheKind::Instruction => "i",
                };
                let mib = size / (1024 * 1024);
                let size = if mib >= 1024 {
                    format!("{}G", mib / 1024)
                } else if mib > 0 {
                    format!("{mib}M")
                } else {
                    format!("{}K", size / 1024)
                };
                let count = if instances > 1 {
                    format!("x{instances}")
                } else {
                    String::new()
                };
                format!("L{level}{kind}:{size}{count}")
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// True when any core has more than one hardware thread.
    #[must_use]
    pub fn smt_enabled(&self) -> bool {
        self.logical_cpus.iter().any(LogicalCpu::shares_core)
    }
}

fn physical_cpus_insert(map: &mut BTreeMap<CoreKey, PhysicalCore>, cpu: &LogicalCpu) {
    map.entry(cpu.core)
        .and_modify(|core| {
            core.logical_cpus.push(cpu.id);
        })
        .or_insert_with(|| PhysicalCore {
            key: cpu.core,
            logical_cpus: vec![cpu.id],
            numa_node: cpu.numa_node,
            class: cpu.class,
            last_level_cache: None,
            llc_shared: false,
        });
}
