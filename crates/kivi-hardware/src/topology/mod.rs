//! Project-owned model of the physical machine.
//!
//! Kivi never exposes an operating-system topology type to its own crates.
//! Everything above this module speaks in the identifiers defined here:
//! [`CoreKey`], [`NumaNodeId`], [`LogicalCpuId`]. A backend may be replaced
//! without touching placement, capability, or provider code.
//!
//! Discovery never fails and never blocks. A machine whose topology cannot be
//! read yields [`TopologySource::FallbackSingleCore`], a one-core one-node
//! model. Every consumer must treat that model as a valid machine.
//!
//! Backends:
//!
//! * Windows: `GetLogicalProcessorInformationEx` for core, package, die and
//!   NUMA relations, plus `GetSystemCpuSetInformation` for efficiency class,
//!   which is how performance and efficiency cores are told apart on hybrid
//!   parts.
//! * Linux: `/sys/devices/system/cpu` for cores and siblings,
//!   `/sys/devices/system/node` for memory nodes and tiers, and
//!   `/sys/bus/pci/devices` for device locality.

mod model;

#[cfg(any(target_os = "linux", target_os = "android"))]
mod linux;
#[cfg(windows)]
mod windows;

pub use model::{
    CacheKind, CacheLevel, CoreKey, CpuClass, CpuId, DeviceClass, HugePagePools, IoDevice,
    LogicalCpu, MemoryNode, MemoryTier, NumaNodeId, PackageId, PhysicalCore, Topology,
    TopologySource,
};

use crate::cpuset::CpuSet;

impl Topology {
    /// Reads the machine topology, restricted to the CPUs this process may
    /// actually run on. Never fails: an unreadable topology degrades to
    /// [`TopologySource::FallbackSingleCore`].
    #[must_use]
    pub fn discover() -> Self {
        Self::discover_within(&CpuSet::current().unwrap_or_default())
    }

    /// Reads the machine topology but keeps only the given CPUs.
    ///
    /// Containers and cgroup CPU limits produce a topology wider than the
    /// process may use. Excluding those CPUs is not optional: offering a
    /// worker a core it cannot be scheduled on produces silent placement
    /// failure at the first bound.
    #[must_use]
    pub fn discover_within(allowed: &CpuSet) -> Self {
        #[cfg(windows)]
        {
            if let Some(mut topology) = windows::discover(allowed) {
                topology.retain(allowed);
                return topology;
            }
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            if let Some(mut topology) = linux::discover(allowed) {
                topology.retain(allowed);
                return topology;
            }
        }
        Self::single_core()
    }
}
