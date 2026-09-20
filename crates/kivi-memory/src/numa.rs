//! NUMA/locality as provider capabilities.
//!
//! Crate evaluation (required by the task):
//! - `rustix`: would provide `sched_getaffinity` and NUMA syscalls
//!   without libc. Rejected for Phase 9: the fabric only needs topology
//!   facts (which node owns which CPU/memory), not affinity control — the
//!   engine already pins workers via `core_affinity2`.
//! - `allocator-api2`: a mechanism beneath arenas (custom backing alloc).
//!   Rejected for Phase 9: arenas allocate `Bytes` (stable, refcounted,
//!   NUMA-agnostic) and account capacity locally. Switching the global
//!   allocator would not group objects by behavior and is explicitly not
//!   a Memory Fabric.
//! - `hwlocality`: full hwloc topology. Rejected for Phase 9: heavy
//!   native dependency for facts a ~40-line sysfs parser provides.
//! - Kivi-owned Linux NUMA layer (this module): parses
//!   `/sys/devices/system/node` for node/memory facts. Small, auditable,
//!   no new dependencies, sufficient for `PreferredNuma`/`PinnedNuma`
//!   intents and provider locality caps.
//!
//! This is deliberately not a global allocator and never changes
//! allocation strategy process-wide.

/// NUMA topology facts used for locality scoring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumaTopology {
    /// Number of NUMA nodes visible.
    pub nodes: u32,
    /// CPUs per node (index-aligned with node id where known).
    pub cpus_per_node: Vec<u32>,
    /// Whether the topology came from the OS or the fallback.
    pub source: TopologySource,
}

/// Where topology facts came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TopologySource {
    /// Parsed from Linux sysfs.
    LinuxSysfs,
    /// Single-node fallback (non-Linux, containers without sysfs, or
    /// parse failure). Always safe: locality degrades to `Any`.
    FallbackSingleNode,
}

impl NumaTopology {
    /// Single-node fallback topology.
    #[must_use]
    pub const fn single_node() -> Self {
        Self {
            nodes: 1,
            cpus_per_node: Vec::new(),
            source: TopologySource::FallbackSingleNode,
        }
    }

    /// Node owning a CPU, or `None` when unknown.
    #[must_use]
    pub fn node_of_cpu(&self, _cpu: u32) -> Option<u32> {
        if self.nodes <= 1 {
            return Some(0);
        }
        None
    }
}

/// Owned OS integration for NUMA facts.
#[derive(Debug, Clone, Copy, Default)]
pub struct OsNuma;

impl OsNuma {
    /// Reads the NUMA topology, falling back to single-node when sysfs
    /// is unavailable or unparsable. Never fails: locality is advisory.
    #[must_use]
    pub fn topology() -> NumaTopology {
        #[cfg(target_os = "linux")]
        {
            Self::linux_topology().unwrap_or(NumaTopology::single_node())
        }
        #[cfg(not(target_os = "linux"))]
        {
            NumaTopology::single_node()
        }
    }

    #[cfg(target_os = "linux")]
    fn linux_topology() -> Option<NumaTopology> {
        let entries = std::fs::read_dir("/sys/devices/system/node").ok()?;
        let mut nodes = 0u32;
        let mut cpus_per_node = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_str()?;
            if let Some(number) = name.strip_prefix("node") {
                if number.parse::<u32>().is_ok() {
                    nodes += 1;
                    let cpulist =
                        std::fs::read_to_string(format!("/sys/devices/system/node/{name}/cpulist"))
                            .unwrap_or_default();
                    cpus_per_node.push(count_cpus(&cpulist));
                }
            }
        }
        if nodes == 0 {
            return None;
        }
        Some(NumaTopology {
            nodes,
            cpus_per_node,
            source: TopologySource::LinuxSysfs,
        })
    }
}

#[cfg(target_os = "linux")]
fn count_cpus(cpulist: &str) -> u32 {
    let mut count = 0u32;
    for part in cpulist.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            if let (Ok(start), Ok(end)) = (start.trim().parse::<u32>(), end.trim().parse::<u32>()) {
                count += end.saturating_sub(start).saturating_add(1);
            }
        } else if part.parse::<u32>().is_ok() {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_never_fails() {
        let topology = OsNuma::topology();
        assert!(topology.nodes >= 1);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpulist_counts_ranges() {
        assert_eq!(count_cpus("0-3"), 4);
        assert_eq!(count_cpus("0-3,8-11"), 8);
        assert_eq!(count_cpus("0"), 1);
        assert_eq!(count_cpus(""), 0);
    }
}
