//! Linux topology backend.
//!
//! Everything comes from sysfs, which needs no library, no `libc` topology
//! binding and no privilege:
//!
//! * `/sys/devices/system/cpu/cpuN/topology/{core_id,physical_package_id,die_id}`
//!   identifies the physical core of each logical processor.
//! * `/sys/devices/system/node/nodeN/{cpulist,meminfo,hugepages}` gives memory
//!   nodes, their processors, their capacity and their explicit huge-page
//!   pools. A node whose `nodeN/hugepages` or the matching CXL entries exist
//!   is a CXL or pmem tier rather than DRAM.
//! * `/sys/devices/system/cpu/cpuN/cache/indexM/` gives cache geometry and
//!   `shared_cpu_list` gives the processors sharing each instance.
//! * `/sys/bus/pci/devices/*/numa_node` and `/sys/class/{nvme,infiniband,dax}`
//!   give device locality.
//!
//! Anything unreadable is simply absent from the model. A container that
//! mounts a partial sysfs still gets a valid topology over whatever is
//! visible.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use super::{
    CacheKind, CacheLevel, CoreKey, CpuClass, CpuId, HugePagePools, LogicalCpu, MemoryNode,
    MemoryTier, Topology,
};
use crate::cpuset::{CpuSet, parse_cpu_list};
use crate::topology::{NumaNodeId, TopologySource};

const SYSFS: &str = "/sys/devices/system";

pub(super) fn discover(allowed: &CpuSet) -> Option<Topology> {
    let logical_cpus = read_logical_cpus()?;
    let cache = read_cache(&logical_cpus);
    Some(Topology::from_parts(
        TopologySource::LinuxSysfs,
        logical_cpus,
        cache,
        read_memory_nodes(),
        crate::device::discover(),
        allowed,
    ))
}

/// The set of directories matching a `cpuN` pattern.
fn cpu_dirs() -> Option<Vec<u32>> {
    let mut cpus: Vec<u32> = fs::read_dir(format!("{SYSFS}/cpu"))
        .ok()?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            name.strip_prefix("cpu")?.parse::<u32>().ok()
        })
        .collect();
    if cpus.is_empty() {
        return None;
    }
    cpus.sort_unstable();
    Some(cpus)
}

fn read_topology_u32(cpu: u32, field: &str) -> Option<u32> {
    let raw = fs::read_to_string(format!("{SYSFS}/cpu/cpu{cpu}/topology/{field}")).ok()?;
    // A single-core package reports package id -1; treat that as package 0.
    // `try_from` rather than `as` because a value above `u32::MAX` is not a
    // topology the rest of the model can represent, and silently truncating it
    // would merge two distinct cores.
    u32::try_from(raw.trim().parse::<i64>().ok()?.max(0)).ok()
}

fn read_logical_cpus() -> Option<Vec<LogicalCpu>> {
    let cpus = cpu_dirs()?;
    let parsed: Vec<(CpuId, CoreKey)> = cpus
        .iter()
        .filter_map(|cpu| {
            Some((
                CpuId(*cpu),
                CoreKey {
                    package: read_topology_u32(*cpu, "physical_package_id")?,
                    die: read_topology_u32(*cpu, "die_id").unwrap_or(0),
                    core: read_topology_u32(*cpu, "core_id")?,
                },
            ))
        })
        .collect();
    if parsed.is_empty() {
        return None;
    }
    // Linux has no efficiency-class interface in sysfs, so no CPU is classified
    // and the machine is reported homogeneous. A hetero part running Linux is
    // treated as uniform cores, which is the conservative choice: pinning a
    // latency-sensitive owner to an efficiency core is a miss, but mislabelling
    // a performance core as an efficiency core would push those owners off the
    // fast cores entirely.
    let nodes = node_of_cpu_map();
    let siblings: BTreeMap<CoreKey, Vec<CpuId>> = parsed.iter().fold(
        BTreeMap::new(),
        |mut map: BTreeMap<CoreKey, Vec<CpuId>>, (cpu, key)| {
            map.entry(*key).or_default().push(*cpu);
            map
        },
    );
    Some(
        parsed
            .into_iter()
            .map(|(id, key)| LogicalCpu {
                id,
                core: key,
                smt_siblings: siblings.get(&key).cloned().unwrap_or_default(),
                // A processor the kernel did not place on a node is on node 0
                // in every kernel that omits the list, and claiming a different
                // node would be a guess. Node 0 is the documented single-node
                // answer, not a claim about a machine that has more.
                numa_node: nodes.get(&id).copied().unwrap_or(NumaNodeId(0)),
                efficiency_class: None,
                class: CpuClass::Unknown,
            })
            .collect(),
    )
}

fn node_of_cpu_map() -> BTreeMap<CpuId, NumaNodeId> {
    let mut map = BTreeMap::new();
    let Ok(entries) = fs::read_dir(format!("{SYSFS}/node")) else {
        return map;
    };
    for entry in entries.filter_map(Result::ok) {
        let name = entry.file_name().into_string().unwrap_or_default();
        let Some(id) = name
            .strip_prefix("node")
            .and_then(|rest| rest.parse::<u32>().ok())
        else {
            continue;
        };
        let path = format!("{SYSFS}/node/{name}/cpulist");
        let Ok(raw) = fs::read_to_string(path) else {
            continue;
        };
        for cpu in parse_cpu_list(&raw) {
            map.insert(CpuId(cpu), NumaNodeId(id));
        }
    }
    map
}

fn read_cache(cpus: &[LogicalCpu]) -> Vec<CacheLevel> {
    // Cache geometry is per physical core and identical across its hardware
    // threads, so one probe per core is enough.
    let mut seen: BTreeSet<CoreKey> = BTreeSet::new();
    let mut levels: Vec<CacheLevel> = Vec::new();
    for cpu in cpus {
        if !seen.insert(cpu.core) {
            continue;
        }
        let base = format!("{SYSFS}/cpu/cpu{}/cache", cpu.id.0);
        let Ok(entries) = fs::read_dir(&base) else {
            continue;
        };
        for entry in entries.filter_map(Result::ok) {
            // Only `indexN` directories describe a cache level; the kernel
            // also puts `shared_cpu_map` and a `uevent` beside them.
            let is_index = entry
                .file_name()
                .into_string()
                .is_ok_and(|name| name.starts_with("index"));
            if !is_index {
                continue;
            }
            if let Some(level) = read_cache_instance(&entry.path(), cpus)
                && !levels.contains(&level)
            {
                levels.push(level);
            }
        }
    }
    levels
}

fn read_cache_instance(dir: &Path, cpus: &[LogicalCpu]) -> Option<CacheLevel> {
    let level = u8::try_from(read_u64(dir.join("level"))?).ok()?;
    let kind = match read_u64(dir.join("type")).unwrap_or(0) {
        1 => CacheKind::Instruction,
        2 => CacheKind::Data,
        _ => CacheKind::Unified,
    };
    let line_bytes = u32::try_from(read_u64(dir.join("coherency_line_size"))?).ok()?;
    let size_bytes = read_u64(dir.join("size"))?;
    let shared: BTreeSet<CpuId> = fs::read_to_string(dir.join("shared_cpu_list"))
        .ok()
        .map(|raw| {
            parse_cpu_list(&raw)
                .into_iter()
                .map(CpuId)
                .filter(|cpu| cpus.iter().any(|entry| entry.id == *cpu))
                .collect()
        })
        .unwrap_or_default();
    if shared.is_empty() {
        return None;
    }
    let mut shared_by: Vec<CoreKey> = cpus
        .iter()
        .filter(|cpu| shared.contains(&cpu.id))
        .map(|cpu| cpu.core)
        .collect();
    shared_by.sort_unstable();
    shared_by.dedup();
    Some(CacheLevel {
        level,
        kind,
        line_bytes,
        size_bytes,
        shared_by,
    })
}

fn read_memory_nodes() -> Vec<MemoryNode> {
    let Ok(entries) = fs::read_dir(format!("{SYSFS}/node")) else {
        return vec![MemoryNode {
            id: NumaNodeId(0),
            cpus: Vec::new(),
            total_bytes: None,
            tier: MemoryTier::Unknown,
            huge_pages: HugePagePools::default(),
        }];
    };
    let mut nodes: Vec<MemoryNode> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let id = NumaNodeId(name.strip_prefix("node")?.parse::<u32>().ok()?);
            let dir = entry.path();
            let cpus = fs::read_to_string(dir.join("cpulist"))
                .map(|raw| parse_cpu_list(&raw).into_iter().map(CpuId).collect())
                .unwrap_or_default();
            Some(MemoryNode {
                id,
                cpus,
                total_bytes: meminfo_total_bytes(&dir.join("meminfo")),
                tier: node_tier(&dir),
                huge_pages: HugePagePools {
                    pages_2mib: huge_page_count(&dir.join("hugepages/hugepages-2048kB")),
                    pages_1gib: huge_page_count(&dir.join("hugepages/hugepages-1048576kB")),
                },
            })
        })
        .collect();
    nodes.sort_by_key(|node| node.id);
    if nodes.is_empty() {
        nodes.push(MemoryNode {
            id: NumaNodeId(0),
            cpus: Vec::new(),
            total_bytes: None,
            tier: MemoryTier::Unknown,
            huge_pages: HugePagePools::default(),
        });
    }
    nodes
}

/// `MemTotal` in `nodeN/meminfo` is in kibibytes.
///
/// The line is prefixed with the node id - `Node 0 MemTotal:  32485044 kB` - so
/// matching on the bare field name is what makes every node report zero
/// capacity and the memory fabric conclude the machine has no memory.
fn meminfo_total_bytes(path: &Path) -> Option<u64> {
    let raw = fs::read_to_string(path).ok()?;
    for line in raw.lines() {
        let Some((_, rest)) = line.rsplit_once("MemTotal:") else {
            continue;
        };
        let kib: u64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
        return Some(kib * 1024);
    }
    None
}

/// A node is CXL or pmem when the kernel says so.
///
/// `/sys/devices/system/node/nodeN/` gains a `cxl` directory when CXL memory is
/// attached to the node, and pmem nodes carry the `pmem` flag in
/// `/sys/bus/nd/devices/`. The directory is the cheaper and more reliable
/// signal, so it wins when present.
fn node_tier(dir: &Path) -> MemoryTier {
    if dir.join("cxl").is_dir() {
        return MemoryTier::Cxl;
    }
    if dir.join("pmem").is_dir() {
        return MemoryTier::Pmem;
    }
    MemoryTier::Ddr
}

fn huge_page_count(path: &Path) -> u64 {
    fs::read_to_string(path.join("nr_hugepages"))
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

fn read_u64(path: impl AsRef<Path>) -> Option<u64> {
    let raw = fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    // The kernel prints cache sizes with a `K` suffix.
    let (digits, multiplier) = match trimmed.strip_suffix('K') {
        Some(rest) => (rest.trim(), 1024_u64),
        None => (trimmed, 1),
    };
    digits.parse::<u64>().ok().map(|value| value * multiplier)
}

#[cfg(test)]
mod tests {
    use crate::cpuset::parse_cpu_list;

    #[test]
    fn a_cpu_list_parsing_covers_runs_singletons_and_junk() {
        assert_eq!(parse_cpu_list("0-3,8,10-11"), vec![0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(parse_cpu_list(" 5 , 5 , 7 "), vec![5, 7]);
        assert!(parse_cpu_list("").is_empty());
        assert!(parse_cpu_list("garbage").is_empty());
    }
}
