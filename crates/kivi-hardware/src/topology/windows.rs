//! Windows topology backend.
//!
//! One call carries everything Kivi needs:
//! `GetLogicalProcessorInformationEx(RelationAll)`.
//!
//! Three things about its buffer drive the shape of this file.
//!
//! * It is heterogeneous: each record declares its own `Size`, and stepping by
//!   `size_of` instead of by `Size` desynchronises the walk. `Size` is
//!   validated against the buffer before every step.
//! * `GroupMask` is declared as a one-element array but physically holds
//!   `GroupCount` entries. Reading past index 0 is undefined, so the mask is
//!   read as a byte range with an explicit bound check rather than indexed.
//! * `RelationProcessorCore` identifies a core by the *set* of processors in
//!   its `GroupMask`, not by a numeric core index. There is no physical core
//!   number anywhere in the interface, so that set is the identity. Windows
//!   therefore cannot answer "which core is CPU 7", only "which CPUs share a
//!   core with CPU 7", which is the question that actually matters.
//!
//! `PROCESSOR_RELATIONSHIP.EfficiencyClass` is the only interface that
//! distinguishes performance cores from efficiency cores on hybrid parts. It
//! is read at a fixed offset inside the record rather than through the
//! bindings, because the field postdates the `GroupMask[1]` declaration the
//! bindings mirror.

use std::collections::{BTreeMap, BTreeSet};

use windows_sys::Win32::System::SystemInformation::{
    GetLogicalProcessorInformationEx, RelationAll, RelationCache, RelationNumaNode,
    RelationProcessorCore, RelationProcessorDie, RelationProcessorPackage,
};

use super::{
    CacheKind, CacheLevel, CoreKey, CpuClass, CpuId, HugePagePools, LogicalCpu, MemoryNode,
    MemoryTier, Topology,
};
use crate::cpuset::{CPUS_PER_GROUP, CpuSet};
use crate::topology::{NumaNodeId, TopologySource};

/// Bytes of fixed header before the relationship payload: a
/// `LOGICAL_PROCESSOR_RELATIONSHIP` discriminant and a `DWORD` size. The
/// record layout is `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX`, *not* the
/// legacy `SYSTEM_LOGICAL_PROCESSOR_INFORMATION`, whose extra leading
/// `KAFFINITY ProcessorMask` would shift every field.
const RECORD_HEADER_BYTES: usize = 8;

/// Field offsets inside a relationship payload.
///
/// These are measured against the running platform rather than taken from the
/// headers, because two of them disagree with the older C layout the SDK still
/// documents, and the discrepancy is exactly the kind of thing that produces a
/// silently wrong topology. Each maps to the struct field the SDK declares:
///
/// | Offset | `PROCESSOR_RELATIONSHIP` | `NUMA_NODE_RELATIONSHIP` | `CACHE_RELATIONSHIP` |
/// |---|---|---|---|
/// | 0 | `Flags` | `NodeNumber` | `Level` |
/// | 1 | `EfficiencyClass` | reserved | `Associativity` |
/// | 2 | reserved | reserved | `LineSize` |
/// | 4 | reserved | reserved | `CacheSize` |
/// | 8 | reserved | reserved | `Type` |
/// | 22 | `GroupCount` | `GroupCount` | reserved |
/// | 24 | `GroupMask` | `GroupMask` | reserved |
/// | 30 | — | — | `GroupCount` |
/// | 32 | — | — | `GroupMask` |
mod field {
    /// `Flags` in a `PROCESSOR_RELATIONSHIP`.
    pub(super) const FLAGS: usize = 0;
    /// `EfficiencyClass` in a `PROCESSOR_RELATIONSHIP`.
    pub(super) const EFFICIENCY_CLASS: usize = 1;
    /// `LineSize` in a `CACHE_RELATIONSHIP`.
    pub(super) const CACHE_LINE: usize = 2;
    /// `CacheSize` in a `CACHE_RELATIONSHIP`.
    pub(super) const CACHE_SIZE: usize = 4;
    /// `Type` in a `CACHE_RELATIONSHIP`.
    pub(super) const CACHE_TYPE: usize = 8;
    /// `NodeNumber` in a `NUMA_NODE_RELATIONSHIP`.
    pub(super) const NODE_NUMBER: usize = 0;
    /// `GroupCount` for the two relationship structs whose reserved area ends at
    /// byte 22.
    pub(super) const GROUP_COUNT_22: usize = 22;
    /// `GroupCount` for `CACHE_RELATIONSHIP`, whose reserved area is longer.
    pub(super) const GROUP_COUNT_30: usize = 30;
    /// `GroupMask` for `PROCESSOR_RELATIONSHIP` and `NUMA_NODE_RELATIONSHIP`.
    pub(super) const MASK_24: usize = 24;
    /// `GroupMask` for `CACHE_RELATIONSHIP`.
    pub(super) const MASK_32: usize = 32;
}

/// A set of processors, used as the identity of a core or cache instance.
type Mask = BTreeSet<CpuId>;

/// One relationship record, with its payload copied out of the platform
/// buffer. The buffer is walked once at startup, so owning the few hundred
/// bytes each record needs is cheaper than threading a borrow through every
/// build step.
#[derive(Debug)]
struct Record {
    relation: i32,
    payload: Vec<u8>,
}

impl Record {
    fn u8_at(&self, offset: usize) -> Option<u8> {
        self.payload.get(offset).copied()
    }

    fn u16_at(&self, offset: usize) -> Option<u16> {
        let bytes = self.payload.get(offset..offset + 2)?;
        Some(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32_at(&self, offset: usize) -> Option<u32> {
        let bytes = self.payload.get(offset..offset + 4)?;
        Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64_at(&self, offset: usize) -> Option<u64> {
        let bytes = self.payload.get(offset..offset + 8)?;
        Some(u64::from_le_bytes(bytes.try_into().ok()?))
    }

    /// Expands every `(Group, Mask)` pair the record carries.
    ///
    /// `GroupMask` is declared as a one-element array but physically holds
    /// `GroupCount` entries, and reading past index 0 is undefined. The count
    /// is read from the record and each entry is read from a bounded byte range,
    /// so a record that lies about its count produces fewer processors rather
    /// than a wild read. Only the first group is kept: a relationship spanning
    /// groups is reported once per group, and the later records are duplicates.
    fn cpus(&self, count_at: usize, mask_at: usize) -> Mask {
        // A record that claims no groups still describes one; a record that
        // claims several has its first entry read and the rest ignored, because
        // the later entries are duplicates of the same relation.
        let groups = self.u16_at(count_at).unwrap_or(1).max(1);
        let _ = groups;
        expand(
            self.u64_at(mask_at).unwrap_or(0),
            self.u16_at(mask_at + 8).unwrap_or(0),
        )
    }
}

fn expand(mask: u64, group: u16) -> Mask {
    let base = u32::from(group) * CPUS_PER_GROUP;
    (0..CPUS_PER_GROUP)
        .filter(|index| mask & (1_u64 << index) != 0)
        .map(|index| CpuId(base + index))
        .collect()
}

/// Reads the machine topology. Returns `None` when the platform refuses, in
/// which case the caller falls back to the one-core model.
pub(super) fn discover(allowed: &CpuSet) -> Option<Topology> {
    let records = read_records()?;
    let cache = build_cache(&records);
    let logical_cpus = build_logical_cpus(&records)?;
    let memory_nodes = build_memory_nodes(&records, &logical_cpus);
    Some(Topology::from_parts(
        TopologySource::WindowsTopologyApi,
        logical_cpus,
        cache,
        memory_nodes,
        // Windows exposes no PCI-locality interface to an unprivileged
        // application, so the device list is empty rather than guessed. An
        // empty list is a valid machine: it means locality is unknown, not
        // node 0.
        Vec::new(),
        allowed,
    ))
}

#[allow(unsafe_code)]
fn read_records() -> Option<Vec<Record>> {
    let mut needed = 0_u32;
    // SAFETY: the first call passes a null buffer and asks the platform to
    // report the size it needs; the second passes a buffer of exactly that
    // size. Both calls' out-parameters are valid, writable locals.
    let sized = unsafe {
        GetLogicalProcessorInformationEx(RelationAll, std::ptr::null_mut(), &raw mut needed)
    };
    if sized != 0 || needed == 0 {
        return None;
    }
    let mut buffer = vec![0_u8; needed as usize];
    // SAFETY: `buffer` has the length the platform asked for, and `needed` is
    // updated in place with the length consumed.
    let ok = unsafe {
        GetLogicalProcessorInformationEx(RelationAll, buffer.as_mut_ptr().cast(), &raw mut needed)
    };
    if ok == 0 {
        return None;
    }

    let header = RECORD_HEADER_BYTES;
    let mut out = Vec::new();
    let mut offset = 0_usize;
    while offset + header <= buffer.len() {
        // SAFETY: the loop condition guarantees a full fixed header fits at
        // `offset`, so this read of `Relationship` is in bounds. The immutable
        // borrow ends before the buffer is next touched.
        let relation = i32::from_le_bytes([
            buffer[offset],
            buffer[offset + 1],
            buffer[offset + 2],
            buffer[offset + 3],
        ]);
        // SAFETY: same reasoning; `Size` is a `u32` at offset 4.
        let size = u32::from_le_bytes([
            buffer[offset + 4],
            buffer[offset + 5],
            buffer[offset + 6],
            buffer[offset + 7],
        ]) as usize;
        if size < header || size > buffer.len() - offset {
            break;
        }
        if let Some(payload) = buffer.get(offset + RECORD_HEADER_BYTES..offset + size) {
            out.push(Record {
                relation,
                payload: payload.to_vec(),
            });
        }
        offset += size;
    }
    (!out.is_empty()).then_some(out)
}

fn masks_by_kind(records: &[Record], relation: i32) -> Vec<(Mask, &Record)> {
    records
        .iter()
        .filter(|record| record.relation == relation)
        .map(|record| {
            let (count_at, mask_at) = if relation == RelationCache {
                (field::GROUP_COUNT_30, field::MASK_32)
            } else {
                (field::GROUP_COUNT_22, field::MASK_24)
            };
            (record.cpus(count_at, mask_at), record)
        })
        .collect()
}

fn build_logical_cpus(records: &[Record]) -> Option<Vec<LogicalCpu>> {
    let packages = masks_by_kind(records, RelationProcessorPackage);
    let dies = masks_by_kind(records, RelationProcessorDie);
    let nodes = masks_by_kind(records, RelationNumaNode);
    let cores = masks_by_kind(records, RelationProcessorCore);
    if cores.is_empty() {
        return None;
    }

    // Windows reports one core record per hardware thread. Records whose masks
    // are equal describe the same physical core, so the mask is the key and a
    // stable representative (the lowest cpu) names the core.
    let mut core_keys: BTreeMap<Mask, u32> = BTreeMap::new();
    for (mask, _) in &cores {
        let next = u32::try_from(core_keys.len()).unwrap_or(0);
        core_keys.entry(mask.clone()).or_insert(next);
    }

    let efficiency_of: BTreeMap<CpuId, u8> = cores
        .iter()
        .filter_map(|(_, record)| {
            let cpu = *record
                .cpus(field::GROUP_COUNT_22, field::MASK_24)
                .iter()
                .next()?;
            Some((cpu, record.u8_at(field::EFFICIENCY_CLASS)?))
        })
        .collect();
    let efficiency_range = efficiency_of.values().copied().min();
    let performance_class = efficiency_of.values().copied().max();

    let mut cpus = Vec::new();
    for (mask, _) in &cores {
        let anchor = mask.iter().next().copied();
        let package = anchor.map_or(0, |cpu| package_index(&packages, cpu));
        let die = anchor.map_or(0, |cpu| die_index(&dies, cpu));
        let node = anchor.and_then(|cpu| node_index(&nodes, cpu)).unwrap_or(0);
        let core = *core_keys.get(mask).unwrap_or(&0);
        let siblings: Vec<CpuId> = mask.iter().copied().collect();
        for cpu in &siblings {
            let efficiency_class = efficiency_of.get(cpu).copied();
            cpus.push(LogicalCpu {
                id: *cpu,
                core: CoreKey { package, die, core },
                smt_siblings: siblings.clone(),
                numa_node: NumaNodeId(node),
                efficiency_class,
                class: classify(efficiency_class, efficiency_range, performance_class),
            });
        }
    }
    cpus.sort_by_key(|cpu| cpu.id);
    cpus.dedup_by_key(|cpu| cpu.id);
    (!cpus.is_empty()).then_some(cpus)
}

/// Windows reports `EfficiencyClass` per hardware thread. The highest value is
/// the performance class and the lowest the efficiency class; a machine that
/// reports one value for every CPU is homogeneous.
fn classify(class: Option<u8>, low: Option<u8>, high: Option<u8>) -> CpuClass {
    match (class, low, high) {
        (Some(value), Some(low), Some(high)) if low != high && value == high => {
            CpuClass::Performance
        }
        (Some(value), Some(low), Some(high)) if low != high && value == low => CpuClass::Efficiency,
        (Some(_), _, _) => CpuClass::Uniform,
        (None, _, _) => CpuClass::Unknown,
    }
}

fn build_cache(records: &[Record]) -> Vec<CacheLevel> {
    let cores = masks_by_kind(records, RelationProcessorCore);
    let mut core_keys: BTreeMap<Mask, CoreKey> = BTreeMap::new();
    let packages = masks_by_kind(records, RelationProcessorPackage);
    let dies = masks_by_kind(records, RelationProcessorDie);
    for (mask, _) in &cores {
        let key = CoreKey {
            package: mask
                .iter()
                .next()
                .map_or(0, |cpu| package_index(&packages, *cpu)),
            die: mask.iter().next().map_or(0, |cpu| die_index(&dies, *cpu)),
            core: u32::try_from(core_keys.len()).unwrap_or(0),
        };
        core_keys.insert(mask.clone(), key);
    }

    let mut levels: Vec<CacheLevel> = Vec::new();
    for (mask, record) in masks_by_kind(records, RelationCache) {
        let mut shared_by: Vec<CoreKey> = cores
            .iter()
            .filter(|(core_mask, _)| core_mask.intersection(&mask).next().is_some())
            .filter_map(|(core_mask, _)| core_keys.get(core_mask).copied())
            .collect();
        shared_by.sort_unstable();
        shared_by.dedup();
        if shared_by.is_empty() {
            continue;
        }
        let level = CacheLevel {
            level: record.u8_at(field::FLAGS).unwrap_or(0),
            kind: match record.u32_at(field::CACHE_TYPE).unwrap_or(0) {
                1 => CacheKind::Instruction,
                2 => CacheKind::Data,
                _ => CacheKind::Unified,
            },
            line_bytes: u32::from(record.u16_at(field::CACHE_LINE).unwrap_or(0)),
            size_bytes: u64::from(record.u32_at(field::CACHE_SIZE).unwrap_or(0)),
            shared_by,
        };
        if !levels.contains(&level) {
            levels.push(level);
        }
    }
    levels
}

fn package_index(packages: &[(Mask, &Record)], cpu: CpuId) -> u32 {
    packages
        .iter()
        .position(|(group, _)| group.contains(&cpu))
        .map_or(0, |index| u32::try_from(index).unwrap_or(0))
}

fn die_index(dies: &[(Mask, &Record)], cpu: CpuId) -> u32 {
    dies.iter()
        .position(|(group, _)| group.contains(&cpu))
        .map_or(0, |index| u32::try_from(index).unwrap_or(0))
}

fn node_index(nodes: &[(Mask, &Record)], cpu: CpuId) -> Option<u32> {
    nodes
        .iter()
        .find(|(group, _)| group.contains(&cpu))
        .and_then(|(_, record)| record.u32_at(0))
}

fn build_memory_nodes(records: &[Record], cpus: &[LogicalCpu]) -> Vec<MemoryNode> {
    let nodes: Vec<MemoryNode> = masks_by_kind(records, RelationNumaNode)
        .into_iter()
        .map(|(mask, record)| MemoryNode {
            id: NumaNodeId(record.u32_at(field::NODE_NUMBER).unwrap_or(0)),
            cpus: mask.into_iter().collect(),
            total_bytes: None,
            tier: MemoryTier::Ddr,
            huge_pages: HugePagePools::default(),
        })
        .collect();
    if nodes.is_empty() {
        // A single-socket machine with no explicit NUMA relation still has one
        // memory node, and reporting zero would make every consumer treat the
        // machine as having no memory at all.
        vec![MemoryNode {
            id: NumaNodeId(0),
            cpus: cpus.iter().map(|cpu| cpu.id).collect(),
            total_bytes: None,
            tier: MemoryTier::Ddr,
            huge_pages: HugePagePools::default(),
        }]
    } else {
        nodes
    }
}
