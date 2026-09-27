//! Memory topology, read once from `kivi-hardware`.
//!
//! This module used to parse `/sys/devices/system/node` on its own and
//! reported `0` for every processor's node because `node_of_cpu` discarded its
//! argument. Both problems are gone: the topology is now read by the same code
//! that placed the workers, so the fabric's notion of a node and the thread's
//! actual node cannot disagree.
//!
//! The types are re-exported rather than wrapped so no consumer can hold a
//! topology that came from anywhere else.

pub use kivi_hardware::topology::{MemoryNode, MemoryTier, NumaNodeId, Topology, TopologySource};

/// Reads the machine's memory topology.
#[must_use]
pub fn topology() -> &'static Topology {
    kivi_hardware::snapshot().topology_logical()
}

/// The number of memory nodes. One on a uniform machine, including a machine
/// whose topology could not be read.
#[must_use]
pub fn node_count() -> u32 {
    u32::try_from(topology().numa_node_count())
        .unwrap_or(1)
        .max(1)
}

/// The node a processor belongs to, or `None` when the platform did not say.
///
/// Unlike the previous stub this never claims `0` for an unknown processor: a
/// caller that receives `None` must treat locality as undisclosed, which is a
/// different decision from "node 0".
#[must_use]
pub fn node_of_cpu(cpu: kivi_hardware::topology::CpuId) -> Option<NumaNodeId> {
    topology().cpu(cpu).map(|cpu| cpu.numa_node)
}

/// Whether the machine has more than one memory node. Cheap enough for a
/// startup probe, never for a hot path.
#[must_use]
pub fn is_multi_node() -> bool {
    node_count() > 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_machine_always_has_at_least_one_node() {
        assert!(node_count() >= 1);
    }

    #[test]
    fn discovery_is_shared_with_placement() {
        // The fabric must read the same snapshot the placement used, or the two
        // can disagree about which node a thread is on.
        assert!(std::ptr::eq(
            topology(),
            kivi_hardware::snapshot().topology_logical()
        ));
    }

    #[test]
    fn unknown_processors_are_unknown_not_node_zero() {
        assert!(node_of_cpu(kivi_hardware::topology::CpuId(u32::MAX)).is_none());
    }
}
