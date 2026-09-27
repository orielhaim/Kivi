//! Topology-aware worker placement.
//!
//! Kivi already owns its threads: a data worker, a consensus worker, a
//! durability lane, an off-core lane, a redundancy repair lane, a checkpoint
//! lane, a network reactor. Each is a long-lived owner with a stable job, so
//! each is a placement decision made once at startup, not a scheduling policy
//! re-evaluated per request.
//!
//! Four strategies are implemented so that enabling one is a measurement
//! rather than an assumption:
//!
//! | Strategy | What it does |
//! |---|---|
//! | [`PlacementStrategy::Unbound`] | Nothing. The OS scheduler decides. |
//! | [`PlacementStrategy::LogicalIndex`] | Thread `i` may run on logical CPU `i`. What a naive implementation does. |
//! | [`PlacementStrategy::PhysicalCore`] | Thread `i` owns physical core `i`, addressed by one hardware thread. |
//! | [`PlacementStrategy::NumaAware`] | Physical cores, spread across memory nodes, latency-critical roles on the lowest-latency cores. |
//!
//! Every strategy returns a [`PlacementPlan`] that says what it did and what
//! it could not do, so an operator never has to guess whether pinning took
//! effect. No strategy is ever allowed to prevent a worker from starting: a
//! machine with fewer cores than owners wraps, and a machine whose topology
//! could not be read falls back to unbound.

use std::collections::BTreeMap;
use std::fmt;

use crate::cpuset::CpuSet;
use crate::topology::{CoreKey, CpuClass, CpuId, Topology};

/// What a thread does. Latency sensitivity and memory intensity drive
/// placement, and a role is the only place that knowledge belongs: the
/// placement algorithm never learns anything about tablets, WALs, or fabrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum PlacementRole {
    /// Front-end data worker. Owns tablets and runs the Compio reactor.
    DataWorker,
    /// Consensus replica or leader loop. Latency critical.
    ConsensusWorker,
    /// The single write-ahead-log barrier lane. Every durable write waits for
    /// it, so it is the most latency critical thread in the process.
    DurabilityLane,
    /// Off-core materialization I/O lane. Bandwidth heavy, latency tolerant.
    OffcoreLane,
    /// Redundancy repair and scrub worker. Bandwidth heavy, background.
    RepairWorker,
    /// Checkpoint build and publish. Bandwidth heavy, background.
    CheckpointWorker,
    /// Redundancy fragment transfer client. Network bound.
    NetworkReactor,
}

impl PlacementRole {
    /// Every role, for callers that enumerate Kivi's owner threads.
    pub const ALL: &'static [Self] = &[
        Self::DataWorker,
        Self::ConsensusWorker,
        Self::DurabilityLane,
        Self::OffcoreLane,
        Self::RepairWorker,
        Self::CheckpointWorker,
        Self::NetworkReactor,
    ];

    /// How much the role's tail latency is felt by a client.
    ///
    /// The durability lane is ranked above the data workers: a data worker
    /// answers a request, but every data worker in the process waits on the
    /// durability lane's barrier.
    #[must_use]
    pub fn latency_sensitivity(self) -> LatencySensitivity {
        match self {
            Self::DurabilityLane | Self::ConsensusWorker => LatencySensitivity::Critical,
            Self::DataWorker | Self::NetworkReactor => LatencySensitivity::Sensitive,
            Self::OffcoreLane | Self::RepairWorker | Self::CheckpointWorker => {
                LatencySensitivity::Background
            }
        }
    }

    /// Whether the role's working set is large enough that cross-node memory
    /// traffic dominates its cache traffic.
    #[must_use]
    pub fn memory_intensity(self) -> MemoryIntensity {
        match self {
            Self::OffcoreLane | Self::RepairWorker | Self::CheckpointWorker => {
                MemoryIntensity::Streaming
            }
            Self::DataWorker | Self::ConsensusWorker => MemoryIntensity::WorkingSet,
            Self::DurabilityLane | Self::NetworkReactor => MemoryIntensity::Streaming,
        }
    }
}

/// Tail-latency exposure of a role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LatencySensitivity {
    /// A client's request is waiting on this thread right now.
    Critical,
    /// One worker's stalls show up as that worker's latency.
    Sensitive,
    /// Nobody is waiting; the work is throughput bound.
    Background,
}

impl LatencySensitivity {
    /// Placement order, lowest first. A separate rank rather than the derived
    /// `Ord` because the declaration order reads best in the other direction:
    /// "critical, then sensitive, then background" is the *severity* order, not
    /// the placement order, and conflating the two produced a plan that handed
    /// the best cores to background work.
    const fn rank(self) -> u8 {
        match self {
            Self::Critical => 0,
            Self::Sensitive => 1,
            Self::Background => 2,
        }
    }
}

/// Whether a role's memory traffic is a streaming copy or a resident
/// working set. Streaming traffic follows the worker to any node; a resident
/// working set is cheapest where the data already is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryIntensity {
    /// A bounded, repeatedly touched set.
    WorkingSet,
    /// A pass over large buffers.
    Streaming,
}

/// How workers are mapped onto cores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PlacementStrategy {
    /// No pinning. Correct everywhere, and the only strategy that is correct
    /// when the topology is unknown.
    #[default]
    Unbound,
    /// One logical CPU per worker, by index.
    LogicalIndex,
    /// One physical core per worker, addressed by a single hardware thread.
    PhysicalCore,
    /// Physical cores spread across memory nodes, latency-critical roles on
    /// the best cores and on one node each so that a barrier never waits on a
    /// remote memory fetch.
    NumaAware,
}

impl PlacementStrategy {
    /// Parses the configuration spelling used by the server command line and
    /// the `KIVI_PLACEMENT` environment variable.
    ///
    /// # Errors
    ///
    /// Returns a human-readable message naming the accepted spellings.
    pub fn parse(spec: &str) -> Result<Self, String> {
        match spec.trim().to_ascii_lowercase().as_str() {
            "none" | "unbound" | "off" => Ok(Self::Unbound),
            "auto" | "simple" | "logical" => Ok(Self::LogicalIndex),
            "core" | "physical" | "physical-core" => Ok(Self::PhysicalCore),
            "numa" | "numa-aware" | "aware" => Ok(Self::NumaAware),
            other => Err(format!(
                "unknown placement strategy {other:?}; expected one of \
                 none, simple, physical-core, numa-aware"
            )),
        }
    }
}

impl fmt::Display for PlacementStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Unbound => "none",
            Self::LogicalIndex => "simple",
            Self::PhysicalCore => "physical-core",
            Self::NumaAware => "numa-aware",
        };
        f.write_str(name)
    }
}

/// One thread to place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkerSlot {
    /// What the thread does.
    pub role: PlacementRole,
    /// Which thread of that role this is, counting from zero.
    pub ordinal: usize,
}

impl WorkerSlot {
    /// Builds a slot.
    #[must_use]
    pub const fn new(role: PlacementRole, ordinal: usize) -> Self {
        Self { role, ordinal }
    }
}

impl fmt::Display for WorkerSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}#{ordinal}", self.role, ordinal = self.ordinal)
    }
}

/// Where one thread will run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    /// The slot this placement is for.
    pub slot: WorkerSlot,
    /// Processors the thread may run on. Empty means unbound.
    pub cpus: CpuSet,
    /// The physical core the placement claims, when one was claimed.
    pub core: Option<CoreKey>,
    /// The memory node the placement targets, when one was chosen.
    pub numa_node: Option<crate::topology::NumaNodeId>,
}

impl fmt::Display for Placement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{slot} -> cpus={cpus}",
            slot = self.slot,
            cpus = self.cpus
        )?;
        match (self.core, self.numa_node) {
            (Some(core), Some(node)) => write!(f, " core={core} node={node}")?,
            (Some(core), None) => write!(f, " core={core}")?,
            (None, Some(node)) => write!(f, " node={node}")?,
            (None, None) => {}
        }
        Ok(())
    }
}

/// A reason a placement could not be honoured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlacementShortfall {
    /// More workers of one role than the machine has cores to give.
    RoleOverflow {
        /// The role that did not fit.
        role: PlacementRole,
        /// How many workers wanted placement.
        requested: usize,
        /// How many could be placed.
        placed: usize,
    },
    /// More workers than cores, so cores are shared.
    CoreSharing {
        /// The core more than one worker was given.
        core: CoreKey,
    },
    /// Two owners were placed on hardware threads of the same core. Only the
    /// naive strategy can do this, and it is the reason the others exist.
    SmtSiblingSharing {
        /// The core both owners landed on.
        core: CoreKey,
    },
    /// The machine has one memory node, so node spreading is a no-op.
    SingleNode,
    /// The machine has no efficiency cores, so class selection is a no-op.
    HomogeneousCpu,
    /// The topology could not be read, so nothing was placed.
    UnknownTopology,
}

/// What the plan did and what it could not do.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PlacementDiagnostics {
    /// The strategy that produced this plan.
    pub strategy: PlacementStrategy,
    /// Physical cores visible after the process affinity mask was applied.
    pub physical_cores: usize,
    /// Logical processors visible after the process affinity mask was applied.
    pub logical_cpus: usize,
    /// Memory nodes visible.
    pub numa_nodes: usize,
    /// Sockets visible.
    pub packages: usize,
    /// Highest cache level present, zero when unknown.
    pub last_level_cache: u8,
    /// True when performance and efficiency cores are both present.
    pub heterogeneous: bool,
    /// Cores whose last-level cache is shared with another core.
    pub llc_coupled_cores: usize,
    /// Every reason the plan could not fully honour its intent.
    pub shortfalls: Vec<PlacementShortfall>,
}

impl PlacementDiagnostics {
    /// One line for logs and the admin surface.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut text = format!(
            "strategy={} cores={} cpus={} nodes={} packages={} llc=L{} smt_coupled={} heterogeneous={}",
            self.strategy,
            self.physical_cores,
            self.logical_cpus,
            self.numa_nodes,
            self.packages,
            self.last_level_cache,
            self.llc_coupled_cores,
            self.heterogeneous,
        );
        for shortfall in &self.shortfalls {
            let reason = match shortfall {
                PlacementShortfall::RoleOverflow {
                    role,
                    requested,
                    placed,
                } => format!(" {role:?} {placed}/{requested}"),
                PlacementShortfall::CoreSharing { core } => format!(" shared={core}"),
                PlacementShortfall::SmtSiblingSharing { core } => format!(" smt-shared={core}"),
                PlacementShortfall::SingleNode => " single-node".to_string(),
                PlacementShortfall::HomogeneousCpu => " homogeneous-cpu".to_string(),
                PlacementShortfall::UnknownTopology => " unknown-topology".to_string(),
            };
            text.push_str(&reason);
        }
        text
    }
}

/// A resolved placement for every requested slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementPlan {
    /// The strategy that produced this plan.
    pub strategy: PlacementStrategy,
    /// One entry per requested slot, in request order.
    pub placements: Vec<Placement>,
    /// What the plan could not do.
    pub diagnostics: PlacementDiagnostics,
}

impl PlacementPlan {
    /// The placement for one slot.
    #[must_use]
    pub fn placement(&self, slot: WorkerSlot) -> Option<&Placement> {
        self.placements.iter().find(|item| item.slot == slot)
    }

    /// Binds the calling thread to its planned processors.
    ///
    /// Returns `Ok(None)` for an unbound plan. A refusal is reported, never
    /// swallowed: a worker that believes it is pinned and is not produces
    /// measurements that mean nothing.
    ///
    /// # Errors
    ///
    /// Returns [`crate::cpuset::BindError`] when the platform rejects the set.
    pub fn bind(
        &self,
        slot: WorkerSlot,
    ) -> Result<Option<crate::cpuset::Binding>, crate::cpuset::BindError> {
        let Some(placement) = self.placement(slot) else {
            return Ok(None);
        };
        crate::cpuset::bind_current_thread(&placement.cpus)
    }
}

/// Builds a placement plan for a fixed set of slots.
///
/// The plan is computed once, at startup, from one topology snapshot. No
/// hot path re-reads the topology, and no plan is ever recomputed while
/// workers are running.
#[must_use]
pub fn plan(
    topology: &Topology,
    strategy: PlacementStrategy,
    slots: &[WorkerSlot],
) -> PlacementPlan {
    let diagnostics = base_diagnostics(topology, strategy);
    match strategy {
        PlacementStrategy::Unbound => unbound(topology, strategy, slots, diagnostics),
        PlacementStrategy::LogicalIndex => logical_index(topology, strategy, slots, diagnostics),
        PlacementStrategy::PhysicalCore => physical(topology, strategy, slots, diagnostics, false),
        PlacementStrategy::NumaAware => physical(topology, strategy, slots, diagnostics, true),
    }
}

fn base_diagnostics(topology: &Topology, strategy: PlacementStrategy) -> PlacementDiagnostics {
    PlacementDiagnostics {
        strategy,
        physical_cores: topology.physical_core_count(),
        logical_cpus: topology.logical_cpu_count(),
        numa_nodes: topology.numa_node_count(),
        packages: topology.package_count(),
        last_level_cache: topology.last_level_cache(),
        heterogeneous: topology.heterogeneous,
        llc_coupled_cores: topology.llc_coupled_cores().len(),
        shortfalls: Vec::new(),
    }
}

fn unbound(
    topology: &Topology,
    strategy: PlacementStrategy,
    slots: &[WorkerSlot],
    mut diagnostics: PlacementDiagnostics,
) -> PlacementPlan {
    if topology.source == crate::topology::TopologySource::FallbackSingleCore
        && topology.logical_cpu_count() == 1
    {
        diagnostics
            .shortfalls
            .push(PlacementShortfall::UnknownTopology);
    }
    PlacementPlan {
        strategy,
        placements: slots
            .iter()
            .map(|slot| Placement {
                slot: *slot,
                cpus: CpuSet::empty(),
                core: None,
                numa_node: None,
            })
            .collect(),
        diagnostics,
    }
}

fn logical_index(
    topology: &Topology,
    strategy: PlacementStrategy,
    slots: &[WorkerSlot],
    mut diagnostics: PlacementDiagnostics,
) -> PlacementPlan {
    // The naive strategy: every owner in the process takes the next logical
    // processor. It is here to be measured against, and on a machine with SMT it
    // is the one that puts two latency-sensitive owners on two hardware threads
    // of the same core while leaving another core idle.
    let mut used: BTreeMap<CpuId, usize> = BTreeMap::new();
    let placements: Vec<Placement> = slots
        .iter()
        .enumerate()
        .map(|(index, slot)| {
            let cpu = topology.cpu_of(index % topology.logical_cpu_count().max(1));
            if let Some(id) = cpu {
                let count = used.entry(id).or_insert(0);
                *count += 1;
                if *count > 1
                    && topology
                        .cpu(id)
                        .is_some_and(crate::topology::LogicalCpu::shares_core)
                {
                    diagnostics
                        .shortfalls
                        .push(PlacementShortfall::SmtSiblingSharing {
                            core: topology.cpu(id).map_or(
                                CoreKey {
                                    package: 0,
                                    die: 0,
                                    core: 0,
                                },
                                |cpu| cpu.core,
                            ),
                        });
                }
            }
            Placement {
                slot: *slot,
                core: cpu.and_then(|id| topology.cpu(id).map(|entry| entry.core)),
                numa_node: cpu.and_then(|id| topology.cpu(id).map(|entry| entry.numa_node)),
                cpus: cpu.map_or_else(CpuSet::empty, CpuSet::single),
            }
        })
        .collect();
    PlacementPlan {
        strategy,
        placements,
        diagnostics,
    }
}

/// Orders candidate cores for a machine.
///
/// Best first: performance class, then the node with the most cores (so
/// spreading leaves a node free), then a core whose last-level cache is not
/// shared (LLC contention is a real tail-latency cost), then core identity for
/// stability.
fn candidate_order(topology: &Topology) -> Vec<CoreKey> {
    let mut cores = topology.physical_cores.clone();
    let biggest_node = topology
        .memory_nodes
        .iter()
        .max_by_key(|node| node.cpus.len())
        .map_or(0, |node| node.id.0);
    cores.sort_by_key(|core| {
        (
            u8::from(core.class == CpuClass::Efficiency),
            u32::from(core.numa_node.0 != biggest_node),
            u8::from(core.llc_shared),
            core.key,
        )
    });
    cores.into_iter().map(|core| core.key).collect()
}

fn physical(
    topology: &Topology,
    strategy: PlacementStrategy,
    slots: &[WorkerSlot],
    mut diagnostics: PlacementDiagnostics,
    node_aware: bool,
) -> PlacementPlan {
    let order = candidate_order(topology);
    if order.is_empty() {
        diagnostics
            .shortfalls
            .push(PlacementShortfall::UnknownTopology);
        return unbound(topology, strategy, slots, diagnostics);
    }
    if node_aware && topology.numa_node_count() <= 1 {
        diagnostics.shortfalls.push(PlacementShortfall::SingleNode);
    }
    if !topology.heterogeneous {
        diagnostics
            .shortfalls
            .push(PlacementShortfall::HomogeneousCpu);
    }

    // Latency-critical roles claim the front of the order, so the best cores go
    // to the durability lane and the consensus workers before background work
    // takes whatever is left.
    let mut ranked: Vec<&WorkerSlot> = slots.iter().collect();
    ranked.sort_by_key(|slot| {
        (
            slot.role.latency_sensitivity().rank(),
            slot.role,
            slot.ordinal,
        )
    });

    // `numa_aware` walks the order in node-sized strides so consecutive owners
    // land on different memory nodes. With one node the stride is one and the
    // result equals `physical_core`, which is exactly right.
    let stride = if node_aware {
        topology.numa_node_count().max(1)
    } else {
        1
    };
    let mut assigned: BTreeMap<WorkerSlot, Placement> = BTreeMap::new();
    let mut used: BTreeMap<CoreKey, usize> = BTreeMap::new();
    let mut per_role: BTreeMap<PlacementRole, (usize, usize)> = BTreeMap::new();

    for (index, slot) in ranked.iter().enumerate() {
        let stride_offset = (index / order.len()) * stride;
        let key = order[(index + stride_offset) % order.len()];
        let count = used.entry(key).or_insert(0);
        *count += 1;
        if *count > 1 {
            diagnostics
                .shortfalls
                .push(PlacementShortfall::CoreSharing { core: key });
        }
        let Some(core) = topology.core(key) else {
            continue;
        };
        let cpus = CpuSet::single(core.anchor_cpu());
        let entry = per_role.entry(slot.role).or_insert((0, 0));
        entry.0 += 1;
        // `placed` counts workers that actually received a processor. The
        // previous form counted workers that received *none*, which made every
        // successful plan report a `RoleOverflow` shortfall: a diagnostic that
        // says "0 of 4 data workers placed" next to four printed placements is
        // worse than no diagnostic, because an operator reads the shortfall
        // list and concludes the plan failed.
        if !cpus.is_empty() {
            entry.1 += 1;
        }
        assigned.insert(
            **slot,
            Placement {
                slot: **slot,
                cpus,
                core: Some(key),
                numa_node: Some(core.numa_node),
            },
        );
    }

    for (role, (requested, placed)) in per_role {
        if placed < requested {
            diagnostics
                .shortfalls
                .push(PlacementShortfall::RoleOverflow {
                    role,
                    requested,
                    placed,
                });
        }
    }

    let placements = slots
        .iter()
        .map(|slot| {
            assigned.get(slot).cloned().unwrap_or(Placement {
                slot: *slot,
                cpus: CpuSet::empty(),
                core: None,
                numa_node: None,
            })
        })
        .collect();
    PlacementPlan {
        strategy,
        placements,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{
        CacheKind, CacheLevel, CoreKey, CpuId, HugePagePools, LogicalCpu, MemoryNode, MemoryTier,
        NumaNodeId, TopologySource,
    };

    fn synthetic(nodes: u32, cores_per_node: u32, threads: u32) -> Topology {
        let mut cpus = Vec::new();
        let mut cache = Vec::new();
        let mut memory_nodes = Vec::new();
        for node in 0..nodes {
            let mut node_cpus = Vec::new();
            for core in 0..cores_per_node {
                let key = CoreKey {
                    package: 0,
                    die: 0,
                    core: node * cores_per_node + core,
                };
                let first = node * cores_per_node * threads + core * threads;
                let mut siblings = Vec::new();
                for thread in 0..threads {
                    siblings.push(CpuId(first + thread));
                }
                for id in &siblings {
                    cpus.push(LogicalCpu {
                        id: *id,
                        core: key,
                        smt_siblings: siblings.clone(),
                        numa_node: NumaNodeId(node),
                        efficiency_class: None,
                        class: CpuClass::Uniform,
                    });
                    node_cpus.push(*id);
                }
            }
            cache.push(CacheLevel {
                level: 3,
                kind: CacheKind::Unified,
                line_bytes: 64,
                size_bytes: 32 * 1024 * 1024,
                shared_by: (0..cores_per_node)
                    .map(|core| CoreKey {
                        package: 0,
                        die: 0,
                        core: node * cores_per_node + core,
                    })
                    .collect(),
            });
            memory_nodes.push(MemoryNode {
                id: NumaNodeId(node),
                cpus: node_cpus,
                total_bytes: Some(64 * 1024 * 1024 * 1024),
                tier: MemoryTier::Ddr,
                huge_pages: HugePagePools::default(),
            });
        }
        Topology::from_parts(
            TopologySource::LinuxSysfs,
            cpus,
            cache,
            memory_nodes,
            Vec::new(),
            &CpuSet::range(nodes * cores_per_node * threads),
        )
    }

    fn slots() -> Vec<WorkerSlot> {
        vec![
            WorkerSlot::new(PlacementRole::DurabilityLane, 0),
            WorkerSlot::new(PlacementRole::DataWorker, 0),
            WorkerSlot::new(PlacementRole::DataWorker, 1),
            WorkerSlot::new(PlacementRole::DataWorker, 2),
            WorkerSlot::new(PlacementRole::RepairWorker, 0),
        ]
    }

    #[test]
    fn single_core_topology_is_always_placeable() {
        let topology = Topology::single_core();
        for strategy in [
            PlacementStrategy::Unbound,
            PlacementStrategy::LogicalIndex,
            PlacementStrategy::PhysicalCore,
            PlacementStrategy::NumaAware,
        ] {
            let plan = plan(&topology, strategy, &slots());
            assert_eq!(plan.placements.len(), slots().len());
        }
    }

    #[test]
    fn unbound_binds_nothing() {
        let topology = synthetic(1, 8, 2);
        let plan = plan(&topology, PlacementStrategy::Unbound, &slots());
        assert!(plan.placements.iter().all(|item| item.cpus.is_empty()));
        assert_eq!(
            plan.bind(WorkerSlot::new(PlacementRole::DataWorker, 0)),
            Ok(None)
        );
    }

    #[test]
    fn physical_core_places_on_distinct_cores_when_it_can() {
        let topology = synthetic(1, 8, 2);
        let plan = plan(&topology, PlacementStrategy::PhysicalCore, &slots());
        let cores: Vec<CoreKey> = plan
            .placements
            .iter()
            .filter_map(|item| item.core)
            .collect();
        assert_eq!(cores.len(), slots().len());
        let unique: std::collections::BTreeSet<CoreKey> = cores.iter().copied().collect();
        assert_eq!(unique.len(), cores.len());
    }

    #[test]
    fn physical_core_never_places_two_workers_on_smt_siblings() {
        let topology = synthetic(1, 8, 2);
        let plan = plan(&topology, PlacementStrategy::PhysicalCore, &slots());
        for placement in &plan.placements {
            let cpu = placement.cpus.iter().next().expect("one cpu per core");
            let entry = topology.cpu(cpu).expect("known cpu");
            assert_eq!(entry.smt_siblings.first(), Some(&entry.id));
        }
    }

    #[test]
    fn numa_aware_spreads_across_nodes() {
        let topology = synthetic(2, 4, 1);
        let plan = plan(&topology, PlacementStrategy::NumaAware, &slots());
        let nodes: std::collections::BTreeSet<u32> = plan
            .placements
            .iter()
            .filter_map(|item| item.numa_node.map(|node| node.0))
            .collect();
        assert_eq!(nodes.len(), 2, "expected both nodes used, got {nodes:?}");
    }

    #[test]
    fn numa_aware_on_one_node_reports_the_shortfall() {
        let topology = synthetic(1, 8, 1);
        let plan = plan(&topology, PlacementStrategy::NumaAware, &slots());
        assert!(
            plan.diagnostics
                .shortfalls
                .contains(&PlacementShortfall::SingleNode)
        );
    }

    #[test]
    fn overflow_is_reported_not_hidden() {
        let topology = synthetic(1, 1, 1);
        let many: Vec<WorkerSlot> = (0..4)
            .map(|ordinal| WorkerSlot::new(PlacementRole::DataWorker, ordinal))
            .collect();
        let plan = plan(&topology, PlacementStrategy::PhysicalCore, &many);
        assert!(
            plan.diagnostics
                .shortfalls
                .iter()
                .any(|shortfall| { matches!(shortfall, PlacementShortfall::CoreSharing { .. }) })
        );
        assert_eq!(plan.placements.len(), 4, "every worker still starts");
    }

    #[test]
    fn a_fully_placed_plan_reports_no_role_overflow() {
        // The failure this catches is a plan that placed every worker and still
        // said it placed none of them. A shortfall list that lies makes every
        // other line in it untrustworthy.
        for (nodes, cores, threads, slots) in [
            (1_u32, 8_u32, 1_u32, slots()),
            (2, 4, 2, slots()),
            (1, 32, 2, slots()),
        ] {
            let topology = synthetic(nodes, cores, threads);
            for strategy in [
                PlacementStrategy::LogicalIndex,
                PlacementStrategy::PhysicalCore,
                PlacementStrategy::NumaAware,
            ] {
                let plan = plan(&topology, strategy, &slots);
                assert!(
                    !plan
                        .diagnostics
                        .shortfalls
                        .iter()
                        .any(|s| matches!(s, PlacementShortfall::RoleOverflow { .. })),
                    "{strategy} on {nodes}x{cores}x{threads} reported overflow: {}",
                    plan.diagnostics.summary()
                );
            }
        }
    }

    #[test]
    fn durability_lane_claims_the_front_of_the_order() {
        let topology = synthetic(1, 8, 1);
        let plan = plan(&topology, PlacementStrategy::PhysicalCore, &slots());
        let lane = plan
            .placement(WorkerSlot::new(PlacementRole::DurabilityLane, 0))
            .expect("lane placed")
            .core;
        let repair = plan
            .placement(WorkerSlot::new(PlacementRole::RepairWorker, 0))
            .expect("repair placed")
            .core;
        assert_ne!(lane, repair);
    }

    #[test]
    fn strategy_parses_every_spelling() {
        assert_eq!(
            PlacementStrategy::parse("none"),
            Ok(PlacementStrategy::Unbound)
        );
        assert_eq!(
            PlacementStrategy::parse("physical-core"),
            Ok(PlacementStrategy::PhysicalCore)
        );
        assert_eq!(
            PlacementStrategy::parse(" NUMA "),
            Ok(PlacementStrategy::NumaAware)
        );
        assert!(PlacementStrategy::parse("turbo").is_err());
    }

    #[test]
    fn diagnostics_render_the_shortfalls() {
        let topology = synthetic(1, 2, 1);
        let plan = plan(&topology, PlacementStrategy::NumaAware, &slots());
        let summary = plan.diagnostics.summary();
        assert!(summary.contains("single-node"), "{summary}");
        assert!(summary.contains("homogeneous-cpu"), "{summary}");
    }
}
