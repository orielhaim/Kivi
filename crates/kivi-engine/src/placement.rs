//! Where Kivi's owner threads run, decided once from the real machine.
//!
//! Kivi's execution model already names its threads: one data worker per
//! worker index, one durability lane, one off-core lane, one chunk lane, one
//! checkpoint thread, and one reactor inside each networked worker. Each is a
//! long-lived owner with a stable job, so each is a placement decision made once
//! at startup rather than a scheduling policy re-evaluated per request.
//!
//! This module is the engine's view of `kivi-hardware`'s placement plan. It
//! exists so that no engine type has to know about CPU ids, cores, or NUMA
//! nodes, and so that a thread which cannot bind says so at startup instead of
//! running unplaced and reporting measurements that mean nothing.
//!
//! ## Binding is reported, not assumed
//!
//! A refusal to bind is an error the caller must handle, and the channel-only
//! worker path and the networked worker path both propagate it. That is
//! deliberate: a worker that believes it is pinned and is not produces
//! benchmark numbers that cannot be trusted, and Phase 13's whole method is
//! measurement.

use std::sync::Arc;

use kivi_hardware::{
    BindError, CpuId, PlacementDiagnostics, PlacementPlan, PlacementRole, PlacementStrategy,
    Topology, WorkerSlot, plan,
};

/// Every worker slot the engine places.
///
/// The count per role is fixed by the engine's own ownership model, so this is
/// the single place that says "Kivi runs exactly this many of these".
pub const DEFAULT_ROLES: &[(PlacementRole, usize)] = &[
    (PlacementRole::DurabilityLane, 1),
    (PlacementRole::DataWorker, 4),
    (PlacementRole::ConsensusWorker, 2),
    (PlacementRole::OffcoreLane, 4),
    (PlacementRole::RepairWorker, 2),
    (PlacementRole::CheckpointWorker, 1),
    (PlacementRole::NetworkReactor, 4),
];

/// Why a thread could not be placed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PlacementError {
    /// The platform refused the cpu set.
    #[error("could not bind {role:?}#{ordinal} to {cpus}: {reason}")]
    Refused {
        /// The slot that could not be bound.
        role: PlacementRole,
        /// Which thread of that role.
        ordinal: usize,
        /// The set the plan asked for.
        cpus: String,
        /// The platform's reason.
        reason: String,
    },
}

/// How a node's owner threads are placed on the machine.
///
/// Physical configuration, as opposed to cluster state: it comes from the
/// operator's command line and environment, and it never replicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HardwareConfig {
    /// How workers map onto cores.
    pub strategy: PlacementStrategy,
    /// Whether large long-lived arenas are advised for huge pages.
    pub huge_pages: kivi_hardware::pages::HugePagePolicy,
    /// Whether the off-core materialization store bypasses the page cache.
    pub direct_io: kivi_hardware::io::DirectIoPolicy,
    /// Whether per-thread hardware performance counters are opened.
    pub pmu: bool,
}

impl Default for HardwareConfig {
    /// Everything off.
    ///
    /// The default is deliberately inert: a machine Kivi has never run on must
    /// behave the way it did before the hardware layer existed, and an
    /// operator opts into placement and accelerators explicitly. The one
    /// exception is the strategy, whose default is measured rather than chosen
    /// (see `docs/hardware.md`); it is resolved by
    /// [`HardwareConfig::resolve`] so the measurement, not this `Default`, is
    /// what decides.
    fn default() -> Self {
        Self {
            strategy: PlacementStrategy::Unbound,
            huge_pages: kivi_hardware::pages::HugePagePolicy::Inherit,
            direct_io: kivi_hardware::io::DirectIoPolicy::Adaptive,
            pmu: false,
        }
    }
}

impl HardwareConfig {
    /// Resolves the strategy from the environment when it was not set
    /// explicitly, so a deployment can be placed without editing a config file.
    #[must_use]
    pub fn resolve(mut self) -> Self {
        if self.strategy == PlacementStrategy::Unbound
            && let Ok(spec) = std::env::var("KIVI_PLACEMENT")
            && let Ok(strategy) = PlacementStrategy::parse(&spec)
        {
            self.strategy = strategy;
        }
        self
    }
}

/// A resolved placement plan shared by every thread the engine starts.
///
/// Cloning is a reference-count bump: threads started later, and threads
/// started by subsystems the engine does not own, all read the same plan.
#[derive(Debug, Clone)]
pub struct ThreadPlacement {
    plan: Arc<PlacementPlan>,
    topology: Arc<Topology>,
}

impl ThreadPlacement {
    /// Resolves a plan for `worker_count` data workers and the fixed number of
    /// every other role.
    #[must_use]
    pub fn resolve(strategy: PlacementStrategy, worker_count: usize) -> Self {
        Self::resolve_with(strategy, DEFAULT_ROLES, worker_count)
    }

    /// Resolves a plan for an explicit role census. `worker_count` sets how many
    /// data workers, chunk-facing off-core lanes and network reactors the plan
    /// should expect, because those are per worker.
    #[must_use]
    pub fn resolve_with(
        strategy: PlacementStrategy,
        roles: &[(PlacementRole, usize)],
        worker_count: usize,
    ) -> Self {
        let topology = Topology::discover();
        let mut slots = Vec::new();
        for (role, count) in roles {
            let count = if *role == PlacementRole::DataWorker
                || *role == PlacementRole::OffcoreLane
                || *role == PlacementRole::NetworkReactor
            {
                (*count).max(worker_count)
            } else {
                *count
            };
            slots.extend((0..count).map(|ordinal| WorkerSlot::new(*role, ordinal)));
        }
        let plan = plan(&topology, strategy, &slots);
        Self {
            plan: Arc::new(plan),
            topology: Arc::new(topology),
        }
    }

    /// An all-unbound plan, for tests and for embedders that manage their own
    /// threads. Diagnostics still describe the real machine, because that is
    /// what an operator needs to see.
    #[must_use]
    pub fn unbound() -> Self {
        Self::resolve(PlacementStrategy::Unbound, 1)
    }

    /// Binds the calling thread to its slot, then reports where it landed.
    ///
    /// # Errors
    ///
    /// Returns [`PlacementError::Refused`] when the platform rejects the plan's
    /// cpu set. A thread that cannot be placed does not start, so a deployment
    /// either has its placement or knows it does not.
    pub fn bind(&self, slot: WorkerSlot) -> Result<Option<kivi_hardware::Binding>, PlacementError> {
        let refused = |error: BindError| PlacementError::Refused {
            role: slot.role,
            ordinal: slot.ordinal,
            cpus: self
                .plan
                .placement(slot)
                .map_or_else(|| "any".to_string(), |placement| placement.cpus.to_string()),
            reason: error.to_string(),
        };
        self.plan.bind(slot).map_err(refused)
    }

    /// The plan, for diagnostics and for subsystems that want to read their own
    /// assignment without binding.
    #[must_use]
    pub fn plan(&self) -> &PlacementPlan {
        &self.plan
    }

    /// The machine this plan was resolved against.
    #[must_use]
    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    /// What the plan did and what it could not do.
    #[must_use]
    pub fn diagnostics(&self) -> &PlacementDiagnostics {
        &self.plan.diagnostics
    }

    /// One line for the startup log and the admin surface.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} | {}",
            self.topology.summary(),
            self.plan.diagnostics.summary()
        )
    }

    /// The processor a bound thread landed on, for a diagnostic that records
    /// where a thread actually is rather than where it was asked to go.
    #[must_use]
    pub fn current_cpu(&self) -> Option<CpuId> {
        kivi_hardware::current_cpu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_role_is_covered() {
        let placement = ThreadPlacement::resolve(PlacementStrategy::PhysicalCore, 2);
        for role in PlacementRole::ALL {
            let wanted = slot_count(*role, 2);
            let placed = (0..wanted)
                .filter(|ordinal| {
                    placement
                        .plan()
                        .placement(WorkerSlot::new(*role, *ordinal))
                        .is_some()
                })
                .count();
            assert_eq!(placed, wanted, "{role:?} is not fully placed");
        }
    }

    fn slot_count(role: PlacementRole, workers: usize) -> usize {
        DEFAULT_ROLES
            .iter()
            .find(|(candidate, _)| *candidate == role)
            .map_or(0, |(_, count)| {
                if matches!(
                    role,
                    PlacementRole::DataWorker
                        | PlacementRole::OffcoreLane
                        | PlacementRole::NetworkReactor
                ) {
                    (*count).max(workers)
                } else {
                    *count
                }
            })
    }

    #[test]
    fn unbound_plans_bind_nothing() {
        let placement = ThreadPlacement::resolve(PlacementStrategy::Unbound, 1);
        assert_eq!(
            placement.bind(WorkerSlot::new(PlacementRole::DataWorker, 0)),
            Ok(None)
        );
    }

    #[test]
    fn the_summary_always_describes_the_machine() {
        let summary = ThreadPlacement::unbound().summary();
        assert!(summary.contains("strategy=none"), "{summary}");
        assert!(summary.contains("logical="), "{summary}");
    }
}
