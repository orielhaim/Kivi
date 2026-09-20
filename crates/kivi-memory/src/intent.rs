//! Declarative materialization requirements (RFC §35).
//!
//! This module is the Declarative Memory Services boundary: software
//! describes *what* a materialization must provide and the runtime chooses
//! *which* provider satisfies it. Nothing here names a device, a tier, or
//! an ordering of providers. Providers advertise measured capabilities
//! ([`crate::provider::ProviderCaps`]); the planner scores them against an
//! intent. Adding a future device (CXL, PM, RDMA, DPA) means registering a
//! new provider with capabilities, never changing this intent schema.
//!
//! Design notes from the required reading:
//! - DMS (CIDR 2026): intent/provider/calibration split is the load-bearing
//!   boundary. Intent is stable; providers and calibration evolve.
//! - Tiered Memory Beyond Hotness (OSDI 2025): intent carries latency and
//!   tail targets plus reconstruction cost, so placement never reduces to
//!   access frequency.

use kivi_types::TabletId;

/// Opaque tenant tag for per-tenant movement budgets.
///
/// The fabric never interprets this beyond budget accounting; namespaces
/// map their own tenant model onto it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TenantTag(pub u64);

/// How strongly an object must survive process/host loss *as a
/// materialization*.
///
/// This is not the namespace durability contract (which the Durability
/// Fabric owns). It tells the planner whether a volatile representation
/// alone is acceptable or whether a stable copy must also exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VolatilityAllowed {
    /// Any representation may be lost at any time; the caller can
    /// reconstruct from the authoritative source (Midas-style soft state).
    Ephemeral,
    /// Loss of the representation costs a re-materialization but never
    /// loses acknowledged state (a durable source exists elsewhere).
    Reconstructible,
    /// This materialization itself must survive process restart.
    MustSurviveRestart,
}

/// Read/write coherence the materialization must provide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoherenceRequirement {
    /// Single-owner exclusive access. The common case: the owning worker
    /// mutates; no other core observes the bytes concurrently.
    Exclusive,
    /// Concurrent read-mostly sharing is allowed (immutable snapshots,
    /// read-only mapped access to chunks).
    SharedRead,
}

/// Who may observe this materialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharingScope {
    /// Only the owning worker.
    WorkerLocal,
    /// Any worker on this node.
    NodeLocal,
    /// Remote readers (bulk chunk sources, DPA caches).
    Cluster,
}

/// NUMA/locality constraint on the materialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LocalityRequirement {
    /// No constraint; any node-local memory is fine.
    Any,
    /// Prefer the owning worker's NUMA node; spilling elsewhere is allowed
    /// under pressure but scored worse.
    PreferredNuma {
        /// Opaque NUMA node index from [`crate::numa::NumaTopology`].
        node: u32,
    },
    /// Must stay on the owning worker's NUMA node.
    PinnedNuma {
        /// Opaque NUMA node index.
        node: u32,
    },
}

/// Declarative requirements for one object's materialization.
///
/// All fields are *requirements*, not mechanisms. Two objects with the same
/// bytes but different intents may legitimately live on different
/// providers; two providers may satisfy the same intent with different
/// costs, and the planner picks by measured cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializationIntent {
    /// Target mean access latency in nanoseconds, if the namespace states
    /// one. `None` means latency-insensitive (archive-adjacent).
    pub access_latency_target_ns: Option<u64>,
    /// Target p99 access latency in nanoseconds, if stated.
    pub tail_latency_target_ns: Option<u64>,
    /// Expected reads per second (planner hint, not a reservation).
    pub expected_read_rate: u64,
    /// Expected writes per second (planner hint, not a reservation).
    pub expected_write_rate: u64,
    /// What loss of *this representation* is allowed.
    pub volatility_allowed: VolatilityAllowed,
    /// Whether the representation alone must be restart-durable.
    pub durability_required: bool,
    /// Coherence the representation must provide.
    pub coherence: CoherenceRequirement,
    /// Who may observe the representation.
    pub sharing: SharingScope,
    /// Sustained bandwidth the workload needs, bytes per second.
    pub bandwidth_requirement_bps: u64,
    /// Whether compute near the data is desired (future DPA/CXL hook).
    /// Informational in Phase 9; no provider is required to honor it.
    pub compute_near_data: bool,
    /// Whether the bytes must be encrypted at rest on this provider.
    pub encryption_required: bool,
    /// Locality constraint.
    pub locality: LocalityRequirement,
    /// Relative cost weight in basis points (100 = neutral). Higher means
    /// the tenant prefers cheaper materializations.
    pub cost_weight_bps: u32,
    /// Reclamation priority: lower values are reclaimed first under
    /// pressure. Cold/archive-adjacent objects use low values; latency
    /// critical objects use high values.
    pub reclaim_priority: u32,
    /// Owning tablet (for migration/split/merge fencing).
    pub tablet: Option<TabletId>,
    /// Owning tenant (for per-tenant movement budgets).
    pub tenant: Option<TenantTag>,
}

impl MaterializationIntent {
    /// Hot small-object default: fast local access, reconstructible from
    /// the authoritative store, no special constraints.
    #[must_use]
    pub const fn hot() -> Self {
        Self {
            access_latency_target_ns: Some(2_000),
            tail_latency_target_ns: Some(10_000),
            expected_read_rate: 1_000,
            expected_write_rate: 100,
            volatility_allowed: VolatilityAllowed::Reconstructible,
            durability_required: false,
            coherence: CoherenceRequirement::Exclusive,
            sharing: SharingScope::WorkerLocal,
            bandwidth_requirement_bps: 0,
            compute_near_data: false,
            encryption_required: false,
            locality: LocalityRequirement::Any,
            cost_weight_bps: 100,
            reclaim_priority: 200,
            tablet: None,
            tenant: None,
        }
    }

    /// Cold default: latency-insensitive, cheap, reclaimed first.
    #[must_use]
    pub const fn cold() -> Self {
        Self {
            access_latency_target_ns: None,
            tail_latency_target_ns: None,
            expected_read_rate: 1,
            expected_write_rate: 0,
            volatility_allowed: VolatilityAllowed::Reconstructible,
            durability_required: false,
            coherence: CoherenceRequirement::SharedRead,
            sharing: SharingScope::NodeLocal,
            bandwidth_requirement_bps: 0,
            compute_near_data: false,
            encryption_required: false,
            locality: LocalityRequirement::Any,
            cost_weight_bps: 300,
            reclaim_priority: 10,
            tablet: None,
            tenant: None,
        }
    }

    /// Whether this intent permits a volatile-only representation.
    #[must_use]
    pub const fn allows_volatile(&self) -> bool {
        !self.durability_required
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hot_prefers_latency_and_cold_prefers_cost() {
        let hot = MaterializationIntent::hot();
        let cold = MaterializationIntent::cold();
        assert!(hot.access_latency_target_ns.is_some());
        assert!(cold.access_latency_target_ns.is_none());
        assert!(hot.reclaim_priority > cold.reclaim_priority);
        assert!(hot.allows_volatile());
    }
}
