//! Provider capabilities and registry.
//!
//! There is no fixed tier hierarchy anywhere in this crate. Providers are
//! registered dynamically under an opaque [`ProviderId`] with a measured
//! [`ProviderCaps`] descriptor. The planner scores providers against a
//! [`MaterializationIntent`](crate::intent::MaterializationIntent); adding a
//! future device means registering new capabilities, never editing an enum.
//!
//! Well-known provider kinds ([`ProviderKind::DRAM`], `COMPRESSED`, `NVME`,
//! `SIM`) are `&'static str` constants for diagnostics and tests, not an
//! ordering. A deployment with only DRAM + `NVMe` registers two providers; a
//! future CXL deployment registers a third. Placement never branches on
//! kind equality for correctness, only for cost scoring.

use crate::intent::{CoherenceRequirement, LocalityRequirement, SharingScope};

/// Opaque provider identifier assigned at registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProviderId(pub u32);

impl ProviderId {
    /// Reserved identifier for "no provider" (placement deferred).
    pub const NONE: Self = Self(u32::MAX);
}

/// Well-known provider kind tags.
///
/// These are human-readable labels, not a hierarchy. Two providers may
/// share a kind tag (e.g. two `NVMe` devices); the planner distinguishes
/// them by calibrated capabilities, never by tag order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProviderKind(pub &'static str);

impl ProviderKind {
    /// Local DRAM materialization.
    pub const DRAM: Self = Self("dram");
    /// Compressed DRAM representation (explicit, not swap).
    pub const COMPRESSED: Self = Self("compressed-dram");
    /// Local `NVMe` demotion target.
    pub const NVME: Self = Self("nvme");
    /// Deterministic simulated provider for tests.
    pub const SIM: Self = Self("sim");
    /// Future extension point (CXL, PM, RDMA, DPA, ...).
    pub const EXTENSION: Self = Self("extension");
}

/// NUMA/locality properties of a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalityCaps {
    /// NUMA node this provider's memory is attached to, if known.
    pub numa_node: Option<u32>,
    /// Whether access from a remote NUMA node pays extra latency.
    pub remote_penalty_ns: u64,
    /// Whether the provider is reachable from any worker on this node.
    pub node_shared: bool,
}

/// Static + calibrated capabilities of one provider.
///
/// Static fields describe what the provider *can* do; loaded fields
/// (maintained by [`crate::calibrate::Calibrator`]) describe what it
/// *currently costs*. The planner multiplies the two: a provider that
/// cannot satisfy a hard requirement is excluded, the rest are scored by
/// measured cost.
///
/// Capability bits are intentionally independent toggles (any combination
/// is meaningful), so they stay flat instead of nesting enums.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderCaps {
    /// Human-readable kind tag (diagnostics only).
    pub kind: ProviderKind,
    /// NUMA/locality properties.
    pub locality: LocalityCaps,
    /// Volatile only (`true` for DRAM/compressed) or restart-durable.
    pub durable: bool,
    /// Supports direct concurrent reads without exclusive fencing.
    pub shared_read: bool,
    /// Supports in-place mutation (DRAM arenas) vs rewrite-only (`NVMe`).
    pub mutable_in_place: bool,
    /// Supports Direct I/O / raw-device access as an optimization.
    /// Baseline correctness never requires it.
    pub direct_io: bool,
    /// Supports Flexible Data Placement hints as an optimization.
    pub fdp: bool,
    /// Advertised mean read latency in nanoseconds (calibrated).
    pub read_latency_ns: u64,
    /// Advertised mean write latency in nanoseconds (calibrated).
    pub write_latency_ns: u64,
    /// Advertised sustained read bandwidth, bytes per second.
    pub read_bandwidth_bps: u64,
    /// Advertised sustained write bandwidth, bytes per second.
    pub write_bandwidth_bps: u64,
    /// Cost per stored byte per second in abstract cost units.
    pub cost_per_byte: u64,
    /// CPU cost per decompression in nanoseconds per byte (compressed
    /// providers; zero elsewhere).
    pub decompress_ns_per_byte: u64,
    /// Total capacity in bytes (`u64::MAX` for effectively unbounded).
    pub capacity_bytes: u64,
    /// Free bytes (maintained by the provider, observed by calibration).
    pub free_bytes: u64,
    /// Whether the provider is currently healthy.
    pub healthy: bool,
}

impl ProviderCaps {
    /// Local DRAM baseline: fast, mutable, volatile, node-local.
    #[must_use]
    pub const fn dram(numa_node: Option<u32>, capacity_bytes: u64) -> Self {
        Self {
            kind: ProviderKind::DRAM,
            locality: LocalityCaps {
                numa_node,
                remote_penalty_ns: 40,
                node_shared: true,
            },
            durable: false,
            shared_read: false,
            mutable_in_place: true,
            direct_io: false,
            fdp: false,
            read_latency_ns: 90,
            write_latency_ns: 90,
            read_bandwidth_bps: 20_000_000_000,
            write_bandwidth_bps: 20_000_000_000,
            cost_per_byte: 100,
            decompress_ns_per_byte: 0,
            capacity_bytes,
            free_bytes: capacity_bytes,
            healthy: true,
        }
    }

    /// Compressed DRAM: slower access, CPU cost, much cheaper per byte.
    #[must_use]
    pub const fn compressed(numa_node: Option<u32>, capacity_bytes: u64) -> Self {
        Self {
            kind: ProviderKind::COMPRESSED,
            locality: LocalityCaps {
                numa_node,
                remote_penalty_ns: 40,
                node_shared: true,
            },
            durable: false,
            shared_read: false,
            mutable_in_place: false,
            direct_io: false,
            fdp: false,
            read_latency_ns: 600,
            write_latency_ns: 2_500,
            read_bandwidth_bps: 5_000_000_000,
            write_bandwidth_bps: 1_500_000_000,
            cost_per_byte: 25,
            decompress_ns_per_byte: 2,
            capacity_bytes,
            free_bytes: capacity_bytes,
            healthy: true,
        }
    }

    /// Local `NVMe`: durable, high latency, rewrite-only, cheap per byte.
    #[must_use]
    pub const fn nvme(capacity_bytes: u64) -> Self {
        Self {
            kind: ProviderKind::NVME,
            locality: LocalityCaps {
                numa_node: None,
                remote_penalty_ns: 0,
                node_shared: true,
            },
            durable: true,
            shared_read: true,
            mutable_in_place: false,
            direct_io: false,
            fdp: false,
            read_latency_ns: 25_000,
            write_latency_ns: 35_000,
            read_bandwidth_bps: 3_000_000_000,
            write_bandwidth_bps: 2_000_000_000,
            cost_per_byte: 3,
            decompress_ns_per_byte: 0,
            capacity_bytes,
            free_bytes: capacity_bytes,
            healthy: true,
        }
    }

    /// Whether this provider can satisfy the hard requirements of an
    /// intent. Soft preferences (latency targets, cost) affect scoring,
    /// never exclusion, except when the provider is unhealthy.
    #[must_use]
    pub fn satisfies(
        &self,
        coherence: CoherenceRequirement,
        sharing: SharingScope,
        durability_required: bool,
        locality: LocalityRequirement,
        encryption_required: bool,
    ) -> bool {
        if !self.healthy {
            return false;
        }
        if durability_required && !self.durable {
            return false;
        }
        // Encryption is a deployment property, not a Phase 9 provider
        // property: no Phase 9 provider offers it, so an intent requiring
        // it excludes all Phase 9 providers (fail closed into the
        // correctness path, per RFC §2).
        if encryption_required {
            return false;
        }
        match (coherence, self.mutable_in_place, self.shared_read) {
            (CoherenceRequirement::Exclusive, _, _)
            | (CoherenceRequirement::SharedRead, _, true) => {}
            (CoherenceRequirement::SharedRead, _, false) => return false,
        }
        match sharing {
            SharingScope::WorkerLocal => {}
            SharingScope::NodeLocal => {
                if !self.locality.node_shared {
                    return false;
                }
            }
            SharingScope::Cluster => {
                if !self.shared_read {
                    return false;
                }
            }
        }
        match locality {
            LocalityRequirement::Any | LocalityRequirement::PreferredNuma { .. } => true,
            LocalityRequirement::PinnedNuma { node } => self.locality.numa_node == Some(node),
        }
    }
}

/// One registered provider: identifier plus current capabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderDescriptor {
    /// Assigned identifier.
    pub id: ProviderId,
    /// Current capabilities (static + calibrated).
    pub caps: ProviderCaps,
}

/// Dynamic provider registry. No ordering, no hierarchy: iteration order
/// is registration order and carries no placement meaning.
#[derive(Debug, Default)]
pub struct ProviderRegistry {
    providers: Vec<ProviderDescriptor>,
}

impl ProviderRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    /// Registers a provider, returning its identifier.
    pub fn register(&mut self, caps: ProviderCaps) -> ProviderId {
        let id = ProviderId(u32::try_from(self.providers.len()).unwrap_or(u32::MAX));
        self.providers.push(ProviderDescriptor { id, caps });
        id
    }

    /// Returns all registered providers.
    #[must_use]
    pub fn all(&self) -> &[ProviderDescriptor] {
        &self.providers
    }

    /// Looks up one provider's capabilities.
    #[must_use]
    pub fn get(&self, id: ProviderId) -> Option<&ProviderCaps> {
        self.providers
            .iter()
            .find(|descriptor| descriptor.id == id)
            .map(|descriptor| &descriptor.caps)
    }

    /// Mutably looks up one provider's capabilities (for calibration).
    pub fn get_mut(&mut self, id: ProviderId) -> Option<&mut ProviderCaps> {
        self.providers
            .iter_mut()
            .find(|descriptor| descriptor.id == id)
            .map(|descriptor| &mut descriptor.caps)
    }

    /// Number of registered providers.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.providers.len()
    }

    /// Whether any provider is registered.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent::MaterializationIntent;

    #[test]
    fn registry_assigns_stable_ids_without_hierarchy() {
        let mut registry = ProviderRegistry::new();
        let dram = registry.register(ProviderCaps::dram(Some(0), 1 << 30));
        let nvme = registry.register(ProviderCaps::nvme(1 << 40));
        assert_ne!(dram, nvme);
        assert_eq!(registry.len(), 2);
        assert!(registry.get(dram).is_some());
    }

    #[test]
    fn durability_requirement_excludes_volatile_providers() {
        let dram = ProviderCaps::dram(Some(0), 1 << 30);
        let nvme = ProviderCaps::nvme(1 << 40);
        let mut intent = MaterializationIntent::hot();
        intent.durability_required = true;
        assert!(!dram.satisfies(
            intent.coherence,
            intent.sharing,
            intent.durability_required,
            intent.locality,
            intent.encryption_required
        ));
        // `NVMe` is durable but hot intent wants WorkerLocal+Exclusive;
        // `NVMe` is shared_read+rewrite-only which still satisfies the hard
        // gate (sharing WorkerLocal needs nothing extra).
        assert!(nvme.satisfies(
            intent.coherence,
            intent.sharing,
            intent.durability_required,
            intent.locality,
            intent.encryption_required
        ));
    }
}
