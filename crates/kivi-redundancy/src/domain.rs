//! Failure domains and node health: what counts as independent.
//!
//! Placement understands failure independence today at node granularity
//! while the model carries rack/zone/region labels so those scopes can be
//! enforced later without changing asset semantics. A valid layout never
//! counts two fragments in the same *relevant* failure domain as
//! independent protection: the [`FailureScope`] selected for an intent
//! decides which label must differ.
//!
//! Node lifecycle reuses the control plane's vocabulary
//! (`Active`/`Draining`/`Unavailable`/…) through [`NodeHealth`] without
//! building a second membership system: the fabric consumes health snapshots
//! and never owns admission.

use kivi_types::NodeId;

/// Which scope must differ for two fragments to count as independent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FailureScope {
    /// Fragments must sit on different nodes (baseline, always enforced).
    Node,
    /// Fragments must sit in different racks (when topology names them).
    Rack,
    /// Fragments must sit in different zones (when topology names them).
    Zone,
    /// Fragments must sit in different regions (when topology names them).
    Region,
}

impl FailureScope {
    /// Returns the baseline scope.
    #[must_use]
    pub const fn default() -> Self {
        Self::Node
    }
}

/// Where one node physically lives.
///
/// Empty labels mean "unknown": the planner treats unknown labels as one
/// shared domain (fail closed — never assumes independence it cannot see).
/// Adding rack/zone/region enforcement later only reads more fields here;
/// [`crate::AssetId`] semantics never change.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FailureDomain {
    /// Node identity (always known).
    pub node: NodeId,
    /// Rack label (empty when unknown).
    pub rack: String,
    /// Zone label (empty when unknown).
    pub zone: String,
    /// Region label (empty when unknown).
    pub region: String,
}

impl FailureDomain {
    /// Builds a node-only domain (rack/zone/region unknown).
    #[must_use]
    pub const fn node_only(node: NodeId) -> Self {
        Self {
            node,
            rack: String::new(),
            zone: String::new(),
            region: String::new(),
        }
    }

    /// Builds a fully labeled domain.
    #[must_use]
    pub fn full(node: NodeId, rack: String, zone: String, region: String) -> Self {
        Self {
            node,
            rack,
            zone,
            region,
        }
    }

    /// Key that must differ for independence under `scope`.
    ///
    /// Unknown labels collapse to one shared key (`"?"`), so two nodes with
    /// unnamed racks never count as rack-independent.
    #[must_use]
    pub fn independence_key(&self, scope: FailureScope) -> String {
        match scope {
            FailureScope::Node => format!("node:{}", self.node.as_u64()),
            FailureScope::Rack => {
                if self.rack.is_empty() {
                    "?".to_owned()
                } else {
                    format!("rack:{}", self.rack)
                }
            }
            FailureScope::Zone => {
                if self.zone.is_empty() {
                    "?".to_owned()
                } else {
                    format!("zone:{}", self.zone)
                }
            }
            FailureScope::Region => {
                if self.region.is_empty() {
                    "?".to_owned()
                } else {
                    format!("region:{}", self.region)
                }
            }
        }
    }
}

/// Health of one node as seen by the fabric (mirror of the control-plane
/// lifecycle, consumed as a snapshot — never owned here).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeHealth {
    /// Serving and eligible for new placements.
    Active,
    /// Transiently suspect; keeps placements, never triggers repair alone.
    Suspect,
    /// Repair-worthy outage; excluded from new placements.
    Unavailable,
    /// Operator drain; excluded from new placements, replicas migrate away.
    Draining,
    /// Drained; holds no desired data.
    Drained,
    /// Joining; not yet eligible.
    Joining,
    /// Removed; tombstoned.
    Removed,
}

impl NodeHealth {
    /// Whether the node may receive new fragments.
    #[must_use]
    pub const fn accepts_new(self) -> bool {
        matches!(self, Self::Active)
    }

    /// Whether the node is expected to serve reads.
    #[must_use]
    pub const fn serves_reads(self) -> bool {
        matches!(self, Self::Active | Self::Suspect | Self::Draining)
    }

    /// Whether the node's fragments need replacement.
    #[must_use]
    pub const fn needs_repair(self) -> bool {
        matches!(self, Self::Unavailable | Self::Draining)
    }
}

/// One placement candidate: identity, failure domain, health, weight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeDescriptor {
    /// Stable node identity.
    pub id: NodeId,
    /// Physical failure domain.
    pub domain: FailureDomain,
    /// Current health snapshot.
    pub health: NodeHealth,
    /// Capacity weight (`1` = default; `0` is normalized to `1`).
    pub weight: u32,
}

impl NodeDescriptor {
    /// Builds an active node-only descriptor.
    #[must_use]
    pub const fn active(id: NodeId) -> Self {
        Self {
            id,
            domain: FailureDomain::node_only(id),
            health: NodeHealth::Active,
            weight: 1,
        }
    }

    /// Effective weight (never zero).
    #[must_use]
    pub fn effective_weight(&self) -> u32 {
        if self.weight == 0 { 1 } else { self.weight }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_scope_always_distinguishes_nodes() {
        let a = FailureDomain::node_only(NodeId::from_u64(1));
        let b = FailureDomain::node_only(NodeId::from_u64(2));
        assert_ne!(
            a.independence_key(FailureScope::Node),
            b.independence_key(FailureScope::Node)
        );
    }

    #[test]
    fn unknown_labels_collapse_fail_closed() {
        let a = FailureDomain::node_only(NodeId::from_u64(1));
        let b = FailureDomain::node_only(NodeId::from_u64(2));
        assert_eq!(
            a.independence_key(FailureScope::Rack),
            b.independence_key(FailureScope::Rack)
        );
        let c = FailureDomain::full(
            NodeId::from_u64(3),
            "rack-a".to_owned(),
            String::new(),
            String::new(),
        );
        let d = FailureDomain::full(
            NodeId::from_u64(4),
            "rack-b".to_owned(),
            String::new(),
            String::new(),
        );
        assert_ne!(
            c.independence_key(FailureScope::Rack),
            d.independence_key(FailureScope::Rack)
        );
    }

    #[test]
    fn health_predicates_mirror_control_lifecycle() {
        assert!(NodeHealth::Active.accepts_new());
        assert!(!NodeHealth::Draining.accepts_new());
        assert!(!NodeHealth::Suspect.accepts_new());
        assert!(NodeHealth::Unavailable.needs_repair());
        assert!(NodeHealth::Draining.needs_repair());
        assert!(!NodeHealth::Suspect.needs_repair());
    }
}
