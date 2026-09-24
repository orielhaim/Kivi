//! Deterministic failure-domain-aware placement.
//!
//! The planner maps a layout's fragment count onto live nodes so that no
//! two fragments counted as independent protection share the same relevant
//! failure domain. Scoring is deterministic rendezvous hashing over
//! `(asset, generation, fragment index, node)`, so every correct participant
//! derives the same desired placement without coordination; ties break by
//! node id. The model cleanly distinguishes desired, healthy, degraded, and
//! reconstruction-target views.

use std::collections::{HashMap, HashSet};

use kivi_types::NodeId;

use crate::{
    AssetId, FailureScope, LocalityRequirement, NodeDescriptor, RedundancyError, SchemeParams,
};

/// Where one fragment should live.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FragmentPlacement {
    /// Fragment index (`0..total`; replication copies are `0..copies`).
    pub index: u32,
    /// Node that should hold it.
    pub node: NodeId,
    /// Independence key at plan time (diagnostics, domain audit).
    pub domain_key: String,
}

/// Desired placement for one generation: every fragment's home.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DesiredPlacement {
    /// Asset placed.
    pub asset: AssetId,
    /// Generation placed.
    pub generation: u64,
    /// Scope enforced.
    pub scope: FailureScope,
    /// One entry per fragment, sorted by index.
    pub fragments: Vec<FragmentPlacement>,
}

impl DesiredPlacement {
    /// Nodes holding fragments (sorted, deduplicated).
    #[must_use]
    pub fn nodes(&self) -> Vec<NodeId> {
        let mut out: Vec<NodeId> = self.fragments.iter().map(|f| f.node).collect();
        out.sort_by_key(|node| node.as_u64());
        out.dedup_by_key(|node| node.as_u64());
        out
    }

    /// Node for one fragment index, if placed.
    #[must_use]
    pub fn node_for(&self, index: u32) -> Option<NodeId> {
        self.fragments
            .iter()
            .find(|f| f.index == index)
            .map(|f| f.node)
    }
}

/// Currently-healthy placement: desired fragments that are live and verified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthyPlacement {
    /// Fragment indexes with verified healthy bytes.
    pub healthy: Vec<u32>,
    /// Fragment indexes known missing or corrupt.
    pub missing: Vec<u32>,
}

/// Degraded placement: what remains plus what must move.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DegradedPlacement {
    /// Healthy fragment indexes.
    pub healthy: Vec<u32>,
    /// Indexes needing replacement and why.
    pub need_repair: Vec<u32>,
    /// Whether reconstruction is still possible now.
    pub reconstructable: bool,
}

/// Reconstruction targets: where replacement fragments should go.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconstructionTargets {
    /// `(fragment index, target node)` pairs.
    pub targets: Vec<(u32, NodeId)>,
}

/// Mixes one `u64` with FNV-1a (deterministic across platforms).
fn mix(mut state: u64, word: u64) -> u64 {
    state ^= word;
    state = state.wrapping_mul(0x1000_0000_01B3);
    state ^= state >> 29;
    state = state.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    state
}

/// Deterministic score for `(asset, generation, index, node)`.
fn score(asset: &AssetId, generation: u64, index: u32, node: NodeId) -> u64 {
    let mut state: u64 = 0xCBF2_9CE4_8422_2325;
    for chunk in asset.hash.chunks(8) {
        let mut word = [0u8; 8];
        word.copy_from_slice(chunk);
        state = mix(state, u64::from_le_bytes(word));
    }
    state = mix(state, u64::from(asset.kind.as_u8()));
    state = mix(state, asset.domain);
    state = mix(state, generation);
    state = mix(state, u64::from(index));
    state = mix(state, node.as_u64());
    // Weight is applied by the caller (replicated scoring rounds); the hash
    // itself stays weight-free so audits reproduce it exactly.
    state
}

/// Plans desired placement for `total` fragments.
///
/// Eligible nodes are `accepts_new` and allowed by `allowed` labels (empty =
/// any). Fragments pick the highest-scoring eligible node whose independence
/// key is still unused; when independence is impossible (fewer domains than
/// fragments), the planner reuses the least-loaded domain but reports the
/// collision so health math never over-counts protection (see
/// [`independent_survivable`]).
///
/// # Errors
///
/// Returns [`RedundancyError::Placement`] when no eligible node exists or
/// `total` is zero.
pub fn plan_placement(
    asset: AssetId,
    generation: u64,
    total: u32,
    scope: FailureScope,
    nodes: &[NodeDescriptor],
    allowed: &[String],
) -> Result<DesiredPlacement, RedundancyError> {
    if total == 0 {
        return Err(RedundancyError::Placement {
            detail: "placement needs at least one fragment".to_owned(),
        });
    }
    let eligible: Vec<&NodeDescriptor> = nodes
        .iter()
        .filter(|n| {
            n.health.accepts_new()
                && (allowed.is_empty()
                    || allowed.iter().any(|label| {
                        label == &n.domain.rack
                            || label == &n.domain.zone
                            || label == &n.domain.region
                    }))
        })
        .collect();
    if eligible.is_empty() {
        return Err(RedundancyError::Placement {
            detail: "no eligible nodes accept new fragments".to_owned(),
        });
    }
    let mut fragments = Vec::with_capacity(total as usize);
    let mut used_keys: HashSet<String> = HashSet::new();
    let mut load: HashMap<NodeId, u32> = HashMap::new();
    for index in 0..total {
        // Rank eligible nodes by deterministic score (tie: node id).
        let mut ranked: Vec<(&NodeDescriptor, u64)> = eligible
            .iter()
            .map(|n| (*n, score(&asset, generation, index, n.id)))
            .collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.id.as_u64().cmp(&b.0.id.as_u64())));
        // Prefer an unused independence key; fall back to least-loaded node
        // when domains are exhausted (collision is recorded in the key and
        // discounted by health accounting).
        let mut chosen: Option<&NodeDescriptor> = None;
        for (node, _) in &ranked {
            let key = node.domain.independence_key(scope);
            if !used_keys.contains(&key) {
                chosen = Some(*node);
                break;
            }
        }
        let pick = chosen.or_else(|| {
            ranked
                .iter()
                .min_by_key(|(n, _)| load.get(&n.id).copied().unwrap_or(0))
                .map(|(n, _)| *n)
        });
        let Some(node) = pick else {
            return Err(RedundancyError::Placement {
                detail: "placement ranking produced no candidate".to_owned(),
            });
        };
        used_keys.insert(node.domain.independence_key(scope));
        *load.entry(node.id).or_insert(0) += 1;
        fragments.push(FragmentPlacement {
            index,
            node: node.id,
            domain_key: node.domain.independence_key(scope),
        });
    }
    fragments.sort_by_key(|f| f.index);
    Ok(DesiredPlacement {
        asset,
        generation,
        scope,
        fragments,
    })
}

/// Plans placement and enforces an explicit locality contract.
#[allow(clippy::too_many_arguments, clippy::missing_errors_doc)]
pub fn plan_placement_with_locality(
    asset: AssetId,
    generation: u64,
    total: u32,
    scope: FailureScope,
    nodes: &[NodeDescriptor],
    allowed: &[String],
    locality: LocalityRequirement,
) -> Result<DesiredPlacement, RedundancyError> {
    let placement = plan_placement(asset, generation, total, scope, nodes, allowed)?;
    let domains = match locality {
        LocalityRequirement::None => return Ok(placement),
        LocalityRequirement::RackSpanning => placement
            .fragments
            .iter()
            .filter_map(|fragment| {
                nodes
                    .iter()
                    .find(|node| node.id == fragment.node)
                    .map(|node| node.domain.rack.clone())
            })
            .collect::<HashSet<_>>(),
        LocalityRequirement::ZoneSpanning => placement
            .fragments
            .iter()
            .filter_map(|fragment| {
                nodes
                    .iter()
                    .find(|node| node.id == fragment.node)
                    .map(|node| node.domain.zone.clone())
            })
            .collect::<HashSet<_>>(),
    };
    if domains.len() < total as usize || domains.iter().any(String::is_empty) {
        return Err(RedundancyError::Placement {
            detail: format!(
                "{locality:?} requires {total} distinct domains, found {}",
                domains.len()
            ),
        });
    }
    Ok(placement)
}

/// Counts how many independent failures a set of healthy fragments survives.
///
/// Only fragments on distinct independence keys count toward protection:
/// `survivable = distinct_healthy_keys.saturating_sub(required)`, capped by
/// the scheme's own tolerance. Collided placements therefore never inflate
/// the answer.
#[allow(clippy::implicit_hasher)]
#[must_use]
pub fn independent_survivable(
    params: SchemeParams,
    scope: FailureScope,
    placement: &DesiredPlacement,
    healthy_indexes: &[u32],
    nodes_by_id: &HashMap<NodeId, crate::NodeDescriptor>,
) -> u32 {
    let _ = scope;
    let mut keys: HashSet<String> = HashSet::new();
    for index in healthy_indexes {
        if let Some(slot) = placement.fragments.iter().find(|f| f.index == *index) {
            let key = nodes_by_id.get(&slot.node).map_or_else(
                || slot.domain_key.clone(),
                |n| n.domain.independence_key(placement.scope),
            );
            keys.insert(key);
        }
    }
    // Distinct healthy keys <= total fragments <=40, fits `u32`.
    #[allow(clippy::cast_possible_truncation)]
    let distinct = keys.len() as u32;
    let required = params.required_pieces();
    let scheme_cap = params.tolerance();
    distinct.saturating_sub(required).min(scheme_cap)
}

/// Derives reconstruction targets for missing indexes: highest-scoring
/// eligible nodes excluding current holders (deterministic, stable).
#[must_use]
pub fn reconstruction_targets(
    asset: AssetId,
    generation: u64,
    missing: &[u32],
    exclude: &[NodeId],
    nodes: &[NodeDescriptor],
) -> ReconstructionTargets {
    let eligible: Vec<&NodeDescriptor> = nodes.iter().filter(|n| n.health.accepts_new()).collect();
    let mut targets = Vec::new();
    for index in missing {
        let mut ranked: Vec<(&NodeDescriptor, u64)> = eligible
            .iter()
            .filter(|n| !exclude.contains(&n.id))
            .map(|n| (*n, score(&asset, generation, *index, n.id)))
            .collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.id.as_u64().cmp(&b.0.id.as_u64())));
        if let Some((node, _)) = ranked.first() {
            targets.push((*index, node.id));
        }
    }
    ReconstructionTargets { targets }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssetKind, NodeHealth};

    fn nodes(count: u64) -> Vec<NodeDescriptor> {
        (1..=count)
            .map(|id| NodeDescriptor {
                id: NodeId::from_u64(id),
                domain: crate::FailureDomain::node_only(NodeId::from_u64(id)),
                health: NodeHealth::Active,
                weight: 1,
            })
            .collect()
    }

    fn asset() -> AssetId {
        AssetId::new(AssetKind::Chunk, 7, [11; 32]).expect("asset")
    }

    #[test]
    fn placement_is_deterministic_and_independent() {
        let cluster = nodes(5);
        let first =
            plan_placement(asset(), 3, 3, FailureScope::Node, &cluster, &[]).expect("plans");
        let second =
            plan_placement(asset(), 3, 3, FailureScope::Node, &cluster, &[]).expect("plans");
        assert_eq!(first, second);
        let mut holders: Vec<NodeId> = first.fragments.iter().map(|f| f.node).collect();
        holders.sort_by_key(|node| node.as_u64());
        holders.dedup_by_key(|node| node.as_u64());
        assert_eq!(holders.len(), 3, "node scope keeps fragments apart");
        // Different generations place differently (fencing domain).
        let other =
            plan_placement(asset(), 4, 3, FailureScope::Node, &cluster, &[]).expect("plans");
        assert_ne!(first.fragments, other.fragments);
    }

    #[test]
    fn locality_contract_rejects_domain_collisions() {
        let cluster = vec![
            NodeDescriptor {
                id: NodeId::from_u64(1),
                domain: crate::FailureDomain::full(
                    NodeId::from_u64(1),
                    "rack-a".to_owned(),
                    "zone-a".to_owned(),
                    String::new(),
                ),
                health: NodeHealth::Active,
                weight: 1,
            },
            NodeDescriptor {
                id: NodeId::from_u64(2),
                domain: crate::FailureDomain::full(
                    NodeId::from_u64(2),
                    "rack-a".to_owned(),
                    "zone-a".to_owned(),
                    String::new(),
                ),
                health: NodeHealth::Active,
                weight: 1,
            },
        ];
        assert!(
            plan_placement_with_locality(
                asset(),
                1,
                2,
                FailureScope::Rack,
                &cluster,
                &[],
                LocalityRequirement::RackSpanning,
            )
            .is_err()
        );
    }

    #[test]
    fn collided_domains_never_overcount_protection() {
        // Two nodes, three fragments: one domain must collide.
        let cluster = nodes(2);
        let placement =
            plan_placement(asset(), 1, 3, FailureScope::Node, &cluster, &[]).expect("plans");
        let by_id: HashMap<NodeId, NodeDescriptor> =
            cluster.into_iter().map(|n| (n.id, n)).collect();
        let params = SchemeParams::Replication(crate::ReplicationParams { copies: 3 });
        // All three healthy but only two distinct nodes: survivable = 1, not 2.
        let survivable =
            independent_survivable(params, FailureScope::Node, &placement, &[0, 1, 2], &by_id);
        assert_eq!(survivable, 1);
    }

    #[test]
    fn no_eligible_nodes_fails_closed() {
        let mut cluster = nodes(2);
        for node in &mut cluster {
            node.health = NodeHealth::Unavailable;
        }
        assert!(plan_placement(asset(), 1, 2, FailureScope::Node, &cluster, &[]).is_err());
    }
}
