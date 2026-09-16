//! In-memory control state plus canonical snapshot encoding.
//!
//! The control state machine applies [`ControlMutation`]s in log order.
//! Snapshots persist the full registry/placement/plan image with the same
//! canonical little-endian discipline as every other Kivi durable format.

use std::collections::{BTreeMap, BTreeSet};

use kivi_types::{NodeId, TabletId};

use crate::migration::{MigrationPlan, MigrationPlanId};
use crate::mutation::{ControlMutation, MigrationPlanIdAlias};
use crate::node::{NodeRecord, NodeState};
use crate::placement::{DesiredReplicaSet, PlacementVersion};

/// Monotonic cluster generation: every control-membership or
/// placement-epoch transition advances it. Stale create/remove/complete
/// messages carrying an older generation never mutate newer state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClusterGeneration(pub u64);

impl ClusterGeneration {
    /// First generation, assigned at cluster bootstrap.
    pub const INITIAL: Self = Self(1);

    /// Wraps a raw generation value.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// Why a control mutation was refused at apply time.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ControlApplyError {
    /// A node-state transition is not on the lifecycle path.
    #[error("illegal node transition for {node}: {detail}")]
    IllegalTransition {
        /// Affected node.
        node: u64,
        /// Human-readable cause.
        detail: String,
    },
    /// The named node is unknown to the registry.
    #[error("unknown node {node}")]
    UnknownNode {
        /// Missing node.
        node: u64,
    },
    /// A stale generation-fenced action was rejected (safe no-op, never a
    /// corruption: the caller's generation predates current placement).
    #[error("stale generation: plan {plan} carries {carried}, current is {current}")]
    StaleGeneration {
        /// Affected plan.
        plan: u64,
        /// Generation the writer observed.
        carried: u64,
        /// Current placement generation.
        current: u64,
    },
    /// The named plan is unknown.
    #[error("unknown migration plan {plan}")]
    UnknownPlan {
        /// Missing plan.
        plan: u64,
    },
}

/// Why a control snapshot image was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ControlSnapshotError {
    /// Truncated image where framing promised bytes.
    #[error("truncated control snapshot: {detail}")]
    Truncated {
        /// What was missing.
        detail: &'static str,
    },
    /// Trailing bytes after the framed image.
    #[error("trailing bytes in control snapshot")]
    TrailingBytes,
    /// An embedded node record failed to decode.
    #[error("undecodable node record: {0}")]
    BadNode(#[from] crate::node::NodeError),
    /// An embedded desired placement failed to decode.
    #[error("undecodable desired placement: {0}")]
    BadPlacement(#[from] crate::placement::PlacementError),
    /// An embedded migration plan failed to decode.
    #[error("undecodable migration plan: {0}")]
    BadMigration(#[from] crate::migration::MigrationError),
}

/// Authoritative replicated control image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlState {
    /// Admitted nodes by id.
    nodes: BTreeMap<u64, NodeRecord>,
    /// Desired voter set per tablet.
    placements: BTreeMap<u64, DesiredReplicaSet>,
    /// Live migration plans by id.
    migrations: BTreeMap<u64, MigrationPlan>,
    /// Current placement version (advanced by every desired-placement
    /// change and every plan creation).
    placement_version: PlacementVersion,
    /// Current cluster generation.
    generation: ClusterGeneration,
    /// Next plan id to assign.
    next_plan_id: u64,
}

impl ControlState {
    /// Empty control image at initial versions.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            nodes: BTreeMap::new(),
            placements: BTreeMap::new(),
            migrations: BTreeMap::new(),
            placement_version: PlacementVersion::INITIAL,
            generation: ClusterGeneration::INITIAL,
            next_plan_id: 1,
        }
    }

    /// Bootstraps the initial image: nodes plus per-tablet desired sets.
    #[must_use]
    pub fn bootstrap(nodes: Vec<NodeRecord>, placements: Vec<DesiredReplicaSet>) -> Self {
        let mut state = Self::empty();
        for record in nodes {
            state.nodes.insert(record.node.as_u64(), record);
        }
        for desired in placements {
            state.placements.insert(desired.tablet.as_u64(), desired);
        }
        state
    }

    /// Returns the current placement version.
    #[must_use]
    pub const fn placement_version(&self) -> PlacementVersion {
        self.placement_version
    }

    /// Returns the current cluster generation.
    #[must_use]
    pub const fn generation(&self) -> ClusterGeneration {
        self.generation
    }

    /// Returns the next plan id without assigning it.
    #[must_use]
    pub const fn next_plan_id(&self) -> MigrationPlanId {
        MigrationPlanId::from_u64(self.next_plan_id)
    }

    /// Looks up one node.
    #[must_use]
    pub fn node(&self, node: NodeId) -> Option<&NodeRecord> {
        self.nodes.get(&node.as_u64())
    }

    /// Iterates all nodes in id order.
    pub fn nodes(&self) -> impl Iterator<Item = &NodeRecord> {
        self.nodes.values()
    }

    /// Looks up one tablet's desired set.
    #[must_use]
    pub fn desired(&self, tablet: TabletId) -> Option<&DesiredReplicaSet> {
        self.placements.get(&tablet.as_u64())
    }

    /// Iterates all desired sets in tablet order.
    pub fn placements(&self) -> impl Iterator<Item = &DesiredReplicaSet> {
        self.placements.values()
    }

    /// Looks up one migration plan.
    #[must_use]
    pub fn migration(&self, id: MigrationPlanId) -> Option<&MigrationPlan> {
        self.migrations.get(&id.as_u64())
    }

    /// Iterates all plans in id order.
    pub fn migrations(&self) -> impl Iterator<Item = &MigrationPlan> {
        self.migrations.values()
    }

    /// Live (non-terminal) plans for one tablet.
    #[must_use]
    pub fn live_plans_for(&self, tablet: TabletId) -> Vec<&MigrationPlan> {
        self.migrations
            .values()
            .filter(|plan| plan.tablet == tablet && !plan.phase.is_terminal())
            .collect()
    }

    /// Applies one committed control mutation in log order.
    ///
    /// # Errors
    ///
    /// Returns [`ControlApplyError`] on illegal lifecycle transitions,
    /// unknown nodes/plans, or stale fenced generations. Stale
    /// generations are safe rejections, never corruption.
    pub fn apply(&mut self, mutation: &ControlMutation) -> Result<(), ControlApplyError> {
        use ControlApplyError as Fault;
        match mutation {
            ControlMutation::RegisterNode { record } => {
                self.nodes
                    .entry(record.node.as_u64())
                    .or_insert_with(|| record.clone());
                Ok(())
            }
            ControlMutation::SetNodeState { node, state } => {
                let Some(record) = self.nodes.get_mut(&node.as_u64()) else {
                    return Err(Fault::UnknownNode {
                        node: node.as_u64(),
                    });
                };
                check_transition(record.state, *state, node.as_u64())?;
                record.state = *state;
                Ok(())
            }
            ControlMutation::SetDesiredPlacement { desired } => {
                self.placements
                    .insert(desired.tablet.as_u64(), desired.clone());
                self.placement_version = PlacementVersion::from_u64(
                    self.placement_version
                        .as_u64()
                        .max(desired.version.as_u64()),
                );
                Ok(())
            }
            ControlMutation::CreateMigration { plan } => {
                self.migrations.insert(plan.id.as_u64(), plan.clone());
                self.placement_version = PlacementVersion::from_u64(
                    self.placement_version
                        .as_u64()
                        .max(plan.generation.as_u64()),
                );
                self.next_plan_id = self.next_plan_id.max(plan.id.as_u64() + 1);
                Ok(())
            }
            ControlMutation::AdvanceMigration {
                plan,
                phase,
                generation,
            } => {
                let id = MigrationPlanId::from_u64(plan.as_u64());
                let Some(existing) = self.migrations.get(&id.as_u64()).cloned() else {
                    return Err(Fault::UnknownPlan { plan: id.as_u64() });
                };
                if *generation != existing.generation {
                    return Err(Fault::StaleGeneration {
                        plan: id.as_u64(),
                        carried: generation.as_u64(),
                        current: existing.generation.as_u64(),
                    });
                }
                let mut advanced = existing.advance(*phase);
                let _ = MigrationPlanIdAlias::from_u64(0);
                advanced.phase = *phase;
                self.migrations.insert(id.as_u64(), advanced);
                Ok(())
            }
            ControlMutation::RemoveMigration { plan } => {
                let id = MigrationPlanId::from_u64(plan.as_u64());
                let Some(existing) = self.migrations.get(&id.as_u64()) else {
                    return Err(Fault::UnknownPlan { plan: id.as_u64() });
                };
                if !existing.phase.is_terminal() {
                    return Err(Fault::IllegalTransition {
                        node: id.as_u64(),
                        detail: "only terminal plans may be removed".to_owned(),
                    });
                }
                self.migrations.remove(&id.as_u64());
                Ok(())
            }
        }
    }

    /// Returns ids of tablets whose desired set names `node`.
    #[must_use]
    pub fn tablets_desiring(&self, node: NodeId) -> Vec<TabletId> {
        self.placements
            .values()
            .filter(|desired| desired.contains(node))
            .map(|desired| desired.tablet)
            .collect()
    }

    /// Active (placement-eligible) node ids in order.
    #[must_use]
    pub fn active_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .values()
            .filter(|record| record.state == NodeState::Active)
            .map(|record| record.node)
            .collect()
    }

    /// Whether `node` may be removed safely: drained and desired by no
    /// tablet. Actual voter membership is checked by the reconciler
    /// against live Raft state, not here.
    #[must_use]
    pub fn removable(&self, node: NodeId) -> bool {
        let drained = self
            .node(node)
            .is_some_and(|record| record.state == NodeState::Drained);
        drained && self.tablets_desiring(node).is_empty()
    }

    /// Encodes a canonical snapshot image.
    #[must_use]
    pub fn encode_snapshot(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.placement_version.as_u64().to_le_bytes());
        out.extend_from_slice(&self.generation.as_u64().to_le_bytes());
        out.extend_from_slice(&self.next_plan_id.to_le_bytes());
        push_map(&mut out, self.nodes.values().map(NodeRecord::encode_to_vec));
        push_map(
            &mut out,
            self.placements
                .values()
                .map(DesiredReplicaSet::encode_to_vec),
        );
        push_map(
            &mut out,
            self.migrations.values().map(MigrationPlan::encode_to_vec),
        );
        out
    }

    /// Decodes a canonical snapshot image.
    ///
    /// # Errors
    ///
    /// Returns [`ControlSnapshotError`] on structural failure.
    pub fn decode_snapshot(input: &[u8]) -> Result<Self, ControlSnapshotError> {
        use ControlSnapshotError as Fault;
        if input.len() < 24 {
            return Err(Fault::Truncated { detail: "header" });
        }
        let placement_version =
            PlacementVersion::from_u64(u64::from_le_bytes(input[..8].try_into().unwrap_or([0; 8])));
        let generation = ClusterGeneration::from_u64(u64::from_le_bytes(
            input[8..16].try_into().unwrap_or([0; 8]),
        ));
        let next_plan_id = u64::from_le_bytes(input[16..24].try_into().unwrap_or([0; 8]));
        let mut rest = &input[24..];
        let mut nodes = BTreeMap::new();
        let mut placements = BTreeMap::new();
        let mut migrations = BTreeMap::new();
        let node_blobs = take_map(rest).map_err(|detail| Fault::Truncated { detail })?;
        rest = node_blobs.1;
        for blob in node_blobs.0 {
            let record = NodeRecord::decode_exact(blob)?;
            nodes.insert(record.node.as_u64(), record);
        }
        let placement_blobs = take_map(rest).map_err(|detail| Fault::Truncated { detail })?;
        rest = placement_blobs.1;
        for blob in placement_blobs.0 {
            let desired = DesiredReplicaSet::decode_exact(blob)?;
            placements.insert(desired.tablet.as_u64(), desired);
        }
        let migration_blobs = take_map(rest).map_err(|detail| Fault::Truncated { detail })?;
        rest = migration_blobs.1;
        for blob in migration_blobs.0 {
            let plan = MigrationPlan::decode_exact(blob)?;
            migrations.insert(plan.id.as_u64(), plan);
        }
        if !rest.is_empty() {
            return Err(Fault::TrailingBytes);
        }
        Ok(Self {
            nodes,
            placements,
            migrations,
            placement_version,
            generation,
            next_plan_id: next_plan_id.max(1),
        })
    }
}

/// Validates one lifecycle transition.
fn check_transition(from: NodeState, to: NodeState, node: u64) -> Result<(), ControlApplyError> {
    use ControlApplyError as Fault;
    use NodeState as State;
    if from == to {
        return Ok(());
    }
    let legal = match from {
        State::Joining => matches!(to, State::Active | State::Removed),
        State::Active => matches!(to, State::Draining | State::Removed),
        State::Draining => matches!(to, State::Drained | State::Active),
        State::Drained => matches!(to, State::Removed | State::Active),
        State::Removed => false,
    };
    if legal {
        Ok(())
    } else {
        Err(Fault::IllegalTransition {
            node,
            detail: format!("cannot move {from:?} to {to:?}"),
        })
    }
}

fn push_map(out: &mut Vec<u8>, blobs: impl Iterator<Item = Vec<u8>>) {
    let blobs: Vec<Vec<u8>> = blobs.collect();
    let count = u32::try_from(blobs.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&count.to_le_bytes());
    for blob in &blobs {
        let len = u32::try_from(blob.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(blob);
    }
}

fn take_map(mut input: &[u8]) -> Result<(Vec<&[u8]>, &[u8]), &'static str> {
    if input.len() < 4 {
        return Err("truncated control snapshot map");
    }
    let count = u32::from_le_bytes(input[..4].try_into().unwrap_or([0; 4])) as usize;
    input = &input[4..];
    let mut out = Vec::with_capacity(count.min(1024));
    for _ in 0..count {
        if input.len() < 4 {
            return Err("truncated control snapshot entry");
        }
        let len = u32::from_le_bytes(input[..4].try_into().unwrap_or([0; 4])) as usize;
        input = &input[4..];
        if input.len() < len {
            return Err("truncated control snapshot entry body");
        }
        out.push(&input[..len]);
        input = &input[len..];
    }
    Ok((out, input))
}

/// Returns the set of tablet ids in a desired map (test/diagnostic use).
#[must_use]
pub fn tablet_set(state: &ControlState) -> BTreeSet<TabletId> {
    state.placements().map(|desired| desired.tablet).collect()
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use super::*;
    use crate::migration::MigrationPhase;
    use crate::node::NodeState;
    use crate::placement::PlacementVersion;

    fn record(id: u64, state: NodeState) -> NodeRecord {
        NodeRecord {
            node: NodeId::from_u64(id),
            peer: format!("127.0.0.1:{}", 9100 + id)
                .parse::<SocketAddr>()
                .expect("peer"),
            native: format!("127.0.0.1:{}", 9000 + id)
                .parse::<SocketAddr>()
                .expect("native"),
            admin: format!("127.0.0.1:{}", 19000 + id)
                .parse::<SocketAddr>()
                .expect("admin"),
            cert_fingerprint: [u8::try_from(id % 256).unwrap_or(0xFF); 32],
            failure_domain: String::new(),
            weight: 1,
            state,
        }
    }

    #[test]
    fn lifecycle_enforced() {
        let mut state = ControlState::empty();
        state
            .apply(&ControlMutation::RegisterNode {
                record: record(4, NodeState::Joining),
            })
            .expect("register");
        // Joining -> Drained is illegal.
        assert!(matches!(
            state.apply(&ControlMutation::SetNodeState {
                node: NodeId::from_u64(4),
                state: NodeState::Drained,
            }),
            Err(ControlApplyError::IllegalTransition { .. })
        ));
        state
            .apply(&ControlMutation::SetNodeState {
                node: NodeId::from_u64(4),
                state: NodeState::Active,
            })
            .expect("activate");
        // Unknown node fails.
        assert!(matches!(
            state.apply(&ControlMutation::SetNodeState {
                node: NodeId::from_u64(9),
                state: NodeState::Active,
            }),
            Err(ControlApplyError::UnknownNode { .. })
        ));
    }

    #[test]
    fn stale_generation_rejected() {
        use crate::migration::MigrationPlanId;
        let mut state = ControlState::empty();
        let plan = MigrationPlan::new(
            MigrationPlanId::from_u64(1),
            TabletId::from_u64(1),
            PlacementVersion::from_u64(5),
            NodeId::from_u64(1),
            NodeId::from_u64(4),
            vec![
                NodeId::from_u64(2),
                NodeId::from_u64(3),
                NodeId::from_u64(4),
            ],
        )
        .expect("plan");
        state
            .apply(&ControlMutation::CreateMigration { plan })
            .expect("create");
        assert!(matches!(
            state.apply(&ControlMutation::AdvanceMigration {
                plan: MigrationPlanIdAlias::from_u64(1),
                phase: MigrationPhase::Ready,
                generation: PlacementVersion::from_u64(4),
            }),
            Err(ControlApplyError::StaleGeneration { .. })
        ));
    }

    #[test]
    fn snapshot_round_trips() {
        let mut state = ControlState::bootstrap(
            vec![record(1, NodeState::Active), record(4, NodeState::Joining)],
            vec![],
        );
        let desired = DesiredReplicaSet::new(
            TabletId::from_u64(7),
            vec![
                NodeId::from_u64(1),
                NodeId::from_u64(2),
                NodeId::from_u64(3),
            ],
            PlacementVersion::INITIAL,
        )
        .expect("desired");
        state
            .apply(&ControlMutation::SetDesiredPlacement { desired })
            .expect("placement");
        let back = ControlState::decode_snapshot(&state.encode_snapshot()).expect("decodes");
        assert_eq!(back, state);
    }
}
