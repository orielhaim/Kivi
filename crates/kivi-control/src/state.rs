//! In-memory control state plus canonical snapshot encoding.
//!
//! The control state machine applies [`ControlMutation`]s in log order.
//! Snapshots persist the full registry/placement/plan image with the same
//! canonical little-endian discipline as every other Kivi durable format.

use std::collections::{BTreeMap, BTreeSet};

use kivi_types::{NodeId, TabletId};

use crate::merge::{MergePlan, MergePlanId};
use crate::migration::{MigrationPlan, MigrationPlanId};
use crate::mutation::{ControlMutation, MergePlanIdAlias, MigrationPlanIdAlias, SplitPlanIdAlias};
use crate::node::{NodeRecord, NodeState};
use crate::placement::{DesiredReplicaSet, PlacementVersion};
use crate::split::{SplitPlan, SplitPlanId};

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
    /// The named split plan is unknown.
    #[error("unknown split plan {plan}")]
    UnknownSplit {
        /// Missing plan.
        plan: u64,
    },
    /// The named merge plan is unknown.
    #[error("unknown merge plan {plan}")]
    UnknownMerge {
        /// Missing plan.
        plan: u64,
    },
    /// A topology plan conflicts with a live plan on the same tablet
    /// (one tablet, one live plan).
    #[error("topology conflict on tablet {tablet}: {detail}")]
    TopologyConflict {
        /// Affected tablet.
        tablet: u64,
        /// Human-readable cause.
        detail: String,
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
    /// An embedded split plan failed to decode.
    #[error("undecodable split plan: {0}")]
    BadSplit(#[from] crate::split::SplitError),
    /// An embedded merge plan failed to decode.
    #[error("undecodable merge plan: {0}")]
    BadMerge(#[from] crate::merge::MergeError),
}

/// Authoritative replicated control image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlState {
    /// Admitted nodes by id.
    nodes: BTreeMap<u64, NodeRecord>,
    /// Desired voter set per tablet.
    placements: BTreeMap<u64, DesiredReplicaSet>,
    /// Migration plans by id (live + terminal until removed).
    migrations: BTreeMap<u64, MigrationPlan>,
    /// Split plans by id (shares the plan-id sequence with migrations).
    splits: BTreeMap<u64, SplitPlan>,
    /// Merge plans by id (shares the plan-id sequence with migrations).
    merges: BTreeMap<u64, MergePlan>,
    /// Current placement version (advanced by every desired-placement
    /// change and every plan creation).
    placement_version: PlacementVersion,
    /// Current cluster generation.
    generation: ClusterGeneration,
    /// Next plan id to assign (unique across migrations/splits/merges).
    next_plan_id: u64,
    /// Next tablet id to allocate for split children / merge targets.
    /// Initialized to one past the genesis tablet count; persisted so a
    /// control-leader restart never reallocates a child id.
    next_tablet_id: u64,
}

impl ControlState {
    /// Empty control image at initial versions.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            nodes: BTreeMap::new(),
            placements: BTreeMap::new(),
            migrations: BTreeMap::new(),
            splits: BTreeMap::new(),
            merges: BTreeMap::new(),
            placement_version: PlacementVersion::INITIAL,
            generation: ClusterGeneration::INITIAL,
            next_plan_id: 1,
            next_tablet_id: 1,
        }
    }

    /// Bootstraps the initial image: nodes plus per-tablet desired sets.
    /// The tablet allocator starts past the highest genesis tablet id so
    /// dynamic children never collide with static tiles.
    #[must_use]
    pub fn bootstrap(nodes: Vec<NodeRecord>, placements: Vec<DesiredReplicaSet>) -> Self {
        let mut state = Self::empty();
        for record in nodes {
            state.nodes.insert(record.node.as_u64(), record);
        }
        let mut highest_tablet = 0u64;
        for desired in placements {
            highest_tablet = highest_tablet.max(desired.tablet.as_u64());
            state.placements.insert(desired.tablet.as_u64(), desired);
        }
        state.next_tablet_id = highest_tablet.saturating_add(1).max(1);
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

    /// Returns the next tablet id for dynamic allocation without
    /// assigning it.
    #[must_use]
    pub const fn next_tablet_id(&self) -> TabletId {
        TabletId::from_u64(self.next_tablet_id)
    }

    /// Allocates `count` consecutive tablet ids, advancing the persisted
    /// allocator. Returns the first id; callers derive the rest.
    pub fn allocate_tablet_ids(&mut self, count: u64) -> TabletId {
        let first = self.next_tablet_id.max(1);
        self.next_tablet_id = first.saturating_add(count.max(1)).max(1);
        TabletId::from_u64(first)
    }

    /// Fresh tablet ids for plan creation (read-only): the persisted
    /// allocator floored by one past the highest placed tablet, so plans
    /// created against pre-allocator images (genesis placed tiles without
    /// advancing it) still avoid collisions. `CreateSplit`/`CreateMerge`
    /// advance the persisted allocator on commit.
    #[must_use]
    pub fn fresh_tablet_ids(&self, count: u64) -> Vec<TabletId> {
        let highest_placed = self.placements.keys().copied().max().unwrap_or(0);
        let first = self
            .next_tablet_id
            .max(highest_placed.saturating_add(1))
            .max(1);
        (0..count.max(1))
            .map(|offset| TabletId::from_u64(first.saturating_add(offset)))
            .collect()
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

    /// Looks up one split plan.
    #[must_use]
    pub fn split(&self, id: SplitPlanId) -> Option<&SplitPlan> {
        self.splits.get(&id.as_u64())
    }

    /// Iterates all split plans in id order.
    pub fn splits(&self) -> impl Iterator<Item = &SplitPlan> {
        self.splits.values()
    }

    /// Live split plans touching `tablet` (parent or child).
    #[must_use]
    pub fn splits_for(&self, tablet: TabletId) -> Vec<&SplitPlan> {
        self.splits
            .values()
            .filter(|plan| !plan.phase.is_terminal() && plan.tablets().contains(&tablet))
            .collect()
    }

    /// Looks up one merge plan.
    #[must_use]
    pub fn merge(&self, id: MergePlanId) -> Option<&MergePlan> {
        self.merges.get(&id.as_u64())
    }

    /// Iterates all merge plans in id order.
    pub fn merges(&self) -> impl Iterator<Item = &MergePlan> {
        self.merges.values()
    }

    /// Live merge plans touching `tablet` (parent or merged target).
    #[must_use]
    pub fn merges_for(&self, tablet: TabletId) -> Vec<&MergePlan> {
        self.merges
            .values()
            .filter(|plan| !plan.phase.is_terminal() && plan.tablets().contains(&tablet))
            .collect()
    }

    /// Whether any live topology plan (migration, split, or merge)
    /// touches `tablet`: plan conflict rule — one tablet, one live plan.
    /// Repair wins over split/merge: the planner refuses to split or merge
    /// degraded tablets (see planner eligibility).
    #[must_use]
    pub fn has_live_topology(&self, tablet: TabletId) -> bool {
        !self.live_plans_for(tablet).is_empty()
            || !self.splits_for(tablet).is_empty()
            || !self.merges_for(tablet).is_empty()
    }

    /// Applies one committed control mutation in log order.
    ///
    /// # Errors
    ///
    /// Returns [`ControlApplyError`] on illegal lifecycle transitions,
    /// unknown nodes/plans, or stale fenced generations. Stale
    /// generations are safe rejections, never corruption.
    #[allow(clippy::too_many_lines)]
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
                // Keep the dynamic tablet allocator past every placed
                // tablet (genesis places static tiles through this path,
                // so split children never collide with them even though
                // genesis never explicitly allocates).
                self.next_tablet_id = self
                    .next_tablet_id
                    .max(desired.tablet.as_u64().saturating_add(1));
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
            ControlMutation::CreateSplit { plan } => {
                // Conflict rule: one tablet, one live plan.
                for tablet in plan.tablets() {
                    if self.has_live_topology(tablet) {
                        return Err(Fault::TopologyConflict {
                            tablet: tablet.as_u64(),
                            detail: "tablet already has a live topology plan".to_owned(),
                        });
                    }
                }
                self.splits.insert(plan.id.as_u64(), plan.clone());
                self.placement_version = PlacementVersion::from_u64(
                    self.placement_version
                        .as_u64()
                        .max(plan.generation.as_u64()),
                );
                self.next_plan_id = self.next_plan_id.max(plan.id.as_u64() + 1);
                self.next_tablet_id = self
                    .next_tablet_id
                    .max(plan.left.as_u64().saturating_add(1))
                    .max(plan.right.as_u64().saturating_add(1));
                // Child placements inherit the parent replica set at the
                // plan generation (logical split first; physical rebalance
                // follows as ordinary tablets).
                for child in [plan.left, plan.right] {
                    if let Ok(desired) =
                        DesiredReplicaSet::new(child, plan.replicas.clone(), plan.generation)
                    {
                        self.placements.insert(child.as_u64(), desired);
                    }
                }
                Ok(())
            }
            ControlMutation::AdvanceSplit {
                plan,
                phase,
                generation,
            } => {
                let id = SplitPlanId::from_u64(plan.as_u64());
                let Some(existing) = self.splits.get(&id.as_u64()).cloned() else {
                    return Err(Fault::UnknownSplit { plan: id.as_u64() });
                };
                if *generation != existing.generation {
                    return Err(Fault::StaleGeneration {
                        plan: id.as_u64(),
                        carried: generation.as_u64(),
                        current: existing.generation.as_u64(),
                    });
                }
                let mut advanced = existing.advance(*phase);
                let _ = SplitPlanIdAlias::from_u64(0);
                advanced.phase = *phase;
                self.splits.insert(id.as_u64(), advanced);
                Ok(())
            }
            ControlMutation::RemoveSplit { plan } => {
                let id = SplitPlanId::from_u64(plan.as_u64());
                let Some(existing) = self.splits.get(&id.as_u64()) else {
                    return Err(Fault::UnknownSplit { plan: id.as_u64() });
                };
                if !existing.phase.is_terminal() {
                    return Err(Fault::IllegalTransition {
                        node: id.as_u64(),
                        detail: "only terminal plans may be removed".to_owned(),
                    });
                }
                self.splits.remove(&id.as_u64());
                Ok(())
            }
            ControlMutation::CreateMerge { plan } => {
                for tablet in plan.tablets() {
                    if self.has_live_topology(tablet) {
                        return Err(Fault::TopologyConflict {
                            tablet: tablet.as_u64(),
                            detail: "tablet already has a live topology plan".to_owned(),
                        });
                    }
                }
                self.merges.insert(plan.id.as_u64(), plan.clone());
                self.placement_version = PlacementVersion::from_u64(
                    self.placement_version
                        .as_u64()
                        .max(plan.generation.as_u64()),
                );
                self.next_plan_id = self.next_plan_id.max(plan.id.as_u64() + 1);
                self.next_tablet_id = self
                    .next_tablet_id
                    .max(plan.merged.as_u64().saturating_add(1));
                if let Ok(desired) =
                    DesiredReplicaSet::new(plan.merged, plan.replicas.clone(), plan.generation)
                {
                    self.placements.insert(plan.merged.as_u64(), desired);
                }
                Ok(())
            }
            ControlMutation::AdvanceMerge {
                plan,
                phase,
                generation,
            } => {
                let id = MergePlanId::from_u64(plan.as_u64());
                let Some(existing) = self.merges.get(&id.as_u64()).cloned() else {
                    return Err(Fault::UnknownMerge { plan: id.as_u64() });
                };
                if *generation != existing.generation {
                    return Err(Fault::StaleGeneration {
                        plan: id.as_u64(),
                        carried: generation.as_u64(),
                        current: existing.generation.as_u64(),
                    });
                }
                let mut advanced = existing.advance(*phase);
                let _ = MergePlanIdAlias::from_u64(0);
                advanced.phase = *phase;
                self.merges.insert(id.as_u64(), advanced);
                Ok(())
            }
            ControlMutation::RemoveMerge { plan } => {
                let id = MergePlanId::from_u64(plan.as_u64());
                let Some(existing) = self.merges.get(&id.as_u64()) else {
                    return Err(Fault::UnknownMerge { plan: id.as_u64() });
                };
                if !existing.phase.is_terminal() {
                    return Err(Fault::IllegalTransition {
                        node: id.as_u64(),
                        detail: "only terminal plans may be removed".to_owned(),
                    });
                }
                self.merges.remove(&id.as_u64());
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

    /// Encodes a canonical snapshot image (v2: splits, merges, tablet
    /// allocator; no legacy v1 compatibility — see stage notes).
    #[must_use]
    pub fn encode_snapshot(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.placement_version.as_u64().to_le_bytes());
        out.extend_from_slice(&self.generation.as_u64().to_le_bytes());
        out.extend_from_slice(&self.next_plan_id.to_le_bytes());
        out.extend_from_slice(&self.next_tablet_id.to_le_bytes());
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
        push_map(&mut out, self.splits.values().map(SplitPlan::encode_to_vec));
        push_map(&mut out, self.merges.values().map(MergePlan::encode_to_vec));
        out
    }

    /// Decodes a canonical snapshot image.
    ///
    /// # Errors
    ///
    /// Returns [`ControlSnapshotError`] on structural failure.
    pub fn decode_snapshot(input: &[u8]) -> Result<Self, ControlSnapshotError> {
        use ControlSnapshotError as Fault;
        if input.len() < 32 {
            return Err(Fault::Truncated { detail: "header" });
        }
        let placement_version =
            PlacementVersion::from_u64(u64::from_le_bytes(input[..8].try_into().unwrap_or([0; 8])));
        let generation = ClusterGeneration::from_u64(u64::from_le_bytes(
            input[8..16].try_into().unwrap_or([0; 8]),
        ));
        let next_plan_id = u64::from_le_bytes(input[16..24].try_into().unwrap_or([0; 8]));
        let next_tablet_id = u64::from_le_bytes(input[24..32].try_into().unwrap_or([0; 8]));
        let mut rest = &input[32..];
        let mut nodes = BTreeMap::new();
        let mut placements = BTreeMap::new();
        let mut migrations = BTreeMap::new();
        let mut splits = BTreeMap::new();
        let mut merges = BTreeMap::new();
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
        let split_blobs = take_map(rest).map_err(|detail| Fault::Truncated { detail })?;
        rest = split_blobs.1;
        for blob in split_blobs.0 {
            let plan = SplitPlan::decode_exact(blob)?;
            splits.insert(plan.id.as_u64(), plan);
        }
        let merge_blobs = take_map(rest).map_err(|detail| Fault::Truncated { detail })?;
        rest = merge_blobs.1;
        for blob in merge_blobs.0 {
            let plan = MergePlan::decode_exact(blob)?;
            merges.insert(plan.id.as_u64(), plan);
        }
        if !rest.is_empty() {
            return Err(Fault::TrailingBytes);
        }
        Ok(Self {
            nodes,
            placements,
            splits,
            merges,
            migrations,
            placement_version,
            generation,
            next_plan_id: next_plan_id.max(1),
            next_tablet_id: next_tablet_id.max(1),
        })
    }
}

/// Validates one lifecycle transition, including the liveness path
/// (`Active <-> Suspect -> Unavailable -> Active`) which is driven by the
/// failure detector (ephemeral observations) through replicated
/// `SetNodeState` commits (never by the detector alone).
fn check_transition(from: NodeState, to: NodeState, node: u64) -> Result<(), ControlApplyError> {
    use ControlApplyError as Fault;
    use NodeState as State;
    if from == to {
        return Ok(());
    }
    let legal = match from {
        State::Joining => matches!(to, State::Active | State::Removed),
        State::Active => matches!(
            to,
            State::Suspect | State::Unavailable | State::Draining | State::Removed
        ),
        State::Suspect => matches!(
            to,
            State::Active | State::Unavailable | State::Draining | State::Removed
        ),
        State::Unavailable => matches!(to, State::Active | State::Draining | State::Removed),
        State::Draining => matches!(to, State::Drained | State::Active | State::Removed),
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
