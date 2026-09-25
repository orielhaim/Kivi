//! Typed control-plane mutations.
//!
//! The control plane is never arbitrary string KV entries. Every change is
//! a project-owned [`ControlMutation`] with canonical encoding;
//! `OpenRaft` Rust structs are never persisted as canonical control
//! state.

use kivi_types::NodeId;

use crate::layouts::{LAYOUT_KEY_LEN, LayoutKey, LayoutRecord};
use crate::merge::MergePlan;
use crate::migration::MigrationPlan;
use crate::node::NodeRecord;
use crate::placement::{DesiredReplicaSet, PlacementVersion};
use crate::split::SplitPlan;
use crate::topology::PlanId;

/// Framing version of [`ControlMutation`].
pub const CONTROL_MUTATION_VERSION: u16 = 1;

/// Typed control-plane change: the only writes the control group commits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlMutation {
    /// Admit a node (starts as `Joining`; promotion to `Active` is an
    /// explicit [`ControlMutation::SetNodeState`]).
    RegisterNode {
        /// Admitted record.
        record: NodeRecord,
    },
    /// Move one node through its lifecycle.
    SetNodeState {
        /// Affected node.
        node: NodeId,
        /// New state.
        state: crate::node::NodeState,
    },
    /// Publish a tablet's desired voter set (bumps placement version).
    SetDesiredPlacement {
        /// Desired set.
        desired: DesiredReplicaSet,
    },
    /// Publish a migration plan and its desired placement atomically (phase `Planned`).
    CreateMigration {
        /// Plan to persist.
        plan: MigrationPlan,
    },
    /// Advance a plan's phase (generation-fenced: stale advances are
    /// rejected by [`ControlState`](crate::state::ControlState)).
    AdvanceMigration {
        /// Affected plan.
        plan: PlanId,
        /// New phase.
        phase: crate::migration::MigrationPhase,
        /// Fencing generation the writer observed.
        generation: PlacementVersion,
    },
    /// Remove a terminally completed/failed plan.
    RemoveMigration {
        /// Affected plan.
        plan: PlanId,
    },
    /// Persist a new split plan (phase `Planned`).
    CreateSplit {
        /// Plan to persist.
        plan: SplitPlan,
    },
    /// Advance a split plan's phase (generation-fenced).
    AdvanceSplit {
        /// Affected plan.
        plan: PlanId,
        /// New phase.
        phase: crate::split::SplitPhase,
        /// Fencing generation the writer observed.
        generation: PlacementVersion,
    },
    /// Remove a terminal split plan.
    RemoveSplit {
        /// Affected plan.
        plan: PlanId,
    },
    /// Persist a new merge plan (phase `Planned`).
    CreateMerge {
        /// Plan to persist.
        plan: MergePlan,
    },
    /// Advance a merge plan's phase (generation-fenced).
    AdvanceMerge {
        /// Affected plan.
        plan: PlanId,
        /// New phase.
        phase: crate::merge::MergePhase,
        /// Fencing generation the writer observed.
        generation: PlacementVersion,
    },
    /// Remove a terminal merge plan.
    RemoveMerge {
        /// Affected plan.
        plan: PlanId,
    },
    /// Register a namespace (idempotent: re-registering the identical
    /// record is a no-op; a conflicting layout is rejected).
    RegisterNamespace {
        /// Namespace record.
        record: crate::catalog::NamespaceRecord,
    },
    /// Persist a new secondary index definition (starts `Building`).
    CreateIndex {
        /// Index record.
        record: crate::catalog::IndexRecord,
    },
    /// Advance an index lifecycle state (`Building → Ready → Dropping`).
    AdvanceIndex {
        /// Affected index.
        id: u64,
        /// New state discriminant.
        state: u8,
    },
    /// Remove a `Dropping` index definition.
    RemoveIndex {
        /// Affected index.
        id: u64,
    },
    /// Publish one asset's redundancy layout generation (generation-fenced
    /// at apply: exactly current + 1, or exactly 1 when absent).
    PublishRedundancyLayout {
        /// Map key: parsed asset identity riding alongside the opaque bytes.
        key: LayoutKey,
        /// Offered control generation. Duplicates `record.control_generation`
        /// so fencing never parses the opaque `record.layout` bytes.
        generation: u64,
        /// Opaque canonical layout bytes plus their generation.
        record: LayoutRecord,
    },
    /// Drop one asset's published redundancy layout (idempotent).
    DropRedundancyAsset {
        /// Map key to remove.
        key: LayoutKey,
    },
}

/// Why a control mutation was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ControlMutationError {
    /// Framing version this binary cannot speak.
    #[error("unsupported control-mutation version {found}")]
    UnsupportedVersion {
        /// Observed version.
        found: u16,
    },
    /// Truncated input where framing promised bytes.
    #[error("truncated control mutation")]
    Truncated,
    /// A length prefix exceeds the input.
    #[error("declared length {len} exceeds input {available}")]
    Oversized {
        /// Declared length.
        len: usize,
        /// Bytes actually available.
        available: usize,
    },
    /// Unknown mutation tag.
    #[error("unknown control mutation tag {tag}")]
    BadTag {
        /// Observed tag.
        tag: u8,
    },
    /// Trailing bytes after the framed mutation.
    #[error("trailing bytes in control mutation")]
    TrailingBytes,
    /// An embedded record failed to decode.
    #[error("undecodable control mutation body: {detail}")]
    BadBody {
        /// Human-readable cause.
        detail: String,
    },
}

impl ControlMutation {
    /// Encodes the canonical bytes: `version u16, tag u8, body_len u32,
    /// body[...]`.
    #[must_use]
    pub fn encode_to_vec(&self) -> Vec<u8> {
        let (tag, body) = match self {
            Self::RegisterNode { record } => (0u8, record.encode_to_vec()),
            Self::SetNodeState { node, state } => {
                let mut body = Vec::with_capacity(9);
                body.extend_from_slice(&node.as_u64().to_le_bytes());
                body.push(state.encode_byte());
                (1u8, body)
            }
            Self::SetDesiredPlacement { desired } => (2u8, desired.encode_to_vec()),
            Self::CreateMigration { plan } => (3u8, plan.encode_to_vec()),
            Self::AdvanceMigration {
                plan,
                phase,
                generation,
            } => {
                let mut body = Vec::with_capacity(8 + 1 + 8);
                body.extend_from_slice(&plan.as_u64().to_le_bytes());
                body.push(phase.encode_byte());
                body.extend_from_slice(&generation.as_u64().to_le_bytes());
                (4u8, body)
            }
            Self::RemoveMigration { plan } => {
                let mut body = Vec::with_capacity(8);
                body.extend_from_slice(&plan.as_u64().to_le_bytes());
                (5u8, body)
            }
            Self::CreateSplit { plan } => (6u8, plan.encode_to_vec()),
            Self::AdvanceSplit {
                plan,
                phase,
                generation,
            } => {
                let mut body = Vec::with_capacity(8 + 1 + 8);
                body.extend_from_slice(&plan.as_u64().to_le_bytes());
                body.push(phase.encode_byte());
                body.extend_from_slice(&generation.as_u64().to_le_bytes());
                (7u8, body)
            }
            Self::RemoveSplit { plan } => {
                let mut body = Vec::with_capacity(8);
                body.extend_from_slice(&plan.as_u64().to_le_bytes());
                (8u8, body)
            }
            Self::CreateMerge { plan } => (9u8, plan.encode_to_vec()),
            Self::AdvanceMerge {
                plan,
                phase,
                generation,
            } => {
                let mut body = Vec::with_capacity(8 + 1 + 8);
                body.extend_from_slice(&plan.as_u64().to_le_bytes());
                body.push(phase.encode_byte());
                body.extend_from_slice(&generation.as_u64().to_le_bytes());
                (10u8, body)
            }
            Self::RemoveMerge { plan } => {
                let mut body = Vec::with_capacity(8);
                body.extend_from_slice(&plan.as_u64().to_le_bytes());
                (11u8, body)
            }
            Self::RegisterNamespace { record } => (12u8, record.encode_to_vec()),
            Self::CreateIndex { record } => (13u8, record.encode_to_vec()),
            Self::AdvanceIndex { id, state } => {
                let mut body = Vec::with_capacity(9);
                body.extend_from_slice(&id.to_le_bytes());
                body.push(*state);
                (14u8, body)
            }
            Self::RemoveIndex { id } => {
                let mut body = Vec::with_capacity(8);
                body.extend_from_slice(&id.to_le_bytes());
                (15u8, body)
            }
            Self::PublishRedundancyLayout {
                key,
                generation,
                record,
            } => {
                let key_bytes = key.encode_to_vec();
                let record_bytes = record.encode_to_vec();
                let mut body = Vec::with_capacity(key_bytes.len() + 8 + record_bytes.len());
                body.extend_from_slice(&key_bytes);
                body.extend_from_slice(&generation.to_le_bytes());
                body.extend_from_slice(&record_bytes);
                (16u8, body)
            }
            Self::DropRedundancyAsset { key } => (17u8, key.encode_to_vec()),
        };
        let mut out = Vec::with_capacity(2 + 1 + 4 + body.len());
        out.extend_from_slice(&CONTROL_MUTATION_VERSION.to_le_bytes());
        out.push(tag);
        let len = u32::try_from(body.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    /// Decodes exactly one mutation.
    ///
    /// # Errors
    ///
    /// Returns [`ControlMutationError`] on version mismatch, truncation,
    /// unknown tags, undecodable bodies, or trailing bytes.
    #[allow(clippy::too_many_lines)]
    pub fn decode_exact(input: &[u8]) -> Result<Self, ControlMutationError> {
        use ControlMutationError as Fault;
        if input.len() < 2 + 1 + 4 {
            return Err(Fault::Truncated);
        }
        let version = u16::from_le_bytes(input[..2].try_into().unwrap_or([0; 2]));
        if version != CONTROL_MUTATION_VERSION {
            return Err(Fault::UnsupportedVersion { found: version });
        }
        let tag = input[2];
        let len = u32::from_le_bytes(input[3..7].try_into().unwrap_or([0; 4])) as usize;
        if input.len() < 7 + len {
            return Err(Fault::Oversized {
                len,
                available: input.len(),
            });
        }
        let body = &input[7..7 + len];
        let rest = &input[7 + len..];
        if !rest.is_empty() {
            return Err(Fault::TrailingBytes);
        }
        match tag {
            0 => {
                let record = NodeRecord::decode_exact(body).map_err(|error| Fault::BadBody {
                    detail: error.to_string(),
                })?;
                Ok(Self::RegisterNode { record })
            }
            1 => {
                if body.len() != 9 {
                    return Err(Fault::Truncated);
                }
                let node =
                    NodeId::from_u64(u64::from_le_bytes(body[..8].try_into().unwrap_or([0; 8])));
                let state = crate::node::NodeState::decode_byte(body[8]).map_err(|error| {
                    Fault::BadBody {
                        detail: error.to_string(),
                    }
                })?;
                Ok(Self::SetNodeState { node, state })
            }
            2 => {
                let desired =
                    DesiredReplicaSet::decode_exact(body).map_err(|error| Fault::BadBody {
                        detail: error.to_string(),
                    })?;
                Ok(Self::SetDesiredPlacement { desired })
            }
            3 => {
                let plan = MigrationPlan::decode_exact(body).map_err(|error| Fault::BadBody {
                    detail: error.to_string(),
                })?;
                Ok(Self::CreateMigration { plan })
            }
            4 => {
                if body.len() != 8 + 1 + 8 {
                    return Err(Fault::Truncated);
                }
                let plan =
                    PlanId::from_u64(u64::from_le_bytes(body[..8].try_into().unwrap_or([0; 8])));
                let phase =
                    crate::migration::MigrationPhase::decode_byte(body[8]).map_err(|error| {
                        Fault::BadBody {
                            detail: error.to_string(),
                        }
                    })?;
                let generation = PlacementVersion::from_u64(u64::from_le_bytes(
                    body[9..17].try_into().unwrap_or([0; 8]),
                ));
                Ok(Self::AdvanceMigration {
                    plan,
                    phase,
                    generation,
                })
            }
            5 => {
                if body.len() != 8 {
                    return Err(Fault::Truncated);
                }
                let plan =
                    PlanId::from_u64(u64::from_le_bytes(body[..8].try_into().unwrap_or([0; 8])));
                Ok(Self::RemoveMigration { plan })
            }
            6 => {
                let plan = SplitPlan::decode_exact(body).map_err(|error| Fault::BadBody {
                    detail: error.to_string(),
                })?;
                Ok(Self::CreateSplit { plan })
            }
            7 => {
                if body.len() != 8 + 1 + 8 {
                    return Err(Fault::Truncated);
                }
                let plan =
                    PlanId::from_u64(u64::from_le_bytes(body[..8].try_into().unwrap_or([0; 8])));
                let phase = crate::split::SplitPhase::decode_byte(body[8]).map_err(|error| {
                    Fault::BadBody {
                        detail: error.to_string(),
                    }
                })?;
                let generation = PlacementVersion::from_u64(u64::from_le_bytes(
                    body[9..17].try_into().unwrap_or([0; 8]),
                ));
                Ok(Self::AdvanceSplit {
                    plan,
                    phase,
                    generation,
                })
            }
            8 => {
                if body.len() != 8 {
                    return Err(Fault::Truncated);
                }
                let plan =
                    PlanId::from_u64(u64::from_le_bytes(body[..8].try_into().unwrap_or([0; 8])));
                Ok(Self::RemoveSplit { plan })
            }
            9 => {
                let plan = MergePlan::decode_exact(body).map_err(|error| Fault::BadBody {
                    detail: error.to_string(),
                })?;
                Ok(Self::CreateMerge { plan })
            }
            10 => {
                if body.len() != 8 + 1 + 8 {
                    return Err(Fault::Truncated);
                }
                let plan =
                    PlanId::from_u64(u64::from_le_bytes(body[..8].try_into().unwrap_or([0; 8])));
                let phase = crate::merge::MergePhase::decode_byte(body[8]).map_err(|error| {
                    Fault::BadBody {
                        detail: error.to_string(),
                    }
                })?;
                let generation = PlacementVersion::from_u64(u64::from_le_bytes(
                    body[9..17].try_into().unwrap_or([0; 8]),
                ));
                Ok(Self::AdvanceMerge {
                    plan,
                    phase,
                    generation,
                })
            }
            11 => {
                if body.len() != 8 {
                    return Err(Fault::Truncated);
                }
                let plan =
                    PlanId::from_u64(u64::from_le_bytes(body[..8].try_into().unwrap_or([0; 8])));
                Ok(Self::RemoveMerge { plan })
            }
            12 => {
                let record =
                    crate::catalog::NamespaceRecord::decode_exact(body).map_err(|error| {
                        Fault::BadBody {
                            detail: error.to_string(),
                        }
                    })?;
                Ok(Self::RegisterNamespace { record })
            }
            13 => {
                let record = crate::catalog::IndexRecord::decode_exact(body).map_err(|error| {
                    Fault::BadBody {
                        detail: error.to_string(),
                    }
                })?;
                Ok(Self::CreateIndex { record })
            }
            14 => {
                if body.len() != 9 {
                    return Err(Fault::Truncated);
                }
                let id = u64::from_le_bytes(body[..8].try_into().unwrap_or([0; 8]));
                Ok(Self::AdvanceIndex { id, state: body[8] })
            }
            15 => {
                if body.len() != 8 {
                    return Err(Fault::Truncated);
                }
                let id = u64::from_le_bytes(body[..8].try_into().unwrap_or([0; 8]));
                Ok(Self::RemoveIndex { id })
            }
            16 => {
                if body.len() < LAYOUT_KEY_LEN + 8 + 12 {
                    return Err(Fault::Truncated);
                }
                let key = LayoutKey::decode_exact(&body[..LAYOUT_KEY_LEN]).map_err(|error| {
                    Fault::BadBody {
                        detail: error.to_string(),
                    }
                })?;
                let generation = u64::from_le_bytes(
                    body[LAYOUT_KEY_LEN..LAYOUT_KEY_LEN + 8]
                        .try_into()
                        .unwrap_or([0; 8]),
                );
                let record =
                    LayoutRecord::decode_exact(&body[LAYOUT_KEY_LEN + 8..]).map_err(|error| {
                        Fault::BadBody {
                            detail: error.to_string(),
                        }
                    })?;
                if generation != record.control_generation {
                    return Err(Fault::BadBody {
                        detail: format!(
                            "publish generation {generation} disagrees with record {}",
                            record.control_generation
                        ),
                    });
                }
                Ok(Self::PublishRedundancyLayout {
                    key,
                    generation,
                    record,
                })
            }
            17 => {
                if body.len() != LAYOUT_KEY_LEN {
                    return Err(Fault::Truncated);
                }
                let key = LayoutKey::decode_exact(body).map_err(|error| Fault::BadBody {
                    detail: error.to_string(),
                })?;
                Ok(Self::DropRedundancyAsset { key })
            }
            _ => Err(Fault::BadTag { tag }),
        }
    }

    /// Returns the tablet this mutation's fencing generation binds to, if
    /// any (plan advances fence on the plan's generation).
    #[must_use]
    pub const fn generation(&self) -> Option<PlacementVersion> {
        match self {
            Self::AdvanceMigration { generation, .. }
            | Self::AdvanceSplit { generation, .. }
            | Self::AdvanceMerge { generation, .. } => Some(*generation),
            Self::RegisterNode { .. }
            | Self::SetNodeState { .. }
            | Self::SetDesiredPlacement { .. }
            | Self::CreateMigration { .. }
            | Self::RemoveMigration { .. }
            | Self::CreateSplit { .. }
            | Self::RemoveSplit { .. }
            | Self::CreateMerge { .. }
            | Self::RemoveMerge { .. }
            | Self::RegisterNamespace { .. }
            | Self::CreateIndex { .. }
            | Self::AdvanceIndex { .. }
            | Self::RemoveIndex { .. }
            | Self::PublishRedundancyLayout { .. }
            | Self::DropRedundancyAsset { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use kivi_types::{NodeId, TabletId};

    use super::*;
    use crate::migration::MigrationPhase;
    use crate::node::NodeState;
    use crate::placement::PlacementVersion;

    fn record() -> NodeRecord {
        NodeRecord {
            node: NodeId::from_u64(4),
            peer: "127.0.0.1:9144".parse::<SocketAddr>().expect("peer"),
            native: "127.0.0.1:9044".parse::<SocketAddr>().expect("native"),
            admin: "127.0.0.1:19444".parse::<SocketAddr>().expect("admin"),
            cert_fingerprint: [9u8; 32],
            failure_domain: String::new(),
            weight: 1,
            state: NodeState::Joining,
        }
    }

    #[test]
    fn control_mutations_round_trip() {
        let cases = [
            ControlMutation::RegisterNode { record: record() },
            ControlMutation::SetNodeState {
                node: NodeId::from_u64(4),
                state: NodeState::Active,
            },
            ControlMutation::AdvanceMigration {
                plan: PlanId::from_u64(7),
                phase: MigrationPhase::Ready,
                generation: PlacementVersion::from_u64(2),
            },
            ControlMutation::RemoveMigration {
                plan: PlanId::from_u64(7),
            },
            ControlMutation::AdvanceSplit {
                plan: PlanId::from_u64(7),
                phase: crate::split::SplitPhase::ChildrenAllocated,
                generation: PlacementVersion::from_u64(2),
            },
            ControlMutation::RemoveSplit {
                plan: PlanId::from_u64(7),
            },
            ControlMutation::AdvanceMerge {
                plan: PlanId::from_u64(7),
                phase: crate::merge::MergePhase::TargetAllocated,
                generation: PlacementVersion::from_u64(2),
            },
            ControlMutation::RemoveMerge {
                plan: PlanId::from_u64(7),
            },
        ];
        for mutation in cases {
            let back = ControlMutation::decode_exact(&mutation.encode_to_vec()).expect("decodes");
            assert_eq!(back, mutation);
        }
        let plan = MigrationPlan::new(
            PlanId::from_u64(1),
            TabletId::from_u64(3),
            PlacementVersion::INITIAL,
            NodeId::from_u64(1),
            NodeId::from_u64(4),
            vec![
                NodeId::from_u64(2),
                NodeId::from_u64(3),
                NodeId::from_u64(4),
            ],
        )
        .expect("plan");
        let create = ControlMutation::CreateMigration { plan };
        let back = ControlMutation::decode_exact(&create.encode_to_vec()).expect("decodes");
        assert_eq!(back, create);
    }

    #[test]
    fn redundancy_mutations_round_trip() {
        use crate::layouts::{LayoutKey, LayoutRecord};
        let asset = kivi_redundancy::AssetId::new(kivi_redundancy::AssetKind::Chunk, 7, [11; 32])
            .expect("asset");
        let key = LayoutKey::from_asset(&asset);
        let publish = ControlMutation::PublishRedundancyLayout {
            key,
            generation: 2,
            record: LayoutRecord::new(2, vec![9u8; 24]),
        };
        let back = ControlMutation::decode_exact(&publish.encode_to_vec()).expect("decodes");
        assert_eq!(back, publish);
        let drop = ControlMutation::DropRedundancyAsset { key };
        let back = ControlMutation::decode_exact(&drop.encode_to_vec()).expect("decodes");
        assert_eq!(back, drop);
        assert_eq!(publish.generation(), None);
        assert_eq!(drop.generation(), None);
    }

    #[test]
    fn redundancy_mutation_mismatch_and_shapes_fail() {
        use crate::layouts::{LayoutKey, LayoutRecord};
        let asset = kivi_redundancy::AssetId::new(kivi_redundancy::AssetKind::Chunk, 7, [11; 32])
            .expect("asset");
        let key = LayoutKey::from_asset(&asset);
        // Explicit generation must duplicate the record generation.
        let mismatched = ControlMutation::PublishRedundancyLayout {
            key,
            generation: 3,
            record: LayoutRecord::new(2, vec![9u8; 24]),
        };
        assert!(matches!(
            ControlMutation::decode_exact(&mismatched.encode_to_vec()),
            Err(ControlMutationError::BadBody { .. })
        ));
        // Zero-generation records never decode.
        let zero = ControlMutation::PublishRedundancyLayout {
            key,
            generation: 0,
            record: LayoutRecord::new(0, vec![9u8; 24]),
        };
        assert!(matches!(
            ControlMutation::decode_exact(&zero.encode_to_vec()),
            Err(ControlMutationError::BadBody { .. })
        ));
        // Wrong drop body length truncates like every other fixed body.
        let drop = ControlMutation::DropRedundancyAsset { key };
        let mut short = drop.encode_to_vec();
        short.pop();
        assert!(matches!(
            ControlMutation::decode_exact(&short),
            Err(ControlMutationError::Oversized { .. } | ControlMutationError::Truncated)
        ));
    }

    #[test]
    fn corrupt_framing_fails() {
        let good = ControlMutation::SetNodeState {
            node: NodeId::from_u64(1),
            state: NodeState::Draining,
        }
        .encode_to_vec();
        assert!(ControlMutation::decode_exact(&good[..3]).is_err());
        let mut trailed = good.clone();
        trailed.push(0xFF);
        assert!(matches!(
            ControlMutation::decode_exact(&trailed),
            Err(ControlMutationError::TrailingBytes)
        ));
        let mut versioned = good;
        versioned[0] = 0xFF;
        assert!(matches!(
            ControlMutation::decode_exact(&versioned),
            Err(ControlMutationError::UnsupportedVersion { .. })
        ));
    }
}
