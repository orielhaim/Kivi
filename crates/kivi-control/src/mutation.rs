//! Typed control-plane mutations.
//!
//! The control plane is never arbitrary string KV entries. Every change is
//! a project-owned [`ControlMutation`] with canonical encoding;
//! `OpenRaft` Rust structs are never persisted as canonical control
//! state.

use kivi_types::{NodeId, TabletId};

use crate::migration::MigrationPlan;
use crate::node::NodeRecord;
use crate::placement::{DesiredReplicaSet, PlacementVersion};

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
    /// Persist a new migration plan (phase `Planned`).
    CreateMigration {
        /// Plan to persist.
        plan: MigrationPlan,
    },
    /// Advance a plan's phase (generation-fenced: stale advances are
    /// rejected by [`ControlState`](crate::state::ControlState)).
    AdvanceMigration {
        /// Affected plan.
        plan: MigrationPlanIdAlias,
        /// New phase.
        phase: crate::migration::MigrationPhase,
        /// Fencing generation the writer observed.
        generation: PlacementVersion,
    },
    /// Remove a terminally completed/failed plan.
    RemoveMigration {
        /// Affected plan.
        plan: MigrationPlanIdAlias,
    },
}

/// Identity of one migration plan (field-level alias avoiding a circular
/// module import in this enum's docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MigrationPlanIdAlias(pub u64);

impl MigrationPlanIdAlias {
    /// Wraps a raw plan id.
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

impl From<crate::migration::MigrationPlanId> for MigrationPlanIdAlias {
    fn from(id: crate::migration::MigrationPlanId) -> Self {
        Self(id.as_u64())
    }
}

impl From<MigrationPlanIdAlias> for crate::migration::MigrationPlanId {
    fn from(id: MigrationPlanIdAlias) -> Self {
        Self::from_u64(id.0)
    }
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
                let plan = MigrationPlanIdAlias::from_u64(u64::from_le_bytes(
                    body[..8].try_into().unwrap_or([0; 8]),
                ));
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
                let plan = MigrationPlanIdAlias::from_u64(u64::from_le_bytes(
                    body[..8].try_into().unwrap_or([0; 8]),
                ));
                Ok(Self::RemoveMigration { plan })
            }
            _ => Err(Fault::BadTag { tag }),
        }
    }

    /// Returns the tablet this mutation's fencing generation binds to, if
    /// any (migration advances fence on the plan's generation).
    #[must_use]
    pub const fn generation(&self) -> Option<PlacementVersion> {
        match self {
            Self::AdvanceMigration { generation, .. } => Some(*generation),
            Self::RegisterNode { .. }
            | Self::SetNodeState { .. }
            | Self::SetDesiredPlacement { .. }
            | Self::CreateMigration { .. }
            | Self::RemoveMigration { .. } => None,
        }
    }
}

/// Type alias so tablet ids stay greppable at control-mutation sites.
#[allow(dead_code)]
pub(crate) type TabletRef = TabletId;

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use kivi_types::NodeId;

    use super::*;
    use crate::migration::{MigrationPhase, MigrationPlanId};
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
                plan: MigrationPlanIdAlias::from_u64(7),
                phase: MigrationPhase::Ready,
                generation: PlacementVersion::from_u64(2),
            },
            ControlMutation::RemoveMigration {
                plan: MigrationPlanIdAlias::from_u64(7),
            },
        ];
        for mutation in cases {
            let back = ControlMutation::decode_exact(&mutation.encode_to_vec()).expect("decodes");
            assert_eq!(back, mutation);
        }
        let plan = MigrationPlan::new(
            MigrationPlanId::from_u64(1),
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
