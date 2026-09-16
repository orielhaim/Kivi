//! Persistent migration plans.
//!
//! Tablet movement is a multi-step distributed operation. Every step is
//! persisted through the control plane, so migration progress survives
//! process crashes and control-leader changes: a new leader reads the
//! persisted plans, observes actual group state, and continues safely.
//! No in-memory workflow continuation is ever required.

use kivi_types::{NodeId, TabletId};

use crate::placement::PlacementVersion;

/// Identity of one migration plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MigrationPlanId(pub u64);

impl MigrationPlanId {
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

/// Migration lifecycle phase.
///
/// The state machine is intentionally small: every step is idempotent and
/// re-derivable from (persisted plan, observed actual membership).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MigrationPhase {
    /// Plan committed; target replica creation not yet observed.
    Planned,
    /// Target replica exists locally; learner registration pending.
    TargetStarting,
    /// Learner registered on the tablet leader; catch-up in flight.
    LearnerAdded,
    /// Learner caught up (blocking learner observed line-rate); ready to
    /// promote.
    Ready,
    /// Joint/uniform membership change in flight.
    MembershipChanging,
    /// New uniform membership committed; source retirement pending.
    Committed,
    /// Source replica tombstoned locally; plan done pending cleanup.
    SourceRetiring,
    /// Terminal success.
    Completed,
    /// Terminal failure with a human-readable reason kept out of band.
    Failed,
}

impl MigrationPhase {
    /// Whether the phase is terminal (no further reconciler work).
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed)
    }

    /// Encodes the phase as a canonical discriminant byte.
    #[must_use]
    pub const fn encode_byte(self) -> u8 {
        match self {
            Self::Planned => 0,
            Self::TargetStarting => 1,
            Self::LearnerAdded => 2,
            Self::Ready => 3,
            Self::MembershipChanging => 4,
            Self::Committed => 5,
            Self::SourceRetiring => 6,
            Self::Completed => 7,
            Self::Failed => 8,
        }
    }

    /// Decodes a canonical discriminant byte.
    ///
    /// # Errors
    ///
    /// Returns [`MigrationError::BadPhase`] for unknown discriminants.
    pub const fn decode_byte(byte: u8) -> Result<Self, MigrationError> {
        match byte {
            0 => Ok(Self::Planned),
            1 => Ok(Self::TargetStarting),
            2 => Ok(Self::LearnerAdded),
            3 => Ok(Self::Ready),
            4 => Ok(Self::MembershipChanging),
            5 => Ok(Self::Committed),
            6 => Ok(Self::SourceRetiring),
            7 => Ok(Self::Completed),
            8 => Ok(Self::Failed),
            _ => Err(MigrationError::BadPhase { found: byte }),
        }
    }
}

/// Why a migration plan was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum MigrationError {
    /// Unknown phase discriminant in canonical bytes.
    #[error("unknown migration phase discriminant {found}")]
    BadPhase {
        /// Observed discriminant.
        found: u8,
    },
    /// Truncated canonical input.
    #[error("truncated migration plan")]
    Truncated,
    /// Trailing bytes after the framed plan.
    #[error("trailing bytes in migration plan")]
    TrailingBytes,
    /// Source and target are the same node (no movement).
    #[error("migration source {node} equals target")]
    NoMovement {
        /// The repeated node.
        node: u64,
    },
}

/// One persisted tablet move: exactly one replica replaced.
///
/// Conceptually `actual voters {A,B,C}` with `desired {B,C,D}` moves
/// `from = A` to `to = D`. The `generation` fences stale control
/// actions: a create/remove/complete carrying an older generation than
/// the tablet's current placement version is rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationPlan {
    /// Plan identity (control-plane-assigned, never reused).
    pub id: MigrationPlanId,
    /// Tablet moving.
    pub tablet: TabletId,
    /// Placement generation at plan creation (fencing).
    pub generation: PlacementVersion,
    /// Replica leaving.
    pub from: NodeId,
    /// Replica joining.
    pub to: NodeId,
    /// Full desired voter set after the move (sorted, includes `to`,
    /// excludes `from`).
    pub desired_voters: Vec<NodeId>,
    /// Current phase.
    pub phase: MigrationPhase,
}

impl MigrationPlan {
    /// Builds a plan, validating that source and target differ.
    ///
    /// # Errors
    ///
    /// Returns [`MigrationError::NoMovement`] when `from == to`.
    pub fn new(
        id: MigrationPlanId,
        tablet: TabletId,
        generation: PlacementVersion,
        from: NodeId,
        to: NodeId,
        desired_voters: Vec<NodeId>,
    ) -> Result<Self, MigrationError> {
        if from == to {
            return Err(MigrationError::NoMovement {
                node: from.as_u64(),
            });
        }
        let mut voters = desired_voters;
        voters.sort_by_key(|node: &NodeId| node.as_u64());
        voters.dedup();
        Ok(Self {
            id,
            tablet,
            generation,
            from,
            to,
            desired_voters: voters,
            phase: MigrationPhase::Planned,
        })
    }

    /// Moves the plan to a new phase (same id/tablet/generation/voters).
    #[must_use]
    pub fn advance(&self, phase: MigrationPhase) -> Self {
        Self {
            id: self.id,
            tablet: self.tablet,
            generation: self.generation,
            from: self.from,
            to: self.to,
            desired_voters: self.desired_voters.clone(),
            phase,
        }
    }

    /// Encodes the canonical bytes.
    ///
    /// Layout: `id u64, tablet u64, generation u64, from u64, to u64,
    /// phase u8, count u32, voters[count] u64`.
    #[must_use]
    pub fn encode_to_vec(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 * 5 + 1 + 4 + 8 * self.desired_voters.len());
        out.extend_from_slice(&self.id.as_u64().to_le_bytes());
        out.extend_from_slice(&self.tablet.as_u64().to_le_bytes());
        out.extend_from_slice(&self.generation.as_u64().to_le_bytes());
        out.extend_from_slice(&self.from.as_u64().to_le_bytes());
        out.extend_from_slice(&self.to.as_u64().to_le_bytes());
        out.push(self.phase.encode_byte());
        let count = u32::try_from(self.desired_voters.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&count.to_le_bytes());
        for voter in &self.desired_voters {
            out.extend_from_slice(&voter.as_u64().to_le_bytes());
        }
        out
    }

    /// Decodes exactly one plan.
    ///
    /// # Errors
    ///
    /// Returns [`MigrationError`] on truncation, unknown phases,
    /// trailing bytes, or source/target incoherence.
    pub fn decode_exact(input: &[u8]) -> Result<Self, MigrationError> {
        use MigrationError as Fault;
        if input.len() < 8 * 5 + 1 + 4 {
            return Err(Fault::Truncated);
        }
        let id =
            MigrationPlanId::from_u64(u64::from_le_bytes(input[..8].try_into().unwrap_or([0; 8])));
        let tablet = TabletId::from_u64(u64::from_le_bytes(
            input[8..16].try_into().unwrap_or([0; 8]),
        ));
        let generation = PlacementVersion::from_u64(u64::from_le_bytes(
            input[16..24].try_into().unwrap_or([0; 8]),
        ));
        let from = NodeId::from_u64(u64::from_le_bytes(
            input[24..32].try_into().unwrap_or([0; 8]),
        ));
        let to = NodeId::from_u64(u64::from_le_bytes(
            input[32..40].try_into().unwrap_or([0; 8]),
        ));
        let phase = MigrationPhase::decode_byte(input[40])?;
        let count = u32::from_le_bytes(input[41..45].try_into().unwrap_or([0; 4])) as usize;
        if input.len() != 45 + 8 * count {
            return Err(if input.len() < 45 + 8 * count {
                Fault::Truncated
            } else {
                Fault::TrailingBytes
            });
        }
        let mut voters = Vec::with_capacity(count);
        for index in 0..count {
            let at = 45 + 8 * index;
            voters.push(NodeId::from_u64(u64::from_le_bytes(
                input[at..at + 8].try_into().unwrap_or([0; 8]),
            )));
        }
        Self::new(id, tablet, generation, from, to, voters).map(|mut plan| {
            plan.phase = phase;
            plan
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> MigrationPlan {
        MigrationPlan::new(
            MigrationPlanId::from_u64(1),
            TabletId::from_u64(17),
            PlacementVersion::from_u64(3),
            NodeId::from_u64(1),
            NodeId::from_u64(4),
            vec![
                NodeId::from_u64(4),
                NodeId::from_u64(2),
                NodeId::from_u64(3),
            ],
        )
        .expect("valid")
    }

    #[test]
    fn plan_round_trips() {
        let plan = sample();
        assert_eq!(
            plan.desired_voters,
            vec![
                NodeId::from_u64(2),
                NodeId::from_u64(3),
                NodeId::from_u64(4)
            ]
        );
        let back = MigrationPlan::decode_exact(&plan.encode_to_vec()).expect("decodes");
        assert_eq!(back, plan);
        assert!(!plan.phase.is_terminal());
        assert!(MigrationPhase::Completed.is_terminal());
    }

    #[test]
    fn self_move_fails() {
        assert!(matches!(
            MigrationPlan::new(
                MigrationPlanId::from_u64(1),
                TabletId::from_u64(1),
                PlacementVersion::INITIAL,
                NodeId::from_u64(2),
                NodeId::from_u64(2),
                vec![NodeId::from_u64(2)],
            ),
            Err(MigrationError::NoMovement { .. })
        ));
    }
}
