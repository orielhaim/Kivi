//! Persisted tablet-merge plans: the inverse of split for adjacent
//! compatible tablets.
//!
//! ```text
//! Left [A,M) + Right [M,Z)  →  Merged [A,Z)
//! ```
//!
//! Requirements: adjacent ranges, compatible namespace/semantics, and
//! compatible-enough placement. Only adjacent merges are allowed (never
//! arbitrary non-adjacent pairs). The merged tablet is a normal tablet
//! after cutover; during preparation it is transitional.
//!
//! Dedup safety mirrors split: the merged tablet unions both parents'
//! session sets. The ranges are disjoint so ordinary key state cannot
//! conflict; per-session sequences are globally unique per session, so the
//! union is collision-free in practice (a conflicting duplicate keeps the
//! higher commit and never re-executes).

use kivi_types::{NodeId, TabletId};

use crate::placement::PlacementVersion;
use crate::topology::PlanId;

/// Merge lifecycle phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MergePhase {
    /// Plan committed; merged tablet not yet allocated.
    Planned,
    /// Merged tablet allocated (id, union range, placement persisted).
    TargetAllocated,
    /// Merged group ensured; base unioned locally from both parents.
    BaseSeeded,
    /// Both parents fenced; final tails installed.
    Fenced,
    /// Proven: the merged group holds the planned voters, has a live leader,
    /// and is not measurably behind. The only phase from which routing may be
    /// published - see [`SplitPhase`](crate::split::SplitPhase), whose argument
    /// applies identically here.
    TargetReady,
    /// Directory cutover published (parents retired with redirect,
    /// merged active).
    CutoverCommitted,
    /// Parent replicas tombstoned; plan done pending cleanup.
    ParentsRetiring,
    /// Terminal success.
    Completed,
    /// Terminal failure.
    Failed,
}

impl MergePhase {
    /// Whether the phase is terminal.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed)
    }

    /// Whether readiness gating is reported for this phase. See
    /// [](crate::split::SplitPhase::needs_children_ready):
    /// the same three phases, for the same reasons, against the merge target.
    #[must_use]
    pub const fn needs_target_ready(self) -> bool {
        matches!(
            self,
            Self::BaseSeeded | Self::Fenced | Self::CutoverCommitted
        )
    }

    /// Every phase, for exhaustive checks over the lifecycle.
    pub const ALL: [Self; 9] = [
        Self::Planned,
        Self::TargetAllocated,
        Self::BaseSeeded,
        Self::Fenced,
        Self::TargetReady,
        Self::CutoverCommitted,
        Self::ParentsRetiring,
        Self::Completed,
        Self::Failed,
    ];

    pub(crate) fn can_advance_to(self, next: Self) -> bool {
        self == next
            || matches!(
                (self, next),
                (Self::Planned, Self::TargetAllocated | Self::Failed)
                    | (Self::TargetAllocated, Self::BaseSeeded | Self::Failed)
                    | (Self::BaseSeeded, Self::Fenced | Self::Failed)
                    | (
                        Self::Fenced,
                        Self::BaseSeeded | Self::TargetReady | Self::Failed
                    )
                    | (Self::TargetReady, Self::CutoverCommitted | Self::Failed)
                    | (Self::CutoverCommitted, Self::ParentsRetiring | Self::Failed)
                    | (Self::ParentsRetiring, Self::Completed | Self::Failed)
            )
    }

    /// Encodes the phase as a canonical discriminant byte.
    #[must_use]
    pub const fn encode_byte(self) -> u8 {
        match self {
            Self::Planned => 0,
            Self::TargetAllocated => 1,
            Self::BaseSeeded => 2,
            Self::Fenced => 3,
            Self::TargetReady => 6,
            Self::CutoverCommitted => 4,
            Self::ParentsRetiring => 5,
            Self::Completed => 7,
            Self::Failed => 8,
        }
    }

    /// Decodes a canonical discriminant byte.
    ///
    /// # Errors
    ///
    /// Returns [`MergeError::BadPhase`] for unknown discriminants.
    pub const fn decode_byte(byte: u8) -> Result<Self, MergeError> {
        match byte {
            0 => Ok(Self::Planned),
            1 => Ok(Self::TargetAllocated),
            2 => Ok(Self::BaseSeeded),
            3 => Ok(Self::Fenced),
            4 => Ok(Self::CutoverCommitted),
            5 => Ok(Self::ParentsRetiring),
            6 => Ok(Self::TargetReady),
            7 => Ok(Self::Completed),
            8 => Ok(Self::Failed),
            _ => Err(MergeError::BadPhase { found: byte }),
        }
    }
}

/// Why a merge plan was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MergeError {
    /// Unknown phase discriminant.
    #[error("unknown merge phase discriminant {found}")]
    BadPhase {
        /// Observed discriminant.
        found: u8,
    },
    /// Truncated canonical input.
    #[error("truncated merge plan")]
    Truncated,
    /// Trailing bytes after the framed plan.
    #[error("trailing bytes in merge plan")]
    TrailingBytes,
    /// Degenerate identities (target equals a parent, parents equal).
    #[error("merge identities must all differ")]
    BadIdentities,
}

/// One persisted tablet merge: left + right into merged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergePlan {
    /// Plan identity (shares the control `next_plan_id` sequence).
    pub id: PlanId,
    /// Left parent (`[A,M)`).
    pub left: TabletId,
    /// Right parent (`[M,Z)`).
    pub right: TabletId,
    /// Merged tablet (`[A,Z)`).
    pub merged: TabletId,
    /// Merged replica set (union/intersection resolved at allocation;
    /// defaults to the left parent's desired voters).
    pub replicas: Vec<NodeId>,
    /// Placement generation at plan creation (fencing).
    pub generation: PlacementVersion,
    /// Current phase.
    pub phase: MergePhase,
}

impl MergePlan {
    /// Builds a plan, validating identities differ.
    ///
    /// # Errors
    ///
    /// Returns [`MergeError::BadIdentities`] on degenerate identities.
    pub fn new(
        id: PlanId,
        left: TabletId,
        right: TabletId,
        merged: TabletId,
        replicas: Vec<NodeId>,
        generation: PlacementVersion,
    ) -> Result<Self, MergeError> {
        if left == right || left == merged || right == merged {
            return Err(MergeError::BadIdentities);
        }
        let mut replicas = replicas;
        replicas.sort_by_key(|node| node.as_u64());
        replicas.dedup();
        Ok(Self {
            id,
            left,
            right,
            merged,
            replicas,
            generation,
            phase: MergePhase::Planned,
        })
    }

    /// Moves the plan to a new phase.
    #[must_use]
    pub fn advance(&self, phase: MergePhase) -> Self {
        Self {
            id: self.id,
            left: self.left,
            right: self.right,
            merged: self.merged,
            replicas: self.replicas.clone(),
            generation: self.generation,
            phase,
        }
    }

    /// All tablets this plan touches.
    #[must_use]
    pub fn tablets(&self) -> [TabletId; 3] {
        [self.left, self.right, self.merged]
    }

    /// All replica hosts this plan touches (for authoritative observation).
    #[must_use]
    pub fn hosts(&self) -> Vec<NodeId> {
        self.replicas.clone()
    }

    /// Encodes the canonical bytes.
    ///
    /// Layout: `id u64, left u64, right u64, merged u64, generation u64,
    /// phase u8, count u32, replicas[count] u64`.
    #[must_use]
    pub fn encode_to_vec(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 * 5 + 1 + 4 + 8 * self.replicas.len());
        out.extend_from_slice(&self.id.as_u64().to_le_bytes());
        out.extend_from_slice(&self.left.as_u64().to_le_bytes());
        out.extend_from_slice(&self.right.as_u64().to_le_bytes());
        out.extend_from_slice(&self.merged.as_u64().to_le_bytes());
        out.extend_from_slice(&self.generation.as_u64().to_le_bytes());
        out.push(self.phase.encode_byte());
        let count = u32::try_from(self.replicas.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&count.to_le_bytes());
        for replica in &self.replicas {
            out.extend_from_slice(&replica.as_u64().to_le_bytes());
        }
        out
    }

    /// Decodes exactly one plan.
    ///
    /// # Errors
    ///
    /// Returns [`MergeError`] on truncation, unknown phases, trailing
    /// bytes, or degenerate identities.
    pub fn decode_exact(input: &[u8]) -> Result<Self, MergeError> {
        use MergeError as Fault;
        if input.len() < 8 * 5 + 1 + 4 {
            return Err(Fault::Truncated);
        }
        let id = PlanId::from_u64(u64::from_le_bytes(input[..8].try_into().unwrap_or([0; 8])));
        let left = TabletId::from_u64(u64::from_le_bytes(
            input[8..16].try_into().unwrap_or([0; 8]),
        ));
        let right = TabletId::from_u64(u64::from_le_bytes(
            input[16..24].try_into().unwrap_or([0; 8]),
        ));
        let merged = TabletId::from_u64(u64::from_le_bytes(
            input[24..32].try_into().unwrap_or([0; 8]),
        ));
        let generation = PlacementVersion::from_u64(u64::from_le_bytes(
            input[32..40].try_into().unwrap_or([0; 8]),
        ));
        let phase = MergePhase::decode_byte(input[40])?;
        let count = u32::from_le_bytes(input[41..45].try_into().unwrap_or([0; 4])) as usize;
        if input.len() != 45 + 8 * count {
            return Err(if input.len() < 45 + 8 * count {
                Fault::Truncated
            } else {
                Fault::TrailingBytes
            });
        }
        let mut replicas = Vec::with_capacity(count);
        for index in 0..count {
            let at = 45 + 8 * index;
            replicas.push(NodeId::from_u64(u64::from_le_bytes(
                input[at..at + 8].try_into().unwrap_or([0; 8]),
            )));
        }
        Self::new(id, left, right, merged, replicas, generation).map(|mut plan| {
            plan.phase = phase;
            plan
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_round_trips() {
        let plan = MergePlan::new(
            PlanId::from_u64(2),
            TabletId::from_u64(101),
            TabletId::from_u64(102),
            TabletId::from_u64(201),
            vec![NodeId::from_u64(1), NodeId::from_u64(2)],
            PlacementVersion::from_u64(6),
        )
        .expect("valid");
        let back = MergePlan::decode_exact(&plan.encode_to_vec()).expect("decodes");
        assert_eq!(back, plan);
        assert!(!plan.phase.is_terminal());
    }

    #[test]
    fn degenerate_identities_fail() {
        assert!(matches!(
            MergePlan::new(
                PlanId::from_u64(1),
                TabletId::from_u64(1),
                TabletId::from_u64(1),
                TabletId::from_u64(2),
                vec![NodeId::from_u64(1)],
                PlacementVersion::INITIAL,
            ),
            Err(MergeError::BadIdentities)
        ));
    }

    /// Every phase that can follow `from`, per the transition table.
    fn successors(from: MergePhase) -> Vec<MergePhase> {
        MergePhase::ALL
            .into_iter()
            .filter(|to| from.can_advance_to(*to))
            .collect()
    }

    /// Publishing routing to the merge target requires a readiness proof.
    ///
    /// The merge-side statement of the split's invariant. A merge retires *two*
    /// parents once the cutover lands, so a target that cannot serve is not a
    /// degraded range - it is two lost ones.
    #[test]
    fn publishing_routing_requires_a_readiness_proof() {
        assert_eq!(
            successors(MergePhase::Fenced),
            vec![
                MergePhase::BaseSeeded,
                MergePhase::Fenced,
                MergePhase::TargetReady,
                MergePhase::Failed,
            ],
            "Fenced reaches TargetReady; it must not reach CutoverCommitted directly"
        );
        assert_eq!(
            successors(MergePhase::TargetReady),
            vec![
                MergePhase::TargetReady,
                MergePhase::CutoverCommitted,
                MergePhase::Failed,
            ]
        );
        for from in MergePhase::ALL {
            if matches!(from, MergePhase::TargetReady | MergePhase::CutoverCommitted) {
                continue;
            }
            assert!(
                !from.can_advance_to(MergePhase::CutoverCommitted),
                "{from:?} must not publish routing without TargetReady"
            );
        }
    }

    /// A plan cannot walk back to a phase that has not published routing.
    #[test]
    fn a_plan_cannot_regress_past_the_cutover() {
        let past = |phase: MergePhase| {
            matches!(
                phase,
                MergePhase::CutoverCommitted | MergePhase::ParentsRetiring | MergePhase::Completed
            )
        };
        for from in MergePhase::ALL {
            for to in successors(from) {
                if past(from) && !to.is_terminal() {
                    assert!(
                        past(to),
                        "{from:?} -> {to:?} walks back to a phase that has not published \
                         routing; both parents are already tombstones with redirects"
                    );
                }
            }
        }
    }

    /// Every phase round-trips through a discriminant of its own.
    #[test]
    fn every_phase_round_trips_through_its_discriminant() {
        let mut seen = std::collections::BTreeSet::new();
        for phase in MergePhase::ALL {
            let byte = phase.encode_byte();
            assert!(seen.insert(byte), "{phase:?} reuses discriminant {byte}");
            assert_eq!(MergePhase::decode_byte(byte), Ok(phase));
        }
        assert!(matches!(
            MergePhase::decode_byte(9),
            Err(MergeError::BadPhase { found: 9 })
        ));
    }
}
