//! Persisted tablet-split plans: online subdivision of one parent range
//! into two child ranges.
//!
//! A split is a multi-step distributed operation reusing the same
//! observe/decide/execute/persist discipline as migration: every phase is
//! persisted through the control plane, so progress survives control-leader
//! crashes and restarts. Children are normal tablets after cutover (own
//! [`ConsensusGroupId`](kivi_types::TabletId), replica placement, Raft
//! group, checkpoint, state machine); during preparation they are
//! transitional (`Allocated`/`Inactive`, never writable).
//!
//! Online model (brief bounded fence only at cutover):
//!
//! ```text
//! Parent serves
//!   → allocate children ( normal TabletId, midpoint ranges, inherit
//!     parent desired voters)
//!   → ensure child groups on replicas + seed base from local parent
//!     state (logical partition, chunk roots referenced, never copied)
//!   → fence parent briefly, install final tail into children
//!   → atomically publish directory cutover (single validated transition:
//!     seal parent, activate children, retire parent with redirect)
//!   → children serve, parent retires (tombstone, no resurrection)
//! ```
//!
//! Dedup safety: the parent's full session set is copied to BOTH children
//! (small, bounded). A retry routed by key to the correct child finds its
//! original outcome; nothing executes twice.

use kivi_types::{NodeId, TabletId};

use crate::placement::PlacementVersion;

/// Identity of one split plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SplitPlanId(pub u64);

impl SplitPlanId {
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

/// Split lifecycle phase (kept small; every step idempotent).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SplitPhase {
    /// Plan committed; children not yet allocated.
    Planned,
    /// Children allocated (ids, midpoint ranges, placements persisted).
    ChildrenAllocated,
    /// Child groups ensured on replicas; base partitioned locally.
    BaseSeeded,
    /// Parent fenced; final tail installed into children.
    Fenced,
    /// Directory cutover published (parent retired with redirect,
    /// children active). Atomic logical point.
    CutoverCommitted,
    /// Parent replicas tombstoned; plan done pending cleanup.
    ParentRetiring,
    /// Terminal success.
    Completed,
    /// Terminal failure.
    Failed,
}

impl SplitPhase {
    /// Whether the phase is terminal.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed)
    }

    /// Encodes the phase as a canonical discriminant byte.
    #[must_use]
    pub const fn encode_byte(self) -> u8 {
        match self {
            Self::Planned => 0,
            Self::ChildrenAllocated => 1,
            Self::BaseSeeded => 2,
            Self::Fenced => 3,
            Self::CutoverCommitted => 4,
            Self::ParentRetiring => 5,
            Self::Completed => 6,
            Self::Failed => 7,
        }
    }

    /// Decodes a canonical discriminant byte.
    ///
    /// # Errors
    ///
    /// Returns [`SplitError::BadPhase`] for unknown discriminants.
    pub const fn decode_byte(byte: u8) -> Result<Self, SplitError> {
        match byte {
            0 => Ok(Self::Planned),
            1 => Ok(Self::ChildrenAllocated),
            2 => Ok(Self::BaseSeeded),
            3 => Ok(Self::Fenced),
            4 => Ok(Self::CutoverCommitted),
            5 => Ok(Self::ParentRetiring),
            6 => Ok(Self::Completed),
            7 => Ok(Self::Failed),
            _ => Err(SplitError::BadPhase { found: byte }),
        }
    }
}

/// Why a split plan was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SplitError {
    /// Unknown phase discriminant.
    #[error("unknown split phase discriminant {found}")]
    BadPhase {
        /// Observed discriminant.
        found: u8,
    },
    /// Truncated canonical input.
    #[error("truncated split plan")]
    Truncated,
    /// Trailing bytes after the framed plan.
    #[error("trailing bytes in split plan")]
    TrailingBytes,
    /// Child equals parent or children equal each other.
    #[error("split children must differ from parent and each other")]
    BadChildren,
}

/// One persisted tablet split: parent into left/right children.
///
/// `split_hash` is the deterministic midpoint boundary (u128) of the
/// parent hash-prefix range; the architecture permits arbitrary future
/// split points by replacing this field. Child replica sets initially
/// inherit the parent's desired voters (logical split first, physical
/// rebalance follows as ordinary tablets).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitPlan {
    /// Plan identity (shares the control `next_plan_id` sequence with
    /// migrations: ids are unique across plan kinds).
    pub id: SplitPlanId,
    /// Parent tablet subdividing.
    pub parent: TabletId,
    /// Deterministic split boundary (midpoint of parent range).
    pub split_hash: u128,
    /// Left child (`[parent.start, split_hash)`).
    pub left: TabletId,
    /// Right child (`[split_hash, parent.end)`).
    pub right: TabletId,
    /// Child replica set (inherited from parent at allocation).
    pub replicas: Vec<NodeId>,
    /// Placement generation at plan creation (fencing).
    pub generation: PlacementVersion,
    /// Current phase.
    pub phase: SplitPhase,
}

impl SplitPlan {
    /// Builds a plan, validating children differ from the parent and each
    /// other.
    ///
    /// # Errors
    ///
    /// Returns [`SplitError::BadChildren`] on degenerate identities.
    pub fn new(
        id: SplitPlanId,
        parent: TabletId,
        split_hash: u128,
        left: TabletId,
        right: TabletId,
        replicas: Vec<NodeId>,
        generation: PlacementVersion,
    ) -> Result<Self, SplitError> {
        if parent == left || parent == right || left == right {
            return Err(SplitError::BadChildren);
        }
        let mut replicas = replicas;
        replicas.sort_by_key(|node| node.as_u64());
        replicas.dedup();
        Ok(Self {
            id,
            parent,
            split_hash,
            left,
            right,
            replicas,
            generation,
            phase: SplitPhase::Planned,
        })
    }

    /// Moves the plan to a new phase.
    #[must_use]
    pub fn advance(&self, phase: SplitPhase) -> Self {
        Self {
            id: self.id,
            parent: self.parent,
            split_hash: self.split_hash,
            left: self.left,
            right: self.right,
            replicas: self.replicas.clone(),
            generation: self.generation,
            phase,
        }
    }

    /// All tablets this plan touches (parent + children).
    #[must_use]
    pub fn tablets(&self) -> [TabletId; 3] {
        [self.parent, self.left, self.right]
    }

    /// All replica hosts this plan touches (for authoritative observation:
    /// consult these nodes' admin planes, never an embryonic local view).
    #[must_use]
    pub fn hosts(&self) -> Vec<NodeId> {
        self.replicas.clone()
    }

    /// Encodes the canonical bytes.
    ///
    /// Layout: `id u64, parent u64, split_hash u128, left u64, right u64,
    /// generation u64, phase u8, count u32, replicas[count] u64`.
    #[must_use]
    pub fn encode_to_vec(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 * 5 + 16 + 1 + 4 + 8 * self.replicas.len());
        out.extend_from_slice(&self.id.as_u64().to_le_bytes());
        out.extend_from_slice(&self.parent.as_u64().to_le_bytes());
        out.extend_from_slice(&self.split_hash.to_le_bytes());
        out.extend_from_slice(&self.left.as_u64().to_le_bytes());
        out.extend_from_slice(&self.right.as_u64().to_le_bytes());
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
    /// Returns [`SplitError`] on truncation, unknown phases, trailing
    /// bytes, or degenerate identities.
    pub fn decode_exact(input: &[u8]) -> Result<Self, SplitError> {
        use SplitError as Fault;
        if input.len() < 8 + 8 + 16 + 8 + 8 + 8 + 1 + 4 {
            return Err(Fault::Truncated);
        }
        let id = SplitPlanId::from_u64(u64::from_le_bytes(input[..8].try_into().unwrap_or([0; 8])));
        let parent = TabletId::from_u64(u64::from_le_bytes(
            input[8..16].try_into().unwrap_or([0; 8]),
        ));
        let split_hash = u128::from_le_bytes(input[16..32].try_into().unwrap_or([0; 16]));
        let left = TabletId::from_u64(u64::from_le_bytes(
            input[32..40].try_into().unwrap_or([0; 8]),
        ));
        let right = TabletId::from_u64(u64::from_le_bytes(
            input[40..48].try_into().unwrap_or([0; 8]),
        ));
        let generation = PlacementVersion::from_u64(u64::from_le_bytes(
            input[48..56].try_into().unwrap_or([0; 8]),
        ));
        let phase = SplitPhase::decode_byte(input[56])?;
        let count = u32::from_le_bytes(input[57..61].try_into().unwrap_or([0; 4])) as usize;
        if input.len() != 61 + 8 * count {
            return Err(if input.len() < 61 + 8 * count {
                Fault::Truncated
            } else {
                Fault::TrailingBytes
            });
        }
        let mut replicas = Vec::with_capacity(count);
        for index in 0..count {
            let at = 61 + 8 * index;
            replicas.push(NodeId::from_u64(u64::from_le_bytes(
                input[at..at + 8].try_into().unwrap_or([0; 8]),
            )));
        }
        Self::new(id, parent, split_hash, left, right, replicas, generation).map(|mut plan| {
            plan.phase = phase;
            plan
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> SplitPlan {
        SplitPlan::new(
            SplitPlanId::from_u64(1),
            TabletId::from_u64(7),
            1u128 << 127,
            TabletId::from_u64(101),
            TabletId::from_u64(102),
            vec![
                NodeId::from_u64(1),
                NodeId::from_u64(2),
                NodeId::from_u64(3),
            ],
            PlacementVersion::from_u64(5),
        )
        .expect("valid")
    }

    #[test]
    fn split_round_trips() {
        let plan = sample();
        let back = SplitPlan::decode_exact(&plan.encode_to_vec()).expect("decodes");
        assert_eq!(back, plan);
        assert!(!plan.phase.is_terminal());
        assert!(SplitPhase::Completed.is_terminal());
    }

    #[test]
    fn degenerate_children_fail() {
        assert!(matches!(
            SplitPlan::new(
                SplitPlanId::from_u64(1),
                TabletId::from_u64(7),
                0,
                TabletId::from_u64(7),
                TabletId::from_u64(8),
                vec![NodeId::from_u64(1)],
                PlacementVersion::INITIAL,
            ),
            Err(SplitError::BadChildren)
        ));
    }
}
