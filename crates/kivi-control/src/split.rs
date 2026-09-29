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
//!   → PROVE both children hold the planned voters, a live leader, and
//!     are caught up
//!   → atomically publish directory cutover (single validated transition:
//!     seal parent, activate children, tombstone parent with redirect)
//!   → PROVE the children again under published routing
//!   → retire parent replicas
//! ```
//!
//! The two proofs are the point of the phase model. A group that has been
//! asked for is not a group that serves, and `cutover_split` is the instant the
//! parent stops serving - so routing published before the proof leaves the
//! namespace with nothing serving, and retiring the parent afterwards discards
//! the only copy of the data. See [`SplitPhase`].
//!
//! Dedup safety: the parent's full session set is copied to BOTH children
//! (small, bounded). A retry routed by key to the correct child finds its
//! original outcome; nothing executes twice.

use kivi_types::{NodeId, TabletId};

use crate::placement::PlacementVersion;
use crate::topology::PlanId;

/// Split lifecycle phase (kept small; every step idempotent).
///
/// # What a phase promises
///
/// Each phase name is a claim about the cluster, and the transitions below are
/// the only legal ways to make a stronger claim. Two of them are claims that
/// cannot be established by issuing a request, only by observing the result:
///
/// * [`ChildrenReady`](Self::ChildrenReady) - both children hold the planned
///   voters, a live leader, and are caught up.
/// * the gate out of
///   [`CutoverCommitted`](Self::CutoverCommitted) - the same proof, taken
///   again after routing moved, before the parent is retired.
///
/// That second proof is not redundancy. `cutover_split` is the atomic point at
/// which the parent stops serving and the children start, so a child that is
/// not serving when routing publishes leaves the namespace with no serving
/// tablet at all - and retiring the parent afterwards destroys the last copy of
/// the data the children were supposed to have.
///
/// # Why provisioning and readiness are different phases
///
/// `ensure_group` and `initialize_group` return when the *request* is
/// accepted. A group that has been asked for exists as a process object; it has
/// no voters, no leader, and no data. `ChildrenProvisioning` is exactly that
/// state and says so, so no later phase has to guess whether provisioning
/// really finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SplitPhase {
    /// Plan committed; children not yet allocated.
    Planned,
    /// Child identities, ranges, and placement exist; group requests have been
    /// accepted on every planned replica. No child serves yet, and none is
    /// claimed to.
    ChildrenProvisioning,
    /// Child groups ensured on replicas; base partitioned locally.
    BaseSeeded,
    /// Parent fenced; final tail installed into children.
    Fenced,
    /// Proven: every child holds the planned voters, a live leader, and is
    /// within catch-up threshold. This is the only phase from which routing may
    /// be published.
    ChildrenReady,
    /// Directory cutover published (parent tombstoned with redirect,
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

    /// Whether a plan in this phase may be re-driven after a crash or a
    /// control-leader change.
    ///
    /// Everything except the two terminal phases is re-entered by the driver:
    /// each step is written to be idempotent precisely so that a plan can be
    /// picked up mid-flight.
    #[must_use]
    pub const fn is_resumable(self) -> bool {
        !self.is_terminal()
    }

    /// Whether the phase already published routing, and so has made the
    /// children - not the parent - the serving tablets.
    #[must_use]
    pub const fn has_published_routing(self) -> bool {
        matches!(
            self,
            Self::CutoverCommitted | Self::ParentRetiring | Self::Completed
        )
    }

    /// Whether the plan must not regress past the cutover.
    ///
    /// After routing publishes there is nothing to go back to: the parent is a
    /// tombstone with a redirect and the children are the only serving tablets.
    /// A plan that tried to regress would be asking the cluster to serve data it
    /// no longer has.
    #[must_use]
    pub const fn is_past_cutover(self) -> bool {
        self.has_published_routing()
    }

    /// Whether readiness gating is reported for this phase.
    ///
    /// The phases where "the children are not serving" is the reason nothing is
    /// happening: `BaseSeeded`, where the proof runs *before* the fence so an
    /// unready child never holds the parent unwritable; `Fenced`, where it is
    /// re-run under the fence; and `CutoverCommitted`, where it gates retirement
    /// of the only remaining copy of the pre-split range.
    ///
    /// Earlier phases are waiting on a request, and `ChildrenReady` or later have
    /// already cleared it - a block there is a different, louder problem.
    #[must_use]
    pub const fn needs_children_ready(self) -> bool {
        matches!(
            self,
            Self::BaseSeeded | Self::Fenced | Self::CutoverCommitted
        )
    }

    /// Every phase, for exhaustive checks over the lifecycle.
    pub const ALL: [Self; 9] = [
        Self::Planned,
        Self::ChildrenProvisioning,
        Self::BaseSeeded,
        Self::Fenced,
        Self::ChildrenReady,
        Self::CutoverCommitted,
        Self::ParentRetiring,
        Self::Completed,
        Self::Failed,
    ];

    pub(crate) fn can_advance_to(self, next: Self) -> bool {
        self == next
            || matches!(
                (self, next),
                (Self::Planned, Self::ChildrenProvisioning | Self::Failed)
                    | (Self::ChildrenProvisioning, Self::BaseSeeded | Self::Failed)
                    | (Self::BaseSeeded, Self::Fenced | Self::Failed)
                    // `Fenced → BaseSeeded` is the unfence-on-race regression, and
                    // it is the only one: it happens before any routing moved.
                    | (Self::Fenced, Self::BaseSeeded | Self::ChildrenReady | Self::Failed)
                    | (Self::ChildrenReady, Self::CutoverCommitted | Self::Failed)
                    | (Self::CutoverCommitted, Self::ParentRetiring | Self::Failed)
                    | (Self::ParentRetiring, Self::Completed | Self::Failed)
            )
    }

    /// Encodes the phase as a canonical discriminant byte.
    ///
    /// `ChildrenReady` is 6: the value `ParentRetiring` used to hold. Numbers are
    /// never reused for a different meaning, so a plan written by an older build
    /// is rejected rather than silently reinterpreted.
    #[must_use]
    pub const fn encode_byte(self) -> u8 {
        match self {
            Self::Planned => 0,
            Self::ChildrenProvisioning => 1,
            Self::BaseSeeded => 2,
            Self::Fenced => 3,
            Self::ChildrenReady => 6,
            Self::CutoverCommitted => 4,
            Self::ParentRetiring => 5,
            Self::Completed => 7,
            Self::Failed => 8,
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
            1 => Ok(Self::ChildrenProvisioning),
            2 => Ok(Self::BaseSeeded),
            3 => Ok(Self::Fenced),
            4 => Ok(Self::CutoverCommitted),
            5 => Ok(Self::ParentRetiring),
            6 => Ok(Self::ChildrenReady),
            7 => Ok(Self::Completed),
            8 => Ok(Self::Failed),
            _ => Err(SplitError::BadPhase { found: byte }),
        }
    }
}

/// Why a split plan was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
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
/// `split_hash` is the deterministic midpoint boundary (u128) of a hash
/// parent range; `split_key` is the real-key boundary of an ordered parent
/// range (`[start, split_key)` + `[split_key, end)`, strictly interior).
/// Exactly one is set: hash splits use the midpoint, ordered splits use a
/// data-driven pivot (approximately balancing logical bytes). The
/// architecture permits arbitrary future split points by replacing the
/// boundary computation, never the plan shape. Child replica sets
/// initially inherit the parent's desired voters (logical split first,
/// physical rebalance follows as ordinary tablets).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitPlan {
    /// Plan identity (shares the control `next_plan_id` sequence with
    /// migrations: ids are unique across plan kinds).
    pub id: PlanId,
    /// Parent tablet subdividing.
    pub parent: TabletId,
    /// Deterministic split boundary for hash parents (midpoint of parent
    /// range); zero for ordered splits.
    pub split_hash: u128,
    /// Real-key split boundary for ordered parents (`None` for hash).
    pub split_key: Option<Vec<u8>>,
    /// Left child (`[parent.start, boundary)`).
    pub left: TabletId,
    /// Right child (`[boundary, parent.end)`).
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: PlanId,
        parent: TabletId,
        split_hash: u128,
        split_key: Option<Vec<u8>>,
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
            split_key,
            left,
            right,
            replicas,
            generation,
            phase: SplitPhase::Planned,
        })
    }

    /// Whether this is an ordered (real-key) split.
    #[must_use]
    pub fn is_ordered(&self) -> bool {
        self.split_key.is_some()
    }

    /// Moves the plan to a new phase.
    #[must_use]
    pub fn advance(&self, phase: SplitPhase) -> Self {
        Self {
            id: self.id,
            parent: self.parent,
            split_hash: self.split_hash,
            split_key: self.split_key.clone(),
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
    /// generation u64, phase u8, count u32, replicas[count] u64, key_flag
    /// u8, [key_len u32, key bytes]` (key section present iff the flag is
    /// 1; v2 appends the ordered boundary, never reuses tags).
    #[must_use]
    pub fn encode_to_vec(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 * 5 + 16 + 1 + 4 + 8 * self.replicas.len() + 5);
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
        match &self.split_key {
            None => out.push(0),
            Some(key) => {
                out.push(1);
                let len = u32::try_from(key.len()).unwrap_or(u32::MAX);
                out.extend_from_slice(&len.to_le_bytes());
                out.extend_from_slice(key);
            }
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
        if input.len() < 8 + 8 + 16 + 8 + 8 + 8 + 1 + 4 + 1 {
            return Err(Fault::Truncated);
        }
        let id = PlanId::from_u64(u64::from_le_bytes(input[..8].try_into().unwrap_or([0; 8])));
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
        let mut at = 61 + 8 * count;
        if input.len() < at + 1 {
            return Err(Fault::Truncated);
        }
        let mut replicas = Vec::with_capacity(count);
        for index in 0..count {
            let base = 61 + 8 * index;
            replicas.push(NodeId::from_u64(u64::from_le_bytes(
                input[base..base + 8].try_into().unwrap_or([0; 8]),
            )));
        }
        let split_key = match input[at] {
            0 => {
                at += 1;
                None
            }
            1 => {
                at += 1;
                if input.len() < at + 4 {
                    return Err(Fault::Truncated);
                }
                let len =
                    u32::from_le_bytes(input[at..at + 4].try_into().unwrap_or([0; 4])) as usize;
                at += 4;
                if input.len() < at + len {
                    return Err(Fault::Truncated);
                }
                let key = input[at..at + len].to_vec();
                at += len;
                Some(key)
            }
            _ => return Err(Fault::Truncated),
        };
        if input.len() != at {
            return Err(if input.len() < at {
                Fault::Truncated
            } else {
                Fault::TrailingBytes
            });
        }
        Self::new(
            id, parent, split_hash, split_key, left, right, replicas, generation,
        )
        .map(|mut plan| {
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
            PlanId::from_u64(1),
            TabletId::from_u64(7),
            1u128 << 127,
            None,
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

    fn ordered_sample() -> SplitPlan {
        SplitPlan::new(
            PlanId::from_u64(2),
            TabletId::from_u64(8),
            0,
            Some(b"m".to_vec()),
            TabletId::from_u64(103),
            TabletId::from_u64(104),
            vec![NodeId::from_u64(1)],
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
        assert!(!plan.is_ordered());
        assert!(SplitPhase::Completed.is_terminal());
        let ordered = ordered_sample();
        assert!(ordered.is_ordered());
        let back = SplitPlan::decode_exact(&ordered.encode_to_vec()).expect("decodes");
        assert_eq!(back, ordered);
    }

    #[test]
    fn degenerate_children_fail() {
        assert!(matches!(
            SplitPlan::new(
                PlanId::from_u64(1),
                TabletId::from_u64(7),
                0,
                None,
                TabletId::from_u64(7),
                TabletId::from_u64(8),
                vec![NodeId::from_u64(1)],
                PlacementVersion::INITIAL,
            ),
            Err(SplitError::BadChildren)
        ));
    }

    /// Every phase that can follow `from`, per the transition table.
    fn successors(from: SplitPhase) -> Vec<SplitPhase> {
        SplitPhase::ALL
            .into_iter()
            .filter(|to| from.can_advance_to(*to))
            .collect()
    }

    /// Routing cannot be published without a readiness proof behind it.
    ///
    /// `CutoverCommitted` is the instant the parent stops serving and the
    /// children start, so reaching it is the same decision as "serve the
    /// children". Before this table existed, `Fenced → CutoverCommitted` was
    /// legal: a group whose `ensure_group` request had been accepted - no
    /// voters, no leader, no data - could be handed every request in the
    /// namespace, and the parent could then be retired from under it.
    ///
    /// Asserted as a property of the transition table rather than of a call
    /// site, because the table is the only thing every future caller has to go
    /// through.
    #[test]
    fn publishing_routing_requires_a_readiness_proof() {
        assert_eq!(
            successors(SplitPhase::Fenced),
            vec![
                SplitPhase::BaseSeeded,
                SplitPhase::Fenced,
                SplitPhase::ChildrenReady,
                SplitPhase::Failed,
            ],
            "Fenced reaches ChildrenReady; it must not reach CutoverCommitted directly"
        );
        assert_eq!(
            successors(SplitPhase::ChildrenReady),
            vec![
                SplitPhase::ChildrenReady,
                SplitPhase::CutoverCommitted,
                SplitPhase::Failed,
            ],
            "the proof is the only thing between the fence and published routing"
        );
        // The only two ways to hold `CutoverCommitted` are to already hold it
        // (re-driving a plan mid-pass must be a no-op, not an error) and to
        // arrive from `ChildrenReady`.
        for from in SplitPhase::ALL {
            if matches!(
                from,
                SplitPhase::ChildrenReady | SplitPhase::CutoverCommitted
            ) {
                continue;
            }
            assert!(
                !from.can_advance_to(SplitPhase::CutoverCommitted),
                "{from:?} must not publish routing without ChildrenReady"
            );
        }
    }

    /// The parent cannot be retired before the children are authoritative.
    ///
    /// Retirement is irreversible for the parent's data - it removes the only
    /// replica of the pre-split range. So the transition out of
    /// `CutoverCommitted` has to be reachable only from a phase that already
    /// published routing, and no phase may walk back to one that has not.
    #[test]
    fn a_plan_cannot_regress_past_the_cutover() {
        assert_eq!(
            successors(SplitPhase::CutoverCommitted),
            vec![
                SplitPhase::CutoverCommitted,
                SplitPhase::ParentRetiring,
                SplitPhase::Failed,
            ]
        );
        for from in SplitPhase::ALL {
            for to in successors(from) {
                if from.is_past_cutover() && !to.is_terminal() {
                    assert!(
                        to.is_past_cutover(),
                        "{from:?} -> {to:?} walks back to a phase that has not published \
                         routing; the parent is already a tombstone with a redirect"
                    );
                }
            }
        }
    }

    /// A split is resumable from every non-terminal phase and terminal from
    /// none, so a driver that restarts mid-flight always has work it may do.
    #[test]
    fn terminal_and_resumable_phases_partition_the_lifecycle() {
        for phase in SplitPhase::ALL {
            assert_eq!(
                phase.is_resumable(),
                !phase.is_terminal(),
                "{phase:?} is reported as both terminal and resumable"
            );
        }
    }

    /// A phase survives its own encoding, and no discriminant names two phases.
    #[test]
    fn every_phase_round_trips_through_its_discriminant() {
        let mut seen = std::collections::BTreeSet::new();
        for phase in SplitPhase::ALL {
            let byte = phase.encode_byte();
            assert!(seen.insert(byte), "{phase:?} reuses discriminant {byte}");
            assert_eq!(SplitPhase::decode_byte(byte), Ok(phase));
        }
        // 9 is the next free discriminant; anything at or above it is a phase
        // this build does not know, which must be refused rather than guessed.
        assert!(matches!(
            SplitPhase::decode_byte(9),
            Err(SplitError::BadPhase { found: 9 })
        ));
    }

    /// Readiness gating is reported for exactly the phases where "the children
    /// are not serving" explains why nothing is happening.
    #[test]
    fn readiness_is_reported_where_it_can_block() {
        for phase in [
            SplitPhase::BaseSeeded,
            SplitPhase::Fenced,
            SplitPhase::CutoverCommitted,
        ] {
            assert!(phase.needs_children_ready(), "{phase:?} gates on readiness");
        }
        for phase in [
            SplitPhase::Planned,
            SplitPhase::ChildrenProvisioning,
            SplitPhase::ChildrenReady,
            SplitPhase::ParentRetiring,
            SplitPhase::Completed,
            SplitPhase::Failed,
        ] {
            assert!(
                !phase.needs_children_ready(),
                "{phase:?} is not waiting on a readiness proof"
            );
        }
    }
}
