//! Kivi-owned consensus concepts.
//!
//! These types are the only consensus vocabulary the rest of the workspace
//! may use. They are defined in terms of Kivi identities
//! ([`kivi_types`]) and plain data — never in terms of `OpenRaft` types, so
//! an `OpenRaft` upgrade (or replacement) cannot ripple past this crate.

use core::fmt;

use kivi_types::{NodeId, TabletId};

/// Identity of one replicated tablet group.
///
/// This stage replicates exactly one tablet, so the group is 1:1 with its
/// tablet; the newtype keeps that coincidence from becoming structural.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConsensusGroupId(TabletId);

impl ConsensusGroupId {
    /// Wraps the tablet this group replicates.
    #[must_use]
    pub const fn of_tablet(tablet: TabletId) -> Self {
        Self(tablet)
    }

    /// Returns the replicated tablet.
    #[must_use]
    pub const fn tablet(self) -> TabletId {
        self.0
    }
}

impl fmt::Display for ConsensusGroupId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "group({})", self.0)
    }
}

/// Identity of one replica (cluster node) inside a consensus group.
///
/// Backed by [`NodeId`]: the unit of Raft voting and of Kivi's static
/// cluster topology are the same process identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReplicaId(NodeId);

impl ReplicaId {
    /// Wraps the node serving this replica.
    #[must_use]
    pub const fn of_node(node: NodeId) -> Self {
        Self(node)
    }

    /// Returns the serving node.
    #[must_use]
    pub const fn node(self) -> NodeId {
        self.0
    }
}

impl fmt::Display for ReplicaId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "replica({})", self.0)
    }
}

/// Raft term, Kivi-owned. Terms order leader generations; a larger term
/// always supersedes a smaller one for the same group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConsensusTerm(u64);

impl ConsensusTerm {
    /// The zero term: no election has completed.
    pub const INITIAL: Self = Self(0);

    /// Wraps a raw term.
    #[must_use]
    pub const fn new(term: u64) -> Self {
        Self(term)
    }

    /// Returns the raw term.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ConsensusTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "term({})", self.0)
    }
}

/// Raft log index, Kivi-owned. Indices are per-group, 1-based once entries
/// exist; index 0 means "no entries".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConsensusLogIndex(u64);

impl ConsensusLogIndex {
    /// No entries.
    pub const NONE: Self = Self(0);

    /// Wraps a raw index.
    #[must_use]
    pub const fn new(index: u64) -> Self {
        Self(index)
    }

    /// Returns the raw index.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ConsensusLogIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "index({})", self.0)
    }
}

/// This replica's role in its group, as last observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReplicaRole {
    /// Holds a current lease on leadership (observed leader).
    Leader,
    /// Following the current leader.
    Follower,
    /// Campaigning (transient; observed during elections).
    Candidate,
}

impl fmt::Display for ReplicaRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Leader => write!(f, "leader"),
            Self::Follower => write!(f, "follower"),
            Self::Candidate => write!(f, "candidate"),
        }
    }
}

/// Where a non-leader believes clients should retry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LeaderHint {
    /// Best-known leader, if any is known.
    pub leader: Option<ReplicaId>,
}

impl LeaderHint {
    /// No leader currently known.
    #[must_use]
    pub const fn unknown() -> Self {
        Self { leader: None }
    }

    /// Points at a known leader.
    #[must_use]
    pub const fn known(leader: ReplicaId) -> Self {
        Self {
            leader: Some(leader),
        }
    }
}

/// Kivi-owned linearizable-read barrier: the leadership that produced it
/// plus the inclusive apply boundary a `Latest` read must cover.
///
/// This preserves the full 0.10 `ReadLogId` semantics at the crate
/// boundary (leadership identity + apply boundary, never a bare index)
/// so future `AtLeast` tokens, follower reads, and `WriteGuard`
/// freshness work can build on it without recovering thrown-away
/// information. Serving `Latest` still means: the leader proved
/// freshness with a quorum, and local state applied through `boundary`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReadBarrier {
    /// Term of the leadership that produced the barrier.
    pub term: ConsensusTerm,
    /// Leader of the producing leadership.
    pub leader: ReplicaId,
    /// Inclusive apply boundary: serve only from state applied through
    /// at least this log index.
    pub boundary: ConsensusLogIndex,
}

/// Read-only per-group diagnostics for the admin plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConsensusStatus {
    /// Group described.
    pub group: ConsensusGroupId,
    /// This replica.
    pub replica: ReplicaId,
    /// Observed role.
    pub role: ReplicaRole,
    /// Observed term.
    pub term: ConsensusTerm,
    /// Best-known leader.
    pub leader: Option<ReplicaId>,
    /// Last log entry (voted or appended).
    pub last_log: ConsensusLogIndex,
    /// Last committed entry.
    pub committed: ConsensusLogIndex,
    /// Last entry applied to the tablet state machine.
    pub applied: ConsensusLogIndex,
}

/// Kivi-owned consensus failures. These — never `OpenRaft` error types —
/// cross the crate boundary and travel to clients over the native protocol.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ConsensusError {
    /// This replica is not the leader; retry at the hint when present.
    #[error("not leader (hint: {hint:?})")]
    NotLeader {
        /// Where to retry.
        hint: LeaderHint,
    },
    /// No leader is currently known (election in flight or quorum lost).
    #[error("leader unknown")]
    LeaderUnknown,
    /// The consensus subsystem cannot make progress (storage failure,
    /// shutdown, or lost quorum).
    #[error("consensus unavailable: {reason}")]
    Unavailable {
        /// Human-readable cause (logged verbatim, never a client secret).
        reason: String,
    },
    /// The replica is shutting down.
    #[error("consensus shutting down")]
    ShuttingDown,
}

/// Narrow Kivi-owned consensus interface.
///
/// Deliberately small: only operations with actual callers exist here.
/// `OpenRaft` stays an implementation detail behind this trait — including
/// its `Raft`, `LogId`, and `Vote` types, which must never appear in these
/// signatures.
pub trait ConsensusCore: Send + Sync {
    /// Proposes one deterministic mutation envelope for commit. Success
    /// means quorum-committed, leader-applied, and dedup-installed (the
    /// strong write contract); the returned bytes are the deterministic
    /// outcome for the caller's request identity.
    fn propose(
        &self,
        envelope: ProposeEnvelope,
    ) -> impl Future<Output = Result<ProposedOutcome, ConsensusError>> + Send;

    /// Establishes a linearizable-read barrier: on success the leader has
    /// confirmed its leadership with a quorum and the returned index is
    /// the applied watermark a `Latest` read must cover.
    fn ensure_linearizable(
        &self,
    ) -> impl Future<Output = Result<ConsensusLogIndex, ConsensusError>> + Send;

    /// Returns current read-only diagnostics.
    fn status(&self) -> ConsensusStatus;

    /// Stops the group cleanly.
    fn shutdown(&self) -> impl Future<Output = ()> + Send;
}

/// What [`ConsensusCore::propose`] carries across the boundary.
///
/// The consensus layer treats the mutation as opaque deterministic bytes;
/// their meaning (`MutationIR`, request identity, expected outcome) is owned
/// by the tablet layer. Stage G replaces the payload sketch with
/// `ReplicatedMutation`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposeEnvelope {
    /// Group addressed.
    pub group: ConsensusGroupId,
    /// Deterministic mutation bytes (canonical, versioned, Kivi-owned).
    pub payload: Vec<u8>,
}

/// Outcome installed by [`ConsensusCore::propose`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposedOutcome {
    /// Committed log index of the proposal.
    pub index: ConsensusLogIndex,
    /// Deterministic outcome bytes for the request identity.
    pub outcome: Vec<u8>,
}
