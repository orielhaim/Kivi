//! First replicated Kivi node on `OpenRaft` 0.10.
//!
//! [`ReplicatedNode`] binds one consensus group to one process: durable
//! log ([`DurableRaftStore`]), Kivi state
//! machine ([`ReplicatedStateMachine`]),
//! shared peer mesh ([`PeerTransport`]),
//! and the `OpenRaft` instance gluing them through the router boundary.
//!
//! ## Owner thread + channel front
//!
//! 0.10's `single-threaded` mode makes the `Raft` handle `!Send`: it lives
//! on the Compio owner thread spawned here (`kivi-consensus`), alongside
//! the transport tasks and the snapshot reassembly state. The public
//! [`ReplicatedNode`] front stays `Send + Sync` — one bounded
//! [`async_channel`] request queue plus `Send` oneshot replies — so
//! callers on any runtime (including the Tokio admin plane) keep ordinary
//! `Send` futures. No Tokio exists on either side of the boundary.
//!
//! ```text
//! caller (any thread/runtime)
//!   │ propose / barrier / status / serve / seal
//!   ▼
//! bounded owner queue (256)
//!   ▼
//! owner loop (Compio reactor thread)
//!   ├── per-request tasks (never awaited inline: client_write, barriers,
//!   │   and snapshot installs can all pend indefinitely while status,
//!   │   diagnostics, and peer RPCs keep flowing)
//!   ├── Raft handle (!Send, pinned here)
//!   ├── snapshot reassembly table
//!   └── store / machine / transport clones
//! ```
//!
//! ## Leader mutation path
//!
//! ```text
//! client mutation
//!     ↓ leader dedup check (read-only; retries answer from any replica)
//!     ↓ sidecar gate (chunked roots verified durable; chunked-base
//!       range patches restaged reusing untouched chunk ids)
//!     ↓ prepare deterministic MutationIR + expected outcome
//! OpenRaft client_write (quorum durable commit; followers gate the
//! flush on sidecar durability)
//!     ↓ state-machine apply + verification (every replica)
//! dedup outcome available
//!     ↓ reply
//! ```
//!
//! The Raft log is authoritative in replicated mode: no single-node
//! mutation WAL entry is appended alongside it (no double-logging). Large
//! values replicate as tiny `ReplaceChunkedRoot` mutations plus immutable
//! bulk sidecars (see [`crate::sidecar`] and [`crate::gate`]).
//!
//! ## Supported operations
//!
//! The small/state-local semantic set replicates: inline `SET`, `DELETE`,
//! `EXISTS`/reads, conditional `SET` (materialized at prepare),
//! `COUNTER_ADD`, expiry mutations, `PERSIST`, and `SETRANGE` against
//! inline bases. Large values replicate as [`Operation::SetChunked`] and
//! friends once staged durably locally; `SETRANGE` against a chunked base
//! restages here (untouched chunks reuse via content addressing) and
//! followers fetch only missing sidecars before acknowledging.
//!
//! ## Leader-only writes
//!
//! Only the leader accepts strong mutations. Followers answer
//! Kivi-owned [`ConsensusError::NotLeader`] with a leader hint (or
//! [`ConsensusError::LeaderUnknown`]); `OpenRaft` error types never cross
//! the native protocol.
//!
//! ## Linearizable reads and follower reads (Phase 7 consistency layer)
//!
//! Reads execute under explicit contracts through the caller-side read
//! front and the node-wide [`ConsistencyHub`]:
//!
//! ```text
//! Latest .................. quorum `ReadIndex` barrier on the leader
//!                           (0.10 `ensure_linearizable` awaits readiness
//!                           itself), then local applied state; optional
//!                           Almost-Local / roster-lease fast paths reuse
//!                           explicit evidence and fall back otherwise.
//! AtLeast(CommitToken) ..... lineage-validated (tablet + epoch, never
//!                           just position); serves from any replica at or
//!                           beyond the token, waits bounded when behind,
//!                           rejects foreign tokens as `StaleToken`.
//! BoundedStale(duration) ... serves only from a `FreshnessReceipt`
//!                           proving the bound, else escalates to a fresh
//!                           barrier (which satisfies any bound) — never
//!                           weak data labeled successful.
//! Any ..................... local applied state, no freshness claim.
//! ```
//!
//! Every served read returns its proof triple (authority, applied
//! position, serve path). `Latest` on a follower is `NotLeader`.
//!
//! ## Storage health
//!
//! Consensus persistence failure fails closed: the log store's barriers
//! already refuse to acknowledge on I/O faults, and a state machine that
//! latched unhealthy (divergence or snapshot failure) fails reads and is
//! reported through [`NodeStatus`]. Fault injection lives at the storage
//! adapter level (state-machine snapshot barrier, log-store barriers).

use std::cell::RefCell;
use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use kivi_redundancy::LocalFragmentStore;
use kivi_state::{DurableOutcome, OpError, Operation, OperationResult};
use kivi_types::{
    ClusterId, CommitPosition, FreshnessReceipt, IdempotencyKey, MutationIdentity, NamespaceId,
    NodeId, NodeIncarnation, ReadAuthorityProvider, ReadContext, ReadContract, ReadReceipt,
    ReplicaFreshness, ServePath, TabletAuthority, TabletId, Ticks, WallTimestamp,
};
use openraft::storage::RaftStateMachine as _;
use openraft::type_config::async_runtime::watch::WatchReceiver as _;

use crate::cluster::{Bootstrap, ClusterTopology, DurableClusterView, classify_bootstrap};
use crate::command::ConsensusCommand;
use crate::config::{KiviTypeConfig, cluster_config};
use crate::consistency::{ConsistencyHub, ReadPlan, ServedRead, ServedScan};
use crate::gate::SidecarGate;
use crate::mutation::ReplicatedMutation;
use crate::peer::{
    PeerChunkResponse, PeerManifestResponse, PeerRequest, PeerResponse, PeerRpcError,
    PeerSnapshotRequest,
};
use crate::router::{GroupNetworkFactory, PeerRouter, decode_snapshot_meta, serve_peer_request};
use crate::sidecar::SidecarStore;
use crate::state_machine::{ProposalGate, ReplicatedStateMachine};
use crate::store::{ConsensusLogStore, DurableRaftStore};
use crate::transport::{PeerHandler, PeerStats, PeerTransport, TransportConfig};
use crate::types::{
    ConsensusError, ConsensusGroupId, ConsensusLogIndex, ConsensusTerm, ControlProposeError,
    LeaderHint, ReadBarrier, ReplicaId, ReplicaRole,
};

/// How long `AtLeast` waits for local applied coverage before failing.
const AT_LEAST_TIMEOUT: Duration = Duration::from_secs(10);

/// Maximum for one lease-batch round trip. Renew cadence is
/// protocol-timed (engine deadlines); a slow batch only delays
/// convergence to the next tick, never safety.
const LEASE_RPC_TIMEOUT: Duration = Duration::from_secs(1);

/// Maximum for one responder-coverage poll. Healthy responders answer
/// in milliseconds; the bound only cuts off dead links per round.
const COVERAGE_POLL_TIMEOUT: Duration = Duration::from_millis(500);

/// Total bounded wait for responder coverage per write. Replication lag
/// (milliseconds) must not fence a live responder, but a dead one must
/// not stall writes forever: after this window the uncovered responders
/// are fenced and the write fails retryable instead of completing
/// uncovered.
const COVERAGE_WAIT: Duration = Duration::from_secs(3);

/// Depth of the owner request queue. Every request carries its own
/// reply channel, so depth only bounds burst memory; a full queue fails
/// the submitter with `ShuttingDown`-class backpressure instead of
/// growing.
pub(crate) const OWNER_QUEUE: usize = 256;

/// Replicated-node construction parameters. Everything is validated
/// before serving: topology incoherence, identity conflicts, and corrupt
/// history all fail `open`, never mid-operation.
#[derive(Debug)]
pub struct NodeConfig {
    /// Data-directory root (consensus WAL lane + state-machine images
    /// live beneath it; the process lock is held for the node's life).
    pub data_dir: PathBuf,
    /// Namespace served.
    pub namespace: NamespaceId,
    /// Tablet replicated (exactly one in this stage).
    pub tablet: TabletId,
    /// Replica authority (epoch/guard fencing, stage-1 static).
    pub authority: TabletAuthority,
    /// This process's node identity (operator-declared; verified against
    /// the data directory, adopted when fresh).
    pub local: NodeId,
    /// Static cluster definition (identity, members, assignment).
    pub topology: ClusterTopology,
    /// WAL segment rotation target in bytes.
    pub segment_target_bytes: u64,
    /// Peer transport tuning.
    pub transport: TransportConfig,
    /// Expected peer certificates (trust anchors) by node for QUIC/TLS
    /// verification. Every other member must be present; the transport
    /// refuses to open otherwise. Empty with `insecure_peer_tls` (lab
    /// only) or when no peers are configured yet.
    pub peer_certs: HashMap<NodeId, Vec<u8>>,
    /// Test-only escape hatch for `kivi-lab` (ephemeral ports and fresh
    /// data directories per run make static pins impractical there).
    /// Never set in production: without verification any network peer
    /// could impersonate a member.
    pub insecure_peer_tls: bool,
    /// Whether the leader pre-distributes chunked sidecars to followers
    /// before proposing the tiny root (`true` in production). `false`
    /// forces the append-gate fallback path (correctness probe: proves
    /// preflight is only an optimization).
    pub preflight_enabled: bool,
    /// Deployment lease-timing contract (bounded clock-rate drift plus
    /// guard/lease/renew durations). The `RosterLease` backend is
    /// unavailable unless these validate; reads then use Lazy-ALR or the
    /// conservative barrier.
    pub lease_params: kivi_types::LeaseParams,
}

/// Why a replicated node could not open.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum NodeOpenError {
    /// Static topology incoherent.
    #[error("topology invalid: {reason}")]
    Topology {
        /// Human-readable cause.
        reason: String,
    },
    /// Data-directory identity failure.
    #[error("data directory failed: {reason}")]
    DataDir {
        /// Human-readable cause.
        reason: String,
    },
    /// Consensus log store failed to open.
    #[error("consensus log store failed: {reason}")]
    LogStore {
        /// Human-readable cause.
        reason: String,
    },
    /// State machine failed to open.
    #[error("state machine failed: {reason}")]
    StateMachine {
        /// Human-readable cause.
        reason: String,
    },
    /// Peer mesh failed to open.
    #[error("peer mesh failed: {reason}")]
    Transport {
        /// Human-readable cause.
        reason: String,
    },
    /// Startup config conflicts with durable identity/membership.
    #[error("bootstrap refused: {reason}")]
    Bootstrap {
        /// Human-readable cause.
        reason: String,
    },
    /// `OpenRaft` membership installation failed on first formation.
    #[error("group initialization failed: {reason}")]
    Initialize {
        /// Human-readable cause.
        reason: String,
    },
    /// Immutable sidecar store failed to open.
    #[error("sidecar store failed: {reason}")]
    Sidecar {
        /// Human-readable cause.
        reason: String,
    },
    /// Redundancy fragment store failed to open.
    #[error("fragment store failed: {reason}")]
    Fragments {
        /// Human-readable cause.
        reason: String,
    },
    /// Peer TLS identity failed to load or generate.
    #[error("peer TLS failed: {reason}")]
    Tls {
        /// Human-readable cause.
        reason: String,
    },
    /// The owner thread failed to start or died during startup.
    #[error("consensus owner failed: {reason}")]
    Owner {
        /// Human-readable cause.
        reason: String,
    },
}

/// Outcome of one proposed mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProposeOutcome {
    /// Quorum-committed, leader-applied, dedup-installed.
    Applied {
        /// Committed log index.
        index: ConsensusLogIndex,
        /// Deterministic outcome.
        outcome: OperationResult,
    },
    /// Same-identity retry answered from retained state (no proposal).
    Duplicate {
        /// Original outcome.
        outcome: OperationResult,
    },
    /// Read-only operation answered locally (no proposal).
    Read {
        /// Read outcome.
        outcome: OperationResult,
    },
    /// Terminally rejected without mutating (deterministic rejections
    /// never replicate; a retry re-evaluates against converged state).
    Rejected {
        /// Terminal outcome.
        outcome: DurableOutcome,
    },
}

/// Why a proposal failed. Leader routing failures are Kivi-owned
/// ([`ConsensusError`]); validation and capability failures are explicit.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ProposeError {
    /// Consensus routing failure (not leader / unknown / unavailable).
    #[error("{0}")]
    Consensus(#[from] ConsensusError),
    /// A retried identity fell below the session floor.
    #[error("mutation identity expired below the session floor")]
    Expired,
    /// The session holds too many unacknowledged outcomes.
    #[error("session outcome window exhausted")]
    Overloaded,
    /// The operation needs unreplicated chunk sidecars (stage-1
    /// capability boundary).
    #[error("unsupported in replicated mode: {reason}")]
    Unsupported {
        /// What is missing.
        reason: String,
    },
    /// Read-only opcode validation failure.
    #[error("operation rejected: {0}")]
    Op(#[from] OpError),
    /// The tablet is fenced for split/merge cutover: mutating proposes
    /// fail so a bounded final tail installs without loss. Retryable
    /// (shaped as `Overloaded` on the wire); reads still serve.
    #[error("tablet {tablet} fenced for topology cutover")]
    Fenced {
        /// Fenced tablet.
        tablet: TabletId,
    },
    /// Responder coverage could not be established before the bound: the
    /// write stays unacknowledged (never externally completed uncovered)
    /// and the uncovered responder was fenced. Shaped as
    /// `CoverageUncertain` on the wire: the outcome is ambiguous (it may
    /// have committed), so callers must not blindly retry non-idempotent
    /// writes — read-verify first, or re-drive under the same identity.
    #[error("responder coverage timed out: {detail}")]
    ResponderCoverage {
        /// Which responder(s) stayed uncovered.
        detail: String,
    },
}

/// Why a strong read failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ReadError {
    /// Consensus routing failure (not leader / unknown / unavailable).
    #[error("{0}")]
    Consensus(#[from] ConsensusError),
    /// Read validation failure.
    #[error("read rejected: {0}")]
    Op(#[from] OpError),
    /// The local-read path was asked to serve a mutating operation.
    #[error("not a read operation")]
    NotARead,
    /// Applied coverage never arrived before the deadline.
    #[error("applied coverage timed out")]
    CoverageTimeout,
    /// The `AtLeast` token's lineage is foreign to this replica (wrong
    /// tablet, superseded epoch, or unassigned position). Waiting cannot
    /// cure it: refresh the token from a read on the token's lineage.
    #[error("stale read token: {detail}")]
    StaleToken {
        /// Which lineage check failed.
        detail: String,
    },
}

/// Per-replicated-tablet diagnostics for the admin plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeStatus {
    /// Group served.
    pub group: ConsensusGroupId,
    /// This replica.
    pub replica: ReplicaId,
    /// Observed role.
    pub role: ReplicaRole,
    /// Observed term.
    pub term: ConsensusTerm,
    /// Best-known leader.
    pub leader: Option<ReplicaId>,
    /// Last log entry.
    pub last_log: ConsensusLogIndex,
    /// Last committed entry.
    pub committed: ConsensusLogIndex,
    /// Last applied entry.
    pub applied: ConsensusLogIndex,
    /// Applied commit position through the index mapping.
    pub applied_commit: CommitPosition,
    /// Installed snapshot base.
    pub snapshot: Option<ConsensusLogIndex>,
    /// Last purged log index (`None` before the first purge).
    pub purged: Option<ConsensusLogIndex>,
    /// Whether the replica serves reads and applies.
    pub healthy: bool,
    /// Latched failure detail.
    pub detail: Option<String>,
    /// Per-peer connection diagnostics.
    pub peers: HashMap<NodeId, PeerStats>,
    /// Immutable sidecar replication diagnostics.
    pub sidecar: crate::sidecar::SidecarMetricsSnapshot,
    /// Sidecar preflight diagnostics (attempts, quorum-ready, fallbacks).
    pub preflight: crate::preflight::PreflightMetricsSnapshot,
    /// Consistency-layer counters for this tablet: contract mix,
    /// local/authority split, waits, escalations, cache and fast-path
    /// hits/fallbacks, fencing rejects.
    pub consistency: kivi_types::ConsistencySnapshot,
}

type Reply<T> = futures::channel::oneshot::Sender<T>;

/// Handles a dynamically created replica: the worker-built state
/// machine and authority the front inserts into its caller-side maps.
#[derive(Debug, Clone)]
pub(crate) struct NewReplica {
    /// Local state machine for the new tablet.
    pub(crate) machine: ReplicatedStateMachine,
    /// Replica authority.
    pub(crate) authority: TabletAuthority,
}

/// Outcome of one Lazy-ALR synchronization: the ordered boundary whose
/// local apply covers the batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncOutcome {
    /// Raft log index whose local apply covers the batch: the appended
    /// fence, or a subsuming write ordered after batch formation.
    pub boundary: u64,
    /// Whether a concurrent write subsumed the extra fence entry.
    pub subsumed: bool,
}

/// One unit of work for the owner thread. Every variant carries its reply
/// channel; dropping the reply (owner shutdown race) surfaces as
/// `ShuttingDown` on the caller, never a hang.
pub(crate) enum OwnerRequest {
    /// Quorum-commit one deterministic command.
    Propose {
        /// Addressed group.
        group: ConsensusGroupId,
        /// Deterministic command plus expectation.
        command: ConsensusCommand,
        /// Mapped proposal outcome.
        reply: Reply<Result<ProposeOutcome, ProposeError>>,
    },
    /// Quorum-commit one typed control-plane mutation. Control entries
    /// install no client outcome (like barriers and membership), so
    /// this answers the committed index directly instead of forcing a
    /// phantom outcome through the tablet proposal contract.
    ProposeControl {
        /// Addressed group (always the system control group).
        group: ConsensusGroupId,
        /// Typed control mutation.
        mutation: kivi_control::ControlMutation,
        /// Committed index or typed control-proposal error.
        reply: Reply<Result<ConsensusLogIndex, ControlProposeError>>,
    },
    /// Quorum-commit one chunked operation with preflight: the owner
    /// pre-distributes sidecars to a write quorum, then prepares the
    /// deterministic mutation against fresh state (condition evaluation
    /// belongs at this boundary, never before a multi-second preflight)
    /// and commits it. Small same-tablet writes proceed meanwhile: this
    /// request runs on its own task and only the final tiny root enters
    /// Raft ordering.
    ProposeChunked {
        /// Addressed group.
        group: ConsensusGroupId,
        /// Staged chunked operation (sidecars already durable locally).
        op: Operation,
        /// Client identity for exactly-once semantics.
        identity: Option<MutationIdentity>,
        /// Idempotency key, if any.
        idempotency: Option<IdempotencyKey>,
        /// Client-boundary timestamp (replicas apply the leader's
        /// materialized stamp).
        now: WallTimestamp,
        /// Mapped proposal outcome.
        reply: Reply<Result<ProposeOutcome, ProposeError>>,
    },
    /// Establish a `ReadIndex` linearizable barrier.
    Barrier {
        /// Addressed group.
        group: ConsensusGroupId,
        /// Leadership-bound barrier or routing failure.
        reply: Reply<Result<ReadBarrier, ConsensusError>>,
    },
    /// Wait for local applied coverage of `index` (bounded).
    WaitApplied {
        /// Addressed group.
        group: ConsensusGroupId,
        /// Index to cover.
        index: u64,
        /// Maximum to wait (capped owner-side by `AT_LEAST_TIMEOUT`).
        /// The caller derives this from its read context; the owner never
        /// waits longer no matter what arrives here.
        timeout: Duration,
        /// Covered (`Ok`) or timed out (`Err(())`).
        reply: Reply<Result<(), ()>>,
    },
    /// Order one Lazy-ALR batch boundary: on the leader, subsume with the
    /// committed index when it already covers formation, else append the
    /// fence; on a follower, forward the request to the leader over the
    /// peer mesh. Any failure (not leader, stale leader view, transport,
    /// commit refusal) answers `Err`, and the caller falls back — the
    /// fence path never blocks a read past its wait budget.
    ProposeSync {
        /// Addressed group.
        group: ConsensusGroupId,
        /// Fence plus formation coordinates.
        sync: crate::command::AlrFenceSync,
        /// Maximum for the follower→leader forward hop (the fence commit
        /// itself is Raft-bounded; the local boundary wait reuses the
        /// read's wait budget through `WaitApplied`).
        timeout: Duration,
        /// Ordered boundary or the reason to fall back.
        reply: Reply<Result<SyncOutcome, ProposeError>>,
    },
    /// One lease-driver tick for `group`: reconcile term/voters into the
    /// tablet's lease engine, advance timers, announce or grant when
    /// designated, and flush outbound traffic over the peer mesh.
    LeaseTick {
        /// Addressed group.
        group: ConsensusGroupId,
    },
    /// Gather full diagnostics.
    Status {
        /// Addressed group.
        group: ConsensusGroupId,
        /// Diagnostics including per-peer stats (the owner holds the
        /// transport handle, so no second hop is needed).
        reply: Reply<NodeStatus>,
    },
    /// Serve one incoming peer RPC for `group`.
    Serve {
        /// Sending node.
        from: NodeId,
        /// Addressed group.
        group: ConsensusGroupId,
        /// Request body.
        request: PeerRequest,
        /// Family-matched answer or refusal.
        reply: Reply<Result<PeerResponse, PeerRpcError>>,
    },
    /// Trigger a snapshot seal and purge the log through its base.
    SnapshotPurge {
        /// Addressed group.
        group: ConsensusGroupId,
        /// Sealed base or human-readable reason.
        reply: Reply<Result<ConsensusLogIndex, String>>,
    },
    /// Register a learner on the group leader (`add_learner`). The
    /// reconciler calls this on the tablet leader with `blocking=true`
    /// so it returns once the learner is line-rate; promoting earlier
    /// can stall quorum progress.
    AddLearner {
        /// Addressed group.
        group: ConsensusGroupId,
        /// Joining node.
        node: NodeId,
        /// Dialable peer address for the joining node (rides as the
        /// node's `BasicNode::addr` in replicated membership).
        addr: String,
        /// Whether to block until the learner catches up.
        blocking: bool,
        /// Completion or human-readable reason (not-leader, lost
        /// leadership, storage).
        reply: Reply<Result<(), String>>,
    },
    /// Replace the voter set (`change_membership` with joint consensus
    /// handled internally by `OpenRaft`). New voters must already be
    /// learners; removed voters leave or linger per `retain`.
    ChangeMembership {
        /// Addressed group.
        group: ConsensusGroupId,
        /// Desired voters after the move.
        voters: std::collections::BTreeSet<u64>,
        /// Whether removed voters linger as learners (`true`) or leave
        /// the cluster (`false`). Planned migration uses `false`.
        retain: bool,
        /// Completion or human-readable reason.
        reply: Reply<Result<(), String>>,
    },
    /// Ask the leader to hand leadership to `to` (`Trigger::transfer_
    /// leader`). Fire-and-forget: ignored when the local replica is not
    /// the leader; the reconciler re-observes and continues.
    TransferLeader {
        /// Addressed group.
        group: ConsensusGroupId,
        /// Preferred successor (a retained, caught-up voter).
        to: NodeId,
        /// Dispatch result (not election outcome).
        reply: Reply<Result<(), String>>,
    },
    /// Observe actual voter/learner/leader state for the reconciler.
    ObserveMembership {
        /// Addressed group.
        group: ConsensusGroupId,
        /// Current observation.
        reply: Reply<Result<crate::control::ObservedMembership, String>>,
    },
    /// Create an empty local replica for a tablet this node has never
    /// hosted (target-side migration step). Opens the group store and
    /// state machine, spawns its `Raft` **without** initializing
    /// membership (the leader brings it via learner replication), and
    /// starts serving peer RPCs. Idempotent: an existing replica
    /// answers success without rebuilding.
    EnsureGroup {
        /// Addressed group.
        group: ConsensusGroupId,
        /// Tablet to host.
        tablet: TabletId,
        /// Current voter set (sidecar-gate sources).
        voters: std::collections::BTreeSet<u64>,
        /// Placement generation fencing this creation (persisted in the
        /// tombstone check: a stale create never resurrects a newer
        /// retirement).
        generation: u64,
        /// New replica handles (machine + authority for the front maps)
        /// or human-readable reason.
        reply: Reply<Result<NewReplica, String>>,
    },
    /// Retire the local replica after committed membership no longer
    /// requires it. Shuts down its `Raft`, unregisters the group,
    /// stops serving it, and writes a tombstone — durable data is
    /// retained conservatively for future GC. Idempotent.
    RetireGroup {
        /// Addressed group.
        group: ConsensusGroupId,
        /// Tablet retiring.
        tablet: TabletId,
        /// Placement generation fencing this retirement.
        generation: u64,
        /// Completion or human-readable reason.
        reply: Reply<Result<(), String>>,
    },
    /// Initialize one new tablet group exactly once (split child or merge
    /// target): the designated founder replica commits the initial
    /// membership; other replicas join through learner replication, never
    /// a second initialize (that would fork history). Idempotent: an
    /// already-initialized group answers success.
    InitializeGroup {
        /// Addressed group.
        group: ConsensusGroupId,
        /// Tablet initializing.
        tablet: TabletId,
        /// Initial voter set.
        voters: std::collections::BTreeSet<u64>,
        /// Dialable peer address per voter (rides as `BasicNode::addr`).
        peer_addrs: std::collections::BTreeMap<u64, String>,
        /// Completion or human-readable reason.
        reply: Reply<Result<(), String>>,
    },
    /// Suspend one peer link (partition test hook, drain tooling).
    SuspendPeer {
        /// Suspended peer.
        peer: NodeId,
        /// Completion (no payload; the suspend itself is synchronous on
        /// the owner).
        reply: Reply<()>,
    },
    /// Resume a suspended peer link.
    ResumePeer {
        /// Resumed peer.
        peer: NodeId,
        /// Completion.
        reply: Reply<()>,
    },
    /// Stop Raft and the mesh, then exit the owner loop.
    Shutdown,
}

impl OwnerRequest {
    /// Returns the addressed group for group-scoped requests. Node-wide
    /// requests (peer suspend/resume, shutdown) address no group.
    pub(crate) fn group(&self) -> Option<ConsensusGroupId> {
        match self {
            Self::Propose { group, .. }
            | Self::ProposeControl { group, .. }
            | Self::ProposeChunked { group, .. }
            | Self::Barrier { group, .. }
            | Self::WaitApplied { group, .. }
            | Self::ProposeSync { group, .. }
            | Self::LeaseTick { group }
            | Self::Status { group, .. }
            | Self::Serve { group, .. }
            | Self::SnapshotPurge { group, .. }
            | Self::AddLearner { group, .. }
            | Self::ChangeMembership { group, .. }
            | Self::TransferLeader { group, .. }
            | Self::ObserveMembership { group, .. }
            | Self::EnsureGroup { group, .. }
            | Self::RetireGroup { group, .. }
            | Self::InitializeGroup { group, .. } => Some(*group),
            Self::SuspendPeer { .. } | Self::ResumePeer { .. } | Self::Shutdown => None,
        }
    }

    /// Sends the loud "no local replica owns this group" answer for a
    /// request that reached a worker without a matching replica. The
    /// multi-tablet front and the group registry route correctly, so this
    /// fires only on genuine misrouting or unknown groups — never silent,
    /// never a hang. Node-wide requests have no group to mismatch and are
    /// ignored (the worker loop handles them directly).
    pub(crate) fn reply_unroutable(self, local: NodeId) {
        match self {
            Self::Propose { group, reply, .. } | Self::ProposeChunked { group, reply, .. } => {
                let _ = reply.send(Err(ProposeError::Consensus(ConsensusError::Unavailable {
                    reason: format!("group {group} not served by node {}", local.as_u64()),
                })));
            }
            Self::Barrier { group, reply } => {
                let _ = reply.send(Err(ConsensusError::Unavailable {
                    reason: format!("group {group} not served by node {}", local.as_u64()),
                }));
            }
            Self::WaitApplied { reply, .. } => {
                let _ = reply.send(Err(()));
            }
            Self::ProposeSync { group, reply, .. } => {
                let _ = reply.send(Err(ProposeError::Consensus(ConsensusError::Unavailable {
                    reason: format!("group {group} not served by node {}", local.as_u64()),
                })));
            }
            Self::Status { group, reply } => {
                let _ = reply.send(NodeStatus {
                    group,
                    replica: ReplicaId::of_node(local),
                    role: ReplicaRole::Follower,
                    term: ConsensusTerm::INITIAL,
                    leader: None,
                    last_log: ConsensusLogIndex::NONE,
                    committed: ConsensusLogIndex::NONE,
                    applied: ConsensusLogIndex::NONE,
                    applied_commit: CommitPosition::UNASSIGNED,
                    snapshot: None,
                    purged: None,
                    healthy: false,
                    detail: Some(format!(
                        "group {group} not served by node {}",
                        local.as_u64()
                    )),
                    peers: HashMap::new(),
                    sidecar: crate::sidecar::SidecarMetricsSnapshot {
                        bulk_bytes_sent: 0,
                        bulk_bytes_received: 0,
                        manifest_requests: 0,
                        chunk_requests: 0,
                        cache_hits: 0,
                        cache_misses: 0,
                        deduped_bytes: 0,
                        bulk_errors: 0,
                        verification_failures: 0,
                        pending_gated: 0,
                        inflight: 0,
                    },
                    preflight: crate::preflight::PreflightMetricsSnapshot {
                        attempts: 0,
                        quorum_ready: 0,
                        fallbacks: 0,
                        ready_replies: 0,
                        latency_us: 0,
                    },
                    consistency: kivi_types::ConsistencySnapshot::default(),
                });
            }
            Self::Serve { reply, .. } => {
                let _ = reply.send(Err(PeerRpcError {
                    detail: format!("group not served by node {}", local.as_u64()),
                }));
            }
            Self::SnapshotPurge { group, reply, .. } => {
                let _ = reply.send(Err(unroutable(group, local)));
            }
            Self::ProposeControl { group, reply, .. } => {
                let _ = reply.send(Err(ControlProposeError::Unavailable {
                    detail: unroutable(group, local),
                }));
            }
            Self::AddLearner { group, reply, .. }
            | Self::ChangeMembership { group, reply, .. }
            | Self::TransferLeader { group, reply, .. }
            | Self::InitializeGroup { group, reply, .. }
            | Self::RetireGroup { group, reply, .. } => {
                let _ = reply.send(Err(unroutable(group, local)));
            }
            Self::EnsureGroup { group, reply, .. } => {
                let _ = reply.send(Err(unroutable(group, local)));
            }
            Self::ObserveMembership { group, reply } => {
                let _ = reply.send(Err(unroutable(group, local)));
            }
            // Node-wide requests have no group to mismatch, and
            // group-scoped ticks without a replica drop (the next tick
            // retries): neither hangs the caller.
            Self::SuspendPeer { .. }
            | Self::ResumePeer { .. }
            | Self::Shutdown
            | Self::LeaseTick { .. } => {}
        }
    }
}

/// Loud "no local replica owns this group" reason for requests that
/// reached a worker without a matching replica.
fn unroutable(group: ConsensusGroupId, local: NodeId) -> String {
    format!("group {group} not served by node {}", local.as_u64())
}

/// One replicated Kivi node: a `Send + Sync` front over the Compio owner
/// thread driving a single tablet group. The `Raft` handle itself never
/// leaves that thread; every method below is an ordinary `Send` future on
/// any runtime.
pub struct ReplicatedNode {
    machine: ReplicatedStateMachine,
    /// Durable log handle for test/diagnostic introspection (production
    /// reads go through the owner; this never touches the lane).
    #[allow(dead_code)]
    store: DurableRaftStore,
    sidecar: SidecarStore,
    /// Node-wide sidecar preflight metrics (shared with the owner).
    preflight: Arc<crate::preflight::PreflightMetrics>,
    owner_tx: async_channel::Sender<OwnerRequest>,
    owner_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    peer_addr: std::net::SocketAddr,
    namespace: NamespaceId,
    tablet: TabletId,
    authority: TabletAuthority,
    local: NodeId,
    cluster: ClusterId,
    incarnation: kivi_types::NodeIncarnation,
    group: ConsensusGroupId,
    data_dir: PathBuf,
    /// Caller-side consistency hub: evidence, lease engines, strong
    /// cache, providers, metrics. Shared with the owner thread (which
    /// drives lease engines through it). Memory-held by construction — a
    /// restart starts empty, so incarnation handling fails closed without
    /// any explicit clearing.
    hub: Arc<ConsistencyHub>,
    /// Lazy-ALR batch coordinator for the tablet group.
    alr: Arc<crate::alr::AlrCoordinator>,
    _dir_guard: kivi_durability::OpenDir,
}

impl std::fmt::Debug for ReplicatedNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReplicatedNode")
            .field("node", &self.local)
            .field("cluster", &self.cluster)
            .field("tablet", &self.tablet)
            .field("group", &self.group)
            .field("peer", &self.peer_addr)
            .finish_non_exhaustive()
    }
}

impl ReplicatedNode {
    /// Returns the bound peer endpoint.
    #[must_use]
    pub const fn peer_addr(&self) -> std::net::SocketAddr {
        self.peer_addr
    }
}

/// Everything recovery establishes before the owner thread starts:
/// validated topology, directory identity, durable log and state,
/// immutable sidecar store, peer TLS identity, and the dial/membership
/// maps.
struct RecoveredParts {
    opened: kivi_durability::OpenDir,
    store: DurableRaftStore,
    machine: ReplicatedStateMachine,
    sidecar: SidecarStore,
    tls: crate::transport::TlsMaterial,
    voters: std::collections::BTreeSet<u64>,
    peer_addrs: std::collections::BTreeMap<u64, String>,
    peers: HashMap<NodeId, std::net::SocketAddr>,
    group: ConsensusGroupId,
}

/// Parameters crossing into the owner thread at spawn. Everything is
/// owned (`Send`); the `!Send` Raft handle is built inside.
struct OwnerParams<S> {
    store: S,
    machine: ReplicatedStateMachine,
    sidecar: SidecarStore,
    tls: crate::transport::TlsMaterial,
    topology: ClusterTopology,
    transport_config: TransportConfig,
    peer_identity: crate::peer::PeerIdentity,
    peers: HashMap<NodeId, std::net::SocketAddr>,
    voters: std::collections::BTreeSet<u64>,
    peer_addrs: std::collections::BTreeMap<u64, String>,
    local: NodeId,
    group: ConsensusGroupId,
    tablet: TabletId,
    namespace: NamespaceId,
    authority: TabletAuthority,
    preflight: Arc<crate::preflight::PreflightMetrics>,
    preflight_enabled: bool,
    /// Shared consistency hub (lease engines, evidence, metrics).
    hub: Arc<ConsistencyHub>,
    serve_tx: async_channel::Sender<OwnerRequest>,
    requests: async_channel::Receiver<OwnerRequest>,
    ready: futures::channel::oneshot::Sender<Result<std::net::SocketAddr, NodeOpenError>>,
}

/// Validates topology, opens the data directory (adopting operator
/// identity when fresh, advancing the incarnation on restart), and
/// recovers the log and state machine. Synchronous: nothing here needs
/// a runtime; the owner thread (mesh and `Raft`) starts afterwards.
///
/// # Errors
///
/// Returns [`NodeOpenError`] on topology, identity, or storage failures.
#[allow(clippy::too_many_lines)]
fn open_recovered_parts(config: &NodeConfig) -> Result<RecoveredParts, NodeOpenError> {
    use NodeOpenError as Fault;
    config
        .topology
        .validate()
        .map_err(|error| Fault::Topology {
            reason: error.to_string(),
        })?;
    let assignment = config
        .topology
        .assignment(config.tablet)
        .ok_or_else(|| Fault::Topology {
            reason: format!("no tablet assignment for tablet {}", config.tablet),
        })?;
    if !assignment.replicas.contains(&config.local) {
        return Err(Fault::Topology {
            reason: format!("node {} replicates no tablet", config.local),
        });
    }
    // Data-directory identity: adopt the static topology identity on
    // first formation, advance the incarnation on every restart.
    let opened = kivi_durability::open_data_dir_with(
        &config.data_dir,
        Some(kivi_durability::NodeSeed {
            cluster: config.topology.cluster,
            node: config.local,
        }),
    )
    .map_err(|error| Fault::DataDir {
        reason: error.to_string(),
    })?;
    if opened.meta.cluster != config.topology.cluster || opened.meta.node != config.local {
        return Err(Fault::Bootstrap {
            reason: format!(
                "directory identity (cluster {}, node {}) conflicts with topology (cluster {}, node {})",
                opened.meta.cluster, opened.meta.node, config.topology.cluster, config.local,
            ),
        });
    }
    let lane_identity = kivi_durability::LaneIdentity {
        cluster: opened.meta.cluster,
        node: opened.meta.node,
        incarnation: opened.meta.incarnation,
    };
    let group = ConsensusGroupId::of_tablet(config.tablet);
    let store = DurableRaftStore::open(
        &config.data_dir,
        config.namespace,
        group,
        lane_identity,
        config.segment_target_bytes,
    )
    .map_err(|error| Fault::LogStore {
        reason: error.to_string(),
    })?;
    let machine = ReplicatedStateMachine::open(
        &config.data_dir,
        config.namespace,
        config.tablet,
        config.authority,
    )
    .map_err(|error| Fault::StateMachine {
        reason: error.to_string(),
    })?;
    // Immutable sidecar store: content-addressed chunks + manifests under
    // `<data_dir>/sidecar`, one lane, namespace-derived domain. Recovery
    // replays packs; startup verification (applied roots resolvable)
    // happens after the owner starts serving.
    let domain = kivi_types::SecurityDomainId::from_u64(config.namespace.as_u64());
    let (sidecar, _sidecar_thread) =
        SidecarStore::open(&config.data_dir, domain, 256 * 1024 * 1024).map_err(|error| {
            Fault::Sidecar {
                reason: error.to_string(),
            }
        })?;
    let (voters, peer_addrs) = config
        .topology
        .membership_for(config.tablet)
        .map_err(|error| Fault::Topology {
            reason: error.to_string(),
        })?;
    let peers: HashMap<NodeId, std::net::SocketAddr> = config
        .topology
        .nodes
        .iter()
        .map(|descriptor| (descriptor.node, descriptor.peer))
        .collect();
    // Peer TLS identity: this node's certificate (generated on first
    // open) plus the pinned peer certificates from static config.
    let peer_tls_cert = crate::tls::NodeCert::load_or_generate(&config.data_dir, opened.meta.node)
        .map_err(|error| Fault::Tls {
            reason: error.to_string(),
        })?;
    let tls = crate::transport::TlsMaterial {
        cert: peer_tls_cert,
        peer_certs: config.peer_certs.clone(),
        trust: Some(crate::tls::empty_trust()),
        insecure_skip_verify: config.insecure_peer_tls,
    };
    Ok(RecoveredParts {
        opened,
        store,
        machine,
        sidecar,
        tls,
        voters,
        peer_addrs,
        peers,
        group,
    })
}

impl ReplicatedNode {
    /// Opens the node: validates topology, opens the data directory
    /// (adopting operator identity when fresh, advancing the
    /// incarnation), recovers log and state, classifies bootstrap,
    /// verifies-or-installs membership, and starts the peer mesh on the
    /// Compio owner thread. Listeners start only after recovery and
    /// bootstrap complete — a corrupt database never serves.
    ///
    /// # Errors
    ///
    /// Returns [`NodeOpenError`] on topology, identity, storage, mesh, or
    /// bootstrap failures. Restarting with conflicting durable
    /// identity/membership fails loudly; a fresh directory forms the
    /// group exactly once.
    pub async fn open(config: NodeConfig) -> Result<Self, NodeOpenError> {
        use NodeOpenError as Fault;
        let RecoveredParts {
            opened,
            store,
            machine,
            sidecar,
            tls,
            voters,
            peer_addrs,
            peers,
            group,
        } = open_recovered_parts(&config)?;
        // Paused mesh: no H3 request serves before every member of a
        // fresh formation installed membership (static-bootstrap race
        // discipline: a premature granted vote would refuse our own
        // `initialize` as `NotAllowed`).
        let mut transport_config = config.transport.clone();
        transport_config.start_paused = true;
        let peer_identity = crate::peer::PeerIdentity {
            cluster: opened.meta.cluster,
            node: opened.meta.node,
            incarnation: opened.meta.incarnation,
        };
        let (owner_tx, owner_rx) = async_channel::bounded::<OwnerRequest>(OWNER_QUEUE);
        let (ready_tx, ready_rx) =
            futures::channel::oneshot::channel::<Result<std::net::SocketAddr, NodeOpenError>>();
        let router_tx = owner_tx.clone();
        let preflight = Arc::new(crate::preflight::PreflightMetrics::default());
        let (hub, alr) = open_consistency_front(
            &machine,
            &owner_tx,
            group,
            config.tablet,
            config.authority,
            opened.meta.node,
            opened.meta.incarnation,
            config.lease_params,
        );
        let owner_params = OwnerParams {
            store: store.clone(),
            machine: machine.clone(),
            sidecar: sidecar.clone(),
            tls,
            topology: config.topology.clone(),
            transport_config,
            peer_identity,
            peers,
            voters,
            peer_addrs,
            local: opened.meta.node,
            group,
            tablet: config.tablet,
            namespace: config.namespace,
            authority: config.authority,
            preflight: Arc::clone(&preflight),
            preflight_enabled: config.preflight_enabled,
            hub: Arc::clone(&hub),
            serve_tx: router_tx,
            requests: owner_rx,
            ready: ready_tx,
        };
        let thread = spawn_owner_thread(owner_params)?;
        let peer_addr = ready_rx.await.map_err(|_| Fault::Owner {
            reason: "consensus owner died during startup".to_owned(),
        })??;
        // Startup verification: every applied chunked root must resolve
        // locally before serving reads (fail-closed poison, repair via
        // sidecar fetch or snapshot install — never a startup wait).
        verify_chunked_roots(&machine, &sidecar).await;
        Ok(Self {
            machine: machine.clone(),
            store: store.clone(),
            sidecar: sidecar.clone(),
            preflight,
            owner_tx: owner_tx.clone(),
            owner_thread: Mutex::new(Some(thread)),
            peer_addr,
            namespace: config.namespace,
            tablet: config.tablet,
            authority: config.authority,
            local: opened.meta.node,
            cluster: opened.meta.cluster,
            incarnation: opened.meta.incarnation,
            group,
            data_dir: config.data_dir,
            hub: Arc::clone(&hub),
            alr,
            _dir_guard: opened,
        })
    }

    /// Returns the local node identity.
    #[must_use]
    pub const fn node(&self) -> NodeId {
        self.local
    }

    /// Returns the cluster identity.
    #[must_use]
    pub const fn cluster(&self) -> ClusterId {
        self.cluster
    }

    /// Returns this process's incarnation (advanced on every restart).
    #[must_use]
    pub const fn incarnation(&self) -> kivi_types::NodeIncarnation {
        self.incarnation
    }

    /// Returns the data-directory root.
    #[must_use]
    pub fn data_dir(&self) -> &std::path::Path {
        &self.data_dir
    }

    /// Returns the immutable sidecar store (leader staging, read serving,
    /// admin diagnostics). Cloneable; the blocking worker thread is shared.
    #[must_use]
    pub fn sidecar(&self) -> &SidecarStore {
        &self.sidecar
    }

    /// Sends one owner request and awaits its reply. A closed queue or a
    /// dropped reply means the owner is gone: `ShuttingDown`, never a
    /// hang.
    async fn owner_call<T>(
        &self,
        build: impl FnOnce(Reply<T>) -> OwnerRequest,
    ) -> Result<T, ConsensusError> {
        owner_call_on(&self.owner_tx, build).await
    }

    /// Proposes one client mutation through the replicated path:
    /// capability gate, leader dedup check, deterministic prepare,
    /// quorum commit, verified apply. Follows Kivi-owned routing errors;
    /// `OpenRaft` errors never escape.
    ///
    /// The gate/dedup/prepare stages run on the caller (shared machine
    /// state); only the quorum commit hops to the owner thread.
    ///
    /// # Errors
    ///
    /// Returns [`ProposeError`] for routing, dedup, capability, or
    /// validation failures.
    pub async fn propose(
        &self,
        op: &Operation,
        identity: Option<MutationIdentity>,
        idempotency: Option<IdempotencyKey>,
        now: WallTimestamp,
    ) -> Result<ProposeOutcome, ProposeError> {
        propose_caller_side(
            &self.machine,
            &self.sidecar,
            self.namespace,
            self.authority,
            &self.owner_tx,
            self.group,
            op,
            identity,
            idempotency,
            now,
        )
        .await
    }

    /// Serves one read under its contract, returning the outcome with its
    /// proof triple (authority, applied position, serve path).
    ///
    /// `eligibility` gates the roster fast path: point reads pass
    /// [`LeaseEligibility::Eligible`]; transactions, batches, and scans
    /// pass [`LeaseEligibility::ConservativeOnly`].
    ///
    /// # Errors
    ///
    /// Returns [`ReadError`] for routing, lineage, validation, or coverage
    /// failures.
    pub async fn read(
        &self,
        op: &Operation,
        contract: ReadContract,
        ctx: ReadContext,
        eligibility: kivi_types::LeaseEligibility,
    ) -> Result<ServedRead, ReadError> {
        ReadFront::bind(
            &self.machine,
            &self.owner_tx,
            &self.hub,
            &self.alr,
            self.group,
            self.tablet,
            self.authority,
            self.incarnation,
        )
        .read(op, contract, ctx, eligibility)
        .await
    }

    /// Serves one bounded scan page under its contract. Scans are always
    /// lease-ineligible (a multi-tablet scan is not one global snapshot);
    /// each tablet read is individually strong via Lazy-ALR or the
    /// barrier.
    ///
    /// # Errors
    ///
    /// Returns [`ReadError`] for routing, lineage, validation, or coverage
    /// failures.
    pub async fn scan(
        &self,
        spec: &kivi_state::ScanSpec,
        contract: ReadContract,
        ctx: ReadContext,
    ) -> Result<crate::consistency::ServedScan, ReadError> {
        ReadFront::bind(
            &self.machine,
            &self.owner_tx,
            &self.hub,
            &self.alr,
            self.group,
            self.tablet,
            self.authority,
            self.incarnation,
        )
        .scan(
            spec,
            contract,
            ctx,
            kivi_types::LeaseEligibility::ConservativeOnly,
        )
        .await
    }

    /// Switches the read-authority mode (operator/test control for the
    /// Almost-Local and roster-lease prototypes).
    pub fn set_read_provider(&self, provider: ReadAuthorityProvider) {
        self.hub.set_provider(provider);
    }

    /// Freezes this tablet's consistency counters for status surfaces.
    #[must_use]
    pub fn consistency_snapshot(&self) -> kivi_types::ConsistencySnapshot {
        self.hub
            .snapshot(self.tablet, process_ticks())
            .map_or_else(kivi_types::ConsistencySnapshot::default, |(_, snapshot)| {
                snapshot
            })
    }

    /// Reads full diagnostics: group, role, term, leader, log pointers,
    /// snapshot base, health, per-peer stats, and consistency counters.
    pub async fn status(&self) -> NodeStatus {
        // The owner is gone only after shutdown; a dead owner reports a
        // closed status rather than hanging the admin plane.
        let group = self.group;
        let mut status = self
            .owner_call(|reply| OwnerRequest::Status { group, reply })
            .await
            .unwrap_or_else(|_| NodeStatus {
                group: self.group,
                replica: ReplicaId::of_node(self.local),
                role: ReplicaRole::Follower,
                term: ConsensusTerm::INITIAL,
                leader: None,
                last_log: ConsensusLogIndex::NONE,
                committed: ConsensusLogIndex::NONE,
                applied: ConsensusLogIndex::NONE,
                applied_commit: CommitPosition::UNASSIGNED,
                snapshot: None,
                purged: None,
                healthy: false,
                detail: Some("consensus owner shut down".to_owned()),
                peers: HashMap::new(),
                sidecar: self.sidecar.metrics().snapshot(),
                preflight: self.preflight.snapshot(),
                consistency: kivi_types::ConsistencySnapshot::default(),
            });
        // The hub lives caller-side (the owner never touches it), so the
        // front overlays live consistency counters here.
        status.consistency = self.consistency_snapshot();
        status
    }

    /// Seals a snapshot and purges the log through its base through
    /// `OpenRaft`'s own trigger path: `RaftCore` builds the image via
    /// the state machine (durable before streaming) and purges through
    /// the store, recording the logical purge and updating its pointers.
    /// A restart replays only the retained tail. Purge never passes the
    /// installed snapshot base, so history stays recoverable; an
    /// unhealthy machine blocks the snapshot (and hence the purge).
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason when unhealthy, when no entries
    /// are applied yet, or when the snapshot/purge never completes
    /// before the deadline.
    pub async fn snapshot_and_purge(&self) -> Result<ConsensusLogIndex, String> {
        let (reply, rx) = futures::channel::oneshot::channel();
        let group = self.group;
        self.owner_tx
            .send(OwnerRequest::SnapshotPurge { group, reply })
            .await
            .map_err(|_| "consensus owner shut down".to_owned())?;
        rx.await
            .map_err(|_| "consensus owner shut down".to_owned())?
    }

    /// Suspends one peer link (partition test hook, drain tooling).
    pub async fn suspend_peer(&self, peer: NodeId) {
        let (reply, rx) = futures::channel::oneshot::channel();
        if self
            .owner_tx
            .send(OwnerRequest::SuspendPeer { peer, reply })
            .await
            .is_ok()
        {
            let _ = rx.await;
        }
    }

    /// Resumes a suspended peer link.
    pub async fn resume_peer(&self, peer: NodeId) {
        let (reply, rx) = futures::channel::oneshot::channel();
        if self
            .owner_tx
            .send(OwnerRequest::ResumePeer { peer, reply })
            .await
            .is_ok()
        {
            let _ = rx.await;
        }
    }

    /// Stops the Raft instance, the peer mesh, and the owner thread.
    ///
    /// # Panics
    ///
    /// Panics when the owner-handle lock is poisoned (a previous holder
    /// panicked while taking the handle — a lifecycle bug, never a
    /// runtime condition).
    pub async fn shutdown(&self) {
        let _ = self.owner_tx.send(OwnerRequest::Shutdown).await;
        // The owner exits promptly (Raft + mesh shutdown are local);
        // joining here reclaims the thread before the caller proceeds.
        // Shutdown is rare and never latency-sensitive, so blocking the
        // caller beats leaking the thread.
        if let Some(thread) = self.owner_thread.lock().expect("owner handle holds").take() {
            let _ = thread.join();
        }
    }
}

/// Builds the caller-side consistency front: the shared hub (evidence,
/// lease engines, metrics) plus the tablet's Lazy-ALR batch coordinator.
/// Memory-held by construction — a restart starts empty, so incarnation
/// handling fails closed without any explicit clearing.
#[allow(clippy::too_many_arguments)]
fn open_consistency_front(
    machine: &ReplicatedStateMachine,
    owner_tx: &async_channel::Sender<OwnerRequest>,
    group: ConsensusGroupId,
    tablet: TabletId,
    authority: TabletAuthority,
    local: NodeId,
    incarnation: NodeIncarnation,
    lease_params: kivi_types::LeaseParams,
) -> (Arc<ConsistencyHub>, Arc<crate::alr::AlrCoordinator>) {
    let hub = Arc::new(ConsistencyHub::new(
        local,
        incarnation,
        ReadAuthorityProvider::ConservativeLeader,
        lease_params,
    ));
    let alr = Arc::new(crate::alr::AlrCoordinator::bind(
        machine.clone(),
        owner_tx.clone(),
        Arc::clone(&hub),
        group,
        tablet,
        authority,
        incarnation,
        local,
    ));
    (hub, alr)
}

/// Verifies every applied chunked root resolves locally before serving
/// reads. Missing sidecars poison the replica (reads fail closed,
/// readiness reflects unhealthy) rather than serving incomplete state;
/// repair arrives via sidecar fetch or a snapshot install carrying the
/// missing bulk (which heals health). Never blocks startup waiting for
/// peers.
async fn verify_chunked_roots(machine: &ReplicatedStateMachine, sidecar: &SidecarStore) {
    for (manifest, logical_len) in machine.chunked_roots().await {
        match sidecar.check_root(manifest, logical_len).await {
            Ok(true) => {}
            Ok(false) => {
                machine
                    .mark_unhealthy(format!(
                        "applied root {manifest} references missing sidecars; repair required"
                    ))
                    .await;
                break;
            }
            Err(error) => {
                machine
                    .mark_unhealthy(format!("sidecar check failed: {error}"))
                    .await;
                break;
            }
        }
    }
}

/// Process monotonic microseconds since first call (clock source for
/// lease issuance only; deterministic callers pass virtual `Ticks`
/// through `grant_read_lease_at` instead). Like `wall_now` on the server:
/// a boundary clock, never inside deterministic logic.
pub(crate) fn process_ticks() -> Ticks {
    use std::sync::OnceLock;
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    let start = START.get_or_init(std::time::Instant::now);
    Ticks::from_micros(start.elapsed().as_micros().try_into().unwrap_or(u64::MAX))
}

/// Sends one owner request over an explicit queue and awaits its reply.
/// A closed queue or a dropped reply means the owner is gone:
/// `ShuttingDown`, never a hang. Shared by the single-group node and the
/// multi-tablet workers (both fronts stay `Send` while `Raft` handles stay
/// pinned to their reactors).
pub(crate) async fn owner_call_on<T>(
    owner_tx: &async_channel::Sender<OwnerRequest>,
    build: impl FnOnce(Reply<T>) -> OwnerRequest,
) -> Result<T, ConsensusError> {
    let (reply, rx) = futures::channel::oneshot::channel();
    owner_tx
        .send(build(reply))
        .await
        .map_err(|_| ConsensusError::ShuttingDown)?;
    rx.await.map_err(|_| ConsensusError::ShuttingDown)
}

/// Caller-side proposal shared by the single-group node and every
/// multi-tablet replica: leader dedup check, local sidecar staging,
/// chunked preflight dispatch vs. small-write prepare, quorum commit.
/// Only the commit hops to the owner thread; the stages before it run on
/// the caller's shared machine state.
///
/// # Errors
///
/// Returns [`ProposeError`] for routing, dedup, capability, or validation
/// failures.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn propose_caller_side(
    machine: &ReplicatedStateMachine,
    sidecar: &SidecarStore,
    namespace: NamespaceId,
    authority: TabletAuthority,
    owner_tx: &async_channel::Sender<OwnerRequest>,
    group: ConsensusGroupId,
    op: &Operation,
    identity: Option<MutationIdentity>,
    idempotency: Option<IdempotencyKey>,
    now: WallTimestamp,
) -> Result<ProposeOutcome, ProposeError> {
    use kivi_state::StorePrepared;
    // Leader dedup check first: retries answer from any replica's
    // retained state without proposing (lost-response failover). This
    // precedes sidecar work so a retry never stages a second root. Hits
    // answer even on a fenced tablet (the outcome already committed
    // before the fence); only fresh admits fence below.
    if let Some(marker) = identity {
        match machine.check_proposal(&marker).await.map_err(|error| {
            ProposeError::Consensus(ConsensusError::Unavailable {
                reason: error.to_string(),
            })
        })? {
            ProposalGate::Admit => {}
            ProposalGate::Hit { outcome } => {
                return Ok(ProposeOutcome::Duplicate { outcome });
            }
            ProposalGate::Expired => return Err(ProposeError::Expired),
            ProposalGate::Overloaded => return Err(ProposeError::Overloaded),
        }
    }
    // Split/merge cutover fence: fresh mutating proposes fail so the
    // bounded final tail installs without loss. Reads never reach here
    // (they use the read path and still serve during the fence).
    if machine.is_fenced().await {
        return Err(ProposeError::Fenced {
            tablet: group.tablet(),
        });
    }
    // Sidecar staging: chunked roots must already be durable locally
    // (cluster server stages before proposing); chunked-base range patches
    // restage here reusing untouched chunk ids via content addressing.
    let restaged = ensure_proposal_sidecars_for(sidecar, machine, op, now).await?;
    let effective = restaged.as_ref().unwrap_or(op);
    // Chunked operations take the preflight path: the owner
    // pre-distributes sidecars to a write quorum first, then prepares the
    // mutation against fresh state at the proposal boundary. The condition
    // of a conditional write must evaluate after the multi-second
    // preflight — never before it — so the caller must NOT prepare here
    // (a stale expectation would silently overwrite newer state). Small
    // writes keep the fast path below.
    if matches!(
        effective,
        Operation::SetChunked { .. } | Operation::SetConditionalChunked { .. }
    ) {
        let staged = effective.clone();
        return owner_call_on(owner_tx, |reply| OwnerRequest::ProposeChunked {
            group,
            op: staged,
            identity,
            idempotency,
            now,
            reply,
        })
        .await?;
    }
    // Deterministic prepare against current state (read-only).
    let prepared = machine.prepare_operation(effective, now).await?;
    let (mutation, expected) = match prepared {
        StorePrepared::Read(outcome) => return Ok(ProposeOutcome::Read { outcome }),
        StorePrepared::Terminal(outcome) => {
            return Ok(ProposeOutcome::Rejected { outcome });
        }
        StorePrepared::Write { mutation, expected } => (mutation, expected),
    };
    // No separate leader pre-check (`is_leader` is deprecated in
    // favor of the barrier): `client_write` fails fast with
    // `ForwardToLeader` on followers, which maps to `NotLeader` with
    // a hint below. A stale pre-check would only add a
    // lost-leadership race without removing this mapping.
    // Identity-less requests envelop the reserved anonymous identity
    // (no retry contract): the state machine applies them fresh
    // without dedup install, so unrelated anonymous writes never
    // alias. Client adapters must never issue session zero.
    let client = identity.map_or(
        kivi_types::RequestIdentity::new(
            kivi_types::SessionId::from_u128(crate::state_machine::ANONYMOUS_SESSION),
            kivi_types::RequestSeq::from_u64(0),
        ),
        |marker| marker.client,
    );
    let envelope =
        kivi_state::MutationEnvelope::new(namespace, authority, client, idempotency, mutation);
    let command = ConsensusCommand::Tablet(ReplicatedMutation::new(
        envelope,
        expected,
        now,
        identity.map_or(kivi_types::RequestSeq::from_u64(0), |marker| {
            marker.ack_floor
        }),
    ));
    owner_call_on(owner_tx, |reply| OwnerRequest::Propose {
        group,
        command,
        reply,
    })
    .await?
}

/// Caller-side read front shared by the single-group node and every
/// multi-tablet replica: hub-planned contract dispatch over one group's
/// machine, consistency hub, and owner queue.
///
/// Planning is deterministic and synchronous ([`ConsistencyHub::plan`]);
/// only barriers, fence syncs, and waits hop to the owner thread (the Raft
/// handle never leaves its reactor). Every served read returns its
/// [`ServedRead`] proof triple; every contract is either satisfied or fails
/// loudly — never silently weakened.
pub(crate) struct ReadFront<'a> {
    /// Local state machine for reads and applied tracking.
    machine: &'a ReplicatedStateMachine,
    /// Owner queue for barriers, fence syncs, and waits.
    owner_tx: &'a async_channel::Sender<OwnerRequest>,
    /// Node-wide consistency hub (evidence, cache, providers, metrics).
    hub: &'a ConsistencyHub,
    /// Tablet Lazy-ALR batch coordinator.
    alr: &'a crate::alr::AlrCoordinator,
    /// Addressed group.
    group: ConsensusGroupId,
    /// Served tablet.
    tablet: TabletId,
    /// Replica authority (epoch/guard fencing).
    authority: TabletAuthority,
    /// This process's incarnation (lease/proof binding).
    incarnation: NodeIncarnation,
}

impl<'a> ReadFront<'a> {
    /// Binds one group's read path.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind(
        machine: &'a ReplicatedStateMachine,
        owner_tx: &'a async_channel::Sender<OwnerRequest>,
        hub: &'a ConsistencyHub,
        alr: &'a crate::alr::AlrCoordinator,
        group: ConsensusGroupId,
        tablet: TabletId,
        authority: TabletAuthority,
        incarnation: NodeIncarnation,
    ) -> Self {
        Self {
            machine,
            owner_tx,
            hub,
            alr,
            group,
            tablet,
            authority,
            incarnation,
        }
    }

    /// Serves one read under its contract.
    ///
    /// # Errors
    ///
    /// Returns [`ReadError`] for routing, lineage, validation, or coverage
    /// failures.
    pub(crate) async fn read(
        &self,
        op: &Operation,
        contract: ReadContract,
        ctx: ReadContext,
        eligibility: kivi_types::LeaseEligibility,
    ) -> Result<ServedRead, ReadError> {
        let applied = self.machine.status().await.applied_commit;
        let fresh = ReplicaFreshness::new(self.authority, applied, self.incarnation);
        let planned = self
            .hub
            .plan(self.tablet, op, contract, fresh, ctx, eligibility);
        match planned.plan {
            ReadPlan::ServeLocal { path } => {
                if path == ServePath::StrongCache
                    && let Some(hit) = planned.cached
                {
                    let served = ServedRead::new(
                        hit.outcome,
                        ReadReceipt::new(self.authority, hit.position, self.incarnation),
                        path,
                    );
                    self.hub.note_served(self.tablet, contract, false);
                    tracing::debug!("{}", served.describe());
                    return Ok(served);
                    // A planned cache hit without a cached outcome falls
                    // through to a live local read rather than failing.
                }
                if path == ServePath::RosterLease {
                    return self.roster_serve(op, contract, ctx).await;
                }
                if path == ServePath::AlrSync || path == ServePath::AlrSubsumed {
                    // Never planned (execution assigns ALR paths); a
                    // planner emitting one is a bug — barrier instead.
                    return self.barrier_serve(op, contract, ctx, false).await;
                }
                let outcome = read_applied_on(self.machine, op, ctx.now).await?;
                let served = self.finish_live(outcome, path).await;
                // Successful fast-path serves replay under their own
                // evidence: the hub attaches the current receipt when it
                // still validates, else stores a proof-less fill (Any /
                // AtLeast replay only — never freshness).
                self.hub.fill(
                    self.tablet,
                    self.authority,
                    op,
                    &served.outcome,
                    served.receipt.position,
                );
                self.hub.note_served(self.tablet, contract, false);
                tracing::debug!("{}", served.describe());
                Ok(served)
            }
            ReadPlan::BarrierThenServe { escalated } => {
                self.barrier_serve(op, contract, ctx, escalated).await
            }
            ReadPlan::AlrThenServe => self.alr_serve(op, contract, ctx).await,
            ReadPlan::WaitThenServe { index } => self.wait_serve(op, contract, ctx, index).await,
            ReadPlan::Reject { reject } => Err(ReadError::StaleToken {
                detail: reject.to_string(),
            }),
        }
    }

    /// Serves one roster-planned read: revalidate evidence at serve time
    /// (the plan's evidence may have destabilized since), serve live
    /// applied state under it, or fall back to Lazy-ALR. Never serves a
    /// previous value when a covered write has not applied yet — the
    /// responder knows it is behind (no evidence) and falls back.
    async fn roster_serve(
        &self,
        op: &Operation,
        contract: ReadContract,
        ctx: ReadContext,
    ) -> Result<ServedRead, ReadError> {
        let applied = self.machine.status().await.applied_commit;
        if self.hub.evidence(self.tablet, applied, ctx.ticks).is_none() {
            self.hub.note_roster_fallback(self.tablet);
            return self.alr_serve(op, contract, ctx).await;
        }
        let outcome = read_applied_on(self.machine, op, ctx.now).await?;
        let served = self.finish_live(outcome, ServePath::RosterLease).await;
        self.hub.fill(
            self.tablet,
            self.authority,
            op,
            &served.outcome,
            served.receipt.position,
        );
        self.hub.note_served(self.tablet, contract, false);
        tracing::debug!("{}", served.describe());
        Ok(served)
    }

    /// Serves one Lazy-ALR read: join (or form) the tablet's batch, wait
    /// for its boundary to apply locally, then serve live applied state.
    /// Batch failure falls back to the conservative barrier.
    async fn alr_serve(
        &self,
        op: &Operation,
        contract: ReadContract,
        ctx: ReadContext,
    ) -> Result<ServedRead, ReadError> {
        let Some(path) = self.alr.join_batch(ctx).await else {
            return self.barrier_serve(op, contract, ctx, false).await;
        };
        let outcome = read_applied_on(self.machine, op, ctx.now).await?;
        let served = self.finish_live(outcome, path).await;
        self.hub.fill(
            self.tablet,
            self.authority,
            op,
            &served.outcome,
            served.receipt.position,
        );
        self.hub.note_served(self.tablet, contract, false);
        tracing::debug!("{}", served.describe());
        Ok(served)
    }

    /// Serves one bounded scan page under its contract. Each tablet read is
    /// individually strong; the multi-tablet scan is still not one global
    /// snapshot (see `ScanConsistency`).
    ///
    /// # Errors
    ///
    /// Returns [`ReadError`] for routing, lineage, validation, or coverage
    /// failures.
    pub(crate) async fn scan(
        &self,
        spec: &kivi_state::ScanSpec,
        contract: ReadContract,
        ctx: ReadContext,
        eligibility: kivi_types::LeaseEligibility,
    ) -> Result<ServedScan, ReadError> {
        let applied = self.machine.status().await.applied_commit;
        let fresh = ReplicaFreshness::new(self.authority, applied, self.incarnation);
        let planned = self
            .hub
            .plan_scan(self.tablet, contract, fresh, ctx, eligibility);
        match planned.plan {
            ReadPlan::ServeLocal { path } => {
                if path == ServePath::RosterLease {
                    return self.roster_scan(spec, contract, ctx).await;
                }
                if path == ServePath::AlrSync || path == ServePath::AlrSubsumed {
                    return self.barrier_scan(spec, contract, ctx, false).await;
                }
                let page = scan_applied_on(self.machine, spec, ctx.now).await?;
                let served = self.finish_scan(page, path).await;
                self.hub.note_served(self.tablet, contract, false);
                Ok(served)
            }
            ReadPlan::BarrierThenServe { escalated } => {
                self.barrier_scan(spec, contract, ctx, escalated).await
            }
            ReadPlan::AlrThenServe => self.alr_scan(spec, contract, ctx).await,
            ReadPlan::WaitThenServe { index } => self.wait_scan(spec, contract, ctx, index).await,
            ReadPlan::Reject { reject } => Err(ReadError::StaleToken {
                detail: reject.to_string(),
            }),
        }
    }

    /// Scan variant of [`ReadFront::roster_serve`].
    async fn roster_scan(
        &self,
        spec: &kivi_state::ScanSpec,
        contract: ReadContract,
        ctx: ReadContext,
    ) -> Result<ServedScan, ReadError> {
        let applied = self.machine.status().await.applied_commit;
        if self.hub.evidence(self.tablet, applied, ctx.ticks).is_none() {
            self.hub.note_roster_fallback(self.tablet);
            return self.alr_scan(spec, contract, ctx).await;
        }
        let page = scan_applied_on(self.machine, spec, ctx.now).await?;
        let served = self.finish_scan(page, ServePath::RosterLease).await;
        self.hub.note_served(self.tablet, contract, false);
        Ok(served)
    }

    /// Scan variant of [`ReadFront::alr_serve`].
    async fn alr_scan(
        &self,
        spec: &kivi_state::ScanSpec,
        contract: ReadContract,
        ctx: ReadContext,
    ) -> Result<ServedScan, ReadError> {
        let Some(path) = self.alr.join_batch(ctx).await else {
            return self.barrier_scan(spec, contract, ctx, false).await;
        };
        let page = scan_applied_on(self.machine, spec, ctx.now).await?;
        let served = self.finish_scan(page, path).await;
        self.hub.note_served(self.tablet, contract, false);
        Ok(served)
    }

    /// Fresh quorum barrier, then a live local serve. A successful barrier
    /// installs a [`FreshnessReceipt`] (bounded-stale evidence) and fills
    /// the strong cache. Barrier failure after a bounded-stale escalation
    /// counts the bound unprovable.
    async fn barrier_serve(
        &self,
        op: &Operation,
        contract: ReadContract,
        ctx: ReadContext,
        escalated: bool,
    ) -> Result<ServedRead, ReadError> {
        // Leader-only strong read: the `ReadIndex` barrier proves
        // leadership with a quorum AND waits for local applied coverage of
        // its boundary (0.10 `ensure_linearizable` awaits readiness
        // itself), and fails with `ForwardToLeader` on followers. No
        // separate leader pre-check: a stale pre-check only adds a
        // lost-leadership race without removing the mapping below.
        let barrier = owner_call_on(self.owner_tx, |reply| OwnerRequest::Barrier {
            group: self.group,
            reply,
        })
        .await
        .map_err(|error| self.note_barrier_failure(error, escalated))?
        .map_err(|error| self.note_barrier_failure(error, escalated))?;
        let receipt = FreshnessReceipt::new(
            self.authority,
            crate::state_machine::commit_of_index(barrier.boundary.get()),
            ctx.ticks,
            self.incarnation,
        );
        self.hub.note_barrier(self.tablet, receipt);
        let outcome = read_applied_on(self.machine, op, ctx.now).await?;
        let path = if escalated {
            ServePath::BoundedStaleEscalated
        } else {
            ServePath::AuthorityBarrier
        };
        let served = self.finish_live(outcome, path).await;
        self.hub.fill(
            self.tablet,
            self.authority,
            op,
            &served.outcome,
            served.receipt.position,
        );
        self.hub.note_served(self.tablet, contract, true);
        tracing::debug!("{}", served.describe());
        Ok(served)
    }

    /// Bounded catch-up wait for an `AtLeast` token behind the replica,
    /// then a live local serve. The wait runs on the owner thread (reactor
    /// sleep with the context's capped budget); this future stays
    /// runtime-agnostic.
    async fn wait_serve(
        &self,
        op: &Operation,
        contract: ReadContract,
        ctx: ReadContext,
        index: u64,
    ) -> Result<ServedRead, ReadError> {
        let timeout = ctx.capped_wait();
        owner_call_on(self.owner_tx, |reply| OwnerRequest::WaitApplied {
            group: self.group,
            index,
            timeout,
            reply,
        })
        .await
        .map_err(ReadError::Consensus)?
        .map_err(|()| {
            self.hub.note_wait_timeout(self.tablet);
            ReadError::CoverageTimeout
        })?;
        let outcome = read_applied_on(self.machine, op, ctx.now).await?;
        let served = self.finish_live(outcome, ServePath::AtLeastWaited).await;
        self.hub.fill(
            self.tablet,
            self.authority,
            op,
            &served.outcome,
            served.receipt.position,
        );
        self.hub.note_served(self.tablet, contract, false);
        tracing::debug!("{}", served.describe());
        Ok(served)
    }

    /// Barrier variant of [`ReadFront::barrier_serve`] for scan pages.
    async fn barrier_scan(
        &self,
        spec: &kivi_state::ScanSpec,
        contract: ReadContract,
        ctx: ReadContext,
        escalated: bool,
    ) -> Result<ServedScan, ReadError> {
        let barrier = owner_call_on(self.owner_tx, |reply| OwnerRequest::Barrier {
            group: self.group,
            reply,
        })
        .await
        .map_err(|error| self.note_barrier_failure(error, escalated))?
        .map_err(|error| self.note_barrier_failure(error, escalated))?;
        self.hub.note_barrier(
            self.tablet,
            FreshnessReceipt::new(
                self.authority,
                crate::state_machine::commit_of_index(barrier.boundary.get()),
                ctx.ticks,
                self.incarnation,
            ),
        );
        let page = scan_applied_on(self.machine, spec, ctx.now).await?;
        let path = if escalated {
            ServePath::BoundedStaleEscalated
        } else {
            ServePath::AuthorityBarrier
        };
        let served = self.finish_scan(page, path).await;
        self.hub.note_served(self.tablet, contract, true);
        Ok(served)
    }

    /// Wait variant of [`ReadFront::wait_serve`] for scan pages.
    async fn wait_scan(
        &self,
        spec: &kivi_state::ScanSpec,
        contract: ReadContract,
        ctx: ReadContext,
        index: u64,
    ) -> Result<ServedScan, ReadError> {
        let timeout = ctx.capped_wait();
        owner_call_on(self.owner_tx, |reply| OwnerRequest::WaitApplied {
            group: self.group,
            index,
            timeout,
            reply,
        })
        .await
        .map_err(ReadError::Consensus)?
        .map_err(|()| {
            self.hub.note_wait_timeout(self.tablet);
            ReadError::CoverageTimeout
        })?;
        let page = scan_applied_on(self.machine, spec, ctx.now).await?;
        let served = self.finish_scan(page, ServePath::AtLeastWaited).await;
        self.hub.note_served(self.tablet, contract, false);
        Ok(served)
    }

    /// Maps a barrier failure, counting fencing rejects and — after a
    /// bounded-stale escalation — the unprovable bound.
    fn note_barrier_failure(&self, error: ConsensusError, escalated: bool) -> ReadError {
        self.hub.note_fencing_reject(self.tablet);
        if escalated {
            self.hub.note_escalation_failed(self.tablet);
        }
        ReadError::Consensus(error)
    }

    /// Builds the served-read proof from post-serve applied state: the
    /// receipt vouches exactly the state the outcome came from.
    async fn finish_live(&self, outcome: OperationResult, path: ServePath) -> ServedRead {
        let applied = self.machine.status().await.applied_commit;
        ServedRead::new(
            outcome,
            ReadReceipt::new(self.authority, applied, self.incarnation),
            path,
        )
    }

    /// Scan variant of [`ReadFront::finish_live`].
    async fn finish_scan(&self, page: kivi_state::ScanPage, path: ServePath) -> ServedScan {
        let applied = self.machine.status().await.applied_commit;
        ServedScan::new(
            page,
            ReadReceipt::new(self.authority, applied, self.incarnation),
            path,
        )
    }
}

/// Reads local applied state, mapping machine faults honestly. Shared by
/// both fronts. Fabric references fail closed here: consensus stores
/// never hold them (replicated IR carries bytes, never worker-local
/// ids), so a `FabricValue` means a foreign reference slipped past
/// install validation — Unavailable, never absence.
pub(crate) async fn read_applied_on(
    machine: &ReplicatedStateMachine,
    op: &Operation,
    now: WallTimestamp,
) -> Result<OperationResult, ReadError> {
    let outcome = machine
        .read_local(op, now)
        .await
        .map_err(|error| match error {
            crate::state_machine::StateMachineFault::NotARead => ReadError::NotARead,
            other => ReadError::Consensus(ConsensusError::Unavailable {
                reason: other.to_string(),
            }),
        })?;
    if let OperationResult::FabricValue { fabric_id, .. } = &outcome {
        return Err(ReadError::Consensus(ConsensusError::Unavailable {
            reason: format!("worker-local fabric reference {fabric_id} not servable here"),
        }));
    }
    Ok(outcome)
}

/// Scans local applied state, mapping machine faults honestly. Pages
/// naming worker-local fabric references fail closed (same rule as
/// [`read_applied_on`]): consensus scans never project foreign ids.
pub(crate) async fn scan_applied_on(
    machine: &ReplicatedStateMachine,
    spec: &kivi_state::ScanSpec,
    now: WallTimestamp,
) -> Result<kivi_state::ScanPage, ReadError> {
    let page = machine.scan_local(spec, now).await.map_err(|error| {
        ReadError::Consensus(ConsensusError::Unavailable {
            reason: error.to_string(),
        })
    })?;
    if page
        .entries
        .iter()
        .any(|entry| matches!(entry.value, Some(kivi_state::ScannedValue::Fabric { .. })))
    {
        return Err(ReadError::Consensus(ConsensusError::Unavailable {
            reason: "worker-local fabric reference not servable here".to_owned(),
        }));
    }
    Ok(page)
}

/// Spawns the Compio owner thread driving one node's consensus work.
/// Returns the thread handle; readiness (bound peer address or startup
/// failure) arrives separately over the params' `ready` channel so a
/// reactor-construction failure reports instead of hanging the opener.
///
/// # Errors
///
/// Returns [`NodeOpenError::Owner`] when the OS refuses the thread.
fn spawn_owner_thread<S>(
    owner_params: OwnerParams<S>,
) -> Result<std::thread::JoinHandle<()>, NodeOpenError>
where
    S: ConsensusLogStore,
{
    std::thread::Builder::new()
        .name("kivi-consensus".to_owned())
        .spawn(move || {
            // `ready` travels separately so a reactor-construction
            // failure still reaches the opener instead of hanging it.
            let OwnerParams {
                store,
                machine,
                sidecar,
                tls,
                topology,
                transport_config,
                peer_identity,
                peers,
                voters,
                peer_addrs,
                local,
                group,
                tablet,
                namespace,
                authority,
                preflight,
                preflight_enabled,
                hub,
                serve_tx,
                requests,
                ready,
            } = owner_params;
            let runtime = match compio::runtime::Runtime::new() {
                Ok(runtime) => runtime,
                Err(error) => {
                    let _ = ready.send(Err(NodeOpenError::Owner {
                        reason: format!("compio reactor failed to start: {error}"),
                    }));
                    return;
                }
            };
            runtime.block_on(owner_main(OwnerMain {
                store,
                machine,
                sidecar,
                tls,
                topology,
                transport_config,
                peer_identity,
                peers,
                voters,
                peer_addrs,
                local,
                group,
                tablet,
                namespace,
                authority,
                preflight,
                preflight_enabled,
                hub,
                serve_tx,
                requests,
                ready,
            }));
        })
        .map_err(|error| NodeOpenError::Owner {
            reason: format!("consensus owner thread failed to spawn: {error}"),
        })
}

/// Converts a raw voter id into its replica identity.
fn replica_of(id: u64) -> ReplicaId {
    ReplicaId::of_node(NodeId::from_u64(id))
}

pub(crate) type OwnerRaft = openraft::Raft<KiviTypeConfig, ReplicatedStateMachine>;

/// Owner thread main: opens the mesh, spawns Raft, bootstraps
/// membership, starts the mesh, signals readiness, then serves the
/// request loop until `Shutdown`. Runs inside the node's Compio reactor;
/// every `.await` here may drive `!Send` Raft futures.
#[allow(clippy::too_many_lines)]
async fn owner_main<S>(params: OwnerMain<S>)
where
    S: ConsensusLogStore,
{
    let OwnerMain {
        mut store,
        machine,
        sidecar,
        tls,
        topology,
        transport_config,
        peer_identity,
        peers,
        voters,
        peer_addrs,
        local,
        group,
        tablet,
        namespace,
        authority,
        preflight,
        preflight_enabled,
        hub,
        serve_tx,
        requests,
        ready,
    } = params;
    // Mesh: one UDP endpoint bound to the topology's peer address; QUIC
    // handshake plus H3 streams replace the old TCP listener/accept pump.
    // A bind conflict fails loudly so the harness retries formation.
    let bulk_timeout = transport_config.bulk_timeout;
    // Lease-tick channel: cloned before the mesh takes the original.
    let tick_tx = serve_tx.clone();
    let started = open_mesh(
        transport_config,
        peer_identity,
        peer_identity.cluster,
        &peers,
        tls,
        serve_tx,
    )
    .await;
    let (transport, peer_addr) = match started {
        Ok(started) => started,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    // Sidecar durability gate: other replicas are candidate sources
    // (leader-first at fetch time via the entry's leader hint). The gate
    // is installed on the log store before Raft spawns so no append can
    // report durable persistence without its sidecars. A clone serves
    // snapshot installs on the owner thread (same deduplicating fetch).
    let gate = {
        let domain = kivi_types::SecurityDomainId::from_u64(namespace.as_u64());
        let mut sources: Vec<NodeId> = peers
            .keys()
            .copied()
            .filter(|peer| *peer != local)
            .collect();
        sources.sort_by_key(|peer| peer.as_u64());
        let gate = SidecarGate::new(
            sidecar.clone(),
            transport.clone(),
            sources,
            group,
            domain,
            bulk_timeout,
        );
        store.set_gate(gate.clone());
        gate
    };
    let router = PeerRouter::new(transport.clone());
    let raft = match spawn_raft(local, group, router, &store, &machine).await {
        Ok(raft) => raft,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    // The single-group node keeps legacy static semantics: a fresh
    // directory always founds (it serves exactly one fixed group, so
    // there is no join path here to confuse with formation).
    if let Err(error) = bootstrap_group(
        &BootstrapInputs {
            topology: &topology,
            tablet,
            local,
            cluster: peer_identity.cluster,
            node: local,
            voters: &voters,
            peer_addrs: &peer_addrs,
        },
        true,
        &mut store,
        &machine,
        &raft,
    )
    .await
    {
        let _ = ready.send(Err(error));
        return;
    }
    transport.start();
    if ready.send(Ok(peer_addr)).is_err() {
        // Opener gone: shut down cleanly rather than serving nobody.
        let _ = raft.shutdown().await;
        transport.shutdown().await;
        return;
    }
    let mut voter_list: Vec<NodeId> = voters.iter().map(|id| NodeId::from_u64(*id)).collect();
    voter_list.sort_by_key(|node| node.as_u64());
    // Lease-driver ticker: one tick per renew quarter keeps renewals
    // flowing and lapses prompt without busy-looping the reactor. The
    // loop dies with the reactor on shutdown; a full queue (shutting
    // down) ends it.
    {
        let tick_tx = tick_tx.clone();
        let tick_interval = lease_tick_interval(hub.lease_params());
        compio::runtime::spawn(async move {
            loop {
                compio::time::sleep(tick_interval).await;
                if tick_tx
                    .send(OwnerRequest::LeaseTick { group })
                    .await
                    .is_err()
                {
                    break;
                }
            }
        })
        .detach();
    }
    owner_serve_loop(
        OwnerCtx {
            raft: raft.clone(),
            store,
            machine,
            sidecar,
            gate,
            // Single-group nodes serve no fragments: the multi-tablet
            // node threads its store here instead.
            fragments: None,
            transport: transport.clone(),
            group,
            namespace,
            authority,
            tablet,
            local,
            incarnation: peer_identity.incarnation,
            voters: voter_list,
            bulk_timeout,
            preflight,
            preflight_enabled,
            transfers: Rc::new(RefCell::new(HashMap::new())),
            hub,
        },
        raft,
        transport,
        requests,
    )
    .await;
}

/// Lease-tick cadence from the deployment contract: a quarter of the
/// renew interval keeps at most a few ticks between renewals while never
/// busy-looping (clamped to 10ms–1s so degenerate contracts fail loud in
/// validation, not in spinning).
pub(crate) fn lease_tick_interval(params: kivi_types::LeaseParams) -> Duration {
    let quarter = params.renew_interval / 4;
    quarter.clamp(Duration::from_millis(10), Duration::from_secs(1))
}

/// Spawns the Raft instance on the shared router (the fallible Raft
/// half of [`owner_main`], extracted so the main flow stays readable).
///
/// # Errors
///
/// Returns [`NodeOpenError`] when the Raft config is incoherent or the
/// spawn fails.
pub(crate) async fn spawn_raft<S>(
    local: NodeId,
    group: ConsensusGroupId,
    router: PeerRouter,
    store: &S,
    machine: &ReplicatedStateMachine,
) -> Result<OwnerRaft, NodeOpenError>
where
    S: ConsensusLogStore,
{
    use NodeOpenError as Fault;
    let raft_config = cluster_config().map_err(|error| Fault::Topology {
        reason: format!("invalid raft config: {error}"),
    })?;
    OwnerRaft::new(
        local.as_u64(),
        raft_config,
        GroupNetworkFactory::new(router, group),
        store.clone(),
        machine.clone(),
    )
    .await
    .map_err(|error| Fault::Transport {
        reason: format!("raft spawn failed: {error}"),
    })
}

/// Opens the QUIC/H3 peer mesh (the fallible mesh half of
/// [`owner_main`], extracted so the main flow stays readable). Binds one
/// UDP endpoint and spawns dial/serve tasks; callers must already run on
/// a Compio reactor.
///
/// # Errors
///
/// Returns [`NodeOpenError`] when the mesh refuses to open.
async fn open_mesh(
    transport_config: TransportConfig,
    peer_identity: crate::peer::PeerIdentity,
    cluster: ClusterId,
    peers: &HashMap<NodeId, std::net::SocketAddr>,
    tls: crate::transport::TlsMaterial,
    serve_tx: async_channel::Sender<OwnerRequest>,
) -> Result<(PeerTransport, std::net::SocketAddr), NodeOpenError> {
    use NodeOpenError as Fault;
    let handler: Arc<dyn PeerHandler> = Arc::new(ServeRouter { tx: serve_tx });
    PeerTransport::open(
        transport_config,
        peer_identity,
        cluster,
        peers,
        tls,
        handler,
    )
    .await
    .map_err(|error| Fault::Transport {
        reason: error.to_string(),
    })
}

/// Serves the owner request loop until `Shutdown`, then stops Raft and
/// the mesh. The loop never awaits Raft work inline: every request
/// spawns its own task (`client_write`, barriers, snapshot installs can
/// all pend indefinitely, and the loop must keep serving status,
/// diagnostics, and peer RPCs meanwhile — awaiting inline deadlocks
/// any client that overlaps a pending proposal with another request).
/// All tasks stay on this reactor (`!Send` handles never leave it).
async fn owner_serve_loop<S>(
    ctx: OwnerCtx<S>,
    raft: OwnerRaft,
    transport: PeerTransport,
    requests: async_channel::Receiver<OwnerRequest>,
) where
    S: ConsensusLogStore,
{
    while let Ok(request) = requests.recv().await {
        match request {
            OwnerRequest::Shutdown => break,
            other => {
                let ctx = ctx.clone();
                // Detach: compio cancels on handle drop, and the loop
                // must not serialize on long-lived Raft work.
                compio::runtime::spawn(async move {
                    ctx.serve_one(other).await;
                })
                .detach();
            }
        }
    }
    let _ = raft.shutdown().await;
    transport.shutdown().await;
}

/// Owned state on the owner thread: the `!Send` Raft handle, its
/// storage, sidecar store, peer TLS material, shared mesh, and snapshot
/// reassembly state.
struct OwnerMain<S> {
    store: S,
    machine: ReplicatedStateMachine,
    sidecar: SidecarStore,
    tls: crate::transport::TlsMaterial,
    topology: ClusterTopology,
    transport_config: TransportConfig,
    peer_identity: crate::peer::PeerIdentity,
    peers: HashMap<NodeId, std::net::SocketAddr>,
    voters: std::collections::BTreeSet<u64>,
    peer_addrs: std::collections::BTreeMap<u64, String>,
    local: NodeId,
    group: ConsensusGroupId,
    tablet: TabletId,
    namespace: NamespaceId,
    authority: TabletAuthority,
    preflight: Arc<crate::preflight::PreflightMetrics>,
    preflight_enabled: bool,
    /// Shared consistency hub (lease engines, evidence, metrics).
    hub: Arc<ConsistencyHub>,
    serve_tx: async_channel::Sender<OwnerRequest>,
    requests: async_channel::Receiver<OwnerRequest>,
    ready: futures::channel::oneshot::Sender<Result<std::net::SocketAddr, NodeOpenError>>,
}

/// Shared owner state behind the request loop. Cloneable so each
/// request runs on its own task; interior mutability is single-threaded
/// (`Rc<RefCell<..>>`) because nothing here ever leaves the owner
/// reactor.
///
/// Generic over the logical log store so the single-group node and every
/// multi-tablet worker share one implementation: the `Raft` handle type
/// is store-independent, and all group behavior (proposals, barriers,
/// peer RPCs, snapshots, preflight) lives here once.
#[derive(Clone)]
pub(crate) struct OwnerCtx<S> {
    pub(crate) raft: OwnerRaft,
    pub(crate) store: S,
    pub(crate) machine: ReplicatedStateMachine,
    pub(crate) sidecar: SidecarStore,
    pub(crate) gate: SidecarGate,
    /// Node-local redundancy fragment store (`None` when fragments are
    /// disabled: fragment RPCs are refused as shutting down). Shared by
    /// every replica on the node (content-addressed, group-independent);
    /// `Arc` because the store is `!Clone` but `Send + Sync`.
    pub(crate) fragments: Option<Arc<LocalFragmentStore>>,
    pub(crate) transport: PeerTransport,
    pub(crate) group: ConsensusGroupId,
    pub(crate) namespace: NamespaceId,
    pub(crate) authority: TabletAuthority,
    pub(crate) tablet: TabletId,
    pub(crate) local: NodeId,
    pub(crate) incarnation: NodeIncarnation,
    pub(crate) voters: Vec<NodeId>,
    pub(crate) bulk_timeout: Duration,
    pub(crate) preflight: Arc<crate::preflight::PreflightMetrics>,
    pub(crate) preflight_enabled: bool,
    pub(crate) transfers: Rc<RefCell<HashMap<u64, PartialTransfer>>>,
    /// Caller-side consistency hub (shared with the node front): the
    /// owner drives lease engines, polls coverage, and orders ALR fences
    /// through it. Engine steps are pure and fast; planning never blocks
    /// on transport.
    pub(crate) hub: std::sync::Arc<crate::consistency::ConsistencyHub>,
}

/// One in-flight snapshot transfer being reassembled from fragments.
pub(crate) struct PartialTransfer {
    vote: openraft::type_config::alias::VoteOf<KiviTypeConfig>,
    meta: openraft::type_config::alias::SnapshotMetaOf<KiviTypeConfig>,
    bytes: Vec<u8>,
}

/// `PeerHandler` over the owner queue: transport tasks forward every
/// incoming RPC (including snapshot fragments) to the owner loop, which
/// owns the Raft handles and the reassembly table. The future stays
/// `Send` (channel ends only).
#[derive(Clone)]
struct ServeRouter {
    tx: async_channel::Sender<OwnerRequest>,
}

impl PeerHandler for ServeRouter {
    fn handle(
        &self,
        from: NodeId,
        group: ConsensusGroupId,
        request: PeerRequest,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<PeerResponse, PeerRpcError>> + Send + '_>>
    {
        let tx = self.tx.clone();
        Box::pin(async move {
            let (reply, rx) = futures::channel::oneshot::channel();
            tx.send(OwnerRequest::Serve {
                from,
                group,
                request,
                reply,
            })
            .await
            .map_err(|_| PeerRpcError {
                detail: "consensus owner shut down".to_owned(),
            })?;
            rx.await.map_err(|_| PeerRpcError {
                detail: "consensus owner shut down".to_owned(),
            })?
        })
    }
}

impl<S> OwnerCtx<S>
where
    S: ConsensusLogStore,
{
    /// Serves one request on its own task (see the loop above for why
    /// nothing awaits inline there). Group-scoped requests naming another
    /// group are refused loudly (a worker loop misroute, never silent):
    /// the multi-tablet front routes by group before reaching a replica.
    #[allow(clippy::too_many_lines)]
    pub(crate) async fn serve_one(&self, request: OwnerRequest) {
        match request {
            OwnerRequest::Propose {
                group,
                command,
                reply,
            } => {
                if group != self.group {
                    let _ = reply.send(Err(ProposeError::Consensus(ConsensusError::Unavailable {
                        reason: format!("group {group} not served here"),
                    })));
                    return;
                }
                let outcome = self.propose(command).await;
                let _ = reply.send(outcome);
            }
            OwnerRequest::ProposeControl {
                group,
                mutation,
                reply,
            } => {
                if group != self.group {
                    let _ = reply.send(Err(ControlProposeError::Unavailable {
                        detail: format!("group {group} not served here"),
                    }));
                    return;
                }
                let outcome = self.propose_control_typed(mutation).await;
                let _ = reply.send(outcome);
            }
            OwnerRequest::ProposeChunked {
                group,
                op,
                identity,
                idempotency,
                now,
                reply,
            } => {
                if group != self.group {
                    let _ = reply.send(Err(ProposeError::Consensus(ConsensusError::Unavailable {
                        reason: format!("group {group} not served here"),
                    })));
                    return;
                }
                let outcome = self.propose_chunked(op, identity, idempotency, now).await;
                let _ = reply.send(outcome);
            }
            OwnerRequest::Barrier { group, reply } => {
                if group != self.group {
                    let _ = reply.send(Err(ConsensusError::Unavailable {
                        reason: format!("group {group} not served here"),
                    }));
                    return;
                }
                let barrier = self.barrier().await;
                let _ = reply.send(barrier);
            }
            OwnerRequest::WaitApplied {
                group,
                index,
                timeout,
                reply,
            } => {
                if group != self.group {
                    let _ = reply.send(Err(()));
                    return;
                }
                let covered = self.wait_applied(index, timeout).await;
                let _ = reply.send(covered);
            }
            OwnerRequest::ProposeSync {
                group,
                sync,
                timeout,
                reply,
            } => {
                if group != self.group {
                    let _ = reply.send(Err(ProposeError::Consensus(ConsensusError::Unavailable {
                        reason: format!("group {group} not served here"),
                    })));
                    return;
                }
                let outcome = self.propose_sync(sync, timeout).await;
                let _ = reply.send(outcome);
            }
            OwnerRequest::LeaseTick { group } => {
                if group != self.group {
                    return;
                }
                self.lease_tick_drive().await;
            }
            OwnerRequest::Status { group, reply } => {
                if group != self.group {
                    // No status to report for a foreign group; the caller
                    // aggregates per-group statuses and treats a missing
                    // group as its own signal. Reply unhealthy-closed
                    // rather than hanging the admin plane.
                    let _ = reply.send(NodeStatus {
                        group,
                        replica: ReplicaId::of_node(self.transport.local().node),
                        role: ReplicaRole::Follower,
                        term: ConsensusTerm::INITIAL,
                        leader: None,
                        last_log: ConsensusLogIndex::NONE,
                        committed: ConsensusLogIndex::NONE,
                        applied: ConsensusLogIndex::NONE,
                        applied_commit: CommitPosition::UNASSIGNED,
                        snapshot: None,
                        purged: None,
                        healthy: false,
                        detail: Some(format!("group {group} not served here")),
                        peers: HashMap::new(),
                        sidecar: self.sidecar.metrics().snapshot(),
                        preflight: self.preflight.snapshot(),
                        consistency: kivi_types::ConsistencySnapshot::default(),
                    });
                    return;
                }
                let status = self.status().await;
                let _ = reply.send(status);
            }
            OwnerRequest::Serve {
                from,
                group,
                request,
                reply,
            } => {
                let answer = self.serve_rpc(from, group, request).await;
                let _ = reply.send(answer);
            }
            OwnerRequest::SnapshotPurge { group, reply } => {
                if group != self.group {
                    let _ = reply.send(Err(format!("group {group} not served here")));
                    return;
                }
                let sealed = self.snapshot_purge().await;
                let _ = reply.send(sealed);
            }
            OwnerRequest::AddLearner {
                group,
                node,
                addr,
                blocking,
                reply,
            } => {
                if group != self.group {
                    let _ = reply.send(Err(format!("group {group} not served here")));
                    return;
                }
                let outcome = self.add_learner(node, addr, blocking).await;
                let _ = reply.send(outcome);
            }
            OwnerRequest::ChangeMembership {
                group,
                voters,
                retain,
                reply,
            } => {
                if group != self.group {
                    let _ = reply.send(Err(format!("group {group} not served here")));
                    return;
                }
                let outcome = self.change_membership(voters, retain).await;
                let _ = reply.send(outcome);
            }
            OwnerRequest::TransferLeader { group, to, reply } => {
                if group != self.group {
                    let _ = reply.send(Err(format!("group {group} not served here")));
                    return;
                }
                let outcome = self.transfer_leader(to).await;
                let _ = reply.send(outcome);
            }
            OwnerRequest::ObserveMembership { group, reply } => {
                if group != self.group {
                    let _ = reply.send(Err(format!("group {group} not served here")));
                    return;
                }
                let _ = reply.send(Ok(self.observe_membership()));
            }
            OwnerRequest::EnsureGroup { group, reply, .. } => {
                // The single-group node serves exactly one fixed group:
                // dynamic lifecycle lives on the multi-tablet worker loop.
                let _ = reply.send(Err(format!(
                    "group {group} dynamic creation unsupported on a single-group node"
                )));
            }
            OwnerRequest::RetireGroup { group, reply, .. } => {
                let _ = reply.send(Err(format!(
                    "group {group} dynamic retirement unsupported on a single-group node"
                )));
            }
            OwnerRequest::InitializeGroup { group, reply, .. } => {
                let _ = reply.send(Err(format!(
                    "group {group} dynamic initialization unsupported on a single-group node"
                )));
            }
            OwnerRequest::SuspendPeer { peer, reply } => {
                self.transport.suspend_peer(peer).await;
                let _ = reply.send(());
            }
            OwnerRequest::ResumePeer { peer, reply } => {
                self.transport.resume_peer(peer).await;
                let _ = reply.send(());
            }
            // Handled by the loop (breaks it); unreachable here.
            OwnerRequest::Shutdown => {}
        }
    }

    /// Stops this replica's `Raft` instance. Runs on the owning reactor
    /// (the handle is `!Send`); the worker loop calls this for every
    /// local replica before its reactor exits.
    pub(crate) async fn shutdown_raft(&self) {
        let _ = self.raft.shutdown().await;
    }

    /// Quorum-commits one chunked operation with sidecar preflight. Runs
    /// on the owner thread as its own task, so small same-tablet writes
    /// keep entering and committing through Raft while bulk sidecars move:
    /// only the final tiny root participates in Raft ordering.
    ///
    /// Ordering is: stage (done by the caller) → preflight followers to a
    /// write quorum → prepare the mutation against fresh state → propose.
    /// The logical condition of a conditional write evaluates at the
    /// prepare step, after the multi-second preflight, so a long-prepared
    /// write can never silently overwrite newer state. Immutable staging
    /// may happen early; logical evaluation belongs here. Preflight
    /// failure (or disabled preflight) falls back to the append gate: the
    /// proposal proceeds and followers fetch on demand.
    async fn propose_chunked(
        &self,
        op: Operation,
        identity: Option<MutationIdentity>,
        idempotency: Option<IdempotencyKey>,
        now: WallTimestamp,
    ) -> Result<ProposeOutcome, ProposeError> {
        use kivi_state::StorePrepared;
        // Extract the immutable root staged locally by the caller.
        let (manifest, logical_len) = match &op {
            Operation::SetChunked {
                manifest,
                logical_len,
                ..
            }
            | Operation::SetConditionalChunked {
                manifest,
                logical_len,
                ..
            } => (*manifest, *logical_len),
            _ => {
                return Err(ProposeError::Consensus(ConsensusError::Unavailable {
                    reason: "chunked proposal path requires a chunked operation".to_owned(),
                }));
            }
        };
        // Preflight followers concurrently; the leader already holds the
        // root durably (the caller verified). A quorum (including us)
        // durable-ready means prepared followers append immediately and
        // the tiny root commits quickly. Stragglers catch up through the
        // authoritative append gate.
        let followers: Vec<NodeId> = self
            .voters
            .iter()
            .copied()
            .filter(|peer| *peer != self.local)
            .collect();
        crate::preflight::preflight_to_quorum(
            &self.transport,
            self.group,
            manifest,
            logical_len,
            &followers,
            self.voters.len(),
            self.bulk_timeout,
            self.preflight_enabled,
            &self.preflight,
        )
        .await;
        // Fresh prepare against current state: deterministic MutationIR +
        // expected outcome evaluated NOW (after preflight), so concurrent
        // small writes to the same key keep normal proposal/order
        // semantics and conditional writes stay atomic.
        let prepared = self.machine.prepare_operation(&op, now).await?;
        let (mutation, expected) = match prepared {
            StorePrepared::Read(outcome) => return Ok(ProposeOutcome::Read { outcome }),
            StorePrepared::Terminal(outcome) => return Ok(ProposeOutcome::Rejected { outcome }),
            StorePrepared::Write { mutation, expected } => (mutation, expected),
        };
        let client = identity.map_or(
            kivi_types::RequestIdentity::new(
                kivi_types::SessionId::from_u128(crate::state_machine::ANONYMOUS_SESSION),
                kivi_types::RequestSeq::from_u64(0),
            ),
            |marker| marker.client,
        );
        let envelope = kivi_state::MutationEnvelope::new(
            self.namespace,
            self.authority,
            client,
            idempotency,
            mutation,
        );
        let command = ConsensusCommand::Tablet(ReplicatedMutation::new(
            envelope,
            expected,
            now,
            identity.map_or(kivi_types::RequestSeq::from_u64(0), |marker| {
                marker.ack_floor
            }),
        ));
        self.propose(command).await
    }

    /// Quorum-commits one typed control-plane mutation and answers the
    /// committed index (control entries carry no client outcome). Runs
    /// on the owner thread.
    pub(crate) async fn propose_control_typed(
        &self,
        mutation: kivi_control::ControlMutation,
    ) -> Result<ConsensusLogIndex, ControlProposeError> {
        use openraft::errors::{ClientWriteError, RaftError};
        let response = self
            .raft
            .client_write(ConsensusCommand::Control(mutation))
            .await
            .map_err(|error| match &error {
                RaftError::APIError(ClientWriteError::ForwardToLeader(forward)) => {
                    ControlProposeError::NotLeader {
                        leader: forward.leader_id.map(NodeId::from_u64),
                    }
                }
                _ => ControlProposeError::Unavailable {
                    detail: format!("control write failed: {error}"),
                },
            })?;
        Ok(ConsensusLogIndex::new(response.log_id.index))
    }

    /// Quorum-commits one typed control-plane mutation and answers a
    /// human-readable failure for the legacy proposal front.
    #[allow(dead_code)]
    async fn propose_control(
        &self,
        mutation: kivi_control::ControlMutation,
    ) -> Result<ConsensusLogIndex, String> {
        self.propose_control_typed(mutation)
            .await
            .map_err(|error| error.to_string())
    }

    /// Quorum-commits one deterministic command and maps the result onto
    /// the Kivi-owned proposal contract. Runs on the owner thread.
    ///
    /// The commit is internal until [`OwnerCtx::coverage_gate`] passes:
    /// a successful write is not acknowledged before every currently
    /// active roster responder can serve it, so a later local read can
    /// never miss an externally completed write.
    async fn propose(&self, command: ConsensusCommand) -> Result<ProposeOutcome, ProposeError> {
        let response = self
            .raft
            .client_write(command)
            .await
            .map_err(|error| client_write_error(&error))?;
        let index = response.log_id.index;
        let outcome = response.data.outcome().cloned().ok_or_else(|| {
            ProposeError::Consensus(ConsensusError::Unavailable {
                reason: "committed entry carried no outcome".to_owned(),
            })
        })?;
        self.coverage_gate(crate::state_machine::commit_of_index(index))
            .await?;
        Ok(ProposeOutcome::Applied {
            index: ConsensusLogIndex::new(index),
            outcome,
        })
    }

    /// Highest committed Raft log index known locally (`0` before the
    /// first commit). Runs on the owner thread.
    async fn committed_index(&self) -> u64 {
        let mut store = self.store.clone();
        store
            .read_committed()
            .await
            .ok()
            .flatten()
            .map_or(0, |id| id.index)
    }

    /// Lease-driver world view from Raft metrics: current term, voter set,
    /// leadership belief, and highest appended index as a commit position
    /// (`UNASSIGNED` before the first append — a threshold vouching for
    /// nothing). Runs on the owner thread (metrics handles are `!Send`).
    fn driver_view(&self) -> (kivi_types::RosterTerm, Vec<NodeId>, bool, CommitPosition) {
        let metrics = self.raft.metrics().borrow_watched().clone();
        let term = kivi_types::RosterTerm::from_u64(metrics.current_term);
        let leader = metrics.current_leader.map(NodeId::from_u64);
        let is_leader = leader.is_some_and(|node| node == self.local);
        let voters: Vec<NodeId> = metrics
            .membership_config
            .membership()
            .voter_ids()
            .map(NodeId::from_u64)
            .collect();
        let accepted = metrics.last_log_index.map_or(
            CommitPosition::UNASSIGNED,
            crate::state_machine::commit_of_index,
        );
        (term, voters, is_leader, accepted)
    }

    /// Responder-coverage gate: after internal commit, poll every active
    /// roster responder until each applied through `commit` and can serve
    /// from it. Rounds repeat on a short sleep so ordinary replication
    /// lag (milliseconds) never fences a live responder; after the
    /// bounded window the still-uncovered responders are fenced (explicit
    /// revoke, never assumed) and the write fails with an uncertain
    /// outcome instead of completing uncovered. Empty responder sets pass
    /// immediately, so the gate costs nothing unless leases are engaged.
    ///
    /// Before polling, the takeover fence runs: a leader that sat out the
    /// previous term waits out forgotten old-term holds (bounded by the
    /// protocol-derived quarantine) instead of completing writes a
    /// partitioned old responder could miss.
    async fn coverage_gate(&self, commit: CommitPosition) -> Result<(), ProposeError> {
        if let Some(hold) = self.hub.takeover_hold(self.tablet, process_ticks()) {
            compio::time::sleep(hold).await;
        }
        let mut responders = self.hub.covered_grantees(self.tablet, process_ticks());
        responders.retain(|(peer, _)| *peer != self.local);
        if responders.is_empty() {
            return Ok(());
        }
        self.hub.note_coverage_wait(self.tablet);
        responders.sort_by_key(|(peer, _)| peer.as_u64());
        let mut uncovered: Vec<NodeId> = responders.into_iter().map(|(peer, _)| peer).collect();
        let deadline = std::time::Instant::now() + COVERAGE_WAIT;
        loop {
            let mut still_out = Vec::new();
            for peer in uncovered {
                if !self.poll_coverage_once(peer, commit).await {
                    still_out.push(peer);
                }
            }
            uncovered = still_out;
            if uncovered.is_empty() {
                return Ok(());
            }
            if std::time::Instant::now() >= deadline {
                break;
            }
            compio::time::sleep(Duration::from_millis(100)).await;
        }
        let now = process_ticks();
        for peer in &uncovered {
            let revokes = self.hub.lease_revoke_peer(self.tablet, *peer, now);
            self.send_lease_batches(revokes).await;
        }
        self.hub.note_coverage_timeout(self.tablet);
        Err(ProposeError::ResponderCoverage {
            detail: format!(
                "tablet {} commit {} uncovered by responders {:?}; fenced, outcome uncertain",
                self.tablet.as_u64(),
                commit.as_u64(),
                uncovered
                    .iter()
                    .map(|node| node.as_u64())
                    .collect::<Vec<_>>(),
            ),
        })
    }

    /// One coverage poll of one responder: `true` iff it already applied
    /// through `commit` and can serve from it. `false` means "not yet (or
    /// not reachable)" — the caller rounds again, then revokes and fails.
    async fn poll_coverage_once(&self, peer: NodeId, commit: CommitPosition) -> bool {
        use crate::peer::{PeerCoveragePoll, PeerRequest, PeerResponse};
        let request = PeerRequest::CoveragePoll(PeerCoveragePoll {
            commit: commit.as_u64(),
        });
        match self
            .transport
            .call(peer, self.group, request, COVERAGE_POLL_TIMEOUT)
            .await
        {
            Ok(PeerResponse::Coverage(report)) => {
                report.responder == peer.as_u64()
                    && report.healthy
                    && report.applied >= commit.as_u64()
            }
            Ok(_) | Err(_) => false,
        }
    }

    /// Orders one Lazy-ALR batch boundary on the leader:
    ///
    /// * `committed > formation` — a write committed while the batch
    ///   formed subsumes the extra fence. Proof (committed-prefix):
    ///   every write completing before formation committed no later than
    ///   formation began, hence at an index at or below any commit
    ///   observed after receipt; waiting the subsuming commit covers them
    ///   all. No entry is appended.
    /// * `committed == formation` — no newer commit observed: append a
    ///   dedicated fence ordered after formation. Proof: the fence
    ///   commits by majority after receipt (hence after formation), so
    ///   its prefix contains every pre-formation completed write; waiting
    ///   the fence covers them. The fence mutates nothing.
    /// * `committed < formation` — this leadership's view lags the
    ///   former's: deposed or partitioned. Refuse (the caller falls back)
    ///   rather than order a boundary this leadership cannot vouch for.
    async fn resolve_alr_boundary(
        &self,
        sync: crate::command::AlrFenceSync,
    ) -> Result<SyncOutcome, ProposeError> {
        if self.group == ConsensusGroupId::control() {
            return Err(ProposeError::Consensus(ConsensusError::Unavailable {
                reason: "read fences serve tablet groups only".to_owned(),
            }));
        }
        let formation = crate::state_machine::index_of_commit(sync.formation_applied).unwrap_or(0);
        let committed = self.committed_index().await;
        if committed < formation {
            return Err(ProposeError::Consensus(ConsensusError::LeaderUnknown));
        }
        if committed > formation {
            return Ok(SyncOutcome {
                boundary: committed,
                subsumed: true,
            });
        }
        let response = self
            .raft
            .client_write(ConsensusCommand::ReadSync(sync))
            .await
            .map_err(|error| client_write_error(&error))?;
        debug_assert!(
            response.data.outcome().is_none(),
            "fences carry no client outcome"
        );
        let boundary = response.log_id.index;
        self.coverage_gate(crate::state_machine::commit_of_index(boundary))
            .await?;
        Ok(SyncOutcome {
            boundary,
            subsumed: false,
        })
    }

    /// Orders one Lazy-ALR batch boundary from any replica: leaders
    /// resolve locally, followers forward to the leader over the peer
    /// mesh. Every failure answers `Err` for the barrier fallback.
    async fn propose_sync(
        &self,
        sync: crate::command::AlrFenceSync,
        timeout: Duration,
    ) -> Result<SyncOutcome, ProposeError> {
        use crate::peer::{PeerAlrSyncRequest, PeerRequest, PeerResponse};
        if self.group == ConsensusGroupId::control() {
            return Err(ProposeError::Consensus(ConsensusError::Unavailable {
                reason: "read fences serve tablet groups only".to_owned(),
            }));
        }
        if self.is_leader() {
            return self.resolve_alr_boundary(sync).await;
        }
        let leader = self
            .raft
            .metrics()
            .borrow_watched()
            .clone()
            .current_leader
            .map(NodeId::from_u64);
        let Some(leader) = leader else {
            return Err(ProposeError::Consensus(ConsensusError::LeaderUnknown));
        };
        if leader == self.local {
            // Metrics race (leader flag and leader id disagree): refuse
            // rather than forward to self or order unstably.
            return Err(ProposeError::Consensus(ConsensusError::LeaderUnknown));
        }
        let request = PeerRequest::AlrSync(PeerAlrSyncRequest {
            fence_tablet: sync.fence.tablet.as_u64(),
            batch: sync.fence.batch,
            requester: sync.fence.requester.as_u64(),
            formation_applied: sync.formation_applied.as_u64(),
        });
        match self
            .transport
            .call(leader, self.group, request, timeout)
            .await
        {
            Ok(PeerResponse::AlrSync(reply)) => Ok(SyncOutcome {
                boundary: reply.boundary,
                subsumed: reply.subsumed,
            }),
            Ok(unexpected) => Err(ProposeError::Consensus(ConsensusError::Unavailable {
                reason: format!("leader answered ALR sync with {unexpected:?}"),
            })),
            Err(_) => Err(ProposeError::Consensus(ConsensusError::LeaderUnknown)),
        }
    }

    /// Serves one inbound lease batch through the tablet's engine and
    /// answers the replies it produced (piggybacked, same batch shape).
    /// Infallible: unknown tablets and foreign providers ignore the batch
    /// (the engine refuses what it must), so there is no error to report.
    fn serve_lease(&self, from: NodeId, batch: crate::peer::PeerLeaseBatch) -> PeerResponse {
        let replies = self
            .hub
            .lease_receive(self.tablet, from, batch.messages, process_ticks());
        PeerResponse::Lease(crate::peer::PeerLeaseBatch {
            messages: replies
                .into_iter()
                .flat_map(|group| group.messages)
                .collect(),
        })
    }

    /// Serves one Lazy-ALR sync request (leader only): resolve the
    /// boundary and answer it. Refusals fail the follower's batch into
    /// its barrier fallback — never a local serve on a guess.
    async fn serve_alr_sync(
        &self,
        request: crate::peer::PeerAlrSyncRequest,
    ) -> Result<PeerResponse, PeerRpcError> {
        let refused = |detail: String| PeerRpcError { detail };
        if !self.is_leader() {
            return Err(refused(
                "not the leader; ALR syncs order on the leader".to_owned(),
            ));
        }
        if request.fence_tablet != self.tablet.as_u64() {
            return Err(refused(format!(
                "fence for tablet {} reached tablet {}",
                request.fence_tablet,
                self.tablet.as_u64(),
            )));
        }
        let sync = crate::command::AlrFenceSync::new(
            kivi_types::AlrFence::new(
                self.tablet,
                request.batch,
                NodeId::from_u64(request.requester),
            ),
            CommitPosition::from_u64(request.formation_applied),
        );
        match self.resolve_alr_boundary(sync).await {
            Ok(outcome) => Ok(PeerResponse::AlrSync(crate::peer::PeerAlrSyncResponse {
                boundary: outcome.boundary,
                subsumed: outcome.subsumed,
            })),
            Err(error) => Err(refused(format!("ALR sync refused: {error}"))),
        }
    }

    /// Serves one responder-coverage poll from the applied pointer: the
    /// position plus whether this replica can satisfy the logical read
    /// contract from it (healthy flag covers sidecar resolvability and
    /// latched faults — an unhealthy replica covers nothing).
    async fn serve_coverage(
        &self,
        poll: crate::peer::PeerCoveragePoll,
    ) -> Result<PeerResponse, PeerRpcError> {
        let status = self.machine.status().await;
        let _ = poll;
        Ok(PeerResponse::Coverage(crate::peer::PeerCoverageReport {
            responder: self.local.as_u64(),
            incarnation: self.incarnation.as_u64(),
            applied: status.applied_commit.as_u64(),
            healthy: status.healthy,
        }))
    }

    /// One lease-driver tick: reconcile the observed world into the
    /// tablet's engine and flush outbound traffic, following piggybacked
    /// replies one bounded level (the rest rides the next tick; message
    /// loss/duplication/reordering is protocol-tolerated).
    async fn lease_tick_drive(&self) {
        let (term, voters, is_leader, accepted) = self.driver_view();
        let committed_at = self.committed_index().await;
        let committed = if committed_at == 0 {
            CommitPosition::UNASSIGNED
        } else {
            crate::state_machine::commit_of_index(committed_at)
        };
        let outbound = self.hub.lease_drive(
            self.tablet,
            self.authority,
            term,
            voters,
            is_leader,
            accepted,
            committed,
            process_ticks(),
        );
        self.send_lease_batches(outbound).await;
    }

    /// Sends grouped lease batches, feeding each piggybacked reply back
    /// into the engine up to two levels deep.
    async fn send_lease_batches(&self, batches: Vec<crate::consistency::LeaseDriveOut>) {
        let mut pending = batches;
        for _ in 0..3 {
            if pending.is_empty() {
                break;
            }
            let mut next = Vec::new();
            for batch in pending {
                if batch.messages.is_empty() {
                    continue;
                }
                let target = batch.target;
                let request = PeerRequest::Lease(crate::peer::PeerLeaseBatch {
                    messages: batch.messages,
                });
                if let Ok(PeerResponse::Lease(reply)) = self
                    .transport
                    .call(target, self.group, request, LEASE_RPC_TIMEOUT)
                    .await
                {
                    next.extend(self.hub.lease_receive(
                        self.tablet,
                        target,
                        reply.messages,
                        process_ticks(),
                    ));
                }
            }
            pending = next;
        }
    }

    /// Registers a learner on the group leader. Must run on the owner
    /// thread (the `Raft` handle is `!Send`). With `blocking=true` this
    /// returns once the leader observes the learner line-rate
    /// (replication lag within threshold), which is the promotion gate:
    /// never promote a learner that has not caught up.
    async fn add_learner(&self, node: NodeId, addr: String, blocking: bool) -> Result<(), String> {
        self.raft
            .add_learner(node.as_u64(), openraft::BasicNode::new(addr), blocking)
            .await
            .map(|_| ())
            .map_err(|error| format!("add_learner failed: {error}"))
    }

    /// Replaces the voter set through joint consensus (handled
    /// internally by `OpenRaft`: joint config commit, then the uniform
    /// successor). New voters must already be learners. Must run on the
    /// owner thread.
    async fn change_membership(
        &self,
        voters: std::collections::BTreeSet<u64>,
        retain: bool,
    ) -> Result<(), String> {
        self.raft
            .change_membership(voters, retain)
            .await
            .map(|_| ())
            .map_err(|error| format!("change_membership failed: {error}"))
    }

    /// Asks the leader to hand leadership to `to`. Fire-and-forget: a
    /// non-leader ignores it, and a changed leadership races safely —
    /// the reconciler re-observes and continues. Must run on the owner
    /// thread.
    async fn transfer_leader(&self, to: NodeId) -> Result<(), String> {
        self.raft
            .trigger()
            .transfer_leader(to.as_u64())
            .await
            .map_err(|error| format!("transfer_leader dispatch failed: {error}"))
    }

    /// Observes actual voter/learner/leader state from metrics. Runs on
    /// the owner thread (metrics handles are `!Send`).
    fn observe_membership(&self) -> crate::control::ObservedMembership {
        let metrics = self.raft.metrics().borrow_watched().clone();
        let stored = metrics.membership_config;
        let membership = stored.membership();
        let voters = membership.voter_ids().collect();
        let learners = membership.learner_ids().collect();
        // Replication lag per node from the leader's match indexes
        // (followers report unknown: only the leader's view gates
        // promotion).
        let mut lag = std::collections::BTreeMap::new();
        if let Some(replication) = metrics.replication.as_ref() {
            let last = metrics.last_log_index.unwrap_or(0);
            for (node, matched) in replication {
                let matched_index = matched.as_ref().map_or(0, |id| id.index);
                lag.insert(*node, last.saturating_sub(matched_index));
            }
        }
        crate::control::ObservedMembership::with_lag(
            self.group,
            voters,
            learners,
            metrics
                .current_leader
                .map(|node| ReplicaId::of_node(NodeId::from_u64(node))),
            ConsensusTerm::new(metrics.current_term),
            membership.get_joint_config().len() > 1,
            lag,
        )
    }

    /// Establishes a `ReadIndex` linearizable barrier: the leader proves
    /// freshness with a quorum, waits for local applied coverage of the
    /// leadership-bound boundary, and returns the owned barrier. The
    /// leadership identity and boundary are preserved (never collapsed
    /// to a bare index) for future `AtLeast`/follower-read work.
    async fn barrier(&self) -> Result<ReadBarrier, ConsensusError> {
        let fallback = self.leader_fallback();
        let read = self
            .raft
            .ensure_linearizable(openraft::ReadPolicy::ReadIndex)
            .await
            .map_err(|error| map_raft_error(&error, fallback))?;
        Ok(ReadBarrier {
            term: ConsensusTerm::new(read.committed_leader_id().term),
            leader: replica_of(read.committed_leader_id().node_id),
            boundary: ConsensusLogIndex::new(read.index()),
        })
    }

    /// Waits for local applied coverage of `index` on bounded reactor
    /// sleeps; the caller stays runtime-agnostic. `timeout` is capped by
    /// `AT_LEAST_TIMEOUT` no matter what the caller asked: every
    /// queue/retry/wait is bounded. Timeout is `Err(())` (the caller maps
    /// it to `CoverageTimeout`); owner death surfaces through the dropped
    /// reply instead.
    async fn wait_applied(&self, index: u64, timeout: Duration) -> Result<(), ()> {
        let deadline = std::time::Instant::now() + timeout.min(AT_LEAST_TIMEOUT);
        loop {
            if self.machine.status().await.applied_index.unwrap_or(0) >= index {
                return Ok(());
            }
            if std::time::Instant::now() >= deadline {
                return Err(());
            }
            compio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Whether this replica's owner currently believes it leads the group.
    /// Local metrics read, no quorum contact: the Almost-Local fast path
    /// combines it with receipt age and authority checks, and falls back
    /// the moment it reports follower.
    fn is_leader(&self) -> bool {
        self.raft
            .metrics()
            .borrow_watched()
            .current_leader
            .is_some_and(|leader| leader == self.local.as_u64())
    }

    /// Best-effort leader hint from metrics (fills gaps when the error
    /// itself names no leader).
    fn leader_fallback(&self) -> Option<ReplicaId> {
        self.raft
            .metrics()
            .borrow_watched()
            .current_leader
            .map(replica_of)
    }

    /// Gathers full diagnostics on the owner thread (metrics handles are
    /// `!Send`, so this cannot run on the caller).
    async fn status(&self) -> NodeStatus {
        let metrics = self.raft.metrics().borrow_watched().clone();
        let role = match metrics.state {
            openraft::ServerState::Leader => ReplicaRole::Leader,
            openraft::ServerState::Follower => ReplicaRole::Follower,
            openraft::ServerState::Candidate => ReplicaRole::Candidate,
            // Learners do not exist yet; shutdown is reported through
            // `healthy`, so both map to the closest serving role.
            openraft::ServerState::Learner | openraft::ServerState::Shutdown => {
                ReplicaRole::Follower
            }
        };
        let mut store = self.store.clone();
        let committed = store
            .read_committed()
            .await
            .ok()
            .flatten()
            .map_or(0, |id| id.index);
        let mut machine = self.machine.clone();
        let (applied_id, _) = machine.applied_state().await.unwrap_or_default();
        let applied = applied_id.map_or(0, |id| id.index);
        let sm = self.machine.status().await;
        let running_ok = metrics.running_state.is_ok();
        let running_error = match &metrics.running_state {
            Ok(()) => None,
            Err(error) => Some(format!("raft halted: {error}")),
        };
        let detail = match (sm.last_error, running_error) {
            (Some(sm_error), Some(raft_error)) => Some(format!("{sm_error}; {raft_error}")),
            (Some(sm_error), None) => Some(sm_error),
            (None, Some(raft_error)) => Some(raft_error),
            (None, None) => None,
        };
        // Sidecar diagnostics are best-effort (never fail status).
        let sidecar_metrics = self.sidecar.metrics().snapshot();
        NodeStatus {
            group: self.group,
            replica: ReplicaId::of_node(self.transport.local().node),
            role,
            term: ConsensusTerm::new(metrics.current_term),
            leader: metrics.current_leader.map(replica_of),
            last_log: ConsensusLogIndex::new(metrics.last_log_index.unwrap_or(0)),
            committed: ConsensusLogIndex::new(committed),
            applied: ConsensusLogIndex::new(applied),
            applied_commit: sm.applied_commit,
            snapshot: sm.snapshot_index.map(ConsensusLogIndex::new),
            purged: metrics.purged.map(|id| ConsensusLogIndex::new(id.index)),
            healthy: sm.healthy && running_ok,
            detail,
            peers: self.transport.stats().await,
            sidecar: sidecar_metrics,
            preflight: self.preflight.snapshot(),
            consistency: kivi_types::ConsistencySnapshot::default(),
        }
    }

    /// Serves one peer RPC for the owned group. Election/append/pre-vote
    /// go straight to Raft; snapshot fragments reassemble here and
    /// install whole via `install_full_snapshot` once complete; manifest
    /// and chunk sidecars serve from the local sidecar store on the bulk
    /// lane (content identity only, never pack offsets); redundancy
    /// fragments serve from the node-local fragment store on the bulk
    /// lane (opaque proto bytes, incarnation-fenced, group-independent);
    /// preflight prepares immutable roots through the durability gate on the bulk
    /// lane (same acquisition primitives as the append gate, replying
    /// only once durable — never consensus, never logical state).
    async fn serve_rpc(
        &self,
        from: NodeId,
        group: ConsensusGroupId,
        request: PeerRequest,
    ) -> Result<PeerResponse, PeerRpcError> {
        // Sidecars and redundancy fragments are group-independent
        // (content-addressed, domain-scoped) and decode with a zero
        // tablet: serve before the group check.
        match request {
            PeerRequest::Manifest(request) => return self.serve_manifest(request).await,
            PeerRequest::Chunk(request) => return self.serve_chunk(request).await,
            PeerRequest::Prepare(request) => return self.serve_prepare(request).await,
            PeerRequest::Fragment(bytes) => {
                return self.serve_redundancy_fragment(&bytes);
            }
            _ => {}
        }
        if group != self.group {
            return Err(PeerRpcError {
                detail: format!("group {group} not served here"),
            });
        }
        match request {
            PeerRequest::Snapshot(fragment) => self.serve_fragment(fragment).await,
            PeerRequest::Lease(batch) => Ok(self.serve_lease(from, batch)),
            PeerRequest::AlrSync(request) => self.serve_alr_sync(request).await,
            PeerRequest::CoveragePoll(poll) => self.serve_coverage(poll).await,
            PeerRequest::ControlForward(request) => {
                self.serve_control_forward(group, request).await
            }
            other => serve_peer_request(&self.raft, other).await,
        }
    }

    async fn serve_control_forward(
        &self,
        group: ConsensusGroupId,
        request: crate::peer::PeerControlForwardRequest,
    ) -> Result<PeerResponse, PeerRpcError> {
        if group != ConsensusGroupId::control() {
            return Err(PeerRpcError {
                detail: format!("control forward refused for {group}"),
            });
        }
        let mutation =
            kivi_control::ControlMutation::decode_exact(&request.mutation).map_err(|error| {
                PeerRpcError {
                    detail: format!("control forward mutation decode failed: {error}"),
                }
            })?;
        match self.propose_control_typed(mutation).await {
            Ok(index) => Ok(PeerResponse::ControlForward(
                crate::peer::PeerControlForwardResponse { index: index.get() },
            )),
            Err(ControlProposeError::NotLeader { leader }) => {
                let detail = leader.map_or_else(
                    || "control forward: leader unknown".to_owned(),
                    |leader| format!("control forward: leader {}", leader.as_u64()),
                );
                Err(PeerRpcError { detail })
            }
            Err(ControlProposeError::Unavailable { detail }) => Err(PeerRpcError { detail }),
        }
    }

    /// Serves one manifest fetch from the local sidecar store.
    async fn serve_manifest(
        &self,
        request: crate::peer::PeerManifestRequest,
    ) -> Result<PeerResponse, PeerRpcError> {
        let refused = |detail: String| PeerRpcError { detail };
        let id = kivi_types::ManifestId::from_bytes(request.manifest);
        if !id.is_valid() {
            return Err(refused("zero manifest id".to_owned()));
        }
        let canonical = self
            .sidecar
            .read_manifest(id)
            .await
            .map_err(|error| refused(format!("manifest {id} unavailable: {error}")))?;
        self.sidecar
            .metrics()
            .bulk_bytes_sent
            .fetch_add(canonical.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(PeerResponse::Manifest(PeerManifestResponse {
            manifest: request.manifest,
            canonical,
        }))
    }

    /// Serves one chunk fetch from the local sidecar store.
    async fn serve_chunk(
        &self,
        request: crate::peer::PeerChunkRequest,
    ) -> Result<PeerResponse, PeerRpcError> {
        let refused = |detail: String| PeerRpcError { detail };
        let id = kivi_types::ChunkId::from_bytes(request.chunk);
        if !id.is_valid() {
            return Err(refused("zero chunk id".to_owned()));
        }
        let bytes = self
            .sidecar
            .read_chunk(id)
            .await
            .map_err(|error| refused(format!("chunk {id} unavailable: {error}")))?;
        self.sidecar
            .metrics()
            .bulk_bytes_sent
            .fetch_add(bytes.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(PeerResponse::Chunk(PeerChunkResponse {
            chunk: request.chunk,
            bytes,
        }))
    }

    /// Serves one sidecar preflight: ensures the named immutable root is
    /// locally durable and verified through the exact same acquisition
    /// primitives as the append gate and snapshot recovery, replying only
    /// once the durability requirement is satisfied. Mutates no logical
    /// state and creates no Raft decision. Abandoned preflight data stays
    /// as reusable content-addressed cache.
    async fn serve_prepare(
        &self,
        request: crate::peer::PeerPrepareRequest,
    ) -> Result<PeerResponse, PeerRpcError> {
        let refused = |detail: String| PeerRpcError { detail };
        let id = kivi_types::ManifestId::from_bytes(request.manifest);
        if !id.is_valid() {
            return Err(refused("zero manifest id".to_owned()));
        }
        self.gate
            .ensure_root(id, request.logical_len, None)
            .await
            .map_err(|error| {
                refused(format!(
                    "preflight root {id} unavailable: {}",
                    crate::gate::sidecar_io_error(&error)
                ))
            })?;
        Ok(PeerResponse::Prepared(crate::peer::PeerPreparedResponse {
            manifest: request.manifest,
        }))
    }

    /// Serves one redundancy fragment RPC from the node-local fragment
    /// store: proto-decode, incarnation fence, then dispatch (stage
    /// verifies hashes before storing; fetch never serves bad bytes).
    /// Group-independent like the sidecars — any replica serves from the
    /// shared store. Protocol-level faults refuse the RPC; stale, missing,
    /// and oversize verdicts return as encoded `Refused` answers; bulk
    /// bytes are metered by the transport lane counters, never here.
    fn serve_redundancy_fragment(&self, bytes: &[u8]) -> Result<PeerResponse, PeerRpcError> {
        crate::fragments::handle_fragment_rpc(
            self.fragments.as_deref(),
            self.incarnation.as_u64(),
            bytes,
        )
    }

    /// Buffers one snapshot fragment; on `done`, validates and installs
    /// the whole image. Gaps, meta conflicts, and mid-transfer restarts
    /// (fresh transfer id) reset or refuse loudly — the sender retries
    /// the transfer from scratch with a new id, never resumes a
    /// corrupted reassembly.
    async fn serve_fragment(
        &self,
        fragment: PeerSnapshotRequest,
    ) -> Result<PeerResponse, PeerRpcError> {
        let refused = |detail: String| PeerRpcError { detail };
        // Decode-first, buffer-second: metadata faults refuse before any
        // bytes are retained.
        let meta = decode_snapshot_meta(&fragment.meta)
            .map_err(|error| refused(format!("snapshot metadata invalid: {error}")))?;
        let vote = crate::router::decode_wire_vote(&fragment.vote);
        // A fresh transfer id always starts a new reassembly (a retried
        // transfer never continues an aborted one's bytes). The map
        // borrow never crosses an await: all branches below drop it
        // before any async work.
        let step = {
            let mut transfers = self.transfers.borrow_mut();
            let partial = transfers
                .entry(fragment.transfer)
                .or_insert(PartialTransfer {
                    vote,
                    meta: meta.clone(),
                    bytes: Vec::new(),
                });
            if partial.meta != meta {
                transfers.remove(&fragment.transfer);
                Err(refused(
                    "snapshot transfer metadata changed mid-stream".to_owned(),
                ))
            } else if fragment.offset != partial.bytes.len() as u64 {
                // Gap or overlap: drop the partial attempt; the sender's
                // next transfer (fresh id) starts clean.
                let buffered = partial.bytes.len();
                transfers.remove(&fragment.transfer);
                Err(refused(format!(
                    "snapshot fragment gap: offset {} != buffered {buffered}",
                    fragment.offset,
                )))
            } else {
                partial.bytes.extend_from_slice(&fragment.data);
                Ok((partial.vote, fragment.done))
            }
        };
        let (ack_vote, done) = step?;
        if !done {
            return Ok(PeerResponse::Snapshot(crate::peer::PeerSnapshotResponse {
                vote: crate::router::encode_wire_vote(&ack_vote),
            }));
        }
        let partial = self
            .transfers
            .borrow_mut()
            .remove(&fragment.transfer)
            .expect("transfer buffered above");
        self.install_transfer(partial).await
    }

    /// Pre-validates and installs one reassembled snapshot. The
    /// prechecks keep a forged or skewed transfer out of
    /// `install_full_snapshot`, which panics (by upstream contract) on
    /// vote/committed inconsistency instead of returning an error.
    async fn install_transfer(
        &self,
        partial: PartialTransfer,
    ) -> Result<PeerResponse, PeerRpcError> {
        use openraft::type_config::alias::SnapshotOf;
        let refused = |detail: String| PeerRpcError { detail };
        // Stale vote: the sender lost leadership mid-transfer. Refuse
        // without touching Raft (a smaller vote would trip the
        // install-time leadership assert).
        let local_vote = {
            let mut store = self.store.clone();
            store.read_vote().await.map_err(|error| {
                refused(format!("snapshot precheck failed to read vote: {error}"))
            })?
        };
        if let Some(local) = local_vote
            && vote_is_stale(&local, &partial.vote)
        {
            return Err(refused("snapshot transfer vote is stale".to_owned()));
        }
        // Committed-history conflict: a snapshot at or behind committed
        // history contradicts it unless it is exactly the committed
        // prefix (which `install_full_snapshot` accepts as a no-op).
        // Anything else would trip the install-time committed assert.
        if let Some(last) = partial.meta.last_log_id {
            let mut store = self.store.clone();
            let committed = store.read_committed().await.map_err(|error| {
                refused(format!("snapshot precheck failed to read commit: {error}"))
            })?;
            if let Some(committed) = committed
                && (last.index < committed.index
                    || (last.index == committed.index && last != committed))
            {
                return Err(refused(
                    "snapshot contradicts locally committed history".to_owned(),
                ));
            }
        }
        // Differential sidecar catch-up: the snapshot body carries only
        // manifest references; bulk bytes stay in chunk packs. Fetch only
        // missing immutable objects before installing, so the install never
        // lands a root it cannot reconstruct. Transfer is content-addressed:
        // a follower holding 95% of chunks downloads only the missing 5%.
        // Fabric references never cross nodes (worker-local ids): a
        // snapshot naming one is refused — the leader must ship bytes
        // (or chunked sidecars), never foreign fabric names.
        if let Ok(decoded) = crate::state_machine::ReplicatedTablet::decode_snapshot(
            &partial.bytes,
            self.namespace,
            self.tablet,
        ) {
            let leader_hint = Some(NodeId::from_u64(partial.vote.leader_id.node_id));
            for (_key, object) in &decoded.objects {
                if let kivi_state::LogicalValue::Chunked(chunked) = object.value() {
                    self.gate
                        .ensure_root(chunked.manifest, chunked.logical_len, leader_hint)
                        .await
                        .map_err(|error| {
                            refused(format!("snapshot sidecar unavailable: {error}"))
                        })?;
                }
                if let kivi_state::LogicalValue::Fabric(fabric) = object.value() {
                    return Err(refused(format!(
                        "snapshot names foreign fabric reference {}: leader must ship bytes",
                        fabric.id
                    )));
                }
            }
        }
        let snapshot = SnapshotOf::<KiviTypeConfig, std::io::Cursor<Vec<u8>>> {
            meta: partial.meta,
            snapshot: std::io::Cursor::new(partial.bytes),
        };
        let response = self
            .raft
            .install_full_snapshot(partial.vote, snapshot)
            .await
            .map_err(|error| refused(format!("snapshot install refused: {error}")))?;
        Ok(PeerResponse::Snapshot(crate::peer::PeerSnapshotResponse {
            vote: crate::router::encode_wire_vote(&response.vote),
        }))
    }

    /// Triggers a snapshot seal, waits for the durable base, then purges
    /// the log through it — all on the owner thread (reactor sleeps; the
    /// caller stays runtime-agnostic).
    async fn snapshot_purge(&self) -> Result<ConsensusLogIndex, String> {
        if !self.machine.status().await.healthy {
            return Err("state machine unhealthy: snapshot blocked".to_owned());
        }
        self.raft
            .trigger()
            .snapshot()
            .await
            .map_err(|error| format!("snapshot trigger refused: {error}"))?;
        // Triggers return at once; wait for the durable base to land.
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        let base = loop {
            if let Some(base) = self.machine.snapshotted_index().await {
                break base;
            }
            if std::time::Instant::now() >= deadline {
                return Err("snapshot never sealed before the deadline".to_owned());
            }
            compio::time::sleep(Duration::from_millis(50)).await;
        };
        self.raft
            .trigger()
            .purge_log(base)
            .await
            .map_err(|error| format!("purge trigger refused: {error}"))?;
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        loop {
            let purged = self
                .raft
                .metrics()
                .borrow_watched()
                .purged
                .map_or(0, |id| id.index);
            if purged >= base {
                break;
            }
            if std::time::Instant::now() >= deadline {
                return Err(format!(
                    "log never purged through {base} before the deadline (purged {purged})"
                ));
            }
            compio::time::sleep(Duration::from_millis(50)).await;
        }
        Ok(ConsensusLogIndex::new(base))
    }
}

/// Whether `incoming` is strictly older than `local` under Raft vote
/// order (term, then leader, then committed-beats-uncommitted). Used to
/// refuse stale snapshot transfers before they reach the install-time
/// leadership assert.
fn vote_is_stale(
    local: &openraft::type_config::alias::VoteOf<KiviTypeConfig>,
    incoming: &openraft::type_config::alias::VoteOf<KiviTypeConfig>,
) -> bool {
    (
        local.leader_id.term,
        local.leader_id.node_id,
        local.committed,
    ) > (
        incoming.leader_id.term,
        incoming.leader_id.node_id,
        incoming.committed,
    )
}

/// Inputs to [`bootstrap_group`], bundled so the arg count stays readable.
pub(crate) struct BootstrapInputs<'a> {
    pub(crate) topology: &'a ClusterTopology,
    pub(crate) tablet: TabletId,
    pub(crate) local: NodeId,
    pub(crate) cluster: ClusterId,
    pub(crate) node: NodeId,
    pub(crate) voters: &'a std::collections::BTreeSet<u64>,
    pub(crate) peer_addrs: &'a std::collections::BTreeMap<u64, String>,
}

/// Installs or verifies group membership: first formation installs the
/// joint membership exactly once; rejoins recover existing Raft state
/// (or wait for learner replication when the group is locally fresh)
/// and fail loudly on identity conflicts instead of initializing a new
/// group. Initializing on rejoin would fork a live group's history —
/// `founding` (true only on a brand-new data directory) gates it.
///
/// Runs on the owner thread (Raft futures are `!Send`).
///
/// # Errors
///
/// Returns [`NodeOpenError::Initialize`] when first formation fails, or
/// [`NodeOpenError::Bootstrap`] when restart config conflicts with
/// durable state.
pub(crate) async fn bootstrap_group<S>(
    inputs: &BootstrapInputs<'_>,
    founding: bool,
    store: &mut S,
    machine: &ReplicatedStateMachine,
    raft: &OwnerRaft,
) -> Result<(), NodeOpenError>
where
    S: ConsensusLogStore,
{
    use NodeOpenError as Fault;
    let BootstrapInputs {
        topology,
        tablet,
        local,
        cluster,
        node,
        voters,
        peer_addrs,
    } = *inputs;
    // Any trace of vote, log, or snapshot base means existing — recover
    // it, never re-initialize.
    let bootstrap = classify_from_store(store, machine).await;
    match bootstrap {
        Bootstrap::Fresh if founding => {
            let members: std::collections::BTreeMap<u64, openraft::BasicNode> = voters
                .iter()
                .map(|voter| {
                    (
                        *voter,
                        openraft::BasicNode::new(
                            peer_addrs.get(voter).cloned().unwrap_or_default(),
                        ),
                    )
                })
                .collect();
            raft.initialize(members)
                .await
                .map_err(|error| Fault::Initialize {
                    reason: error.to_string(),
                })?;
            Ok(())
        }
        // Locally fresh on rejoin: wait for learner replication (the
        // leader brings state); initializing here would fork history.
        Bootstrap::Fresh => Ok(()),
        Bootstrap::Existing => {
            // Recover durable state, never re-initialize. Voter sets
            // are not compared (control-plane moves outlive restarts);
            // removed-while-down replicas were tombstoned at open
            // through the desired oracle before reaching this worker.
            crate::cluster::verify_against_durable(
                topology,
                tablet,
                local,
                &DurableClusterView { cluster, node },
            )
            .map_err(|error| Fault::Bootstrap {
                reason: error.to_string(),
            })
        }
    }
}

/// Decides fresh vs. existing from durable signals: any vote, any log
/// entry, or any snapshot base means existing.
pub(crate) async fn classify_from_store<S>(
    store: &mut S,
    machine: &ReplicatedStateMachine,
) -> Bootstrap
where
    S: ConsensusLogStore,
{
    let has_vote = matches!(store.read_vote().await, Ok(Some(_)));
    let has_log = store
        .get_log_state()
        .await
        .is_ok_and(|state| state.last_log_id.is_some());
    let has_snapshot = machine.snapshotted_index().await.is_some();
    classify_bootstrap(has_vote, has_log, has_snapshot)
}

/// Maps a linearizable-read failure onto Kivi-owned routing errors: a
/// named leader becomes a hint (falling back to the metrics view when
/// the error itself is leaderless), quorum loss without a leader is
/// unknown, and fatal storage faults are unavailable. Never a phantom
/// success: losing leadership between check and commit surfaces here as
/// `NotLeader`.
fn map_raft_error(
    error: &openraft::errors::RaftError<
        KiviTypeConfig,
        openraft::errors::LinearizableReadError<KiviTypeConfig>,
    >,
    fallback: Option<ReplicaId>,
) -> ConsensusError {
    use openraft::errors::{LinearizableReadError, RaftError};
    match error {
        RaftError::Fatal(fatal) => ConsensusError::Unavailable {
            reason: format!("consensus fatal: {fatal}"),
        },
        RaftError::APIError(LinearizableReadError::ForwardToLeader(forward)) => {
            ConsensusError::NotLeader {
                hint: LeaderHint {
                    leader: forward.leader_id.map(replica_of).or(fallback),
                },
            }
        }
        RaftError::APIError(LinearizableReadError::QuorumNotEnough(_)) => {
            ConsensusError::LeaderUnknown
        }
    }
}

/// Maps a `client_write` failure onto proposal errors. Membership-change
/// failures are unreachable for data writes and fail closed as
/// unavailable.
fn client_write_error(
    error: &openraft::errors::RaftError<
        KiviTypeConfig,
        openraft::errors::ClientWriteError<KiviTypeConfig>,
    >,
) -> ProposeError {
    use openraft::errors::{ClientWriteError, RaftError};
    match &error {
        RaftError::APIError(ClientWriteError::ForwardToLeader(forward)) => {
            ProposeError::Consensus(ConsensusError::NotLeader {
                hint: LeaderHint {
                    leader: forward.leader_id.map(replica_of),
                },
            })
        }
        RaftError::APIError(ClientWriteError::ChangeMembershipError(install)) => {
            ProposeError::Consensus(ConsensusError::Unavailable {
                reason: format!("unexpected membership verdict on a data write: {install}"),
            })
        }
        RaftError::Fatal(fatal) => ProposeError::Consensus(ConsensusError::Unavailable {
            reason: format!("consensus fatal: {fatal}"),
        }),
    }
}

/// Ensures proposal sidecars are durable before replication.
///
/// * `SetChunked` / `SetConditionalChunked`: the referenced root must
///   already be durable locally (cluster server stages before proposing).
///   Missing sidecars fail closed as unavailable — the leader never
///   proposes a root referencing volatile buffers.
/// * `SetRange` against a chunked base: restages by reading the old value
///   via the sidecar store, applying the patch (zero-pad gaps, Redis
///   `SETRANGE` semantics), staging the new value (untouched chunk ids
///   reuse via content addressing), and returning a restaged
///   `SetConditionalChunked` preserving expiry (`Keep`). Followers fetch
///   only missing chunks via the durability gate.
/// * All other operations need no sidecars.
///
/// Returns `Ok(None)` to proceed with `op`, `Ok(Some(restaged))` to
/// proceed with a restaged operation instead.
#[allow(clippy::too_many_lines)]
/// Proves one transactional write's chunked reference names durable local
/// payload. Plain writes carry no references and pass trivially. Fabric
/// references never validate here: ids are worker-local, so a `PutFabric`
/// on the propose path is a foreign name and fails closed.
async fn check_txn_write_sidecars(
    sidecar: &SidecarStore,
    write: &kivi_state::TxnWrite,
) -> Result<(), ProposeError> {
    use crate::types::ConsensusError;
    if let kivi_state::TxnWriteKind::PutFabric { fabric_id, .. } = &write.kind {
        return Err(ProposeError::Consensus(ConsensusError::Unavailable {
            reason: format!(
                "transactional fabric write references worker-local payload {fabric_id}: replicate bytes"
            ),
        }));
    }
    let kivi_state::TxnWriteKind::PutChunked {
        manifest,
        logical_len,
    } = &write.kind
    else {
        return Ok(());
    };
    let durable = sidecar
        .check_root(*manifest, *logical_len)
        .await
        .map_err(|error| {
            ProposeError::Consensus(ConsensusError::Unavailable {
                reason: format!("sidecar check failed: {error}"),
            })
        })?;
    if !durable {
        return Err(ProposeError::Consensus(ConsensusError::Unavailable {
            reason: format!(
                "transactional chunked write references unavailable payload: {manifest}"
            ),
        }));
    }
    Ok(())
}

/// Proves one chunked root durable locally before proposing: the manifest
/// must name staged payload, or the proposal fails loudly — a later commit
/// can never meet a durable root with unavailable payload.
async fn check_chunked_root(
    sidecar: &SidecarStore,
    manifest: kivi_types::ManifestId,
    logical_len: u64,
) -> Result<(), ProposeError> {
    let durable = sidecar
        .check_root(manifest, logical_len)
        .await
        .map_err(|error| {
            ProposeError::Consensus(ConsensusError::Unavailable {
                reason: format!("sidecar check failed: {error}"),
            })
        })?;
    if !durable {
        return Err(ProposeError::Consensus(ConsensusError::Unavailable {
            reason: format!("chunked root {manifest} not durable locally; stage before proposing"),
        }));
    }
    Ok(())
}

/// Stages one transactional medium `Put` into chunked sidecars,
/// returning the rewritten write. Every other write kind passes through
/// after the usual sidecar proof (chunked references must name durable
/// local payload; fabric references fail closed as foreign names).
async fn stage_txn_put(
    sidecar: &SidecarStore,
    write: &kivi_state::TxnWrite,
) -> Result<kivi_state::TxnWrite, ProposeError> {
    if let kivi_state::TxnWriteKind::Put(value) = &write.kind
        && value.len() > 256
        && (value.len() as u64) <= kivi_chunk::policy::DEFAULT_INLINE_THRESHOLD
    {
        let staged = sidecar.stage_value(value).await.map_err(|error| {
            ProposeError::Consensus(ConsensusError::Unavailable {
                reason: format!("medium stage failed: {error}"),
            })
        })?;
        return Ok(kivi_state::TxnWrite {
            key: write.key.clone(),
            kind: kivi_state::TxnWriteKind::PutChunked {
                manifest: staged.manifest,
                logical_len: staged.logical_len,
            },
            expect: write.expect,
        });
    }
    check_txn_write_sidecars(sidecar, write).await?;
    Ok(write.clone())
}

/// Stages one medium `Set`/`SetConditional` into chunked sidecars,
/// returning the restaged operation (`None` when the op is not a medium
/// write). The Raft log carries only tiny deterministic roots while the
/// content-addressed sidecars move through the durability gate (same
/// contract as explicit uploads). Bounds mirror the engine's size split
/// (tiny inline below, chunk threshold above): tiny values stay inline,
/// large values ride explicit streaming uploads.
async fn stage_medium_set(
    sidecar: &SidecarStore,
    op: &Operation,
) -> Result<Option<Operation>, ProposeError> {
    // Medium values replicate as chunked sidecars, never inline bytes.
    const MEDIUM_INLINE_MAX: usize = 256;
    if let Operation::Set { key, value } = op
        && value.len() > MEDIUM_INLINE_MAX
        && (value.len() as u64) <= kivi_chunk::policy::DEFAULT_INLINE_THRESHOLD
    {
        let staged = sidecar.stage_value(value).await.map_err(|error| {
            ProposeError::Consensus(ConsensusError::Unavailable {
                reason: format!("medium stage failed: {error}"),
            })
        })?;
        return Ok(Some(Operation::SetChunked {
            key: key.clone(),
            manifest: staged.manifest,
            logical_len: staged.logical_len,
        }));
    }
    if let Operation::SetConditional {
        key,
        value,
        condition,
        expiry,
    } = op
        && value.len() > MEDIUM_INLINE_MAX
        && (value.len() as u64) <= kivi_chunk::policy::DEFAULT_INLINE_THRESHOLD
    {
        let staged = sidecar.stage_value(value).await.map_err(|error| {
            ProposeError::Consensus(ConsensusError::Unavailable {
                reason: format!("medium stage failed: {error}"),
            })
        })?;
        return Ok(Some(Operation::SetConditionalChunked {
            key: key.clone(),
            manifest: staged.manifest,
            logical_len: staged.logical_len,
            condition: *condition,
            expiry: *expiry,
        }));
    }
    Ok(None)
}

/// Stages medium transactional `Put`s into chunked sidecars, returning
/// the rewritten operation (`None` when nothing changed). Chunked
/// references still prove the usual durability contract; fabric
/// references fail closed as foreign names.
async fn stage_txn_sidecars(
    sidecar: &SidecarStore,
    op: &Operation,
) -> Result<Option<Operation>, ProposeError> {
    if let Operation::TxnPrepare { write, .. } = op {
        let staged = stage_txn_put(sidecar, write).await?;
        if staged.kind == write.kind {
            return Ok(None);
        }
        let Operation::TxnPrepare {
            txn,
            coordinator,
            digest,
            ..
        } = op
        else {
            unreachable!("matched TxnPrepare above");
        };
        return Ok(Some(Operation::TxnPrepare {
            txn: *txn,
            coordinator: *coordinator,
            write: staged,
            digest: *digest,
        }));
    }
    if let Operation::TxnCommitLocal { writes, .. } = op {
        let mut staged_writes = Vec::with_capacity(writes.len());
        let mut changed = false;
        for write in writes {
            let staged = stage_txn_put(sidecar, write).await?;
            changed |= staged.kind != write.kind;
            staged_writes.push(staged);
        }
        if !changed {
            return Ok(None);
        }
        let Operation::TxnCommitLocal { txn, .. } = op else {
            unreachable!("matched TxnCommitLocal above");
        };
        return Ok(Some(Operation::TxnCommitLocal {
            txn: *txn,
            writes: staged_writes,
        }));
    }
    Ok(None)
}

async fn ensure_proposal_sidecars_for(
    sidecar: &SidecarStore,
    machine: &ReplicatedStateMachine,
    op: &Operation,
    now: WallTimestamp,
) -> Result<Option<Operation>, ProposeError> {
    // Medium values replicate as chunked sidecars, never inline bytes.
    if let Some(restaged) = stage_medium_set(sidecar, op).await? {
        return Ok(Some(restaged));
    }
    if let Some(restaged) = stage_txn_sidecars(sidecar, op).await? {
        return Ok(Some(restaged));
    }
    match op {
        Operation::SetChunked {
            manifest,
            logical_len,
            ..
        }
        | Operation::SetConditionalChunked {
            manifest,
            logical_len,
            ..
        } => {
            check_chunked_root(sidecar, *manifest, *logical_len).await?;
            Ok(None)
        }
        // Fabric references never enter replicated proposals: ids are
        // worker-local names, meaningless on any other replica. Medium
        // values replicate as chunked sidecars (staged above); each
        // replica manages its own physical materialization locally.
        Operation::SetFabric { .. } | Operation::SetConditionalFabric { .. } => {
            Err(ProposeError::Unsupported {
                reason: "fabric references never enter consensus proposals; replicate bytes"
                    .to_owned(),
            })
        }
        // Transactional writes stage medium `Put`s above; chunked
        // references prove the same durability contract as plain roots
        // before a participant may say Prepared.
        Operation::TxnPrepare { write, .. } => {
            check_txn_write_sidecars(sidecar, write).await?;
            Ok(None)
        }
        Operation::TxnCommitLocal { writes, .. } => {
            for write in writes {
                check_txn_write_sidecars(sidecar, write).await?;
            }
            Ok(None)
        }
        Operation::SetRange { key, offset, patch } => {
            let base = machine.peek_base(key, now).await.map_err(|error| {
                ProposeError::Consensus(ConsensusError::Unavailable { reason: error })
            })?;
            if base != crate::state_machine::RangeBase::Chunked {
                return Ok(None);
            }
            // Chunked base: restage. Read the reference via a local Get,
            // resolve bytes via the sidecar store, patch, stage anew.
            let reference = machine
                .read_local(&Operation::Get { key: key.clone() }, now)
                .await
                .map_err(|error| {
                    ProposeError::Consensus(ConsensusError::Unavailable {
                        reason: format!("chunked base unreadable: {error}"),
                    })
                })?;
            let (manifest, logical_len) = match reference {
                kivi_state::OperationResult::ChunkedValue {
                    manifest,
                    logical_len,
                } => (manifest, logical_len),
                kivi_state::OperationResult::Value(_) | kivi_state::OperationResult::Length(_) => {
                    // Raced to inline between peek and read; proceed inline.
                    return Ok(None);
                }
                other => {
                    return Err(ProposeError::Consensus(ConsensusError::Unavailable {
                        reason: format!("unexpected chunked base read: {other:?}"),
                    }));
                }
            };
            let old = sidecar
                .read_value(manifest, logical_len)
                .await
                .map_err(|error| {
                    ProposeError::Consensus(ConsensusError::Unavailable {
                        reason: format!("chunked base bytes unreadable: {error}"),
                    })
                })?;
            let patched = apply_range_patch(&old, *offset, patch).map_err(ProposeError::Op)?;
            if patched.len() as u64 > kivi_chunk::policy::MAX_LEGACY_VALUE_BYTES {
                return Err(ProposeError::Unsupported {
                    reason: "range result exceeds 64 MiB; rewrite through streaming upload"
                        .to_owned(),
                });
            }
            let staged = sidecar.stage_value(&patched).await.map_err(|error| {
                ProposeError::Consensus(ConsensusError::Unavailable {
                    reason: format!("restage failed: {error}"),
                })
            })?;
            Ok(Some(Operation::SetConditionalChunked {
                key: key.clone(),
                manifest: staged.manifest,
                logical_len: staged.logical_len,
                condition: kivi_state::SetCondition::Always,
                expiry: kivi_state::ExpiryPolicy::Keep,
            }))
        }
        _ => Ok(None),
    }
}

/// Applies a Redis-`SETRANGE` patch: overwrites `[offset, offset+len)`
/// zero-padding past-the-end gaps. Pure; mirrors `SpliceBytes` apply.
fn apply_range_patch(
    base: &[u8],
    offset: u64,
    patch: &bytes::Bytes,
) -> Result<Vec<u8>, kivi_state::OpError> {
    use kivi_state::OpError;
    let offset = usize::try_from(offset).map_err(|_| OpError::StaleRangeBase)?;
    let patch_len = patch.len();
    let new_len = offset.saturating_add(patch_len).max(base.len());
    if new_len as u64 > kivi_chunk::policy::MAX_LEGACY_VALUE_BYTES {
        return Err(OpError::StaleRangeBase);
    }
    let mut out = vec![0u8; new_len];
    out[..base.len()].copy_from_slice(base);
    out[offset..offset + patch_len].copy_from_slice(patch);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::time::Duration;

    use kivi_state::{Key, Operation, OperationResult};
    use kivi_types::{
        ClusterId, MutationIdentity, NamespaceId, NodeId, ReadContext, ReadContract,
        RequestIdentity, RequestSeq, ServePath, SessionId, TabletAuthority, TabletEpoch, TabletId,
        Ticks, WallTimestamp, WriteGuardGeneration,
    };

    use super::{NodeConfig, ProposeError, ProposeOutcome, ReadError, ReplicatedNode};
    use crate::cluster::{ClusterTopology, NodeDescriptor, TabletAssignment};
    use crate::types::{ConsensusError, ReplicaRole};

    const CLUSTER: u128 = 0x0C10_57E2;
    const NS: NamespaceId = NamespaceId::from_u64(1);
    const TABLET: TabletId = TabletId::from_u64(9);
    const NOW: WallTimestamp = WallTimestamp::from_micros(1_000_000);
    const CTX: ReadContext =
        ReadContext::new(NOW, Ticks::from_micros(1_000_000), Duration::from_secs(5));

    fn authority() -> TabletAuthority {
        TabletAuthority::new(TABLET, TabletEpoch::INITIAL, WriteGuardGeneration::INITIAL)
    }

    /// Fast lease timing for tests: 500ms guard/lease, 50ms renewals
    /// (12.5ms ticks), 1000ppm drift. Restart quarantine is ~1s.
    fn test_lease_params() -> kivi_types::LeaseParams {
        kivi_types::LeaseParams {
            guard_duration: Duration::from_millis(500),
            lease_duration: Duration::from_millis(500),
            renew_interval: Duration::from_millis(50),
            max_drift_ppm: 1_000,
        }
    }

    /// Slow lease timing for failover-transition tests: 5s leases make
    /// exclusion windows (~5s) deterministically outlast elections
    /// (~1s), so refusal-then-success sequences never race the clock.
    /// Restart quarantine is ~5.5s.
    fn slow_lease_params() -> kivi_types::LeaseParams {
        kivi_types::LeaseParams {
            guard_duration: Duration::from_millis(500),
            lease_duration: Duration::from_millis(5_000),
            renew_interval: Duration::from_millis(50),
            max_drift_ppm: 1_000,
        }
    }

    /// Lease-eligible point reads (the common test shape).
    const ELIGIBLE: kivi_types::LeaseEligibility = kivi_types::LeaseEligibility::Eligible;

    fn identity(session: u128, seq: u64) -> MutationIdentity {
        MutationIdentity::new(
            RequestIdentity::new(SessionId::from_u128(session), RequestSeq::from_u64(seq)),
            RequestSeq::from_u64(0),
        )
    }

    fn block_on<F, T>(future: F) -> T
    where
        F: Future<Output = T>,
    {
        compio::runtime::Runtime::new()
            .expect("compio test runtime")
            .block_on(future)
    }

    /// Static 1×3 topology over three loopback peer addresses.
    fn test_topology(addrs: &[std::net::SocketAddr]) -> ClusterTopology {
        ClusterTopology {
            cluster: ClusterId::from_u128(CLUSTER),
            nodes: vec![
                NodeDescriptor {
                    node: NodeId::from_u64(1),
                    peer: addrs[0],
                    native: "127.0.0.1:9001".parse().expect("addr"),
                },
                NodeDescriptor {
                    node: NodeId::from_u64(2),
                    peer: addrs[1],
                    native: "127.0.0.1:9002".parse().expect("addr"),
                },
                NodeDescriptor {
                    node: NodeId::from_u64(3),
                    peer: addrs[2],
                    native: "127.0.0.1:9003".parse().expect("addr"),
                },
            ],
            tablets: vec![TabletAssignment {
                tablet: TABLET,
                replicas: vec![
                    NodeId::from_u64(1),
                    NodeId::from_u64(2),
                    NodeId::from_u64(3),
                ],
            }],
        }
    }

    /// Opens one test node: scratch directory and static identity. QUIC
    /// binds its UDP socket inside the owner thread, so no pre-bound
    /// listener crosses the boundary (insecure test TLS: loopback only).
    async fn open_test_node(
        dir: &std::path::Path,
        id: u64,
        topology: &ClusterTopology,
    ) -> ReplicatedNode {
        open_test_node_with_params(dir, id, topology, test_lease_params()).await
    }

    /// Opens one test node with explicit lease timing (failover tests
    /// need slower leases than the default fast ones).
    async fn open_test_node_with_params(
        dir: &std::path::Path,
        id: u64,
        topology: &ClusterTopology,
        lease_params: kivi_types::LeaseParams,
    ) -> ReplicatedNode {
        ReplicatedNode::open(NodeConfig {
            data_dir: dir.to_owned(),
            namespace: NS,
            tablet: TABLET,
            authority: authority(),
            local: NodeId::from_u64(id),
            topology: topology.clone(),
            segment_target_bytes: 1024 * 1024,
            transport: crate::transport::TransportConfig::default(),
            peer_certs: std::collections::HashMap::new(),
            insecure_peer_tls: true,
            preflight_enabled: true,
            lease_params,
        })
        .await
        .expect("node opens")
    }

    /// Probes one free loopback UDP port for the QUIC endpoint.
    fn probe_udp() -> std::net::SocketAddr {
        let socket = std::net::UdpSocket::bind("127.0.0.1:0").expect("probe binds");
        socket.local_addr().expect("probe addr")
    }

    /// One test cluster: three nodes on loopback with probed ports and
    /// scratch data directories. The topology (with the probed ports) is
    /// returned for restarts reusing the same addresses.
    async fn cluster() -> (Vec<ReplicatedNode>, Vec<tempfile::TempDir>, ClusterTopology) {
        slow_cluster_with_params(test_lease_params()).await
    }

    /// One test cluster with explicit lease timing.
    async fn slow_cluster_with_params(
        lease_params: kivi_types::LeaseParams,
    ) -> (Vec<ReplicatedNode>, Vec<tempfile::TempDir>, ClusterTopology) {
        let addrs = vec![probe_udp(), probe_udp(), probe_udp()];
        let topology = test_topology(&addrs);
        let mut nodes = Vec::new();
        let mut dirs = Vec::new();
        for id in [1u64, 2, 3] {
            let dir = tempfile::tempdir().expect("scratch");
            nodes.push(open_test_node_with_params(dir.path(), id, &topology, lease_params).await);
            dirs.push(dir);
        }
        (nodes, dirs, topology)
    }

    /// Restarts the node at `index` from its existing directory with the
    /// same topology (the bootstrap-existing path: no re-initialization).
    /// UDP has no `TIME_WAIT`, so the restarted node rebinds its topology
    /// port verbatim and the mesh heals through outbound dials.
    async fn restart(
        dirs: &[tempfile::TempDir],
        topology: &ClusterTopology,
        index: usize,
    ) -> ReplicatedNode {
        open_test_node(dirs[index].path(), index as u64 + 1, topology).await
    }

    /// Waits for exactly one agreed leader, then confirms it stays the
    /// leader across consecutive polls (a single agreeing poll can catch
    /// a transition mid-handover).
    async fn wait_leader(nodes: &[ReplicatedNode]) -> usize {
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        let mut stable = 0usize;
        let mut last = usize::MAX;
        loop {
            let mut leaders = Vec::new();
            for (index, node) in nodes.iter().enumerate() {
                if node.status().await.role == ReplicaRole::Leader {
                    leaders.push(index);
                }
            }
            if leaders.len() == 1 && leaders[0] == last {
                stable += 1;
                if stable >= 3 {
                    return leaders[0];
                }
            } else {
                stable = 0;
                last = leaders.first().copied().unwrap_or(usize::MAX);
            }
            assert!(
                std::time::Instant::now() < deadline,
                "cluster never elected exactly one stable leader"
            );
            compio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Waits until a leader other than `exclude` is stable.
    async fn wait_leader_except(nodes: &[ReplicatedNode], exclude: usize) -> usize {
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        let mut stable = 0usize;
        let mut last = usize::MAX;
        loop {
            let mut leaders = Vec::new();
            for (index, node) in nodes.iter().enumerate() {
                if index != exclude && node.status().await.role == ReplicaRole::Leader {
                    leaders.push(index);
                }
            }
            if leaders.len() == 1 && leaders[0] == last {
                stable += 1;
                if stable >= 3 {
                    return leaders[0];
                }
            } else {
                stable = 0;
                last = leaders.first().copied().unwrap_or(usize::MAX);
            }
            assert!(
                std::time::Instant::now() < deadline,
                "majority never elected without node {exclude}"
            );
            compio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Symmetrically partitions the leader from every follower (both
    /// directions suspended) and waits until every suspended link reads
    /// disconnected on both ends: proposals after this point cannot have
    /// flown before the partition closed (no phantom quorum from
    /// in-flight RPCs).
    async fn partition_leader(nodes: &[ReplicatedNode], leader: usize) {
        let leader_id = NodeId::from_u64(leader as u64 + 1);
        for (index, node) in nodes.iter().enumerate() {
            if index == leader {
                continue;
            }
            let peer = NodeId::from_u64(index as u64 + 1);
            nodes[leader].suspend_peer(peer).await;
            node.suspend_peer(leader_id).await;
        }
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let mut down = true;
            for (index, node) in nodes.iter().enumerate() {
                let status = node.status().await;
                for (peer, stats) in &status.peers {
                    // A leader link has exactly one end at the leader.
                    let mine = index == leader;
                    let theirs = peer.as_u64() == leader as u64 + 1;
                    if mine != theirs && (stats.connected_control || stats.connected_bulk) {
                        down = false;
                    }
                }
            }
            if down {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "partition never fully closed"
            );
            compio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Heals every partition in the cluster.
    async fn heal_all(nodes: &[ReplicatedNode]) {
        for node in nodes {
            for peer in [1u64, 2, 3] {
                node.resume_peer(NodeId::from_u64(peer)).await;
            }
        }
    }

    /// Waits until every node applied at least `index`.
    async fn wait_applied(nodes: &[ReplicatedNode], index: u64) {
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            let mut ready = true;
            for node in nodes {
                if node.status().await.applied.get() < index {
                    ready = false;
                }
            }
            if ready {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "cluster never applied index {index}"
            );
            compio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Opens a cluster, elects, writes `x=1` through the leader, and
    /// waits for full application. Returns nodes, directories, topology,
    /// leader index, and the write index.
    async fn elected_cluster() -> (
        Vec<ReplicatedNode>,
        Vec<tempfile::TempDir>,
        ClusterTopology,
        usize,
        u64,
    ) {
        elected_cluster_with_params(test_lease_params()).await
    }

    /// Opens a cluster with explicit lease timing, elects, writes `x=1`
    /// through the leader, and waits for full application.
    async fn elected_cluster_with_params(
        lease_params: kivi_types::LeaseParams,
    ) -> (
        Vec<ReplicatedNode>,
        Vec<tempfile::TempDir>,
        ClusterTopology,
        usize,
        u64,
    ) {
        let (nodes, dirs, topology) = slow_cluster_with_params(lease_params).await;
        let leader = wait_leader(&nodes).await;
        let outcome = nodes[leader]
            .propose(
                &Operation::Set {
                    key: Key::from("x"),
                    value: bytes::Bytes::from_static(b"1"),
                },
                None,
                None,
                NOW,
            )
            .await
            .expect("leader proposes");
        let index = match outcome {
            ProposeOutcome::Applied { index, .. } => index.get(),
            other => panic!("expected Applied, got {other:?}"),
        };
        wait_applied(&nodes, index).await;
        (nodes, dirs, topology, leader, index)
    }

    /// Leader writes and linearizable reads serve the committed value.
    #[test]
    fn leader_writes_and_reads_latest() {
        block_on(async {
            let (nodes, _dirs, _topology, leader, _index) = elected_cluster().await;
            let read = nodes[leader]
                .read(
                    &Operation::Get {
                        key: Key::from("x"),
                    },
                    ReadContract::Latest,
                    CTX,
                    ELIGIBLE,
                )
                .await
                .expect("latest reads");
            assert_eq!(
                read.outcome,
                OperationResult::Value(Some(bytes::Bytes::from_static(b"1")))
            );
            for node in &nodes {
                node.shutdown().await;
            }
        });
    }

    /// `AtLeast` serves from a lagging-but-covered follower without
    /// contacting the leader; foreign lineages reject as `StaleToken`
    /// without waiting; the strong cache replays `Any`.
    #[test]
    fn at_least_covers_follower_and_rejects_foreign_lineage() {
        use kivi_types::{CommitPosition, CommitToken, TabletEpoch, TabletId};

        block_on(async {
            let (nodes, _dirs, _topology, leader, _index) = elected_cluster().await;
            let follower = (leader + 1) % 3;
            // The leader's receipt vouches its applied position.
            let receipt = nodes[leader]
                .read(
                    &Operation::Get {
                        key: Key::from("x"),
                    },
                    ReadContract::Latest,
                    CTX,
                    ELIGIBLE,
                )
                .await
                .expect("leader reads")
                .receipt;
            // A covered follower serves without a barrier: local serve,
            // no authority contact.
            let served = nodes[follower]
                .read(
                    &Operation::Get {
                        key: Key::from("x"),
                    },
                    ReadContract::AtLeast(receipt.token()),
                    CTX,
                    ELIGIBLE,
                )
                .await
                .expect("covered follower serves");
            assert_eq!(
                served.outcome,
                OperationResult::Value(Some(bytes::Bytes::from_static(b"1")))
            );
            let metrics = nodes[follower].status().await.consistency;
            assert_eq!(metrics.reads_by_contract.1, 1, "one at-least read");
            assert_eq!(metrics.authority_serves, 0, "no barrier taken");
            assert_eq!(metrics.local_serves, 1, "served locally");
            // A token for another tablet never validates here.
            let foreign = CommitToken::new(
                TabletId::from_u64(4242),
                TabletEpoch::INITIAL,
                CommitPosition::FIRST,
            );
            let error = nodes[follower]
                .read(
                    &Operation::Get {
                        key: Key::from("x"),
                    },
                    ReadContract::AtLeast(foreign),
                    CTX,
                    ELIGIBLE,
                )
                .await
                .expect_err("foreign tablet rejects");
            assert!(
                matches!(error, ReadError::StaleToken { .. }),
                "wrong tablet is a stale token, got {error:?}"
            );
            // A token from a superseded epoch rejects too: split/merge
            // lineage never accidentally validates.
            let old_epoch =
                CommitToken::new(TABLET, TabletEpoch::from_u64(999), CommitPosition::FIRST);
            let error = nodes[follower]
                .read(
                    &Operation::Get {
                        key: Key::from("x"),
                    },
                    ReadContract::AtLeast(old_epoch),
                    CTX,
                    ELIGIBLE,
                )
                .await
                .expect_err("superseded epoch rejects");
            assert!(
                matches!(error, ReadError::StaleToken { .. }),
                "stale epoch is a stale token, got {error:?}"
            );
            let metrics = nodes[follower].status().await.consistency;
            assert_eq!(
                metrics.at_least_lineage_rejects, 2,
                "both foreign tokens counted"
            );
            // The strong cache replays `Any` on a second read of one key.
            let any = || Operation::Get {
                key: Key::from("y-absent"),
            };
            let first = nodes[follower]
                .read(&any(), ReadContract::Any, CTX, ELIGIBLE)
                .await
                .expect("first any serves");
            assert_eq!(first.outcome, OperationResult::Value(None));
            let second = nodes[follower]
                .read(&any(), ReadContract::Any, CTX, ELIGIBLE)
                .await
                .expect("second any serves");
            assert_eq!(second.outcome, OperationResult::Value(None));
            let metrics = nodes[follower].status().await.consistency;
            assert_eq!(metrics.strong_cache_hits, 1, "second any replays");
            for node in &nodes {
                node.shutdown().await;
            }
        });
    }

    /// `BoundedStale` without evidence escalates instead of serving weak
    /// data: on a follower the escalation barrier fails as `NotLeader`
    /// (a redirect at the server layer), never as a value. On the leader
    /// a prior `Latest` installs the receipt and the bound serves from
    /// proof.
    #[test]
    fn bounded_stale_escalates_without_proof_and_hits_with_it() {
        use core::time::Duration;

        block_on(async {
            let (nodes, _dirs, _topology, leader, _index) = elected_cluster().await;
            let follower = (leader + 1) % 3;
            let bound = ReadContract::BoundedStale {
                max_staleness: Duration::from_secs(60),
            };
            // Cold follower: no receipt, so escalation; the barrier then
            // fails as NotLeader. The bound is never served weakly.
            let error = nodes[follower]
                .read(
                    &Operation::Get {
                        key: Key::from("x"),
                    },
                    bound,
                    CTX,
                    ELIGIBLE,
                )
                .await
                .expect_err("cold follower cannot prove the bound");
            assert!(
                matches!(
                    error,
                    ReadError::Consensus(ConsensusError::NotLeader { .. })
                ),
                "escalation surfaces routing, never weak data: {error:?}"
            );
            let metrics = nodes[follower].status().await.consistency;
            assert_eq!(metrics.bounded_stale_escalations, 1);
            // The leader's `Latest` installs a receipt; the same bound
            // then serves from proof with no second barrier.
            nodes[leader]
                .read(
                    &Operation::Get {
                        key: Key::from("x"),
                    },
                    ReadContract::Latest,
                    CTX,
                    ELIGIBLE,
                )
                .await
                .expect("leader barriers");
            let served = nodes[leader]
                .read(
                    &Operation::Get {
                        key: Key::from("x"),
                    },
                    bound,
                    CTX,
                    ELIGIBLE,
                )
                .await
                .expect("proven bound serves");
            assert_eq!(
                served.outcome,
                OperationResult::Value(Some(bytes::Bytes::from_static(b"1")))
            );
            let metrics = nodes[leader].status().await.consistency;
            assert_eq!(metrics.bounded_stale_hits, 1, "proof hit counted");
            assert_eq!(
                metrics.authority_serves, 1,
                "only the Latest took a barrier"
            );
            for node in &nodes {
                node.shutdown().await;
            }
        });
    }

    /// Lazy-ALR serves `Latest` on leaders and followers without any
    /// barrier and without any timing assumption: one fence per batch,
    /// local execution after the boundary applies.
    #[test]
    fn lazy_alr_serves_without_barriers_or_clocks() {
        use kivi_types::ReadAuthorityProvider;

        block_on(async {
            let (nodes, _dirs, _topology, leader, _index) = elected_cluster().await;
            let follower = (leader + 1) % 3;
            for node in &nodes {
                node.set_read_provider(ReadAuthorityProvider::AlmostLocal);
            }
            let get = || Operation::Get {
                key: Key::from("x"),
            };
            // Leader: batch orders a fence, no barrier is taken.
            let served = nodes[leader]
                .read(&get(), ReadContract::Latest, CTX, ELIGIBLE)
                .await
                .expect("leader ALR serves");
            assert_eq!(
                served.outcome,
                OperationResult::Value(Some(bytes::Bytes::from_static(b"1")))
            );
            assert!(
                matches!(served.path, ServePath::AlrSync | ServePath::AlrSubsumed),
                "leader serves from its ALR batch, got {:?}",
                served.path
            );
            let metrics = nodes[leader].status().await.consistency;
            assert_eq!(metrics.alr_batches, 1);
            assert_eq!(metrics.authority_serves, 0, "no barrier on the ALR path");
            // Follower: forwards the fence order to the leader, executes
            // the batch locally once the boundary applies — still no
            // barrier, still exact.
            let served = nodes[follower]
                .read(&get(), ReadContract::Latest, CTX, ELIGIBLE)
                .await
                .expect("follower ALR serves");
            assert_eq!(
                served.outcome,
                OperationResult::Value(Some(bytes::Bytes::from_static(b"1")))
            );
            assert!(
                matches!(served.path, ServePath::AlrSync | ServePath::AlrSubsumed),
                "follower serves from its ALR batch, got {:?}",
                served.path
            );
            let metrics = nodes[follower].status().await.consistency;
            assert_eq!(metrics.authority_serves, 0, "follower takes no barrier");
            for node in &nodes {
                node.shutdown().await;
            }
        });
    }

    /// Roster leases serve `Latest` locally on responders; a leader
    /// transition voids old-roster authority (term fencing) so an old
    /// responder either sees the new write or fails over — never a stale
    /// success. Partitioning the old responder forces the failure leg.
    ///
    /// Slow lease timing (5s leases): exclusion windows deterministically
    /// outlast elections, so the refuse-then-succeed sequence never races
    /// the clock.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn roster_lease_serves_until_term_fencing_forces_fallback() {
        use kivi_types::ReadAuthorityProvider;

        block_on(async {
            let (nodes, _dirs, _topology, leader, _index) =
                elected_cluster_with_params(slow_lease_params()).await;
            for node in &nodes {
                node.set_read_provider(ReadAuthorityProvider::RosterLease);
            }
            let get = || Operation::Get {
                key: Key::from("x"),
            };
            // Wait for full-mesh activation, not just one responder:
            // every node must hold live exclusions to all three peers
            // (covered == 3 everywhere). Only then does killing the
            // leader deterministically leave a live exclusion covering
            // the dead node on the successor — the refusal below must
            // not race activation.
            let responder = (leader + 1) % 3;
            let mut served = None;
            for _ in 0..600 {
                let attempt = nodes[responder]
                    .read(&get(), ReadContract::Latest, CTX, ELIGIBLE)
                    .await
                    .expect("reads always answer while connected");
                if attempt.path == ServePath::RosterLease {
                    served = Some(attempt);
                }
                let mut meshed = served.is_some();
                for node in &nodes {
                    if node.consistency_snapshot().covered_responders != 3 {
                        meshed = false;
                    }
                }
                if meshed {
                    break;
                }
                compio::time::sleep(Duration::from_millis(50)).await;
            }
            let served = served.expect("roster stabilizes after quarantine");
            assert_eq!(
                served.outcome,
                OperationResult::Value(Some(bytes::Bytes::from_static(b"1")))
            );
            let metrics = nodes[responder].status().await.consistency;
            assert!(metrics.roster_hits >= 1, "served from the roster");
            assert_eq!(metrics.authority_serves, 0, "no barrier on the fast path");
            // Kill the leader: the successor's term voids the old roster.
            // Two protocol-correct outcomes exist here and the test must
            // accept either (never assume one timing):
            //
            // * Transition window observed: the dead leader's exclusion
            //   still covers the successor, so the write FAILS retryable
            //   instead of completing uncovered (revocation is impossible
            //   from a dead node, so the exclusion must lapse).
            // * Re-roster won the race: the successor already holds a new
            //   term roster with quorum coverage excluding the dead node,
            //   so the write SUCCEEDS covered. Refusing it here would be
            //   the bug, not the other way around.
            //
            // Either way the invariant under test holds downstream: an
            // old responder either sees the new write or fails over —
            // never a stale success. A bare `expect_err` here would turn
            // scheduler timing (grant tasks vs test thread) into a fake
            // protocol failure.
            nodes[leader].shutdown().await;
            let new_leader = wait_leader(&nodes).await;
            assert_ne!(new_leader, leader, "leadership moved on");
            let pre = nodes[new_leader].status().await.consistency;
            let outcome = nodes[new_leader]
                .propose(
                    &Operation::Set {
                        key: Key::from("x"),
                        value: bytes::Bytes::from_static(b"2"),
                    },
                    None,
                    None,
                    NOW,
                )
                .await;
            let post = nodes[new_leader].status().await.consistency;
            match outcome {
                Err(ProposeError::ResponderCoverage { .. }) => {}
                Ok(ProposeOutcome::Applied { outcome, .. }) => {
                    assert!(
                        matches!(outcome, OperationResult::Stored { .. }),
                        "covered fast-forward write commits, got {outcome:?}"
                    );
                    assert!(
                        post.covered_responders >= 2,
                        "fast-forward write was quorum-covered \
                         [pre: covered={} | post: covered={} roster={:?}]",
                        pre.covered_responders,
                        post.covered_responders,
                        post.roster,
                    );
                }
                other => panic!(
                    "transition write must refuse-covered or commit-covered, got {other:?} \
                     [pre: covered={} roster={:?} | post: covered={} roster={:?}]",
                    pre.covered_responders, pre.roster, post.covered_responders, post.roster,
                ),
            }
            // Past the exclusion lapse (slow timing: ~5s lease), the
            // gate clears and the write succeeds; the old responder's
            // next read either sees it (ALR over the live link) or fails
            // over — never stale.
            compio::time::sleep(Duration::from_millis(6_000)).await;
            nodes[new_leader]
                .propose(
                    &Operation::Set {
                        key: Key::from("x"),
                        value: bytes::Bytes::from_static(b"2"),
                    },
                    None,
                    None,
                    NOW,
                )
                .await
                .expect("write succeeds once the exclusion lapses");
            let outcome = nodes[responder]
                .read(&get(), ReadContract::Latest, CTX, ELIGIBLE)
                .await
                .expect("connected responder converges")
                .outcome;
            assert_eq!(
                outcome,
                OperationResult::Value(Some(bytes::Bytes::from_static(b"2"))),
                "old responder sees the post-failover write, never stale data"
            );
            // Partition a pure follower (neither old nor new leader) from
            // the new leader: while its lease is still valid it keeps
            // serving — the covered value, never stale data.
            let partitioned = (0..3)
                .find(|index| *index != new_leader && *index != leader)
                .expect("a pure follower exists");
            nodes[partitioned]
                .suspend_peer(nodes[new_leader].node())
                .await;
            let outcome = nodes[partitioned]
                .read(&get(), ReadContract::Latest, CTX, ELIGIBLE)
                .await
                .expect("valid lease still serves under partition")
                .outcome;
            assert_eq!(
                outcome,
                OperationResult::Value(Some(bytes::Bytes::from_static(b"2"))),
                "partitioned responder serves covered state, never stale"
            );
            // Past the hold lapse, with no leader leg (hence no stable
            // roster), no ALR forward, and no follower barrier, the read
            // fails instead of succeeding stale.
            compio::time::sleep(Duration::from_millis(6_000)).await;
            let error = nodes[partitioned]
                .read(&get(), ReadContract::Latest, CTX, ELIGIBLE)
                .await
                .expect_err("lapsed responder cannot prove freshness");
            assert!(
                matches!(error, ReadError::Consensus(_)),
                "partition fails over as routing, got {error:?}"
            );
            for node in &nodes {
                node.shutdown().await;
            }
        });
    }

    /// Restart clears memory-held evidence but preserves token meaning:
    /// after the leader restarts and the cluster re-elects, `BoundedStale`
    /// escalates (no resurrected receipt) while a pre-restart `AtLeast`
    /// token still covers.
    #[test]
    fn restart_drops_evidence_and_preserves_valid_tokens() {
        use core::time::Duration;

        block_on(async {
            let (mut nodes, dirs, topology, leader, _index) = elected_cluster().await;
            let token = nodes[leader]
                .read(
                    &Operation::Get {
                        key: Key::from("x"),
                    },
                    ReadContract::Latest,
                    CTX,
                    ELIGIBLE,
                )
                .await
                .expect("pre-restart read")
                .receipt
                .token();
            // Restart the leader in place; the cluster re-elects with the
            // rejoined replica (fresh hub, replayed durable state).
            nodes[leader].shutdown().await;
            drop(nodes.remove(leader));
            nodes.insert(leader, restart(&dirs, &topology, leader).await);
            let new_leader = wait_leader(&nodes).await;
            // No resurrected receipts anywhere: BoundedStale escalates to
            // a fresh barrier even though the data is present.
            let bound = ReadContract::BoundedStale {
                max_staleness: Duration::from_secs(60),
            };
            let served = nodes[new_leader]
                .read(
                    &Operation::Get {
                        key: Key::from("x"),
                    },
                    bound,
                    CTX,
                    ELIGIBLE,
                )
                .await
                .expect("escalation serves after restart");
            assert_eq!(
                served.outcome,
                OperationResult::Value(Some(bytes::Bytes::from_static(b"1")))
            );
            let metrics = nodes[new_leader].status().await.consistency;
            assert_eq!(
                metrics.bounded_stale_escalations, 1,
                "restart left no usable receipt"
            );
            // The pre-restart token names the same lineage and still
            // covers: restart preserves valid tokens, never silently
            // redefines them.
            let served = nodes[new_leader]
                .read(
                    &Operation::Get {
                        key: Key::from("x"),
                    },
                    ReadContract::AtLeast(token),
                    CTX,
                    ELIGIBLE,
                )
                .await
                .expect("pre-restart token survives restart");
            assert_eq!(
                served.outcome,
                OperationResult::Value(Some(bytes::Bytes::from_static(b"1")))
            );
            for node in &nodes {
                node.shutdown().await;
            }
        });
    }

    /// Same-identity retry returns the original outcome without
    /// re-executing (exactly-once across the replicated path).
    #[test]
    fn same_identity_retry_is_duplicate() {
        block_on(async {
            let (nodes, _dirs, _topology, leader, _index) = elected_cluster().await;
            let add = Operation::CounterAdd {
                key: Key::from("n"),
                delta: 1,
            };
            let first = nodes[leader]
                .propose(&add, Some(identity(0x5E55, 1)), None, NOW)
                .await
                .expect("counter applies");
            assert!(matches!(first, ProposeOutcome::Applied { .. }));
            let retry = nodes[leader]
                .propose(&add, Some(identity(0x5E55, 1)), None, NOW)
                .await
                .expect("retry answers");
            assert!(
                matches!(retry, ProposeOutcome::Duplicate { .. }),
                "same identity retries without re-executing, got {retry:?}"
            );
            let value = nodes[leader]
                .read(
                    &Operation::CounterGet {
                        key: Key::from("n"),
                    },
                    ReadContract::Latest,
                    CTX,
                    ELIGIBLE,
                )
                .await
                .expect("reads counter");
            assert_eq!(value.outcome, OperationResult::Counter(Some(1)));
            for node in &nodes {
                node.shutdown().await;
            }
        });
    }

    /// Followers hint the leader (never a phantom commit), refuse
    /// `Latest`, serve `Any`, and chunk sidecars fail explicitly.
    #[test]
    fn follower_hints_and_weak_reads() {
        block_on(async {
            let (nodes, _dirs, _topology, leader, _index) = elected_cluster().await;
            let follower = (leader + 1) % 3;
            // Follower mutations fail with a leader hint (never OpenRaft
            // types, never a phantom commit).
            let error = nodes[follower]
                .propose(
                    &Operation::Set {
                        key: Key::from("y"),
                        value: bytes::Bytes::from_static(b"2"),
                    },
                    None,
                    None,
                    NOW,
                )
                .await
                .expect_err("follower refuses writes");
            match error {
                ProposeError::Consensus(ConsensusError::NotLeader { hint }) => {
                    assert_eq!(
                        hint.leader.map(|leader| leader.node().as_u64()),
                        Some(leader as u64 + 1)
                    );
                }
                other => panic!("expected NotLeader with hint, got {other:?}"),
            }
            // Follower `Latest` refuses; `Any` serves applied state.
            let error = nodes[follower]
                .read(
                    &Operation::Get {
                        key: Key::from("x"),
                    },
                    ReadContract::Latest,
                    CTX,
                    ELIGIBLE,
                )
                .await
                .expect_err("follower refuses Latest");
            assert!(matches!(
                error,
                ReadError::Consensus(ConsensusError::NotLeader { .. })
            ));
            let weak = nodes[follower]
                .read(
                    &Operation::Get {
                        key: Key::from("x"),
                    },
                    ReadContract::Any,
                    CTX,
                    ELIGIBLE,
                )
                .await
                .expect("follower serves Any");
            assert_eq!(
                weak.outcome,
                OperationResult::Value(Some(bytes::Bytes::from_static(b"1")))
            );
            // Undurable chunked roots fail closed (never proposed): the
            // leader stages sidecars before proposing, so a fake manifest
            // is unavailable, never silently replicated.
            let error = nodes[leader]
                .propose(
                    &Operation::SetChunked {
                        key: Key::from("big"),
                        manifest: kivi_types::ManifestId::from_bytes([0x11; 32]),
                        logical_len: 1_000_000,
                    },
                    None,
                    None,
                    NOW,
                )
                .await
                .expect_err("undurable chunked root must not propose");
            assert!(
                matches!(
                    error,
                    ProposeError::Consensus(ConsensusError::Unavailable { .. })
                ),
                "fake manifest fails as unavailable, got {error:?}"
            );
            for node in &nodes {
                node.shutdown().await;
            }
        });
    }

    /// Restart reuses durable state (new incarnation, no re-bootstrap)
    /// and the restarted replica catches up and converges.
    #[test]
    fn restarted_replica_catches_up() {
        block_on(async {
            let (mut nodes, dirs, topology) = cluster().await;
            let leader = wait_leader(&nodes).await;
            for (session, value) in [0xA1u128, 0xA2, 0xA3].into_iter().zip([1i64, 2, 3]) {
                nodes[leader]
                    .propose(
                        &Operation::CounterAdd {
                            key: Key::from("n"),
                            delta: value,
                        },
                        Some(identity(session, 1)),
                        None,
                        NOW,
                    )
                    .await
                    .expect("writes apply");
            }
            let status = nodes[leader].status().await;
            wait_applied(&nodes, status.applied.get()).await;
            // Restart a follower from its data directory.
            let follower = (leader + 1) % 3;
            let incarnation = nodes[follower].incarnation();
            nodes[follower].shutdown().await;
            drop(nodes.remove(follower));
            let restarted = restart(&dirs, &topology, follower).await;
            assert!(
                restarted.incarnation().as_u64() > incarnation.as_u64(),
                "incarnation advances across restarts"
            );
            // Convergence without repair: the restarted replica applies the
            // retained tail up to the pre-restart applied watermark.
            let watermark = status.applied;
            let deadline = std::time::Instant::now() + Duration::from_secs(20);
            loop {
                if restarted.status().await.applied >= watermark {
                    break;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "restarted replica never converged"
                );
                compio::time::sleep(Duration::from_millis(50)).await;
            }
            let value = restarted
                .read(
                    &Operation::CounterGet {
                        key: Key::from("n"),
                    },
                    ReadContract::Any,
                    CTX,
                    ELIGIBLE,
                )
                .await
                .expect("converged replica reads");
            assert_eq!(value.outcome, OperationResult::Counter(Some(6)));
            restarted.shutdown().await;
            for node in &nodes {
                node.shutdown().await;
            }
        });
    }

    /// Minority fencing: an isolated node must never acknowledge a new
    /// strong mutation, while the majority elects and proceeds. Healing
    /// converges with no dual acknowledged history.
    ///
    /// The isolated proposal resolves in one of two correct ways: it
    /// stays pending while the node still considers itself leader without
    /// quorum, or it fails fast with a routing error once the node steps
    /// down or campaigns (0.10 elections move promptly). `Applied` is the
    /// only forbidden outcome — acknowledging without quorum.
    #[test]
    fn isolated_leader_cannot_acknowledge() {
        block_on(async {
            let (nodes, _dirs, _topology) = cluster().await;
            let leader = wait_leader(&nodes).await;
            nodes[leader]
                .propose(
                    &Operation::CounterAdd {
                        key: Key::from("n"),
                        delta: 1,
                    },
                    Some(identity(0xD0, 1)),
                    None,
                    NOW,
                )
                .await
                .expect("baseline applies");
            let base = nodes[leader].status().await.applied.get();
            wait_applied(&nodes, base).await;
            // `suspend_peer` returns only after the evicted tasks exited
            // and their sockets closed, so the partition is airtight for
            // the fencing verdict below (no dying-writer flush can still
            // complete quorum).
            partition_leader(&nodes, leader).await;
            // The majority elects without the isolated node.
            let majority = wait_leader_except(&nodes, leader).await;
            // The isolated node must not acknowledge a new strong
            // mutation: without quorum its commit can never complete.
            let verdict = compio::time::timeout(
                Duration::from_secs(5),
                nodes[leader].propose(
                    &Operation::CounterAdd {
                        key: Key::from("n"),
                        delta: 100,
                    },
                    Some(identity(0xD1, 1)),
                    None,
                    NOW,
                ),
            )
            .await;
            match verdict {
                // Still leader, still pending, or stepped down with a
                // prompt routing failure: no ack without quorum either way.
                Err(_) | Ok(Err(_)) => {}
                Ok(Ok(outcome)) => {
                    panic!("isolated node acknowledged without quorum: {outcome:?}");
                }
            }
            // The majority side proceeds.
            nodes[majority]
                .propose(
                    &Operation::CounterAdd {
                        key: Key::from("n"),
                        delta: 10,
                    },
                    Some(identity(0xD2, 1)),
                    None,
                    NOW,
                )
                .await
                .expect("majority writes");
            let read = nodes[majority]
                .read(
                    &Operation::CounterGet {
                        key: Key::from("n"),
                    },
                    ReadContract::Latest,
                    CTX,
                    ELIGIBLE,
                )
                .await
                .expect("majority reads");
            assert_eq!(read.outcome, OperationResult::Counter(Some(11)));
            // Heal: the isolated entry (never quorum-committed) vanishes and
            // every replica converges on the majority history — no dual
            // acknowledged history exists anywhere.
            heal_all(&nodes).await;
            wait_applied(&nodes, nodes[majority].status().await.applied.get()).await;
            for node in &nodes {
                let value = node
                    .read(
                        &Operation::CounterGet {
                            key: Key::from("n"),
                        },
                        ReadContract::Any,
                        CTX,
                        ELIGIBLE,
                    )
                    .await
                    .expect("converged read");
                assert_eq!(value.outcome, OperationResult::Counter(Some(11)));
            }
            for node in &nodes {
                node.shutdown().await;
            }
        });
    }

    /// Log conflict repair: a real divergent uncommitted suffix on the
    /// isolated leader is replaced by the authoritative log on heal,
    /// through the persisted `RaftTruncate` record — and the record
    /// survives a restart.
    #[test]
    fn conflicting_suffix_truncates_through_durable_record() {
        block_on(async {
            let (nodes, dirs, topology) = cluster().await;
            let leader = wait_leader(&nodes).await;
            nodes[leader]
                .propose(
                    &Operation::CounterAdd {
                        key: Key::from("n"),
                        delta: 1,
                    },
                    Some(identity(0xE0, 1)),
                    None,
                    NOW,
                )
                .await
                .expect("baseline applies");
            let base = nodes[leader].status().await.applied.get();
            wait_applied(&nodes, base).await;
            partition_leader(&nodes, leader).await;
            // The isolated leader appends an uncommitted suffix entry: it
            // lands in its local log (last_log advances) but can never
            // commit without quorum. The proposal goes out immediately —
            // well inside the election timeout — while the node still
            // considers itself leader; if it already stepped down or
            // campaigns, the scenario cannot start and the test reports
            // that instead of hanging.
            assert_eq!(
                nodes[leader].status().await.role,
                crate::types::ReplicaRole::Leader,
                "isolated node must still lead to seed divergence"
            );
            let divergent = Operation::CounterAdd {
                key: Key::from("n"),
                delta: 100,
            };
            let pending = nodes[leader].propose(&divergent, Some(identity(0xE1, 1)), None, NOW);
            let mut appended = false;
            let mut early: Option<String> = None;
            {
                futures::pin_mut!(pending);
                let deadline = std::time::Instant::now() + Duration::from_secs(10);
                loop {
                    if nodes[leader].status().await.last_log.get() > base {
                        appended = true;
                        break;
                    }
                    if let Some(settled) =
                        futures::future::FutureExt::now_or_never(pending.as_mut())
                    {
                        // Settled without appending (stepped down or
                        // campaigning mid-scenario): report instead of hanging.
                        early = Some(format!("{settled:?}"));
                        break;
                    }
                    if std::time::Instant::now() >= deadline {
                        break;
                    }
                    compio::time::sleep(Duration::from_millis(50)).await;
                }
            }
            assert!(
                appended,
                "suffix entry must append locally, never commit (early: {early:?})"
            );
            // The majority elects and commits a divergent entry.
            let majority = wait_leader_except(&nodes, leader).await;
            nodes[majority]
                .propose(
                    &Operation::CounterAdd {
                        key: Key::from("n"),
                        delta: 10,
                    },
                    Some(identity(0xE2, 1)),
                    None,
                    NOW,
                )
                .await
                .expect("majority commits divergently");
            // Heal: the authoritative log replaces the stale suffix and
            // every replica converges on the majority history.
            heal_all(&nodes).await;
            wait_applied(&nodes, nodes[majority].status().await.applied.get()).await;
            for node in &nodes {
                let value = node
                    .read(
                        &Operation::CounterGet {
                            key: Key::from("n"),
                        },
                        ReadContract::Any,
                        CTX,
                        ELIGIBLE,
                    )
                    .await
                    .expect("converged read");
                assert_eq!(
                    value.outcome,
                    OperationResult::Counter(Some(11)),
                    "divergent +100 must not survive anywhere"
                );
            }
            // The replacement went through the persisted Kivi truncate
            // record — not a memory-only path.
            assert!(
                nodes[leader].store.cached_truncate_count().await >= 1,
                "stale suffix replacement must record a durable truncate"
            );
            // The record survives a restart: reopening replays the marker
            // and converges identically.
            nodes[leader].shutdown().await;
            drop(nodes);
            let reopened = restart(&dirs, &topology, leader).await;
            assert!(
                reopened.store.cached_truncate_count().await >= 1,
                "truncate record must survive restart"
            );
            reopened.shutdown().await;
        });
    }

    /// Medium values replicate as chunked sidecars, never inline bytes,
    /// never fabric ids: a 2 KiB plain `Set` restages to `SetChunked` with
    /// a deterministic manifest, every replica stores a chunked root (no
    /// 2 KiB inline `Bytes`, no `Fabric` in the snapshot), sidecars resolve
    /// exact bytes on every replica, and the value survives leader death
    /// plus a follower restart through the sidecar fetch path.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn medium_set_replicates_as_chunked_sidecar() {
        use kivi_state::LogicalValue;

        // Deterministic 2 KiB value (chunk-boundary agnostic fill).
        fn medium_value() -> Vec<u8> {
            (0..2048u32)
                .map(|i| u8::try_from(i % 251).expect("remainder fits in u8"))
                .collect()
        }

        block_on(async {
            let (mut nodes, dirs, topology) = cluster().await;
            let leader = wait_leader(&nodes).await;
            let value = medium_value();
            assert_eq!(value.len(), 2048);
            let key = Key::from("medium");
            let op = Operation::Set {
                key: key.clone(),
                value: bytes::Bytes::from(value.clone()),
            };
            let outcome = nodes[leader]
                .propose(&op, None, None, NOW)
                .await
                .expect("medium Set proposes");
            let index = match outcome {
                ProposeOutcome::Applied { index, .. } => index.get(),
                other => panic!("medium Set must apply, got {other:?}"),
            };
            wait_applied(&nodes, index).await;
            // Every replica stores a chunked root, never inline bytes or a
            // worker-local fabric id. `Any` serves local applied state
            // without a barrier (already converged above).
            let get = Operation::Get { key: key.clone() };
            let mut manifests = Vec::new();
            for node in &nodes {
                let served = node
                    .read(&get, ReadContract::Any, CTX, ELIGIBLE)
                    .await
                    .expect("follower serves Any")
                    .outcome;
                match served {
                    OperationResult::ChunkedValue {
                        manifest,
                        logical_len,
                    } => {
                        assert_eq!(logical_len, 2048, "chunked root names 2 KiB");
                        manifests.push(manifest);
                        // Sidecar fetch path: every replica (leader and both
                        // followers) reconstructs the exact bytes.
                        let resolved = node
                            .sidecar()
                            .read_value(manifest, logical_len)
                            .await
                            .expect("sidecar resolves");
                        assert_eq!(resolved, value, "sidecar bytes exact");
                    }
                    OperationResult::Value(_) | OperationResult::FabricValue { .. } => {
                        panic!("medium must store as Chunked, got {served:?}");
                    }
                    other => panic!("Get of a medium must be chunked, got {other:?}"),
                }
            }
            assert_eq!(manifests.len(), 3);
            assert_eq!(manifests[0], manifests[1], "manifest replicates");
            assert_eq!(manifests[1], manifests[2], "manifest replicates");
            // Deterministic manifest: the same bytes proposed again (new key)
            // stage to the same manifest instead of fresh sidecars.
            let key_b = Key::from("medium-b");
            nodes[leader]
                .propose(
                    &Operation::Set {
                        key: key_b.clone(),
                        value: bytes::Bytes::from(value.clone()),
                    },
                    None,
                    None,
                    NOW,
                )
                .await
                .expect("second medium proposes");
            let second_index = nodes[leader].status().await.applied.get();
            wait_applied(&nodes, second_index).await;
            let served_b = nodes[leader]
                .read(
                    &Operation::Get { key: key_b },
                    ReadContract::Any,
                    CTX,
                    ELIGIBLE,
                )
                .await
                .expect("second medium reads")
                .outcome;
            match served_b {
                OperationResult::ChunkedValue { manifest, .. } => {
                    assert_eq!(manifest, manifests[0], "same bytes stage same manifest");
                }
                other => panic!("second medium must be chunked, got {other:?}"),
            }
            // Snapshot decode: the durable image holds a chunked root, no
            // 2 KiB inline `Bytes`, and no `Fabric` anywhere.
            let base = nodes[leader]
                .snapshot_and_purge()
                .await
                .expect("snapshot seals");
            assert!(base.get() >= index, "purge covers the medium write");
            let snapshots_dir = dirs[leader]
                .path()
                .join("consensus-sm")
                .join(TABLET.as_u64().to_string())
                .join("snapshots");
            let name =
                std::fs::read_to_string(snapshots_dir.join("CURRENT")).expect("CURRENT readable");
            let image =
                std::fs::read(snapshots_dir.join(name.trim())).expect("snapshot image readable");
            let decoded =
                crate::state_machine::ReplicatedTablet::decode_snapshot(&image, NS, TABLET)
                    .expect("snapshot decodes");
            let mut found_chunked = false;
            for (stored_key, object) in &decoded.objects {
                match object.value() {
                    LogicalValue::Bytes(inline) => {
                        assert_ne!(inline.len(), 2048, "no inline 2 KiB bytes in the snapshot");
                    }
                    LogicalValue::Fabric(fabric) => {
                        panic!("no fabric ids in the snapshot, got {fabric:?}");
                    }
                    LogicalValue::Chunked(chunked) if stored_key.as_bytes() == b"medium" => {
                        assert_eq!(chunked.logical_len, 2048);
                        assert_eq!(chunked.manifest, manifests[0]);
                        found_chunked = true;
                    }
                    _ => {}
                }
            }
            assert!(found_chunked, "medium stored as a chunked root");
            // Kill the leader: the survivors serve the exact bytes from
            // their durable sidecars without the dead leader.
            nodes[leader].shutdown().await;
            let elected = wait_leader(&nodes).await;
            assert_ne!(elected, leader, "a new leader emerges");
            let served = nodes[elected]
                .read(&get, ReadContract::Any, CTX, ELIGIBLE)
                .await
                .expect("survivor serves")
                .outcome;
            match served {
                OperationResult::ChunkedValue {
                    manifest,
                    logical_len,
                } => {
                    assert_eq!(manifest, manifests[0]);
                    let resolved = nodes[elected]
                        .sidecar()
                        .read_value(manifest, logical_len)
                        .await
                        .expect("survivor sidecar resolves");
                    assert_eq!(resolved, value);
                }
                other => panic!("survivor must serve chunked, got {other:?}"),
            }
            // Restart the dead leader from its directory: it catches up
            // (snapshot + tail) and its sidecars resolve the same bytes.
            drop(nodes.remove(leader));
            nodes.insert(leader, restart(&dirs, &topology, leader).await);
            wait_applied(&nodes, second_index).await;
            let served = nodes[leader]
                .read(&get, ReadContract::Any, CTX, ELIGIBLE)
                .await
                .expect("restarted replica serves")
                .outcome;
            match served {
                OperationResult::ChunkedValue {
                    manifest,
                    logical_len,
                } => {
                    assert_eq!(manifest, manifests[0]);
                    let resolved = nodes[leader]
                        .sidecar()
                        .read_value(manifest, logical_len)
                        .await
                        .expect("restarted sidecar resolves");
                    assert_eq!(resolved, value);
                }
                other => panic!("restarted replica must serve chunked, got {other:?}"),
            }
            for node in &nodes {
                node.shutdown().await;
            }
        });
    }

    /// Snapshot seals, purge advances the logical base through the
    /// durable record, and the base survives a restart.
    #[test]
    fn snapshot_purge_and_restart() {
        block_on(async {
            let (nodes, dirs, topology) = cluster().await;
            let leader = wait_leader(&nodes).await;
            for (session, value) in [0xB1u128, 0xB2, 0xB3, 0xB4, 0xB5]
                .into_iter()
                .zip([1i64, 2, 3, 4, 5])
            {
                nodes[leader]
                    .propose(
                        &Operation::CounterAdd {
                            key: Key::from("n"),
                            delta: value,
                        },
                        Some(identity(session, 1)),
                        None,
                        NOW,
                    )
                    .await
                    .expect("writes apply");
            }
            // Purge trails replication: wait for every replica to apply the
            // writes first.
            let last = nodes[leader].status().await.last_log.get();
            wait_applied(&nodes, last).await;
            let base = nodes[leader]
                .snapshot_and_purge()
                .await
                .expect("snapshot and purge");
            assert!(base.get() >= 5, "purge covers the writes");
            // The purge base is visible in diagnostics and the tail still
            // serves: one more write commits past it.
            let status = nodes[leader].status().await;
            assert_eq!(status.purged, Some(base));
            nodes[leader]
                .propose(
                    &Operation::CounterAdd {
                        key: Key::from("n"),
                        delta: 100,
                    },
                    Some(identity(0xC0, 1)),
                    None,
                    NOW,
                )
                .await
                .expect("post-purge write applies");
            // Restart the cluster from its directories: the purge record
            // replays, the snapshot base holds, and new writes succeed.
            // Every node shuts down first (dropping a live node would leak
            // its owner thread and its bound peer port).
            let incarnation = nodes[leader].incarnation();
            for node in &nodes {
                node.shutdown().await;
            }
            drop(nodes);
            let mut fresh = Vec::new();
            for (index, _dir) in dirs.iter().enumerate() {
                fresh.push(restart(&dirs, &topology, index).await);
            }
            assert!(
                fresh[leader].incarnation().as_u64() > incarnation.as_u64(),
                "leader incarnation advances"
            );
            let elected = wait_leader(&fresh).await;
            fresh[elected]
                .propose(
                    &Operation::CounterAdd {
                        key: Key::from("n"),
                        delta: 1000,
                    },
                    Some(identity(0xC1, 1)),
                    None,
                    NOW,
                )
                .await
                .expect("post-restart write applies");
            let value = fresh[elected]
                .read(
                    &Operation::CounterGet {
                        key: Key::from("n"),
                    },
                    ReadContract::Latest,
                    CTX,
                    ELIGIBLE,
                )
                .await
                .expect("post-restart latest reads");
            // 1+2+3+4+5+100+1000, exactly once each.
            assert_eq!(value.outcome, OperationResult::Counter(Some(1115)));
            for node in &fresh {
                node.shutdown().await;
            }
        });
    }
}
