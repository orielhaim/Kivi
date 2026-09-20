//! Kivi-backed `OpenRaft` state machine (stage M).
//!
//! Committed entries flow through the existing deterministic state path —
//! no second database implementation exists:
//!
//! ```text
//! OpenRaft committed entry
//!         ↓
//! ReplicatedMutation (canonical Kivi bytes)
//!         ↓
//! MutationEnvelope → Mutation IR (existing types)
//!         ↓
//! ObjectStore::apply + outcome_for (existing deterministic apply)
//!         ↓
//! ReplicatedOutcome (installed + replicated dedup)
//! ```
//!
//! ## `CommitPosition` decision (task D: Option 1)
//!
//! For replicated strong tablets there is exactly one order counter: the
//! Raft log index. [`CommitPosition`] is the
//! applied Raft log position through a typed `+1` offset:
//!
//! ```text
//! CommitPosition = raft_index + 1
//! ```
//!
//! The offset exists because `OpenRaft` numbers its first entry `0`
//! while [`CommitPosition::UNASSIGNED`](kivi_types::CommitPosition::UNASSIGNED)
//! (`0`) means "no log slot". The mapping is total, monotonic, and
//! invertible ([`commit_of_index`] / [`index_of_commit`]); the tablet
//! commit chain retires for the replicated tablet and the Raft log is the
//! authoritative mutation history (no double-logging).
//!
//! ## Applied tracking (task B)
//!
//! The machine distinguishes `last_log` (log store), `committed` (log
//! store pointer), and `applied` (here). Runtime `applied` lives in memory
//! and advances per batch; *durable* applied state is the installed
//! snapshot plus a crash-safe applied file, both written only on
//! snapshot build/install. Recovery therefore never assumes
//! `persisted == applied`: restart loads the snapshot base, reports it as
//! applied, and `OpenRaft` replays committed-but-not-applied entries from
//! the retained log through [`apply`](ReplicatedStateMachine::apply)
//! exactly once. Replay is safe because apply is deterministic and dedup
//! installs are idempotent (same identity overwrites the identical
//! entry). Per-batch object fsync would reintroduce a second durable
//! history; the Raft log already is that history, so snapshots stay the
//! sole object durability.
//!
//! ## Replicated dedup (task C)
//!
//! [`SessionId`] / [`RequestSeq`]
//! floors and retained [`DurableOutcome`]s are
//! tablet state, snapshotted with objects. The ack floor rides in the
//! replicated command ([`ReplicatedMutation::ack_floor`](crate::mutation::ReplicatedMutation::ack_floor))
//! so floor advances replicate in log order and every replica retains the
//! identical outcome set. Same identity applied on every replica → same
//! logical mutation → same outcome; a same-identity retry (including on a
//! new leader after failover) returns the original outcome without
//! re-executing.
//!
//! Only mutating `Write` proposals replicate. Terminal rejections
//! (overflow, wrong-type) are side-effect-free pure functions of converged
//! state: the leader answers them locally without proposing, and a retry
//! re-evaluates deterministically. They consume no log slot anywhere, so no
//! replica diverges.
//!
//! ## Deterministic verification (task E)
//!
//! Every applied command recomputes its outcome and compares it against the
//! proposal's `expected`. A mismatch is state-machine divergence or
//! corruption — never an ordinary client error — and surfaces as a storage
//! `IO` error (which halts the `OpenRaft` node) plus a latched unhealthy
//! flag the admin plane reports. The same holds for authority/namespace
//! mismatches, expired identities at apply time (the leader's pre-proposal
//! check admitted them, so reaching apply means divergence), and
//! unresolvable splices (leader admission restages chunked bases first).
//!
//! ## Snapshot substrate
//!
//! The snapshot is the Kivi checkpoint state in canonical bytes: objects,
//! sessions, applied pointer, and membership — the same logical concepts
//! `LiveTablet` checkpoints, encoded here
//! because this crate must not depend on the engine. Small single-tablet
//! state is sufficient for this stage; chunk-aware differential transfer
//! belongs to the next stage (chunked roots snapshot as manifest
//! references; bulk bytes stay in chunk packs).
//!
//! Snapshot metadata binds the applied [`LogId`](openraft::LogId) and the
//! membership `OpenRaft` requires — and nothing else. There is no
//! snapshot transfer id in logical metadata (0.10 removed it upstream;
//! Kivi's checkpoint identity is the sealed image name plus its bytes,
//! while transfer-session identity rides only the network fragment
//! envelope). Install validates everything, swaps state only after the
//! snapshot file *and* the applied file are durable, and never marks
//! installation complete before that.
//!
//! ## Purge coordination
//!
//! The machine never purges the log itself. The node layer may purge only
//! through [`snapshotted_index`](ReplicatedStateMachine::snapshotted_index)
//! (the installed snapshot base), so the logical [`RaftPurge`](kivi_durability::RaftPurge)
//! record always grounds in durable snapshot state across
//! snapshot/purge/restart.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use kivi_codec::{Decode, Encode, decode_byte_vec, encode_bytes};
use kivi_state::{
    DurableOutcome, Key, LogicalValue, Mutation, ObjectStore, ObjectVersion, Operation,
    OperationResult, StoredObject, outcome_for,
};
use kivi_types::{
    CommitPosition, Expiry, NamespaceId, RequestSeq, SessionId, TabletAuthority, TabletId,
    UnixMicros,
};
use openraft::storage::{RaftSnapshotBuilder, RaftStateMachine, SnapshotMeta};
use openraft::type_config::alias::{
    EntryPayloadOf, LogIdOf, SnapshotMetaOf, SnapshotOf, StoredMembershipOf,
};
use openraft::vote::RaftLeaderId as _;
use openraft::{BasicNode, EntryPayload, Membership};

use crate::command::ConsensusCommand;
use crate::config::KiviTypeConfig;
use crate::mutation::{ReplicatedMutation, ReplicatedOutcome};

/// Maximum retained unacknowledged outcomes per session on one replica.
/// Mirrors the single-node engine's session cap so both paths backstop
/// identically; past this, apply fails loudly (divergence) rather than
/// silently evicting an outcome a retry might need.
pub const SESSION_OUTCOME_CAP: usize = 1024;

/// Reserved identity for requests without a retry contract (embedded
/// callers sharing fate with the process): session `0`, sequence `0`.
/// Anonymous commands apply, verify, and advance `applied` exactly like
/// identified ones but install no dedup entry — and every later anonymous
/// command applies fresh rather than colliding on a shared fake identity.
/// Client adapters must never issue session `0`.
pub const ANONYMOUS_SESSION: u128 = 0;

/// Snapshot framing magic (`KVSM`: Kivi state machine), little-endian.
const SNAPSHOT_MAGIC: u32 = 0x4D53_564B;
/// Control snapshot framing magic (`KVSC`: Kivi control), little-endian.
const CONTROL_SNAPSHOT_MAGIC: u32 = 0x4353_564B;
/// Control snapshot framing version.
const CONTROL_SNAPSHOT_VERSION: u16 = 1;
/// Applied-file framing magic (`KVSA`: Kivi state-machine applied).
const APPLIED_MAGIC: u32 = 0x4153_564B;
/// Snapshot framing version (v2 binds prepared intents to their write-set
/// digests; v1 images are rejected loudly, never migrated).
const SNAPSHOT_VERSION: u16 = 2;
/// Applied-file framing version.
const APPLIED_VERSION: u16 = 1;

/// Logical-value tags for the snapshot body (snapshot-scoped, documented
/// here; never reused once assigned).
const VALUE_BYTES: u8 = 1;
/// Chunked-root tag: manifest reference only, bulk bytes stay in packs.
const VALUE_CHUNKED: u8 = 2;
/// Strict-counter tag.
const VALUE_COUNTER: u8 = 3;
/// Commutative-counter tag (order-free sum; never an ordinal).
const VALUE_COMMUTATIVE: u8 = 4;
/// Bounded-counter tag (value plus explicit escrow share).
const VALUE_BOUNDED: u8 = 5;
/// Semaphore tag (capacity plus live permits).
const VALUE_SEMAPHORE: u8 = 6;
/// Lease tag (holder plus monotonic fencing).
const VALUE_LEASE: u8 = 7;
/// Stream-shard tag (cursor plus retained entries).
const VALUE_STREAM_SHARD: u8 = 8;
/// Fabric-root tag: fabric reference only (id, length, version), never
/// bulk bytes. Fabric ids are worker-local names: a snapshot carrying
/// one only restores on a worker whose fabric staged it (same-process
/// restart via the journal); cross-node installs must resolve foreign
/// ids through sidecar bytes first, and serve paths fail closed on
/// unresolvable references instead of serving absence.
const VALUE_FABRIC: u8 = 9;

/// Converts an applied Raft log index into its [`CommitPosition`]
/// (task D, Option 1: `CommitPosition = raft_index + 1`; the `+1` preserves
/// the `UNASSIGNED` sentinel because `OpenRaft` 0.9 numbers entries from 0).
#[must_use]
pub const fn commit_of_index(index: u64) -> CommitPosition {
    CommitPosition::from_u64(index.saturating_add(1))
}

/// Inverts [`commit_of_index`]: the Raft log index a commit position names.
/// `UNASSIGNED` maps to `None` (it names no log slot).
#[must_use]
pub const fn index_of_commit(commit: CommitPosition) -> Option<u64> {
    if commit.as_u64() == 0 {
        None
    } else {
        Some(commit.as_u64() - 1)
    }
}

/// Durable applied pointer: the last log id the state machine applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppliedPointer {
    /// Term of the last applied entry.
    pub term: u64,
    /// Leader of the last applied entry.
    pub leader: u64,
    /// Index of the last applied entry.
    pub index: u64,
}

impl AppliedPointer {
    /// Builds a pointer from its parts.
    #[must_use]
    pub const fn new(term: u64, leader: u64, index: u64) -> Self {
        Self {
            term,
            leader,
            index,
        }
    }

    /// Converts an `OpenRaft` log id into its owned form. The 0.10
    /// `leader_id_adv` leader always names its node directly (total
    /// order), so no `Option` unwraps remain on this path.
    #[must_use]
    pub fn of_log_id(id: LogIdOf<KiviTypeConfig>) -> Self {
        Self {
            term: id.leader_id.term,
            leader: id.leader_id.node_id,
            index: id.index,
        }
    }

    /// Converts back into an `OpenRaft` log id.
    #[must_use]
    pub fn openraft(self) -> LogIdOf<KiviTypeConfig> {
        use openraft::impls::leader_id_adv::LeaderId;
        openraft::LogId::new(LeaderId::new(self.term, self.leader), self.index)
    }
}

/// One retained dedup outcome: enough to answer a retried identity
/// byte-for-byte. Mirrors the engine's dedup entry through the
/// [`commit_of_index`] mapping so both paths shape identical replies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetainedOutcome {
    /// Commit position that installed this outcome (Raft index `+1`).
    pub commit: CommitPosition,
    /// Originating opcode discriminant (response shaping).
    pub opcode: u8,
    /// The recorded outcome.
    pub outcome: DurableOutcome,
}

/// Exported topology copy: live objects plus all dedup sessions
/// `(session, floor, outcomes-by-seq)`. Objects are cheap clones;
/// sessions copy wholesale to every child for exactly-once retries.
pub type TopologyCopy = (
    Vec<(kivi_state::Key, kivi_state::StoredObject)>,
    Vec<(SessionId, RequestSeq, BTreeMap<RequestSeq, RetainedOutcome>)>,
);

/// One session's replicated dedup state: the durable floor plus retained
/// outcomes above it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionState {
    floor: RequestSeq,
    outcomes: BTreeMap<RequestSeq, RetainedOutcome>,
}

impl Default for SessionState {
    fn default() -> Self {
        Self {
            floor: RequestSeq::from_u64(0),
            outcomes: BTreeMap::new(),
        }
    }
}

impl SessionState {
    /// Advances the floor over acknowledged sequences, dropping outcomes
    /// the client will never retry.
    pub fn advance_floor(&mut self, ack: RequestSeq) {
        if ack.as_u64() > self.floor.as_u64() {
            self.floor = ack;
            self.outcomes.retain(|seq, _| seq.as_u64() > ack.as_u64());
        }
    }

    /// Returns the acknowledgement floor.
    #[must_use]
    pub const fn floor(&self) -> RequestSeq {
        self.floor
    }

    /// Iterates retained outcomes in sequence order.
    pub fn outcomes(&self) -> impl Iterator<Item = (&RequestSeq, &RetainedOutcome)> {
        self.outcomes.iter()
    }

    /// Rebuilds a session from exported parts (split/merge topology copy).
    #[must_use]
    pub fn from_parts(floor: RequestSeq, outcomes: BTreeMap<RequestSeq, RetainedOutcome>) -> Self {
        Self { floor, outcomes }
    }
}

/// Representation class of one key's live base: the admission-peek the
/// node capability gate and the engine planner share (chunked bases need
/// lane restaging before any byte patch may replicate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeBase {
    /// No live object (absent or expired).
    Absent,
    /// Resident byte string (safe to splice inline).
    Inline,
    /// Non-bytes value (counters and semantic objects: range patches
    /// reject deterministically).
    NonBytes,
    /// Chunked root (needs restaging; unreplicated in this stage).
    Chunked,
}

/// Read-only leader-side proposal decision: the dedup verdict before
/// anything replicates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProposalGate {
    /// Fresh identity: prepare and propose normally.
    Admit,
    /// Identity already completed: reply with the stored outcome.
    Hit {
        /// The recorded outcome.
        outcome: OperationResult,
    },
    /// Identity at or below the floor: gone, never re-executable.
    Expired,
    /// Session window exhausted: reject before replicating anything.
    Overloaded,
}

/// Why state-machine application failed. Every variant is divergence,
/// corruption, or I/O — never an ordinary client error (task E).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum StateMachineFault {
    /// Re-application produced a different outcome than the proposal
    /// carried: replicas diverged or history is corrupt.
    #[error(
        "deterministic apply diverged at index {index}: expected {expected:?}, recomputed {found:?}"
    )]
    Divergence {
        /// Raft log index of the divergent entry.
        index: u64,
        /// Outcome the proposal carried.
        expected: OperationResult,
        /// Outcome deterministic apply recomputed.
        found: OperationResult,
    },
    /// A dedup hit reproduced a different outcome than the proposal
    /// carried (same divergence class through the dedup path).
    #[error("dedup hit diverged at index {index}: retained {retained:?}, proposed {expected:?}")]
    DedupDivergence {
        /// Raft log index of the entry.
        index: u64,
        /// Outcome retained for the identity.
        retained: OperationResult,
        /// Outcome the proposal carried.
        expected: OperationResult,
    },
    /// The committed envelope names a different namespace or tablet.
    #[error(
        "committed envelope misrouted: envelope ({namespace}, tablet {tablet}), replica ({expected_namespace}, tablet {expected_tablet})"
    )]
    Misrouted {
        /// Envelope namespace.
        namespace: u64,
        /// Envelope tablet.
        tablet: u64,
        /// Replica namespace.
        expected_namespace: u64,
        /// Replica tablet.
        expected_tablet: u64,
    },
    /// The committed envelope's authority is stale or foreign.
    #[error("committed envelope authority mismatch: {detail}")]
    AuthorityMismatch {
        /// Human-readable mismatch detail.
        detail: String,
    },
    /// The identity arrived at or below the replicated floor: the
    /// leader's pre-proposal check admitted it, so reaching apply means
    /// the replicas diverged on floor state.
    #[error("committed identity {session}:{seq} is expired below floor {floor} at index {index}")]
    ExpiredIdentity {
        /// Session half.
        session: u128,
        /// Sequence half.
        seq: u64,
        /// Floor half.
        floor: u64,
        /// Entry index.
        index: u64,
    },
    /// The session holds too many unacknowledged outcomes: the leader's
    /// pre-proposal check admitted it, so reaching apply means divergence.
    #[error("session {session} outcome window exhausted at index {index}")]
    SessionOverloaded {
        /// Session half.
        session: u128,
        /// Entry index.
        index: u64,
    },
    /// Deterministic apply itself failed (only reachable on divergent
    /// state: same log on same state never produces it).
    #[error("deterministic apply failed at index {index}: {detail}")]
    ApplyFailed {
        /// Entry index.
        index: u64,
        /// Failure detail.
        detail: String,
    },
    /// A snapshot or applied-file image failed structural validation.
    #[error("snapshot image corrupt: {detail}")]
    CorruptImage {
        /// Human-readable detail.
        detail: String,
    },
    /// Durable tablet state violates an invariant the state machine
    /// itself maintains (e.g. a non-`Completed` outcome retained where
    /// only `Completed` outcomes are ever installed): corruption or a
    /// deterministic-apply bug, never a client error.
    #[error("replicated tablet state corrupt: {detail}")]
    CorruptState {
        /// Human-readable detail.
        detail: String,
    },
    /// Durable I/O for applied/snapshot state failed.
    #[error("state-machine I/O failed: {detail}")]
    Io {
        /// Human-readable detail.
        detail: String,
    },
    /// The local-read path was asked to serve a mutating operation.
    #[error("local read path cannot serve a mutating operation")]
    NotARead,
    /// A control-plane mutation was rejected at apply time (illegal
    /// lifecycle transition, unknown node/plan, or stale fenced
    /// generation). Stale generations are safe rejections, never
    /// corruption — but they still fail the entry loudly so the
    /// divergence cannot commit silently.
    #[error("control mutation rejected at index {index}: {detail}")]
    ControlRejected {
        /// Raft log index of the rejected entry.
        index: u64,
        /// Human-readable cause.
        detail: String,
    },
    /// A command reached the wrong group kind (tablet command on the
    /// control group, or control mutation on a tablet group).
    #[error("command misrouted to wrong group kind: {detail}")]
    WrongGroup {
        /// Human-readable cause.
        detail: String,
    },
}

/// Whether the mutation originated from a persist-expiry request,
/// derived deterministically from the mutation bytes (only
/// `PersistExpiry` produces `SetExpiry{NEVER}`).
fn is_persist_expiry(mutation: &Mutation) -> bool {
    matches!(mutation, Mutation::SetExpiry { expiry, .. } if *expiry == Expiry::NEVER)
}

/// The deterministic tablet core: [`ObjectStore`] plus replicated dedup,
/// applied pointer, and membership. Synchronous with no I/O: the async
/// wrapper owns durability (applied file, snapshot files).
#[derive(Debug)]
pub struct ReplicatedTablet {
    namespace: NamespaceId,
    tablet: TabletId,
    authority: TabletAuthority,
    store: ObjectStore,
    sessions: HashMap<SessionId, SessionState>,
    applied: Option<AppliedPointer>,
    membership: StoredMembershipOf<KiviTypeConfig>,
    /// Replicated control image. Meaningful only on the system control
    /// group ([`is_control`](Self::is_control)); tablet groups leave it
    /// empty forever.
    control: kivi_control::ControlState,
}

impl ReplicatedTablet {
    /// Builds an empty replica for `tablet` under `authority`.
    #[must_use]
    pub fn new(namespace: NamespaceId, tablet: TabletId, authority: TabletAuthority) -> Self {
        Self {
            namespace,
            tablet,
            authority,
            store: ObjectStore::new(),
            sessions: HashMap::new(),
            applied: None,
            membership: openraft::StoredMembership::default(),
            control: kivi_control::ControlState::empty(),
        }
    }

    /// Whether this is the system control group (applies control-plane
    /// mutations instead of tablet writes).
    #[must_use]
    pub fn is_control(&self) -> bool {
        self.tablet.as_u64() == crate::types::CONTROL_TABLET_RAW
    }

    /// Returns the replicated control image (meaningful on the control
    /// group; empty elsewhere).
    #[must_use]
    pub const fn control(&self) -> &kivi_control::ControlState {
        &self.control
    }

    /// Returns the tablet identity.
    #[must_use]
    pub const fn tablet(&self) -> TabletId {
        self.tablet
    }

    /// Returns the current authority triple.
    #[must_use]
    pub const fn authority(&self) -> TabletAuthority {
        self.authority
    }

    /// Returns the last applied pointer (`None` before the first apply).
    #[must_use]
    pub const fn applied(&self) -> Option<AppliedPointer> {
        self.applied
    }

    /// Returns the last applied commit position (`UNASSIGNED` before the
    /// first apply) through the [`commit_of_index`] mapping.
    #[must_use]
    pub fn applied_commit(&self) -> CommitPosition {
        self.applied.map_or(CommitPosition::UNASSIGNED, |pointer| {
            commit_of_index(pointer.index)
        })
    }

    /// Returns the underlying object store (read-only inspection for the
    /// linearizable-read path and diagnostics).
    #[must_use]
    pub const fn store(&self) -> &ObjectStore {
        &self.store
    }

    /// Enables or disables the ordered key index (ordered namespaces
    /// enable it; hash namespaces leave it off). Enabling rebuilds the
    /// index deterministically from current objects, so late enablement is
    /// still exact.
    pub fn set_ordered_indexing(&mut self, enabled: bool) {
        self.store.set_ordered_indexing(enabled);
    }

    /// Whether the ordered index is currently maintained.
    #[must_use]
    pub fn ordered_index_enabled(&self) -> bool {
        self.store.ordered_index_enabled()
    }

    /// Approximate median split key of this replica. The ordered index is
    /// rebuilt first when disabled (same lazy rule as [`scan_local`](Self::scan_local)),
    /// so a median is available whenever the tablet holds enough keys —
    /// split planning never observes a permanently unindexed tablet.
    #[must_use]
    pub fn median_split_key(&mut self) -> Option<kivi_state::Key> {
        if !self.store.ordered_index_enabled() {
            self.store.rebuild_ordered_index();
        }
        self.store.median_split_key()
    }

    /// Serves one bounded local scan page. The ordered index is rebuilt
    /// first when disabled, so the index always matches logical state
    /// before serving scans (deterministic state, never a cache).
    ///
    /// # Errors
    ///
    /// Returns [`ScanError`](kivi_state::ScanError) for malformed specs.
    /// The caller must have established its read contract first; this never
    /// talks to a quorum itself.
    pub fn scan_local(
        &mut self,
        spec: &kivi_state::ScanSpec,
        now: UnixMicros,
    ) -> Result<kivi_state::ScanPage, kivi_state::ScanError> {
        if !self.store.ordered_index_enabled() {
            self.store.rebuild_ordered_index();
        }
        self.store.scan_range(spec, now)
    }

    /// Prepared transaction intents currently reserving keys, in key order
    /// (split/merge fencing and the resolver consult this).
    #[must_use]
    pub fn pending_intents(&self) -> Vec<kivi_state::TxnIntent> {
        self.store.snapshot_intents()
    }

    /// Number of prepared intents on this replica.
    #[must_use]
    pub fn pending_intent_count(&self) -> usize {
        self.store.pending_intent_count()
    }

    /// Logical telemetry at `now` for split/merge policy.
    #[must_use]
    pub fn tablet_stats(&self, now: UnixMicros) -> kivi_state::StoreStats {
        self.store.stats(now)
    }

    /// Removes and returns every prepared intent (topology cutover
    /// migrates intents to the keys' new owners; never discards them).
    pub fn drain_intents(&mut self) -> Vec<kivi_state::TxnIntent> {
        let intents = self.store.snapshot_intents();
        for intent in &intents {
            self.store.stage_intent_for_key(intent.key.clone(), None);
        }
        intents
    }

    /// Installs one migrated intent verbatim (cutover path only).
    pub fn restore_intent(&mut self, intent: kivi_state::TxnIntent) {
        self.store.restore_intent(intent);
    }

    /// Returns one session's dedup state, if this replica has seen it.
    #[must_use]
    pub fn session(&self, session: SessionId) -> Option<&SessionState> {
        self.sessions.get(&session)
    }

    /// Exports a full copy of live objects plus all dedup sessions for
    /// topology copy (split/merge base). Objects are cheap clones
    /// (`Bytes` refcounts; chunked roots reference the same immutable
    /// `ManifestId` — never a chunk copy). Callers partition objects by
    /// [`PartitionHasher`](kivi_state::PartitionHasher); sessions copy
    /// wholesale to every child (bounded, and required for exactly-once
    /// retries across the cutover).
    #[must_use]
    pub fn export_topology_copy(&self) -> TopologyCopy {
        let objects = self.store.snapshot_entries();
        let sessions = self
            .sessions
            .iter()
            .map(|(session, state)| {
                (
                    *session,
                    state.floor(),
                    state
                        .outcomes()
                        .map(|(seq, entry)| (*seq, entry.clone()))
                        .collect(),
                )
            })
            .collect();
        (objects, sessions)
    }

    /// Installs a topology copy into an inactive (not yet serving) child
    /// or merge target: replaces all objects and the full session set.
    /// Safe only before the target activates (it serves nothing yet, so
    /// overwrite cannot lose acknowledged writes); after cutover the
    /// target is an ordinary tablet and this is never called again.
    pub fn install_topology_copy(&mut self, copy: TopologyCopy) {
        let (objects, sessions) = copy;
        for (key, _) in self.store.snapshot_entries() {
            self.store.remove_stored(&key);
        }
        for (key, object) in objects {
            self.store.put_stored(key, object);
        }
        self.sessions.clear();
        for (session, floor, outcomes) in sessions {
            self.sessions
                .insert(session, SessionState::from_parts(floor, outcomes));
        }
    }

    /// Returns the voter set of the currently held membership.
    #[must_use]
    pub fn membership_voters(&self) -> std::collections::BTreeSet<u64> {
        self.membership.membership().voter_ids().collect()
    }

    /// Read-only proposal gate for the leader pipeline: the dedup
    /// decision without any state change. `Admit` means the leader may
    /// prepare and propose; `Hit` answers the retry from retained state;
    /// `Expired`/`Overloaded` reject before anything replicates. Floor
    /// advances and outcome installs happen exactly once at apply time in
    /// log order on every replica, so this check mutates nothing.
    ///
    /// # Errors
    ///
    /// Returns [`StateMachineFault::CorruptState`] when retained state
    /// violates the completed-only invariant (fail loudly, never answer
    /// from corrupt retention).
    pub fn check_proposal(
        &self,
        session: SessionId,
        seq: RequestSeq,
        ack_floor: RequestSeq,
    ) -> Result<ProposalGate, StateMachineFault> {
        let Some(state) = self.sessions.get(&session) else {
            return Ok(ProposalGate::Admit);
        };
        let floor = state.floor.as_u64().max(ack_floor.as_u64());
        if seq.as_u64() <= floor {
            return Ok(ProposalGate::Expired);
        }
        if let Some(retained) = state.outcomes.get(&seq) {
            match &retained.outcome {
                DurableOutcome::Completed(result) => {
                    return Ok(ProposalGate::Hit {
                        outcome: result.clone(),
                    });
                }
                DurableOutcome::Rejected(_) | DurableOutcome::VersionExhausted => {
                    return Err(StateMachineFault::CorruptState {
                        detail: format!(
                            "session {session} sequence {seq} retains a non-completed outcome"
                        ),
                    });
                }
            }
        }
        // The cap counts post-advance retention (the ack accompanying
        // this proposal frees its outcomes first), exactly as apply does.
        let retained_after = state
            .outcomes
            .keys()
            .filter(|seq| seq.as_u64() > floor)
            .count();
        if retained_after >= SESSION_OUTCOME_CAP {
            return Ok(ProposalGate::Overloaded);
        }
        Ok(ProposalGate::Admit)
    }

    /// Prepares one operation against current state without mutating: the
    /// leader pipeline calls this to materialize the deterministic
    /// [`Mutation`] and its expected outcome before proposing. Read-only
    /// opcodes answer inline; mutating opcodes yield the mutation to
    /// replicate or the terminal rejection to answer locally.
    ///
    /// # Errors
    ///
    /// Returns [`OpError`](kivi_state::OpError) for read-only opcode failures only; mutating
    /// failures arrive as [`StorePrepared::Terminal`](kivi_state::StorePrepared::Terminal).
    pub fn prepare_operation(
        &self,
        op: &Operation,
        now: UnixMicros,
    ) -> Result<kivi_state::StorePrepared, kivi_state::OpError> {
        self.store.prepare_durable(op, now)
    }

    /// Serves a local read against applied state. The caller must have
    /// established a linearizable barrier first (task Q); this never talks
    /// to a quorum itself.
    ///
    /// # Errors
    ///
    /// Returns [`OpError`](kivi_state::OpError) for read validation failures (wrong type);
    /// passing a mutating operation is a caller bug and fails as
    /// [`StateMachineFault::NotARead`]. Reads never mutate.
    pub fn read_local(
        &self,
        op: &Operation,
        now: UnixMicros,
    ) -> Result<OperationResult, StateMachineFault> {
        match self
            .store
            .prepare(op, now)
            .map_err(|error| StateMachineFault::ApplyFailed {
                index: self.applied.map_or(0, |pointer| pointer.index),
                detail: format!("local read rejected: {error}"),
            })? {
            kivi_state::Prepared::Read(result) => Ok(result),
            kivi_state::Prepared::Write(_) => Err(StateMachineFault::NotARead),
        }
    }

    /// Applies one committed log entry: blank/membership bookkeeping, a
    /// full deterministic tablet command with dedup and verification, or
    /// a typed control-plane mutation on the control group.
    ///
    /// # Errors
    ///
    /// Returns [`StateMachineFault`] on divergence, misrouting, expiry,
    /// control rejection, or apply failure — all fail-loudly classes,
    /// never client errors.
    pub fn apply_entry(
        &mut self,
        log_id: LogIdOf<KiviTypeConfig>,
        payload: &EntryPayloadOf<KiviTypeConfig>,
    ) -> Result<ReplicatedOutcome, StateMachineFault> {
        let index = log_id.index;
        match payload {
            EntryPayload::Blank => {
                self.applied = Some(AppliedPointer::of_log_id(log_id));
                Ok(ReplicatedOutcome::none())
            }
            EntryPayload::Membership(membership) => {
                self.membership = openraft::StoredMembership::new(Some(log_id), membership.clone());
                self.applied = Some(AppliedPointer::of_log_id(log_id));
                Ok(ReplicatedOutcome::none())
            }
            EntryPayload::Normal(command) => {
                let outcome = self.apply_envelope(index, log_id, command)?;
                self.applied = Some(AppliedPointer::of_log_id(log_id));
                Ok(outcome)
            }
        }
    }

    /// Dispatches one committed envelope by group kind: tablet commands
    /// apply through the deterministic tablet path, control mutations
    /// through replicated control state, read fences advance the applied
    /// pointer without touching logical state. Cross-kind commands fail as
    /// [`WrongGroup`](StateMachineFault::WrongGroup), never silently.
    fn apply_envelope(
        &mut self,
        index: u64,
        log_id: LogIdOf<KiviTypeConfig>,
        command: &ConsensusCommand,
    ) -> Result<ReplicatedOutcome, StateMachineFault> {
        use StateMachineFault as Fault;
        match command {
            ConsensusCommand::Tablet(mutation) => {
                if self.is_control() {
                    return Err(Fault::WrongGroup {
                        detail: format!(
                            "tablet command for tablet {} reached the control group at index {index}",
                            mutation.envelope().tablet().as_u64(),
                        ),
                    });
                }
                self.apply_command(index, mutation)
            }
            ConsensusCommand::Control(mutation) => {
                if !self.is_control() {
                    return Err(Fault::WrongGroup {
                        detail: format!(
                            "control mutation reached tablet {} at index {index}",
                            self.tablet.as_u64(),
                        ),
                    });
                }
                self.control
                    .apply(mutation)
                    .map_err(|error| Fault::ControlRejected {
                        index,
                        detail: error.to_string(),
                    })?;
                let _ = log_id;
                Ok(ReplicatedOutcome::none())
            }
            ConsensusCommand::ReadSync(sync) => {
                // Lazy-ALR fence: ordering without mutation. The applied
                // pointer advances (the caller in `apply_entry` records it),
                // which is the entire effect — and exactly the effect the
                // waiting batch needs. No objects, sessions, floors, or
                // outcomes change, so there is nothing to verify and no
                // dedup to install; a fence replays identically.
                if self.is_control() {
                    return Err(Fault::WrongGroup {
                        detail: format!(
                            "read fence for tablet {} reached the control group at index {index}",
                            sync.fence.tablet.as_u64(),
                        ),
                    });
                }
                if sync.fence.tablet != self.tablet {
                    return Err(Fault::Misrouted {
                        namespace: self.namespace.as_u64(),
                        tablet: sync.fence.tablet.as_u64(),
                        expected_namespace: self.namespace.as_u64(),
                        expected_tablet: self.tablet.as_u64(),
                    });
                }
                let _ = log_id;
                Ok(ReplicatedOutcome::none())
            }
        }
    }

    /// Applies one committed deterministic command with replicated dedup
    /// and expected-outcome verification.
    fn apply_command(
        &mut self,
        index: u64,
        command: &ReplicatedMutation,
    ) -> Result<ReplicatedOutcome, StateMachineFault> {
        use StateMachineFault as Fault;
        let envelope = command.envelope();
        if envelope.namespace() != self.namespace || envelope.tablet() != self.tablet {
            return Err(Fault::Misrouted {
                namespace: envelope.namespace().as_u64(),
                tablet: envelope.tablet().as_u64(),
                expected_namespace: self.namespace.as_u64(),
                expected_tablet: self.tablet.as_u64(),
            });
        }
        envelope
            .authority()
            .check_against(&self.authority)
            .map_err(|mismatch| Fault::AuthorityMismatch {
                detail: mismatch.to_string(),
            })?;
        let identity = envelope.client();
        let session = identity.session();
        let seq = identity.seq();
        // Anonymous commands (reserved session/sequence zero: no retry
        // contract) skip dedup entirely — lookup and install alike — so
        // unrelated anonymous writes never alias one fake identity.
        if session.as_u128() == ANONYMOUS_SESSION && seq.as_u64() == 0 {
            return self.apply_fresh(index, command, envelope);
        }
        // Read-only dedup verdict first (borrows end before mutation).
        let hit = self.dedup_lookup(session, seq, command.ack_floor(), index)?;
        if let Some(retained) = hit {
            // Same-identity retry (lost-response failover path): return
            // the original outcome without re-executing, after verifying
            // the proposal agrees with history.
            if &retained != command.expected() {
                return Err(Fault::DedupDivergence {
                    index,
                    retained,
                    expected: command.expected().clone(),
                });
            }
            return Ok(ReplicatedOutcome::new(retained));
        }
        let outcome = self.apply_fresh(index, command, envelope)?;
        let result = outcome
            .outcome()
            .cloned()
            .expect("fresh apply installs outcomes");
        // The floor advances in log order on every replica identically
        // (the watermark replicates inside the command), so retained sets
        // converge exactly. Installed here, exactly once per apply.
        let state = self.sessions.entry(session).or_default();
        state.advance_floor(command.ack_floor());
        let mutation = envelope.operation().clone();
        state.outcomes.insert(
            seq,
            RetainedOutcome {
                commit: commit_of_index(index),
                opcode: kivi_protocol::mutation_opcode(&mutation, is_persist_expiry(&mutation))
                    .as_u8(),
                outcome: DurableOutcome::Completed(result),
            },
        );
        Ok(outcome)
    }

    /// Read-only dedup lookup: advances nothing, installs nothing.
    /// Returns the retained outcome on a hit, `None` when fresh.
    fn dedup_lookup(
        &self,
        session: SessionId,
        seq: RequestSeq,
        ack_floor: RequestSeq,
        index: u64,
    ) -> Result<Option<OperationResult>, StateMachineFault> {
        use StateMachineFault as Fault;
        let Some(state) = self.sessions.get(&session) else {
            return Ok(None);
        };
        let floor = state.floor.as_u64().max(ack_floor.as_u64());
        if seq.as_u64() <= floor {
            return Err(Fault::ExpiredIdentity {
                session: session.as_u128(),
                seq: seq.as_u64(),
                floor,
                index,
            });
        }
        match state.outcomes.get(&seq) {
            None => {
                // Post-advance retention, matching `check_proposal` and
                // the install below (the ack frees its outcomes first).
                let retained_after = state
                    .outcomes
                    .keys()
                    .filter(|seq| seq.as_u64() > floor)
                    .count();
                if retained_after >= SESSION_OUTCOME_CAP {
                    return Err(Fault::SessionOverloaded {
                        session: session.as_u128(),
                        index,
                    });
                }
                Ok(None)
            }
            Some(retained) => match &retained.outcome {
                DurableOutcome::Completed(result) => Ok(Some(result.clone())),
                DurableOutcome::Rejected(_) | DurableOutcome::VersionExhausted => {
                    Err(Fault::CorruptState {
                        detail: format!(
                            "session {} sequence {} retains a non-completed outcome at index {index}",
                            session.as_u128(),
                            seq.as_u64(),
                        ),
                    })
                }
            },
        }
    }

    /// Applies one command fresh: deterministic apply plus
    /// expected-outcome verification, without touching dedup state.
    /// Shared by identified applies (which install afterwards) and
    /// anonymous applies (which install nothing).
    fn apply_fresh(
        &mut self,
        index: u64,
        command: &ReplicatedMutation,
        envelope: &kivi_state::MutationEnvelope,
    ) -> Result<ReplicatedOutcome, StateMachineFault> {
        use StateMachineFault as Fault;
        // The envelope carries the typed Mutation IR directly (the
        // leader's pipeline prepared and materialized it, including
        // resolved conditional expiries); replicas apply it verbatim.
        // Chunked roots apply as manifest references with no I/O: the
        // leader pipeline stages sidecars before proposing, so apply meets
        // no reference it cannot store.
        let mutation = envelope.operation().clone();
        let outcome = self
            .store
            .apply(&mutation, command.now())
            .map_err(|error| Fault::ApplyFailed {
                index,
                detail: error.to_string(),
            })?;
        let result = outcome_for(&mutation, &outcome, is_persist_expiry(&mutation));
        if &result != command.expected() {
            return Err(Fault::Divergence {
                index,
                expected: command.expected().clone(),
                found: result,
            });
        }
        Ok(ReplicatedOutcome::new(result))
    }
}

/// Snapshot image decode failure (structural only; semantic checks like
/// authority fencing happen at install/open time, never here).
#[derive(Debug, Clone, PartialEq, Eq)]
enum ImageFault {
    Truncated,
    Oversized { len: usize },
    BadMagic { found: u32 },
    UnsupportedVersion { found: u16 },
    BadCrc { found: u32, computed: u32 },
    TrailingBytes,
    BadKey,
    BadBytes,
    BadValueTag { tag: u8 },
    BadOutcome,
    BadVersion,
    BadExpiry,
    BadChunkedRef,
    BadFabricRef,
    BadAddress,
}

impl From<ImageFault> for StateMachineFault {
    fn from(fault: ImageFault) -> Self {
        Self::CorruptImage {
            detail: format!("snapshot image invalid: {fault:?}"),
        }
    }
}

/// Encodes one prepared transaction intent for snapshots (deterministic
/// key-ordered section after sessions). Carries the write-set digest so
/// restarts never lose the confused-deputy binding.
fn encode_intent(out: &mut Vec<u8>, intent: &kivi_state::TxnIntent) {
    use kivi_codec::Encode;
    out.extend_from_slice(&intent.id.as_bytes());
    push_u64(out, intent.coordinator.as_u64());
    intent.key.encode(out);
    intent.write.expect.encode(out);
    intent.write.kind.encode(out);
    out.extend_from_slice(&intent.digest);
    match intent.observed {
        None => out.push(0),
        Some(version) => {
            out.push(1);
            version.encode(out);
        }
    }
    push_u64(out, intent.prepared_at);
}

/// Decodes one snapshot intent at the cursor.
fn decode_snapshot_intents(
    body: &[u8],
    at: &mut usize,
) -> Result<Vec<kivi_state::TxnIntent>, StateMachineFault> {
    use StateMachineFault as Fault;
    let count = snap_take_count(body, at, "intent")?;
    if count > 1_000_000 {
        return Err(Fault::CorruptImage {
            detail: "snapshot intent count absurd".to_owned(),
        });
    }
    let mut intents = Vec::with_capacity(count.min(1024));
    for _ in 0..count {
        let raw = snap_take(body, at, 16)?;
        let id = kivi_state::TxnId::from_bytes(raw.try_into().unwrap_or([0; 16]));
        let coordinator = kivi_types::TabletId::from_u64(snap_take_u64(body, at)?);
        let (key, used) = Key::decode(&body[*at..]).map_err(|_| Fault::CorruptImage {
            detail: "snapshot intent key invalid".to_owned(),
        })?;
        *at += used;
        let (expect, used) =
            kivi_state::TxnExpect::decode(&body[*at..]).map_err(|_| Fault::CorruptImage {
                detail: "snapshot intent expectation invalid".to_owned(),
            })?;
        *at += used;
        let (kind, used) =
            kivi_state::TxnWriteKind::decode(&body[*at..]).map_err(|_| Fault::CorruptImage {
                detail: "snapshot intent write invalid".to_owned(),
            })?;
        *at += used;
        let digest_raw = snap_take(body, at, 32)?;
        let digest: [u8; 32] = digest_raw.try_into().map_err(|_| Fault::CorruptImage {
            detail: "snapshot intent digest invalid".to_owned(),
        })?;
        let observed = match snap_take(body, at, 1)?[0] {
            0 => None,
            1 => {
                let (version, used) =
                    ObjectVersion::decode(&body[*at..]).map_err(|_| Fault::CorruptImage {
                        detail: "snapshot intent version invalid".to_owned(),
                    })?;
                *at += used;
                Some(version)
            }
            _ => {
                return Err(Fault::CorruptImage {
                    detail: "snapshot intent observed tag invalid".to_owned(),
                });
            }
        };
        let prepared_at = snap_take_u64(body, at)?;
        intents.push(kivi_state::TxnIntent {
            id,
            coordinator,
            key: key.clone(),
            write: kivi_state::TxnWrite { key, kind, expect },
            digest,
            observed,
            prepared_at,
        });
    }
    Ok(intents)
}

/// Encodes one stored object value with its version and expiry.
fn encode_object(out: &mut Vec<u8>, object: &StoredObject) {
    match object.value() {
        LogicalValue::Bytes(bytes) => {
            out.push(VALUE_BYTES);
            encode_bytes(out, bytes);
        }
        LogicalValue::Chunked(chunked) => {
            out.push(VALUE_CHUNKED);
            chunked.encode(out);
        }
        LogicalValue::Fabric(fabric) => {
            out.push(VALUE_FABRIC);
            fabric.encode(out);
        }
        LogicalValue::StrictCounter(value) => {
            out.push(VALUE_COUNTER);
            out.extend_from_slice(&value.to_le_bytes());
        }
        LogicalValue::CommutativeCounter(value) => {
            out.push(VALUE_COMMUTATIVE);
            out.extend_from_slice(&value.to_le_bytes());
        }
        LogicalValue::BoundedCounter(state) => {
            out.push(VALUE_BOUNDED);
            state.encode(out);
        }
        LogicalValue::Semaphore(state) => {
            out.push(VALUE_SEMAPHORE);
            state.encode(out);
        }
        LogicalValue::Lease(state) => {
            out.push(VALUE_LEASE);
            state.encode(out);
        }
        LogicalValue::StreamShard(shard) => {
            out.push(VALUE_STREAM_SHARD);
            shard.encode(out);
        }
    }
    object.version().encode(out);
    object.expiry().encode(out);
}

/// Decodes one stored object value with its version and expiry.
fn decode_object(input: &[u8]) -> Result<(StoredObject, usize), ImageFault> {
    use ImageFault as Fault;
    if input.is_empty() {
        return Err(Fault::Truncated);
    }
    let (value, mut at) = match input[0] {
        VALUE_BYTES => {
            let (bytes, used) = decode_byte_vec(&input[1..]).map_err(|_| Fault::BadBytes)?;
            (LogicalValue::Bytes(bytes::Bytes::from(bytes)), 1 + used)
        }
        VALUE_CHUNKED => {
            let (chunked, used) =
                kivi_state::ChunkedRef::decode(&input[1..]).map_err(|_| Fault::BadChunkedRef)?;
            (LogicalValue::Chunked(chunked), 1 + used)
        }
        VALUE_FABRIC => {
            let (fabric, used) =
                kivi_state::FabricRef::decode(&input[1..]).map_err(|_| Fault::BadFabricRef)?;
            (LogicalValue::Fabric(fabric), 1 + used)
        }
        VALUE_COUNTER => {
            if input.len() < 1 + 8 {
                return Err(Fault::Truncated);
            }
            let value = i64::from_le_bytes(input[1..9].try_into().unwrap_or([0; 8]));
            (LogicalValue::StrictCounter(value), 9)
        }
        VALUE_COMMUTATIVE => {
            if input.len() < 1 + 8 {
                return Err(Fault::Truncated);
            }
            let value = i64::from_le_bytes(input[1..9].try_into().unwrap_or([0; 8]));
            (LogicalValue::CommutativeCounter(value), 9)
        }
        VALUE_BOUNDED => {
            let (state, used) = kivi_state::BoundedCounterState::decode(&input[1..])
                .map_err(|_| Fault::BadBytes)?;
            if !state.invariant_holds() {
                return Err(Fault::BadValueTag { tag: VALUE_BOUNDED });
            }
            (LogicalValue::BoundedCounter(state), 1 + used)
        }
        VALUE_SEMAPHORE => {
            let (state, used) =
                kivi_state::SemaphoreState::decode(&input[1..]).map_err(|_| Fault::BadBytes)?;
            if !state.invariant_holds() {
                return Err(Fault::BadValueTag {
                    tag: VALUE_SEMAPHORE,
                });
            }
            (LogicalValue::Semaphore(state), 1 + used)
        }
        VALUE_LEASE => {
            let (state, used) =
                kivi_state::LeaseState::decode(&input[1..]).map_err(|_| Fault::BadBytes)?;
            (LogicalValue::Lease(state), 1 + used)
        }
        VALUE_STREAM_SHARD => {
            let (shard, used) =
                kivi_state::StreamShardState::decode(&input[1..]).map_err(|_| Fault::BadBytes)?;
            (LogicalValue::StreamShard(shard), 1 + used)
        }
        tag => return Err(Fault::BadValueTag { tag }),
    };
    let (version, used) = ObjectVersion::decode(&input[at..]).map_err(|_| Fault::BadVersion)?;
    at += used;
    let (expiry, used) = Expiry::decode(&input[at..]).map_err(|_| Fault::BadExpiry)?;
    at += used;
    Ok((StoredObject::restore(value, version, expiry), at))
}

/// Encodes the `OpenRaft` membership in its owned (voters + dialable nodes)
/// shape, mirroring the durable [`RaftMembership`](kivi_durability::RaftMembership)
/// record layout so log and snapshot membership never drift.
fn encode_membership(out: &mut Vec<u8>, membership: &StoredMembershipOf<KiviTypeConfig>) {
    let (voters, nodes) = membership_nodes(membership);
    push_u64(out, voters.len() as u64);
    for voter in &voters {
        push_u64(out, *voter);
    }
    push_u64(out, nodes.len() as u64);
    for (id, addr) in &nodes {
        push_u64(out, *id);
        encode_bytes(out, addr.as_bytes());
    }
}

/// Extracts the single-group voter set and node map from stored
/// membership (this stage replicates one tablet with one voter group;
/// multi-group membership arrives with dynamic membership, out of scope).
fn membership_nodes(
    membership: &StoredMembershipOf<KiviTypeConfig>,
) -> (Vec<u64>, Vec<(u64, String)>) {
    let mut voters: Vec<u64> = membership.membership().voter_ids().collect();
    voters.sort_unstable();
    let mut nodes: Vec<(u64, String)> = membership
        .membership()
        .nodes()
        .map(|(id, node)| (*id, node.addr.clone()))
        .collect();
    nodes.sort_by_key(|(id, _)| *id);
    (voters, nodes)
}

/// Decodes membership nodes back into `OpenRaft` form with its log id.
fn decode_membership(
    input: &[u8],
    log_id: Option<LogIdOf<KiviTypeConfig>>,
) -> Result<(StoredMembershipOf<KiviTypeConfig>, usize), ImageFault> {
    use ImageFault as Fault;
    let (voters, mut at) = take_u64_list(input)?;
    let (count, used) = take_u64(&input[at..])?;
    at += used;
    let count = usize::try_from(count).map_err(|_| Fault::Oversized { len: usize::MAX })?;
    if count.saturating_mul(12) > input.len().saturating_sub(at) {
        return Err(Fault::Oversized {
            len: count.saturating_mul(12),
        });
    }
    let mut nodes = BTreeMap::new();
    for _ in 0..count {
        let (id, used) = take_u64(&input[at..])?;
        at += used;
        let (addr, used) = decode_byte_vec(&input[at..]).map_err(|_| Fault::BadAddress)?;
        at += used;
        let addr = core::str::from_utf8(&addr).map_err(|_| Fault::BadAddress)?;
        nodes.insert(id, BasicNode::new(addr.to_owned()));
    }
    let voter_set: BTreeSet<u64> = voters.into_iter().collect();
    // The default (empty) membership is a real state — a fresh replica
    // that never installed one — and must round-trip exactly. Strict
    // validation applies to non-empty memberships only.
    if voter_set.is_empty() && nodes.is_empty() {
        return Ok((openraft::StoredMembership::default(), at));
    }
    let membership = Membership::new(vec![voter_set], nodes).map_err(|_| Fault::BadAddress)?;
    Ok((openraft::StoredMembership::new(log_id, membership), at))
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn take_u64(input: &[u8]) -> Result<(u64, usize), ImageFault> {
    if input.len() < 8 {
        return Err(ImageFault::Truncated);
    }
    Ok((
        u64::from_le_bytes(input[..8].try_into().unwrap_or([0; 8])),
        8,
    ))
}

fn take_u64_list(input: &[u8]) -> Result<(Vec<u64>, usize), ImageFault> {
    let (count, mut at) = take_u64(input)?;
    let count = usize::try_from(count).map_err(|_| ImageFault::Oversized { len: usize::MAX })?;
    if count.saturating_mul(8) > input.len().saturating_sub(at) {
        return Err(ImageFault::Oversized {
            len: count.saturating_mul(8),
        });
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let (word, used) = take_u64(&input[at..])?;
        out.push(word);
        at += used;
    }
    Ok((out, at))
}

/// Frames `body` with magic + version + CRC (project-owned framing per
/// RFC §53: magic, version, length-capable blobs, CRC; immutable
/// artifacts additionally carry content identity via the snapshot id).
fn frame_image(magic: u32, version: u16, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + 2 + body.len() + 4);
    out.extend_from_slice(&magic.to_le_bytes());
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(body);
    out.extend_from_slice(&crc32c::crc32c(body).to_le_bytes());
    out
}

/// Validates framing and returns the body.
fn unframe_image(input: &[u8], magic: u32, version: u16) -> Result<&[u8], ImageFault> {
    use ImageFault as Fault;
    if input.len() < 4 + 2 + 4 {
        return Err(Fault::Truncated);
    }
    let found_magic = u32::from_le_bytes(input[..4].try_into().unwrap_or([0; 4]));
    if found_magic != magic {
        return Err(Fault::BadMagic { found: found_magic });
    }
    let found_version = u16::from_le_bytes(input[4..6].try_into().unwrap_or([0; 2]));
    if found_version != version {
        return Err(Fault::UnsupportedVersion {
            found: found_version,
        });
    }
    let body = &input[6..input.len() - 4];
    let found_crc = u32::from_le_bytes(input[input.len() - 4..].try_into().unwrap_or([0; 4]));
    let computed = crc32c::crc32c(body);
    if found_crc != computed {
        return Err(Fault::BadCrc {
            found: found_crc,
            computed,
        });
    }
    Ok(body)
}

impl ReplicatedTablet {
    /// Encodes the full snapshot body: identity, applied pointer,
    /// membership, objects (sorted by key for determinism), and sessions
    /// (sorted by session, outcomes in sequence order).
    #[must_use]
    pub fn encode_snapshot_body(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u64(&mut out, self.namespace.as_u64());
        push_u64(&mut out, self.tablet.as_u64());
        push_u64(&mut out, self.authority.epoch().as_u64());
        push_u64(&mut out, self.authority.guard().as_u64());
        let applied = self.applied.unwrap_or(AppliedPointer::new(0, 0, 0));
        push_u64(&mut out, applied.term);
        push_u64(&mut out, applied.leader);
        push_u64(&mut out, applied.index);
        // Membership log id rides along so the snapshot grounds
        // `applied_state` exactly (task T metadata requirement). The
        // encode side stays total (`unwrap_or(0)`): images from any source
        // must re-encode without panicking; only the live protocol path
        // (`of_log_id`) is strict.
        match self.membership.log_id() {
            None => out.push(0),
            Some(id) => {
                out.push(1);
                push_u64(&mut out, id.leader_id.term);
                push_u64(&mut out, id.leader_id.node_id);
                push_u64(&mut out, id.index);
            }
        }
        encode_membership(&mut out, &self.membership);
        let mut objects = self.store.snapshot_entries();
        objects.sort_by(|(a, _), (b, _)| a.as_bytes().cmp(b.as_bytes()));
        push_u64(&mut out, objects.len() as u64);
        for (key, object) in &objects {
            key.encode(&mut out);
            encode_object(&mut out, object);
        }
        let mut sessions: Vec<(&SessionId, &SessionState)> = self.sessions.iter().collect();
        sessions.sort_by_key(|(session, _)| session.as_u128());
        push_u64(&mut out, sessions.len() as u64);
        for (session, state) in sessions {
            out.extend_from_slice(&session.as_u128().to_le_bytes());
            push_u64(&mut out, state.floor.as_u64());
            push_u64(&mut out, state.outcomes.len() as u64);
            for (seq, retained) in &state.outcomes {
                push_u64(&mut out, seq.as_u64());
                push_u64(&mut out, retained.commit.as_u64());
                out.push(retained.opcode);
                let outcome_bytes = retained.outcome.encode_to_vec();
                encode_bytes(&mut out, &outcome_bytes);
            }
        }
        // Ordered-index flag plus prepared transaction intents (key order):
        // intents are deterministic state and must survive snapshot
        // transfer and restart; the index itself rebuilds from objects.
        out.push(u8::from(self.store.ordered_index_enabled()));
        let intents = self.store.snapshot_intents();
        push_u64(&mut out, intents.len() as u64);
        for intent in &intents {
            encode_intent(&mut out, intent);
        }
        out
    }

    /// Encodes the framed snapshot image (transport + file bytes).
    ///
    /// The control group encodes replicated control state under its own
    /// magic; tablet groups encode objects/sessions as before. The branch
    /// is by group kind, so neither format ever aliases the other.
    #[must_use]
    pub fn encode_snapshot(&self) -> Vec<u8> {
        if self.is_control() {
            return frame_image(
                CONTROL_SNAPSHOT_MAGIC,
                CONTROL_SNAPSHOT_VERSION,
                &self.encode_control_body(),
            );
        }
        frame_image(
            SNAPSHOT_MAGIC,
            SNAPSHOT_VERSION,
            &self.encode_snapshot_body(),
        )
    }

    /// Encodes the control snapshot body: identity, applied pointer,
    /// membership log id, membership, plus the canonical control image.
    #[must_use]
    pub fn encode_control_body(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u64(&mut out, self.namespace.as_u64());
        push_u64(&mut out, self.tablet.as_u64());
        let applied = self.applied.unwrap_or(AppliedPointer::new(0, 0, 0));
        push_u64(&mut out, applied.term);
        push_u64(&mut out, applied.leader);
        push_u64(&mut out, applied.index);
        match self.membership.log_id() {
            None => out.push(0),
            Some(id) => {
                out.push(1);
                push_u64(&mut out, id.leader_id.term);
                push_u64(&mut out, id.leader_id.node_id);
                push_u64(&mut out, id.index);
            }
        }
        encode_membership(&mut out, &self.membership);
        let control = self.control.encode_snapshot();
        push_u64(&mut out, control.len() as u64);
        out.extend_from_slice(&control);
        out
    }

    /// Decodes a framed snapshot image into tablet state, validating
    /// identity against the expected replica. Structural faults fail here;
    /// authority fencing is checked by the caller with full context.
    ///
    /// # Errors
    ///
    /// Returns [`StateMachineFault`] on framing, CRC, identity, or content
    /// failures.
    pub fn decode_snapshot(
        bytes: &[u8],
        namespace: NamespaceId,
        tablet: TabletId,
    ) -> Result<DecodedSnapshot, StateMachineFault> {
        use StateMachineFault as Fault;
        if tablet.as_u64() == crate::types::CONTROL_TABLET_RAW {
            return Self::decode_control_snapshot(bytes, namespace, tablet);
        }
        let body = unframe_image(bytes, SNAPSHOT_MAGIC, SNAPSHOT_VERSION).map_err(|fault| {
            Fault::CorruptImage {
                detail: format!("snapshot framing invalid: {fault:?}"),
            }
        })?;
        let mut at = 0;
        let head = decode_snapshot_head(body, &mut at, namespace, tablet)?;
        let (membership, used) =
            decode_membership(&body[at..], head.membership_log).map_err(|fault| {
                Fault::CorruptImage {
                    detail: format!("snapshot membership invalid: {fault:?}"),
                }
            })?;
        at += used;
        let objects = decode_snapshot_objects(body, &mut at)?;
        let sessions = decode_snapshot_sessions(body, &mut at)?;
        let ordered_index = match snap_take(body, &mut at, 1)?[0] {
            0 => false,
            1 => true,
            _ => {
                return Err(Fault::CorruptImage {
                    detail: "snapshot ordered-index tag invalid".to_owned(),
                });
            }
        };
        let intents = decode_snapshot_intents(body, &mut at)?;
        if at != body.len() {
            return Err(Fault::CorruptImage {
                detail: "snapshot body has trailing bytes".to_owned(),
            });
        }
        Ok(DecodedSnapshot {
            epoch: kivi_types::TabletEpoch::from_u64(head.epoch),
            guard: kivi_types::WriteGuardGeneration::from_u64(head.guard),
            applied: if head.index == 0 && head.term == 0 && head.leader == 0 {
                None
            } else {
                Some(AppliedPointer::new(head.term, head.leader, head.index))
            },
            membership,
            objects,
            sessions,
            ordered_index,
            intents,
            control: None,
        })
    }

    /// Decodes a framed control snapshot image into control state.
    fn decode_control_snapshot(
        bytes: &[u8],
        namespace: NamespaceId,
        tablet: TabletId,
    ) -> Result<DecodedSnapshot, StateMachineFault> {
        use StateMachineFault as Fault;
        let body = unframe_image(bytes, CONTROL_SNAPSHOT_MAGIC, CONTROL_SNAPSHOT_VERSION).map_err(
            |fault| Fault::CorruptImage {
                detail: format!("control snapshot framing invalid: {fault:?}"),
            },
        )?;
        let mut at = 0;
        let found_namespace = snap_take_u64(body, &mut at)?;
        let found_tablet = snap_take_u64(body, &mut at)?;
        if found_namespace != namespace.as_u64() || found_tablet != tablet.as_u64() {
            return Err(Fault::Misrouted {
                namespace: found_namespace,
                tablet: found_tablet,
                expected_namespace: namespace.as_u64(),
                expected_tablet: tablet.as_u64(),
            });
        }
        let term = snap_take_u64(body, &mut at)?;
        let leader = snap_take_u64(body, &mut at)?;
        let index = snap_take_u64(body, &mut at)?;
        let membership_flag = snap_take(body, &mut at, 1)?[0];
        let membership_log = if membership_flag == 0 {
            None
        } else {
            use openraft::impls::leader_id_adv::LeaderId;
            let term = snap_take_u64(body, &mut at)?;
            let node = snap_take_u64(body, &mut at)?;
            let index = snap_take_u64(body, &mut at)?;
            Some(openraft::LogId::new(LeaderId::new(term, node), index))
        };
        let (membership, used) =
            decode_membership(&body[at..], membership_log).map_err(|fault| {
                Fault::CorruptImage {
                    detail: format!("control snapshot membership invalid: {fault:?}"),
                }
            })?;
        at += used;
        let len = snap_take_count(body, &mut at, "control image")?;
        let image = snap_take(body, &mut at, len)?;
        if at != body.len() {
            return Err(Fault::CorruptImage {
                detail: "control snapshot body has trailing bytes".to_owned(),
            });
        }
        let control = kivi_control::ControlState::decode_snapshot(image).map_err(|detail| {
            Fault::CorruptImage {
                detail: format!("control image invalid: {detail}"),
            }
        })?;
        Ok(DecodedSnapshot {
            epoch: kivi_types::TabletEpoch::INITIAL,
            guard: kivi_types::WriteGuardGeneration::INITIAL,
            applied: if index == 0 && term == 0 && leader == 0 {
                None
            } else {
                Some(AppliedPointer::new(term, leader, index))
            },
            membership,
            objects: Vec::new(),
            sessions: Vec::new(),
            ordered_index: false,
            intents: Vec::new(),
            control: Some(control),
        })
    }

    /// Installs decoded snapshot state wholesale (install path only: the
    /// caller proves durability before and after).
    fn install_decoded(&mut self, decoded: DecodedSnapshot) {
        if let Some(control) = decoded.control {
            self.control = control;
            self.applied = decoded.applied;
            self.membership = decoded.membership;
            return;
        }
        self.store = ObjectStore::new();
        for (key, object) in decoded.objects {
            self.store.put_stored(key, object);
        }
        // Unresolved intents are never discarded on restore; the ordered
        // index rebuilds deterministically when enabled.
        for intent in decoded.intents {
            self.store.restore_intent(intent);
        }
        if decoded.ordered_index {
            self.store.rebuild_ordered_index();
        }
        self.sessions.clear();
        for (session, floor, outcomes) in decoded.sessions {
            self.sessions
                .insert(session, SessionState { floor, outcomes });
        }
        self.applied = decoded.applied;
        self.membership = decoded.membership;
    }

    /// Installs replicated control state directly (control-group open
    /// path shares this with snapshot install).
    #[allow(dead_code)]
    fn install_control(&mut self, control: kivi_control::ControlState) {
        self.control = control;
    }
}

/// Takes `n` bytes at the cursor (snapshot-body walker shared by every
/// section decoder below).
fn snap_take<'a>(input: &'a [u8], at: &mut usize, n: usize) -> Result<&'a [u8], StateMachineFault> {
    if input.len() < *at + n {
        return Err(StateMachineFault::CorruptImage {
            detail: "snapshot body truncated".to_owned(),
        });
    }
    let slice = &input[*at..*at + n];
    *at += n;
    Ok(slice)
}

/// Takes one little-endian word at the cursor.
fn snap_take_u64(input: &[u8], at: &mut usize) -> Result<u64, StateMachineFault> {
    Ok(u64::from_le_bytes(
        snap_take(input, at, 8)?.try_into().unwrap_or([0; 8]),
    ))
}

/// Takes a `usize` count at the cursor, rejecting address-space overflow.
fn snap_take_count(input: &[u8], at: &mut usize, what: &str) -> Result<usize, StateMachineFault> {
    let count = snap_take_u64(input, at)?;
    usize::try_from(count).map_err(|_| StateMachineFault::CorruptImage {
        detail: format!("snapshot {what} count overflows address space"),
    })
}

/// Snapshot identity header: replica coordinates, applied pointer, and
/// the membership log id grounding `applied_state`.
struct SnapshotHead {
    epoch: u64,
    guard: u64,
    term: u64,
    leader: u64,
    index: u64,
    membership_log: Option<LogIdOf<KiviTypeConfig>>,
}

/// Decodes the snapshot identity header, fencing the replica coordinates.
fn decode_snapshot_head(
    body: &[u8],
    at: &mut usize,
    namespace: NamespaceId,
    tablet: TabletId,
) -> Result<SnapshotHead, StateMachineFault> {
    use StateMachineFault as Fault;
    let found_namespace = snap_take_u64(body, at)?;
    let found_tablet = snap_take_u64(body, at)?;
    if found_namespace != namespace.as_u64() || found_tablet != tablet.as_u64() {
        return Err(Fault::Misrouted {
            namespace: found_namespace,
            tablet: found_tablet,
            expected_namespace: namespace.as_u64(),
            expected_tablet: tablet.as_u64(),
        });
    }
    let epoch = snap_take_u64(body, at)?;
    let guard = snap_take_u64(body, at)?;
    let term = snap_take_u64(body, at)?;
    let leader = snap_take_u64(body, at)?;
    let index = snap_take_u64(body, at)?;
    let membership_log = match snap_take(body, at, 1)?[0] {
        0 => None,
        1 => {
            let term = snap_take_u64(body, at)?;
            let leader = snap_take_u64(body, at)?;
            let index = snap_take_u64(body, at)?;
            Some(openraft::LogId::new(
                openraft::impls::leader_id_adv::LeaderId::new(term, leader),
                index,
            ))
        }
        _ => {
            return Err(Fault::CorruptImage {
                detail: "snapshot membership tag invalid".to_owned(),
            });
        }
    };
    Ok(SnapshotHead {
        epoch,
        guard,
        term,
        leader,
        index,
        membership_log,
    })
}

/// Decodes the snapshot object section at the cursor.
fn decode_snapshot_objects(
    body: &[u8],
    at: &mut usize,
) -> Result<Vec<(Key, StoredObject)>, StateMachineFault> {
    use StateMachineFault as Fault;
    let object_count = snap_take_count(body, at, "object")?;
    let mut objects = Vec::with_capacity(object_count.min(1_000_000));
    for _ in 0..object_count {
        let (key, used) = Key::decode(&body[*at..]).map_err(|_| ImageFault::BadKey)?;
        *at += used;
        let (object, used) = decode_object(&body[*at..]).map_err(|fault| Fault::CorruptImage {
            detail: format!("snapshot object invalid: {fault:?}"),
        })?;
        *at += used;
        objects.push((key, object));
    }
    Ok(objects)
}

/// Decoded session section: floors plus retained outcomes in order.
type DecodedSessions = Vec<(SessionId, RequestSeq, BTreeMap<RequestSeq, RetainedOutcome>)>;

/// Decodes the snapshot session section at the cursor.
fn decode_snapshot_sessions(
    body: &[u8],
    at: &mut usize,
) -> Result<DecodedSessions, StateMachineFault> {
    use StateMachineFault as Fault;
    let session_count = snap_take_count(body, at, "session")?;
    let mut sessions = Vec::with_capacity(session_count.min(1_000_000));
    for _ in 0..session_count {
        let raw = snap_take(body, at, 16)?;
        let session = SessionId::from_u128(u128::from_le_bytes(raw.try_into().unwrap_or([0; 16])));
        let floor = RequestSeq::from_u64(snap_take_u64(body, at)?);
        let outcome_count = snap_take_count(body, at, "outcome")?;
        let mut outcomes = BTreeMap::new();
        for _ in 0..outcome_count {
            let seq = RequestSeq::from_u64(snap_take_u64(body, at)?);
            let commit = CommitPosition::from_u64(snap_take_u64(body, at)?);
            let opcode = snap_take(body, at, 1)?[0];
            let (bytes, used) = decode_byte_vec(&body[*at..]).map_err(|_| Fault::CorruptImage {
                detail: "snapshot outcome blob invalid".to_owned(),
            })?;
            *at += used;
            let outcome =
                DurableOutcome::decode_exact(&bytes).map_err(|_| ImageFault::BadOutcome)?;
            outcomes.insert(
                seq,
                RetainedOutcome {
                    commit,
                    opcode,
                    outcome,
                },
            );
        }
        sessions.push((session, floor, outcomes));
    }
    Ok(sessions)
}

/// Snapshot state decoded from an image, awaiting a durability-gated
/// install.
#[derive(Debug)]
pub struct DecodedSnapshot {
    /// Tablet epoch at capture.
    pub epoch: kivi_types::TabletEpoch,
    /// Write-guard generation at capture.
    pub guard: kivi_types::WriteGuardGeneration,
    /// Applied pointer at capture (`None` when nothing applied).
    pub applied: Option<AppliedPointer>,
    /// Membership at capture.
    pub membership: StoredMembershipOf<KiviTypeConfig>,
    /// Committed objects.
    pub objects: Vec<(Key, StoredObject)>,
    /// Sessions with floors and retained outcomes.
    pub sessions: DecodedSessions,
    /// Whether the ordered index was enabled at capture.
    pub ordered_index: bool,
    /// Prepared transaction intents at capture (never discarded).
    pub intents: Vec<kivi_state::TxnIntent>,
    /// Replicated control image (control snapshots only; `None` for
    /// tablet snapshots).
    pub control: Option<kivi_control::ControlState>,
}

/// Encodes the applied-file body: applied pointer plus membership (so
/// `applied_state` survives the loss of every snapshot file).
fn encode_applied_body(
    applied: Option<AppliedPointer>,
    membership: &StoredMembershipOf<KiviTypeConfig>,
) -> Vec<u8> {
    let mut out = Vec::new();
    match applied {
        None => out.push(0),
        Some(pointer) => {
            out.push(1);
            push_u64(&mut out, pointer.term);
            push_u64(&mut out, pointer.leader);
            push_u64(&mut out, pointer.index);
        }
    }
    let log_id = membership.log_id();
    match log_id {
        None => out.push(0),
        Some(id) => {
            out.push(1);
            push_u64(&mut out, id.leader_id.term);
            push_u64(&mut out, id.leader_id.node_id);
            push_u64(&mut out, id.index);
        }
    }
    encode_membership(&mut out, membership);
    out
}

/// Crash-safe file write through the owned durability primitive
/// ([`kivi_durability::fs::write_atomic_sync`]: sibling temp file, file
/// sync, atomic rename, directory sync where the platform allows). A crash
/// leaves the old complete image or the new complete image, never a torn
/// one — so recovery treats any structural fault as corruption (loud),
/// never as a tail to repair. No second crash protocol is invented here.
fn write_crash_safe(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    kivi_durability::fs::write_atomic_sync(path, bytes)
}

/// Snapshot file name for an applied base: sortable, unique, parseable.
/// The file name (plus the sealed bytes) IS the Kivi checkpoint identity:
/// no separate snapshot id exists in logical metadata anymore.
fn snapshot_file_name(index: u64, term: u64) -> String {
    format!("snapshot-{index:020}-{term:020}.bin")
}

/// Why the state machine directory could not be opened.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum StateMachineOpenError {
    /// The state directory or its files are unreachable.
    #[error("state-machine I/O failed: {reason}")]
    Io {
        /// Human-readable cause.
        reason: String,
    },
    /// A snapshot or applied image is structurally corrupt.
    #[error("state-machine image corrupt: {detail}")]
    Corrupt {
        /// Human-readable detail.
        detail: String,
    },
    /// A durable image names a different replica identity.
    #[error("state-machine identity mismatch: {detail}")]
    Identity {
        /// Human-readable detail.
        detail: String,
    },
}

/// Read-only state-machine diagnostics for the admin plane (task AE).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateMachineStatus {
    /// Tablet replicated.
    pub tablet: TabletId,
    /// Last applied log index (`None` before the first apply).
    pub applied_index: Option<u64>,
    /// Last applied commit position through [`commit_of_index`].
    pub applied_commit: CommitPosition,
    /// Installed snapshot base index (`None` before the first snapshot).
    pub snapshot_index: Option<u64>,
    /// Whether the machine serves reads and applies.
    pub healthy: bool,
    /// Last latched failure detail (`None` while healthy).
    pub last_error: Option<String>,
}

/// One sealed snapshot: its `OpenRaft` metadata plus the durable image
/// bytes the metadata describes.
type SealedSnapshot = (SnapshotMetaOf<KiviTypeConfig>, Vec<u8>);

/// Durable `OpenRaft` state machine over the Kivi deterministic core.
/// Cloneable: clones share the same serialized state (the `Raft` object
/// owns one; the node layer holds another for reads and diagnostics).
#[derive(Clone, Debug)]
pub struct ReplicatedStateMachine {
    shared: Arc<futures::lock::Mutex<StateMachineInner>>,
}

#[derive(Debug)]
struct StateMachineInner {
    tablet: ReplicatedTablet,
    snapshots_dir: PathBuf,
    applied_path: PathBuf,
    current_path: PathBuf,
    current_snapshot: Option<SealedSnapshot>,
    healthy: bool,
    last_error: Option<String>,
    inject_snapshot_write_failure: bool,
    /// Split/merge cutover fence: while set, mutating proposes fail with
    /// `Fenced` (shaped as retryable `Overloaded`) so a bounded final
    /// tail can install without losing writes. Reads still serve (parent
    /// state is final during the fence). Ephemeral: re-applied from the
    /// persisted split/merge plan phase after a restart.
    fenced: bool,
}

impl ReplicatedStateMachine {
    /// Opens (or creates) the replica's state-machine directory
    /// `<data_dir>/consensus-sm/<tablet>/`: loads the installed snapshot
    /// (adopting a fully written but unlinked newer image when the crash
    /// order left one), verifies identity, and reports the snapshot base
    /// as applied so `OpenRaft` replays the retained tail exactly once.
    ///
    /// # Errors
    ///
    /// Returns [`StateMachineOpenError`] on I/O failure, corrupt images,
    /// or identity mismatch. Never re-bootstraps over durable state.
    pub fn open(
        data_dir: &Path,
        namespace: NamespaceId,
        tablet: TabletId,
        authority: TabletAuthority,
    ) -> Result<Self, StateMachineOpenError> {
        use StateMachineOpenError as Fault;
        let state_dir = data_dir
            .join("consensus-sm")
            .join(tablet.as_u64().to_string());
        let snapshots_dir = state_dir.join("snapshots");
        let applied_path = state_dir.join("applied.bin");
        let current_path = snapshots_dir.join("CURRENT");
        std::fs::create_dir_all(&snapshots_dir).map_err(|error| Fault::Io {
            reason: error.to_string(),
        })?;
        let current_name = std::fs::read_to_string(&current_path)
            .ok()
            .map(|name| name.trim().to_owned())
            .filter(|name| !name.is_empty());
        let applied_bytes = std::fs::read(&applied_path).ok();
        // The applied file names the snapshot it was sealed with; when it
        // names a complete image newer than CURRENT (crash between the
        // applied write and the CURRENT switch), adopting it is safe: the
        // image was fsynced before the applied file.
        let mut snapshot_name = current_name;
        if let Some(bytes) = &applied_bytes
            && let Ok(body) = unframe_image(bytes, APPLIED_MAGIC, APPLIED_VERSION)
            && let Ok((_, _, name)) = decode_applied_file_body(body)
        {
            let candidate = snapshots_dir.join(&name);
            if !name.is_empty() && candidate.is_file() && snapshot_name.as_ref() != Some(&name) {
                snapshot_name = Some(name);
            }
        }
        let mut tablet_core = ReplicatedTablet::new(namespace, tablet, authority);
        let mut current_snapshot = None;
        if let Some(name) = snapshot_name {
            let bytes = std::fs::read(snapshots_dir.join(&name)).map_err(|error| Fault::Io {
                reason: error.to_string(),
            })?;
            let decoded =
                ReplicatedTablet::decode_snapshot(&bytes, namespace, tablet).map_err(|fault| {
                    Fault::Corrupt {
                        detail: fault.to_string(),
                    }
                })?;
            if decoded.epoch != authority.epoch() || decoded.guard != authority.guard() {
                return Err(Fault::Identity {
                    detail: format!(
                        "snapshot authority epoch {} guard {} != replica authority {authority}",
                        decoded.epoch, decoded.guard,
                    ),
                });
            }
            let applied = decoded.applied;
            tablet_core.install_decoded(decoded);
            // The sealed file name is the checkpoint identity (no
            // transfer id in logical metadata); the bytes carry the
            // rest. `name` is unused beyond locating the image.
            let _ = name;
            let meta = SnapshotMeta {
                last_log_id: applied.map(AppliedPointer::openraft),
                last_membership: tablet_core.membership.clone(),
            };
            current_snapshot = Some((meta, bytes));
        }
        // The applied file additionally grounds membership when no
        // snapshot exists yet (fresh replica that applied entries but
        // never snapshotted): adopt its membership, never its objects
        // (there are none durable outside snapshots — the log replays).
        if current_snapshot.is_none()
            && let Some(bytes) = applied_bytes
        {
            let body = unframe_image(&bytes, APPLIED_MAGIC, APPLIED_VERSION).map_err(|fault| {
                Fault::Corrupt {
                    detail: format!("applied file invalid: {fault:?}"),
                }
            })?;
            let (_, membership, _) =
                decode_applied_file_body(body).map_err(|fault| Fault::Corrupt {
                    detail: format!("applied file invalid: {fault:?}"),
                })?;
            tablet_core.membership = membership;
        }
        Ok(Self {
            shared: Arc::new(futures::lock::Mutex::new(StateMachineInner {
                tablet: tablet_core,
                snapshots_dir,
                applied_path,
                current_path,
                current_snapshot,
                healthy: true,
                last_error: None,
                inject_snapshot_write_failure: false,
                fenced: false,
            })),
        })
    }

    /// Live object count of this replica (best-effort sizing signal for
    /// automatic split/merge policy).
    pub async fn object_count(&self) -> usize {
        self.shared.lock().await.tablet.store().len()
    }

    /// Logical telemetry of this replica for split/merge policy.
    pub async fn tablet_stats(&self, now: UnixMicros) -> kivi_state::StoreStats {
        self.shared.lock().await.tablet.tablet_stats(now)
    }

    /// Approximate median split key of this replica (self-healing: the
    /// ordered index rebuilds on demand, so medians work on tablets that
    /// were never scanned).
    pub async fn median_split_key(&self) -> Option<Vec<u8>> {
        self.shared
            .lock()
            .await
            .tablet
            .median_split_key()
            .map(|key| key.as_bytes().to_vec())
    }

    /// Number of prepared transaction intents on this replica.
    pub async fn pending_intent_count(&self) -> usize {
        self.shared.lock().await.tablet.pending_intent_count()
    }

    /// Prepared transaction intents on this replica, in key order
    /// (resolver consults this; never discards here).
    pub async fn snapshot_intents(&self) -> Vec<kivi_state::TxnIntent> {
        self.shared.lock().await.tablet.pending_intents()
    }

    /// Removes and returns every prepared intent (cutover migration).
    pub async fn drain_intents(&self) -> Vec<kivi_state::TxnIntent> {
        self.shared.lock().await.tablet.drain_intents()
    }

    /// Installs one migrated intent verbatim (cutover path only).
    pub async fn restore_intent(&self, intent: kivi_state::TxnIntent) {
        self.shared.lock().await.tablet.restore_intent(intent);
    }

    /// Whether this replica maintains the ordered index.
    pub async fn ordered_index_enabled(&self) -> bool {
        self.shared.lock().await.tablet.ordered_index_enabled()
    }

    /// Enables or disables the ordered key index on this replica.
    pub async fn set_ordered_indexing(&self, enabled: bool) {
        self.shared
            .lock()
            .await
            .tablet
            .set_ordered_indexing(enabled);
    }

    /// Serves one bounded local scan page (rebuilding the ordered index
    /// first when disabled, so scans always meet exact state).
    ///
    /// # Errors
    ///
    /// Returns [`StateMachineFault`] when unhealthy, or the store's
    /// [`ScanError`](kivi_state::ScanError) for malformed specs.
    pub async fn scan_local(
        &self,
        spec: &kivi_state::ScanSpec,
        now: UnixMicros,
    ) -> Result<kivi_state::ScanPage, StateMachineFault> {
        let mut inner = self.shared.lock().await;
        if !inner.healthy {
            return Err(StateMachineFault::CorruptState {
                detail: "state machine unhealthy".to_owned(),
            });
        }
        inner
            .tablet
            .scan_local(spec, now)
            .map_err(|error| StateMachineFault::CorruptState {
                detail: format!("local scan failed: {error}"),
            })
    }

    /// Whether this replica is fenced for split/merge cutover (mutating
    /// proposes fail; reads still serve).
    pub async fn is_fenced(&self) -> bool {
        self.shared.lock().await.fenced
    }

    /// Sets or clears the cutover fence (idempotent; reconciler-driven).
    pub async fn set_fenced(&self, fenced: bool) {
        self.shared.lock().await.fenced = fenced;
    }

    /// Exports a topology copy of this replica's live state (split/merge
    /// base): all objects plus all dedup sessions.
    pub async fn export_topology_copy(&self) -> TopologyCopy {
        self.shared.lock().await.tablet.export_topology_copy()
    }

    /// Installs a topology copy into this (inactive) replica, replacing
    /// all objects and sessions. Caller must ensure the target serves
    /// nothing yet.
    pub async fn install_topology_copy(&self, copy: TopologyCopy) {
        self.shared.lock().await.tablet.install_topology_copy(copy);
    }

    /// Read-only leader-side proposal gate: the dedup verdict without
    /// any state change (the node pipeline calls this before preparing).
    ///
    /// # Errors
    ///
    /// Returns [`StateMachineFault`] when retained state is corrupt.
    pub async fn check_proposal(
        &self,
        marker: &kivi_types::MutationIdentity,
    ) -> Result<ProposalGate, StateMachineFault> {
        self.shared.lock().await.tablet.check_proposal(
            marker.client.session(),
            marker.client.seq(),
            marker.ack_floor,
        )
    }

    /// Prepares one operation against current state without mutating (the
    /// node pipeline calls this to materialize the proposal).
    ///
    /// # Errors
    ///
    /// Returns [`kivi_state::OpError`] for read-only opcode failures only.
    pub async fn prepare_operation(
        &self,
        op: &kivi_state::Operation,
        now: UnixMicros,
    ) -> Result<kivi_state::StorePrepared, kivi_state::OpError> {
        self.shared.lock().await.tablet.prepare_operation(op, now)
    }

    /// Returns a clone of the replicated control image (meaningful on
    /// the control group; empty elsewhere). Read-only observers —
    /// routing, admin, reconciler — use this; only committed control
    /// entries mutate it.
    pub async fn tablet_control(&self) -> kivi_control::ControlState {
        self.shared.lock().await.tablet.control().clone()
    }

    /// Peeks the representation class of one key's live base (the node
    /// capability gate uses this to refuse range patches against chunked
    /// bases before proposing).
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason when unhealthy.
    pub async fn peek_base(
        &self,
        key: &kivi_state::Key,
        now: UnixMicros,
    ) -> Result<RangeBase, String> {
        let inner = self.shared.lock().await;
        if !inner.healthy {
            return Err(format!(
                "state machine unhealthy: {}",
                inner.last_error.as_deref().unwrap_or("unknown")
            ));
        }
        Ok(match inner.tablet.store().get(key, now) {
            None => RangeBase::Absent,
            Some(object) => match object.value() {
                kivi_state::LogicalValue::Bytes(_) => RangeBase::Inline,
                kivi_state::LogicalValue::Chunked(_) => RangeBase::Chunked,
                // Defensive: consensus stores never hold fabric roots
                // (replicated IR carries bytes, never worker-local ids),
                // so any other value fails closed downstream.
                _ => RangeBase::NonBytes,
            },
        })
    }

    /// Returns read-only diagnostics (admin plane; never blocks apply).
    pub async fn status(&self) -> StateMachineStatus {
        let inner = self.shared.lock().await;
        StateMachineStatus {
            tablet: inner.tablet.tablet(),
            applied_index: inner.tablet.applied.map(|pointer| pointer.index),
            applied_commit: inner.tablet.applied_commit(),
            snapshot_index: inner
                .current_snapshot
                .as_ref()
                .and_then(|(meta, _)| meta.last_log_id.map(|id| id.index)),
            healthy: inner.healthy,
            last_error: inner.last_error.clone(),
        }
    }

    /// Serves a local read against applied state. The caller must have
    /// established a linearizable barrier first; unhealthy machines fail
    /// closed instead of serving possibly poisoned state.
    ///
    /// # Errors
    ///
    /// Returns [`StateMachineFault`] when unhealthy, on a mutating
    /// operation, or on read rejection.
    pub async fn read_local(
        &self,
        op: &Operation,
        now: UnixMicros,
    ) -> Result<OperationResult, StateMachineFault> {
        let inner = self.shared.lock().await;
        if !inner.healthy {
            return Err(StateMachineFault::Io {
                detail: format!(
                    "state machine unhealthy: {}",
                    inner.last_error.as_deref().unwrap_or("unknown")
                ),
            });
        }
        inner.tablet.read_local(op, now)
    }

    /// Arms fault injection for snapshot/install persistence (task S
    /// storage-health path): the next snapshot build or install fails its
    /// durability barrier, latches unhealthy, and blocks purge. Test and
    /// chaos hooks only; never armed in normal serving.
    pub async fn set_snapshot_write_failure(&self, fail: bool) {
        self.shared.lock().await.inject_snapshot_write_failure = fail;
    }

    /// Lists live chunked roots (manifest + length) for startup
    /// verification and repair. Returns empty when no large values are
    /// live; never fails (an unhealthy machine yields its last known
    /// roots for best-effort repair, never an error).
    pub async fn chunked_roots(&self) -> Vec<(kivi_types::ManifestId, u64)> {
        let inner = self.shared.lock().await;
        inner
            .tablet
            .store()
            .snapshot_entries()
            .into_iter()
            .filter_map(|(_key, object)| match object.value() {
                kivi_state::LogicalValue::Chunked(chunked) => {
                    Some((chunked.manifest, chunked.logical_len))
                }
                _ => None,
            })
            .collect()
    }

    /// Marks the replica unhealthy with a latched detail (startup
    /// verification found applied roots referencing missing sidecars).
    /// Reads fail closed; a later repair (sidecar fetch + re-verify, or a
    /// snapshot install carrying the missing bulk) heals via the snapshot
    /// path, which resets health on validated install. Never blocks
    /// consensus transport: the replica stays reachable for repair.
    pub async fn mark_unhealthy(&self, detail: String) {
        let mut inner = self.shared.lock().await;
        inner.healthy = false;
        inner.last_error = Some(detail);
    }

    /// Returns the voter set of the currently held membership (snapshot
    /// or applied-file ground truth after open; live tracking after).
    /// The node restart path uses this when the retained log holds no
    /// membership entry (purged past, with the snapshot grounding it).
    pub async fn member_voters(&self) -> std::collections::BTreeSet<u64> {
        self.shared.lock().await.tablet.membership_voters()
    }

    /// Returns the installed snapshot base index (`None` before the first
    /// snapshot). The node layer may purge the log only through this
    /// index (task V).
    pub async fn snapshotted_index(&self) -> Option<u64> {
        self.shared
            .lock()
            .await
            .current_snapshot
            .as_ref()
            .and_then(|(meta, _)| meta.last_log_id.map(|id| id.index))
    }
}

/// Latches the machine unhealthy with its failure detail (divergence and
/// durability faults halt this replica loudly; the admin plane reports
/// the latch while the process exits or awaits repair).
fn poison(inner: &mut StateMachineInner, detail: String) {
    inner.healthy = false;
    inner.last_error = Some(detail);
}

fn sm_io_error(fault: &StateMachineFault) -> std::io::Error {
    std::io::Error::other(format!("state machine failed: {fault}"))
}

fn snapshot_io_error(fault: &StateMachineFault) -> std::io::Error {
    std::io::Error::other(format!("state machine snapshot failed: {fault}"))
}

/// Plans one crash-safe snapshot seal: image bytes plus the file names
/// they must land under. Pure computation under the machine lock; the
/// blocking file work runs afterwards on a blocking thread (never the
/// Compio reactor), and the caller installs the result back under the
/// lock. Splitting plan/execute/install keeps `fsync` off the reactor
/// while the machine lock stays an async mutex.
struct SealPlan {
    bytes: Vec<u8>,
    image_path: PathBuf,
    applied_path: PathBuf,
    applied_body: Vec<u8>,
    current_path: PathBuf,
    current_body: Vec<u8>,
}

fn plan_seal(
    inner: &StateMachineInner,
    bytes: Vec<u8>,
    index: u64,
    term: u64,
) -> Result<SealPlan, StateMachineFault> {
    use StateMachineFault as Fault;
    if inner.inject_snapshot_write_failure {
        return Err(Fault::Io {
            detail: "injected snapshot-write failure".to_owned(),
        });
    }
    let name = snapshot_file_name(index, term);
    let applied_body =
        encode_applied_file_body(inner.tablet.applied, &inner.tablet.membership, &name);
    Ok(SealPlan {
        bytes,
        image_path: inner.snapshots_dir.join(&name),
        applied_path: inner.applied_path.clone(),
        applied_body: frame_image(APPLIED_MAGIC, APPLIED_VERSION, &applied_body),
        current_path: inner.current_path.clone(),
        current_body: name.into_bytes(),
    })
}

/// Executes a seal plan on the calling thread: image file, then the
/// applied file naming it, then the CURRENT switch — each crash-safe
/// (sibling temp file, file sync, atomic rename). A crash leaves the old
/// complete image or the new complete image, never a torn one.
fn execute_seal(plan: &SealPlan) -> Result<(), StateMachineFault> {
    use StateMachineFault as Fault;
    write_crash_safe(&plan.image_path, &plan.bytes).map_err(|error| Fault::Io {
        detail: format!("snapshot image not durable: {error}"),
    })?;
    write_crash_safe(&plan.applied_path, &plan.applied_body).map_err(|error| Fault::Io {
        detail: format!("applied file not durable: {error}"),
    })?;
    write_crash_safe(plan.current_path.as_path(), &plan.current_body).map_err(|error| {
        Fault::Io {
            detail: format!("snapshot CURRENT not durable: {error}"),
        }
    })?;
    Ok(())
}

/// Runs a seal plan off the reactor: file `fsync`s never stall async
/// progress (the old Tokio island used `block_in_place` for the same
/// reason; on single-threaded Compio the work moves to a blocking
/// thread instead).
async fn seal_off_reactor(plan: SealPlan) -> Result<(), StateMachineFault> {
    compio::runtime::spawn_blocking(move || execute_seal(&plan))
        .await
        .map_err(|_| StateMachineFault::Io {
            detail: "snapshot seal thread failed".to_owned(),
        })?
}

/// Encodes the applied-file body: applied pointer, membership, and the
/// snapshot file name sealed with them.
fn encode_applied_file_body(
    applied: Option<AppliedPointer>,
    membership: &StoredMembershipOf<KiviTypeConfig>,
    snapshot_name: &str,
) -> Vec<u8> {
    let mut out = encode_applied_body(applied, membership);
    encode_bytes(&mut out, snapshot_name.as_bytes());
    out
}

/// One applied file: the pointer, the membership grounding it, and the
/// snapshot file name sealed with them.
type AppliedFile = (
    Option<AppliedPointer>,
    StoredMembershipOf<KiviTypeConfig>,
    String,
);

/// Decodes the applied-file body plus its snapshot name.
fn decode_applied_file_body(body: &[u8]) -> Result<AppliedFile, ImageFault> {
    let ((applied, membership), used) = decode_applied_prefix(body)?;
    let rest = body.get(used..).ok_or(ImageFault::Truncated)?;
    let (name, consumed) = decode_byte_vec(rest).map_err(|_| ImageFault::BadAddress)?;
    if used + consumed != body.len() {
        return Err(ImageFault::TrailingBytes);
    }
    let name = core::str::from_utf8(&name).map_err(|_| ImageFault::BadAddress)?;
    Ok((applied, membership, name.to_owned()))
}

/// Snapshot builder serving the sealed image bytes.
#[derive(Debug)]
pub struct SealedSnapshotBuilder {
    meta: SnapshotMetaOf<KiviTypeConfig>,
    data: Vec<u8>,
}

impl SealedSnapshotBuilder {
    /// Returns the sealed snapshot metadata (base log id, membership,
    /// checkpoint identity).
    #[must_use]
    pub const fn meta(&self) -> &SnapshotMetaOf<KiviTypeConfig> {
        &self.meta
    }
}

impl RaftSnapshotBuilder<KiviTypeConfig> for SealedSnapshotBuilder {
    type SnapshotData = Cursor<Vec<u8>>;

    #[allow(clippy::unused_async_trait_impl)]
    async fn build_snapshot(
        &mut self,
    ) -> Result<SnapshotOf<KiviTypeConfig, Cursor<Vec<u8>>>, std::io::Error> {
        Ok(openraft::storage::Snapshot {
            meta: self.meta.clone(),
            snapshot: Cursor::new(self.data.clone()),
        })
    }
}

impl RaftStateMachine<KiviTypeConfig> for ReplicatedStateMachine {
    type SnapshotBuilder = SealedSnapshotBuilder;
    type SnapshotData = Cursor<Vec<u8>>;

    async fn applied_state(
        &mut self,
    ) -> Result<
        (
            Option<LogIdOf<KiviTypeConfig>>,
            StoredMembershipOf<KiviTypeConfig>,
        ),
        std::io::Error,
    > {
        let inner = self.shared.lock().await;
        Ok((
            inner.tablet.applied.map(AppliedPointer::openraft),
            inner.tablet.membership.clone(),
        ))
    }

    async fn apply<Strm>(&mut self, entries: Strm) -> Result<(), std::io::Error>
    where
        Strm: futures::Stream<
                Item = Result<openraft::storage::EntryResponder<KiviTypeConfig>, std::io::Error>,
            > + Unpin
            + openraft::OptionalSend,
    {
        use futures::StreamExt as _;
        let mut entries = entries;
        // The whole batch applies under one lock acquisition: apply is
        // pure in-memory work (no awaits inside), so the reactor never
        // stalls here. Each entry's response streams back through its
        // `ApplyResponder` immediately — no `Vec` buffering — and
        // `client_write` on the leader resolves entry by entry.
        let mut inner = self.shared.lock().await;
        if !inner.healthy {
            let fault = StateMachineFault::Io {
                detail: format!(
                    "state machine unhealthy: {}",
                    inner.last_error.as_deref().unwrap_or("unknown")
                ),
            };
            return Err(sm_io_error(&fault));
        }
        while let Some(item) = entries.next().await {
            let (entry, responder) = item?;
            let outcome = match inner.tablet.apply_entry(entry.log_id, &entry.payload) {
                Ok(outcome) => outcome,
                Err(fault) => {
                    poison(&mut inner, fault.to_string());
                    return Err(sm_io_error(&fault));
                }
            };
            if let Some(responder) = responder {
                responder.send(outcome);
            }
        }
        Ok(())
    }

    async fn try_create_snapshot_builder(&mut self, _force: bool) -> Option<Self::SnapshotBuilder> {
        // Sealing happens here (not in `build_snapshot`): the bytes
        // `OpenRaft` streams must already be durable, otherwise a crash
        // after streaming could leave followers ahead of any durable
        // image on the leader. A poisoned machine still seals (the node
        // layer blocks purge and reads while unhealthy, and the seal
        // records the latched state for forensics, never for serving).
        // Kivi always seals synchronously (`force` is moot): deferring
        // would only delay replication that already waits.
        let (bytes, applied, membership) = {
            let inner = self.shared.lock().await;
            (
                inner.tablet.encode_snapshot(),
                inner.tablet.applied,
                inner.tablet.membership.clone(),
            )
        };
        let (index, term) = applied.map_or((0, 0), |pointer| (pointer.index, pointer.term));
        // The plan borrows nothing: `inject_snapshot_write_failure` is
        // read under the lock, file work runs off-reactor below.
        let plan = {
            let inner = self.shared.lock().await;
            match plan_seal(&inner, bytes.clone(), index, term) {
                Ok(plan) => plan,
                Err(fault) => {
                    drop(inner);
                    let mut inner = self.shared.lock().await;
                    poison(&mut inner, fault.to_string());
                    return Some(SealedSnapshotBuilder {
                        meta: openraft::storage::SnapshotMeta {
                            last_log_id: applied.map(AppliedPointer::openraft),
                            last_membership: membership,
                        },
                        data: bytes,
                    });
                }
            }
        };
        let meta = openraft::storage::SnapshotMeta {
            last_log_id: applied.map(AppliedPointer::openraft),
            last_membership: membership,
        };
        match seal_off_reactor(plan).await {
            Ok(()) => {
                let mut inner = self.shared.lock().await;
                inner.current_snapshot = Some((meta.clone(), bytes.clone()));
            }
            Err(fault) => {
                let mut inner = self.shared.lock().await;
                poison(&mut inner, fault.to_string());
            }
        }
        Some(SealedSnapshotBuilder { meta, data: bytes })
    }

    async fn get_snapshot_builder(&mut self) -> Self::SnapshotBuilder {
        self.try_create_snapshot_builder(true)
            .await
            .expect("kivi always seals synchronously")
    }

    async fn install_snapshot(
        &mut self,
        meta: &SnapshotMetaOf<KiviTypeConfig>,
        snapshot: Cursor<Vec<u8>>,
    ) -> Result<(), std::io::Error> {
        let bytes = snapshot.into_inner();
        let (namespace, tablet) = {
            let inner = self.shared.lock().await;
            (inner.tablet.namespace, inner.tablet.tablet())
        };
        let decoded = ReplicatedTablet::decode_snapshot(&bytes, namespace, tablet)
            .map_err(|fault| snapshot_io_error(&fault));
        let decoded = match decoded {
            Ok(decoded) => decoded,
            Err(error) => {
                let mut inner = self.shared.lock().await;
                poison(&mut inner, error.to_string());
                return Err(error);
            }
        };
        // Fencing: the image authority must equal this replica's.
        {
            let inner = self.shared.lock().await;
            if decoded.epoch != inner.tablet.authority.epoch()
                || decoded.guard != inner.tablet.authority.guard()
            {
                let fault = StateMachineFault::AuthorityMismatch {
                    detail: format!(
                        "snapshot authority epoch {} guard {} != replica {}",
                        decoded.epoch, decoded.guard, inner.tablet.authority
                    ),
                };
                drop(inner);
                let mut inner = self.shared.lock().await;
                poison(&mut inner, fault.to_string());
                return Err(snapshot_io_error(&fault));
            }
        }
        // Monotonicity: installs never move the applied base backwards.
        let incoming = decoded.applied.map_or(0, |pointer| pointer.index);
        let current = self
            .shared
            .lock()
            .await
            .tablet
            .applied()
            .map_or(0, |pointer| pointer.index);
        if incoming < current {
            let fault = StateMachineFault::CorruptImage {
                detail: format!("stale snapshot install: base {incoming} < applied {current}"),
            };
            let mut inner = self.shared.lock().await;
            poison(&mut inner, fault.to_string());
            return Err(snapshot_io_error(&fault));
        }
        // Durability first: the image, applied file, and CURRENT switch
        // seal off-reactor before the in-memory swap. Never mark
        // installed before its required state is durable.
        let plan = {
            let inner = self.shared.lock().await;
            match plan_seal(
                &inner,
                bytes.clone(),
                incoming,
                decoded.applied.map_or(0, |pointer| pointer.term),
            ) {
                Ok(plan) => plan,
                Err(fault) => {
                    drop(inner);
                    let mut inner = self.shared.lock().await;
                    poison(&mut inner, fault.to_string());
                    return Err(snapshot_io_error(&fault));
                }
            }
        };
        if let Err(fault) = seal_off_reactor(plan).await {
            let mut inner = self.shared.lock().await;
            poison(&mut inner, fault.to_string());
            return Err(snapshot_io_error(&fault));
        }
        let mut inner = self.shared.lock().await;
        inner.tablet.install_decoded(decoded);
        inner.current_snapshot = Some((meta.clone(), bytes));
        // A validated leader snapshot heals a poisoned machine: every
        // byte of state now came from the installed image, so the latch
        // lifts and reads resume. (Log-side divergence, if any, is
        // `OpenRaft`'s to truncate, not the state machine's.)
        inner.healthy = true;
        inner.last_error = None;
        Ok(())
    }

    async fn get_current_snapshot(
        &mut self,
    ) -> Result<Option<SnapshotOf<KiviTypeConfig, Cursor<Vec<u8>>>>, std::io::Error> {
        let inner = self.shared.lock().await;
        Ok(inner
            .current_snapshot
            .clone()
            .map(|(meta, data)| openraft::storage::Snapshot {
                meta,
                snapshot: Cursor::new(data),
            }))
    }
}

/// One applied prefix: the pointer plus the membership grounding it.
type AppliedPrefix = (Option<AppliedPointer>, StoredMembershipOf<KiviTypeConfig>);

/// Decodes the applied prefix, returning the consumed length so the file
/// variant can locate its trailing snapshot-name blob.
fn decode_applied_prefix(body: &[u8]) -> Result<(AppliedPrefix, usize), ImageFault> {
    use ImageFault as Fault;
    if body.is_empty() {
        return Err(Fault::Truncated);
    }
    let mut at: usize;
    let applied = match body[0] {
        0 => {
            at = 1;
            None
        }
        1 => {
            if body.len() < 1 + 24 {
                return Err(Fault::Truncated);
            }
            let term = u64::from_le_bytes(body[1..9].try_into().unwrap_or([0; 8]));
            let leader = u64::from_le_bytes(body[9..17].try_into().unwrap_or([0; 8]));
            let index = u64::from_le_bytes(body[17..25].try_into().unwrap_or([0; 8]));
            at = 25;
            Some(AppliedPointer::new(term, leader, index))
        }
        _ => return Err(Fault::BadValueTag { tag: body[0] }),
    };
    if body.len() <= at {
        return Err(Fault::Truncated);
    }
    let log_id = match body[at] {
        0 => {
            at += 1;
            None
        }
        1 => {
            if body.len() < at + 1 + 24 {
                return Err(Fault::Truncated);
            }
            let term = u64::from_le_bytes(body[at + 1..at + 9].try_into().unwrap_or([0; 8]));
            let leader = u64::from_le_bytes(body[at + 9..at + 17].try_into().unwrap_or([0; 8]));
            let index = u64::from_le_bytes(body[at + 17..at + 25].try_into().unwrap_or([0; 8]));
            at += 25;
            Some(openraft::LogId::new(
                openraft::impls::leader_id_adv::LeaderId::new(term, leader),
                index,
            ))
        }
        tag => return Err(Fault::BadValueTag { tag }),
    };
    let (membership, used) = decode_membership(&body[at..], log_id)?;
    at += used;
    Ok(((applied, membership), at))
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use kivi_state::{Key, Mutation, MutationEnvelope, ObjectVersion, OperationResult};
    use kivi_types::{
        NamespaceId, RequestIdentity, RequestSeq, SessionId, TabletAuthority, TabletEpoch,
        TabletId, UnixMicros, WriteGuardGeneration,
    };
    use openraft::storage::RaftStateMachine;

    use super::{
        ReplicatedStateMachine, ReplicatedTablet, StateMachineFault, commit_of_index,
        index_of_commit,
    };
    use crate::config::KiviTypeConfig;
    use crate::mutation::{ReplicatedMutation, ReplicatedOutcome};
    use openraft::vote::RaftLeaderId as _;
    use std::future::Future;

    const NS: NamespaceId = NamespaceId::from_u64(1);
    const TABLET: TabletId = TabletId::from_u64(9);
    const NOW: UnixMicros = UnixMicros::from_micros(1_000_000);

    fn authority() -> TabletAuthority {
        TabletAuthority::new(TABLET, TabletEpoch::INITIAL, WriteGuardGeneration::INITIAL)
    }

    fn tablet() -> ReplicatedTablet {
        ReplicatedTablet::new(NS, TABLET, authority())
    }

    fn command(
        session: u128,
        seq: u64,
        mutation: Mutation,
        expected: OperationResult,
    ) -> crate::command::ConsensusCommand {
        crate::command::ConsensusCommand::Tablet(ReplicatedMutation::new(
            MutationEnvelope::new(
                NS,
                authority(),
                RequestIdentity::new(SessionId::from_u128(session), RequestSeq::from_u64(seq)),
                None,
                mutation,
            ),
            expected,
            NOW,
            RequestSeq::from_u64(0),
        ))
    }

    fn log_id(index: u64) -> openraft::type_config::alias::LogIdOf<crate::config::KiviTypeConfig> {
        use openraft::impls::leader_id_adv::LeaderId;
        openraft::LogId::new(LeaderId::new(1, 1), index)
    }

    fn block_on<F, T>(future: F) -> T
    where
        F: Future<Output = T>,
    {
        compio::runtime::Runtime::new()
            .expect("compio test runtime")
            .block_on(future)
    }

    #[test]
    fn commit_position_is_the_raft_index_plus_one() {
        // Index 0 is a real entry; the +1 preserves UNASSIGNED.
        assert_eq!(commit_of_index(0), kivi_types::CommitPosition::FIRST);
        assert_eq!(
            commit_of_index(41),
            kivi_types::CommitPosition::from_u64(42)
        );
        assert_eq!(index_of_commit(kivi_types::CommitPosition::FIRST), Some(0));
        assert_eq!(
            index_of_commit(kivi_types::CommitPosition::from_u64(42)),
            Some(41)
        );
        assert_eq!(
            index_of_commit(kivi_types::CommitPosition::UNASSIGNED),
            None
        );
    }

    /// Applies one command at `index`, panicking on divergence (test
    /// shorthand: every case below commits known-good history).
    fn apply(
        tablet: &mut ReplicatedTablet,
        index: u64,
        cmd: crate::command::ConsensusCommand,
    ) -> ReplicatedOutcome {
        tablet
            .apply_entry(log_id(index), &openraft::EntryPayload::Normal(cmd))
            .expect("test history applies")
    }

    #[test]
    fn apply_serves_transaction_prepare_and_finalize() {
        use kivi_state::{TxnExpect, TxnId, TxnWriteKind};
        let mut tablet = tablet();
        let txn = TxnId::derive(7, 1, 0);
        // Prepare reserves without touching user state.
        let prepare = Mutation::TxnPrepare {
            txn,
            coordinator: TABLET.as_u64(),
            key: Key::from("k"),
            expect: TxnExpect::Absent,
            write: TxnWriteKind::Put(bytes::Bytes::from_static(b"v")),
            digest: [0xD1; 32],
        };
        let outcome = apply(
            &mut tablet,
            0,
            command(1, 1, prepare, OperationResult::TxnPrepared),
        );
        let _ = outcome;
        assert_eq!(tablet.pending_intent_count(), 1);
        assert!(tablet.store().get(&Key::from("k"), NOW).is_none());
        // A normal mutation against the reserved key diverges loudly at
        // apply (prepare blocks it first on every honest path).
        assert!(
            tablet
                .store
                .apply(
                    &Mutation::PutBytes {
                        key: Key::from("k"),
                        value: bytes::Bytes::from_static(b"race"),
                    },
                    NOW,
                )
                .is_err()
        );
        // Finalize-commit applies the prepared write.
        let finalize = Mutation::TxnFinalize {
            txn,
            key: Key::from("k"),
            commit: true,
            digest: [0xD1; 32],
        };
        apply(
            &mut tablet,
            1,
            command(
                1,
                2,
                finalize,
                OperationResult::TxnFinalized {
                    applied: true,
                    version: Some(ObjectVersion::FIRST),
                },
            ),
        );
        assert_eq!(tablet.pending_intent_count(), 0);
        assert!(tablet.store().get(&Key::from("k"), NOW).is_some());
    }

    #[test]
    fn snapshot_round_trips_intents_and_ordered_flag() {
        use kivi_state::{TxnExpect, TxnId, TxnWriteKind};
        let mut tablet = tablet();
        tablet.set_ordered_indexing(true);
        let txn = TxnId::derive(3, 1, 0);
        apply(
            &mut tablet,
            0,
            command(
                1,
                1,
                Mutation::TxnPrepare {
                    txn,
                    coordinator: TABLET.as_u64(),
                    key: Key::from("k"),
                    expect: TxnExpect::Any,
                    write: TxnWriteKind::Put(bytes::Bytes::from_static(b"v")),
                    digest: [0xD1; 32],
                },
                OperationResult::TxnPrepared,
            ),
        );
        let image = tablet.encode_snapshot();
        let decoded = ReplicatedTablet::decode_snapshot(&image, NS, TABLET).expect("decodes");
        assert!(decoded.ordered_index);
        assert_eq!(decoded.intents.len(), 1);
        assert_eq!(decoded.intents[0].id, txn);
        // Install restores intents verbatim and rebuilds the index.
        let mut rebuilt = ReplicatedTablet::new(NS, TABLET, authority());
        rebuilt.install_decoded(decoded);
        assert!(rebuilt.ordered_index_enabled());
        assert_eq!(rebuilt.pending_intent_count(), 1);
        assert!(rebuilt.store.verify_ordered_index());
    }

    #[test]
    fn scan_serves_ordered_pages_locally() {
        use kivi_state::{ScanDirection, ScanProjection, ScanSpec};
        let mut tablet = tablet();
        apply(
            &mut tablet,
            0,
            command(
                1,
                1,
                Mutation::PutBytes {
                    key: Key::from("b"),
                    value: bytes::Bytes::from_static(b"2"),
                },
                OperationResult::Stored {
                    version: ObjectVersion::FIRST,
                },
            ),
        );
        apply(
            &mut tablet,
            1,
            command(
                1,
                2,
                Mutation::PutBytes {
                    key: Key::from("a"),
                    value: bytes::Bytes::from_static(b"1"),
                },
                OperationResult::Stored {
                    version: ObjectVersion::FIRST,
                },
            ),
        );
        let spec = ScanSpec::new(
            None,
            None,
            ScanDirection::Forward,
            100,
            1 << 20,
            ScanProjection::KeysOnly,
        )
        .expect("spec");
        let page = tablet.scan_local(&spec, NOW).expect("scan");
        assert!(page.exhausted);
        assert_eq!(page.entries.len(), 2);
        assert_eq!(page.entries[0].key, Key::from("a"));
        assert!(tablet.store.verify_ordered_index());
    }

    #[test]
    fn apply_serves_set_delete_and_reads() {
        let mut tablet = tablet();
        let outcome = apply(
            &mut tablet,
            0,
            command(
                1,
                1,
                Mutation::PutBytes {
                    key: Key::from("k"),
                    value: bytes::Bytes::from_static(b"v"),
                },
                OperationResult::Stored {
                    version: ObjectVersion::FIRST,
                },
            ),
        );
        assert_eq!(
            outcome,
            ReplicatedOutcome::new(OperationResult::Stored {
                version: ObjectVersion::FIRST,
            })
        );
        assert_eq!(tablet.applied_commit(), commit_of_index(0));
        let exists = tablet
            .read_local(
                &kivi_state::Operation::Exists {
                    key: Key::from("k"),
                },
                NOW,
            )
            .expect("exists reads");
        assert_eq!(exists, OperationResult::Exists(true));
        let get = tablet
            .read_local(
                &kivi_state::Operation::Get {
                    key: Key::from("k"),
                },
                NOW,
            )
            .expect("get reads");
        assert_eq!(
            get,
            OperationResult::Value(Some(bytes::Bytes::from_static(b"v")))
        );
        let outcome = apply(
            &mut tablet,
            1,
            command(
                1,
                2,
                Mutation::Delete {
                    key: Key::from("k"),
                },
                OperationResult::Deleted { existed: true },
            ),
        );
        assert_eq!(
            outcome,
            ReplicatedOutcome::new(OperationResult::Deleted { existed: true })
        );
        assert_eq!(tablet.applied_commit(), commit_of_index(1));
    }

    #[test]
    fn apply_serves_counter_expiry_and_conditional() {
        let mut tablet = tablet();
        apply(
            &mut tablet,
            0,
            command(
                7,
                1,
                Mutation::CounterAdd {
                    key: Key::from("n"),
                    delta: 41,
                },
                OperationResult::CounterUpdated {
                    value: 41,
                    version: ObjectVersion::FIRST,
                },
            ),
        );
        apply(
            &mut tablet,
            1,
            command(
                7,
                2,
                Mutation::SetExpiry {
                    key: Key::from("n"),
                    expiry: kivi_types::Expiry::at(UnixMicros::from_micros(2_000_000)),
                },
                OperationResult::ExpirySet { applied: true },
            ),
        );
        apply(
            &mut tablet,
            2,
            command(
                7,
                3,
                Mutation::SetExpiry {
                    key: Key::from("n"),
                    expiry: kivi_types::Expiry::NEVER,
                },
                OperationResult::ExpiryPersisted { removed: true },
            ),
        );
        // Conditional SET arrives materialized (resolved expiry stamp).
        apply(
            &mut tablet,
            3,
            command(
                8,
                1,
                Mutation::PutBytesWithExpiry {
                    key: Key::from("c"),
                    value: bytes::Bytes::from_static(b"v"),
                    expiry: kivi_types::Expiry::NEVER,
                },
                OperationResult::ConditionalSet {
                    applied: true,
                    version: Some(ObjectVersion::FIRST),
                },
            ),
        );
        assert_eq!(tablet.applied_commit(), commit_of_index(3));
    }

    #[test]
    fn apply_serves_splice_preserving_surroundings() {
        let mut tablet = tablet();
        apply(
            &mut tablet,
            0,
            command(
                8,
                1,
                Mutation::PutBytes {
                    key: Key::from("s"),
                    value: bytes::Bytes::from_static(b" offen"),
                },
                OperationResult::Stored {
                    version: ObjectVersion::FIRST,
                },
            ),
        );
        // SETRANGE splice preserves bytes around the patch.
        apply(
            &mut tablet,
            1,
            command(
                8,
                2,
                Mutation::SpliceBytes {
                    key: Key::from("s"),
                    offset: 0,
                    patch: bytes::Bytes::from_static(b"c"),
                },
                OperationResult::Stored {
                    version: ObjectVersion::from_u64(2),
                },
            ),
        );
        let get = tablet
            .read_local(
                &kivi_state::Operation::Get {
                    key: Key::from("s"),
                },
                NOW,
            )
            .expect("reads spliced");
        assert_eq!(
            get,
            OperationResult::Value(Some(bytes::Bytes::from_static(b"coffen")))
        );
        assert_eq!(tablet.applied_commit(), commit_of_index(1));
    }

    #[test]
    fn same_identity_retry_returns_the_original_without_reexecuting() {
        let mut tablet = tablet();
        let first = command(
            0x5E55,
            1,
            Mutation::CounterAdd {
                key: Key::from("n"),
                delta: 1,
            },
            OperationResult::CounterUpdated {
                value: 1,
                version: ObjectVersion::FIRST,
            },
        );
        let outcome = tablet
            .apply_entry(log_id(0), &openraft::EntryPayload::Normal(first))
            .expect("first applies");
        assert_eq!(
            outcome,
            ReplicatedOutcome::new(OperationResult::CounterUpdated {
                value: 1,
                version: ObjectVersion::FIRST,
            })
        );
        // Lost-response retry: the identical command commits again (new
        // leader, new index) and must NOT execute twice.
        let retry = command(
            0x5E55,
            1,
            Mutation::CounterAdd {
                key: Key::from("n"),
                delta: 1,
            },
            OperationResult::CounterUpdated {
                value: 1,
                version: ObjectVersion::FIRST,
            },
        );
        let outcome = tablet
            .apply_entry(log_id(1), &openraft::EntryPayload::Normal(retry))
            .expect("retry hits dedup");
        assert_eq!(
            outcome,
            ReplicatedOutcome::new(OperationResult::CounterUpdated {
                value: 1,
                version: ObjectVersion::FIRST,
            }),
            "retry returns the original outcome"
        );
        // Counter proves single execution.
        let get = tablet
            .read_local(
                &kivi_state::Operation::CounterGet {
                    key: Key::from("n"),
                },
                NOW,
            )
            .expect("reads");
        assert_eq!(get, OperationResult::Counter(Some(1)));
        // The applied cursor still advances over the dedup hit.
        assert_eq!(tablet.applied_commit(), commit_of_index(1));
    }

    #[test]
    fn verification_mismatch_fails_loudly() {
        let mut tablet = tablet();
        let lying = command(
            1,
            1,
            Mutation::CounterAdd {
                key: Key::from("n"),
                delta: 1,
            },
            OperationResult::CounterUpdated {
                value: 999,
                version: ObjectVersion::FIRST,
            },
        );
        let fault = tablet
            .apply_entry(log_id(0), &openraft::EntryPayload::Normal(lying))
            .expect_err("wrong expectation diverges");
        assert!(
            matches!(fault, StateMachineFault::Divergence { .. }),
            "loud divergence, got {fault:?}"
        );
    }

    #[test]
    fn expired_identity_at_apply_is_divergence() {
        let mut tablet = tablet();
        // Unknown sessions always admit (mirroring the single-node
        // engine): the floor only constrains sessions the replica has
        // actually seen.
        let fresh = command(
            1,
            1,
            Mutation::Delete {
                key: Key::from("k"),
            },
            OperationResult::Deleted { existed: false },
        );
        let fresh_inner = fresh.as_tablet().expect("tablet command").clone();
        let fresh = crate::command::ConsensusCommand::Tablet(ReplicatedMutation::new(
            fresh_inner.envelope().clone(),
            fresh_inner.expected().clone(),
            fresh_inner.now(),
            RequestSeq::from_u64(1),
        ));
        tablet
            .apply_entry(log_id(0), &openraft::EntryPayload::Normal(fresh))
            .expect("unknown session admits");
        // Advance the session floor past sequence 3 with an ack: a later
        // committed entry for sequence 3 means floor divergence (the
        // leader's pre-proposal check would have rejected it).
        let advance = command(
            1,
            5,
            Mutation::Delete {
                key: Key::from("k"),
            },
            OperationResult::Deleted { existed: false },
        );
        let advance_inner = advance.as_tablet().expect("tablet command").clone();
        let advance = crate::command::ConsensusCommand::Tablet(ReplicatedMutation::new(
            advance_inner.envelope().clone(),
            advance_inner.expected().clone(),
            advance_inner.now(),
            RequestSeq::from_u64(4),
        ));
        tablet
            .apply_entry(log_id(1), &openraft::EntryPayload::Normal(advance))
            .expect("floor advances");
        let stale = command(
            1,
            3,
            Mutation::Delete {
                key: Key::from("k"),
            },
            OperationResult::Deleted { existed: false },
        );
        let fault = tablet
            .apply_entry(log_id(2), &openraft::EntryPayload::Normal(stale))
            .expect_err("expired identity fails");
        assert!(
            matches!(fault, StateMachineFault::ExpiredIdentity { .. }),
            "got {fault:?}"
        );
    }

    #[test]
    fn misrouted_and_stale_envelopes_fail() {
        let mut tablet = tablet();
        // Foreign tablet.
        let foreign_authority = TabletAuthority::new(
            TabletId::from_u64(10),
            TabletEpoch::INITIAL,
            WriteGuardGeneration::INITIAL,
        );
        let foreign = crate::command::ConsensusCommand::Tablet(ReplicatedMutation::new(
            MutationEnvelope::new(
                NS,
                foreign_authority,
                RequestIdentity::new(SessionId::from_u128(1), RequestSeq::from_u64(1)),
                None,
                Mutation::Delete {
                    key: Key::from("k"),
                },
            ),
            OperationResult::Deleted { existed: false },
            NOW,
            RequestSeq::from_u64(0),
        ));
        assert!(matches!(
            tablet.apply_entry(log_id(0), &openraft::EntryPayload::Normal(foreign)),
            Err(StateMachineFault::Misrouted { .. })
        ));
        // Stale epoch on the right tablet.
        let stale_authority = TabletAuthority::new(
            TABLET,
            TabletEpoch::from_u64(999),
            WriteGuardGeneration::INITIAL,
        );
        let stale = crate::command::ConsensusCommand::Tablet(ReplicatedMutation::new(
            MutationEnvelope::new(
                NS,
                stale_authority,
                RequestIdentity::new(SessionId::from_u128(1), RequestSeq::from_u64(2)),
                None,
                Mutation::Delete {
                    key: Key::from("k"),
                },
            ),
            OperationResult::Deleted { existed: false },
            NOW,
            RequestSeq::from_u64(0),
        ));
        assert!(matches!(
            tablet.apply_entry(log_id(1), &openraft::EntryPayload::Normal(stale)),
            Err(StateMachineFault::AuthorityMismatch { .. })
        ));
    }

    #[test]
    fn blank_and_membership_entries_advance_applied_without_outcomes() {
        let mut tablet = tablet();
        let blank = tablet
            .apply_entry(log_id(0), &openraft::EntryPayload::Blank)
            .expect("blank applies");
        assert_eq!(blank, ReplicatedOutcome::none());
        let membership = openraft::Membership::new(
            vec![BTreeSet::from([1u64, 2, 3])],
            BTreeMap::from([
                (1u64, openraft::BasicNode::new("127.0.0.1:9001".to_owned())),
                (2u64, openraft::BasicNode::new("127.0.0.1:9002".to_owned())),
                (3u64, openraft::BasicNode::new("127.0.0.1:9003".to_owned())),
            ]),
        )
        .expect("test membership validates");
        let installed = tablet
            .apply_entry(log_id(1), &openraft::EntryPayload::Membership(membership))
            .expect("membership installs");
        assert_eq!(installed, ReplicatedOutcome::none());
        assert_eq!(tablet.applied_commit(), commit_of_index(1));
    }

    #[test]
    fn snapshot_images_round_trip_with_dedup_and_membership() {
        let mut tablet = tablet();
        tablet
            .apply_entry(log_id(0), &openraft::EntryPayload::Blank)
            .expect("blank");
        let membership = openraft::Membership::new(
            vec![BTreeSet::from([1u64, 2, 3])],
            BTreeMap::from([
                (1u64, openraft::BasicNode::new("n1".to_owned())),
                (2u64, openraft::BasicNode::new("n2".to_owned())),
                (3u64, openraft::BasicNode::new("n3".to_owned())),
            ]),
        )
        .expect("test membership validates");
        tablet
            .apply_entry(log_id(1), &openraft::EntryPayload::Membership(membership))
            .expect("membership");
        let set = command(
            1,
            1,
            Mutation::PutBytes {
                key: Key::from("k"),
                value: bytes::Bytes::from_static(b"v"),
            },
            OperationResult::Stored {
                version: ObjectVersion::FIRST,
            },
        );
        tablet
            .apply_entry(log_id(2), &openraft::EntryPayload::Normal(set))
            .expect("set");
        let image = tablet.encode_snapshot();
        let decoded = ReplicatedTablet::decode_snapshot(&image, NS, TABLET).expect("decodes");
        assert_eq!(decoded.applied.map(|pointer| pointer.index), Some(2));
        assert_eq!(decoded.objects.len(), 1);
        assert_eq!(decoded.sessions.len(), 1);
        // Re-encoding the installed state is byte-identical (sorted,
        // deterministic).
        let mut rebuilt = ReplicatedTablet::new(NS, TABLET, authority());
        rebuilt.install_decoded(decoded);
        assert_eq!(rebuilt.encode_snapshot(), image);
        // Foreign identity fails loudly, never cross-installs.
        assert!(matches!(
            ReplicatedTablet::decode_snapshot(&image, NS, TabletId::from_u64(10)),
            Err(StateMachineFault::Misrouted { .. })
        ));
        // Corrupt bytes fail loudly.
        let mut damaged = image.clone();
        damaged[10] ^= 0xFF;
        assert!(
            ReplicatedTablet::decode_snapshot(&damaged, NS, TABLET).is_err(),
            "CRC must catch damage"
        );
    }

    /// One counter write entry at `index` for session 1.
    fn counter_entry(
        index: u64,
        seq: u64,
        delta: i64,
        value: i64,
        version: u64,
    ) -> openraft::type_config::alias::EntryOf<KiviTypeConfig> {
        use openraft::impls::leader_id_adv::LeaderId;
        openraft::Entry {
            log_id: openraft::LogId::new(LeaderId::new(1, 1), index),
            payload: openraft::EntryPayload::Normal(command(
                1,
                seq,
                Mutation::CounterAdd {
                    key: Key::from("n"),
                    delta,
                },
                OperationResult::CounterUpdated {
                    value,
                    version: ObjectVersion::from_u64(version),
                },
            )),
        }
    }

    /// Drives entries through the streaming trait `apply` (no responders:
    /// trait-level tests observe converged state via reads instead).
    async fn apply_all(
        sm: &mut ReplicatedStateMachine,
        entries: Vec<openraft::type_config::alias::EntryOf<KiviTypeConfig>>,
    ) {
        sm.apply(futures::stream::iter(
            entries.into_iter().map(|entry| Ok((entry, None))),
        ))
        .await
        .expect("applies");
    }

    /// Opens a machine and drives membership at 0, blank at 1, and two
    /// counter writes at 2-3 through the real `RaftStateMachine` trait.
    async fn seeded_counter_sm(dir: &tempfile::TempDir) -> ReplicatedStateMachine {
        use openraft::impls::leader_id_adv::LeaderId;
        let mut sm =
            ReplicatedStateMachine::open(dir.path(), NS, TABLET, authority()).expect("opens");
        let (applied, _) = sm.applied_state().await.expect("applied state");
        assert_eq!(applied, None);
        let membership = openraft::Membership::new(
            vec![BTreeSet::from([1u64])],
            BTreeMap::from([(1u64, openraft::BasicNode::new("n1".to_owned()))]),
        )
        .expect("test membership validates");
        let writes = vec![
            openraft::Entry {
                log_id: openraft::LogId::new(LeaderId::new(1, 1), 0),
                payload: openraft::EntryPayload::Membership(membership),
            },
            openraft::Entry {
                log_id: openraft::LogId::new(LeaderId::new(1, 1), 1),
                payload: openraft::EntryPayload::Blank,
            },
            counter_entry(2, 1, 10, 10, 1),
            counter_entry(3, 2, 20, 30, 2),
        ];
        apply_all(&mut sm, writes).await;
        let get = sm
            .read_local(
                &kivi_state::Operation::CounterGet {
                    key: Key::from("n"),
                },
                NOW,
            )
            .await
            .expect("seeded state reads");
        assert_eq!(get, OperationResult::Counter(Some(30)));
        sm
    }

    /// Applies through the trait surface and seals a snapshot base.
    #[test]
    fn state_machine_applies_and_seals_snapshot() {
        block_on(async {
            let dir = tempfile::tempdir().expect("scratch");
            let mut sm = seeded_counter_sm(&dir).await;
            let status = sm.status().await;
            assert_eq!(status.applied_index, Some(3));
            assert!(status.healthy);
            let builder = sm.get_snapshot_builder().await;
            assert_eq!(builder.meta.last_log_id.map(|id| id.index), Some(3));
            assert_eq!(sm.snapshotted_index().await, Some(3));
        });
    }

    /// Reopen reports the snapshot base; the retained tail replays exactly
    /// once and replicas converge.
    #[test]
    fn state_machine_recovers_through_snapshot_plus_replay() {
        block_on(async {
            let dir = tempfile::tempdir().expect("scratch");
            let mut sm = seeded_counter_sm(&dir).await;
            let _sealed = sm.get_snapshot_builder().await;
            apply_all(&mut sm, vec![counter_entry(4, 3, 5, 35, 3)]).await;
            drop(sm);
            // Reopen: base 3 durable, tail (4) replays from the log.
            let mut reopened =
                ReplicatedStateMachine::open(dir.path(), NS, TABLET, authority()).expect("reopens");
            let (applied, membership) = reopened.applied_state().await.expect("applied");
            assert_eq!(applied.map(|id| id.index), Some(3));
            assert!(membership.membership().voter_ids().any(|id| id == 1));
            apply_all(&mut reopened, vec![counter_entry(4, 3, 5, 35, 3)]).await;
            let get = reopened
                .read_local(
                    &kivi_state::Operation::CounterGet {
                        key: Key::from("n"),
                    },
                    NOW,
                )
                .await
                .expect("converged");
            assert_eq!(get, OperationResult::Counter(Some(35)));
        });
    }

    /// Crash between the applied-file write and the CURRENT switch still
    /// recovers: the applied file names a complete image, which open
    /// adopts instead of the stale CURRENT.
    #[test]
    fn open_adopts_unlinked_snapshot_image() {
        use openraft::Entry;
        use openraft::impls::leader_id_adv::LeaderId;

        block_on(async {
            let dir = tempfile::tempdir().expect("scratch");
            let mut sm =
                ReplicatedStateMachine::open(dir.path(), NS, TABLET, authority()).expect("opens");
            apply_all(
                &mut sm,
                vec![Entry {
                    log_id: openraft::LogId::new(LeaderId::new(1, 1), 0),
                    payload: openraft::EntryPayload::Normal(command(
                        1,
                        1,
                        Mutation::PutBytes {
                            key: Key::from("k"),
                            value: bytes::Bytes::from_static(b"v"),
                        },
                        OperationResult::Stored {
                            version: ObjectVersion::FIRST,
                        },
                    )),
                }],
            )
            .await;
            let _sealed = sm.get_snapshot_builder().await;
            assert_eq!(sm.snapshotted_index().await, Some(0));
            drop(sm);
            // Simulate the crash window: CURRENT gone, image + applied intact.
            std::fs::remove_file(
                dir.path()
                    .join("consensus-sm")
                    .join("9")
                    .join("snapshots")
                    .join("CURRENT"),
            )
            .expect("removes CURRENT");
            let mut reopened =
                ReplicatedStateMachine::open(dir.path(), NS, TABLET, authority()).expect("reopens");
            let (applied, _) = reopened.applied_state().await.expect("applied");
            assert_eq!(applied.map(|id| id.index), Some(0));
            let get = reopened
                .read_local(
                    &kivi_state::Operation::Get {
                        key: Key::from("k"),
                    },
                    NOW,
                )
                .await
                .expect("adopted image serves");
            assert_eq!(
                get,
                OperationResult::Value(Some(bytes::Bytes::from_static(b"v")))
            );
        });
    }

    /// Fault injection: a failed snapshot barrier latches unhealthy with
    /// its detail; reads fail closed; a validated install heals the
    /// machine.
    #[test]
    fn snapshot_failure_poisions_and_install_heals() {
        use openraft::Entry;
        use openraft::impls::leader_id_adv::LeaderId;

        block_on(async {
            let dir = tempfile::tempdir().expect("scratch");
            let mut sm =
                ReplicatedStateMachine::open(dir.path(), NS, TABLET, authority()).expect("opens");
            apply_all(
                &mut sm,
                vec![Entry {
                    log_id: openraft::LogId::new(LeaderId::new(1, 1), 0),
                    payload: openraft::EntryPayload::Normal(command(
                        1,
                        1,
                        Mutation::PutBytes {
                            key: Key::from("k"),
                            value: bytes::Bytes::from_static(b"v"),
                        },
                        OperationResult::Stored {
                            version: ObjectVersion::FIRST,
                        },
                    )),
                }],
            )
            .await;
            sm.set_snapshot_write_failure(true).await;
            let _failed = sm.get_snapshot_builder().await;
            let status = sm.status().await;
            assert!(!status.healthy, "failed seal poisons");
            assert!(status.last_error.is_some());
            assert!(
                sm.read_local(
                    &kivi_state::Operation::Get {
                        key: Key::from("k")
                    },
                    NOW
                )
                .await
                .is_err()
            );
            // Heal with a validated install of the good image.
            sm.set_snapshot_write_failure(false).await;
            let image = {
                let inner = sm.shared.lock().await;
                inner.tablet.encode_snapshot()
            };
            let meta = openraft::storage::SnapshotMeta {
                last_log_id: Some(openraft::LogId::new(LeaderId::new(1, 1), 0)),
                last_membership: openraft::StoredMembership::default(),
            };
            sm.install_snapshot(&meta, std::io::Cursor::new(image))
                .await
                .expect("install heals");
            assert!(sm.status().await.healthy);
        });
    }
}
