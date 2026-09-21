//! Worker-local batched commit pipeline: group commit with one durability
//! barrier per physical batch.
//!
//! The required logical path:
//!
//! ```text
//! mutation requests
//!        ↓
//! prepare in deterministic order (batch-local overlay; committed untouched)
//!        ↓
//! PendingCommitBatch
//!        ↓
//! single WAL physical batch
//!        ↓
//! single durability barrier (dedicated lane thread, never the DataWorker)
//!        ↓
//! CommitProof
//!        ↓
//! apply mutations in original order (+ verify against preparation)
//!        ↓
//! install dedup outcomes
//!        ↓
//! release replies
//! ```
//!
//! One durability barrier acknowledges many mutations. The architecture
//! preserves one logical order per tablet, deterministic `MutationIR`,
//! exact `ApplyOutcome`, and durability-before-acknowledgement.
//!
//! ## Why batches form without artificial delay
//!
//! A batch seals when it is full (`max_ops`/`max_bytes`), when the oldest
//! entry exceeds `max_wait`, or when the currently available queued work
//! is drained — whichever comes first. There is no linger timer: while a
//! batch is in flight on the durability lane, later arrivals accumulate in
//! the admission queue (same-tablet mutations must stay ordered behind
//! it), so the next batch forms from a full queue under load. A lone
//! mutation seals immediately and pays exactly one barrier.
//!
//! ## Speculative-state isolation
//!
//! Preparation runs against a batch-local overlay (`TabletOverlay`)
//! containing only touched keys, staged over the immutable committed
//! store. Committed state is never written before the durability proof.
//! Ordinary reads always answer from committed state (they linearize
//! before any uncommitted batch), so no request can observe speculation.
//!
//! ## Failure semantics
//!
//! A failed batch burns no commit positions (per-tablet positions rewind
//! to their pre-batch values — nothing persisted, so reuse is safe and
//! recovery chains stay contiguous), touches no committed state, and
//! fails exactly its affected requests. Session floors never advance
//! before apply, so retries prepare fresh.

use std::collections::{HashMap, HashSet, VecDeque};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, bounded};
use kivi_durability::{
    CommitProof, DurabilityError, DurabilityLevel, DurabilityProvider, PersistIntent,
    StorageHealth, WalRecord, WorkerLaneStats,
    wal::{MutationRecord, OutcomeRecord},
};
use kivi_state::{
    Key, Mutation, ObjectStore, Operation, OperationResult, StorePrepared, StoredObject,
};
use kivi_types::{CommitPosition, MutationIdentity, NamespaceId, RequestSeq, SessionId, TabletId};

use crate::tablet::{IdentityGate, LiveTablet, TabletError};
use crate::worker::{LaneAccess, WorkerRequestError, WorkerResponse};

/// Completion-detection quantum: how long a worker parks waiting for a
/// durability proof before re-polling. This is NOT batching — batches seal
/// on drain/full/deadline regardless of this value. It only bounds how
/// quickly a finished fsync is noticed (100µs against a ~12ms barrier).
pub(crate) const COMPLETION_PARK: Duration = Duration::from_micros(100);

/// Submit-channel depth to the lane thread. The coordinator keeps at most
/// one batch in flight, so one slot would suffice; a small margin keeps a
/// burst seal from ever blocking the worker on channel capacity.
const LANE_SUBMIT_DEPTH: usize = 16;

/// Completion-channel depth. Same one-in-flight discipline: completions
/// are consumed in order, so this never fills in practice.
const LANE_COMPLETE_DEPTH: usize = 16;

/// Recent batch sizes kept for exact windowed quantiles (bounded ring, no
/// unbounded growth, no histogram dependency on the data path).
const BATCH_SIZE_WINDOW: usize = 1024;

/// Upper bound the batching policy accepts: the WAL physical format holds
/// at most [`kivi_durability::wal::WAL_MAX_RECORDS_PER_BATCH`] records per
/// batch, and the policy must never promise more.
pub const BATCH_POLICY_MAX_OPS: usize = kivi_durability::wal::WAL_MAX_RECORDS_PER_BATCH;

/// How long a blocked shutdown flush waits for one proof before failing
/// closed (a wedged barrier is a durability violation, never a hang).
const FLUSH_PROOF_TIMEOUT: Duration = Duration::from_secs(30);

/// Batch-closing policy. Replaceable without touching tablet semantics:
/// adaptive or device-specific strategies implement the same close rules
/// against these limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchPolicy {
    /// Maximum mutations per physical batch (`1` = immediate durability).
    pub max_ops: usize,
    /// Maximum encoded record bytes per batch (framing excluded).
    pub max_bytes: u64,
    /// Maximum age of the oldest unsealed entry before forced seal.
    pub max_wait: Duration,
    /// Minimum age of the oldest unsealed entry before a drained (but
    /// unfilled) batch may seal. Lets concurrently arriving mutations
    /// join the batch instead of each sealing alone; a lone mutation
    /// pays at most this on top of its barrier. Zero seals drained
    /// batches immediately.
    pub linger: Duration,
}

impl BatchPolicy {
    /// Conservative production default: batches stay far below the WAL
    /// record cap, the byte cap fits comfortably in one segment write, and
    /// the wait backstop only binds under sustained trickle load
    /// (drain-close seals idle batches immediately).
    pub const DEFAULT_MAX_OPS: usize = 256;
    /// Default byte cap: 1 MiB of records per barrier.
    pub const DEFAULT_MAX_BYTES: u64 = 1024 * 1024;
    /// Default oldest-entry deadline: 500µs.
    pub const DEFAULT_MAX_WAIT: Duration = Duration::from_micros(500);
    /// Default drain linger: 200µs. Concurrent arrivals within one
    /// linger window join the batch; the cost to a lone mutation is a
    /// fraction of one barrier. Tuned by measurement (see benchmarks).
    pub const DEFAULT_LINGER: Duration = Duration::from_micros(200);

    /// Production default policy.
    #[must_use]
    pub const fn default_policy() -> Self {
        Self {
            max_ops: Self::DEFAULT_MAX_OPS,
            max_bytes: Self::DEFAULT_MAX_BYTES,
            max_wait: Self::DEFAULT_MAX_WAIT,
            linger: Self::DEFAULT_LINGER,
        }
    }

    /// Strict immediate durability: every mutation seals alone. Same
    /// pipeline, same code path — the baseline group commit measures
    /// against (spec K).
    #[must_use]
    pub const fn immediate() -> Self {
        Self {
            max_ops: 1,
            max_bytes: u64::MAX,
            max_wait: Duration::ZERO,
            linger: Duration::ZERO,
        }
    }

    /// Validates the policy against the physical format limits.
    ///
    /// # Errors
    ///
    /// Returns a reason when the policy could promise an unencodable batch.
    pub const fn validate(&self) -> Result<(), &'static str> {
        if self.max_ops == 0 {
            return Err("batch max_ops must be nonzero");
        }
        if self.max_ops > BATCH_POLICY_MAX_OPS {
            return Err("batch max_ops exceeds the WAL record cap");
        }
        if self.max_bytes == 0 {
            return Err("batch max_bytes must be nonzero");
        }
        Ok(())
    }
}

/// Cumulative commit-pipeline counters plus bounded windowed distributions.
/// Everything an operator needs for the `logical writes / fsync` ratio.
#[derive(Debug, Clone, Default)]
pub struct CommitMetricsSnapshot {
    /// Logical mutations committed (applied after proof).
    pub logical_mutations: u64,
    /// Physical WAL batches sealed.
    pub physical_batches: u64,
    /// Durability barriers issued (one fsync per batch today).
    pub barriers: u64,
    /// Payload + framing bytes made durable.
    pub bytes_durable: u64,
    /// Batches that failed persistence (requests failed, state untouched).
    pub failed_batches: u64,
    /// Mean batch size (mutations per sealed batch).
    pub batch_size_avg: f64,
    /// Largest sealed batch.
    pub batch_size_max: u64,
    /// Windowed batch-size quantiles (recent 1024 batches).
    pub batch_size_p50: u64,
    /// Windowed batch-size quantiles (recent 1024 batches).
    pub batch_size_p95: u64,
    /// Windowed batch-size quantiles (recent 1024 batches).
    pub batch_size_p99: u64,
    /// Mean oldest-entry wait before seal.
    pub oldest_wait_avg: Duration,
    /// Worst oldest-entry wait before seal.
    pub oldest_wait_max: Duration,
    /// Mean barrier latency (submit to proof).
    pub barrier_latency_avg: Duration,
    /// Worst barrier latency (submit to proof).
    pub barrier_latency_max: Duration,
    /// Currently queued (admitted, unprepared) requests.
    pub queue_depth: usize,
    /// Deepest queue observed.
    pub queue_depth_max: usize,
    /// Whether a batch is currently on the durability lane.
    pub in_flight: bool,
}

/// One admitted request awaiting batch preparation.
#[derive(Debug)]
pub(crate) struct PendingEntry {
    tablet: TabletId,
    op: Operation,
    opcode: u8,
    is_mutating: bool,
    now: kivi_types::WallTimestamp,
    identity: Option<MutationIdentity>,
    respond: Sender<WorkerResponse>,
    admitted_at: Instant,
    /// Staging pins for a chunked upload, held admission→apply so chunk
    /// GC cannot collect staged-but-uncommitted packs underneath this
    /// entry. Dropped (unpinning) on every exit path, including
    /// preparation failure. One entry per staged value: ordinary writes
    /// stage at most one, transactional batches stage several.
    pinned: Vec<crate::chunk_lane::PinnedUpload>,
    /// Staged fabric seals: medium values staged into the worker fabric
    /// at admission. Preparation converts them into seal payloads with
    /// predicted versions; failure paths retire the ids immediately so
    /// uncommitted stages never accumulate.
    fabric_staged: Vec<crate::fabric::StagedSeal>,
}

impl PendingEntry {
    /// Admits one request. The opcode discriminant and its
    /// mutating/read-only classification derive once here from the
    /// authoritative protocol table — the coordinator never re-derives
    /// them, so classification cannot skew mid-batch.
    pub(crate) fn new(
        tablet: TabletId,
        op: Operation,
        now: kivi_types::WallTimestamp,
        identity: Option<MutationIdentity>,
        respond: Sender<WorkerResponse>,
    ) -> Self {
        let opcode = kivi_protocol::operation_opcode(&op);
        Self {
            tablet,
            op,
            opcode: opcode.as_u8(),
            is_mutating: opcode.is_mutating(),
            now,
            identity,
            respond,
            admitted_at: Instant::now(),
            pinned: Vec::new(),
            fabric_staged: Vec::new(),
        }
    }

    /// Attaches a staging-pin guard carried admission→apply (see the
    /// field docs). `None` (the default) is every inline request.
    pub(crate) fn with_pinned(mut self, pinned: Option<crate::chunk_lane::PinnedUpload>) -> Self {
        if let Some(pinned) = pinned {
            self.pinned.push(pinned);
        }
        self
    }

    /// Attaches several staging-pin guards (transactional batches stage
    /// one value per large put; all ride admission→apply together).
    pub(crate) fn with_pins(mut self, pins: Vec<crate::chunk_lane::PinnedUpload>) -> Self {
        self.pinned.extend(pins);
        self
    }

    /// Attaches staged fabric seals (medium values staged into the worker
    /// fabric at admission; all ride admission→seal together).
    pub(crate) fn with_fabric_staged(mut self, staged: Vec<crate::fabric::StagedSeal>) -> Self {
        self.fabric_staged.extend(staged);
        self
    }
}

/// One committed-state read suspended on promotion: the linearization
/// (root id + version at prepare time) plus the fabric park sequence.
/// Resume re-fences against the live root before answering, so a
/// delayed promotion can never serve bytes the read was not authorized
/// to observe.
#[derive(Debug)]
struct SuspendedRead {
    /// Tablet the read runs against.
    tablet: TabletId,
    /// Key being read (root re-validation on resume).
    key: kivi_state::Key,
    /// Logical wall time of the read (expiry evaluation on resume).
    now: kivi_types::WallTimestamp,
    /// `GetRange` window (`None` for full `Get`): applied after a
    /// resolved root, so range reads never serve more than asked.
    range: Option<(u64, u64)>,
    /// Fabric park sequence holding the lane reply.
    park: u64,
    /// Where the outcome goes (bounded to one message).
    respond: Sender<WorkerResponse>,
}

/// Extracts a `GetRange` window (`None` for every other opcode): applied
/// after a fabric root resolves, so range reads slice the exact bytes
/// the read was authorized to observe.
fn read_range(op: &kivi_state::Operation) -> Option<(u64, u64)> {
    match op {
        kivi_state::Operation::GetRange { offset, len, .. } => Some((*offset, *len)),
        _ => None,
    }
}

/// Slices resolved bytes through a `GetRange` window (full bytes for
/// `Get`): the single shaping point for fabric reads, so hit, park, and
/// bridge paths cannot disagree.
fn shape_resolved(bytes: bytes::Bytes, range: Option<(u64, u64)>) -> kivi_state::OperationResult {
    match range {
        Some((offset, len)) => {
            kivi_state::OperationResult::Value(Some(kivi_state::slice_range(&bytes, offset, len)))
        }
        None => kivi_state::OperationResult::Value(Some(bytes)),
    }
}

/// How one sealed item completes after the durability proof.
#[derive(Debug)]
enum PreparedKind {
    /// Apply the mutation, verify against `expected`, install dedup.
    Mutation {
        /// Mutation to apply in batch order.
        mutation: Mutation,
        /// Outcome apply must reproduce (preparation prediction).
        expected: OperationResult,
        /// Originating request shape for response mapping.
        is_persist_expiry: bool,
    },
    /// Install the terminal outcome for retries, no state change.
    Terminal {
        /// Outcome to install and reply.
        outcome: kivi_state::DurableOutcome,
        /// Originating opcode discriminant.
        opcode: u8,
    },
}

/// The appliable half of one sealed batch member (the WAL-encodable half
/// travels in the batch's record list).
#[derive(Debug)]
struct ApplyData {
    tablet: TabletId,
    commit: CommitPosition,
    kind: PreparedKind,
    identity: Option<MutationIdentity>,
    now: kivi_types::WallTimestamp,
    respond: Sender<WorkerResponse>,
    /// Staging pins, moved from the admission and held through the apply:
    /// the journal records the commit before this drops, so GC always sees
    /// staged data as either pinned or journaled, never neither. One entry
    /// per staged value (transactional batches stage several).
    pinned: Vec<crate::chunk_lane::PinnedUpload>,
}

/// Batch-local overlay for one tablet: pending state for touched keys only
/// (`None` = deleted within the batch), staged over the immutable
/// committed store. Never cloned wholesale; never visible to reads.
///
/// Intents stage the same way as objects: `pending_intents` carries this
/// key's post-batch intent (`None` = resolved within the batch), layered
/// over the committed intent table. Scratch preparation for one key always
/// sees committed-plus-pending intents, so durable predictions match the
/// later committed apply exactly — the same contract objects already keep.
#[derive(Debug, Default)]
struct TabletOverlay {
    pending: HashMap<Key, Option<StoredObject>>,
    intents: HashMap<Key, Option<kivi_state::TxnIntent>>,
}

impl TabletOverlay {
    /// Prepares `op` against committed-plus-pending state and evolves the
    /// overlay for a `Write` exactly as the later committed apply will.
    /// Pure except for the overlay itself: committed is only borrowed.
    fn prepare(
        &mut self,
        committed: &ObjectStore,
        op: &Operation,
        now: kivi_types::WallTimestamp,
    ) -> Result<StorePrepared, TabletError> {
        let key = op.key();
        // Materialize a scratch store holding exactly this key's current
        // view state. Single-key operations read no other state (every
        // prepare path resolves through the op key alone), so a one-entry
        // scratch prepares identically to the full store while cloning
        // only the touched root.
        let mut scratch = ObjectStore::new();
        match self.pending.get(key) {
            Some(Some(object)) => scratch.put_stored(key.clone(), object.clone()),
            Some(None) => {}
            None => {
                if let Some(object) = committed.get_stored(key) {
                    scratch.put_stored(key.clone(), object.clone());
                }
            }
        }
        // Stage exactly this key's intent view: overlay-staged intents win,
        // otherwise the committed intent (if any). Single-key operations
        // read no intent state beyond their key, so this one-entry staging
        // predicts identically to the full store.
        match self.intents.get(key) {
            Some(Some(intent)) => scratch.restore_intent(intent.clone()),
            Some(None) => {}
            None => {
                if let Some(intent) = committed.cloned_intent_for_key(key) {
                    scratch.restore_intent(intent);
                }
            }
        }
        let prepared = scratch.prepare_durable(op, now)?;
        if let StorePrepared::Write { mutation, .. } = &prepared {
            // Evolve the overlay through the same deterministic apply the
            // committed store will run after durability. A failure here is
            // a preparation-time request error (nothing is durable yet),
            // mirroring the live path's apply-error mapping.
            scratch.apply(mutation, now).map_err(TabletError::Apply)?;
            let staged = scratch.get_stored(mutation.key()).cloned();
            self.pending.insert(key.clone(), staged);
            self.intents
                .insert(key.clone(), scratch.cloned_intent_for_key(mutation.key()));
        }
        Ok(prepared)
    }

    /// Prepares a same-tablet atomic commit against committed-plus-pending
    /// state: stages every touched key into a scratch store (overlay roots
    /// win, otherwise committed roots, plus both intent layers), runs the
    /// atomic [`commit_local`](kivi_state::ObjectStore::commit_local) there,
    /// and stages every post-commit root back into the overlay so later
    /// batch items observe the commit. Prediction and later committed apply
    /// agree by construction on identical state.
    fn prepare_local_commit(
        &mut self,
        committed: &ObjectStore,
        txn: kivi_state::TxnId,
        writes: &[kivi_state::TxnWrite],
        now: kivi_types::WallTimestamp,
    ) -> Result<StorePrepared, TabletError> {
        use kivi_state::{LocalCommitError, OperationResult};
        if writes.is_empty() {
            return Err(TabletError::Op(kivi_state::OpError::TxnConflict));
        }
        let mut touched: std::collections::BTreeSet<Key> = std::collections::BTreeSet::new();
        for write in writes {
            touched.insert(write.key.clone());
        }
        let mut scratch = ObjectStore::new();
        for key in &touched {
            match self.pending.get(key) {
                Some(Some(object)) => scratch.put_stored(key.clone(), object.clone()),
                Some(None) => {}
                None => {
                    if let Some(object) = committed.get_stored(key) {
                        scratch.put_stored(key.clone(), object.clone());
                    }
                }
            }
            match self.intents.get(key) {
                Some(Some(intent)) => scratch.restore_intent(intent.clone()),
                Some(None) => {}
                None => {
                    if let Some(intent) = committed.cloned_intent_for_key(key) {
                        scratch.restore_intent(intent);
                    }
                }
            }
        }
        match scratch.commit_local(writes, now) {
            Ok(_) => {
                for key in &touched {
                    self.pending
                        .insert(key.clone(), scratch.get_stored(key).cloned());
                    self.intents
                        .insert(key.clone(), scratch.cloned_intent_for_key(key));
                }
                Ok(StorePrepared::Write {
                    mutation: kivi_state::Mutation::TxnCommitLocal {
                        txn,
                        writes: writes.to_vec(),
                    },
                    expected: OperationResult::TxnLocalCommitted {
                        versions: kivi_state::local_versions(&scratch, writes, now),
                    },
                })
            }
            Err(LocalCommitError::Validation(op)) => Err(TabletError::Op(op)),
            Err(LocalCommitError::Exhausted) => {
                Err(TabletError::Apply(kivi_state::ApplyError::VersionExhausted))
            }
            Err(LocalCommitError::Diverged(error)) => Err(TabletError::Apply(error)),
            // Future validation failures fail closed as divergence.
            Err(_) => Err(TabletError::Apply(kivi_state::ApplyError::TxnDiverged)),
        }
    }
}

/// Builds seal payloads for every staged fabric id the prepared mutation
/// references, stamping predicted authoritative versions from the
/// expected outcome. Predict/apply agreement makes these versions exact;
/// unreferenced staged ids are the caller's to retire (unmet conditions,
/// terminal outcomes).
///
/// 2PC intents seal twice: `TxnPrepare` persists the staged bytes under
/// the staging version (`0`, never a root) so a crash cannot strand a
/// prepared intent without payload, and pins the id until the intent
/// resolves; `TxnFinalize` re-seals the same bytes under the predicted
/// authoritative version, superseding the staging record.
fn fabric_payloads_for(
    tablet: TabletId,
    mutation: &Mutation,
    expected: &kivi_state::OperationResult,
    staged: &[crate::fabric::StagedSeal],
) -> Vec<crate::fabric::FabricSealPayload> {
    use kivi_state::{Mutation as M, OperationResult as R};
    let sealed: Vec<(u64, u64)> = match (mutation, expected) {
        (M::ReplaceFabricRoot { fabric_id, .. }, R::Stored { version }) => {
            vec![(*fabric_id, version.as_u64())]
        }
        (
            M::ReplaceFabricRootWithExpiry { fabric_id, .. },
            R::ConditionalSet {
                applied: true,
                version: Some(version),
            },
        ) => vec![(*fabric_id, version.as_u64())],
        (M::TxnCommitLocal { writes, .. }, R::TxnLocalCommitted { versions }) => writes
            .iter()
            .zip(versions.iter())
            .filter_map(|(write, version)| match (&write.kind, version) {
                (kivi_state::TxnWriteKind::PutFabric { fabric_id, .. }, Some(version)) => {
                    Some((*fabric_id, version.as_u64()))
                }
                _ => None,
            })
            .collect(),
        (
            M::TxnPrepare {
                write: kivi_state::TxnWriteKind::PutFabric { fabric_id, .. },
                ..
            },
            _,
        ) => vec![(*fabric_id, 0)],
        (
            M::TxnFinalize { .. },
            R::TxnFinalized {
                applied: true,
                version: Some(version),
            },
        ) => staged
            .iter()
            .map(|seal| (seal.fabric_id, version.as_u64()))
            .collect(),
        _ => Vec::new(),
    };
    sealed
        .into_iter()
        .filter_map(|(fabric_id, version)| {
            staged
                .iter()
                .find(|seal| seal.fabric_id == fabric_id)
                .map(|seal| crate::fabric::FabricSealPayload {
                    fabric_id,
                    tablet,
                    key: mutation.key().clone(),
                    version,
                    bytes: seal.bytes.clone(),
                })
        })
        .collect()
}

/// Estimated wire length of one WAL record body contribution, mirroring
/// `encode_record_body` (8-byte frame + fixed prefix + identity + blobs).
/// Used for the `max_bytes` close rule; the lane's authoritative
/// `bytes_durable` comes from the real encoder.
fn record_wire_len(record: &WalRecord) -> u64 {
    use kivi_codec::Encode;
    const FRAME: u64 = 8;
    const PREFIX: u64 = 48 + 1;
    let (identity, blobs) = match record {
        WalRecord::Mutation(entry) => (
            entry.identity,
            4 + entry.mutation.encoded_len() as u64 + 4 + entry.expected.encoded_len() as u64,
        ),
        WalRecord::Outcome(entry) => (entry.identity, 4 + entry.outcome.encoded_len() as u64),
    };
    let identity_len: u64 = if identity.is_some() {
        1 + 16 + 8 + 8
    } else {
        1
    };
    FRAME + PREFIX + identity_len + blobs
}

/// Batch under construction: appliable items plus their WAL records, in
/// deterministic queue order.
#[derive(Debug)]
struct OpenBatch {
    applies: Vec<ApplyData>,
    records: Vec<WalRecord>,
    bytes: u64,
    overlays: HashMap<TabletId, TabletOverlay>,
    /// Pre-batch commit positions per touched tablet (rollback on failure).
    commit_base: HashMap<TabletId, CommitPosition>,
    oldest_admitted: Instant,
    /// Staged fabric payloads sealed with this batch (durable-before-
    /// reference): the lane persists every payload plus its journal entry
    /// before the WAL barrier.
    fabric_payloads: Vec<crate::fabric::FabricSealPayload>,
    /// When formation started (bounds the formation window: under
    /// continuous arrivals a batch must still seal instead of growing
    /// while the queue never drains).
    formed_at: Instant,
}

/// Batch on the durability lane: sealed records plus everything needed to
/// apply or roll back on completion.
#[derive(Debug)]
struct InflightBatch {
    batch_seq: u64,
    tablets: HashSet<TabletId>,
    applies: Vec<ApplyData>,
    commit_base: HashMap<TabletId, CommitPosition>,
    submitted_at: Instant,
    oldest_admitted: Instant,
    /// Staged fabric ids sealed with this batch: retired when the batch
    /// fails (nothing references them), live when it commits (roots name
    /// them and seal locators reconstruct them).
    fabric_staged: Vec<u64>,
}

/// One sealed unit of work for the lane thread.
#[derive(Debug)]
pub(crate) struct SealJob {
    batch_seq: u64,
    records: Vec<WalRecord>,
    /// Staged fabric payloads: the lane persists every payload and its
    /// journal entry BEFORE the WAL barrier, establishing
    /// durable-before-reference ordering for fabric roots in `records`.
    fabric_payloads: Vec<crate::fabric::FabricSealPayload>,
}

/// Lane-thread commands: seals plus infrequent checkpoint-time
/// maintenance. One channel, processed strictly in order — a seal and a
/// reclaim can never interleave mid-batch.
#[derive(Debug)]
pub(crate) enum LaneCommand {
    /// Persist one batch and report its proof.
    Seal(SealJob),
    /// Force-rotate the active segment (checkpoint-time simplification).
    SealActive {
        /// Where the outcome goes.
        reply: Sender<Result<(), DurabilityError>>,
    },
    /// Strictly scan sealed segments for the reclamation planner.
    ScanSealed {
        /// Segments to scan (must all be sealed).
        segments: Vec<u64>,
        /// Where per-segment summaries (or the first failure) go.
        reply: Sender<
            Result<Vec<kivi_durability::SealedSegmentSummary>, kivi_durability::RecoveryError>,
        >,
    },
    /// Delete sealed segments (never the active tail).
    DeleteSegments {
        /// Segments to remove.
        segments: Vec<u64>,
        /// Where the deleted sequences go (best-effort subset on error).
        reply: Sender<Vec<u64>>,
    },
    /// Report numbering watermarks for floor computation.
    FloorSnapshot {
        /// Where the snapshot goes.
        reply: Sender<LaneFloorSnapshot>,
    },
}

/// Numbering watermarks one lane reports for reclaim planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LaneFloorSnapshot {
    /// The lane.
    pub lane: u16,
    /// Active (still-written) segment.
    pub active_segment: u64,
    /// First batch of the active segment (or the upcoming sequence).
    pub active_first_batch: u64,
    /// One past the highest batch ever issued.
    pub next_batch: u64,
}

/// Durability proof (or failure) plus the lane's authoritative snapshot —
/// stats ride home on every completion, so admin never round-trips.
#[derive(Debug)]
pub(crate) struct SealOutcome {
    batch_seq: u64,
    result: Result<CommitProof, DurabilityError>,
    lane_stats: WorkerLaneStats,
    /// Durable locators for the sealed fabric payloads (empty when the
    /// batch staged nothing): the worker mirrors these into memory as
    /// the reconstruction sources for the new roots.
    fabric_locators: Vec<crate::fabric::FabricSealLocator>,
}

/// Fabric seal store: the durability lane's single-writer handle over
/// the worker's `materializations.dat` plus the fabric journal. Lives on
/// the lane thread; the worker thread never touches these files.
#[derive(Debug)]
pub struct FabricSealStore {
    provider: kivi_memory::NvmeProvider,
    journal_path: std::path::PathBuf,
}

impl FabricSealStore {
    /// Opens (creating) the material provider and journal parent.
    ///
    /// # Errors
    ///
    /// Returns [`DurabilityError`] when directories or files cannot be
    /// opened, or the record format is newer than this build.
    pub(crate) fn open(paths: &crate::fabric::FabricPaths) -> Result<Self, DurabilityError> {
        std::fs::create_dir_all(paths.material_dir()).map_err(|error| DurabilityError::Io {
            op: "create fabric material dir",
            message: error.to_string(),
            code: None,
        })?;
        let provider = kivi_memory::NvmeProvider::open(
            &paths.material_dir(),
            kivi_memory::NvmeOptions::default(),
        )
        .map_err(|error| DurabilityError::Io {
            op: "open fabric material provider",
            message: error.to_string(),
            code: None,
        })?;
        Ok(Self {
            provider,
            journal_path: paths.journal(),
        })
    }

    /// Persists every payload plus its journal entry under one barrier,
    /// returning locators in input order. The caller seals the WAL only
    /// after this returns success (durable-before-reference).
    ///
    /// # Errors
    ///
    /// Returns [`DurabilityError`] on any filesystem failure; partial
    /// prefixes stay in the files and are validated or truncated by
    /// recovery (torn tails) and journal load (CRC per record).
    pub(crate) fn seal_payloads(
        &mut self,
        payloads: &[crate::fabric::FabricSealPayload],
    ) -> Result<Vec<crate::fabric::FabricSealLocator>, DurabilityError> {
        let map_io = |op: &'static str| {
            move |error: std::io::Error| DurabilityError::Io {
                op,
                message: error.to_string(),
                code: None,
            }
        };
        let mut locators = Vec::with_capacity(payloads.len());
        let mut entries = Vec::with_capacity(payloads.len());
        for payload in payloads {
            let (offset, len, checksum) =
                self.provider
                    .append(&payload.bytes)
                    .map_err(|error| DurabilityError::Io {
                        op: "append fabric material record",
                        message: error.to_string(),
                        code: None,
                    })?;
            entries.push(crate::fabric::JournalEntry {
                fabric_id: payload.fabric_id,
                tablet: payload.tablet,
                key: payload.key.as_bytes().to_vec(),
                version: payload.version,
                file: crate::fabric::JournalFile::Material,
                offset,
                len,
                checksum,
            });
            locators.push(crate::fabric::FabricSealLocator {
                fabric_id: payload.fabric_id,
                tablet: payload.tablet,
                key: payload.key.clone(),
                version: payload.version,
                offset,
                len,
                checksum,
            });
        }
        if !entries.is_empty() {
            crate::fabric::journal_append_batch(&self.journal_path, &entries)
                .map_err(map_io("append fabric journal"))?;
        }
        Ok(locators)
    }
}

/// Runs one closure under the WAL lane, exclusive or shared.
type LaneRunner<'a> = &'a mut dyn FnMut(&mut dyn FnMut(&mut kivi_durability::LocalWalLane));

/// Reports one failed seal: builds the outcome under the lane (for
/// lane stats) and sends it. Kept beside
/// [`seal_one_batch`] so the seal arm stays reviewable.
fn fail_seal(
    job: &SealJob,
    error: &DurabilityError,
    with_lane: LaneRunner<'_>,
    complete: &Sender<SealOutcome>,
) {
    let mut outcome: Option<SealOutcome> = None;
    with_lane(&mut |inner| {
        outcome = Some(SealOutcome {
            batch_seq: job.batch_seq,
            result: Err(error.clone()),
            lane_stats: inner.lane_stats(),
            fabric_locators: Vec::new(),
        });
    });
    let _ = complete.send(outcome.expect("seal always reports"));
}

/// Seals one batch: fabric payloads first (durable-before-reference),
/// then the WAL barrier. A fabric failure retires nothing here — the
/// coordinator retires staged ids when it collects the failed proof —
/// but unreferenced records never accumulate because every staged id
/// rides exactly one seal.
fn seal_one_batch(
    job: &SealJob,
    with_lane: LaneRunner<'_>,
    seal_store: &mut Option<FabricSealStore>,
    complete: &Sender<SealOutcome>,
) {
    // Fabric payloads first (durable-before-reference): a seal failure
    // below then also retires the staged ids on the worker, so
    // unreferenced records never accumulate behind a failed batch.
    let fabric_sealed = match seal_store.as_mut() {
        Some(store) if !job.fabric_payloads.is_empty() => {
            Some(store.seal_payloads(&job.fabric_payloads))
        }
        _ => None,
    };
    if !job.fabric_payloads.is_empty() && seal_store.is_none() {
        fail_seal(
            job,
            &DurabilityError::Io {
                op: "seal fabric payloads",
                message: "no fabric seal store on this lane".to_owned(),
                code: None,
            },
            with_lane,
            complete,
        );
        return;
    }
    let fabric_locators = match fabric_sealed {
        Some(Err(error)) => {
            fail_seal(job, &error, with_lane, complete);
            return;
        }
        Some(Ok(locators)) => locators,
        None => Vec::new(),
    };
    let mut outcome: Option<SealOutcome> = None;
    with_lane(&mut |inner| {
        let result = inner.append_batch(&PersistIntent {
            records: &job.records,
            level: DurabilityLevel::Sync,
        });
        outcome = Some(SealOutcome {
            batch_seq: job.batch_seq,
            result,
            lane_stats: inner.lane_stats(),
            fabric_locators: fabric_locators.clone(),
        });
    });
    // The coordinator always waits for every seal it submits
    // (poll or flush); a gone waiter only happens past a
    // fail-closed panic, in which case dropping the proof is
    // moot.
    let _ = complete.send(outcome.expect("seal always reports"));
}

/// Dedicated durability-lane thread: owns the WAL lane, appends sealed
/// batches, syncs, and reports. The `DataWorker` never touches the
/// filesystem; the lane thread never touches tablet state. Maintenance
/// commands run between seals in arrival order. When a fabric seal store
/// is present, staged fabric payloads persist (records plus journal,
/// one barrier) before the WAL barrier of the same seal.
fn lane_thread_main(
    lane: LaneAccess,
    submit: Receiver<LaneCommand>,
    complete: &Sender<SealOutcome>,
    mut seal_store: Option<FabricSealStore>,
) {
    let mut lane = lane;
    let mut with_lane = |job: &mut dyn FnMut(&mut kivi_durability::LocalWalLane)| {
        match &mut lane {
            LaneAccess::Exclusive(inner) => job(inner),
            LaneAccess::Shared(shared) => {
                if let Ok(mut inner) = shared.lock() {
                    job(&mut inner);
                } else {
                    tracing::error!("shared WAL lane mutex poisoned");
                    // Fail-closed: poison the process loudly rather than
                    // serve durability from a broken lane.
                    panic!("shared WAL lane mutex poisoned");
                }
            }
        }
    };
    for command in submit {
        match command {
            LaneCommand::Seal(job) => {
                seal_one_batch(&job, &mut with_lane, &mut seal_store, complete);
            }
            LaneCommand::SealActive { reply } => {
                let mut result = Ok(());
                with_lane(&mut |inner| {
                    result = inner.seal_active_segment();
                });
                let _ = reply.send(result);
            }
            LaneCommand::ScanSealed { segments, reply } => {
                let mut summaries = Vec::with_capacity(segments.len());
                let mut failure = None;
                with_lane(&mut |inner| {
                    for segment in &segments {
                        match inner.scan_sealed(*segment) {
                            Ok(summary) => summaries.push(summary),
                            Err(error) => {
                                failure = Some(error);
                                break;
                            }
                        }
                    }
                });
                let _ = reply.send(match failure {
                    Some(error) => Err(error),
                    None => Ok(summaries),
                });
            }
            LaneCommand::DeleteSegments { segments, reply } => {
                let mut deleted = Vec::new();
                with_lane(&mut |inner| {
                    for segment in &segments {
                        match inner.delete_segment(*segment) {
                            Ok(()) => deleted.push(*segment),
                            Err(error) => {
                                tracing::warn!(
                                    segment,
                                    ?error,
                                    "WAL reclaim delete failed; segment kept"
                                );
                                break;
                            }
                        }
                    }
                });
                let _ = reply.send(deleted);
            }
            LaneCommand::FloorSnapshot { reply } => {
                let mut snapshot = None;
                with_lane(&mut |inner| {
                    snapshot = Some(LaneFloorSnapshot {
                        lane: inner.lane(),
                        active_segment: inner.active_segment(),
                        active_first_batch: inner.active_first_batch(),
                        next_batch: inner.next_batch_seq(),
                    });
                });
                let _ = reply.send(snapshot.expect("snapshot always reports"));
            }
        }
    }
}

/// Handle to the dedicated durability-lane thread.
#[derive(Debug)]
pub struct LaneHandle {
    submit: Sender<LaneCommand>,
    thread: Option<JoinHandle<()>>,
}

impl LaneHandle {
    /// Spawns the lane thread owning `lane`, reporting completions on
    /// `complete`.
    ///
    /// # Errors
    ///
    /// Returns [`DurabilityError`] when the OS refuses the thread.
    pub(crate) fn spawn(
        lane: LaneAccess,
        complete: &Sender<SealOutcome>,
        seal_store: Option<FabricSealStore>,
    ) -> Result<Self, DurabilityError> {
        let (submit, receive) = bounded::<LaneCommand>(LANE_SUBMIT_DEPTH);
        let complete_thread = complete.clone();
        let thread = thread::Builder::new()
            .name("kivi-durability-lane".to_owned())
            .spawn(move || lane_thread_main(lane, receive, &complete_thread, seal_store))
            .map_err(|error| DurabilityError::Io {
                op: "spawn durability lane thread",
                message: error.to_string(),
                code: None,
            })?;
        Ok(Self {
            submit,
            thread: Some(thread),
        })
    }

    /// Joins the lane thread. The submit channel must already be severed
    /// (all handles dropped) or this blocks until the lane drains — which
    /// is exactly the shutdown semantic wanted.
    fn join(&mut self) {
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }

    /// Severs submission and joins the lane thread. Severing must precede
    /// joining: the lane parks in `submit.recv()`, so joining while any
    /// sender lives waits forever. (Notable: `Drop` runs before fields
    /// drop, so a `Drop` impl that merely joins would deadlock on its own
    /// still-live submit sender — it must sever first.)
    fn shutdown(&mut self) {
        self.submit = bounded::<LaneCommand>(0).0;
        self.join();
    }

    /// Clones direct maintenance access to the lane thread (checkpoint
    /// worker path; seals keep flowing through the coordinator).
    pub(crate) fn maintenance(&self) -> LaneMaintenance {
        LaneMaintenance {
            submit: self.submit.clone(),
        }
    }

    /// Builds a handle around an externally driven submit channel (tests
    /// act as the lane: they read seals and answer proofs or failures on
    /// demand). No thread is owned, so shutdown is a no-op.
    #[cfg(test)]
    pub(crate) fn manual(submit: Sender<LaneCommand>) -> Self {
        Self {
            submit,
            thread: None,
        }
    }
}

/// Worker-local commit coordinator: admission queue, batch formation,
/// durability submission, ordered completion. Lives on the `DataWorker`
/// thread (plain struct for the embedded loop, behind `RefCell` on the
/// reactor); the lane thread owns all filesystem work.
pub struct CommitCoordinator {
    policy: BatchPolicy,
    queue: VecDeque<PendingEntry>,
    open: Option<OpenBatch>,
    inflight: Option<InflightBatch>,
    next_batch_seq: u64,
    lane: LaneHandle,
    complete: Receiver<SealOutcome>,
    /// Identities admitted but not yet applied (open batch plus in-flight
    /// batch): retries of these hold their queue position instead of
    /// preparing twice.
    unapplied_identities: HashSet<(SessionId, RequestSeq)>,
    /// Per-session admitted-but-unapplied counts: the overload cap stays
    /// exact across batch boundaries.
    unapplied_sessions: HashMap<SessionId, usize>,
    health: StorageHealth,
    lane_stats: WorkerLaneStats,
    /// Committed-state reads suspended on promotion (polled, never
    /// blocking): linearization captured at prepare, re-fenced on resume.
    suspended: VecDeque<SuspendedRead>,
    logical_mutations: u64,
    physical_batches: u64,
    barriers: u64,
    bytes_durable: u64,
    failed_batches: u64,
    batch_sizes: VecDeque<u32>,
    batch_size_count: u64,
    batch_size_total: u64,
    batch_size_max: u64,
    oldest_wait_total: Duration,
    oldest_wait_max: Duration,
    barrier_latency_total: Duration,
    barrier_latency_max: Duration,
    queue_depth_max: usize,
}

impl core::fmt::Debug for CommitCoordinator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CommitCoordinator")
            .field("policy", &self.policy)
            .field("queued", &self.queue.len())
            .field("open", &self.open.as_ref().map(|open| open.applies.len()))
            .field("in_flight", &self.inflight.is_some())
            .field("health", &self.health)
            .field("logical_mutations", &self.logical_mutations)
            .field("physical_batches", &self.physical_batches)
            .finish_non_exhaustive()
    }
}

impl Drop for CommitCoordinator {
    /// Backstop teardown: severs the lane and joins it. Every owner
    /// flushes explicitly before drop (answering all admitted requests),
    /// so by the time this runs there is normally no unapplied work left.
    /// If work does remain (engine-gone paths), abandoning it is safe:
    /// nothing unapplied was ever persisted, committed state is
    /// untouched, and any sealed batch replays from the WAL on restart.
    fn drop(&mut self) {
        self.lane.shutdown();
    }
}

impl CommitCoordinator {
    /// Builds a coordinator and spawns its lane thread.
    ///
    /// # Errors
    ///
    /// Returns [`DurabilityError`] when the policy is invalid or the lane
    /// thread cannot spawn.
    pub(crate) fn spawn(
        policy: BatchPolicy,
        lane: LaneAccess,
        initial_stats: WorkerLaneStats,
        seal_store: Option<FabricSealStore>,
    ) -> Result<Self, DurabilityError> {
        policy
            .validate()
            .map_err(|reason| DurabilityError::InvalidConfig { reason })?;
        let (complete_tx, complete) = bounded::<SealOutcome>(LANE_COMPLETE_DEPTH);
        Ok(Self {
            policy,
            queue: VecDeque::new(),
            open: None,
            inflight: None,
            next_batch_seq: 1,
            lane: LaneHandle::spawn(lane, &complete_tx, seal_store)?,
            complete,
            unapplied_identities: HashSet::new(),
            unapplied_sessions: HashMap::new(),
            health: StorageHealth::Healthy,
            suspended: VecDeque::new(),
            lane_stats: initial_stats,
            logical_mutations: 0,
            physical_batches: 0,
            barriers: 0,
            bytes_durable: 0,
            failed_batches: 0,
            batch_sizes: VecDeque::new(),
            batch_size_count: 0,
            batch_size_total: 0,
            batch_size_max: 0,
            oldest_wait_total: Duration::ZERO,
            oldest_wait_max: Duration::ZERO,
            barrier_latency_total: Duration::ZERO,
            barrier_latency_max: Duration::ZERO,
            queue_depth_max: 0,
        })
    }

    /// Builds a coordinator around a test-driven lane: the caller reads
    /// seals from `seals` and answers on the coordinator's completion
    /// channel, proving or failing batches on demand. No thread exists.
    #[cfg(test)]
    pub(crate) fn with_manual_lane(
        policy: BatchPolicy,
        initial_stats: WorkerLaneStats,
    ) -> (Self, Receiver<LaneCommand>, Sender<SealOutcome>) {
        policy.validate().expect("test policy valid");
        let (submit_tx, submit_rx) = bounded::<LaneCommand>(LANE_SUBMIT_DEPTH);
        let (complete_tx, complete) = bounded::<SealOutcome>(LANE_COMPLETE_DEPTH);
        let coordinator = Self {
            policy,
            queue: VecDeque::new(),
            open: None,
            inflight: None,
            next_batch_seq: 1,
            lane: LaneHandle::manual(submit_tx),
            complete,
            unapplied_identities: HashSet::new(),
            unapplied_sessions: HashMap::new(),
            health: StorageHealth::Healthy,
            suspended: VecDeque::new(),
            lane_stats: initial_stats,
            logical_mutations: 0,
            physical_batches: 0,
            barriers: 0,
            bytes_durable: 0,
            failed_batches: 0,
            batch_sizes: VecDeque::new(),
            batch_size_count: 0,
            batch_size_total: 0,
            batch_size_max: 0,
            oldest_wait_total: Duration::ZERO,
            oldest_wait_max: Duration::ZERO,
            barrier_latency_total: Duration::ZERO,
            barrier_latency_max: Duration::ZERO,
            queue_depth_max: 0,
        };
        (coordinator, submit_rx, complete_tx)
    }

    /// Admits one request to the queue (bounded upstream by the worker's
    /// request channel; the queue itself never exceeds what was admitted).
    pub(crate) fn admit(&mut self, entry: PendingEntry) {
        self.queue.push_back(entry);
        self.queue_depth_max = self.queue_depth_max.max(self.queue.len());
    }

    /// Direct maintenance access to the durability lane (checkpoint worker
    /// path; seals keep flowing through this coordinator).
    pub(crate) fn maintenance(&self) -> LaneMaintenance {
        self.lane.maintenance()
    }

    /// Whether any request is admitted-but-unanswered, suspended on
    /// promotion, or any batch is on the lane: what the worker's park
    /// logic consults.
    #[must_use]
    pub(crate) fn has_pending(&self) -> bool {
        !self.queue.is_empty()
            || self.open.is_some()
            || self.inflight.is_some()
            || !self.suspended.is_empty()
    }

    /// Current coarse lane health (cached from the last seal outcome).
    #[must_use]
    pub(crate) const fn health(&self) -> StorageHealth {
        self.health
    }

    /// Last authoritative lane snapshot (rides home on every seal).
    #[must_use]
    pub(crate) const fn lane_stats(&self) -> WorkerLaneStats {
        self.lane_stats
    }

    /// Keys currently touched by queued, open, in-flight, or suspended
    /// operations. Sweeps consult this to avoid deleting a key whose
    /// prediction is already staged - a sweep/delete race would otherwise
    /// diverge the post-durability verification. Suspended reads join the
    /// set: a sweep must not delete a key whose promotion is in flight.
    #[must_use]
    pub(crate) fn touched_keys(&self) -> HashSet<Key> {
        let mut keys = HashSet::new();
        for entry in &self.queue {
            if entry.is_mutating {
                keys.insert(entry.op.key().clone());
            }
        }
        if let Some(open) = &self.open {
            for overlay in open.overlays.values() {
                keys.extend(overlay.pending.keys().cloned());
            }
        }
        if let Some(inflight) = &self.inflight {
            for apply in &inflight.applies {
                match &apply.kind {
                    PreparedKind::Mutation { mutation, .. } => {
                        keys.extend(mutation.keys().into_iter().cloned());
                    }
                    PreparedKind::Terminal { .. } => {}
                }
            }
        }
        for suspended in &self.suspended {
            keys.insert(suspended.key.clone());
        }
        keys
    }

    /// Fabric ids in the open or in-flight batch (sealed or sealing):
    /// checkpoint journal GC must retain their records until the batch
    /// applies, even though no live root names them yet.
    #[must_use]
    pub(crate) fn pending_fabric_ids(&self) -> Vec<u64> {
        let mut ids = Vec::new();
        if let Some(open) = &self.open {
            ids.extend(open.fabric_payloads.iter().map(|payload| payload.fabric_id));
        }
        if let Some(inflight) = &self.inflight {
            ids.extend(inflight.fabric_staged.iter().copied());
        }
        ids
    }

    /// Pumps the pipeline once: drains ready completions, answers every
    /// fast path (reads, unknown tablets, settled identities), then forms
    /// and seals at most one batch. Never blocks; never touches the
    /// filesystem. Safe to call from any worker pump point (run loop,
    /// bridge cycle, connection waiter).
    ///
    /// Fast paths run even with a batch in flight: committed-state reads
    /// and settled identities never wait on a barrier. Fabric reads that
    /// miss residency suspend into the coordinator's parked set instead
    /// of blocking; later polls resume them.
    pub(crate) fn poll(
        &mut self,
        tablets: &mut HashMap<TabletId, LiveTablet>,
        namespace: NamespaceId,
        fabric: &mut crate::fabric::TabletFabric,
    ) {
        self.drain_completions(tablets, fabric);
        self.poll_suspended(tablets, fabric);
        self.answer_fast_paths(tablets, fabric);
        if self.inflight.is_none() {
            self.prepare_mutations(tablets, namespace, fabric);
        }
    }

    /// Blocks until every admitted request is answered (shutdown path on
    /// plain worker threads; the reactor uses its own wait loop).
    ///
    /// # Panics
    ///
    /// Panics when the lane thread dies mid-flush or a completion carries
    /// an impossible batch sequence: both are fail-closed durability
    /// violations, never silent.
    pub(crate) fn flush_blocking(
        &mut self,
        tablets: &mut HashMap<TabletId, LiveTablet>,
        namespace: NamespaceId,
        fabric: &mut crate::fabric::TabletFabric,
    ) {
        loop {
            self.poll(tablets, namespace, fabric);
            if !self.has_pending() {
                return;
            }
            let Ok(outcome) = self.complete.recv_timeout(FLUSH_PROOF_TIMEOUT) else {
                panic!("durability lane died during shutdown flush");
            };
            self.apply_completion(tablets, outcome, fabric);
        }
    }
}

/// Cloneable maintenance access to a durability lane: the checkpoint
/// worker sends lane commands straight to the lane thread through this
/// (never through the reactor). Seals keep flowing through the owning
/// coordinator; maintenance runs between seals in arrival order.
#[derive(Debug, Clone)]
pub struct LaneMaintenance {
    submit: Sender<LaneCommand>,
}

impl LaneMaintenance {
    /// Sends one maintenance command to the lane thread and waits for its
    /// reply (checkpoint worker threads only — never the reactor).
    /// A dead or wedged lane fails loudly after a bounded wait instead of
    /// hanging reclamation.
    fn round_trip<T>(
        &self,
        build: impl FnOnce(Sender<T>) -> LaneCommand,
    ) -> Result<T, DurabilityError> {
        let (reply_tx, reply_rx) = bounded::<T>(1);
        self.submit
            .send(build(reply_tx))
            .map_err(|_| DurabilityError::Io {
                op: "submit WAL maintenance command",
                message: "durability lane thread is gone".to_owned(),
                code: None,
            })?;
        reply_rx
            .recv_timeout(FLUSH_PROOF_TIMEOUT)
            .map_err(|_| DurabilityError::Io {
                op: "await WAL maintenance command",
                message: "durability lane thread is wedged".to_owned(),
                code: None,
            })
    }

    /// Force-rotates the active segment (checkpoint-time simplification).
    /// Blocking: checkpoint worker threads only.
    ///
    /// # Errors
    ///
    /// Returns [`DurabilityError`] when the rotation cannot publish or the
    /// lane is gone.
    pub(crate) fn seal_active_sync(&self) -> Result<(), DurabilityError> {
        self.round_trip(|reply| LaneCommand::SealActive { reply })?
    }

    /// Strictly scans sealed segments for the reclamation planner.
    /// Blocking: checkpoint worker threads only.
    ///
    /// # Errors
    ///
    /// Returns [`DurabilityError`] on transport failure or the first
    /// strict scan failure (as a lane I/O error carrying the validation
    /// message — the planner keeps everything on any error).
    pub(crate) fn scan_sealed_sync(
        &self,
        segments: Vec<u64>,
    ) -> Result<Vec<kivi_durability::SealedSegmentSummary>, DurabilityError> {
        self.round_trip(|reply| LaneCommand::ScanSealed { segments, reply })?
            .map_err(|error| DurabilityError::Io {
                op: "scan sealed WAL segment",
                message: format!("{error:?}"),
                code: None,
            })
    }

    /// Deletes sealed segments (never the active tail). Returns what was
    /// actually removed; transport failure errors.
    /// Blocking: checkpoint worker threads only.
    ///
    /// # Errors
    ///
    /// Returns [`DurabilityError`] when the lane is gone.
    pub(crate) fn delete_segments_sync(
        &self,
        segments: Vec<u64>,
    ) -> Result<Vec<u64>, DurabilityError> {
        self.round_trip(|reply| LaneCommand::DeleteSegments { segments, reply })
    }

    /// Reports numbering watermarks for floor computation.
    /// Blocking: checkpoint worker threads only.
    ///
    /// # Errors
    ///
    /// Returns [`DurabilityError`] when the lane is gone.
    pub(crate) fn floor_snapshot_sync(&self) -> Result<LaneFloorSnapshot, DurabilityError> {
        self.round_trip(|reply| LaneCommand::FloorSnapshot { reply })
    }
}

impl CommitCoordinator {
    /// Full metrics snapshot for the admin plane (control channel).
    // Averages are f64 approximations over cumulative u64 counters — the
    // admin plane wants ratios, not exact rationals; precision beyond 2^53
    // cumulative bytes is meaningless here.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub(crate) fn metrics_snapshot(&self) -> CommitMetricsSnapshot {
        let mut sizes: Vec<u32> = self.batch_sizes.iter().copied().collect();
        sizes.sort_unstable();
        // Exact integer ranks (`ceil(num*n/den)`): no float domain at all.
        let quantile = |num: usize, den: usize| -> u64 {
            if sizes.is_empty() {
                return 0;
            }
            let rank = num
                .saturating_mul(sizes.len())
                .saturating_add(den.saturating_sub(1))
                .checked_div(den.max(1))
                .unwrap_or(sizes.len())
                .max(1)
                .min(sizes.len());
            u64::from(sizes[rank - 1])
        };
        let batches = self.batch_size_count.max(1);
        let divisor = u32::try_from(batches).unwrap_or(u32::MAX);
        CommitMetricsSnapshot {
            logical_mutations: self.logical_mutations,
            physical_batches: self.physical_batches,
            barriers: self.barriers,
            bytes_durable: self.bytes_durable,
            failed_batches: self.failed_batches,
            batch_size_avg: if batches == 0 {
                0.0
            } else {
                self.batch_size_total as f64 / batches as f64
            },
            batch_size_max: self.batch_size_max,
            batch_size_p50: quantile(1, 2),
            batch_size_p95: quantile(95, 100),
            batch_size_p99: quantile(99, 100),
            oldest_wait_avg: self.oldest_wait_total / divisor,
            oldest_wait_max: self.oldest_wait_max,
            barrier_latency_avg: self.barrier_latency_total / divisor,
            barrier_latency_max: self.barrier_latency_max,
            queue_depth: self.queue.len(),
            queue_depth_max: self.queue_depth_max,
            in_flight: self.inflight.is_some(),
        }
    }

    /// Applies every ready completion waiting on the lane channel.
    fn drain_completions(
        &mut self,
        tablets: &mut HashMap<TabletId, LiveTablet>,
        fabric: &mut crate::fabric::TabletFabric,
    ) {
        while let Ok(outcome) = self.complete.try_recv() {
            self.apply_completion(tablets, outcome, fabric);
        }
    }

    /// Applies one seal outcome: on proof, ordered verified apply plus
    /// dedup install and replies; on storage failure, position rollback
    /// and per-request failure with committed state untouched. Seal
    /// locators mirror into the fabric as durable reconstruction sources;
    /// staged ids retire when the batch fails (nothing references them).
    fn apply_completion(
        &mut self,
        tablets: &mut HashMap<TabletId, LiveTablet>,
        outcome: SealOutcome,
        fabric: &mut crate::fabric::TabletFabric,
    ) {
        let inflight = self.inflight.take().expect("completion without a batch");
        assert_eq!(
            outcome.batch_seq, inflight.batch_seq,
            "lane completion {} does not match in-flight batch {}",
            outcome.batch_seq, inflight.batch_seq
        );
        self.lane_stats = outcome.lane_stats;
        let barrier_latency = inflight.submitted_at.elapsed();
        self.barrier_latency_total += barrier_latency;
        self.barrier_latency_max = self.barrier_latency_max.max(barrier_latency);
        match outcome.result {
            Ok(proof) => {
                self.physical_batches += 1;
                self.barriers += proof.fsyncs;
                self.bytes_durable += proof.bytes_durable;
                let size = inflight.applies.len();
                self.logical_mutations += size as u64;
                self.record_batch_size(size, inflight.oldest_admitted.elapsed());
                for locator in outcome.fabric_locators {
                    fabric.note_sealed(crate::fabric::JournalEntry {
                        fabric_id: locator.fabric_id,
                        tablet: locator.tablet,
                        key: locator.key.as_bytes().to_vec(),
                        version: locator.version,
                        file: crate::fabric::JournalFile::Material,
                        offset: locator.offset,
                        len: locator.len,
                        checksum: locator.checksum,
                    });
                }
                for apply in inflight.applies {
                    self.forget_identity(apply.identity.as_ref());
                    let live = tablets.get_mut(&apply.tablet).expect("tablet live");
                    apply_item(live, apply, fabric);
                }
            }
            Err(error) => {
                self.failed_batches += 1;
                self.health = match &error {
                    DurabilityError::NoSpace { .. } => StorageHealth::ReadOnly,
                    _ => StorageHealth::Failed,
                };
                // Nothing persisted, so pre-batch positions are reusable:
                // restore them and recovery chains stay contiguous. Staged
                // fabric ids retire: no root will ever name them. Prepared
                // intents never formed here, so their pins release first
                // (otherwise the pin would outlive the intent it protects).
                for apply in &inflight.applies {
                    if let PreparedKind::Mutation {
                        mutation: Mutation::TxnPrepare { write, .. },
                        ..
                    } = &apply.kind
                        && let kivi_state::TxnWriteKind::PutFabric { fabric_id, .. } = write
                    {
                        fabric.unpin_and_sweep(*fabric_id);
                    }
                }
                for id in inflight.fabric_staged {
                    fabric.retire_or_defer(id);
                }
                for apply in &inflight.applies {
                    self.forget_identity(apply.identity.as_ref());
                    if let Some(base) = inflight.commit_base.get(&apply.tablet)
                        && let Some(live) = tablets.get_mut(&apply.tablet)
                    {
                        live.rewind_commit(*base);
                    }
                    let _ = apply
                        .respond
                        .try_send(Err(WorkerRequestError::Storage(error.clone())));
                }
            }
        }
    }

    /// Drops one applied-or-failed identity from the unapplied sets.
    fn forget_identity(&mut self, identity: Option<&MutationIdentity>) {
        let Some(marker) = identity else {
            return;
        };
        self.unapplied_identities
            .remove(&(marker.client.session(), marker.client.seq()));
        if let Some(count) = self.unapplied_sessions.get_mut(&marker.client.session()) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.unapplied_sessions.remove(&marker.client.session());
            }
        }
    }

    /// Answers everything that needs no WAL: committed-state reads,
    /// unknown tablets, and settled identities (expired, hit,
    /// overloaded). Admit-gated mutations stay queued for the prepare
    /// phase. Always runs — even with a batch in flight — because none of
    /// these observe uncommitted state. Fabric reads that miss residency
    /// suspend into the parked set (never block); later polls resume them.
    fn answer_fast_paths(
        &mut self,
        tablets: &mut HashMap<TabletId, LiveTablet>,
        fabric: &mut crate::fabric::TabletFabric,
    ) {
        let mut cursor = 0usize;
        while cursor < self.queue.len() {
            let tablet_id = self.queue[cursor].tablet;
            let Some(live) = tablets.get(&tablet_id) else {
                let entry = self.queue.remove(cursor).expect("cursor valid");
                let _ = entry
                    .respond
                    .try_send(Err(WorkerRequestError::UnknownTablet { tablet: tablet_id }));
                continue;
            };
            if !self.queue[cursor].is_mutating {
                // Committed-state reads linearize before any uncommitted
                // batch: always safe, never gated, never persisted.
                // Fabric references resolve here (synchronous hit) or
                // suspend into the parked set (promotion miss); either
                // way the queue position releases.
                let entry = self.queue.remove(cursor).expect("cursor valid");
                let live = tablets.get_mut(&tablet_id).expect("presence checked");
                match live
                    .execute(&entry.op, entry.now)
                    .map_err(WorkerRequestError::Tablet)
                {
                    Ok(kivi_state::OperationResult::FabricValue {
                        fabric_id,
                        logical_len,
                        version,
                    }) => self.answer_fast_read(
                        tablets,
                        fabric,
                        tablet_id,
                        entry,
                        kivi_state::FabricRef {
                            id: fabric_id,
                            logical_len,
                            version,
                        },
                    ),
                    outcome => {
                        let _ = entry.respond.try_send(outcome);
                    }
                }
                continue;
            }
            let gate = self.queue[cursor]
                .identity
                .as_ref()
                .map(|marker| live.check_identity(marker));
            match gate {
                // No identity, or a fresh one: the prepare phase decides
                // (needs the open batch, the tablet gate, and the exact
                // session-cap accounting).
                None | Some(IdentityGate::Admit) => {
                    cursor += 1;
                }
                Some(IdentityGate::Expired) => {
                    let entry = self.queue.remove(cursor).expect("cursor valid");
                    let _ = entry
                        .respond
                        .try_send(Err(WorkerRequestError::DedupExpired));
                }
                Some(IdentityGate::Hit { outcome, .. }) => {
                    let entry = self.queue.remove(cursor).expect("cursor valid");
                    let _ = entry.respond.try_send(map_terminal(outcome));
                }
                Some(IdentityGate::Overloaded) => {
                    let entry = self.queue.remove(cursor).expect("cursor valid");
                    let _ = entry
                        .respond
                        .try_send(Err(WorkerRequestError::SessionOverloaded));
                }
            }
        }
    }

    /// Answers one fast-path read that prepared a fabric reference:
    /// synchronous residency answers inline, a promotion miss suspends
    /// into the parked set (polled, never blocking). The entry's queue
    /// position is already released.
    fn answer_fast_read(
        &mut self,
        tablets: &mut HashMap<TabletId, LiveTablet>,
        fabric: &mut crate::fabric::TabletFabric,
        tablet: TabletId,
        entry: PendingEntry,
        reference: kivi_state::FabricRef,
    ) {
        let current = tablets
            .get(&tablet)
            .and_then(|live| live.store().get(entry.op.key(), entry.now));
        let range = read_range(&entry.op);
        match fabric.resolve_sync(&reference, current) {
            Ok(Some(bytes)) => {
                let _ = entry.respond.try_send(Ok(shape_resolved(bytes, range)));
            }
            Ok(None) => {
                // Promotion miss: submit the lane promotion and park.
                // A missing locator means the root moved mid-answer.
                match fabric.submit_parked_promote(
                    tablet,
                    entry.op.key().clone(),
                    &reference,
                    current,
                ) {
                    Ok(park) => {
                        self.suspended.push_back(SuspendedRead {
                            tablet,
                            key: entry.op.key().clone(),
                            now: entry.now,
                            range,
                            park,
                            respond: entry.respond,
                        });
                    }
                    Err(error) => {
                        let _ = entry.respond.try_send(Err(map_fabric_error(error)));
                    }
                }
            }
            Err(error) => {
                let _ = entry.respond.try_send(Err(map_fabric_error(error)));
            }
        }
    }

    /// Polls suspended promotion reads: ready replies re-fence against
    /// the live root and answer; stale roots count and fail closed.
    /// Expiry re-checks at the request's `now`: fenced bytes are only
    /// servable while the root is still live, so an expired or deleted
    /// key answers absence even when the promotion succeeded.
    fn poll_suspended(
        &mut self,
        tablets: &mut HashMap<TabletId, LiveTablet>,
        fabric: &mut crate::fabric::TabletFabric,
    ) {
        let mut cursor = 0usize;
        while cursor < self.suspended.len() {
            let tablet = self.suspended[cursor].tablet;
            let key = self.suspended[cursor].key.clone();
            let park = self.suspended[cursor].park;
            let now = self.suspended[cursor].now;
            let current = tablets
                .get(&tablet)
                .and_then(|live| live.store().get(&key, now));
            match fabric.poll_parked(park, current) {
                Ok(None) => {
                    cursor += 1;
                }
                Ok(Some(bytes)) => {
                    let suspended = self.suspended.remove(cursor).expect("cursor valid");
                    let answer = match current {
                        None => kivi_state::OperationResult::Value(None),
                        Some(_) => shape_resolved(bytes, suspended.range),
                    };
                    let _ = suspended.respond.try_send(Ok(answer));
                }
                Err(error) => {
                    let suspended = self.suspended.remove(cursor).expect("cursor valid");
                    match current {
                        // Expired or deleted while parked: absence, not a
                        // stale-completion error (the fence agrees: no
                        // root names the pinned version anymore).
                        None => {
                            let _ = suspended
                                .respond
                                .try_send(Ok(kivi_state::OperationResult::Value(None)));
                        }
                        Some(_) => {
                            let _ = suspended.respond.try_send(Err(map_fabric_error(error)));
                        }
                    }
                }
            }
        }
    }

    /// Prepares admit-gated mutations into the open batch in deterministic
    /// arrival order, then seals and submits. Runs only with no batch in
    /// flight (single-in-flight invariant): same-tablet mutations hold
    /// their positions in the fast phase above until the gate clears.
    ///
    /// Sealing rules, in order: full (`max_ops`/`max_bytes`), oldest past
    /// `max_wait`, or drained with the oldest past `linger` (lets
    /// concurrent arrivals join instead of each sealing alone). A drained
    /// young batch stays open for the next poll.
    fn prepare_mutations(
        &mut self,
        tablets: &mut HashMap<TabletId, LiveTablet>,
        namespace: NamespaceId,
        fabric: &mut crate::fabric::TabletFabric,
    ) {
        let mut cursor = 0usize;
        while cursor < self.queue.len() {
            if self.batch_is_full() {
                break;
            }
            // Fast paths above answered every read and unknown tablet, so
            // only mutations remain. Anything else is an internal
            // contract break — fail loudly, never spin.
            assert!(
                self.queue[cursor].is_mutating,
                "prepare phase reached a read: fast paths must answer those"
            );
            let tablet_id = self.queue[cursor].tablet;
            if self.is_held(tablet_id, self.queue[cursor].identity.as_ref()) {
                cursor += 1;
                continue;
            }
            // Exact session-cap accounting: live window plus this batch's
            // unapplied tail (grows as this same loop prepares).
            if let Some(marker) = self.queue[cursor].identity {
                let live = tablets.get(&tablet_id).expect("tablet live");
                let live_len = live_session_len(live, &marker);
                let unapplied = self
                    .unapplied_sessions
                    .get(&marker.client.session())
                    .copied()
                    .unwrap_or(0);
                if live_len + unapplied >= crate::tablet::SESSION_OUTCOME_CAP {
                    let entry = self.queue.remove(cursor).expect("cursor valid");
                    let _ = entry
                        .respond
                        .try_send(Err(WorkerRequestError::SessionOverloaded));
                    continue;
                }
            }
            // Prepare into the open batch (per-tablet overlay shared
            // across same-tablet entries, so hot keys batch together).
            let entry = self.queue.remove(cursor).expect("cursor valid");
            let live = tablets.get_mut(&tablet_id).expect("tablet live");
            match self.prepare_into_open(namespace, live, entry, fabric) {
                PrepareOutcome::Prepared => {}
                PrepareOutcome::RequestFailed { respond, error } => {
                    let _ = respond.try_send(Err(error));
                }
            }
        }
        if self.seal_is_due() {
            self.seal_open();
        }
    }

    /// Whether the open batch tripped a close rule: full by count or
    /// bytes, or its formation window outlasted `max_wait`. Note the
    /// window is measured from formation start, not oldest entry age — a
    /// backlogged queue must drain into the batch (sealing immediately
    /// after) rather than abort after one entry, which would serialize
    /// every backlog into size-1 batches.
    fn batch_is_full(&self) -> bool {
        match &self.open {
            None => false,
            Some(open) => {
                open.applies.len() >= self.policy.max_ops
                    || open.bytes >= self.policy.max_bytes
                    || open.formed_at.elapsed() >= self.policy.max_wait
            }
        }
    }

    /// Whether the open batch should seal now: full, past its deadline,
    /// drained past linger, or drained with only held entries left (held
    /// entries can never join this batch, so waiting out the linger would
    /// only delay it).
    fn seal_is_due(&self) -> bool {
        match &self.open {
            None => false,
            Some(open) if open.applies.is_empty() => false,
            Some(open) => {
                self.batch_is_full()
                    || open.oldest_admitted.elapsed() >= self.policy.linger
                    || !self.queue.is_empty()
            }
        }
    }

    /// Whether a mutation must hold its queue position: its tablet has a
    /// batch on the lane (per-tablet ordering), or its identity is
    /// already admitted but unapplied (a retry that will dedup-hit after
    /// apply instead of preparing twice).
    fn is_held(&self, tablet: TabletId, identity: Option<&MutationIdentity>) -> bool {
        if let Some(inflight) = &self.inflight
            && inflight.tablets.contains(&tablet)
        {
            return true;
        }
        if let Some(marker) = identity
            && self
                .unapplied_identities
                .contains(&(marker.client.session(), marker.client.seq()))
        {
            return true;
        }
        false
    }

    /// Prepares one admitted mutation into the open batch: overlay
    /// prepare, commit assign, record build. Reads never reach here.
    // One linear pipeline stage (stage overlay → assign commit → build
    // record → append); splitting it would scatter the prepare order the
    // borrow checker currently enforces in one place.
    #[allow(clippy::too_many_lines)]
    fn prepare_into_open(
        &mut self,
        namespace: NamespaceId,
        live: &mut LiveTablet,
        entry: PendingEntry,
        fabric: &mut crate::fabric::TabletFabric,
    ) -> PrepareOutcome {
        debug_assert!(entry.is_mutating);
        let tablet_id = live.id();
        // Stage the overlay against a short committed borrow, then assign
        // the commit once the immutable borrow ends (the borrow checker
        // enforces the order; preparation never observes assignment).
        let prepared = {
            let committed: &ObjectStore = live.store();
            let open = self.open.get_or_insert_with(|| OpenBatch {
                applies: Vec::new(),
                records: Vec::new(),
                bytes: 0,
                overlays: HashMap::new(),
                commit_base: HashMap::new(),
                oldest_admitted: entry.admitted_at,
                fabric_payloads: Vec::new(),
                formed_at: Instant::now(),
            });
            open.oldest_admitted = open.oldest_admitted.min(entry.admitted_at);
            let overlay = open.overlays.entry(tablet_id).or_default();
            match &entry.op {
                Operation::TxnCommitLocal { txn, writes } => {
                    overlay.prepare_local_commit(committed, *txn, writes, entry.now)
                }
                _ => overlay.prepare(committed, &entry.op, entry.now),
            }
        };
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                for staged in &entry.fabric_staged {
                    fabric.retire_or_defer(staged.fabric_id);
                }
                return PrepareOutcome::RequestFailed {
                    respond: entry.respond,
                    error: WorkerRequestError::Tablet(error),
                };
            }
        };
        let authority = live.authority();
        let open = self.open.as_mut().expect("open batch");
        open.commit_base
            .entry(tablet_id)
            .or_insert_with(|| live.next_commit());
        let commit = match live.assign_commit() {
            Ok(commit) => commit,
            Err(error) => {
                for staged in &entry.fabric_staged {
                    fabric.retire_or_defer(staged.fabric_id);
                }
                return PrepareOutcome::RequestFailed {
                    respond: entry.respond,
                    error: WorkerRequestError::Tablet(error),
                };
            }
        };
        let is_persist_expiry = matches!(entry.op, Operation::PersistExpiry { .. });
        // Fabric seal payloads are derived inside the Write arm (where the
        // mutation and expected outcome are both bound); terminal outcomes
        // reference nothing, so their staged ids retire just below.
        let mut fabric_payloads = Vec::new();
        // Prepared intents pin their staged ids until the intent resolves
        // (finalize-commit re-seals under the authoritative version,
        // finalize-abort releases): a later Commit always finds bytes.
        let mut prepare_pins: Vec<u64> = Vec::new();
        let (record, kind) = match prepared {
            StorePrepared::Read(_) => {
                // Mutating opcodes never prepare reads (the store
                // guarantees it with a panic of its own). Reaching here
                // means the store contract broke after a commit was
                // already assigned - fail the whole worker loudly rather
                // than burn a position silently. The process dies; nothing
                // persisted, so recovery stays contiguous.
                panic!("mutating opcode prepared a read after commit assignment");
            }
            StorePrepared::Write { mutation, expected } => {
                fabric_payloads =
                    fabric_payloads_for(tablet_id, &mutation, &expected, &entry.fabric_staged);
                if matches!(mutation, Mutation::TxnPrepare { .. }) {
                    prepare_pins.extend(fabric_payloads.iter().map(|payload| payload.fabric_id));
                }
                let record = WalRecord::Mutation(MutationRecord {
                    namespace,
                    tablet: tablet_id,
                    epoch: authority.epoch(),
                    guard: authority.guard(),
                    commit,
                    now: entry.now,
                    opcode: entry.opcode,
                    identity: entry.identity,
                    mutation: mutation.clone(),
                    expected: expected.clone(),
                });
                (
                    record,
                    PreparedKind::Mutation {
                        mutation,
                        expected,
                        is_persist_expiry,
                    },
                )
            }
            StorePrepared::Terminal(outcome) => {
                let record = WalRecord::Outcome(OutcomeRecord {
                    namespace,
                    tablet: tablet_id,
                    epoch: authority.epoch(),
                    guard: authority.guard(),
                    commit,
                    now: entry.now,
                    opcode: entry.opcode,
                    identity: entry.identity,
                    outcome: outcome.clone(),
                });
                (
                    record,
                    PreparedKind::Terminal {
                        outcome,
                        opcode: entry.opcode,
                    },
                )
            }
        };
        if let Some(marker) = &entry.identity {
            self.unapplied_identities
                .insert((marker.client.session(), marker.client.seq()));
            *self
                .unapplied_sessions
                .entry(marker.client.session())
                .or_insert(0) += 1;
        }
        // Fabric seal payloads: every staged id the mutation references
        // rides the seal with its predicted version; staged ids the
        // mutation dropped (unmet conditions, terminal outcomes) retire
        // immediately — nothing will ever reference them.
        let open = self.open.as_mut().expect("open batch");
        let referenced: std::collections::HashSet<u64> = fabric_payloads
            .iter()
            .map(|payload| payload.fabric_id)
            .collect();
        for staged in &entry.fabric_staged {
            if !referenced.contains(&staged.fabric_id) {
                fabric.retire_or_defer(staged.fabric_id);
            }
        }
        open.fabric_payloads.extend(fabric_payloads);
        for pinned in prepare_pins {
            fabric.pin(pinned);
        }
        open.bytes += record_wire_len(&record);
        open.records.push(record);
        open.applies.push(ApplyData {
            tablet: tablet_id,
            commit,
            kind,
            identity: entry.identity,
            now: entry.now,
            respond: entry.respond,
            pinned: entry.pinned,
        });
        PrepareOutcome::Prepared
    }

    /// Seals the open batch (if it holds anything) and submits it to the
    /// lane. An empty open batch is simply dropped.
    fn seal_open(&mut self) {
        let Some(mut open) = self.open.take() else {
            return;
        };
        if open.applies.is_empty() {
            return;
        }
        debug_assert!(self.inflight.is_none(), "one batch in flight at most");
        let batch_seq = self.next_batch_seq;
        self.next_batch_seq += 1;
        let mut tablets = HashSet::new();
        for apply in &open.applies {
            tablets.insert(apply.tablet);
        }
        // With at most one batch in flight and a 16-deep submit channel,
        // this never blocks unless the lane thread died — which fails
        // closed below instead of queuing into the void.
        let fabric_payloads = std::mem::take(&mut open.fabric_payloads);
        let fabric_staged: Vec<u64> = fabric_payloads
            .iter()
            .map(|payload| payload.fabric_id)
            .collect();
        if self
            .lane
            .submit
            .send(LaneCommand::Seal(SealJob {
                batch_seq,
                records: open.records,
                fabric_payloads,
            }))
            .is_err()
        {
            panic!("durability lane thread died with a sealed batch pending");
        }
        self.inflight = Some(InflightBatch {
            batch_seq,
            tablets,
            applies: open.applies,
            commit_base: open.commit_base,
            submitted_at: Instant::now(),
            oldest_admitted: open.oldest_admitted,
            fabric_staged,
        });
    }

    /// Records one sealed batch size plus its oldest-entry wait.
    fn record_batch_size(&mut self, size: usize, oldest_wait: Duration) {
        self.batch_size_count += 1;
        self.batch_size_total += size as u64;
        self.batch_size_max = self.batch_size_max.max(size as u64);
        if self.batch_sizes.len() >= BATCH_SIZE_WINDOW {
            self.batch_sizes.pop_front();
        }
        // Histogram window stores u32; batches are capped by policy
        // max_bytes, so saturation only triggers on absurd configs — and a
        // saturated histogram bucket beats a wrapped one.
        self.batch_sizes
            .push_back(u32::try_from(size).unwrap_or(u32::MAX));
        self.oldest_wait_total += oldest_wait;
        self.oldest_wait_max = self.oldest_wait_max.max(oldest_wait);
    }
}

/// Outcome of preparing one entry into the open batch.
enum PrepareOutcome {
    /// The entry joined the open batch.
    Prepared,
    /// Preparation failed the request (no WAL, no state change).
    RequestFailed {
        /// Where the failure goes.
        respond: Sender<WorkerResponse>,
        /// The failure.
        error: WorkerRequestError,
    },
}

/// Live retained-outcome count for one session (exact cap accounting adds
/// the batch's unapplied tail to this).
fn live_session_len(live: &LiveTablet, marker: &MutationIdentity) -> usize {
    live.dedup_state(marker.client.session())
        .map_or(0, super::tablet::SessionDedup::len)
}

/// Maps a stored terminal outcome onto a reply (dedup fast path).
fn map_terminal(outcome: kivi_state::DurableOutcome) -> WorkerResponse {
    use kivi_state::DurableOutcome;
    match outcome {
        DurableOutcome::Completed(result) => Ok(result),
        DurableOutcome::Rejected(error) => Err(WorkerRequestError::Tablet(TabletError::Op(error))),
        DurableOutcome::VersionExhausted => Err(WorkerRequestError::Tablet(TabletError::Apply(
            kivi_state::ApplyError::VersionExhausted,
        ))),
    }
}

/// Maps a fabric failure onto the request error surface: saturation is
/// retryable backpressure, everything else fails the single operation
/// closed with logical state untouched.
fn map_fabric_error(error: crate::fabric::FabricError) -> WorkerRequestError {
    match error {
        crate::fabric::FabricError::Overloaded => WorkerRequestError::SessionOverloaded,
        other => WorkerRequestError::Fabric(other),
    }
}

/// Applies one proven item in batch order: verified apply, floor advance,
/// reply.
///
/// # Panics
///
/// Panics when the post-durability apply fails: durable truth exists at
/// this point, so a normal request error would be dishonest (spec H).
/// Verification mismatch already panics inside `commit_persisted`.
fn apply_item(live: &mut LiveTablet, apply: ApplyData, fabric: &mut crate::fabric::TabletFabric) {
    // Held through the reply: the journal records this commit during the
    // apply below, so pins release only after journaling is done.
    let _pinned = apply.pinned;
    match apply.kind {
        PreparedKind::Mutation {
            mutation,
            expected,
            is_persist_expiry,
        } => {
            // Pre-image for post-apply retirement: when this mutation
            // supersedes a fabric root, the old materialization retires
            // (or defers while pinned) after success.
            let previous = live
                .store()
                .get(mutation.key(), apply.now)
                .and_then(kivi_state::StoredObject::fabric_ref);
            // Prepared-intent pins release here: commit and abort both
            // end the reservation (commit's re-sealed record now protects
            // reconstruction; aborts also retire the staged id).
            let intent_pin = match &mutation {
                Mutation::TxnFinalize { key, commit, .. } => live
                    .store()
                    .intent_for_key(key)
                    .and_then(|intent| match &intent.write.kind {
                        kivi_state::TxnWriteKind::PutFabric { fabric_id, .. } => {
                            Some((*fabric_id, *commit))
                        }
                        _ => None,
                    }),
                _ => None,
            };
            let outcome = live
                .commit_persisted(
                    &mutation,
                    &expected,
                    is_persist_expiry,
                    apply.identity.as_ref(),
                    apply.commit,
                    apply.now,
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "post-durability apply failed on tablet {} commit {}: {error:?}",
                        live.id().as_u64(),
                        apply.commit.as_u64(),
                    )
                });
            if let Some(post) = live.store().get(mutation.key(), apply.now) {
                fabric.retire_superseded(previous, post);
            }
            if let Some((fabric_id, commit)) = intent_pin {
                fabric.unpin_and_sweep(fabric_id);
                if !commit {
                    fabric.retire_or_defer(fabric_id);
                }
            }
            if let Some(marker) = &apply.identity {
                live.advance_session_floor(marker.client.session(), marker.ack_floor);
            }
            let _ = apply.respond.try_send(Ok(outcome));
        }
        PreparedKind::Terminal { outcome, opcode } => {
            let committed =
                live.commit_terminal(&outcome, apply.identity.as_ref(), apply.commit, opcode);
            if let Some(marker) = &apply.identity {
                live.advance_session_floor(marker.client.session(), marker.ack_floor);
            }
            let _ = apply.respond.try_send(map_terminal(committed));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_state::{Key, OperationResult};
    use kivi_tablet::{DirectorySnapshot, HashPrefix, PartitionRange};
    use kivi_types::{
        ClusterId, NamespaceId, NodeId, NodeIncarnation, TabletAuthority, TabletEpoch,
    };

    const NS: NamespaceId = NamespaceId::from_u64(1);
    const NOW: kivi_types::WallTimestamp = kivi_types::WallTimestamp::from_micros(1_000_000);

    fn live_tablet() -> LiveTablet {
        let tablet = TabletId::from_u64(1);
        let snapshot = DirectorySnapshot::bootstrap(
            NS,
            tablet,
            PartitionRange::Hash(HashPrefix::new(0, 0).expect("root")),
            TabletEpoch::INITIAL,
            kivi_types::WriteGuardGeneration::INITIAL,
        )
        .expect("genesis");
        let descriptor = snapshot.get(tablet).expect("descriptor").clone();
        let authority = TabletAuthority::new(
            tablet,
            TabletEpoch::INITIAL,
            kivi_types::WriteGuardGeneration::INITIAL,
        );
        LiveTablet::from_descriptor(&descriptor, authority).expect("live")
    }

    fn test_lane(dir: &std::path::Path) -> (LaneAccess, WorkerLaneStats) {
        let lane = kivi_durability::LocalWalLane::open(
            dir,
            0,
            kivi_durability::LaneIdentity {
                cluster: ClusterId::from_u128(7),
                node: NodeId::from_u64(9),
                incarnation: NodeIncarnation::INITIAL,
            },
            1024 * 1024,
        )
        .expect("lane opens");
        let stats = lane.lane_stats();
        (LaneAccess::Exclusive(lane), stats)
    }

    fn drive(
        coord: &mut CommitCoordinator,
        tablets: &mut HashMap<TabletId, LiveTablet>,
        fabric: &mut crate::fabric::TabletFabric,
    ) {
        for _ in 0..300 {
            coord.poll(tablets, NS, fabric);
            if !coord.has_pending() {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("coordinator did not drain");
    }

    fn test_fabric() -> (
        crate::fabric::TabletFabric,
        kivi_memory::OffcoreLaneGuard,
        tempfile::TempDir,
    ) {
        let dir = tempfile::tempdir().expect("scratch");
        let paths = crate::fabric::FabricPaths {
            root: dir.path().to_owned(),
        };
        let (fabric, guard) = crate::fabric::TabletFabric::open(
            kivi_types::WorkerId::from_u64(0),
            &paths,
            1 << 20,
            16 << 20,
        )
        .expect("fabric opens");
        (fabric, guard, dir)
    }

    #[test]
    fn three_counters_commit_in_one_batch() {
        let (mut fabric, _fabric_guard, _fabric_dir) = test_fabric();
        let scratch = tempfile::tempdir().expect("scratch");
        let (lane, stats) = test_lane(scratch.path());
        let mut coord = CommitCoordinator::spawn(BatchPolicy::default_policy(), lane, stats, None)
            .expect("spawn");
        let mut tablets = HashMap::new();
        tablets.insert(TabletId::from_u64(1), live_tablet());
        let mut receivers = Vec::new();
        for _ in 0..3 {
            let (respond, receive) = bounded::<WorkerResponse>(1);
            coord.admit(PendingEntry::new(
                TabletId::from_u64(1),
                Operation::CounterAdd {
                    key: Key::from("n"),
                    delta: 1,
                },
                NOW,
                None,
                respond,
            ));
            receivers.push(receive);
        }
        drive(&mut coord, &mut tablets, &mut fabric);
        for receive in receivers {
            let outcome = receive.try_recv().expect("answered");
            assert!(matches!(
                outcome,
                Ok(OperationResult::CounterUpdated { .. })
            ));
        }
        let metrics = coord.metrics_snapshot();
        assert_eq!(metrics.logical_mutations, 3);
        assert_eq!(
            metrics.physical_batches, 1,
            "one barrier for three: {metrics:?}"
        );
        assert_eq!(metrics.barriers, 1);
        assert_eq!(metrics.failed_batches, 0);
    }

    fn manual_coordinator() -> (
        CommitCoordinator,
        HashMap<TabletId, LiveTablet>,
        Receiver<LaneCommand>,
        Sender<SealOutcome>,
    ) {
        let stats = WorkerLaneStats {
            lane: 0,
            active_segment: 0,
            segments: 0,
            health: StorageHealth::Healthy,
            stats: kivi_durability::LaneStats::default(),
        };
        // Deterministic sealing: the step-by-step tests below assert exact
        // pipeline sequencing per poll, so the drain linger is zero here.
        // Linger timing itself is covered by the policy validation tests
        // and the `drive`-pumped integration test above.
        let policy = BatchPolicy {
            linger: Duration::ZERO,
            ..BatchPolicy::default_policy()
        };
        let (coord, seals, complete) = CommitCoordinator::with_manual_lane(policy, stats);
        let mut tablets = HashMap::new();
        tablets.insert(TabletId::from_u64(1), live_tablet());
        (coord, tablets, seals, complete)
    }

    fn admit_counter(
        coord: &mut CommitCoordinator,
        key: &str,
        delta: i64,
        identity: Option<MutationIdentity>,
    ) -> Receiver<WorkerResponse> {
        let (respond, receive) = bounded::<WorkerResponse>(1);
        coord.admit(PendingEntry::new(
            TabletId::from_u64(1),
            Operation::CounterAdd {
                key: Key::from(key),
                delta,
            },
            NOW,
            identity,
            respond,
        ));
        receive
    }

    fn admit_get(coord: &mut CommitCoordinator, key: &str) -> Receiver<WorkerResponse> {
        let (respond, receive) = bounded::<WorkerResponse>(1);
        coord.admit(PendingEntry::new(
            TabletId::from_u64(1),
            Operation::CounterGet {
                key: Key::from(key),
            },
            NOW,
            None,
            respond,
        ));
        receive
    }

    fn prove(
        seals: &Receiver<LaneCommand>,
        complete: &Sender<SealOutcome>,
        stats: WorkerLaneStats,
    ) -> usize {
        let LaneCommand::Seal(job) = seals.try_recv().expect("a batch sealed") else {
            panic!("expected a seal command");
        };
        let records = job.records.len();
        complete
            .send(SealOutcome {
                batch_seq: job.batch_seq,
                result: Ok(CommitProof {
                    batch_seq: job.batch_seq,
                    bytes_durable: 100,
                    fsyncs: 1,
                }),
                lane_stats: stats,
                fabric_locators: Vec::new(),
            })
            .expect("coordinator waits");
        records
    }

    fn fail(seals: &Receiver<LaneCommand>, complete: &Sender<SealOutcome>, stats: WorkerLaneStats) {
        let LaneCommand::Seal(job) = seals.try_recv().expect("a batch sealed") else {
            panic!("expected a seal command");
        };
        complete
            .send(SealOutcome {
                batch_seq: job.batch_seq,
                result: Err(DurabilityError::Io {
                    op: "test injected failure",
                    message: "injected".to_owned(),
                    code: None,
                }),
                lane_stats: stats,
                fabric_locators: Vec::new(),
            })
            .expect("coordinator waits");
    }

    fn lane_stats() -> WorkerLaneStats {
        WorkerLaneStats {
            lane: 0,
            active_segment: 1,
            segments: 1,
            health: StorageHealth::Healthy,
            stats: kivi_durability::LaneStats::default(),
        }
    }

    #[test]
    fn failed_batch_burns_no_positions_and_touches_no_state() {
        let (mut fabric, _fabric_guard, _fabric_dir) = test_fabric();
        let (mut coord, mut tablets, seals, complete) = manual_coordinator();
        let first = admit_counter(&mut coord, "n", 1, None);
        let second = admit_counter(&mut coord, "n", 1, None);
        coord.poll(&mut tablets, NS, &mut fabric);
        fail(&seals, &complete, lane_stats());
        coord.poll(&mut tablets, NS, &mut fabric);
        // Both requests fail explicitly...
        assert!(matches!(
            first.try_recv(),
            Ok(Err(WorkerRequestError::Storage(_)))
        ));
        assert!(matches!(
            second.try_recv(),
            Ok(Err(WorkerRequestError::Storage(_)))
        ));
        // ...committed state is untouched...
        let live = tablets.get(&TabletId::from_u64(1)).expect("tablet");
        assert_eq!(live.next_commit(), CommitPosition::UNASSIGNED);
        assert_eq!(live.applied_commit(), CommitPosition::UNASSIGNED);
        assert_eq!(live.store().len(), 0);
        assert_eq!(coord.metrics_snapshot().failed_batches, 1);
        // ...and the burned positions are reusable: the retry assigns 1,2
        // and recovery chains stay contiguous.
        let retry = admit_counter(&mut coord, "n", 5, None);
        coord.poll(&mut tablets, NS, &mut fabric);
        prove(&seals, &complete, lane_stats());
        coord.poll(&mut tablets, NS, &mut fabric);
        assert!(matches!(
            retry.try_recv(),
            Ok(Ok(OperationResult::CounterUpdated { value: 5, .. }))
        ));
        let live = tablets.get(&TabletId::from_u64(1)).expect("tablet");
        assert_eq!(live.next_commit().as_u64(), 1);
        assert_eq!(live.applied_commit().as_u64(), 1);
    }

    #[test]
    fn reads_answer_committed_state_while_a_batch_is_in_flight() {
        let (mut fabric, _fabric_guard, _fabric_dir) = test_fabric();
        let (mut coord, mut tablets, seals, complete) = manual_coordinator();
        let write = admit_counter(&mut coord, "n", 7, None);
        coord.poll(&mut tablets, NS, &mut fabric);
        assert_eq!(seals.len(), 1, "write sealed and submitted"); // The batch is on the (manual) lane, unproven: an arriving read
        // must answer from committed state immediately — never wait, never
        // see speculation.
        let read = admit_get(&mut coord, "n");
        coord.poll(&mut tablets, NS, &mut fabric);
        assert!(
            matches!(read.try_recv(), Ok(Ok(OperationResult::Counter(None)))),
            "committed state has no counter yet"
        );
        assert!(write.try_recv().is_err(), "write still unproven");
        // Prove the batch: the write applies afterwards, exactly once.
        prove(&seals, &complete, lane_stats());
        coord.poll(&mut tablets, NS, &mut fabric);
        assert!(matches!(
            write.try_recv(),
            Ok(Ok(OperationResult::CounterUpdated { value: 7, .. }))
        ));
        let metrics = coord.metrics_snapshot();
        assert_eq!(metrics.logical_mutations, 1);
        assert_eq!(metrics.physical_batches, 1);
    }

    #[test]
    fn same_tablet_holds_position_behind_its_inflight_batch() {
        let (mut fabric, _fabric_guard, _fabric_dir) = test_fabric();
        let (mut coord, mut tablets, seals, complete) = manual_coordinator();
        let first = admit_counter(&mut coord, "n", 1, None);
        coord.poll(&mut tablets, NS, &mut fabric);
        assert_eq!(seals.len(), 1, "first batch submitted");
        // Same tablet, still unproven: the second mutation holds its queue
        // position instead of overtaking into a racy second batch.
        let second = admit_counter(&mut coord, "n", 10, None);
        coord.poll(&mut tablets, NS, &mut fabric);
        assert!(second.try_recv().is_err(), "held behind in-flight");
        assert_eq!(seals.len(), 1, "no second seal while gated");
        prove(&seals, &complete, lane_stats());
        coord.poll(&mut tablets, NS, &mut fabric);
        // First applies; second is still queued (it forms the next batch).
        assert!(matches!(
            first.try_recv(),
            Ok(Ok(OperationResult::CounterUpdated { value: 1, .. }))
        ));
        assert!(second.try_recv().is_err(), "second not yet applied");
        // Pump until the second batch seals and prove it.
        coord.poll(&mut tablets, NS, &mut fabric);
        prove(&seals, &complete, lane_stats());
        coord.poll(&mut tablets, NS, &mut fabric);
        assert!(matches!(
            second.try_recv(),
            Ok(Ok(OperationResult::CounterUpdated { value: 11, .. }))
        ));
    }

    #[test]
    fn retry_of_inflight_identity_holds_then_dedup_hits() {
        use kivi_types::{RequestIdentity, SessionId};
        let (mut fabric, _fabric_guard, _fabric_dir) = test_fabric();
        let (mut coord, mut tablets, seals, complete) = manual_coordinator();
        let session = SessionId::from_u128(0xA1);
        let marker = MutationIdentity::new(
            RequestIdentity::new(session, kivi_types::RequestSeq::from_u64(1)),
            kivi_types::RequestSeq::from_u64(0),
        );
        let first = admit_counter(&mut coord, "n", 1, Some(marker));
        coord.poll(&mut tablets, NS, &mut fabric);
        assert_eq!(seals.len(), 1, "original submitted");
        // The reply is "lost": the client retries the same identity while
        // the original is unproven. It must hold, not prepare twice.
        let retry = admit_counter(&mut coord, "n", 1, Some(marker));
        coord.poll(&mut tablets, NS, &mut fabric);
        assert!(retry.try_recv().is_err(), "retry holds its position");
        assert_eq!(seals.len(), 1, "no duplicate seal");
        prove(&seals, &complete, lane_stats());
        coord.poll(&mut tablets, NS, &mut fabric);
        // Original applied once; the retry dedup-hits the same outcome.
        let original = first.try_recv().expect("original answered");
        let repeated = retry.try_recv().expect("retry answered");
        assert_eq!(original, repeated);
        assert!(matches!(
            original,
            Ok(OperationResult::CounterUpdated { value: 1, .. })
        ));
        let metrics = coord.metrics_snapshot();
        assert_eq!(metrics.logical_mutations, 1, "executed exactly once");
    }

    #[test]
    fn immediate_policy_seals_every_mutation_alone() {
        let (mut fabric, _fabric_guard, _fabric_dir) = test_fabric();
        let stats = lane_stats();
        let (mut coord, seals, complete) =
            CommitCoordinator::with_manual_lane(BatchPolicy::immediate(), stats);
        let mut tablets = HashMap::new();
        tablets.insert(TabletId::from_u64(1), live_tablet());
        let mut receivers = Vec::new();
        for _ in 0..3 {
            receivers.push(admit_counter(&mut coord, "n", 1, None));
        }
        // Same pipeline, max_ops 1: three seals, three proofs.
        for _ in 0..3 {
            coord.poll(&mut tablets, NS, &mut fabric);
            prove(&seals, &complete, lane_stats());
            coord.poll(&mut tablets, NS, &mut fabric);
        }
        for receive in receivers {
            assert!(receive.try_recv().is_ok(), "all answered");
        }
        let metrics = coord.metrics_snapshot();
        assert_eq!(metrics.logical_mutations, 3);
        assert_eq!(metrics.physical_batches, 3, "immediate: {metrics:?}");
        assert_eq!(metrics.barriers, 3);
        assert_eq!(metrics.batch_size_max, 1);
    }

    #[test]
    fn batch_policy_rejects_unencodable_limits() {
        assert!(BatchPolicy::default_policy().validate().is_ok());
        assert!(BatchPolicy::immediate().validate().is_ok());
        assert!(
            BatchPolicy {
                max_ops: 0,
                ..BatchPolicy::default_policy()
            }
            .validate()
            .is_err()
        );
        assert!(
            BatchPolicy {
                max_ops: BATCH_POLICY_MAX_OPS + 1,
                ..BatchPolicy::default_policy()
            }
            .validate()
            .is_err()
        );
        assert!(
            BatchPolicy {
                max_bytes: 0,
                ..BatchPolicy::default_policy()
            }
            .validate()
            .is_err()
        );
    }
}
