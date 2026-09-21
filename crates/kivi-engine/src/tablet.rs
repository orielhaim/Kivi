//! Live tablets: runtime mutable tablet replicas owned by exactly one worker.
//!
//! A [`LiveTablet`] pairs immutable directory metadata (identity, authority)
//! with its mutable [`ObjectStore`]. It is deliberately distinct from
//! [`TabletDescriptor`]: descriptors are
//! shared routing facts, live tablets are single-owner execution state.
//! There is no lock anywhere in this path — the owning worker thread is the
//! only mutator, and execution is plain synchronous Rust once a request
//! arrives.
//!
//! Execution mirrors the future replicated path exactly: [`prepare`](LiveTablet::prepare)
//! validates, [`apply_mutation`](LiveTablet::apply_mutation) commits, and
//! [`execute`](LiveTablet::execute) runs both atomically for the single-node
//! engine. Expiry reclamation also flows through explicit [`Delete`](kivi_state::Mutation::Delete)
//! mutations via [`sweep_expired`](LiveTablet::sweep_expired), so a future
//! consensus layer can propose the same mutations without changing tablet logic.

use core::fmt;
use std::collections::{BTreeMap, HashMap};

use kivi_checkpoint::layout::BandDirty;
use kivi_state::{
    ApplyError, ApplyOutcome, DurableOutcome, Key, Mutation, ObjectStore, OpError, Operation,
    OperationResult, Prepared, StorePrepared,
};
use kivi_tablet::TabletDescriptor;
use kivi_types::{
    CommitPosition, MutationIdentity, NamespaceId, RequestSeq, SessionId, TabletAuthority,
    TabletId, WallTimestamp,
};

/// Local per-tablet counters. Plain integers behind the owner thread —
/// aggregation across workers happens outside the hot path.
///
/// Transaction and semantic counters record replicated apply truth (what
/// committed), not driver attempts: `txn_prepares` counts reserved
/// intents, `txn_local_commits` counts atomic batch commits, and the
/// `rejected_*` family counts terminal rejections that persisted nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TabletMetrics {
    /// Operations executed (reads and writes).
    pub ops_total: u64,
    /// Read operations executed.
    pub reads: u64,
    /// Mutations applied.
    pub writes: u64,
    /// Expired objects reclaimed through explicit delete mutations.
    pub expired_reclaimed: u64,
    /// Same-tablet atomic commits applied (`TxnCommitLocal`).
    pub txn_local_commits: u64,
    /// Cross-tablet prepares that reserved an intent.
    pub txn_prepares: u64,
    /// Prepares that met OCC/intent contention.
    pub txn_prepare_conflicts: u64,
    /// Intent resolutions applied (commit or abort).
    pub txn_finalizes: u64,
    /// Finalizes that met a foreign or digest-mismatched intent.
    pub txn_finalize_conflicts: u64,
    /// Commutative-counter mutations applied.
    pub commutative_ops: u64,
    /// Bounded-counter mutations applied (creates and adds).
    pub bounded_ops: u64,
    /// Escrow share moves applied.
    pub escrow_moves: u64,
    /// Semaphore mutations applied (creates, acquisitions, releases).
    pub semaphore_ops: u64,
    /// Lease acquisitions applied.
    pub lease_acquires: u64,
    /// Lease renewals applied.
    pub lease_renews: u64,
    /// Lease releases applied.
    pub lease_releases: u64,
    /// Stream mutations applied (creates, appends, trims).
    pub stream_ops: u64,
    /// Terminal `TxnConflict` rejections (nothing persisted).
    pub rejected_conflicts: u64,
    /// Terminal `BoundedExceeded` rejections.
    pub rejected_rights: u64,
    /// Terminal `SemaphoreExhausted` rejections.
    pub rejected_capacity: u64,
    /// Terminal `LeaseConflict` rejections.
    pub rejected_lease: u64,
    /// Terminal `StaleFencing` rejections.
    pub rejected_fencing: u64,
    /// Terminal `StreamFull` rejections.
    pub rejected_stream: u64,
    /// Terminal `NotFound` rejections.
    pub rejected_not_found: u64,
    /// Terminal `WrongType` rejections.
    pub rejected_wrong_type: u64,
    /// Terminal `CounterOverflow` rejections.
    pub rejected_overflow: u64,
}

impl TabletMetrics {
    /// Records one applied mutation by category. Exhaustive over
    /// variants (no wildcard): a new mutation must be counted explicitly.
    pub fn note_applied(&mut self, mutation: &kivi_state::Mutation) {
        use kivi_state::Mutation as M;
        match mutation {
            M::TxnCommitLocal { .. } => self.txn_local_commits += 1,
            M::TxnPrepare { .. } => self.txn_prepares += 1,
            M::TxnFinalize { .. } => self.txn_finalizes += 1,
            M::CommutativeAdd { .. } => self.commutative_ops += 1,
            M::BoundedCreate { .. } | M::BoundedAdd { .. } => self.bounded_ops += 1,
            M::EscrowSetShare { .. } => self.escrow_moves += 1,
            M::SemaphoreCreate { .. } | M::SemaphoreAcquire { .. } | M::SemaphoreRelease { .. } => {
                self.semaphore_ops += 1;
            }
            M::LeaseAcquire { .. } => self.lease_acquires += 1,
            M::LeaseRenew { .. } => self.lease_renews += 1,
            M::LeaseRelease { .. } => self.lease_releases += 1,
            M::StreamCreate { .. } | M::StreamAppend { .. } | M::StreamTrim { .. } => {
                self.stream_ops += 1;
            }
            M::PutBytes { .. }
            | M::Delete { .. }
            | M::CounterAdd { .. }
            | M::SetExpiry { .. }
            | M::ReplaceChunkedRoot { .. }
            | M::ReplaceFabricRoot { .. }
            | M::SpliceBytes { .. }
            | M::PutBytesWithExpiry { .. }
            | M::ReplaceChunkedRootWithExpiry { .. }
            | M::ReplaceFabricRootWithExpiry { .. } => {}
        }
    }

    /// Records one applied transaction outcome that carries no mutation
    /// of its own (prepare conflicts, finalize conflicts).
    pub fn note_txn_outcome(&mut self, outcome: &kivi_state::ApplyOutcome) {
        use kivi_state::ApplyOutcome as O;
        if outcome == &O::TxnConflict {
            self.txn_prepare_conflicts += 1;
        }
    }

    /// Records one terminal rejection by error category.
    pub fn note_rejection(&mut self, error: &kivi_state::OpError) {
        use kivi_state::OpError as E;
        match error {
            E::TxnConflict => self.rejected_conflicts += 1,
            E::BoundedExceeded => self.rejected_rights += 1,
            E::SemaphoreExhausted => self.rejected_capacity += 1,
            E::LeaseConflict => self.rejected_lease += 1,
            E::StaleFencing => self.rejected_fencing += 1,
            E::StreamFull => self.rejected_stream += 1,
            E::NotFound => self.rejected_not_found += 1,
            E::WrongType { .. } => self.rejected_wrong_type += 1,
            E::CounterOverflow => self.rejected_overflow += 1,
            E::StaleRangeBase => {}
        }
    }
}

/// Live tablet failure modes.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TabletError {
    /// The authority triple names a different tablet than constructed.
    #[error("authority {authority} does not name tablet {tablet}")]
    AuthorityMismatch {
        /// Tablet the live instance was built for.
        tablet: TabletId,
        /// Authority triple supplied.
        authority: TabletAuthority,
    },
    /// Operation validation failed; nothing was mutated.
    #[error("operation rejected: {0}")]
    Op(#[from] OpError),
    /// Mutation application failed; see the variant for what held.
    #[error("mutation failed: {0}")]
    Apply(#[from] ApplyError),
    /// Local scan rejected (no ordered index, or a malformed scan spec).
    #[error("scan rejected: {0}")]
    OpScan(#[from] kivi_state::ScanError),
    /// This tablet's commit positions are exhausted; it can accept no
    /// further durable mutations (practically unreachable: 2^64 slots).
    #[error("tablet {tablet} commit position space exhausted")]
    CommitExhausted {
        /// The affected tablet.
        tablet: TabletId,
    },
}

/// Maximum retained unacknowledged outcomes per session on one tablet.
/// Past this, new mutations for the session fail explicitly
/// (`SessionOverloaded`) instead of silently evicting an outcome that a
/// later retry might need. The client disciplines itself to 256
/// outstanding; this is the backstop, not the target.
pub const SESSION_OUTCOME_CAP: usize = 1024;

/// One retained outcome: enough to answer a retried identity byte-for-byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DedupEntry {
    /// Commit position that installed this outcome.
    pub commit: CommitPosition,
    /// Originating opcode discriminant (response shaping).
    pub opcode: u8,
    /// The recorded outcome.
    pub outcome: DurableOutcome,
}

/// One session's dedup state on one tablet: the durable floor plus the
/// retained outcomes above it. The floor survives with an empty outcome
/// map (a compact floor record); sessions are never garbage-collected in
/// this stage — correctness over reclamation, and the eventual
/// session-retention design must preserve the "never execute again" rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionDedup {
    floor: RequestSeq,
    outcomes: BTreeMap<RequestSeq, DedupEntry>,
}

impl SessionDedup {
    /// Empty state: nothing acknowledged, nothing retained.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            floor: RequestSeq::from_u64(0),
            outcomes: BTreeMap::new(),
        }
    }

    /// Advances the floor over the client's watermark, dropping outcomes
    /// the client explicitly acknowledged (and will never retry).
    pub fn advance_floor(&mut self, ack: RequestSeq) {
        if ack.as_u64() > self.floor.as_u64() {
            self.floor = ack;
            self.outcomes.retain(|seq, _| seq.as_u64() > ack.as_u64());
        }
    }

    /// Returns the retained outcome for an in-window sequence, if any.
    #[must_use]
    pub fn get(&self, seq: RequestSeq) -> Option<&DedupEntry> {
        self.outcomes.get(&seq)
    }

    /// Iterates retained outcomes in sequence order (checkpoint capture).
    pub fn outcomes(&self) -> impl Iterator<Item = (&RequestSeq, &DedupEntry)> {
        self.outcomes.iter()
    }

    /// Returns the current floor.
    #[must_use]
    pub const fn floor(&self) -> RequestSeq {
        self.floor
    }

    /// Returns the retained outcome count (window accounting).
    #[must_use]
    pub fn len(&self) -> usize {
        self.outcomes.len()
    }

    /// Whether no outcomes are retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.outcomes.is_empty()
    }
}

/// Read-only identity gate: the dedup decision without any state change.
/// The commit pipeline checks this before preparing into an open batch;
/// floor advances and outcome installs happen exactly once at apply time,
/// in batch order, so a failed batch leaves session state untouched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum IdentityGate {
    /// Fresh identity: prepare and persist normally.
    Admit,
    /// Identity already completed: reply with the stored outcome.
    Hit {
        /// The recorded outcome.
        outcome: DurableOutcome,
        /// Originating opcode discriminant.
        opcode: u8,
    },
    /// Identity at or below the floor: gone, never re-executable.
    Expired,
    /// Session window exhausted: reject before persisting anything.
    Overloaded,
}

/// Durable preparation: what the worker must persist (if anything) before
/// replying, plus the fast paths that skip persistence entirely.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DurablePrepared {
    /// Read-only outcome: reply directly, no WAL, no dedup, no commit.
    Read(OperationResult),
    /// Mutation with its predicted outcome: persist a mutation record,
    /// then apply and verify.
    Persist {
        /// Mutation to persist and apply.
        mutation: Mutation,
        /// Outcome `apply` must reproduce.
        expected: OperationResult,
        /// Retry identity with floor, if the request carried one.
        identity: Option<MutationIdentity>,
    },
    /// Terminal outcome with no mutation: persist an outcome-only record,
    /// install dedup, then reply.
    Terminal {
        /// Terminal outcome to persist and replay.
        outcome: DurableOutcome,
        /// Retry identity with floor, if the request carried one.
        identity: Option<MutationIdentity>,
    },
    /// Identity already completed: reply with the stored outcome. No WAL,
    /// no commit advance, no re-execution.
    DedupHit {
        /// The recorded outcome.
        outcome: DurableOutcome,
        /// Originating opcode discriminant.
        opcode: u8,
    },
    /// The identity fell at or below the session floor: its outcome is
    /// gone and it will never execute again. Reply `DedupExpired`
    /// without touching the WAL or the commit sequence.
    Expired,
    /// The session holds too many unacknowledged outcomes: reject before
    /// persisting anything (so no phantom execution can follow).
    Overloaded,
}

/// Checkpoint-restored tablet state: plain data the recovery loader
/// hands to [`LiveTablet::restore_checkpoint`].
#[derive(Debug, Clone)]
pub(crate) struct RestoredTablet {
    /// Restored objects (exact versions, expiries, values).
    pub objects: Vec<(Key, kivi_state::StoredObject)>,
    /// Restored sessions: floor plus retained outcomes in sequence order.
    pub sessions: Vec<RestoredSession>,
    /// Restored prepared intents (never discarded on restart).
    pub intents: Vec<kivi_state::TxnIntent>,
    /// Checkpoint cut both cursors resume from.
    pub cut: CommitPosition,
}

/// One restored session: floor plus retained outcomes in sequence order.
pub(crate) type RestoredSession = (SessionId, RequestSeq, Vec<(RequestSeq, DedupEntry)>);

/// One mutable tablet replica. `Send` (it moves between threads only at
/// construction/migration boundaries) but used from exactly one thread at a
/// time — the owner enforces that, not the type system.
pub struct LiveTablet {
    id: TabletId,
    authority: TabletAuthority,
    store: ObjectStore,
    metrics: TabletMetrics,
    next_commit: CommitPosition,
    /// Last applied commit position (`UNASSIGNED` before the first apply).
    /// Assign (prepare time) and apply (post-durability) are separate
    /// steps in the batched pipeline, so two cursors are needed: assignment
    /// reserves order, application advances truth. Recovery assigns and
    /// applies in lockstep, keeping both identical there.
    applied_commit: CommitPosition,
    dedup: HashMap<SessionId, SessionDedup>,
    /// Checkpoint dirty-band tracking (`None` in ephemeral mode, which
    /// never checkpoints: zero hot-path cost there).
    bands: Option<BandTracker>,
    /// Chunked-root commit journal: `(commit, manifest)` for every
    /// `ReplaceChunkedRoot` applied here, in commit order. Append-only
    /// history for chunk GC: the checkpoint worker prunes entries at or
    /// below the older retained cut (then checkpoint bands plus the WAL
    /// tail cover them), keeping everything newer as GC roots. One push
    /// per chunked commit only — inline traffic never touches it.
    chunked_commits: std::collections::VecDeque<(u64, kivi_types::ManifestId)>,
}

/// Dirty-band tracking for one tablet: the namespace needed to map keys
/// to physical bands plus the bitset apply paths mark.
#[derive(Debug, Clone)]
struct BandTracker {
    namespace: NamespaceId,
    dirty: BandDirty,
}

impl LiveTablet {
    /// Builds a live tablet from directory metadata and a matching authority.
    /// Starts empty; state arrives through normal mutation application
    /// (single-node: direct execution; future: consensus replay / catch-up).
    ///
    /// # Errors
    ///
    /// Returns [`TabletError::AuthorityMismatch`] if `authority` names a
    /// different tablet than `descriptor` describes.
    pub fn from_descriptor(
        descriptor: &TabletDescriptor,
        authority: TabletAuthority,
    ) -> Result<Self, TabletError> {
        if authority.tablet() != descriptor.id() {
            return Err(TabletError::AuthorityMismatch {
                tablet: descriptor.id(),
                authority,
            });
        }
        Ok(Self {
            id: descriptor.id(),
            authority,
            store: ObjectStore::new(),
            metrics: TabletMetrics::default(),
            next_commit: CommitPosition::UNASSIGNED,
            applied_commit: CommitPosition::UNASSIGNED,
            dedup: HashMap::new(),
            bands: None,
            chunked_commits: std::collections::VecDeque::new(),
        })
    }

    /// Enables checkpoint dirty-band tracking for `namespace`. Durable
    /// workers call this once at startup; ephemeral tablets never do, so
    /// their apply paths pay no hashing cost.
    pub fn enable_band_tracking(&mut self, namespace: NamespaceId) {
        self.bands = Some(BandTracker {
            namespace,
            dirty: kivi_checkpoint::layout::BandDirty::clean(),
        });
    }

    /// Returns the dirty-band bitset, if tracking is enabled.
    #[must_use]
    pub fn dirty_bands(&self) -> Option<&BandDirty> {
        self.bands.as_ref().map(|tracker| &tracker.dirty)
    }

    /// Marks the physical checkpoint band holding `key` dirty. No-op when
    /// tracking is disabled. An unmappable key (unsupported layout — only
    /// reachable if formats evolve past this build) fails closed: callers
    /// surface it rather than silently losing dirt.
    fn note_key_dirty(&mut self, key: &Key) {
        if let Some(tracker) = self.bands.as_mut() {
            match kivi_checkpoint::BandLayout::V1.band_of(tracker.namespace, key.as_bytes()) {
                Some(band) => tracker.dirty.mark(band),
                None => {
                    // Layout evolution past this build must be loud, not
                    // silently untracked. Every apply path funnels through
                    // here, so one panic site covers all mutations.
                    panic!("band layout cannot map keys: checkpoint tracking is stale");
                }
            }
        }
    }

    /// Returns the tablet identity.
    #[must_use]
    pub const fn id(&self) -> TabletId {
        self.id
    }

    /// Returns the current authority triple (for fencing checks upstream).
    #[must_use]
    pub const fn authority(&self) -> TabletAuthority {
        self.authority
    }

    /// Returns the underlying object store (read-only inspection).
    #[must_use]
    pub fn store(&self) -> &ObjectStore {
        &self.store
    }

    /// Enables or disables the ordered key index on this tablet's store
    /// (ordered namespaces enable it; hash namespaces leave it off).
    pub fn set_ordered_indexing(&mut self, enabled: bool) {
        self.store.set_ordered_indexing(enabled);
    }

    /// Executes one bounded local scan page against this tablet.
    ///
    /// # Errors
    ///
    /// Returns [`ScanError`](kivi_state::ScanError) when the tablet keeps
    /// no ordered index, or [`TabletError::Op`] for malformed specs. Never
    /// mutates.
    pub fn scan_local(
        &mut self,
        spec: &kivi_state::ScanSpec,
        now: WallTimestamp,
    ) -> Result<kivi_state::ScanPage, TabletError> {
        let page = self
            .store
            .scan_range(spec, now)
            .map_err(TabletError::OpScan)?;
        self.metrics.reads += 1;
        self.metrics.ops_total += 1;
        Ok(page)
    }

    /// Prepared transaction intents currently reserving keys on this
    /// tablet, in key order (resolver and split/merge fencing consult this:
    /// tablets with unresolved intents do not cut over).
    #[must_use]
    pub fn pending_intents(&self) -> Vec<kivi_state::TxnIntent> {
        self.store.snapshot_intents()
    }

    /// Number of prepared intents on this tablet.
    #[must_use]
    pub fn pending_intent_count(&self) -> usize {
        self.store.pending_intent_count()
    }

    /// Installs a recovered intent verbatim (checkpoint restore never
    /// discards unresolved intents).
    pub fn restore_intent(&mut self, intent: kivi_state::TxnIntent) {
        self.store.restore_intent(intent);
    }

    /// Logical telemetry at `now` for split/merge policy.
    #[must_use]
    pub fn tablet_stats(&self, now: WallTimestamp) -> kivi_state::StoreStats {
        self.store.stats(now)
    }

    /// Returns a copy of the local metrics.
    #[must_use]
    pub const fn metrics(&self) -> TabletMetrics {
        self.metrics
    }

    /// Validates an operation without mutating (replica-proposal shape).
    ///
    /// # Errors
    ///
    /// Returns [`TabletError::Op`] for validation failures.
    pub fn prepare(&self, op: &Operation, now: WallTimestamp) -> Result<Prepared, TabletError> {
        Ok(self.store.prepare(op, now)?)
    }

    /// Applies one validated mutation (replica-apply shape).
    ///
    /// # Errors
    ///
    /// Returns [`TabletError::Apply`] when versions cannot advance, counters
    /// overflow, or types diverge.
    pub fn apply_mutation(
        &mut self,
        mutation: &Mutation,
        now: WallTimestamp,
    ) -> Result<ApplyOutcome, TabletError> {
        let outcome = self.store.apply(mutation, now)?;
        self.metrics.writes += 1;
        self.metrics.ops_total += 1;
        self.metrics.note_applied(mutation);
        self.metrics.note_txn_outcome(&outcome);
        Ok(outcome)
    }

    /// Executes one native operation atomically: prepare, then apply when the
    /// operation is a write. This is the single-node fast path; consensus
    /// will split the two phases across proposer and replicas later.
    ///
    /// # Errors
    ///
    /// Returns [`TabletError`] for validation or application failures;
    /// failures never partially mutate.
    ///
    /// # Panics
    ///
    /// Panics if preparation and application disagree on the outcome shape
    /// (e.g. a `Set` yielding a counter outcome). The two phases run
    /// back-to-back on the same state with no interleaving, so disagreement
    /// would indicate an internal logic bug rather than a runtime condition;
    /// the panic makes that unmistakable instead of returning a wrong result.
    pub fn execute(
        &mut self,
        op: &Operation,
        now: WallTimestamp,
    ) -> Result<OperationResult, TabletError> {
        let prepared = self.store.prepare(op, now).map_err(|error| {
            self.metrics.note_rejection(&error);
            TabletError::Op(error)
        })?;
        match prepared {
            Prepared::Read(result) => {
                self.metrics.reads += 1;
                self.metrics.ops_total += 1;
                Ok(result)
            }
            Prepared::Write(mutation) => {
                let outcome = self.store.apply(&mutation, now)?;
                self.metrics.writes += 1;
                self.metrics.ops_total += 1;
                self.metrics.note_applied(&mutation);
                self.metrics.note_txn_outcome(&outcome);
                Ok(kivi_state::outcome_for(
                    &mutation,
                    &outcome,
                    matches!(op, Operation::PersistExpiry { .. }),
                ))
            }
        }
    }

    /// Checks one retry identity against session state without changing
    /// anything: the read-only half of the dedup decision. The commit
    /// pipeline calls this before preparing into an open batch; floor
    /// advances and outcome installs happen at apply time instead.
    pub(crate) fn check_identity(&self, marker: &MutationIdentity) -> IdentityGate {
        let Some(session) = self.dedup.get(&marker.client.session()) else {
            return IdentityGate::Admit;
        };
        // The check runs against the floor this identity would install:
        // preparation advances the floor before deciding, so an
        // adversarial (or replayed) ack at or above the sequence fails
        // closed here exactly as it would on the immediate path.
        let floor = session.floor().as_u64().max(marker.ack_floor.as_u64());
        let seq = marker.client.seq();
        if seq.as_u64() <= floor {
            // Unreachable for well-formed clients (the floor only
            // passes consecutively completed sequences, and a live
            // identity is never completed beneath itself), but a
            // violated invariant must fail closed — never re-execute.
            return IdentityGate::Expired;
        }
        if let Some(entry) = session.get(seq) {
            return IdentityGate::Hit {
                outcome: entry.outcome.clone(),
                opcode: entry.opcode,
            };
        }
        if session.len() >= SESSION_OUTCOME_CAP {
            return IdentityGate::Overloaded;
        }
        IdentityGate::Admit
    }

    /// Advances one session's floor (durable apply path only). Floor moves
    /// only forward and only over client-acknowledged sequences.
    pub(crate) fn advance_session_floor(&mut self, session: SessionId, ack: RequestSeq) {
        if let Some(state) = self.dedup.get_mut(&session) {
            state.advance_floor(ack);
        }
    }

    /// Prepares one operation for the durable pipeline: dedup lookup and
    /// floor advance first, then store preparation. Never persists, never
    /// mutates logical state — the caller persists exactly what this
    /// returns before replying.
    ///
    /// Identity `None` (embedded callers, which share fate with the
    /// process) skips dedup but still goes through the WAL in durable
    /// mode; the worker decides persistence from the returned shape.
    ///
    /// # Errors
    ///
    /// Returns [`TabletError::Op`] for read-only opcode failures only;
    /// mutating failures arrive as [`DurablePrepared::Terminal`].
    pub fn prepare_durable(
        &mut self,
        op: &Operation,
        identity: Option<&MutationIdentity>,
        now: WallTimestamp,
    ) -> Result<DurablePrepared, TabletError> {
        if let Some(marker) = identity {
            match self.check_identity(marker) {
                IdentityGate::Expired => return Ok(DurablePrepared::Expired),
                IdentityGate::Hit { outcome, opcode } => {
                    return Ok(DurablePrepared::DedupHit { outcome, opcode });
                }
                IdentityGate::Overloaded => return Ok(DurablePrepared::Overloaded),
                IdentityGate::Admit => {}
            }
            let session = self
                .dedup
                .entry(marker.client.session())
                .or_insert_with(SessionDedup::empty);
            session.advance_floor(marker.ack_floor);
        }
        match self.store.prepare_durable(op, now)? {
            StorePrepared::Read(result) => Ok(DurablePrepared::Read(result)),
            StorePrepared::Terminal(outcome) => Ok(DurablePrepared::Terminal {
                outcome,
                identity: identity.copied(),
            }),
            StorePrepared::Write { mutation, expected } => Ok(DurablePrepared::Persist {
                mutation,
                expected,
                identity: identity.copied(),
            }),
        }
    }

    /// Assigns this tablet's next commit position (exactly once per
    /// accepted durable request). The caller persists the record under the
    /// returned position before applying anything.
    ///
    /// # Errors
    ///
    /// Returns [`TabletError::CommitExhausted`] when the position space is
    /// exhausted (practically unreachable: 2^64 slots per tablet).
    pub fn assign_commit(&mut self) -> Result<CommitPosition, TabletError> {
        let next = self
            .next_commit
            .next()
            .map_err(|_| TabletError::CommitExhausted { tablet: self.id })?;
        self.next_commit = next;
        Ok(next)
    }

    /// Returns the last assigned commit position (`UNASSIGNED` before the
    /// first durable request or replay).
    #[must_use]
    pub const fn next_commit(&self) -> CommitPosition {
        self.next_commit
    }

    /// Returns the last applied commit position (`UNASSIGNED` before the
    /// first apply). Checkpoint cuts and recovery floors read this, never
    /// the assignment cursor: only applied positions are durable truth.
    #[must_use]
    pub const fn applied_commit(&self) -> CommitPosition {
        self.applied_commit
    }

    /// Advances the applied cursor after a verified apply. The caller must
    /// have applied exactly `applied.next()` — contiguity is checked by
    /// the commit paths, not here.
    fn set_applied(&mut self, commit: CommitPosition) {
        self.applied_commit = commit;
    }

    /// Rewinds the commit sequence to a pre-batch position. The commit
    /// pipeline calls this only when a sealed batch fails persistence:
    /// nothing was written, so the burned positions are safely reusable
    /// and recovery chains stay contiguous. Never called after any
    /// persistence of the rewound positions (that would fork history).
    pub(crate) fn rewind_commit(&mut self, base: CommitPosition) {
        debug_assert!(
            base.as_u64() <= self.next_commit.as_u64(),
            "commit rewind must move backwards"
        );
        self.next_commit = base;
    }

    /// Commits a persisted mutation: applies it, verifies the outcome
    /// against the persisted expectation, installs dedup, and returns the
    /// reply outcome.
    ///
    /// A verification mismatch is process-fatal (panic): durable truth now
    /// exists, so returning a normal recoverable error would be dishonest.
    /// On restart, replay applies the same durable mutation.
    ///
    /// Application must arrive in assignment order: the applied cursor
    /// advances contiguously, so a batch applies 1,2,3 — never 1,3,2.
    /// Out-of-order or repeated application is process-fatal for the same
    /// reason as a value mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`TabletError::Apply`] when application itself fails (only
    /// reachable on divergent state — same log on same state never
    /// produces it, which is exactly what the verification below guards).
    ///
    /// # Panics
    ///
    /// Panics when re-application diverges from the persisted expectation
    /// (corruption, semantic incompatibility, or a deterministic-apply
    /// bug), or when commits apply out of order.
    ///
    /// # Panics
    ///
    /// Panics when re-application diverges from the persisted expectation
    /// (corruption, semantic incompatibility, or a deterministic-apply
    /// bug).
    pub fn commit_persisted(
        &mut self,
        mutation: &Mutation,
        expected: &OperationResult,
        is_persist_expiry: bool,
        identity: Option<&MutationIdentity>,
        commit: CommitPosition,
        now: WallTimestamp,
    ) -> Result<OperationResult, TabletError> {
        assert_eq!(
            self.applied_commit
                .next()
                .expect("applied cursor cannot exhaust before assignment does"),
            commit,
            "durable commits must apply in assignment order on tablet {}",
            self.id.as_u64(),
        );
        let outcome = self.store.apply(mutation, now)?;
        let result = kivi_state::outcome_for(mutation, &outcome, is_persist_expiry);
        assert_eq!(
            &result,
            expected,
            "durable apply diverged on tablet {} commit {}: persisted {expected:?}, recomputed {result:?}",
            self.id.as_u64(),
            commit.as_u64(),
        );
        self.set_applied(commit);
        for key in mutation.keys() {
            self.note_key_dirty(key);
        }
        self.note_chunked_commit(commit, mutation);
        if let Some(marker) = identity {
            self.install_dedup(
                marker.client.session(),
                marker.client.seq(),
                DedupEntry {
                    commit,
                    opcode: kivi_protocol::mutation_opcode(mutation, is_persist_expiry).as_u8(),
                    outcome: DurableOutcome::Completed(result.clone()),
                },
            );
        }
        self.metrics.writes += 1;
        self.metrics.ops_total += 1;
        self.metrics.note_applied(mutation);
        self.metrics.note_txn_outcome(&outcome);
        Ok(result)
    }

    /// Records chunked roots installed by a commit in the GC journal.
    /// Called after every apply (live, replay, restore-conservative): roots
    /// are observed in post-apply state rather than named from the mutation,
    /// so transactionally committed roots (finalize-applied inner writes,
    /// local-commit sets) journal exactly like direct ones, and deletions
    /// journal nothing. Prepared intents journal their referenced manifests
    /// directly (no post-state root exists yet): the entry lingers past an
    /// abort until the prune bound passes — retention-longer, never
    /// shorter — so payload a later Commit needs can never be collected
    /// first. GC always sees staged data as either pinned or journaled,
    /// never neither.
    fn note_chunked_commit(&mut self, commit: CommitPosition, mutation: &Mutation) {
        if let Mutation::TxnPrepare {
            write: kivi_state::TxnWriteKind::PutChunked { manifest, .. },
            ..
        } = mutation
        {
            self.chunked_commits.push_back((commit.as_u64(), *manifest));
        }
        for key in mutation.keys() {
            if let Some(chunked) = self
                .store
                .get_stored(key)
                .and_then(kivi_state::StoredObject::chunk_ref)
            {
                self.chunked_commits
                    .push_back((commit.as_u64(), chunked.manifest));
            }
        }
    }

    /// Commits a persisted terminal outcome (no mutation to apply):
    /// installs dedup and returns the outcome for reply.
    ///
    /// # Panics
    ///
    /// Panics on out-of-order commits or an exhausted commit cursor: both
    /// are coordinator bugs that must fail closed, never silently reorder
    /// durable history.
    pub fn commit_terminal(
        &mut self,
        outcome: &DurableOutcome,
        identity: Option<&MutationIdentity>,
        commit: CommitPosition,
        opcode: u8,
    ) -> DurableOutcome {
        assert_eq!(
            self.applied_commit
                .next()
                .expect("applied cursor cannot exhaust before assignment does"),
            commit,
            "durable commits must apply in assignment order on tablet {}",
            self.id.as_u64(),
        );
        self.set_applied(commit);
        if let Some(marker) = identity {
            self.install_dedup(
                marker.client.session(),
                marker.client.seq(),
                DedupEntry {
                    commit,
                    opcode,
                    outcome: outcome.clone(),
                },
            );
        }
        self.metrics.ops_total += 1;
        if let DurableOutcome::Rejected(error) = outcome {
            self.metrics.note_rejection(error);
        }
        outcome.clone()
    }

    /// Replays one recovered mutation record: validates the commit chain,
    /// re-applies deterministically at the recorded timestamp, verifies
    /// against the recorded expectation, and installs dedup.
    ///
    /// # Errors
    ///
    /// Returns [`kivi_durability::RecoveryError`] on chain breaks or
    /// outcome divergence. Never panics on record contents.
    pub fn apply_recovered_mutation(
        &mut self,
        mutation: &Mutation,
        expected: &OperationResult,
        is_persist_expiry: bool,
        commit: CommitPosition,
        identity: Option<&MutationIdentity>,
        now: WallTimestamp,
    ) -> Result<(), kivi_durability::RecoveryError> {
        use kivi_durability::RecoveryError;
        // Recovery assigns and applies in lockstep, so the check runs
        // against the applied cursor (identical to the assignment cursor
        // here) and both advance together.
        let expected_pos = self
            .applied_commit
            .next()
            .map_err(|_| RecoveryError::ChainBreak {
                tablet: self.id.as_u64(),
                expected: u64::MAX,
                found: commit.as_u64(),
            })?;
        if commit != expected_pos {
            return Err(RecoveryError::ChainBreak {
                tablet: self.id.as_u64(),
                expected: expected_pos.as_u64(),
                found: commit.as_u64(),
            });
        }
        let outcome =
            self.store
                .apply(mutation, now)
                .map_err(|_| RecoveryError::OutcomeMismatch {
                    tablet: self.id.as_u64(),
                    commit: commit.as_u64(),
                })?;
        let result = kivi_state::outcome_for(mutation, &outcome, is_persist_expiry);
        if &result != expected {
            return Err(RecoveryError::OutcomeMismatch {
                tablet: self.id.as_u64(),
                commit: commit.as_u64(),
            });
        }
        self.next_commit = commit;
        self.set_applied(commit);
        for key in mutation.keys() {
            self.note_key_dirty(key);
        }
        self.note_chunked_commit(commit, mutation);
        self.metrics.writes += 1;
        self.metrics.ops_total += 1;
        self.metrics.note_applied(mutation);
        self.metrics.note_txn_outcome(&outcome);
        if let Some(marker) = identity {
            self.install_dedup(
                marker.client.session(),
                marker.client.seq(),
                DedupEntry {
                    commit,
                    opcode: kivi_protocol::mutation_opcode(mutation, is_persist_expiry).as_u8(),
                    outcome: DurableOutcome::Completed(result),
                },
            );
            self.advance_replay_floor(marker);
        }
        Ok(())
    }

    /// Replays one recovered outcome-only record: validates the commit
    /// chain and installs dedup without executing anything.
    ///
    /// # Errors
    ///
    /// Returns [`kivi_durability::RecoveryError`] on chain breaks.
    pub fn install_recovered_outcome(
        &mut self,
        outcome: &DurableOutcome,
        commit: CommitPosition,
        identity: Option<&MutationIdentity>,
        opcode: u8,
    ) -> Result<(), kivi_durability::RecoveryError> {
        use kivi_durability::RecoveryError;
        // Lockstep like the mutation path above: check applied, advance both.
        let expected_pos = self
            .applied_commit
            .next()
            .map_err(|_| RecoveryError::ChainBreak {
                tablet: self.id.as_u64(),
                expected: u64::MAX,
                found: commit.as_u64(),
            })?;
        if commit != expected_pos {
            return Err(RecoveryError::ChainBreak {
                tablet: self.id.as_u64(),
                expected: expected_pos.as_u64(),
                found: commit.as_u64(),
            });
        }
        self.next_commit = commit;
        self.set_applied(commit);
        if let Some(marker) = identity {
            self.install_dedup(
                marker.client.session(),
                marker.client.seq(),
                DedupEntry {
                    commit,
                    opcode,
                    outcome: outcome.clone(),
                },
            );
            self.advance_replay_floor(marker);
        }
        self.metrics.ops_total += 1;
        Ok(())
    }

    /// Records one completed identity's outcome (durable paths only).
    fn install_dedup(&mut self, session: SessionId, seq: RequestSeq, entry: DedupEntry) {
        self.dedup
            .entry(session)
            .or_insert_with(SessionDedup::empty)
            .outcomes
            .insert(seq, entry);
    }

    /// Advances one session's floor during replay from a record's ack.
    /// Identical to the live path's floor advance, so replay converges to
    /// the same retained set the running tablet held.
    fn advance_replay_floor(&mut self, marker: &MutationIdentity) {
        if let Some(session) = self.dedup.get_mut(&marker.client.session()) {
            session.advance_floor(marker.ack_floor);
        }
    }

    /// Returns one session's dedup state, if the tablet has ever seen it.
    #[must_use]
    pub fn dedup_state(&self, session: SessionId) -> Option<&SessionDedup> {
        self.dedup.get(&session)
    }

    /// Collects the chunked-commit journal, pruning entries at or below
    /// `prune_through` (the older retained cut: checkpoint bands plus the
    /// WAL floor cover them from here on). Returns the retained remainder
    /// in commit order for the checkpoint worker's GC roots.
    pub(crate) fn collect_chunked_journal(
        &mut self,
        prune_through: u64,
    ) -> Vec<(u64, kivi_types::ManifestId)> {
        while self
            .chunked_commits
            .front()
            .is_some_and(|(commit, _)| *commit <= prune_through)
        {
            self.chunked_commits.pop_front();
        }
        self.chunked_commits.iter().copied().collect()
    }

    /// Captures a logically immutable checkpoint view at the current
    /// committed cut. The view shares value bytes with committed state
    /// (cheap clones) and takes the dirty bands (clearing the live set).
    /// Open batches are definitionally excluded: only applied positions
    /// are durable truth, so capture never waits and never sees
    /// speculation. Serialization happens off the hot worker from this
    /// view — the first step of the future COW/MVCC checkpoint model.
    pub fn capture_view(&mut self, namespace: NamespaceId) -> kivi_checkpoint::TabletSnapshot {
        use kivi_checkpoint::{OutcomeCheckpoint, SessionCheckpoint};
        let mut sessions: Vec<SessionCheckpoint> = self
            .dedup
            .iter()
            .map(|(session, state)| SessionCheckpoint {
                session: *session,
                floor: state.floor(),
                outcomes: state
                    .outcomes()
                    .map(|(seq, entry)| OutcomeCheckpoint {
                        seq: *seq,
                        commit: entry.commit,
                        opcode: entry.opcode,
                        outcome: entry.outcome.clone(),
                    })
                    .collect(),
            })
            .collect();
        sessions.sort_by_key(|session| session.session.as_u128());
        let dirty = self
            .bands
            .as_mut()
            .map_or_else(BandDirty::clean, |tracker| {
                std::mem::replace(&mut tracker.dirty, BandDirty::clean())
            });
        kivi_checkpoint::TabletSnapshot {
            namespace,
            tablet: self.id,
            epoch: self.authority.epoch(),
            guard: self.authority.guard(),
            cut: self.applied_commit,
            objects: self.store.snapshot_entries(),
            sessions,
            intents: self.store.snapshot_intents(),
            dirty,
        }
    }

    /// Installs checkpoint-restored state: objects, dedup sessions,
    /// prepared intents (never discarded), and both commit cursors at the
    /// cut. Recovery calls this before replaying the WAL tail; the tail's
    /// chain validation resumes from the cut exactly as if the tablet had
    /// applied it live.
    pub(crate) fn restore_checkpoint(&mut self, restored: RestoredTablet) {
        debug_assert!(
            restored.cut.as_u64() >= self.applied_commit.as_u64(),
            "checkpoint restore must not move the cut backwards"
        );
        for (key, object) in restored.objects {
            if let Some(chunked) = object.chunk_ref() {
                // Conservative filing under the restore cut (see
                // `note_chunked_commit`): retention-longer, never shorter.
                self.chunked_commits
                    .push_back((restored.cut.as_u64(), chunked.manifest));
            }
            self.store.put_stored(key, object);
        }
        for intent in restored.intents {
            self.store.restore_intent(intent);
        }
        for (session, floor, outcomes) in restored.sessions {
            let mut state = SessionDedup::empty();
            state.advance_floor(floor);
            for (seq, entry) in outcomes {
                state.outcomes.insert(seq, entry);
            }
            self.dedup.insert(session, state);
        }
        self.next_commit = restored.cut;
        self.applied_commit = restored.cut;
    }

    /// Reclaims up to `limit` expired objects through explicit delete
    /// mutations, returning each reclaimed key with its outcome. Bounded by
    /// `limit` so reclamation never starves foreground work; the driver
    /// repeats sweeps until `collect_expired` comes back empty.
    pub fn sweep_expired(&mut self, now: WallTimestamp, limit: usize) -> Vec<(Key, ApplyOutcome)> {
        self.sweep_expired_except(now, limit, None)
    }

    /// Reclaims expired objects like [`sweep_expired`](Self::sweep_expired)
    /// but never touches keys in `skip`. The commit pipeline passes its
    /// staged keys so a sweep cannot delete a key whose prediction is
    /// already batched (which would diverge post-durability verification).
    /// Skipped keys simply wait for the next sweep.
    pub fn sweep_expired_except(
        &mut self,
        now: WallTimestamp,
        limit: usize,
        skip: Option<&std::collections::HashSet<Key>>,
    ) -> Vec<(Key, ApplyOutcome)> {
        // Over-fetch past skipped keys so a batch-staged prefix cannot
        // starve reclamation behind it; the reclaim bound below still
        // holds exactly.
        let extra = skip.map_or(0, std::collections::HashSet::len);
        let mut reclaimed = Vec::new();
        for key in self.store.collect_expired(now, limit + extra) {
            if reclaimed.len() >= limit {
                break;
            }
            if skip.is_some_and(|touched| touched.contains(&key)) {
                continue;
            }
            // Delete of a swept key cannot fail except on a prepared
            // transaction intent (which fails closed by design): intent
            // resolution owns the key until finalize, so the sweep skips
            // it and reclaims it on a later pass. Skipping is the safe
            // direction either way.
            if let Ok(outcome) = self
                .store
                .apply(&Mutation::Delete { key: key.clone() }, now)
            {
                self.metrics.writes += 1;
                self.metrics.ops_total += 1;
                self.metrics.expired_reclaimed += 1;
                self.note_key_dirty(&key);
                reclaimed.push((key, outcome));
            }
        }
        reclaimed
    }
}

impl fmt::Debug for LiveTablet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiveTablet")
            .field("id", &self.id)
            .field("authority", &self.authority)
            .field("store", &self.store)
            .field("metrics", &self.metrics)
            .field("next_commit", &self.next_commit)
            .field("applied_commit", &self.applied_commit)
            .field("dedup", &self.dedup.len())
            .field("chunked_commits", &self.chunked_commits.len())
            .field(
                "bands",
                &self
                    .bands
                    .as_ref()
                    .map(|tracker| tracker.dirty.dirty_count()),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_state::{Key, LogicalValue, ObjectVersion};
    use kivi_tablet::{DirectorySnapshot, PartitionRange};
    use kivi_types::RequestIdentity;
    use kivi_types::{NamespaceId, TabletEpoch, WriteGuardGeneration};

    const NS: NamespaceId = NamespaceId::from_u64(1);

    fn live() -> LiveTablet {
        let tablet = TabletId::from_u64(1);
        let authority =
            TabletAuthority::new(tablet, TabletEpoch::INITIAL, WriteGuardGeneration::INITIAL);
        let snapshot = DirectorySnapshot::bootstrap(
            NS,
            tablet,
            PartitionRange::Hash(kivi_tablet::HashPrefix::new(0, 0).expect("root")),
            TabletEpoch::INITIAL,
            WriteGuardGeneration::INITIAL,
        )
        .expect("genesis");
        let descriptor = snapshot.get(tablet).expect("descriptor").clone();
        LiveTablet::from_descriptor(&descriptor, authority).expect("live tablet")
    }

    const NOW: WallTimestamp = WallTimestamp::from_micros(1_000_000);

    #[test]
    fn authority_must_name_the_tablet() {
        let tablet = live();
        let wrong = TabletAuthority::new(
            TabletId::from_u64(999),
            TabletEpoch::INITIAL,
            WriteGuardGeneration::INITIAL,
        );
        let snapshot = DirectorySnapshot::bootstrap(
            NS,
            TabletId::from_u64(1),
            PartitionRange::Hash(kivi_tablet::HashPrefix::new(0, 0).expect("root")),
            TabletEpoch::INITIAL,
            WriteGuardGeneration::INITIAL,
        )
        .expect("genesis");
        let descriptor = snapshot.get(TabletId::from_u64(1)).expect("descriptor");
        assert!(matches!(
            LiveTablet::from_descriptor(descriptor, wrong),
            Err(TabletError::AuthorityMismatch { .. })
        ));
        assert_eq!(tablet.id(), TabletId::from_u64(1));
    }

    #[test]
    fn execute_runs_typed_operations_and_counts_metrics() {
        let mut tablet = live();
        tablet
            .execute(
                &Operation::Set {
                    key: Key::from("k"),
                    value: bytes::Bytes::from_static(b"v"),
                },
                NOW,
            )
            .expect("set");
        tablet
            .execute(
                &Operation::CounterAdd {
                    key: Key::from("n"),
                    delta: 3,
                },
                NOW,
            )
            .expect("add");
        tablet
            .execute(
                &Operation::Get {
                    key: Key::from("k"),
                },
                NOW,
            )
            .expect("get");
        let metrics = tablet.metrics();
        assert_eq!(metrics.ops_total, 3);
        assert_eq!(metrics.reads, 1);
        assert_eq!(metrics.writes, 2);
        assert_eq!(metrics.expired_reclaimed, 0);
    }

    #[test]
    fn prepare_apply_split_matches_direct_execute() {
        let mut direct = live();
        let mut split = live();
        let op = Operation::CounterAdd {
            key: Key::from("n"),
            delta: 7,
        };
        let expected = direct.execute(&op, NOW).expect("direct");
        let prepared = split.prepare(&op, NOW).expect("prepare");
        let Prepared::Write(mutation) = prepared else {
            panic!("counter add must prepare a mutation");
        };
        split.apply_mutation(&mutation, NOW).expect("apply");
        assert_eq!(
            split.store().snapshot_sorted(),
            direct.store().snapshot_sorted()
        );
        assert_eq!(
            expected,
            OperationResult::CounterUpdated {
                value: 7,
                version: ObjectVersion::FIRST
            }
        );
    }

    #[test]
    fn sweep_reclaims_through_explicit_mutations() {
        let mut tablet = live();
        for name in ["a", "b", "keep"] {
            tablet
                .execute(
                    &Operation::Set {
                        key: Key::from(name),
                        value: bytes::Bytes::from_static(b"v"),
                    },
                    NOW,
                )
                .expect("set");
            tablet
                .execute(
                    &Operation::ExpireAt {
                        key: Key::from(name),
                        expires_at: WallTimestamp::from_micros(10_000_000),
                    },
                    NOW,
                )
                .expect("expire");
        }
        // Keep one alive by persisting before the deadline.
        tablet
            .execute(
                &Operation::PersistExpiry {
                    key: Key::from("keep"),
                },
                NOW,
            )
            .expect("persist");
        let later = WallTimestamp::from_micros(20_000_000);
        let reclaimed = tablet.sweep_expired(later, 1);
        assert_eq!(reclaimed.len(), 1, "sweep honors its bound");
        let reclaimed = tablet.sweep_expired(later, 100);
        assert_eq!(reclaimed.len(), 1);
        assert_eq!(tablet.metrics().expired_reclaimed, 2);
        assert_eq!(tablet.store().live_count(later), 1);
        assert!(matches!(
            tablet.store().get(&Key::from("keep"), later),
            Some(object) if matches!(object.value(), LogicalValue::Bytes(_))
        ));
    }

    fn session(n: u128) -> SessionId {
        SessionId::from_u128(n)
    }

    fn identity(session_id: u128, seq: u64, ack: u64) -> MutationIdentity {
        MutationIdentity::new(
            RequestIdentity::new(session(session_id), RequestSeq::from_u64(seq)),
            RequestSeq::from_u64(ack),
        )
    }

    fn counter_add(key: &str, delta: i64) -> Operation {
        Operation::CounterAdd {
            key: Key::from(key),
            delta,
        }
    }

    /// Drives one full durable cycle in memory (prepare, assign, commit)
    /// without any WAL: the tablet logic is identical, only persistence is
    /// elided.
    fn durable_cycle(
        tablet: &mut LiveTablet,
        op: &Operation,
        marker: Option<&MutationIdentity>,
    ) -> Result<OperationResult, TabletError> {
        let prepared = tablet.prepare_durable(op, marker, NOW)?;
        match prepared {
            DurablePrepared::Read(result) => Ok(result),
            DurablePrepared::DedupHit { outcome, .. } => match outcome {
                DurableOutcome::Completed(result) => Ok(result),
                DurableOutcome::Rejected(error) => Err(TabletError::Op(error)),
                DurableOutcome::VersionExhausted => {
                    Err(TabletError::Apply(kivi_state::ApplyError::VersionExhausted))
                }
            },
            DurablePrepared::Expired | DurablePrepared::Overloaded => {
                panic!("test never fills the window")
            }
            DurablePrepared::Terminal { outcome, identity } => {
                let commit = tablet.assign_commit()?;
                tablet.commit_terminal(&outcome, identity.as_ref(), commit, 6);
                match outcome {
                    DurableOutcome::Completed(result) => Ok(result),
                    DurableOutcome::Rejected(error) => Err(TabletError::Op(error)),
                    DurableOutcome::VersionExhausted => {
                        Err(TabletError::Apply(kivi_state::ApplyError::VersionExhausted))
                    }
                }
            }
            DurablePrepared::Persist {
                mutation,
                expected,
                identity,
            } => {
                let commit = tablet.assign_commit()?;
                let is_persist = matches!(op, Operation::PersistExpiry { .. });
                tablet.commit_persisted(
                    &mutation,
                    &expected,
                    is_persist,
                    identity.as_ref(),
                    commit,
                    NOW,
                )
            }
        }
    }

    #[test]
    fn durable_retry_returns_original_outcome() {
        let mut tablet = live();
        let marker = identity(0x51, 1, 0);
        let first = durable_cycle(&mut tablet, &counter_add("n", 5), Some(&marker))
            .expect("first executes");
        assert_eq!(
            first,
            OperationResult::CounterUpdated {
                value: 5,
                version: ObjectVersion::FIRST,
            }
        );
        assert_eq!(tablet.next_commit().as_u64(), 1);
        // Same identity again: dedup hit, no re-execution, no new commit.
        let retry =
            durable_cycle(&mut tablet, &counter_add("n", 5), Some(&marker)).expect("retry hits");
        assert_eq!(retry, first);
        assert_eq!(tablet.next_commit().as_u64(), 1);
        // Counter stayed at 5, not 10.
        assert_eq!(
            tablet
                .execute(
                    &Operation::CounterGet {
                        key: Key::from("n")
                    },
                    NOW
                )
                .expect("read"),
            OperationResult::Counter(Some(5))
        );
    }

    #[test]
    fn floor_expiry_never_reexecutes() {
        let mut tablet = live();
        durable_cycle(
            &mut tablet,
            &counter_add("n", 1),
            Some(&identity(0x52, 1, 0)),
        )
        .expect("seq 1");
        durable_cycle(
            &mut tablet,
            &counter_add("n", 1),
            Some(&identity(0x52, 2, 1)),
        )
        .expect("seq 2");
        // Floor is now 1: seq 1 is gone and stays gone.
        let state = tablet.dedup_state(session(0x52)).expect("session tracked");
        assert_eq!(state.floor(), RequestSeq::from_u64(1));
        assert!(state.get(RequestSeq::from_u64(1)).is_none());
        let expired = tablet
            .prepare_durable(&counter_add("n", 1), Some(&identity(0x52, 1, 2)), NOW)
            .expect("prepares");
        assert!(
            matches!(expired, DurablePrepared::Expired),
            "stale identity fails closed, got {expired:?}"
        );
        // And the counter was incremented exactly twice.
        assert_eq!(
            tablet
                .execute(
                    &Operation::CounterGet {
                        key: Key::from("n")
                    },
                    NOW
                )
                .expect("read"),
            OperationResult::Counter(Some(2))
        );
    }

    #[test]
    fn outcome_only_terminal_is_replayable() {
        let mut tablet = live();
        // Bytes first, so the counter add is a terminal wrong-type.
        tablet
            .execute(
                &Operation::Set {
                    key: Key::from("k"),
                    value: bytes::Bytes::from_static(b"v"),
                },
                NOW,
            )
            .expect("set");
        let marker = identity(0x53, 1, 0);
        let result = durable_cycle(&mut tablet, &counter_add("k", 1), Some(&marker));
        assert!(matches!(
            result,
            Err(TabletError::Op(OpError::WrongType { .. }))
        ));
        // Retry replays the rejection without executing.
        let retry = durable_cycle(&mut tablet, &counter_add("k", 1), Some(&marker));
        assert!(matches!(
            retry,
            Err(TabletError::Op(OpError::WrongType { .. }))
        ));
        // Terminal completions also consume commit positions (uniform rule).
        assert_eq!(tablet.next_commit().as_u64(), 1);
    }

    #[test]
    fn replay_divergence_fails_instead_of_lying() {
        use kivi_durability::RecoveryError;
        // A failed replay leaves its application behind by design: in
        // production the first divergence aborts startup and the tablet is
        // never used, so rollback would be theater. Tests mirror that by
        // replaying each scenario on the commit it belongs to.
        let mut tablet = live();
        let add = Mutation::CounterAdd {
            key: Key::from("n"),
            delta: 5,
        };
        // Honest expectation replays cleanly.
        let honest = OperationResult::CounterUpdated {
            value: 5,
            version: ObjectVersion::FIRST,
        };
        tablet
            .apply_recovered_mutation(&add, &honest, false, CommitPosition::from_u64(1), None, NOW)
            .expect("honest replays");
        // ...but only once: the chain rejects a repeated position.
        let err = tablet
            .apply_recovered_mutation(&add, &honest, false, CommitPosition::from_u64(1), None, NOW)
            .expect_err("duplicate position fails");
        assert!(
            matches!(err, RecoveryError::ChainBreak { .. }),
            "got {err:?}"
        );
        // A forged expectation for the next commit fails loudly instead of
        // installing a lie: +1 onto 5 must yield 6, not 7.
        let add_one = Mutation::CounterAdd {
            key: Key::from("n"),
            delta: 1,
        };
        let forged = OperationResult::CounterUpdated {
            value: 7,
            version: ObjectVersion::from_u64(2),
        };
        let err = tablet
            .apply_recovered_mutation(
                &add_one,
                &forged,
                false,
                CommitPosition::from_u64(2),
                None,
                NOW,
            )
            .expect_err("divergence fails");
        assert!(
            matches!(err, RecoveryError::OutcomeMismatch { .. }),
            "got {err:?}"
        );
    }
}
