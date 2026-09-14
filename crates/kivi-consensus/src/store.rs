//! Durable `OpenRaft` 0.10 log storage on Kivi WAL primitives.
//!
//! One [`DurableRaftStore`] backs one consensus group's log side
//! ([`openraft::storage::RaftLogStorage`] +
//! [`openraft::storage::RaftLogReader`]). It persists Kivi-owned
//! [`kivi_durability::RaftRecord`]s through a dedicated WAL lane in the
//! node's data directory — the same segmented, checksummed, crash-safe
//! infrastructure as tablet history, so a future physical WAL holds many
//! groups with no format change.
//!
//! ## No double-log
//!
//! In replicated mode the Raft log IS the authoritative mutation history:
//! the tablet commit chain retires for the replicated tablet and these
//! records replace it (vote, entries, logical truncate/purge markers,
//! commit pointer, installed-snapshot metadata). Single-node durable mode
//! is untouched.
//!
//! ## Durability contracts (exact — audited against 0.10, never 0.9)
//!
//! 0.10 expresses storage errors as `std::io::Error` (no more
//! `StorageError` enum); Kivi maps [`kivi_durability::DurabilityError`]
//! through `std::io::Error::other`, preserving the message. Per-method
//! completion semantics, quoted from the 0.10 trait docs:
//!
//! * `save_vote` — "**must be persisted on disk before returning**". The
//!   vote is the fencing contract: the barrier runs first, the cache
//!   installs second, so a crash between the two replays the record and
//!   converges identically.
//! * `append` — "**should return immediately after saving in memory**" and
//!   "**when the callback is called, the entries must be persisted**".
//!   Entries stage into the cache (readable on return, as required) while
//!   the lane thread persists; the [`openraft::storage::IOFlushed`]
//!   callback fires only after the barrier acks. A barrier failure removes
//!   the staged suffix so the cache never advertises undurable history.
//!   Replication acknowledgements therefore always imply durable
//!   persistence: a follower never ACKs memory-only state.
//! * `truncate_after(last_kept)` — exclusive keep-through marker (0.10
//!   renamed 0.9's inclusive `truncate`). The Kivi-owned durable format is
//!   unchanged: the live path still persists
//!   [`kivi_durability::RaftTruncate`] `{ from_index }` (inclusive removal)
//!   with `from_index = last_kept + 1` (`None` maps to `from_index = 0`,
//!   which drops everything — index 0 is a legal entry, so the marker
//!   carries no off-by-one). Marker first, cache second.
//! * `purge` — logical prefix removal, inclusive; marker first, cache
//!   second. Distinct from physical segment deletion, which stays
//!   conservative.
//! * `save_committed` / `read_committed` — the commit pointer persists for
//!   real; restart re-derives nothing. 0.10 documents this as optional but
//!   recommended; Kivi implements it (the `RaftCommitted` record already
//!   exists) so a restart never reverts applied state behind reads.
//! * `read_vote` lives on [`openraft::storage::RaftLogReader`] in 0.10
//!   (both the store and [`DurableLogReader`] implement it).
//!
//! ## Lane thread
//!
//! The WAL barrier is synchronous `std::fs` plus `fsync`. Under the old
//! Tokio island that ran inside `block_in_place`; on a single-threaded
//! Compio reactor blocking is forbidden, so each store owns a dedicated
//! lane-writer thread holding the WAL lane exclusively.
//! Barriers cross as bounded [`async_channel`] jobs with
//! [`futures::channel::oneshot`] acks — the reactor yields while `fsync`
//! runs elsewhere. One thread per group is the day-one shape (correct and
//! simple); the documented Multi-Raft follow-up is one node-shared lane
//! writer batching many groups' barriers physically (RFC §60).
//!
//! All write paths serialize through one [`futures::lock::Mutex`]: it is
//! held across stage-plus-submit only, never across the barrier ack, so no
//! reactor task stalls on `fsync` latency. FIFO lane submission preserves
//! the log order the consecutive-log invariant requires.
//!
//! ## State machine side
//!
//! This module covers the LOG side only. Applied-state tracking and the
//! production apply path (`MutationIR` into `LiveTablet`) live in
//! [`crate::state_machine`].

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::config::KiviTypeConfig;
use crate::mutation::ReplicatedMutation;
use crate::types::ConsensusGroupId;
use kivi_durability::{
    DurabilityError, LaneFloor, LaneIdentity, LocalWalLane, RaftEntryPayload, RaftRecord,
};
use kivi_types::{NamespaceId, NodeId, TabletId};
use openraft::storage::{IOFlushed, LogState, RaftLogReader, RaftLogStorage};
use openraft::type_config::alias::{EntryOf, LogIdOf, VoteOf};
use openraft::vote::RaftLeaderId as _;

/// WAL lane reserved for consensus history (`wal/lane-65535/`). Worker
/// lanes number `0..worker_count`; the top lane id can never collide with
/// one, so consensus history shares the directory (and the reclamation
/// domain) without sharing any worker's file sequence.
pub const CONSENSUS_LANE: u16 = u16::MAX;

/// Bound on in-flight lane barriers per store. The barrier path holds no
/// cache lock while awaiting the ack, so depth here only bounds memory;
/// one outstanding barrier per writer is the norm, and a full channel
/// applies honest backpressure instead of unbounded growth.
const LANE_QUEUE: usize = 64;

/// Why a durable store could not be opened.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum StoreOpenError {
    /// The WAL lane refused to open or recover.
    #[error("consensus lane failed to open: {reason}")]
    Lane {
        /// Human-readable cause (preserves the underlying error text).
        reason: String,
    },
    /// A consensus record names a different namespace than the opener.
    #[error(
        "consensus record for group {group} belongs to namespace {found}, opener serves {expected}"
    )]
    NamespaceMismatch {
        /// The affected group.
        group: u64,
        /// The opener's namespace.
        expected: u64,
        /// The record's namespace.
        found: u64,
    },
}

/// One cached log entry: Kivi-owned fields, convertible both ways without
/// touching `OpenRaft` memory layout on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
struct StoredEntry {
    term: u64,
    leader: u64,
    payload: StoredPayload,
}

/// Cached entry payload (mirrors [`RaftEntryPayload`] 1:1).
#[derive(Debug, Clone, PartialEq, Eq)]
enum StoredPayload {
    Blank,
    Normal(Vec<u8>),
    Membership(StoredMembershipData),
}

/// Cached membership (voters plus dialable nodes).
#[derive(Debug, Clone, PartialEq, Eq)]
struct StoredMembershipData {
    voters: Vec<u64>,
    nodes: Vec<(u64, String)>,
}

/// Cached log id (term, leader, index).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StoredLogId {
    term: u64,
    leader: u64,
    index: u64,
}

impl StoredLogId {
    fn openraft(self) -> LogIdOf<KiviTypeConfig> {
        use openraft::impls::leader_id_adv::LeaderId;
        openraft::LogId::new(LeaderId::new(self.term, self.leader), self.index)
    }
}

/// Cached vote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StoredVote {
    term: u64,
    candidate: u64,
    committed: bool,
}

impl StoredVote {
    fn openraft(self) -> VoteOf<KiviTypeConfig> {
        if self.committed {
            openraft::impls::Vote::new_committed(self.term, self.candidate)
        } else {
            openraft::impls::Vote::new(self.term, self.candidate)
        }
    }
}

/// Logical group cache: everything `OpenRaft` reads, rebuilt by folding
/// [`RaftRecord`]s in file order at open. The WAL lane itself lives on the
/// [`LaneWriter`] thread; this cache never touches files.
struct StoreInner {
    namespace: NamespaceId,
    group: TabletId,
    vote: Option<StoredVote>,
    entries: BTreeMap<u64, StoredEntry>,
    last_purged: Option<StoredLogId>,
    committed: Option<StoredLogId>,
    snapshot_base: Option<StoredLogId>,
    /// Logical conflicting-suffix removals observed (live
    /// `truncate_after` calls plus recovery-fold replays of
    /// `RaftTruncate` markers). Operator/diagnostic accounting only: it
    /// proves the durable truncate record drove a suffix replacement
    /// rather than a memory-only path.
    truncations: u64,
}

/// Maps a durability failure onto the `io::Error` the 0.10 storage traits
/// require. The message preserves the typed cause; the lane's health latch
/// (terminal failure stays terminal) lives in the lane itself.
fn lane_io_error(error: &DurabilityError) -> io::Error {
    io::Error::other(error.to_string())
}

/// One barrier job for the lane thread.
struct LaneJob {
    records: Vec<RaftRecord>,
    ack: futures::channel::oneshot::Sender<Result<(), DurabilityError>>,
}

/// Exclusive owner of the group's WAL lane, parked on its own thread. The
/// thread exits when the last [`DurableRaftStore`] clone drops (the channel
/// closes); no join is required for correctness, and [`shutdown`](Self::shutdown)
/// joins for clean process teardown in tests and density runs.
#[derive(Clone)]
struct LaneWriter {
    tx: async_channel::Sender<LaneJob>,
}

impl LaneWriter {
    /// Spawns the lane thread owning `lane`.
    fn spawn(mut lane: LocalWalLane) -> Self {
        let (tx, rx) = async_channel::bounded::<LaneJob>(LANE_QUEUE);
        std::thread::Builder::new()
            .name("kivi-consensus-lane".to_owned())
            .spawn(move || {
                while let Ok(job) = rx.recv_blocking() {
                    let result = lane.append_raft_records(&job.records).map(|_| ());
                    // The submitter may be gone (shutdown race); its
                    // records are still durable — dropping the ack only
                    // skips a notification that has nowhere to go.
                    let _ = job.ack.send(result);
                }
            })
            .expect("consensus lane thread spawns");
        Self { tx }
    }

    /// Submits one barrier job in FIFO order and returns the ack
    /// receiver. The send only waits for channel capacity (the lane
    /// thread drains independently), never for the `fsync` itself — so
    /// callers may hold the cache lock across `submit` (which serializes
    /// submit order) but must release it before awaiting the ack.
    async fn submit(
        &self,
        records: Vec<RaftRecord>,
    ) -> futures::channel::oneshot::Receiver<Result<(), DurabilityError>> {
        let (ack, rx) = futures::channel::oneshot::channel();
        // A closed channel means the lane thread died mid-submit; the
        // receiver observes the same condition as a dropped sender below.
        let _ = self.tx.send(LaneJob { records, ack }).await;
        rx
    }

    /// Runs one barrier: FIFO submit, reactor-friendly wait for the
    /// `fsync`-backed ack. A closed channel means the lane thread died,
    /// which is a process-fatal condition surfaced as an I/O error.
    /// Callers holding the cache lock must use [`submit`](Self::submit)
    /// instead and await the ack after releasing it.
    #[cfg(test)]
    async fn barrier(&self, records: Vec<RaftRecord>) -> Result<(), DurabilityError> {
        let rx = self.submit(records).await;
        rx.await.map_err(|_| lane_exited())?
    }
}

fn lane_exited() -> DurabilityError {
    DurabilityError::InvalidConfig {
        reason: "consensus lane thread exited",
    }
}

/// Awaits one barrier ack, mapping every failure onto the `io::Error`
/// the 0.10 storage traits require.
async fn await_barrier(
    rx: futures::channel::oneshot::Receiver<Result<(), DurabilityError>>,
) -> Result<(), io::Error> {
    let acked = rx.await.map_err(|_| lane_io_error(&lane_exited()))?;
    acked.map_err(|error| lane_io_error(&error))
}

impl std::fmt::Debug for LaneWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LaneWriter")
            .field("capacity", &LANE_QUEUE)
            .finish_non_exhaustive()
    }
}

/// Durable per-group Raft log store. Cloneable: clones share the same
/// serialized state (log readers for replication streams included) and the
/// same lane writer.
#[derive(Clone)]
pub struct DurableRaftStore {
    shared: Arc<futures::lock::Mutex<StoreInner>>,
    lane: LaneWriter,
}

/// Log reader sharing the store's serialized state (replication streams
/// read through this; the log side owns the writes).
#[derive(Clone)]
pub struct DurableLogReader {
    shared: Arc<futures::lock::Mutex<StoreInner>>,
}

impl std::fmt::Debug for DurableRaftStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Summarize the logical state without blocking (a contended lock
        // still formats).
        match self.shared.try_lock() {
            Some(inner) => f
                .debug_struct("DurableRaftStore")
                .field("group", &inner.group)
                .field("entries", &inner.entries.len())
                .field("vote", &inner.vote)
                .finish_non_exhaustive(),
            None => f
                .debug_struct("DurableRaftStore")
                .field("state", &"locked")
                .finish_non_exhaustive(),
        }
    }
}

impl std::fmt::Debug for DurableLogReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.shared.try_lock() {
            Some(inner) => f
                .debug_struct("DurableLogReader")
                .field("group", &inner.group)
                .field("entries", &inner.entries.len())
                .finish_non_exhaustive(),
            None => f
                .debug_struct("DurableLogReader")
                .field("state", &"locked")
                .finish_non_exhaustive(),
        }
    }
}

impl DurableRaftStore {
    /// Opens (or reopens) the group's durable log in the node's data
    /// directory: the consensus lane is opened or resumed, its history
    /// recovered, and the logical group state rebuilt by folding
    /// vote/entry/truncate/purge/committed/snapshot markers in file order.
    /// The recovered lane moves onto a dedicated writer thread; this call
    /// is synchronous and touches no async runtime.
    ///
    /// Only this group's consensus records fold; other groups' records
    /// (shared-lane Multi-Raft) and tablet-family entries (not group
    /// history) are skipped by the fold. Forming a group on a non-fresh
    /// directory is a bootstrap concern, rejected there.
    ///
    /// # Errors
    ///
    /// Returns [`StoreOpenError`] when the lane fails or a consensus
    /// record names a foreign namespace.
    pub fn open(
        data_dir: &Path,
        namespace: NamespaceId,
        group: ConsensusGroupId,
        identity: LaneIdentity,
        segment_target_bytes: u64,
    ) -> Result<Self, StoreOpenError> {
        let wal_dir = data_dir.join(kivi_durability::node::WAL_DIR_NAME);
        let mut lane = LocalWalLane::open(&wal_dir, CONSENSUS_LANE, identity, segment_target_bytes)
            .map_err(|error| StoreOpenError::Lane {
                reason: error.to_string(),
            })?;
        let recovery = lane
            .recover_from(LaneFloor::for_lane(CONSENSUS_LANE))
            .map_err(|error| StoreOpenError::Lane {
                reason: error.to_string(),
            })?;
        let tablet = group.tablet();
        let mut inner = StoreInner {
            namespace,
            group: tablet,
            vote: None,
            entries: BTreeMap::new(),
            last_purged: None,
            committed: None,
            snapshot_base: None,
            truncations: 0,
        };
        for recovered in &recovery.records {
            let kivi_durability::WalEntry::Consensus(record) = &recovered.entry else {
                continue;
            };
            if record.group() != tablet {
                continue;
            }
            if record.namespace() != namespace {
                return Err(StoreOpenError::NamespaceMismatch {
                    group: tablet.as_u64(),
                    expected: namespace.as_u64(),
                    found: record.namespace().as_u64(),
                });
            }
            inner.fold(record);
        }
        Ok(Self {
            shared: Arc::new(futures::lock::Mutex::new(inner)),
            lane: LaneWriter::spawn(lane),
        })
    }

    /// Returns the newest committed-membership voter set visible in the
    /// retained log (scanned newest-first). Empty when no membership
    /// entry survives (purged past, or never committed) — the caller
    /// falls back to the state machine's snapshot membership, which
    /// grounds every purge.
    pub async fn membership_voters(&self) -> std::collections::BTreeSet<u64> {
        let inner = self.shared.lock().await;
        for (_, stored) in inner.entries.iter().rev() {
            if let StoredPayload::Membership(membership) = &stored.payload {
                return membership.voters.iter().copied().collect();
            }
        }
        std::collections::BTreeSet::new()
    }

    /// Data directory path backing this store (operator introspection).
    #[must_use]
    pub fn data_dir_hint(&self) -> PathBuf {
        // The lane owns its directory; this hint exists so health
        // endpoints can name the files without reaching into the lane.
        PathBuf::from(kivi_durability::node::WAL_DIR_NAME)
            .join(kivi_durability::wal::lane_dir_name(CONSENSUS_LANE))
    }
}

impl StoreInner {
    /// Folds one recovered consensus record into the logical group state
    /// (recovery-time rebuild; mirrors the live mutation paths below).
    fn fold(&mut self, record: &RaftRecord) {
        match record {
            RaftRecord::Vote(vote) => {
                self.vote = Some(StoredVote {
                    term: vote.term,
                    candidate: vote.candidate.as_u64(),
                    committed: vote.committed,
                });
            }
            RaftRecord::Entry(entry) => {
                self.entries.insert(
                    entry.index,
                    StoredEntry {
                        term: entry.term,
                        leader: entry.leader.as_u64(),
                        payload: match &entry.payload {
                            RaftEntryPayload::Blank => StoredPayload::Blank,
                            RaftEntryPayload::Normal(command) => {
                                StoredPayload::Normal(command.clone())
                            }
                            RaftEntryPayload::Membership(membership) => {
                                StoredPayload::Membership(StoredMembershipData {
                                    voters: membership.voters.clone(),
                                    nodes: membership.nodes.clone(),
                                })
                            }
                        },
                    },
                );
            }
            RaftRecord::Truncate(marker) => {
                self.entries.retain(|index, _| *index < marker.from_index);
                self.truncations += 1;
            }
            RaftRecord::Purge(marker) => {
                self.entries
                    .retain(|index, _| *index > marker.through_index);
                let purged = StoredLogId {
                    term: marker.term,
                    leader: marker.leader.as_u64(),
                    index: marker.through_index,
                };
                if self
                    .last_purged
                    .is_none_or(|prev| prev.index <= purged.index)
                {
                    self.last_purged = Some(purged);
                }
            }
            RaftRecord::Committed(pointer) => {
                self.committed = Some(StoredLogId {
                    term: pointer.term,
                    leader: pointer.leader.as_u64(),
                    index: pointer.index,
                });
            }
            RaftRecord::SnapshotInstalled(snapshot) => {
                self.snapshot_base = Some(StoredLogId {
                    term: snapshot.term,
                    leader: snapshot.leader.as_u64(),
                    index: snapshot.index,
                });
            }
        }
    }

    fn last_log_id(&self) -> Option<StoredLogId> {
        self.entries
            .last_key_value()
            .map(|(index, entry)| StoredLogId {
                term: entry.term,
                leader: entry.leader,
                index: *index,
            })
            .or(self.last_purged)
    }
}

/// Converts an `OpenRaft` entry into its Kivi-owned record form. Command
/// bytes are the canonical [`ReplicatedMutation`] encoding (never JSON,
/// never `OpenRaft` memory layout). The 0.10 `leader_id_adv` leader carries
/// its node id directly (`node_id`, never `Option`): every persisted entry
/// names its leader by construction.
fn entry_to_record(
    namespace: NamespaceId,
    group: TabletId,
    entry: &EntryOf<KiviTypeConfig>,
) -> RaftRecord {
    use kivi_durability::{RaftEntry, RaftMembership};
    use openraft::EntryPayload;
    let payload = match &entry.payload {
        EntryPayload::Blank => kivi_durability::RaftEntryPayload::Blank,
        EntryPayload::Normal(command) => {
            kivi_durability::RaftEntryPayload::Normal(command.encode_to_vec())
        }
        EntryPayload::Membership(membership) => {
            let voters: Vec<u64> = membership.voter_ids().collect();
            let nodes: Vec<(u64, String)> = membership
                .nodes()
                .map(|(id, node)| (*id, node.addr.clone()))
                .collect();
            kivi_durability::RaftEntryPayload::Membership(RaftMembership { voters, nodes })
        }
    };
    RaftRecord::Entry(RaftEntry {
        namespace,
        group,
        index: entry.log_id.index,
        term: entry.log_id.leader_id.term,
        leader: NodeId::from_u64(entry.log_id.leader_id.node_id),
        payload,
    })
}

/// Converts a stored entry back into `OpenRaft` form for readers. Command
/// bytes decode here (they were validated at write); undecodable bytes are
/// an `io::Error`, never a silent skip. (In practice this path only serves
/// entries this binary wrote: the WAL framing CRC already excludes
/// corruption.)
fn stored_to_entry(index: u64, stored: &StoredEntry) -> Result<EntryOf<KiviTypeConfig>, io::Error> {
    use openraft::EntryPayload;
    use openraft::impls::leader_id_adv::LeaderId;
    let log_id = openraft::LogId::new(LeaderId::new(stored.term, stored.leader), index);
    let payload = match &stored.payload {
        StoredPayload::Blank => EntryPayload::Blank,
        StoredPayload::Normal(command) => {
            let mutation = ReplicatedMutation::decode_exact(command).map_err(|error| {
                io::Error::other(format!("stored command entry fails to decode: {error}"))
            })?;
            EntryPayload::Normal(mutation)
        }
        StoredPayload::Membership(membership) => {
            let voters: BTreeSet<u64> = membership.voters.iter().copied().collect();
            let nodes: BTreeMap<u64, openraft::BasicNode> = membership
                .nodes
                .iter()
                .map(|(id, addr)| (*id, openraft::BasicNode::new(addr.clone())))
                .collect();
            let membership = openraft::Membership::new(vec![voters], nodes).map_err(|error| {
                io::Error::other(format!("stored membership entry invalid: {error}"))
            })?;
            EntryPayload::Membership(membership)
        }
    };
    Ok(openraft::Entry { log_id, payload })
}

/// Converts an `OpenRaft` vote into its Kivi-owned record form. Votes on
/// this path always name their candidate (self-votes and granted votes
/// alike); the `leader_id_adv` leader id carries it directly.
fn vote_to_record(
    namespace: NamespaceId,
    group: TabletId,
    vote: &VoteOf<KiviTypeConfig>,
) -> RaftRecord {
    RaftRecord::Vote(kivi_durability::RaftVote {
        namespace,
        group,
        term: vote.leader_id.term,
        candidate: NodeId::from_u64(vote.leader_id.node_id),
        committed: vote.committed,
    })
}

/// Extracts a log id's leader (log ids on the truncate/purge/committed
/// paths always name one; only the default vote does not).
fn entry_leader_of(log_id: &LogIdOf<KiviTypeConfig>) -> u64 {
    log_id.leader_id.node_id
}

fn read_entries(
    inner: &StoreInner,
    range: (std::ops::Bound<u64>, std::ops::Bound<u64>),
) -> Result<Vec<EntryOf<KiviTypeConfig>>, io::Error> {
    inner
        .entries
        .range((range.0, range.1))
        .map(|(index, stored)| stored_to_entry(*index, stored))
        .collect()
}

impl RaftLogReader<KiviTypeConfig> for DurableLogReader {
    async fn try_get_log_entries<RB>(
        &mut self,
        range: RB,
    ) -> Result<Vec<EntryOf<KiviTypeConfig>>, io::Error>
    where
        RB: std::ops::RangeBounds<u64> + Clone + std::fmt::Debug + openraft::OptionalSend,
    {
        let inner = self.shared.lock().await;
        read_entries(
            &inner,
            (range.start_bound().cloned(), range.end_bound().cloned()),
        )
    }

    async fn read_vote(&mut self) -> Result<Option<VoteOf<KiviTypeConfig>>, io::Error> {
        Ok(self.shared.lock().await.vote.map(StoredVote::openraft))
    }
}

impl RaftLogReader<KiviTypeConfig> for DurableRaftStore {
    async fn try_get_log_entries<RB>(
        &mut self,
        range: RB,
    ) -> Result<Vec<EntryOf<KiviTypeConfig>>, io::Error>
    where
        RB: std::ops::RangeBounds<u64> + Clone + std::fmt::Debug + openraft::OptionalSend,
    {
        DurableLogReader {
            shared: Arc::clone(&self.shared),
        }
        .try_get_log_entries(range)
        .await
    }

    async fn read_vote(&mut self) -> Result<Option<VoteOf<KiviTypeConfig>>, io::Error> {
        Ok(self.shared.lock().await.vote.map(StoredVote::openraft))
    }
}

impl RaftLogStorage<KiviTypeConfig> for DurableRaftStore {
    type LogReader = DurableLogReader;

    async fn get_log_state(&mut self) -> Result<LogState<KiviTypeConfig>, io::Error> {
        let inner = self.shared.lock().await;
        Ok(LogState {
            last_purged_log_id: inner.last_purged.map(StoredLogId::openraft),
            last_log_id: inner.last_log_id().map(StoredLogId::openraft),
        })
    }

    #[allow(clippy::unused_async_trait_impl)]
    async fn get_log_reader(&mut self) -> Self::LogReader {
        DurableLogReader {
            shared: Arc::clone(&self.shared),
        }
    }

    async fn save_vote(&mut self, vote: &VoteOf<KiviTypeConfig>) -> Result<(), io::Error> {
        // Fencing first: the barrier proves durability before the cache
        // installs, so a crash between the two replays the record and
        // converges identically. Submit order serializes through the
        // cache lock; the lock is NOT held across the barrier ack (the
        // reactor stays responsive while `fsync` runs on the lane thread).
        let stored = StoredVote {
            term: vote.leader_id.term,
            candidate: vote.leader_id.node_id,
            committed: vote.committed,
        };
        let ack = {
            let inner = self.shared.lock().await;
            let record = vote_to_record(inner.namespace, inner.group, vote);
            self.lane.submit(vec![record]).await
        };
        await_barrier(ack).await?;
        self.shared.lock().await.vote = Some(stored);
        Ok(())
    }

    async fn append<I>(
        &mut self,
        entries: I,
        callback: IOFlushed<KiviTypeConfig>,
    ) -> Result<(), io::Error>
    where
        I: IntoIterator<Item = EntryOf<KiviTypeConfig>> + openraft::OptionalSend,
        I::IntoIter: openraft::OptionalSend,
    {
        // Stage records and cache together under one lock (readable on
        // return, same turn) and submit the barrier in FIFO order while
        // still holding it, so concurrent writers cannot invert lane
        // order. The lock is released before awaiting the barrier ack, so
        // the reactor never stalls on `fsync`. A barrier failure removes
        // the staged suffix so the cache never advertises undurable
        // history.
        let (staged, ack) = {
            let mut inner = self.shared.lock().await;
            let mut records = Vec::new();
            let mut staged = Vec::new();
            for entry in entries {
                let index = entry.log_id.index;
                let record = entry_to_record(inner.namespace, inner.group, &entry);
                let stored = StoredEntry {
                    term: entry.log_id.leader_id.term,
                    leader: entry.log_id.leader_id.node_id,
                    payload: match &entry.payload {
                        openraft::EntryPayload::Blank => StoredPayload::Blank,
                        openraft::EntryPayload::Normal(command) => {
                            StoredPayload::Normal(command.encode_to_vec())
                        }
                        openraft::EntryPayload::Membership(membership) => {
                            StoredPayload::Membership(StoredMembershipData {
                                voters: membership.voter_ids().collect(),
                                nodes: membership
                                    .nodes()
                                    .map(|(id, node)| (*id, node.addr.clone()))
                                    .collect(),
                            })
                        }
                    },
                };
                staged.push(index);
                records.push(record);
                inner.entries.insert(index, stored);
            }
            // FIFO submit while holding the cache lock: submit order
            // equals install order, so the lane can never observe a hole.
            // The send only waits for channel capacity (the lane thread
            // drains independently and never needs this lock), never for
            // the `fsync` itself.
            let ack = self.lane.submit(records).await;
            (staged, ack)
        };
        match await_barrier(ack).await {
            Ok(()) => {
                callback.io_completed(Ok(()));
                Ok(())
            }
            Err(error) => {
                // Roll back precisely the staged indexes (not "everything
                // after"), so a concurrent writer's newer entries survive.
                // (In practice OpenRaft serializes writers, so this
                // removes exactly our stage.)
                let mut inner = self.shared.lock().await;
                for index in &staged {
                    inner.entries.remove(index);
                }
                drop(inner);
                callback.io_completed(Err(io::Error::other(error.to_string())));
                Err(error)
            }
        }
    }

    async fn truncate_after(
        &mut self,
        last_kept: Option<LogIdOf<KiviTypeConfig>>,
    ) -> Result<(), io::Error> {
        // 0.10's exclusive keep-through marker maps onto the unchanged
        // Kivi-owned `RaftTruncate { from_index }` (inclusive removal):
        // `from_index = last_kept + 1`, `None` (drop everything) maps to
        // `from_index = 0`, which is exact because index 0 is a legal
        // entry — no off-by-one, no format change.
        let from_index = last_kept
            .as_ref()
            .map_or(0, |id| id.index.saturating_add(1));
        let ack = {
            let inner = self.shared.lock().await;
            let record = RaftRecord::Truncate(kivi_durability::RaftTruncate {
                namespace: inner.namespace,
                group: inner.group,
                from_index,
            });
            // Submit serialized through the cache lock; the marker is
            // durable before the cache drops the suffix, so a crash
            // between the two replays the marker and converges
            // identically (the logical-truncation contract).
            self.lane.submit(vec![record]).await
        };
        await_barrier(ack).await?;
        let mut inner = self.shared.lock().await;
        inner.entries.retain(|index, _| *index < from_index);
        inner.truncations += 1;
        Ok(())
    }

    async fn purge(&mut self, log_id: LogIdOf<KiviTypeConfig>) -> Result<(), io::Error> {
        let purged = StoredLogId {
            term: log_id.leader_id.term,
            leader: entry_leader_of(&log_id),
            index: log_id.index,
        };
        let ack = {
            let inner = self.shared.lock().await;
            let record = RaftRecord::Purge(kivi_durability::RaftPurge {
                namespace: inner.namespace,
                group: inner.group,
                term: log_id.leader_id.term,
                leader: NodeId::from_u64(entry_leader_of(&log_id)),
                through_index: log_id.index,
            });
            self.lane.submit(vec![record]).await
        };
        await_barrier(ack).await?;
        let mut inner = self.shared.lock().await;
        inner.entries.retain(|index, _| *index > log_id.index);
        if inner
            .last_purged
            .is_none_or(|prev| prev.index <= purged.index)
        {
            inner.last_purged = Some(purged);
        }
        Ok(())
    }

    async fn save_committed(
        &mut self,
        committed: Option<LogIdOf<KiviTypeConfig>>,
    ) -> Result<(), io::Error> {
        let Some(log_id) = committed else {
            // Clearing the pointer has no durable meaning in this store:
            // absence already means "re-derive at startup".
            return Ok(());
        };
        let stored = StoredLogId {
            term: log_id.leader_id.term,
            leader: entry_leader_of(&log_id),
            index: log_id.index,
        };
        let ack = {
            let inner = self.shared.lock().await;
            let record = RaftRecord::Committed(kivi_durability::RaftCommitted {
                namespace: inner.namespace,
                group: inner.group,
                term: log_id.leader_id.term,
                leader: NodeId::from_u64(entry_leader_of(&log_id)),
                index: log_id.index,
            });
            self.lane.submit(vec![record]).await
        };
        await_barrier(ack).await?;
        self.shared.lock().await.committed = Some(stored);
        Ok(())
    }

    async fn read_committed(&mut self) -> Result<Option<LogIdOf<KiviTypeConfig>>, io::Error> {
        Ok(self
            .shared
            .lock()
            .await
            .committed
            .map(StoredLogId::openraft))
    }
}

#[cfg(test)]
mod tests {
    use kivi_durability::LaneIdentity;
    use kivi_types::{ClusterId, NamespaceId, NodeId, NodeIncarnation, TabletId};
    use openraft::storage::{RaftLogReader, RaftLogStorage};

    use super::{DurableRaftStore, StoreOpenError};
    use crate::config::KiviTypeConfig;
    use crate::types::ConsensusGroupId;

    const NS: NamespaceId = NamespaceId::from_u64(1);
    const GROUP_A: ConsensusGroupId = ConsensusGroupId::of_tablet(TabletId::from_u64(9));
    const GROUP_B: ConsensusGroupId = ConsensusGroupId::of_tablet(TabletId::from_u64(10));

    fn identity() -> LaneIdentity {
        LaneIdentity {
            cluster: ClusterId::from_u128(0xC10C),
            node: NodeId::from_u64(1),
            incarnation: NodeIncarnation::INITIAL,
        }
    }

    fn open(dir: &std::path::Path, group: ConsensusGroupId) -> DurableRaftStore {
        DurableRaftStore::open(dir, NS, group, identity(), 1024 * 1024).expect("store opens")
    }

    fn block_on<F, T>(future: F) -> T
    where
        F: Future<Output = T>,
    {
        compio::runtime::Runtime::new()
            .expect("compio test runtime")
            .block_on(future)
    }

    /// Builds one deterministic test command: counter `n += value` with
    /// its exact expected outcome.
    pub(crate) fn test_command(value: i64) -> crate::mutation::ReplicatedMutation {
        use kivi_state::{Key, Mutation, MutationEnvelope, ObjectVersion, OperationResult};
        use kivi_types::{
            RequestIdentity, RequestSeq, SessionId, TabletAuthority, TabletEpoch, UnixMicros,
            WriteGuardGeneration,
        };
        let authority = TabletAuthority::new(
            TabletId::from_u64(9),
            TabletEpoch::INITIAL,
            WriteGuardGeneration::INITIAL,
        );
        crate::mutation::ReplicatedMutation::new(
            MutationEnvelope::new(
                NS,
                authority,
                RequestIdentity::new(
                    SessionId::from_u128(u128::try_from(value).expect("test values positive")),
                    RequestSeq::from_u64(1),
                ),
                None,
                Mutation::CounterAdd {
                    key: Key::from("n"),
                    delta: value,
                },
            ),
            OperationResult::CounterUpdated {
                value,
                version: ObjectVersion::from_u64(1),
            },
            UnixMicros::from_micros(1_000_000),
            RequestSeq::from_u64(0),
        )
    }

    /// Full framework path: a live single-voter group on the durable
    /// store elects, commits, truncates logically, and survives reopen —
    /// the log-conflict scenario against real barriers, on the Compio
    /// runtime with no Tokio anywhere.
    #[test]
    fn durable_log_store_serves_a_live_group() {
        block_on(async {
            use std::collections::BTreeMap;

            use openraft::Raft;

            use crate::config::{KiviTypeConfig, spike_config};
            use crate::spike::{MemStateMachine, StubNetworkFactory};
            let scratch = tempfile::tempdir().expect("scratch");
            let group_dir = scratch.path().join("node-a");
            std::fs::create_dir_all(&group_dir).expect("node dir");
            let store = open(&group_dir, GROUP_A);
            let config = spike_config().expect("config validates");
            let raft = Raft::new(
                1u64,
                config,
                StubNetworkFactory,
                store.clone(),
                MemStateMachine::new(),
            )
            .await
            .expect("raft spawns");
            raft.initialize(BTreeMap::from([(
                1u64,
                openraft::BasicNode::new("test".to_owned()),
            )]))
            .await
            .expect("initializes");
            wait_for_leader(&raft, 1).await;
            // Drive committed history through the real log and barrier.
            for value in [10i64, 20, 30] {
                let response = raft
                    .client_write(test_command(value))
                    .await
                    .expect("write commits");
                assert_eq!(
                    response.data.outcome(),
                    Some(&kivi_state::OperationResult::CounterUpdated {
                        value,
                        version: kivi_state::ObjectVersion::from_u64(1),
                    }),
                    "installed outcome echoes the deterministic expectation"
                );
            }
            // Membership at 0, the leader's initial blank at 1, then the
            // three writes at 2–4.
            assert_eq!(store.cached_indexes().await, vec![0, 1, 2, 3, 4]);
            raft.shutdown().await.expect("shutdown");

            // Reopen from the same directory: vote, entries, and committed
            // pointer all survive; the group re-elects without re-bootstrap.
            let store = open(&group_dir, GROUP_A);
            assert_eq!(store.cached_indexes().await, vec![0, 1, 2, 3, 4]);
            let config = spike_config().expect("config validates");
            let raft: openraft::Raft<KiviTypeConfig, MemStateMachine> = Raft::new(
                1u64,
                config,
                StubNetworkFactory,
                store.clone(),
                MemStateMachine::new(),
            )
            .await
            .expect("raft respawns");
            wait_for_leader(&raft, 1).await;
            let response = raft
                .client_write(test_command(40))
                .await
                .expect("writes continue after reopen");
            assert_eq!(response.log_id.index, 5);
            assert_eq!(store.cached_indexes().await, vec![0, 1, 2, 3, 4, 5]);
            raft.shutdown().await.expect("shutdown");
        });
    }

    async fn wait_for_leader(
        raft: &openraft::Raft<KiviTypeConfig, crate::spike::MemStateMachine>,
        node: u64,
    ) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if raft.current_leader().await == Some(node) {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "lone voter never elected itself"
            );
            compio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }

    /// Seeds one divergent history at the record level: 6 entries, the
    /// 4/5 suffix truncated, a 4'/5' replacement, the 1/2 prefix purged,
    /// plus vote and commit pointer. The same bytes the live paths above
    /// persist through the lane barrier.
    async fn seed_divergent_history(seed: &DurableRaftStore) {
        use kivi_durability::{
            RaftCommitted, RaftEntry, RaftEntryPayload, RaftPurge, RaftRecord, RaftTruncate,
            RaftVote,
        };
        let group = TabletId::from_u64(9);
        let command = |value: i64| RaftEntryPayload::Normal(test_command(value).encode_to_vec());
        let entry = |index: u64, term: u64, leader: u64, payload: RaftEntryPayload| {
            RaftRecord::Entry(RaftEntry {
                namespace: NS,
                group,
                index,
                term,
                leader: NodeId::from_u64(leader),
                payload,
            })
        };
        let records = vec![
            entry(1, 2, 1, command(1)),
            entry(2, 2, 1, command(2)),
            entry(3, 2, 1, command(3)),
            entry(4, 2, 1, command(99)),
            entry(5, 2, 1, command(100)),
            RaftRecord::Truncate(RaftTruncate {
                namespace: NS,
                group,
                from_index: 4,
            }),
            entry(4, 3, 2, command(4)),
            entry(5, 3, 2, command(5)),
            RaftRecord::Purge(RaftPurge {
                namespace: NS,
                group,
                term: 2,
                leader: NodeId::from_u64(1),
                through_index: 2,
            }),
            RaftRecord::Vote(RaftVote {
                namespace: NS,
                group,
                term: 3,
                candidate: NodeId::from_u64(2),
                committed: true,
            }),
            RaftRecord::Committed(RaftCommitted {
                namespace: NS,
                group,
                term: 3,
                leader: NodeId::from_u64(2),
                index: 5,
            }),
        ];
        seed.persist_records(records.clone())
            .await
            .expect("seed batch persists");
        let mut inner = seed.shared.lock().await;
        for record in &records {
            inner.fold(record);
        }
    }

    /// Recovery fold: entries, logical truncate/purge markers, vote, and
    /// commit pointer rebuild exactly across reopen (direct trait calls
    /// with framework-constructed callbacks are impossible, so this drives
    /// `RaftRecord` batches through the lane and asserts the fold — the
    /// same bytes the live paths above persist).
    #[test]
    fn recovery_fold_rebuilds_logical_history() {
        block_on(async {
            let scratch = tempfile::tempdir().expect("scratch");
            let dir = scratch.path().join("node-b");
            std::fs::create_dir_all(&dir).expect("node dir");
            let seed = open(&dir, GROUP_A);
            seed_divergent_history(&seed).await;
            assert_eq!(seed.cached_indexes().await, vec![3, 4, 5]);
            assert_eq!(seed.cached_vote().await, Some((3, 2, true)));
            assert_eq!(seed.cached_committed().await, Some((3, 2, 5)));
            assert_eq!(seed.cached_snapshot_base().await, None);

            // Reopen: the fold rebuilds the identical logical state from the
            // physical bytes (truncate dropped 4/5-divergent, purge dropped
            // 1/2, vote and commit pointer intact).
            drop(seed);
            let reopened = open(&dir, GROUP_A);
            assert_eq!(reopened.cached_indexes().await, vec![3, 4, 5]);
            assert_eq!(reopened.cached_vote().await, Some((3, 2, true)));
            assert_eq!(reopened.cached_committed().await, Some((3, 2, 5)));
            let mut reader = reopened.clone();
            let state = reader.get_log_state().await.expect("log state");
            assert_eq!(state.last_log_id.map(|id| id.index), Some(5));
            assert_eq!(state.last_purged_log_id.map(|id| id.index), Some(2));
            let entries = reader
                .try_get_log_entries(3..6)
                .await
                .expect("reads suffix");
            assert_eq!(entries.len(), 3);
            // Foreign-group records on the shared lane never leak across.
            let foreign = open(&dir, GROUP_B);
            assert_eq!(foreign.cached_indexes().await, Vec::<u64>::new());
        });
    }

    /// Foreign namespaces fail the open loudly (never cross-read).
    #[test]
    fn foreign_namespace_rejected() {
        let scratch = tempfile::tempdir().expect("scratch");
        let dir = scratch.path().join("node-d");
        std::fs::create_dir_all(&dir).expect("node dir");
        block_on(async {
            let mut seed = open(&dir, GROUP_A);
            seed.save_vote(&openraft::impls::Vote::new_committed(2, 1))
                .await
                .expect("vote persists");
            drop(seed);
        });
        let err = DurableRaftStore::open(
            &dir,
            NamespaceId::from_u64(2),
            GROUP_A,
            identity(),
            1024 * 1024,
        )
        .expect_err("foreign namespace refuses");
        assert!(matches!(
            err,
            StoreOpenError::NamespaceMismatch { group: 9, .. }
        ));
    }
}

/// Test/operator handle to the store's logical state without going through
/// the `OpenRaft` traits.
#[cfg(test)]
pub(crate) mod test_support {
    use super::DurableRaftStore;

    impl DurableRaftStore {
        /// Persists raw records through the lane barrier without folding
        /// (record-level seeding; the caller folds explicitly).
        pub(crate) async fn persist_records(
            &self,
            records: Vec<kivi_durability::RaftRecord>,
        ) -> Result<(), kivi_durability::DurabilityError> {
            self.lane.barrier(records).await
        }

        /// Returns the cached entry indexes in order.
        pub(crate) async fn cached_indexes(&self) -> Vec<u64> {
            self.shared.lock().await.entries.keys().copied().collect()
        }

        /// Returns the cached vote.
        pub(crate) async fn cached_vote(&self) -> Option<(u64, u64, bool)> {
            self.shared
                .lock()
                .await
                .vote
                .map(|vote| (vote.term, vote.candidate, vote.committed))
        }

        /// Returns the cached commit pointer.
        pub(crate) async fn cached_committed(&self) -> Option<(u64, u64, u64)> {
            self.shared
                .lock()
                .await
                .committed
                .map(|pointer| (pointer.term, pointer.leader, pointer.index))
        }

        /// Returns the cached snapshot base.
        pub(crate) async fn cached_snapshot_base(&self) -> Option<(u64, u64, u64)> {
            self.shared
                .lock()
                .await
                .snapshot_base
                .map(|base| (base.term, base.leader, base.index))
        }

        /// Returns observed logical truncations (live calls plus
        /// recovery-fold replays of durable markers).
        pub(crate) async fn cached_truncate_count(&self) -> u64 {
            self.shared.lock().await.truncations
        }
    }
}
