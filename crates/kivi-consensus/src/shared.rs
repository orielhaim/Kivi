//! Shared physical consensus durability: one WAL writer batching many groups.
//!
//! The single-tablet stage gave every [`DurableRaftStore`](crate::store::DurableRaftStore)
//! its own lane-writer thread (`kivi-consensus-lane`): one OS thread per Raft
//! group. Multi-Raft replaces that with one node-wide physical writer
//! serving every local group:
//!
//! ```text
//! Group A submissions ─┐
//! Group B submissions ─┼→ shared channel → one writer thread → one WAL lane
//! Group C submissions ─┘   (one physical batch, one fsync/barrier)
//! ```
//!
//! ## Physical vs logical
//!
//! Batching is physical only. Each [`GroupRaftStore`] keeps its own exact
//! logical cache (vote, entries, truncate/purge markers, commit pointer,
//! snapshot base) with the same per-group ordering contracts as the
//! single-group store: FIFO submission per group, marker-first/cache-second
//! for truncations, staged-suffix rollback on barrier failure. A shared
//! physical batch never creates semantic ordering dependencies between
//! groups: recovery dispatches records by [`ConsensusGroupId`] and each
//! group folds only its own history.
//!
//! ## Format
//!
//! No format change was needed: [`RaftRecord`]
//! already carries its consensus group ([`TabletId`])
//! plus namespace on every record kind (10–15 v1), and the WAL framing
//! (`kind u16, version u16, length u32`) multiplexes families already. One
//! physical WAL scan at open dispatches records by group and rebuilds each
//! group's logical state in a single recovery pass.
//!
//! ## Backpressure
//!
//! Bounded throughout: pending submissions, pending bytes, batch records,
//! batch bytes, and batch linger are all capped. A full channel applies
//! honest backpressure to the submitting group instead of unbounded memory
//! growth; a single pathological group cannot consume the writer.
//!
//! ## Failure scope
//!
//! A shared-writer failure fails closed for every group using it: barrier
//! acks carry the I/O error and each group's storage trait maps it onto
//! `io::Error`, so no group reports durable persistence it did not get.
//! Writer death (dropped channel) surfaces identically.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Duration;

use kivi_durability::{
    DurabilityError, LaneFloor, LaneIdentity, LocalWalLane, RaftEntryPayload, RaftRecord,
};
use kivi_types::{NamespaceId, NodeId, TabletId};

use crate::command::ConsensusCommand;
use crate::config::KiviTypeConfig;
use crate::store::CONSENSUS_LANE;
use crate::types::ConsensusGroupId;
use openraft::storage::{IOFlushed, LogState, RaftLogReader, RaftLogStorage};
use openraft::type_config::alias::{EntryOf, LogIdOf, VoteOf};
use openraft::vote::RaftLeaderId as _;

/// Bound on in-flight shared submissions across all groups. The barrier
/// path holds no cache lock while awaiting the ack, so depth here only
/// bounds memory; a full channel applies honest backpressure.
const SHARED_QUEUE: usize = 1024;

/// Maximum estimated bytes queued across all groups before submissions
/// block. Counts encoded-record estimates, not exact framing.
const MAX_PENDING_BYTES: usize = 64 * 1024 * 1024;

/// Maximum records combined into one physical batch.
const MAX_BATCH_RECORDS: usize = 1024;

/// Maximum estimated bytes combined into one physical batch.
const MAX_BATCH_BYTES: usize = 4 * 1024 * 1024;

/// How long a non-full batch waits for concurrent arrivals before sealing.
/// Small enough to never dominate commit latency, large enough to combine
/// cross-group work under concurrent load.
const BATCH_LINGER: Duration = Duration::from_micros(200);

/// Tuning for the shared writer. All bounds explicit; defaults suit the
/// current 3-node static clusters and density runs.
#[derive(Debug, Clone, Copy)]
pub struct SharedDurabilityConfig {
    /// In-flight submission bound (all groups combined).
    pub max_pending: usize,
    /// Queued-bytes bound (all groups combined).
    pub max_pending_bytes: usize,
    /// Records per physical batch.
    pub max_batch_records: usize,
    /// Estimated bytes per physical batch.
    pub max_batch_bytes: usize,
    /// Non-full batch linger before sealing.
    pub linger: Duration,
}

impl Default for SharedDurabilityConfig {
    fn default() -> Self {
        Self {
            max_pending: SHARED_QUEUE,
            max_pending_bytes: MAX_PENDING_BYTES,
            max_batch_records: MAX_BATCH_RECORDS,
            max_batch_bytes: MAX_BATCH_BYTES,
            linger: BATCH_LINGER,
        }
    }
}

/// Why shared durability could not be opened.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SharedOpenError {
    /// The WAL lane refused to open or recover.
    #[error("shared consensus lane failed to open: {reason}")]
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

/// Point-in-time shared-writer metrics for admin and lab benchmarking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedDurabilityMetrics {
    /// Physical batches sealed (each is one WAL batch + one barrier).
    pub physical_batches: u64,
    /// Total logical records persisted.
    pub records: u64,
    /// Total record bytes persisted (estimates).
    pub bytes: u64,
    /// Total barriers (`fsync`s): equals `physical_batches`.
    pub barriers: u64,
    /// Sum of per-batch group counts (divide by batches for mean).
    pub group_appearances: u64,
    /// Largest group count observed in one batch.
    pub max_groups_per_batch: u64,
    /// Largest record count observed in one batch.
    pub max_records_per_batch: u64,
    /// Currently queued submissions (all groups).
    pub queue_depth: u64,
    /// Currently queued bytes (estimates, all groups).
    pub queue_bytes: u64,
    /// Total barrier latency in microseconds (divide by batches for mean).
    pub barrier_latency_us: u64,
}

/// Atomic shared-writer counters. Cloneable; snapshots are point-in-time.
#[derive(Debug, Default)]
pub struct SharedDurabilityStats {
    physical_batches: AtomicU64,
    records: AtomicU64,
    bytes: AtomicU64,
    group_appearances: AtomicU64,
    max_groups_per_batch: AtomicU64,
    max_records_per_batch: AtomicU64,
    queue_depth: AtomicU64,
    queue_bytes: AtomicU64,
    barrier_latency_us: AtomicU64,
}

impl SharedDurabilityStats {
    /// Captures a point-in-time snapshot.
    #[must_use]
    pub fn snapshot(&self) -> SharedDurabilityMetrics {
        let batches = self.physical_batches.load(Ordering::Relaxed);
        SharedDurabilityMetrics {
            physical_batches: batches,
            records: self.records.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            barriers: batches,
            group_appearances: self.group_appearances.load(Ordering::Relaxed),
            max_groups_per_batch: self.max_groups_per_batch.load(Ordering::Relaxed),
            max_records_per_batch: self.max_records_per_batch.load(Ordering::Relaxed),
            queue_depth: self.queue_depth.load(Ordering::Relaxed),
            queue_bytes: self.queue_bytes.load(Ordering::Relaxed),
            barrier_latency_us: self.barrier_latency_us.load(Ordering::Relaxed),
        }
    }
}

/// One barrier job for the shared writer thread.
struct SharedJob {
    records: Vec<RaftRecord>,
    bytes: usize,
    ack: futures::channel::oneshot::Sender<Result<(), DurabilityError>>,
}

fn estimate_bytes(records: &[RaftRecord]) -> usize {
    // Cheap estimate: payload lengths plus fixed framing overhead. Exact
    // framing is computed by the WAL encoder; this only bounds memory.
    records
        .iter()
        .map(|record| match record {
            RaftRecord::Vote(_) | RaftRecord::Purge(_) | RaftRecord::Committed(_) => 64,
            RaftRecord::Entry(entry) => {
                64 + match &entry.payload {
                    RaftEntryPayload::Blank => 0,
                    RaftEntryPayload::Normal(command) => command.len(),
                    RaftEntryPayload::Membership(membership) => {
                        membership.voters.len() * 8
                            + membership
                                .nodes
                                .iter()
                                .map(|(_, addr)| addr.len() + 12)
                                .sum::<usize>()
                    }
                }
            }
            RaftRecord::Truncate(_) => 48,
            RaftRecord::SnapshotInstalled(snapshot) => 64 + snapshot.snapshot_id.len(),
        })
        .sum()
}

fn lane_exited() -> DurabilityError {
    DurabilityError::InvalidConfig {
        reason: "shared consensus lane thread exited",
    }
}

async fn await_shared_barrier(
    rx: futures::channel::oneshot::Receiver<Result<(), DurabilityError>>,
) -> Result<(), io::Error> {
    let acked = rx
        .await
        .map_err(|_| io::Error::other(lane_exited().to_string()))?;
    acked.map_err(|error| io::Error::other(error.to_string()))
}

struct SharedInner {
    tx: async_channel::Sender<SharedJob>,
    stats: SharedDurabilityStats,
    config: SharedDurabilityConfig,
}

/// Shared physical consensus durability: one WAL lane plus one writer
/// thread batching every local group's barriers. Cloneable: clones share
/// the same channel, writer, and metrics.
#[derive(Clone)]
pub struct SharedRaftDurability {
    inner: Arc<SharedInner>,
}

impl std::fmt::Debug for SharedRaftDurability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedRaftDurability")
            .field(
                "queue",
                &self.inner.stats.queue_depth.load(Ordering::Relaxed),
            )
            .finish_non_exhaustive()
    }
}

impl SharedRaftDurability {
    /// Opens (or reopens) the node-wide shared consensus lane, recovers its
    /// history once, and spawns the single writer thread. Returns the
    /// shared handle plus the recovered consensus records in file order
    /// (callers dispatch by group and fold per-group state in one pass).
    ///
    /// # Errors
    ///
    /// Returns [`SharedOpenError`] when the lane fails.
    ///
    /// # Panics
    ///
    /// Panics when the OS refuses the writer thread (process startup
    /// cannot proceed without durability).
    pub fn open(
        data_dir: &Path,
        identity: LaneIdentity,
        segment_target_bytes: u64,
        config: SharedDurabilityConfig,
    ) -> Result<(Self, Vec<RaftRecord>), SharedOpenError> {
        let wal_dir = data_dir.join(kivi_durability::node::WAL_DIR_NAME);
        let mut lane = LocalWalLane::open(&wal_dir, CONSENSUS_LANE, identity, segment_target_bytes)
            .map_err(|error| SharedOpenError::Lane {
                reason: error.to_string(),
            })?;
        let recovery = lane
            .recover_from(LaneFloor::for_lane(CONSENSUS_LANE))
            .map_err(|error| SharedOpenError::Lane {
                reason: error.to_string(),
            })?;
        let mut records = Vec::new();
        for recovered in &recovery.records {
            let kivi_durability::WalEntry::Consensus(record) = &recovered.entry else {
                continue;
            };
            records.push(record.clone());
        }
        let (tx, rx) = async_channel::bounded::<SharedJob>(config.max_pending);
        let stats = SharedDurabilityStats::default();
        // Move the lane onto the writer thread; metrics update through a
        // raw pointer-free channel: the thread owns both lane and a stats
        // handle cloned from the same Arc below.
        let handle = Self {
            inner: Arc::new(SharedInner { tx, stats, config }),
        };
        let writer_handle = Arc::clone(&handle.inner);
        std::thread::Builder::new()
            .name("kivi-consensus-shared".to_owned())
            .spawn(move || writer_main(lane, &rx, &writer_handle))
            .expect("shared consensus lane thread spawns");
        Ok((handle, records))
    }

    /// Submits one group's records for the next physical batch. FIFO per
    /// group is preserved by the caller's submit order (each group's cache
    /// lock serializes its own submits); the shared channel preserves
    /// global arrival order, which refines every per-group order.
    async fn submit(
        &self,
        records: Vec<RaftRecord>,
    ) -> futures::channel::oneshot::Receiver<Result<(), DurabilityError>> {
        let bytes = estimate_bytes(&records);
        self.inner.stats.queue_depth.fetch_add(1, Ordering::Relaxed);
        self.inner
            .stats
            .queue_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        let (ack, rx) = futures::channel::oneshot::channel();
        let _ = self
            .inner
            .tx
            .send(SharedJob {
                records,
                bytes,
                ack,
            })
            .await;
        rx
    }

    /// Records a completed submission (queue gauges only; batch metrics
    /// update on the writer thread).
    fn release(&self, bytes: usize) {
        self.inner.stats.queue_depth.fetch_sub(1, Ordering::Relaxed);
        self.inner
            .stats
            .queue_bytes
            .fetch_sub(bytes as u64, Ordering::Relaxed);
    }

    /// Returns current writer metrics.
    #[must_use]
    pub fn metrics(&self) -> SharedDurabilityMetrics {
        self.inner.stats.snapshot()
    }

    /// Data directory path backing the shared lane (operator introspection).
    #[must_use]
    pub fn data_dir_hint(&self) -> PathBuf {
        PathBuf::from(kivi_durability::node::WAL_DIR_NAME)
            .join(kivi_durability::wal::lane_dir_name(CONSENSUS_LANE))
    }
}

/// Writer thread: batches cross-group submissions into single physical
/// WAL batches (one barrier each), then completes only the callbacks whose
/// records are now durable.
fn writer_main(
    mut lane: LocalWalLane,
    rx: &async_channel::Receiver<SharedJob>,
    shared: &SharedInner,
) {
    use std::collections::HashSet;
    let config = shared.config;
    while let Ok(first) = rx.recv_blocking() {
        let batch_start = std::time::Instant::now();
        let mut jobs = vec![first];
        let mut record_count = jobs[0].records.len();
        let mut byte_count = jobs[0].bytes;
        // Opportunistic cross-group batching: drain everything already
        // queued (concurrent groups pile up while a barrier is in flight),
        // then — only when a single group is active — park for one linger
        // so a near-simultaneous arrival from another group still joins
        // the same physical batch. Bounded by record/byte caps; the first
        // job guarantees progress even under no concurrency.
        let drain_available = |jobs: &mut Vec<SharedJob>,
                               record_count: &mut usize,
                               byte_count: &mut usize| {
            while *record_count < config.max_batch_records && *byte_count < config.max_batch_bytes {
                match rx.try_recv() {
                    Ok(job) => {
                        *record_count += job.records.len();
                        *byte_count += job.bytes;
                        jobs.push(job);
                    }
                    Err(_) => break,
                }
            }
        };
        drain_available(&mut jobs, &mut record_count, &mut byte_count);
        if jobs.len() == 1 && !config.linger.is_zero() {
            std::thread::sleep(config.linger);
            drain_available(&mut jobs, &mut record_count, &mut byte_count);
        }
        let mut combined: Vec<RaftRecord> = Vec::with_capacity(record_count);
        let mut groups = HashSet::new();
        for job in &jobs {
            for record in &job.records {
                groups.insert(record.group());
            }
            combined.extend_from_slice(&job.records);
        }
        let group_count = groups.len() as u64;
        let result = lane.append_raft_records(&combined).map(|_| ());
        let elapsed_us = u64::try_from(batch_start.elapsed().as_micros().min(u128::from(u64::MAX)))
            .unwrap_or(u64::MAX);
        shared
            .stats
            .physical_batches
            .fetch_add(1, Ordering::Relaxed);
        shared
            .stats
            .records
            .fetch_add(combined.len() as u64, Ordering::Relaxed);
        shared
            .stats
            .bytes
            .fetch_add(byte_count as u64, Ordering::Relaxed);
        shared
            .stats
            .group_appearances
            .fetch_add(group_count, Ordering::Relaxed);
        shared
            .stats
            .max_groups_per_batch
            .fetch_max(group_count, Ordering::Relaxed);
        shared
            .stats
            .max_records_per_batch
            .fetch_max(combined.len() as u64, Ordering::Relaxed);
        shared
            .stats
            .barrier_latency_us
            .fetch_add(elapsed_us, Ordering::Relaxed);
        for job in jobs {
            shared.stats.queue_depth.fetch_sub(1, Ordering::Relaxed);
            shared
                .stats
                .queue_bytes
                .fetch_sub(job.bytes as u64, Ordering::Relaxed);
            let _ = job.ack.send(result.clone());
        }
    }
}

/// One cached log entry (mirrors the single-group store exactly).
#[derive(Debug, Clone, PartialEq, Eq)]
struct GroupStoredEntry {
    term: u64,
    leader: u64,
    payload: GroupStoredPayload,
}

/// Cached entry payload.
#[derive(Debug, Clone, PartialEq, Eq)]
enum GroupStoredPayload {
    Blank,
    Normal(Vec<u8>),
    Membership(GroupStoredMembership),
}

/// Cached membership.
#[derive(Debug, Clone, PartialEq, Eq)]
struct GroupStoredMembership {
    voters: Vec<u64>,
    nodes: Vec<(u64, String)>,
}

/// Cached log id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GroupStoredLogId {
    term: u64,
    leader: u64,
    index: u64,
}

impl GroupStoredLogId {
    fn openraft(self) -> LogIdOf<KiviTypeConfig> {
        use openraft::impls::leader_id_adv::LeaderId;
        openraft::LogId::new(LeaderId::new(self.term, self.leader), self.index)
    }
}

/// Cached vote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GroupStoredVote {
    term: u64,
    candidate: u64,
    committed: bool,
}

impl GroupStoredVote {
    fn openraft(self) -> VoteOf<KiviTypeConfig> {
        if self.committed {
            openraft::impls::Vote::new_committed(self.term, self.candidate)
        } else {
            openraft::impls::Vote::new(self.term, self.candidate)
        }
    }
}

/// Logical per-group cache over the shared lane. Folds the group's own
/// recovered records at open; the physical lane is never touched here.
struct GroupInner {
    namespace: NamespaceId,
    group: TabletId,
    vote: Option<GroupStoredVote>,
    entries: BTreeMap<u64, GroupStoredEntry>,
    last_purged: Option<GroupStoredLogId>,
    committed: Option<GroupStoredLogId>,
    snapshot_base: Option<GroupStoredLogId>,
    truncations: u64,
}

impl GroupInner {
    /// Folds one recovered record (recovery-time rebuild; mirrors the live
    /// mutation paths).
    fn fold(&mut self, record: &RaftRecord) {
        match record {
            RaftRecord::Vote(vote) => {
                self.vote = Some(GroupStoredVote {
                    term: vote.term,
                    candidate: vote.candidate.as_u64(),
                    committed: vote.committed,
                });
            }
            RaftRecord::Entry(entry) => {
                self.entries.insert(
                    entry.index,
                    GroupStoredEntry {
                        term: entry.term,
                        leader: entry.leader.as_u64(),
                        payload: match &entry.payload {
                            RaftEntryPayload::Blank => GroupStoredPayload::Blank,
                            RaftEntryPayload::Normal(command) => {
                                GroupStoredPayload::Normal(command.clone())
                            }
                            RaftEntryPayload::Membership(membership) => {
                                GroupStoredPayload::Membership(GroupStoredMembership {
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
                let purged = GroupStoredLogId {
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
                self.committed = Some(GroupStoredLogId {
                    term: pointer.term,
                    leader: pointer.leader.as_u64(),
                    index: pointer.index,
                });
            }
            RaftRecord::SnapshotInstalled(snapshot) => {
                self.snapshot_base = Some(GroupStoredLogId {
                    term: snapshot.term,
                    leader: snapshot.leader.as_u64(),
                    index: snapshot.index,
                });
            }
        }
    }

    fn last_log_id(&self) -> Option<GroupStoredLogId> {
        self.entries
            .last_key_value()
            .map(|(index, entry)| GroupStoredLogId {
                term: entry.term,
                leader: entry.leader,
                index: *index,
            })
            .or(self.last_purged)
    }
}

fn group_entry_to_record(
    namespace: NamespaceId,
    group: TabletId,
    entry: &EntryOf<KiviTypeConfig>,
) -> RaftRecord {
    use kivi_durability::{RaftEntry, RaftMembership};
    use openraft::EntryPayload;
    let payload = match &entry.payload {
        EntryPayload::Blank => RaftEntryPayload::Blank,
        EntryPayload::Normal(command) => RaftEntryPayload::Normal(command.encode_to_vec()),
        EntryPayload::Membership(membership) => {
            let voters: Vec<u64> = membership.voter_ids().collect();
            let nodes: Vec<(u64, String)> = membership
                .nodes()
                .map(|(id, node)| (*id, node.addr.clone()))
                .collect();
            RaftEntryPayload::Membership(RaftMembership { voters, nodes })
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

fn group_stored_to_entry(
    index: u64,
    stored: &GroupStoredEntry,
) -> Result<EntryOf<KiviTypeConfig>, io::Error> {
    use openraft::EntryPayload;
    use openraft::impls::leader_id_adv::LeaderId;
    let log_id = openraft::LogId::new(LeaderId::new(stored.term, stored.leader), index);
    let payload = match &stored.payload {
        GroupStoredPayload::Blank => EntryPayload::Blank,
        GroupStoredPayload::Normal(command) => {
            let mantra = ConsensusCommand::decode_exact(command).map_err(|error| {
                io::Error::other(format!("stored command entry fails to decode: {error}"))
            })?;
            EntryPayload::Normal(mantra)
        }
        GroupStoredPayload::Membership(membership) => {
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

fn group_vote_to_record(
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

fn group_entry_leader_of(log_id: &LogIdOf<KiviTypeConfig>) -> u64 {
    log_id.leader_id.node_id
}

/// Lightweight logical group view over the shared physical durability
/// service: one object per `OpenRaft` group implementing its storage
/// traits, with no OS writer thread inside. Cloneable: clones share the
/// same logical cache, shared writer, and sidecar gate.
#[derive(Clone)]
pub struct GroupRaftStore {
    shared_cache: Arc<futures::lock::Mutex<GroupInner>>,
    durability: SharedRaftDurability,
    gate: Option<crate::gate::SidecarGate>,
}

impl std::fmt::Debug for GroupRaftStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.shared_cache.try_lock() {
            Some(inner) => f
                .debug_struct("GroupRaftStore")
                .field("group", &inner.group)
                .field("entries", &inner.entries.len())
                .field("vote", &inner.vote)
                .finish_non_exhaustive(),
            None => f
                .debug_struct("GroupRaftStore")
                .field("state", &"locked")
                .finish_non_exhaustive(),
        }
    }
}

/// Log reader sharing the group's logical cache.
#[derive(Clone)]
pub struct GroupLogReader {
    shared_cache: Arc<futures::lock::Mutex<GroupInner>>,
}

impl std::fmt::Debug for GroupLogReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.shared_cache.try_lock() {
            Some(inner) => f
                .debug_struct("GroupLogReader")
                .field("group", &inner.group)
                .field("entries", &inner.entries.len())
                .finish_non_exhaustive(),
            None => f
                .debug_struct("GroupLogReader")
                .field("state", &"locked")
                .finish_non_exhaustive(),
        }
    }
}

impl GroupRaftStore {
    /// Builds one group's logical view from its recovered records. The
    /// caller dispatches the shared lane's recovery output by group and
    /// folds each group in one pass — the physical log is never replayed
    /// once per group.
    ///
    /// # Errors
    ///
    /// Returns [`SharedOpenError::NamespaceMismatch`] when a record names
    /// a foreign namespace.
    pub fn from_recovered(
        namespace: NamespaceId,
        group: ConsensusGroupId,
        durability: &SharedRaftDurability,
        records: &[RaftRecord],
    ) -> Result<Self, SharedOpenError> {
        let tablet = group.tablet();
        let mut inner = GroupInner {
            namespace,
            group: tablet,
            vote: None,
            entries: BTreeMap::new(),
            last_purged: None,
            committed: None,
            snapshot_base: None,
            truncations: 0,
        };
        for record in records {
            if record.group() != tablet {
                continue;
            }
            if record.namespace() != namespace {
                return Err(SharedOpenError::NamespaceMismatch {
                    group: tablet.as_u64(),
                    expected: namespace.as_u64(),
                    found: record.namespace().as_u64(),
                });
            }
            inner.fold(record);
        }
        Ok(Self {
            shared_cache: Arc::new(futures::lock::Mutex::new(inner)),
            durability: durability.clone(),
            gate: None,
        })
    }

    /// Installs the sidecar durability gate (wired after the peer mesh and
    /// sidecar store exist).
    pub fn set_gate(&mut self, gate: crate::gate::SidecarGate) {
        self.gate = Some(gate);
    }

    /// Returns the installed gate, if any.
    #[must_use]
    pub fn gate(&self) -> Option<&crate::gate::SidecarGate> {
        self.gate.as_ref()
    }

    /// Returns the newest committed-membership voter set visible in the
    /// retained log (newest-first scan). Empty when none survives — the
    /// caller falls back to the state machine's snapshot membership.
    pub async fn membership_voters(&self) -> BTreeSet<u64> {
        let inner = self.shared_cache.lock().await;
        for (_, stored) in inner.entries.iter().rev() {
            if let GroupStoredPayload::Membership(membership) = &stored.payload {
                return membership.voters.iter().copied().collect();
            }
        }
        BTreeSet::new()
    }

    /// Returns the group served.
    pub async fn group(&self) -> ConsensusGroupId {
        ConsensusGroupId::of_tablet(self.shared_cache.lock().await.group)
    }

    /// Persists raw records through the shared barrier without folding
    /// (record-level seeding; the caller folds explicitly). Test-only.
    #[cfg(test)]
    pub(crate) async fn persist_raw(
        &self,
        records: Vec<RaftRecord>,
    ) -> Result<(), DurabilityError> {
        let bytes = estimate_bytes(&records);
        let rx = self.durability.submit(records).await;
        let result = rx.await.map_err(|_| DurabilityError::InvalidConfig {
            reason: "shared consensus lane thread exited",
        })?;
        self.durability.release(bytes);
        result
    }

    /// Folds one raw record into the logical cache without persisting
    /// (mirrors the live paths; completes record-level seeding).
    /// Test-only.
    #[cfg(test)]
    pub(crate) async fn fold_raw(&self, record: &RaftRecord) {
        self.shared_cache.lock().await.fold(record);
    }

    /// Returns the cached entry indexes in order. Test-only.
    #[cfg(test)]
    pub(crate) async fn cached_indexes(&self) -> Vec<u64> {
        self.shared_cache
            .lock()
            .await
            .entries
            .keys()
            .copied()
            .collect()
    }

    /// Returns the cached vote. Test-only.
    #[cfg(test)]
    pub(crate) async fn cached_vote(&self) -> Option<(u64, u64, bool)> {
        self.shared_cache
            .lock()
            .await
            .vote
            .map(|vote| (vote.term, vote.candidate, vote.committed))
    }

    /// Returns the cached commit pointer. Test-only.
    #[cfg(test)]
    pub(crate) async fn cached_committed(&self) -> Option<(u64, u64, u64)> {
        self.shared_cache
            .lock()
            .await
            .committed
            .map(|pointer| (pointer.term, pointer.leader, pointer.index))
    }

    /// Returns the cached snapshot base. Test-only.
    #[cfg(test)]
    pub(crate) async fn cached_snapshot_base(&self) -> Option<(u64, u64, u64)> {
        self.shared_cache
            .lock()
            .await
            .snapshot_base
            .map(|base| (base.term, base.leader, base.index))
    }

    /// Returns observed logical truncations. Test-only.
    #[cfg(test)]
    pub(crate) async fn cached_truncate_count(&self) -> u64 {
        self.shared_cache.lock().await.truncations
    }
}

impl crate::store::ConsensusLogStore for GroupRaftStore {
    fn set_gate(&mut self, gate: crate::gate::SidecarGate) {
        GroupRaftStore::set_gate(self, gate);
    }

    fn gate(&self) -> Option<&crate::gate::SidecarGate> {
        GroupRaftStore::gate(self)
    }

    async fn membership_voters(&self) -> BTreeSet<u64> {
        GroupRaftStore::membership_voters(self).await
    }
}

fn group_read_entries(
    inner: &GroupInner,
    range: (std::ops::Bound<u64>, std::ops::Bound<u64>),
) -> Result<Vec<EntryOf<KiviTypeConfig>>, io::Error> {
    inner
        .entries
        .range((range.0, range.1))
        .map(|(index, stored)| group_stored_to_entry(*index, stored))
        .collect()
}

impl RaftLogReader<KiviTypeConfig> for GroupLogReader {
    async fn try_get_log_entries<RB>(
        &mut self,
        range: RB,
    ) -> Result<Vec<EntryOf<KiviTypeConfig>>, io::Error>
    where
        RB: std::ops::RangeBounds<u64> + Clone + std::fmt::Debug + openraft::OptionalSend,
    {
        let inner = self.shared_cache.lock().await;
        group_read_entries(
            &inner,
            (range.start_bound().cloned(), range.end_bound().cloned()),
        )
    }

    async fn read_vote(&mut self) -> Result<Option<VoteOf<KiviTypeConfig>>, io::Error> {
        Ok(self
            .shared_cache
            .lock()
            .await
            .vote
            .map(GroupStoredVote::openraft))
    }
}

impl RaftLogReader<KiviTypeConfig> for GroupRaftStore {
    async fn try_get_log_entries<RB>(
        &mut self,
        range: RB,
    ) -> Result<Vec<EntryOf<KiviTypeConfig>>, io::Error>
    where
        RB: std::ops::RangeBounds<u64> + Clone + std::fmt::Debug + openraft::OptionalSend,
    {
        GroupLogReader {
            shared_cache: Arc::clone(&self.shared_cache),
        }
        .try_get_log_entries(range)
        .await
    }

    async fn read_vote(&mut self) -> Result<Option<VoteOf<KiviTypeConfig>>, io::Error> {
        Ok(self
            .shared_cache
            .lock()
            .await
            .vote
            .map(GroupStoredVote::openraft))
    }
}

impl RaftLogStorage<KiviTypeConfig> for GroupRaftStore {
    type LogReader = GroupLogReader;

    async fn get_log_state(&mut self) -> Result<LogState<KiviTypeConfig>, io::Error> {
        let inner = self.shared_cache.lock().await;
        Ok(LogState {
            last_purged_log_id: inner.last_purged.map(GroupStoredLogId::openraft),
            last_log_id: inner.last_log_id().map(GroupStoredLogId::openraft),
        })
    }

    #[allow(clippy::unused_async_trait_impl)]
    async fn get_log_reader(&mut self) -> Self::LogReader {
        GroupLogReader {
            shared_cache: Arc::clone(&self.shared_cache),
        }
    }

    async fn save_vote(&mut self, vote: &VoteOf<KiviTypeConfig>) -> Result<(), io::Error> {
        // Fencing first: the barrier proves durability before the cache
        // installs, so a crash between the two replays the record and
        // converges identically.
        let stored = GroupStoredVote {
            term: vote.leader_id.term,
            candidate: vote.leader_id.node_id,
            committed: vote.committed,
        };
        let (rx, bytes) = {
            let inner = self.shared_cache.lock().await;
            let record = group_vote_to_record(inner.namespace, inner.group, vote);
            let bytes = estimate_bytes(std::slice::from_ref(&record));
            let rx = self.durability.submit(vec![record]).await;
            (rx, bytes)
        };
        let result = await_shared_barrier(rx).await;
        self.durability.release(bytes);
        result?;
        self.shared_cache.lock().await.vote = Some(stored);
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
        // Sidecar durability gate (authoritative, unchanged): before
        // staging or persisting, every chunked root must be locally
        // durable and verified. No lock is held across the fetch.
        let entries: Vec<EntryOf<KiviTypeConfig>> = entries.into_iter().collect();
        if let Some(gate) = self.gate.clone() {
            let mut commands = Vec::new();
            let mut hint = None;
            for entry in &entries {
                if hint.is_none() {
                    hint = Some(NodeId::from_u64(entry.log_id.leader_id.node_id));
                }
                if let openraft::EntryPayload::Normal(command) = &entry.payload {
                    commands.push(command.encode_to_vec());
                }
            }
            if !commands.is_empty() {
                gate.ensure_entries(&commands, hint)
                    .await
                    .map_err(|error| crate::gate::sidecar_io_error(&error))?;
            }
        }
        // Stage records and cache together under one lock (readable on
        // return) and submit the barrier in FIFO order while still
        // holding it. The lock releases before the ack wait.
        let (staged, rx, bytes) = {
            let mut inner = self.shared_cache.lock().await;
            let mut records = Vec::new();
            let mut staged = Vec::new();
            for entry in entries {
                let index = entry.log_id.index;
                let record = group_entry_to_record(inner.namespace, inner.group, &entry);
                let stored = GroupStoredEntry {
                    term: entry.log_id.leader_id.term,
                    leader: entry.log_id.leader_id.node_id,
                    payload: match &entry.payload {
                        openraft::EntryPayload::Blank => GroupStoredPayload::Blank,
                        openraft::EntryPayload::Normal(command) => {
                            GroupStoredPayload::Normal(command.encode_to_vec())
                        }
                        openraft::EntryPayload::Membership(membership) => {
                            GroupStoredPayload::Membership(GroupStoredMembership {
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
            let bytes = estimate_bytes(&records);
            let rx = self.durability.submit(records).await;
            (staged, rx, bytes)
        };
        let result = await_shared_barrier(rx).await;
        self.durability.release(bytes);
        match result {
            Ok(()) => {
                callback.io_completed(Ok(()));
                Ok(())
            }
            Err(error) => {
                let mut inner = self.shared_cache.lock().await;
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
        let from_index = last_kept
            .as_ref()
            .map_or(0, |id| id.index.saturating_add(1));
        let (rx, bytes) = {
            let inner = self.shared_cache.lock().await;
            let record = RaftRecord::Truncate(kivi_durability::RaftTruncate {
                namespace: inner.namespace,
                group: inner.group,
                from_index,
            });
            let bytes = estimate_bytes(std::slice::from_ref(&record));
            let rx = self.durability.submit(vec![record]).await;
            (rx, bytes)
        };
        let result = await_shared_barrier(rx).await;
        self.durability.release(bytes);
        result?;
        let mut inner = self.shared_cache.lock().await;
        inner.entries.retain(|index, _| *index < from_index);
        inner.truncations += 1;
        Ok(())
    }

    async fn purge(&mut self, log_id: LogIdOf<KiviTypeConfig>) -> Result<(), io::Error> {
        let purged = GroupStoredLogId {
            term: log_id.leader_id.term,
            leader: group_entry_leader_of(&log_id),
            index: log_id.index,
        };
        let (rx, bytes) = {
            let inner = self.shared_cache.lock().await;
            let record = RaftRecord::Purge(kivi_durability::RaftPurge {
                namespace: inner.namespace,
                group: inner.group,
                term: log_id.leader_id.term,
                leader: NodeId::from_u64(group_entry_leader_of(&log_id)),
                through_index: log_id.index,
            });
            let bytes = estimate_bytes(std::slice::from_ref(&record));
            let rx = self.durability.submit(vec![record]).await;
            (rx, bytes)
        };
        let result = await_shared_barrier(rx).await;
        self.durability.release(bytes);
        result?;
        let mut inner = self.shared_cache.lock().await;
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
            return Ok(());
        };
        let stored = GroupStoredLogId {
            term: log_id.leader_id.term,
            leader: group_entry_leader_of(&log_id),
            index: log_id.index,
        };
        let (rx, bytes) = {
            let inner = self.shared_cache.lock().await;
            let record = RaftRecord::Committed(kivi_durability::RaftCommitted {
                namespace: inner.namespace,
                group: inner.group,
                term: log_id.leader_id.term,
                leader: NodeId::from_u64(group_entry_leader_of(&log_id)),
                index: log_id.index,
            });
            let bytes = estimate_bytes(std::slice::from_ref(&record));
            let rx = self.durability.submit(vec![record]).await;
            (rx, bytes)
        };
        let result = await_shared_barrier(rx).await;
        self.durability.release(bytes);
        result?;
        self.shared_cache.lock().await.committed = Some(stored);
        Ok(())
    }

    async fn read_committed(&mut self) -> Result<Option<LogIdOf<KiviTypeConfig>>, io::Error> {
        Ok(self
            .shared_cache
            .lock()
            .await
            .committed
            .map(GroupStoredLogId::openraft))
    }
}

/// Dispatches shared-lane recovery output by group: one pass over the
/// physical log, grouped by [`ConsensusGroupId`]. Groups with no records
/// map to empty vecs (fresh groups fold nothing).
#[must_use]
pub fn dispatch_by_group(
    records: &[RaftRecord],
    groups: &[ConsensusGroupId],
) -> HashMap<ConsensusGroupId, Vec<RaftRecord>> {
    let mut out: HashMap<ConsensusGroupId, Vec<RaftRecord>> = HashMap::new();
    for group in groups {
        out.insert(*group, Vec::new());
    }
    for record in records {
        let group = ConsensusGroupId::of_tablet(record.group());
        if let Some(bucket) = out.get_mut(&group) {
            bucket.push(record.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use kivi_types::{ClusterId, NodeIncarnation};

    use super::*;

    fn identity() -> LaneIdentity {
        LaneIdentity {
            cluster: ClusterId::from_u128(0xC10C),
            node: NodeId::from_u64(1),
            incarnation: NodeIncarnation::INITIAL,
        }
    }

    fn block_on<F, T>(future: F) -> T
    where
        F: Future<Output = T>,
    {
        compio::runtime::Runtime::new()
            .expect("test reactor starts")
            .block_on(future)
    }

    #[test]
    fn cross_group_batches_share_one_writer() {
        use openraft::storage::RaftLogStorage as _;
        let dir = tempfile::tempdir().expect("tempdir");
        let group_a = ConsensusGroupId::of_tablet(TabletId::from_u64(101));
        let group_b = ConsensusGroupId::of_tablet(TabletId::from_u64(102));
        let ns = NamespaceId::from_u64(1);
        let (durability, recovered) = SharedRaftDurability::open(
            dir.path(),
            identity(),
            1024 * 1024,
            SharedDurabilityConfig::default(),
        )
        .expect("shared opens");
        assert!(recovered.is_empty());
        let by_group = dispatch_by_group(&recovered, &[group_a, group_b]);
        let mut store_a =
            GroupRaftStore::from_recovered(ns, group_a, &durability, &by_group[&group_a])
                .expect("group A view");
        let mut store_b =
            GroupRaftStore::from_recovered(ns, group_b, &durability, &by_group[&group_b])
                .expect("group B view");
        block_on(async {
            let vote_a = openraft::impls::Vote::new_committed(3, 1);
            let vote_b = openraft::impls::Vote::new_committed(5, 2);
            // Concurrent barriers from two groups share the writer.
            let (left, right) =
                futures::join!(store_a.save_vote(&vote_a), store_b.save_vote(&vote_b));
            left.expect("A vote durable");
            right.expect("B vote durable");
            assert_eq!(
                store_a.read_vote().await.expect("read A"),
                Some(vote_a),
                "A vote exact"
            );
            assert_eq!(
                store_b.read_vote().await.expect("read B"),
                Some(vote_b),
                "B vote exact, never A's"
            );
        });
        let metrics = durability.metrics();
        assert!(metrics.physical_batches >= 1, "at least one batch");
        assert_eq!(metrics.records, 2, "two logical records");
    }

    #[test]
    fn cross_group_purge_and_commit_markers_stay_isolated() {
        use openraft::storage::RaftLogStorage as _;
        let dir = tempfile::tempdir().expect("tempdir");
        let group_a = ConsensusGroupId::of_tablet(TabletId::from_u64(301));
        let group_b = ConsensusGroupId::of_tablet(TabletId::from_u64(302));
        let ns = NamespaceId::from_u64(1);
        let (durability, recovered) = SharedRaftDurability::open(
            dir.path(),
            identity(),
            1024 * 1024,
            SharedDurabilityConfig::default(),
        )
        .expect("shared opens");
        let by_group = dispatch_by_group(&recovered, &[group_a, group_b]);
        let mut store_a =
            GroupRaftStore::from_recovered(ns, group_a, &durability, &by_group[&group_a])
                .expect("group A view");
        let mut store_b =
            GroupRaftStore::from_recovered(ns, group_b, &durability, &by_group[&group_b])
                .expect("group B view");
        block_on(async {
            use openraft::impls::leader_id_adv::LeaderId;
            let vote = openraft::impls::Vote::new_committed(1, 1);
            store_a.save_vote(&vote).await.expect("A vote");
            store_b.save_vote(&vote).await.expect("B vote");
            // A purges through index 5 and records its commit pointer;
            // B's markers must stay absent (logical isolation over the
            // shared physical history — purge is never segment deletion).
            let purged = openraft::LogId::new(LeaderId::new(1, 1), 5);
            store_a.purge(purged).await.expect("A purge");
            store_a
                .save_committed(Some(purged))
                .await
                .expect("A commit");
            let state_b = store_b.get_log_state().await.expect("B state");
            assert_eq!(state_b.last_purged_log_id, None, "B never purged");
            assert_eq!(
                store_b.read_committed().await.expect("B commit"),
                None,
                "B never committed"
            );
            let state_a = store_a.get_log_state().await.expect("A state");
            assert_eq!(state_a.last_purged_log_id, Some(purged), "A purge exact");
            assert_eq!(
                store_a.read_committed().await.expect("A commit"),
                Some(purged),
                "A commit exact"
            );
        });
        // Reopen: one recovery pass rebuilds both groups' markers exactly.
        drop(store_a);
        drop(store_b);
        drop(durability);
        let (durability, recovered) = SharedRaftDurability::open(
            dir.path(),
            identity(),
            1024 * 1024,
            SharedDurabilityConfig::default(),
        )
        .expect("shared reopens");
        let by_group = dispatch_by_group(&recovered, &[group_a, group_b]);
        let mut reopened_a =
            GroupRaftStore::from_recovered(ns, group_a, &durability, &by_group[&group_a])
                .expect("A view reopens");
        let mut reopened_b =
            GroupRaftStore::from_recovered(ns, group_b, &durability, &by_group[&group_b])
                .expect("B view reopens");
        block_on(async {
            use openraft::storage::RaftLogStorage as _;
            assert!(
                reopened_a
                    .get_log_state()
                    .await
                    .expect("A state")
                    .last_purged_log_id
                    .is_some(),
                "A purge survives restart"
            );
            assert!(
                reopened_b
                    .get_log_state()
                    .await
                    .expect("B state")
                    .last_purged_log_id
                    .is_none(),
                "B still never purged after restart"
            );
        });
    }

    #[test]
    fn shared_wal_corruption_fails_recovery_loudly() {
        let dir = tempfile::tempdir().expect("tempdir");
        let group_a = ConsensusGroupId::of_tablet(TabletId::from_u64(401));
        let group_b = ConsensusGroupId::of_tablet(TabletId::from_u64(402));
        let ns = NamespaceId::from_u64(1);
        let (durability, recovered) = SharedRaftDurability::open(
            dir.path(),
            identity(),
            1024 * 1024,
            SharedDurabilityConfig::default(),
        )
        .expect("shared opens");
        let by_group = dispatch_by_group(&recovered, &[group_a, group_b]);
        let mut store_a =
            GroupRaftStore::from_recovered(ns, group_a, &durability, &by_group[&group_a])
                .expect("group A view");
        let mut store_b =
            GroupRaftStore::from_recovered(ns, group_b, &durability, &by_group[&group_b])
                .expect("group B view");
        block_on(async {
            use openraft::storage::RaftLogStorage as _;
            // Two separate physical batches (linger gap between them).
            store_a
                .save_vote(&openraft::impls::Vote::new_committed(1, 1))
                .await
                .expect("A vote durable");
        });
        std::thread::sleep(std::time::Duration::from_millis(5));
        block_on(async {
            use openraft::storage::RaftLogStorage as _;
            store_b
                .save_vote(&openraft::impls::Vote::new_committed(2, 2))
                .await
                .expect("B vote durable");
        });
        drop(store_a);
        drop(store_b);
        drop(durability);
        // Corrupt one byte inside the first batch body (complete-but-wrong
        // mid-file content: never a torn tail, always loud corruption).
        let lane_dir = dir
            .path()
            .join(kivi_durability::node::WAL_DIR_NAME)
            .join(kivi_durability::wal::lane_dir_name(CONSENSUS_LANE));
        let segment = std::fs::read_dir(&lane_dir)
            .expect("lane dir")
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .find(|path| path.extension().is_some_and(|ext| ext == "wal"))
            .expect("a segment file");
        let mut bytes = std::fs::read(&segment).expect("segment bytes");
        // Segment header (56) + batch header (40): land inside the first
        // batch body, past framing, inside record bytes.
        let at = 56 + 40 + 10;
        assert!(bytes.len() > at + 1, "segment holds two batches");
        bytes[at] ^= 0xFF;
        std::fs::write(&segment, &bytes).expect("corrupt segment");
        let reopened = SharedRaftDurability::open(
            dir.path(),
            identity(),
            1024 * 1024,
            SharedDurabilityConfig::default(),
        );
        assert!(
            reopened.is_err(),
            "corrupted shared WAL fails loudly instead of salvaging"
        );
    }

    #[test]
    fn one_group_truncate_never_touches_another() {
        use openraft::storage::RaftLogStorage as _;
        let dir = tempfile::tempdir().expect("tempdir");
        let group_a = ConsensusGroupId::of_tablet(TabletId::from_u64(201));
        let group_b = ConsensusGroupId::of_tablet(TabletId::from_u64(202));
        let ns = NamespaceId::from_u64(1);
        let (durability, recovered) = SharedRaftDurability::open(
            dir.path(),
            identity(),
            1024 * 1024,
            SharedDurabilityConfig::default(),
        )
        .expect("shared opens");
        let by_group = dispatch_by_group(&recovered, &[group_a, group_b]);
        let mut store_a =
            GroupRaftStore::from_recovered(ns, group_a, &durability, &by_group[&group_a])
                .expect("group A view");
        let mut store_b =
            GroupRaftStore::from_recovered(ns, group_b, &durability, &by_group[&group_b])
                .expect("group B view");
        block_on(async {
            let vote = openraft::impls::Vote::new_committed(1, 1);
            store_a.save_vote(&vote).await.expect("A vote");
            store_b.save_vote(&vote).await.expect("B vote");
            // Truncate A fully; B's vote must survive (logical isolation
            // over shared physical history).
            store_a.truncate_after(None).await.expect("A truncate");
            assert_eq!(
                store_b.read_vote().await.expect("read B"),
                Some(vote),
                "B unaffected by A's truncate"
            );
        });
        // Reopen: one recovery pass rebuilds both groups exactly.
        drop(store_a);
        drop(store_b);
        drop(durability);
        let (durability, recovered) = SharedRaftDurability::open(
            dir.path(),
            identity(),
            1024 * 1024,
            SharedDurabilityConfig::default(),
        )
        .expect("shared reopens");
        let by_group = dispatch_by_group(&recovered, &[group_a, group_b]);
        // A's truncate marker replayed; B's vote replayed.
        assert!(
            by_group[&group_a].iter().any(|record| matches!(
                record,
                RaftRecord::Truncate(marker) if marker.from_index == 0
            )),
            "A truncate marker recovered"
        );
        assert!(
            by_group[&group_b]
                .iter()
                .any(|record| matches!(record, RaftRecord::Vote(_))),
            "B vote recovered"
        );
        let mut reopened_b =
            GroupRaftStore::from_recovered(ns, group_b, &durability, &by_group[&group_b])
                .expect("B view reopens");
        block_on(async {
            assert_eq!(
                reopened_b.read_vote().await.expect("read B"),
                Some(openraft::impls::Vote::new_committed(1, 1)),
                "B vote survives restart"
            );
        });
    }
}
