//! Durable `OpenRaft` 0.10 log storage on Kivi WAL primitives.
//!
//! One [`DurableRaftStore`] backs one consensus group's log side
//! ([`openraft::storage::RaftLogStorage`] +
//! [`openraft::storage::RaftLogReader`]). It is a thin façade over one
//! [`GroupRaftStore`](crate::shared::GroupRaftStore): Kivi-owned
//! [`kivi_durability::RaftRecord`]s persist through the node-wide shared
//! WAL writer — the same segmented, checksummed, crash-safe
//! infrastructure as tablet history, with one physical WAL holding every
//! group and no format change.
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
//!   the shared writer persists; the [`openraft::storage::IOFlushed`]
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
//! ## Shared writer thread
//!
//! The WAL barrier is synchronous `std::fs` plus `fsync`. On a
//! single-threaded Compio reactor blocking is forbidden, so barriers
//! cross as bounded channel jobs with oneshot acks to the node-wide
//! shared writer thread — the reactor yields while `fsync` runs
//! elsewhere. There is no thread per group: one writer serves every
//! group on the node (RFC §60), batching concurrent groups' records into
//! single physical batches.
//!
//! All write paths serialize through one [`futures::lock::Mutex`]: it is
//! held across stage-plus-submit only, never across the barrier ack, so no
//! reactor task stalls on `fsync` latency. FIFO submission preserves
//! the per-group log order the consecutive-log invariant requires.
//!
//! ## State machine side
//!
//! This module covers the LOG side only. Applied-state tracking and the
//! production apply path (`MutationIR` into `LiveTablet`) live in
//! [`crate::state_machine`].

use std::collections::BTreeSet;
use std::io;
use std::path::{Path, PathBuf};

use crate::config::KiviTypeConfig;
use crate::types::ConsensusGroupId;
use kivi_durability::LaneIdentity;
use kivi_types::NamespaceId;
use openraft::storage::{IOFlushed, LogState, RaftLogReader, RaftLogStorage};
use openraft::type_config::alias::{EntryOf, LogIdOf, VoteOf};

/// WAL lane reserved for consensus history (`wal/lane-65535/`). Worker
/// lanes number `0..worker_count`; the top lane id can never collide with
/// one, so consensus history shares the directory (and the reclamation
/// domain) without sharing any worker's file sequence.
pub const CONSENSUS_LANE: u16 = u16::MAX;

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

/// Logical per-group durability is implemented once in
/// [`GroupRaftStore`](crate::shared::GroupRaftStore) (logical cache plus
/// the node-wide shared writer); this module keeps the single-group
/// façade ([`DurableRaftStore`]) over it, so one code path serves every
/// group count and no OS thread is ever owned per group.
///
/// Lightweight logical group view over physical consensus durability:
/// one object per `OpenRaft` group implementing its storage traits, with
/// no hidden writer thread inside the view itself (physical writing lives
/// in the node-wide shared writer).
///
/// Both the single-group [`DurableRaftStore`] façade and the multi-group
/// [`GroupRaftStore`](crate::shared::GroupRaftStore) implement this, so
/// the owner machinery (proposals, barriers, peer RPCs, snapshots,
/// preflight) is written once generically and shared by both modes.
pub trait ConsensusLogStore:
    RaftLogStorage<KiviTypeConfig> + RaftLogReader<KiviTypeConfig> + Clone + Send + 'static
{
    /// Installs the sidecar durability gate (wired after the peer mesh and
    /// sidecar store exist).
    fn set_gate(&mut self, gate: crate::gate::SidecarGate);
    /// Returns the installed gate, if any.
    fn gate(&self) -> Option<&crate::gate::SidecarGate>;
    /// Returns the newest committed-membership voter set visible in the
    /// retained log (newest-first scan). Empty when none survives — the
    /// caller falls back to the state machine's snapshot membership.
    fn membership_voters(&self) -> impl Future<Output = BTreeSet<u64>> + Send;
}

/// Durable single-group Raft log store: a thin façade over one
/// [`GroupRaftStore`](crate::shared::GroupRaftStore) on a node-wide
/// shared writer. Cloneable: clones share the same logical cache, writer,
/// and sidecar gate.
///
/// There is deliberately no per-group lane thread here (nor anywhere
/// else): one OS thread per Raft group does not scale to many tablets, so
/// every group — one or one thousand — shares the physical writer (see
/// [`crate::shared`]).
///
/// The gate enforces the core invariant: a replica never reports a Raft
/// append durable while required immutable sidecars are missing. See
/// [`crate::gate`] and [`crate::sidecar`].
#[derive(Clone, Debug)]
pub struct DurableRaftStore {
    view: crate::shared::GroupRaftStore,
}

/// Log reader sharing the store's logical cache (replication streams read
/// through this; the log side owns the writes).
#[derive(Clone, Debug)]
pub struct DurableLogReader {
    view: crate::shared::GroupLogReader,
}

impl DurableRaftStore {
    /// Opens (or reopens) the group's durable log in the node's data
    /// directory through the node-wide shared writer: the shared lane is
    /// opened or resumed, its history recovered once, and this group's
    /// logical state rebuilt by folding its own records in file order.
    /// This call is synchronous and touches no async runtime.
    ///
    /// Only this group's consensus records fold; other groups' records on
    /// the shared lane are skipped by the fold. Forming a group on a
    /// non-fresh directory is a bootstrap concern, rejected there.
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
        // Zero linger: a lone group seals immediately (no batching delay
        // without concurrency); the shared writer still owns the single
        // physical thread, so no per-group thread exists either way.
        let config = crate::shared::SharedDurabilityConfig {
            linger: std::time::Duration::ZERO,
            ..crate::shared::SharedDurabilityConfig::default()
        };
        let (durability, recovered) = crate::shared::SharedRaftDurability::open(
            data_dir,
            identity,
            segment_target_bytes,
            config,
        )
        .map_err(|error| match error {
            crate::shared::SharedOpenError::Lane { reason } => StoreOpenError::Lane { reason },
            crate::shared::SharedOpenError::NamespaceMismatch { .. } => StoreOpenError::Lane {
                reason: "shared lane recovery failed".to_owned(),
            },
        })?;
        let view = crate::shared::GroupRaftStore::from_recovered(
            namespace,
            group,
            &durability,
            &recovered,
        )
        .map_err(|error| match error {
            crate::shared::SharedOpenError::NamespaceMismatch {
                group,
                expected,
                found,
            } => StoreOpenError::NamespaceMismatch {
                group,
                expected,
                found,
            },
            crate::shared::SharedOpenError::Lane { reason } => StoreOpenError::Lane { reason },
        })?;
        Ok(Self { view })
    }

    /// Installs the sidecar durability gate (node open wires this after
    /// the peer mesh and sidecar store exist; the log open itself stays
    /// synchronous and runtime-free).
    pub fn set_gate(&mut self, gate: crate::gate::SidecarGate) {
        self.view.set_gate(gate);
    }

    /// Returns the installed gate, if any.
    #[must_use]
    pub fn gate(&self) -> Option<&crate::gate::SidecarGate> {
        self.view.gate()
    }

    /// Returns the newest committed-membership voter set visible in the
    /// retained log (scanned newest-first). Empty when no membership
    /// entry survives (purged past, or never committed) — the caller
    /// falls back to the state machine's snapshot membership, which
    /// grounds every purge.
    pub async fn membership_voters(&self) -> std::collections::BTreeSet<u64> {
        self.view.membership_voters().await
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

impl ConsensusLogStore for DurableRaftStore {
    fn set_gate(&mut self, gate: crate::gate::SidecarGate) {
        DurableRaftStore::set_gate(self, gate);
    }

    fn gate(&self) -> Option<&crate::gate::SidecarGate> {
        DurableRaftStore::gate(self)
    }

    async fn membership_voters(&self) -> BTreeSet<u64> {
        DurableRaftStore::membership_voters(self).await
    }
}

/// Logical group behavior lives in [`GroupRaftStore`](crate::shared::GroupRaftStore);
/// the façade below delegates every storage-trait call to its view, so one
/// implementation serves single-group nodes and multi-tablet workers alike.
impl RaftLogReader<KiviTypeConfig> for DurableLogReader {
    async fn try_get_log_entries<RB>(
        &mut self,
        range: RB,
    ) -> Result<Vec<EntryOf<KiviTypeConfig>>, io::Error>
    where
        RB: std::ops::RangeBounds<u64> + Clone + std::fmt::Debug + openraft::OptionalSend,
    {
        self.view.try_get_log_entries(range).await
    }

    async fn read_vote(&mut self) -> Result<Option<VoteOf<KiviTypeConfig>>, io::Error> {
        self.view.read_vote().await
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
        self.get_log_reader().await.try_get_log_entries(range).await
    }

    async fn read_vote(&mut self) -> Result<Option<VoteOf<KiviTypeConfig>>, io::Error> {
        self.get_log_reader().await.read_vote().await
    }
}

impl RaftLogStorage<KiviTypeConfig> for DurableRaftStore {
    type LogReader = DurableLogReader;

    async fn get_log_state(&mut self) -> Result<LogState<KiviTypeConfig>, io::Error> {
        self.view.get_log_state().await
    }

    #[allow(clippy::unused_async_trait_impl)]
    async fn get_log_reader(&mut self) -> Self::LogReader {
        DurableLogReader {
            view: self.view.get_log_reader().await,
        }
    }

    async fn save_vote(&mut self, vote: &VoteOf<KiviTypeConfig>) -> Result<(), io::Error> {
        self.view.save_vote(vote).await
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
        self.view.append(entries, callback).await
    }

    async fn truncate_after(
        &mut self,
        last_kept: Option<LogIdOf<KiviTypeConfig>>,
    ) -> Result<(), io::Error> {
        self.view.truncate_after(last_kept).await
    }

    async fn purge(&mut self, log_id: LogIdOf<KiviTypeConfig>) -> Result<(), io::Error> {
        self.view.purge(log_id).await
    }

    async fn save_committed(
        &mut self,
        committed: Option<LogIdOf<KiviTypeConfig>>,
    ) -> Result<(), io::Error> {
        self.view.save_committed(committed).await
    }

    async fn read_committed(&mut self) -> Result<Option<LogIdOf<KiviTypeConfig>>, io::Error> {
        self.view.read_committed().await
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
        for record in &records {
            seed.fold_record(record).await;
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
/// the `OpenRaft` traits. Delegates to the view's test hooks: one
/// implementation serves both store shapes.
#[cfg(test)]
pub(crate) mod test_support {
    use super::DurableRaftStore;

    impl DurableRaftStore {
        /// Persists raw records through the shared barrier without folding
        /// (record-level seeding; the caller folds explicitly).
        pub(crate) async fn persist_records(
            &self,
            records: Vec<kivi_durability::RaftRecord>,
        ) -> Result<(), kivi_durability::DurabilityError> {
            self.view.persist_raw(records).await
        }

        /// Folds one raw record into the logical cache without persisting.
        pub(crate) async fn fold_record(&self, record: &kivi_durability::RaftRecord) {
            self.view.fold_raw(record).await;
        }

        /// Returns the cached entry indexes in order.
        pub(crate) async fn cached_indexes(&self) -> Vec<u64> {
            self.view.cached_indexes().await
        }

        /// Returns the cached vote.
        pub(crate) async fn cached_vote(&self) -> Option<(u64, u64, bool)> {
            self.view.cached_vote().await
        }

        /// Returns the cached commit pointer.
        pub(crate) async fn cached_committed(&self) -> Option<(u64, u64, u64)> {
            self.view.cached_committed().await
        }

        /// Returns the cached snapshot base.
        pub(crate) async fn cached_snapshot_base(&self) -> Option<(u64, u64, u64)> {
            self.view.cached_snapshot_base().await
        }

        /// Returns observed logical truncations (live calls plus
        /// recovery-fold replays of durable markers).
        pub(crate) async fn cached_truncate_count(&self) -> u64 {
            self.view.cached_truncate_count().await
        }
    }
}
