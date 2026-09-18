//! In-memory `OpenRaft` storage and unreachable network for experiments.
//!
//! [`MemLogStore`] and [`MemStateMachine`] implement the exact split traits
//! the production store implements
//! ([`openraft::storage::RaftLogStorage`] +
//! [`openraft::storage::RaftStateMachine`]), so the density experiment and
//! storage-adjacent unit tests exercise the real integration surface — but
//! they persist nothing and must never back a real group (see
//! [`crate::store`] for the WAL-backed store).
//!
//! [`StubNetworkFactory`] mints an unreachable stub: single-voter groups
//! never send RPCs; if one ever does, the error is loud, never silent.
//!
//! The state machine installs the proposer's deterministic outcome verbatim
//! (the command's `expected`). This is deliberately NOT execution — the
//! production state machine re-applies and compares against `expected`
//! instead of trusting it.

use std::collections::BTreeMap;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use openraft::storage::{
    IOFlushed, LogState, RaftLogReader, RaftLogStorage, RaftSnapshotBuilder, RaftStateMachine,
    Snapshot,
};
use openraft::type_config::alias::{
    EntryOf, LogIdOf, SnapshotMetaOf, SnapshotOf, StoredMembershipOf, VoteOf,
};

use crate::config::KiviTypeConfig;

/// Shared in-memory log core behind [`MemLogStore`] clones.
#[derive(Debug, Default)]
struct MemLogCore {
    vote: Option<VoteOf<KiviTypeConfig>>,
    entries: BTreeMap<u64, EntryOf<KiviTypeConfig>>,
    last_purged: Option<LogIdOf<KiviTypeConfig>>,
}

/// In-memory [`RaftLogStorage`] for experiments.
///
/// Correct in-memory semantics (consecutive log, serialized writes via the
/// mutex, logical truncate/purge markers) but zero durability: dropping
/// the store loses everything.
#[derive(Debug, Clone, Default)]
pub struct MemLogStore {
    core: Arc<Mutex<MemLogCore>>,
}

impl MemLogStore {
    /// Creates an empty log store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[allow(clippy::unused_async_trait_impl)]
impl RaftLogReader<KiviTypeConfig> for MemLogStore {
    #[allow(clippy::unused_async_trait_impl)]
    #[allow(clippy::unused_async_trait_impl)]
    async fn try_get_log_entries<RB>(
        &mut self,
        range: RB,
    ) -> Result<Vec<EntryOf<KiviTypeConfig>>, std::io::Error>
    where
        RB: std::ops::RangeBounds<u64> + Clone + std::fmt::Debug + openraft::OptionalSend,
    {
        let core = self.core.lock().expect("mem log lock holds");
        Ok(core
            .entries
            .range((range.start_bound().cloned(), range.end_bound().cloned()))
            .map(|(_, entry)| entry.clone())
            .collect())
    }

    #[allow(clippy::unused_async_trait_impl)]
    async fn read_vote(&mut self) -> Result<Option<VoteOf<KiviTypeConfig>>, std::io::Error> {
        Ok(self.core.lock().expect("mem log lock holds").vote)
    }
}

impl RaftLogStorage<KiviTypeConfig> for MemLogStore {
    type LogReader = Self;

    #[allow(clippy::unused_async_trait_impl)]
    async fn get_log_state(&mut self) -> Result<LogState<KiviTypeConfig>, std::io::Error> {
        let core = self.core.lock().expect("mem log lock holds");
        let last = core
            .entries
            .values()
            .next_back()
            .map(|entry| entry.log_id)
            .or(core.last_purged);
        Ok(LogState {
            last_purged_log_id: core.last_purged,
            last_log_id: last,
        })
    }

    #[allow(clippy::unused_async_trait_impl)]
    async fn get_log_reader(&mut self) -> Self::LogReader {
        self.clone()
    }

    #[allow(clippy::unused_async_trait_impl)]
    async fn save_vote(&mut self, vote: &VoteOf<KiviTypeConfig>) -> Result<(), std::io::Error> {
        self.core.lock().expect("mem log lock holds").vote = Some(*vote);
        Ok(())
    }

    #[allow(clippy::unused_async_trait_impl)]
    async fn append<I>(
        &mut self,
        entries: I,
        callback: IOFlushed<KiviTypeConfig>,
    ) -> Result<(), std::io::Error>
    where
        I: IntoIterator<Item = EntryOf<KiviTypeConfig>> + openraft::OptionalSend,
        I::IntoIter: openraft::OptionalSend,
    {
        {
            let mut core = self.core.lock().expect("mem log lock holds");
            for entry in entries {
                core.entries.insert(entry.log_id.index, entry);
            }
        }
        // In-memory persistence is immediate: report the flush inline.
        callback.io_completed(Ok(()));
        Ok(())
    }

    #[allow(clippy::unused_async_trait_impl)]
    async fn truncate_after(
        &mut self,
        last_kept: Option<LogIdOf<KiviTypeConfig>>,
    ) -> Result<(), std::io::Error> {
        let mut core = self.core.lock().expect("mem log lock holds");
        match last_kept {
            Some(id) => {
                core.entries.split_off(&(id.index + 1));
            }
            None => core.entries.clear(),
        }
        Ok(())
    }

    #[allow(clippy::unused_async_trait_impl)]
    async fn purge(&mut self, log_id: LogIdOf<KiviTypeConfig>) -> Result<(), std::io::Error> {
        let mut core = self.core.lock().expect("mem log lock holds");
        core.entries.retain(|index, _| *index > log_id.index);
        core.last_purged = Some(log_id);
        Ok(())
    }
}

/// Shared in-memory state-machine core behind [`MemStateMachine`] clones.
#[derive(Debug, Default)]
struct MemStateMachineCore {
    last_applied: Option<LogIdOf<KiviTypeConfig>>,
    membership: StoredMembershipOf<KiviTypeConfig>,
    applied_commands: u64,
}

/// In-memory [`RaftStateMachine`] for experiments. Installs the proposer's
/// deterministic outcome verbatim; tracks application order. Spike only.
#[derive(Debug, Clone, Default)]
pub struct MemStateMachine {
    core: Arc<Mutex<MemStateMachineCore>>,
}

impl MemStateMachine {
    /// Creates an empty state machine.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Counts applied command entries (experiment introspection only).
    ///
    /// # Panics
    ///
    /// Panics when the in-memory lock is poisoned (a previous holder
    /// panicked mid-apply — an experiment-harness bug, never a runtime
    /// condition).
    #[must_use]
    pub fn applied_commands(&self) -> u64 {
        self.core
            .lock()
            .expect("mem sm lock holds")
            .applied_commands
    }
}

/// Snapshot builder capturing the applied watermark at build time.
#[derive(Debug)]
pub struct MemSnapshotBuilder {
    meta: SnapshotMetaOf<KiviTypeConfig>,
}

impl RaftSnapshotBuilder<KiviTypeConfig> for MemSnapshotBuilder {
    type SnapshotData = Cursor<Vec<u8>>;

    #[allow(clippy::unused_async_trait_impl)]
    async fn build_snapshot(
        &mut self,
    ) -> Result<SnapshotOf<KiviTypeConfig, Cursor<Vec<u8>>>, std::io::Error> {
        Ok(Snapshot {
            meta: self.meta.clone(),
            snapshot: Cursor::new(Vec::new()),
        })
    }
}

impl RaftStateMachine<KiviTypeConfig> for MemStateMachine {
    type SnapshotBuilder = MemSnapshotBuilder;
    type SnapshotData = Cursor<Vec<u8>>;

    #[allow(clippy::unused_async_trait_impl)]
    async fn applied_state(
        &mut self,
    ) -> Result<
        (
            Option<LogIdOf<KiviTypeConfig>>,
            StoredMembershipOf<KiviTypeConfig>,
        ),
        std::io::Error,
    > {
        let core = self.core.lock().expect("mem sm lock holds");
        Ok((core.last_applied, core.membership.clone()))
    }

    async fn apply<Strm>(&mut self, entries: Strm) -> Result<(), std::io::Error>
    where
        Strm: futures::Stream<
                Item = Result<openraft::storage::EntryResponder<KiviTypeConfig>, std::io::Error>,
            > + Unpin
            + openraft::OptionalSend,
    {
        use futures::StreamExt as _;
        use openraft::EntryPayload;
        let mut entries = entries;
        while let Some(item) = entries.next().await {
            let (entry, responder) = item?;
            let mut core = self.core.lock().expect("mem sm lock holds");
            core.last_applied = Some(entry.log_id);
            let response = match &entry.payload {
                EntryPayload::Blank | EntryPayload::Membership(_) => {
                    if let EntryPayload::Membership(membership) = &entry.payload {
                        core.membership =
                            openraft::StoredMembership::new(Some(entry.log_id), membership.clone());
                    }
                    crate::mutation::ReplicatedOutcome::none()
                }
                EntryPayload::Normal(command) => {
                    core.applied_commands += 1;
                    match command {
                        crate::command::ConsensusCommand::Tablet(mutation) => {
                            crate::mutation::ReplicatedOutcome::new(mutation.expected().clone())
                        }
                        crate::command::ConsensusCommand::Control(_)
                        | crate::command::ConsensusCommand::ReadSync(_) => {
                            crate::mutation::ReplicatedOutcome::none()
                        }
                    }
                }
            };
            drop(core);
            if let Some(responder) = responder {
                responder.send(response);
            }
        }
        Ok(())
    }

    #[allow(clippy::unused_async_trait_impl)]
    async fn get_snapshot_builder(&mut self) -> Self::SnapshotBuilder {
        let core = self.core.lock().expect("mem sm lock holds");
        MemSnapshotBuilder {
            meta: SnapshotMetaOf::<KiviTypeConfig> {
                last_log_id: core.last_applied,
                last_membership: core.membership.clone(),
            },
        }
    }

    #[allow(clippy::unused_async_trait_impl)]
    async fn install_snapshot(
        &mut self,
        meta: &SnapshotMetaOf<KiviTypeConfig>,
        _snapshot: Self::SnapshotData,
    ) -> Result<(), std::io::Error> {
        let mut core = self.core.lock().expect("mem sm lock holds");
        core.last_applied = meta.last_log_id;
        core.membership = meta.last_membership.clone();
        Ok(())
    }

    #[allow(clippy::unused_async_trait_impl)]
    async fn get_current_snapshot(
        &mut self,
    ) -> Result<Option<SnapshotOf<KiviTypeConfig, Cursor<Vec<u8>>>>, std::io::Error> {
        Ok(None)
    }
}

fn unreachable<C: openraft::RaftTypeConfig>(target: u64) -> openraft::errors::RPCError<C> {
    let source = std::io::Error::new(
        std::io::ErrorKind::NotConnected,
        format!("spike stub network: no route to node {target}"),
    );
    openraft::errors::RPCError::Unreachable(openraft::errors::Unreachable::new(&source))
}

fn unreachable_stream<C: openraft::RaftTypeConfig>(
    target: u64,
) -> openraft::errors::StreamingError<C> {
    let source = std::io::Error::new(
        std::io::ErrorKind::NotConnected,
        format!("spike stub network: no route to node {target}"),
    );
    openraft::errors::StreamingError::Unreachable(openraft::errors::Unreachable::new(&source))
}

/// Unreachable single-target network for single-voter experiments.
///
/// A lone voter commits locally and never opens a replication stream; if
/// `OpenRaft` ever asks this stub to send, the `Unreachable` error is loud
/// and triggers backoff rather than silent success.
#[derive(Debug)]
pub struct StubNetwork {
    target: u64,
}

impl openraft::network::RaftNetworkV2<KiviTypeConfig> for StubNetwork {
    type SnapshotData = Cursor<Vec<u8>>;

    #[allow(clippy::unused_async_trait_impl)]
    async fn append_entries(
        &mut self,
        _rpc: openraft::raft::AppendEntriesRequest<KiviTypeConfig>,
        _option: openraft::network::RPCOption,
    ) -> Result<
        openraft::raft::AppendEntriesResponse<KiviTypeConfig>,
        openraft::errors::RPCError<KiviTypeConfig>,
    > {
        Err(unreachable(self.target))
    }

    #[allow(clippy::unused_async_trait_impl)]
    async fn vote(
        &mut self,
        _rpc: openraft::raft::VoteRequest<KiviTypeConfig>,
        _option: openraft::network::RPCOption,
    ) -> Result<
        openraft::raft::VoteResponse<KiviTypeConfig>,
        openraft::errors::RPCError<KiviTypeConfig>,
    > {
        Err(unreachable(self.target))
    }

    #[allow(clippy::unused_async_trait_impl)]
    async fn full_snapshot(
        &mut self,
        _vote: VoteOf<KiviTypeConfig>,
        _snapshot: SnapshotOf<KiviTypeConfig, Cursor<Vec<u8>>>,
        _cancel: impl Future<Output = openraft::errors::ReplicationClosed>
        + openraft::OptionalSend
        + 'static,
        _option: openraft::network::RPCOption,
    ) -> Result<
        openraft::raft::SnapshotResponse<KiviTypeConfig>,
        openraft::errors::StreamingError<KiviTypeConfig>,
    > {
        Err(unreachable_stream(self.target))
    }
}

/// Factory minting [`StubNetwork`] stubs (single-voter experiments only).
#[derive(Debug, Default)]
pub struct StubNetworkFactory;

impl openraft::network::RaftNetworkFactory<KiviTypeConfig> for StubNetworkFactory {
    type Network = StubNetwork;

    #[allow(clippy::unused_async_trait_impl)]
    async fn new_client(&mut self, target: u64, _node: &openraft::BasicNode) -> Self::Network {
        StubNetwork { target }
    }
}
