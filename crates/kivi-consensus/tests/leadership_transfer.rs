//! Leadership handoff to a target that is still inside a replication
//! backoff.
//!
//! Kivi moves leadership for migrations (`transfer_leader` to a retained
//! voter) and rebalancing, and the target of such a move is routinely a
//! node that just restarted or just rejoined. Those targets are exactly
//! the ones sitting inside `OpenRaft`'s replication backoff: the leader
//! failed to reach them, so it is sleeping before it retries. The
//! transfer broadcast is one-shot, and a target that has not caught up
//! refuses it with `LogNotFlushed`, so without a backoff reset the handoff
//! silently never happens.
//!
//! These tests pin the two behaviours Kivi depends on:
//! [`Config::reset_backoff_on_transfer_leader`] for the transfer, and
//! [`openraft::raft::Trigger::reset_backoff`] for a declared peer heal
//! (which [`kivi_consensus`] calls from its resume path).
//!
//! Both run three real in-process `Raft` voters over the real
//! [`GroupRouter`] contract, with a backoff long enough that "the backoff
//! was not cleared" cannot masquerade as a slow test: without a reset the
//! target is asleep for an hour and the test times out in seconds.
//!
//! [`Config::reset_backoff_on_transfer_leader`]: openraft::Config::reset_backoff_on_transfer_leader
//! [`GroupRouter`]: openraft_multi::GroupRouter

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Cursor;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use kivi_consensus::ConsensusGroupId;
use kivi_consensus::config::KiviTypeConfig;
use kivi_consensus::config::cluster_config;
use kivi_consensus::router::GroupNetworkFactory;
use kivi_consensus::spike::{MemLogStore, MemStateMachine};
use kivi_types::TabletId;
use openraft::Raft;
use openraft::type_config::async_runtime::watch::WatchReceiver as _;

/// A delay no test waits out. Chosen far above the per-test deadline so
/// "the backoff was not cleared" fails fast and unambiguously instead of
/// intermittently passing on a slow machine.
const PARALYSIS: &str = "3600s";

/// Generous upper bound for "promptly", across CI oversubscription. Still
/// two orders of magnitude below [`PARALYSIS`].
const PROMPT: Duration = Duration::from_secs(10);

const VOTERS: [u64; 3] = [1, 2, 3];
const TABLET: u64 = 17;

type TestRaft = Raft<KiviTypeConfig, MemStateMachine>;

/// In-process group mesh: one registry keyed by `(node, group)` serving
/// each target's `Raft` directly, with per-target reachability so a test
/// can drop a link exactly as the real mesh drops one. Mirrors
/// [`PeerRouter`](kivi_consensus::router::PeerRouter)'s rule that anything
/// which is not a remote verdict is a retry-safe `Unreachable`.
///
/// Lives here rather than in the crate: it is a test seam with a
/// reachability switch the production router has no use for.
#[derive(Clone, Default)]
struct Registry {
    rafts: Rc<RefCell<BTreeMap<(u64, u64), TestRaft>>>,
    /// Targets currently refusing every RPC, as their in-process peers
    /// would while the link is down.
    isolated: Rc<RefCell<BTreeSet<u64>>>,
}

impl Registry {
    /// Marks `node` reachable or isolated. An isolated node answers every
    /// RPC with `Unreachable`, the same verdict Kivi's mesh gives a peer
    /// whose link is down.
    fn reach(&self, node: u64, reachable: bool) {
        let mut isolated = self.isolated.borrow_mut();
        if reachable {
            isolated.remove(&node);
        } else {
            isolated.insert(node);
        }
    }

    fn target(&self, node: u64) -> Result<TestRaft, openraft::errors::Unreachable<KiviTypeConfig>> {
        if self.isolated.borrow().contains(&node) {
            return Err(unreachable(node, "link down"));
        }
        self.rafts
            .borrow()
            .get(&(node, TABLET))
            .cloned()
            .ok_or_else(|| unreachable(node, "not spawned"))
    }
}

fn unreachable(node: u64, detail: &str) -> openraft::errors::Unreachable<KiviTypeConfig> {
    openraft::errors::Unreachable::new(&std::io::Error::new(
        std::io::ErrorKind::NotConnected,
        format!("registry: node {node} {detail}"),
    ))
}

impl openraft_multi::GroupRouter<KiviTypeConfig, ConsensusGroupId> for Registry {
    type SnapshotData = Cursor<Vec<u8>>;

    async fn append_entries(
        &self,
        target: u64,
        _group: ConsensusGroupId,
        rpc: openraft::raft::AppendEntriesRequest<KiviTypeConfig>,
        _option: openraft::network::RPCOption,
    ) -> Result<
        openraft::raft::AppendEntriesResponse<KiviTypeConfig>,
        openraft::errors::RPCError<KiviTypeConfig>,
    > {
        let raft = self
            .target(target)
            .map_err(openraft::errors::RPCError::Unreachable)?;
        raft.append_entries(rpc).await.map_err(|error| {
            openraft::errors::RPCError::Unreachable(unreachable(target, &error.to_string()))
        })
    }

    async fn vote(
        &self,
        target: u64,
        _group: ConsensusGroupId,
        rpc: openraft::raft::VoteRequest<KiviTypeConfig>,
        _option: openraft::network::RPCOption,
    ) -> Result<
        openraft::raft::VoteResponse<KiviTypeConfig>,
        openraft::errors::RPCError<KiviTypeConfig>,
    > {
        let raft = self
            .target(target)
            .map_err(openraft::errors::RPCError::Unreachable)?;
        raft.vote(rpc).await.map_err(|error| {
            openraft::errors::RPCError::Unreachable(unreachable(target, &error.to_string()))
        })
    }

    async fn pre_vote(
        &self,
        target: u64,
        _group: ConsensusGroupId,
        rpc: openraft::raft::VoteRequest<KiviTypeConfig>,
        _option: openraft::network::RPCOption,
    ) -> Result<
        openraft::raft::VoteResponse<KiviTypeConfig>,
        openraft::errors::RPCError<KiviTypeConfig>,
    > {
        let raft = self
            .target(target)
            .map_err(openraft::errors::RPCError::Unreachable)?;
        raft.pre_vote(rpc).await.map_err(|error| {
            openraft::errors::RPCError::Unreachable(unreachable(target, &error.to_string()))
        })
    }

    async fn transfer_leader(
        &self,
        target: u64,
        _group: ConsensusGroupId,
        rpc: openraft::raft::TransferLeaderRequest<KiviTypeConfig>,
        _option: openraft::network::RPCOption,
    ) -> Result<
        openraft::raft::TransferLeaderResponse<KiviTypeConfig>,
        openraft::errors::RPCError<KiviTypeConfig>,
    > {
        let raft = self
            .target(target)
            .map_err(openraft::errors::RPCError::Unreachable)?;
        raft.handle_transfer_leader(rpc).await.map_err(|error| {
            openraft::errors::RPCError::Unreachable(unreachable(target, &error.to_string()))
        })
    }

    #[allow(clippy::unused_async_trait_impl)]
    async fn full_snapshot(
        &self,
        target: u64,
        _group: ConsensusGroupId,
        _vote: openraft::type_config::alias::VoteOf<KiviTypeConfig>,
        _snapshot: openraft::type_config::alias::SnapshotOf<KiviTypeConfig, Cursor<Vec<u8>>>,
        _cancel: impl Future<Output = openraft::errors::ReplicationClosed>
        + openraft::OptionalSend
        + 'static,
        _option: openraft::network::RPCOption,
    ) -> Result<
        openraft::raft::SnapshotResponse<KiviTypeConfig>,
        openraft::errors::StreamingError<KiviTypeConfig>,
    > {
        // These tests never reach log compaction; a call here would be a
        // harness bug, refused loudly rather than papered over.
        Err(openraft::errors::StreamingError::Unreachable(unreachable(
            target,
            "snapshot transfer not used by this harness",
        )))
    }
}

/// Kivi's production config, with one field widened: the backoff becomes
/// an hour so a missing reset fails the assertion instead of sleeping
/// politely for 200ms. Everything else - including
/// `reset_backoff_on_transfer_leader` - is exactly what Kivi ships.
fn test_config() -> Arc<openraft::Config> {
    let mut config = cluster_config()
        .expect("cluster config validates")
        .as_ref()
        .clone();
    PARALYSIS.clone_into(&mut config.backoff);
    Arc::new(config.validate().expect("widened backoff still validates"))
}

/// A live three-voter group over the shared registry: one leader plus
/// two followers, the topology every real Kivi group has. The returned
/// handles are in [`VOTERS`] order.
async fn group(registry: &Registry) -> Vec<TestRaft> {
    let config = test_config();
    let group_id = ConsensusGroupId::of_tablet(TabletId::from_u64(TABLET));
    let members: BTreeMap<u64, openraft::BasicNode> = VOTERS
        .into_iter()
        .map(|node| (node, openraft::BasicNode::new(format!("n{node}"))))
        .collect();
    let mut rafts = Vec::with_capacity(VOTERS.len());
    for node in VOTERS {
        let raft = Raft::new(
            node,
            Arc::clone(&config),
            GroupNetworkFactory::new(registry.clone(), group_id),
            MemLogStore::new(),
            MemStateMachine::default(),
        )
        .await
        .expect("raft spawns");
        registry
            .rafts
            .borrow_mut()
            .insert((node, TABLET), raft.clone());
        if let Err(error) = raft.initialize(members.clone()).await
            && !matches!(
                error,
                openraft::errors::RaftError::APIError(
                    openraft::errors::InitializeError::NotAllowed(_)
                )
            )
        {
            panic!("node {node} initializes: {error:?}");
        }
        rafts.push(raft);
    }
    rafts
}

/// The leader's own handle (`group` returns handles in [`VOTERS`] order).
fn leader_handle(rafts: &[TestRaft], leader: u64) -> TestRaft {
    rafts[VOTERS
        .iter()
        .position(|voter| *voter == leader)
        .expect("the leader is a voter")]
    .clone()
}

/// Each voter's current view of the leader, in [`VOTERS`] order.
fn leader_views(rafts: &[TestRaft]) -> Vec<Option<u64>> {
    rafts
        .iter()
        .map(|raft| raft.metrics().borrow_watched().current_leader)
        .collect()
}

/// The leader every voter agrees on, or `None` while the group is still
/// converging.
fn leader_of(rafts: &[TestRaft]) -> Option<u64> {
    let mut views = leader_views(rafts).into_iter();
    let first = views.next()?;
    if views.all(|view| view == first) {
        first
    } else {
        None
    }
}

/// Waits until every voter agrees on a leader, then returns it.
async fn elect(rafts: &[TestRaft]) -> u64 {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(leader) = leader_of(rafts) {
            return leader;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "group never elected a leader"
        );
        compio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Waits until every voter agrees that `node` leads. Agreement, not one
/// node's optimism: a transfer that only half-landed leaves the group
/// disagreeing or still on the incumbent.
async fn await_leadership(rafts: &[TestRaft], node: u64) {
    let deadline = std::time::Instant::now() + PROMPT;
    loop {
        let views = leader_views(rafts);
        if views.iter().all(|view| *view == Some(node)) {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "node {node} never became leader within {PROMPT:?} (views {views:?})"
        );
        compio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Waits until the leader's view of `node`'s matched index reaches
/// `index` - i.e. replication to that node has resumed and delivered.
async fn await_matched(leader: &TestRaft, node: u64, index: u64) {
    let deadline = std::time::Instant::now() + PROMPT;
    loop {
        let metrics = leader.metrics().borrow_watched().clone();
        let matched = metrics
            .replication
            .as_ref()
            .and_then(|replication| replication.get(&node))
            .and_then(|matched| matched.as_ref().map(|id| id.index))
            .unwrap_or(0);
        if matched >= index {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "node {node} never reached index {index} within {PROMPT:?} (matched {matched})"
        );
        compio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Puts `target` inside a replication backoff: its link goes down, a
/// proposal through the leader advances the log past it, and the failed
/// append engages the backoff. Returns the leader's handle and the index
/// `target` is now missing, so a later catch-up assertion can name it.
async fn backoff_against(
    rafts: &[TestRaft],
    registry: &Registry,
    leader: u64,
    target: u64,
) -> (TestRaft, u64) {
    let handle = leader_handle(rafts, leader);
    registry.reach(target, false);
    let written = handle
        .client_write(command())
        .await
        .expect("proposal commits with a majority of two");
    // The leader's replication to `target` just failed, engaging the
    // backoff. The one-shot transfer broadcast that follows is refused
    // with `LogNotFlushed` unless the backoff is cleared first.
    let metrics = handle.metrics().borrow_watched().clone();
    assert_eq!(
        metrics.current_leader,
        Some(leader),
        "node {leader} should still lead after the isolated proposal"
    );
    (handle, written.log_id.index)
}

/// A voter other than `leader`, i.e. the only candidates a transfer can
/// target.
fn follower_of(leader: u64) -> u64 {
    VOTERS
        .into_iter()
        .find(|node| *node != leader)
        .expect("a three-voter group has followers")
}

/// One deterministic command; the state machine installs its `expected`
/// verbatim, which is all these tests need.
fn command() -> kivi_consensus::ConsensusCommand {
    use kivi_state::{Key, Mutation, MutationEnvelope, ObjectVersion, OperationResult};
    use kivi_types::{
        NamespaceId, RequestIdentity, RequestSeq, SessionId, TabletAuthority, TabletEpoch,
        WallTimestamp, WriteGuardGeneration,
    };
    kivi_consensus::ConsensusCommand::Tablet(kivi_consensus::ReplicatedMutation::new(
        MutationEnvelope::new(
            NamespaceId::from_u64(1),
            TabletAuthority::new(
                TabletId::from_u64(TABLET),
                TabletEpoch::INITIAL,
                WriteGuardGeneration::INITIAL,
            ),
            RequestIdentity::new(SessionId::from_u128(7), RequestSeq::from_u64(1)),
            None,
            Mutation::CounterAdd {
                key: Key::from("n"),
                delta: 1,
            },
        ),
        OperationResult::CounterUpdated {
            value: 1,
            version: ObjectVersion::FIRST,
        },
        WallTimestamp::from_micros(5),
        RequestSeq::from_u64(0),
    ))
}

/// Leadership moves to a recovered target that is still inside the
/// backoff the leader accrued while it was away.
///
/// This is the migration and rebalance shape Kivi drives through
/// `transfer_leader`: a target that just restarted sits in a backoff, the
/// transfer broadcast reaches it before it has caught up, and the one-shot
/// handoff is refused. `reset_backoff_on_transfer_leader` clears the
/// target's backoff first, so it catches up and wins the election.
#[test]
fn transfer_to_a_recovered_target_lands_promptly() {
    compio::runtime::Runtime::new()
        .expect("compio test runtime")
        .block_on(async {
            let registry = Registry::default();
            let rafts = group(&registry).await;
            let leader = elect(&rafts).await;
            let target = follower_of(leader);

            let (raft, _missing) = backoff_against(&rafts, &registry, leader, target).await;
            registry.reach(target, true);
            raft.trigger()
                .transfer_leader(target)
                .await
                .expect("transfer dispatch");

            // Catch-up and election are the same observable here: the
            // transfer only completes once the target has flushed the
            // leader's last log id. A refused transfer leaves the
            // incumbent leading and this never lands.
            await_leadership(&rafts, target).await;
            for handle in &rafts {
                handle.shutdown().await.expect("group shuts down");
            }
        });
}

/// A declared peer heal ends the backoff on demand: replication to the
/// recovered peer resumes at once instead of waiting out a delay chosen
/// while it was down.
///
/// This is what Kivi's resume path (`POST /v1/peers/resume`, and the
/// chaos harness behind it) buys with
/// [`Trigger::reset_backoff`](openraft::raft::Trigger::reset_backoff).
#[test]
fn reset_backoff_resumes_replication_to_a_healed_peer() {
    compio::runtime::Runtime::new()
        .expect("compio test runtime")
        .block_on(async {
            let registry = Registry::default();
            let rafts = group(&registry).await;
            let leader = elect(&rafts).await;
            let target = follower_of(leader);

            let (raft, missing) = backoff_against(&rafts, &registry, leader, target).await;
            registry.reach(target, true);
            // The peer is reachable again, which is exactly what Kivi knows
            // at this point and nothing more: no transfer is in flight, so
            // only the explicit reset can resume replication.
            raft.trigger()
                .reset_backoff([target])
                .await
                .expect("backoff reset dispatch");

            await_matched(&raft, target, missing).await;
            for handle in &rafts {
                handle.shutdown().await.expect("group shuts down");
            }
        });
}
