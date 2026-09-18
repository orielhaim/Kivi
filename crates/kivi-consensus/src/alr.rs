//! Lazy-ALR batch coordinator: true asynchronous Almost-Local Reads.
//!
//! The LAW impossibility result says purely local linearizable reads are
//! impossible under asynchrony with crashes — so this backend never claims
//! locality without ordering. Instead, pending `Latest` reads batch
//! opportunistically behind one lightweight ordered fence:
//!
//! ```text
//! pending reads ──► opportunistic batch ──► ordered ReadSync fence
//!       (no deliberate delay)      (ordering of a write, no mutation)
//!                            │
//!                            ▼
//!              fence (or a subsuming write) locally applied
//!                            │
//!                            ▼
//!              entire batch executes locally, exact `Latest`
//! ```
//!
//! Batching is strictly opportunistic: the first read with no batch in
//! flight forms one and orders its fence immediately; reads arriving while
//! the fence is outstanding join it; nobody ever waits to fill a batch.
//! A concurrent write ordered after batch formation may subsume the extra
//! fence (the leader says so with a proof — see `ProposeSync`), which the
//! coordinator records without changing the safety argument.
//!
//! The coordinator is node-local per tablet: all waiters share one state
//! machine, so the initiator's boundary wait covers every waiter. Any
//! failure — lost leadership, stale leader view, transport loss, boundary
//! timeout, owner shutdown — resolves to `Err(())`, and the caller falls
//! back to the conservative barrier. Uncertainty always falls back, never
//! serves.

use std::sync::{Arc, Mutex};

use kivi_types::{
    CommitPosition, NodeId, NodeIncarnation, ReadContext, ServePath, TabletAuthority, TabletId,
};

use crate::command::AlrFenceSync;
use crate::consistency::ConsistencyHub;
use crate::node::{OwnerRequest, SyncOutcome, owner_call_on};
use crate::state_machine::ReplicatedStateMachine;
use crate::types::ConsensusGroupId;

/// Result of one batch for one waiter: the ordered boundary (already
/// locally applied by the initiator's wait) plus whether a write subsumed
/// the fence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BatchResult {
    /// Raft log index covering the batch.
    boundary: u64,
    /// Whether a concurrent write subsumed the fence entry.
    subsumed: bool,
}

/// One in-flight batch: the waiters to wake on completion. (The fence
/// identity travels with the initiator's ordering call; waiters only
/// need the boundary result.)
struct InflightBatch {
    /// Waiters to complete (initiator serves itself separately).
    waiters: Vec<futures::channel::oneshot::Sender<BatchResult>>,
}

/// Coordinator interior: batch sequence plus the single in-flight batch.
struct AlrInner {
    /// Batch sequence for fence identities (monotonic per tablet).
    batch_seq: u64,
    /// Currently ordering batch, if any.
    inflight: Option<InflightBatch>,
}

/// Lazy-ALR batch coordinator for one tablet on one replica. `Send + Sync`
/// over a short mutex; only fence formation and waiter registration hold
/// it, never ordering or waits.
pub struct AlrCoordinator {
    /// Batch sequence plus in-flight batch.
    inner: Mutex<AlrInner>,
    /// Local state machine (formation pointer, boundary waits read here).
    machine: ReplicatedStateMachine,
    /// Owner queue (fence proposals, boundary waits).
    owner_tx: async_channel::Sender<OwnerRequest>,
    /// Node-wide consistency hub (batch metrics).
    hub: Arc<ConsistencyHub>,
    /// Addressed group.
    group: ConsensusGroupId,
    /// Served tablet.
    tablet: TabletId,
    /// Replica authority (diagnostics only; the fence carries the tablet).
    authority: TabletAuthority,
    /// This process's incarnation (fence requester binding).
    incarnation: NodeIncarnation,
    /// This replica's node identity (fence requester).
    local: NodeId,
}

impl std::fmt::Debug for AlrCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlrCoordinator")
            .field("tablet", &self.tablet)
            .field("group", &self.group)
            .finish_non_exhaustive()
    }
}

impl AlrCoordinator {
    /// Binds one tablet's coordinator. All handles are caller-side and
    /// `Send`; ordering and waits hop to the owner thread per batch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind(
        machine: ReplicatedStateMachine,
        owner_tx: async_channel::Sender<OwnerRequest>,
        hub: Arc<ConsistencyHub>,
        group: ConsensusGroupId,
        tablet: TabletId,
        authority: TabletAuthority,
        incarnation: NodeIncarnation,
        local: NodeId,
    ) -> Self {
        Self {
            inner: Mutex::new(AlrInner {
                batch_seq: 0,
                inflight: None,
            }),
            machine,
            owner_tx,
            hub,
            group,
            tablet,
            authority,
            incarnation,
            local,
        }
    }

    /// Joins (or forms) the tablet's Lazy-ALR batch and waits for its
    /// boundary to apply locally. Returns the serve path on success — the
    /// caller then reads live applied state — or `None` when the batch
    /// cannot order, in which case the caller takes the conservative
    /// barrier. `None` is always a fallback, never a serve.
    pub async fn join_batch(&self, ctx: ReadContext) -> Option<ServePath> {
        enum Role {
            /// This read initiates: order the fence, wait the boundary,
            /// complete the waiters.
            Initiate {
                /// Fence identity for the log.
                fence: kivi_types::AlrFence,
            },
            /// A batch is already ordering: register and await its result.
            Join {
                /// Channel the initiator completes.
                recv: futures::channel::oneshot::Receiver<BatchResult>,
            },
        }
        let role = {
            let Ok(mut inner) = self.inner.lock() else {
                return None;
            };
            if let Some(inflight) = inner.inflight.as_mut() {
                let (send, recv) = futures::channel::oneshot::channel();
                inflight.waiters.push(send);
                Role::Join { recv }
            } else {
                inner.batch_seq = inner.batch_seq.saturating_add(1);
                let fence = kivi_types::AlrFence::new(self.tablet, inner.batch_seq, self.local);
                inner.inflight = Some(InflightBatch {
                    waiters: Vec::new(),
                });
                Role::Initiate { fence }
            }
        };
        match role {
            Role::Join { recv } => {
                // The initiator's boundary wait covers this replica's
                // single machine, hence every waiter on it: applied only
                // advances, so a completed boundary stays covered.
                recv.await.ok().map(|result| {
                    if result.subsumed {
                        ServePath::AlrSubsumed
                    } else {
                        ServePath::AlrSync
                    }
                })
            }
            Role::Initiate { fence } => self.initiate_batch(fence, ctx).await,
        }
    }

    /// Initiator path: read the live formation pointer, order the fence
    /// (or accept a subsuming boundary), wait for local apply, complete
    /// the waiters, and report the path. Exactly one metric note per
    /// batch: the batch — not each read — is the unit of fence cost.
    async fn initiate_batch(
        &self,
        fence: kivi_types::AlrFence,
        ctx: ReadContext,
    ) -> Option<ServePath> {
        // Live formation pointer: the floor the boundary must cover. Read
        // at initiation (not at registration above, which only reserves
        // the batch slot): the fence orders after this instant, so the
        // prefix proof holds for this exact position.
        let formation = self.machine.status().await.applied_commit;
        let Some(outcome) = self.propose_fence(fence, formation, ctx).await else {
            self.complete_batch(None);
            self.hub.note_alr_fallback(self.tablet);
            return None;
        };
        // Defense in depth: a boundary below formation (stale or corrupt
        // leader answer) can never cover the batch — fail rather than
        // serve from it.
        let formation_index = crate::state_machine::index_of_commit(formation).unwrap_or(0);
        if outcome.boundary < formation_index {
            self.complete_batch(None);
            self.hub.note_alr_fallback(self.tablet);
            return None;
        }
        if self.wait_boundary(outcome.boundary, ctx).await.is_none() {
            self.complete_batch(None);
            self.hub.note_alr_fallback(self.tablet);
            return None;
        }
        let path = if outcome.subsumed {
            ServePath::AlrSubsumed
        } else {
            ServePath::AlrSync
        };
        let result = BatchResult {
            boundary: outcome.boundary,
            subsumed: outcome.subsumed,
        };
        if let Some(readers) = self.complete_batch(Some(result)) {
            self.hub
                .note_alr_batch(self.tablet, readers, outcome.subsumed);
        }
        Some(path)
    }

    /// Orders one fence through the owner (leader appends or subsumes; a
    /// follower forwards to the leader). Proposal failure means batch
    /// failure; the caller falls back to the barrier.
    async fn propose_fence(
        &self,
        fence: kivi_types::AlrFence,
        formation: CommitPosition,
        ctx: ReadContext,
    ) -> Option<SyncOutcome> {
        let sync = AlrFenceSync::new(fence, formation);
        let group = self.group;
        let timeout = ctx.capped_wait();
        match owner_call_on(&self.owner_tx, |reply| OwnerRequest::ProposeSync {
            group,
            sync,
            timeout,
            reply,
        })
        .await
        {
            Ok(Ok(outcome)) => Some(outcome),
            Ok(Err(_)) | Err(_) => None,
        }
    }

    /// Waits for local applied coverage of `boundary` on the read's
    /// bounded budget. Timeout (or owner death) fails the batch: the
    /// caller falls back rather than serving uncovered state.
    async fn wait_boundary(&self, boundary: u64, ctx: ReadContext) -> Option<()> {
        let group = self.group;
        let timeout = ctx.capped_wait();
        match owner_call_on(&self.owner_tx, |reply| OwnerRequest::WaitApplied {
            group,
            index: boundary,
            timeout,
            reply,
        })
        .await
        {
            Ok(Ok(())) => Some(()),
            Ok(Err(())) | Err(_) => None,
        }
    }

    /// Removes the in-flight batch and wakes every waiter with `result`
    /// (`None` fails them all). Returns the total reads the batch covered,
    /// initiator included, for the batch metric (`None` when the hub lock
    /// is poisoned, in which case no metric is recorded).
    fn complete_batch(&self, result: Option<BatchResult>) -> Option<u64> {
        let waiters = {
            let Ok(mut inner) = self.inner.lock() else {
                return None;
            };
            inner
                .inflight
                .take()
                .map_or_else(Vec::new, |batch| batch.waiters)
        };
        let readers = waiters.len() as u64 + 1;
        if let Some(result) = result {
            for waiter in waiters {
                let _ = waiter.send(result);
            }
        }
        // `None` drops the senders: every waiter observes the cancelled
        // batch and falls back. No waiter ever hangs on a dead initiator.
        Some(readers)
    }

    /// Returns the tablet served.
    #[must_use]
    pub const fn tablet(&self) -> TabletId {
        self.tablet
    }

    /// Returns the replica authority (diagnostics).
    #[must_use]
    pub const fn authority(&self) -> TabletAuthority {
        self.authority
    }

    /// Returns this process's incarnation.
    #[must_use]
    pub const fn incarnation(&self) -> NodeIncarnation {
        self.incarnation
    }
}
