//! Follower durability gate: sidecar acquisition before Raft persistence.
//!
//! This is the heart of replicated large values. When `OpenRaft` asks the
//! follower store to persist entries referencing immutable sidecars, this
//! gate ensures every required sidecar is locally durable and verified
//! before the Raft append may complete:
//!
//! ```text
//! Raft append -> detect sidecar dependency
//!   -> if present: verify metadata, persist Raft record, barrier, return
//!   -> if missing: resolve manifest, verify, discover missing chunks,
//!      resolve only missing, verify hashes/length/domain, durably install,
//!      persist Raft entry, barrier, return
//! ```
//!
//! "Resolve" is one path with several sources (see [`SidecarGate`]): the local
//! store, the peer mesh, and then whatever [`SidecarResolver`] is installed.
//! The last of those is what makes redundancy-backed sidecars transparent to
//! replication - a member that lost the whole copy of a manifest rebuilds it
//! from surviving fragments instead of failing the append, so losing a member
//! costs a fetch rather than the whole group's progress.
//!
//! The follower never fakes persistence while transfer is incomplete. A
//! corrupt sidecar fails the transfer, prevents the append from becoming
//! durable, and surfaces as an `io::Error` (which halts the node loudly,
//! never a silent ACK). What changed is *which conditions can reach that
//! point*: a recoverable miss is resolved or waited out, and only content that
//! is genuinely gone or genuinely corrupt gets there.
//!
//! ## Why the gate stays at the flush boundary
//!
//! A sidecar failure here is expensive: `OpenRaft` treats a storage error as
//! fatal, so the node stops participating - including the empty heartbeats
//! that let the rest of the group replace it. Moving the gate earlier (reject
//! the proposal, or defer to apply) would trade that fatal for a different
//! wrongness, because a committed entry naming content the node cannot later
//! materialize is an availability bug on restart, discovered when the operator
//! needs the data most.
//!
//! So the boundary is right and the fix belongs on the resolution side, which is
//! what this module now does. The remaining escalation is reserved for content
//! that is genuinely unrecoverable, and by then "halt loudly" is the honest
//! answer: continuing would mean acknowledging an append whose payload this
//! node can never read.
//!
//! ## `OpenRaft` 0.10 persistence contract (audited, never 0.9)
//!
//! * `append` must return with entries readable; `IOFlushed` fires once
//!   entries are persisted. Replication ACKs imply durable persistence
//!   (see `store.rs` docs): the leader commits only on quorum flushed, so
//!   gating the flush gates the commit.
//! * The sidecar must be local before the entry is durable, because a
//!   committed entry that references content the node cannot later
//!   materialize is an availability bug on restart. That invariant is
//!   unchanged: the gate sits at the flush boundary, and resolution is how
//!   the node satisfies it rather than how it evades it.
//! * The gate runs before the WAL barrier submit (no lock held across the
//!   fetch), so the reactor stays responsive while bulk moves on the
//!   bulk lane.
//! * Already-durable content-addressed chunks remain cached safely if the
//!   suffix truncates mid-fetch; no physical rollback is required.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use kivi_types::{ChunkId, ManifestId, NodeId, SecurityDomainId};

use crate::peer::{PeerChunkRequest, PeerManifestRequest, PeerRequest, PeerResponse};
use crate::resolve::{ResolutionSlots, ResolveError, ResolveFuture};
use crate::sidecar::{ImmutableDependencies, SidecarError, SidecarStore};
use crate::transport::{PeerTransport, TransportError};
use crate::types::ConsensusGroupId;

/// Maximum concurrent chunk fetches per gated root (bounded bulk).
const MAX_CONCURRENT_CHUNK_FETCH: usize = 4;
/// Maximum inflight gated roots (bounded acquisition table).
const MAX_INFLIGHT_ROOTS: usize = 64;

/// How many times a resolver may report "still coming" before the gate gives
/// up waiting and surfaces the condition.
const RESOLVER_PENDING_ROUNDS: usize = 3;
/// First pause between pending rounds; grows linearly with the round.
const PENDING_BACKOFF: Duration = Duration::from_millis(40);

/// Where one root's acquisition stands.
#[derive(Debug, Clone)]
enum FetchState {
    /// A task holds this root. Waiters poll the store until it finishes,
    /// because a finished outcome is only published by removing the entry or
    /// recording a definite failure.
    Inflight,
    /// Finished without producing the content, for a reason that is still
    /// true: no source had it, and the resolver could not rebuild it. Kept so
    /// a late caller learns *why* instead of concluding from the entry's
    /// absence that the fetch "did not complete".
    ///
    /// Only definitive outcomes are kept. A transient one - the resolver
    /// answering "not published yet" while a holder restarts - is a fact about
    /// one instant, and serving it for the retention window would report a
    /// recoverable condition as a permanent loss.
    Failed(SidecarError),
}

/// In-flight sidecar fetches and, briefly, the definitive failures among them.
///
/// One record under one lock: a manifest cannot be inflight in one part and
/// finished in the other, which is the only way this state is consistent.
#[derive(Default)]
struct SidecarFetches {
    /// Current state per root. Absent means "never attempted"; a root that
    /// succeeded is absent again, because the store itself answers that.
    states: HashMap<ManifestId, FetchState>,
    /// Roots with a recorded failure, oldest first (the prune order).
    failed: std::collections::VecDeque<ManifestId>,
}

impl SidecarFetches {
    /// Records a finished acquisition: a definite failure is remembered for
    /// late callers, anything else clears the entry so the next caller makes a
    /// fresh attempt.
    fn finish(&mut self, manifest: ManifestId, outcome: &Result<(), SidecarError>) {
        self.states.remove(&manifest);
        let Some(error) = outcome.as_ref().err() else {
            return;
        };
        if error.is_retryable() {
            return;
        }
        self.states
            .insert(manifest, FetchState::Failed(error.clone()));
        self.failed.push_back(manifest);
    }

    /// Claims the root for this caller, or reports what a finished attempt
    /// already concluded.
    fn claim(&mut self, manifest: ManifestId) -> Claim {
        match self.states.get(&manifest) {
            Some(FetchState::Failed(error)) => Claim::AlreadyFailed(error.clone()),
            Some(FetchState::Inflight) => Claim::Held,
            None => {
                self.states.insert(manifest, FetchState::Inflight);
                Claim::Acquired
            }
        }
    }

    /// Drops the oldest recorded failures past the retention bound.
    ///
    /// A record only helps a caller that arrives moments later, so the bound is
    /// what stops the table growing with every sidecar the node ever touched.
    /// Explicit rather than a timer, so it needs no background work to stay
    /// honest. An entry that has since been re-fetched is inflight again and is
    /// left alone.
    fn prune(&mut self) {
        while self.failed.len() > RETAINED_RESULTS {
            let Some(old) = self.failed.pop_front() else {
                return;
            };
            if matches!(self.states.get(&old), Some(FetchState::Failed(_))) {
                self.states.remove(&old);
            }
        }
    }
}

/// The outcome of claiming a root.
enum Claim {
    /// This caller now owns the acquisition.
    Acquired,
    /// Another task holds it; wait on the store.
    Held,
    /// A finished attempt already concluded, for a reason that still holds.
    AlreadyFailed(SidecarError),
}

/// How many finished fetches keep their outcome readable.
///
/// A waiter needs the record only if it arrives moments after the fetch, so a
/// small window suffices. The bound is what stops the table growing with every
/// sidecar the node ever touched.
const RETAINED_RESULTS: usize = 16;
const SOURCE_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(10);

/// Durability gate for sidecar-backed Raft entries.
///
/// One resolution path, several sources. For every byte the gate installs it
/// asks the same question - "give me this content, complete and verified" -
/// and takes the first verified answer:
///
/// ```text
/// local store (already durable)
///   -> miss
/// peer mesh (whole manifest / whole chunk)
///   -> nobody has it
/// installed resolver (redundancy reconstruction, another storage tier, ...)
///   -> verified bytes
/// install locally, prove durable, return
/// ```
///
/// The gate deliberately does not know *how* the last step works. That is
/// [`SidecarResolver`](crate::resolve::SidecarResolver)'s business, which is
/// what keeps erasure coding out of the Raft path: a missing whole copy is an
/// ordinary miss here, not a dead end.
///
/// A resolver that answers [`ResolveError::Pending`](crate::resolve::ResolveError::Pending)
/// earns bounded retries rather than an immediate failure - that is the
/// difference between a restarting member and a wedged group. Nothing else is
/// retried: an unrecoverable answer is a fixed fact, and a corrupt one is
/// never worth a second attempt.
#[derive(Debug, Clone)]
pub struct SidecarGate {
    store: SidecarStore,
    transport: PeerTransport,
    peers: Arc<Vec<NodeId>>,
    group: ConsensusGroupId,
    domain: SecurityDomainId,
    bulk_timeout: Duration,
    fetches: Arc<futures::lock::Mutex<SidecarFetches>>,
    slots: Arc<ResolutionSlots>,
}

impl SidecarGate {
    /// Creates a gate over a sidecar store and peer mesh.
    ///
    /// `peers` names candidate sidecar sources (other replicas; the local
    /// node need not be listed because the fast path already checked the
    /// local store). Static topology: fixed at node open.
    ///
    /// `slots` carries the resolver installed later (the redundancy plane is
    /// built from an already-open node, so it cannot exist at construction
    /// time). Sharing the node's cells is what makes the gate's fallback and
    /// the propose path's protection one system rather than two.
    #[must_use]
    pub fn new(
        store: SidecarStore,
        transport: PeerTransport,
        peers: Vec<NodeId>,
        group: ConsensusGroupId,
        domain: SecurityDomainId,
        bulk_timeout: Duration,
        slots: Arc<ResolutionSlots>,
    ) -> Self {
        Self {
            store,
            transport,
            peers: Arc::new(peers),
            group,
            domain,
            bulk_timeout,
            fetches: Arc::new(futures::lock::Mutex::new(SidecarFetches::default())),
            slots,
        }
    }

    fn all_sources(&self, hint: Option<NodeId>) -> Vec<NodeId> {
        let mut sources = Vec::with_capacity(self.peers.len());
        if let Some(leader) = hint
            && self.peers.contains(&leader)
        {
            sources.push(leader);
        }
        sources.extend(
            self.peers
                .iter()
                .copied()
                .filter(|peer| Some(*peer) != hint),
        );
        sources
    }

    async fn ordered_sources(&self, hint: Option<NodeId>) -> Vec<NodeId> {
        if hint.is_some() {
            return self.all_sources(hint);
        }
        let stats = self.transport.stats().await;
        let mut live = Vec::with_capacity(self.peers.len());
        let mut offline = Vec::with_capacity(self.peers.len());
        for peer in self.peers.iter().copied() {
            if stats
                .get(&peer)
                .is_some_and(|stats| stats.connected_bulk || stats.connected_control)
            {
                live.push(peer);
            } else {
                offline.push(peer);
            }
        }
        live.sort_by_key(|peer| peer.as_u64());
        offline.sort_by_key(|peer| peer.as_u64());
        live.extend(offline);
        live
    }

    /// Returns the sidecar store.
    #[must_use]
    pub fn store(&self) -> &SidecarStore {
        &self.store
    }

    /// Returns the domain.
    #[must_use]
    pub const fn domain(&self) -> SecurityDomainId {
        self.domain
    }

    /// Extracts immutable dependencies from one replicated command's bytes.
    ///
    /// Returns an empty vector for small/state-local mutations (no
    /// sidecars) and for control-plane commands (never chunked). Chunked
    /// roots - direct or carried inside transaction prepares and local
    /// commits - each contribute one entry, so followers fetch every
    /// payload a commit may reference before the entry may apply.
    #[must_use]
    pub fn dependencies_of(command: &[u8]) -> Vec<ImmutableDependencies> {
        let Some(envelope_command) = crate::command::ConsensusCommand::decode_exact(command).ok()
        else {
            return Vec::new();
        };
        let Some(mutation) = envelope_command.as_tablet() else {
            return Vec::new();
        };
        let envelope = mutation.envelope();
        // Domain binds at apply time from namespace; the gate uses
        // its configured domain (namespace-derived, single-domain
        // lanes). Cross-domain manifests fail verification later.
        let domain = SecurityDomainId::from_u64(0);
        match envelope.operation() {
            kivi_state::Mutation::ReplaceChunkedRoot {
                manifest,
                logical_len,
                ..
            }
            | kivi_state::Mutation::ReplaceChunkedRootWithExpiry {
                manifest,
                logical_len,
                ..
            }
            | kivi_state::Mutation::TxnPrepare {
                write:
                    kivi_state::TxnWriteKind::PutChunked {
                        manifest,
                        logical_len,
                    },
                ..
            } => vec![ImmutableDependencies::new(*manifest, *logical_len, domain)],
            kivi_state::Mutation::TxnCommitLocal { writes, .. } => writes
                .iter()
                .filter_map(|write| match &write.kind {
                    kivi_state::TxnWriteKind::PutChunked {
                        manifest,
                        logical_len,
                    } => Some(ImmutableDependencies::new(*manifest, *logical_len, domain)),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Extracts dependencies with the gate's domain filled in.
    #[must_use]
    pub fn dependencies_of_with_domain(&self, command: &[u8]) -> Vec<ImmutableDependencies> {
        Self::dependencies_of(command)
            .into_iter()
            .map(|mut deps| {
                deps.domain = self.domain;
                deps
            })
            .collect()
    }

    /// Ensures one root is locally durable, fetching missing sidecars from
    /// peers. Concurrent callers for the same manifest coalesce: only one
    /// fetch runs while others wait on the inflight entry (bounded).
    ///
    /// # Errors
    ///
    /// Returns [`SidecarError`] on verification failure, missing source,
    /// overload, or shutdown. Callers map this to `io::Error` so the Raft
    /// append never reports durable persistence for incomplete sidecars.
    pub async fn ensure_root(
        &self,
        manifest: ManifestId,
        logical_len: u64,
        hint: Option<NodeId>,
    ) -> Result<(), SidecarError> {
        self.ensure_root_with_mode(manifest, logical_len, hint, false)
            .await
    }

    pub(crate) async fn ensure_root_preflight(
        &self,
        manifest: ManifestId,
        logical_len: u64,
        hint: Option<NodeId>,
    ) -> Result<(), SidecarError> {
        self.ensure_root_with_mode(manifest, logical_len, hint, true)
            .await
    }

    async fn ensure_root_with_mode(
        &self,
        manifest: ManifestId,
        logical_len: u64,
        hint: Option<NodeId>,
        preflight: bool,
    ) -> Result<(), SidecarError> {
        if !manifest.is_valid() {
            return Err(SidecarError::Corrupt {
                detail: "zero manifest id".to_owned(),
            });
        }
        // Fast path: already durable (dedup hit, no transfer).
        if self.store.check_root(manifest, logical_len).await? {
            self.store
                .metrics()
                .cache_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        self.store
            .metrics()
            .cache_misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Coalesce: bound the inflight table, wait if another task fetches
        // the same root.
        {
            let mut fetches = self.fetches.lock().await;
            if fetches.states.len() >= MAX_INFLIGHT_ROOTS && !fetches.states.contains_key(&manifest)
            {
                return Err(SidecarError::Overloaded {
                    detail: "too many inflight sidecar acquisitions".to_owned(),
                });
            }
            match fetches.claim(manifest) {
                // A definitive outcome is re-served verbatim - the same variant,
                // not a re-wrapped `Missing`, so a caller can still tell a lost
                // copy from a corrupt one.
                Claim::AlreadyFailed(error) => return Err(error),
                Claim::Held => {
                    drop(fetches);
                    return self.wait_for_root(manifest, logical_len).await;
                }
                Claim::Acquired => {}
            }
            self.store
                .metrics()
                .inflight
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let sources = self.ordered_sources(hint).await;
        let result = self
            .fetch_root(manifest, logical_len, &sources, hint, preflight)
            .await;
        {
            // Publish the outcome: a definite failure is remembered so a late
            // caller learns why, and a success or a transient failure clears the
            // entry so the next caller is not served a stale verdict.
            let mut fetches = self.fetches.lock().await;
            fetches.finish(manifest, &result);
            fetches.prune();
            self.store
                .metrics()
                .inflight
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
        result
    }

    /// Ensures all sidecar dependencies across entries (batch entry point
    /// for `append`: resolves each manifest once, reuses across entries).
    ///
    /// # Errors
    ///
    /// Returns [`SidecarError`] when any root cannot be made durable.
    pub async fn ensure_entries(
        &self,
        commands: &[Vec<u8>],
        hint: Option<NodeId>,
    ) -> Result<(), SidecarError> {
        use std::collections::HashSet;
        self.store
            .metrics()
            .pending_gated
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let result = async {
            let mut seen: HashSet<(ManifestId, u64)> = HashSet::new();
            for command in commands {
                for deps in self.dependencies_of_with_domain(command) {
                    if seen.insert((deps.manifest, deps.logical_len)) {
                        self.ensure_root(deps.manifest, deps.logical_len, hint)
                            .await?;
                    }
                }
            }
            Ok(())
        }
        .await;
        self.store
            .metrics()
            .pending_gated
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        result
    }

    async fn wait_for_root(
        &self,
        manifest: ManifestId,
        logical_len: u64,
    ) -> Result<(), SidecarError> {
        // Poll with bounded waits: the fetcher holds the inflight entry
        // until durable or failed. Timeout surfaces as overload (caller
        // retries via Raft replication backoff).
        let deadline = std::time::Instant::now() + self.bulk_timeout * 2;
        loop {
            compio::time::sleep(Duration::from_millis(50)).await;
            if self.store.check_root(manifest, logical_len).await? {
                return Ok(());
            }
            // A finished fetch that could not produce the content records why.
            // Inferring failure from the entry's *absence* is what produced "did
            // not complete" for fetches that had failed for a reason worth
            // acting on - and, separately, for fetches that had succeeded a
            // moment before the absence was observable. Success is not recorded
            // at all: the store is the authority for it, and the check above
            // already reads it.
            if let Some(FetchState::Failed(error)) = self.fetches.lock().await.states.get(&manifest)
            {
                return Err(error.clone());
            }
            if std::time::Instant::now() >= deadline {
                return Err(SidecarError::Overloaded {
                    detail: format!("coalesced fetch for {manifest} timed out"),
                });
            }
        }
    }

    async fn fetch_root(
        &self,
        manifest: ManifestId,
        logical_len: u64,
        sources: &[NodeId],
        hint: Option<NodeId>,
        preflight: bool,
    ) -> Result<(), SidecarError> {
        self.store
            .metrics()
            .manifest_requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // 1. Manifest (single, small): resolve if not durable.
        if !self.store.is_durable_manifest(manifest).await? {
            let canonical = self
                .obtain_manifest(manifest, sources, hint, preflight)
                .await?;
            // Verify before install: id recompute + structural validation
            // + domain + length agreement (never trust peer-supplied ids).
            // Verification is on the *content*, not on where it came from, so
            // a reconstructed manifest faces exactly the same checks as a
            // fetched one and corruption can never reach the store.
            let decoded = kivi_chunk::verify_manifest(manifest, &canonical).map_err(|error| {
                self.store
                    .metrics()
                    .verification_failures
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                SidecarError::from(error)
            })?;
            if decoded.domain != self.domain {
                self.store
                    .metrics()
                    .verification_failures
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(SidecarError::Corrupt {
                    detail: format!("manifest {manifest} names a foreign domain"),
                });
            }
            if decoded.total_len != logical_len {
                self.store
                    .metrics()
                    .verification_failures
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(SidecarError::Corrupt {
                    detail: format!(
                        "manifest total {} != root logical {logical_len}",
                        decoded.total_len
                    ),
                });
            }
            self.store.stage_manifest(manifest, canonical).await?;
        }
        // 2. Discover missing chunks (dedup: only these move).
        let missing = self.store.missing_chunks(manifest).await?;
        if missing.is_empty() {
            self.store.sync().await?;
            // The gate proves durability; live-state references and the
            // content-addressed store own retention.
            if !self.store.check_root(manifest, logical_len).await? {
                return Err(SidecarError::Missing {
                    detail: format!("root {manifest} not durable after no-op fetch"),
                });
            }
            return Ok(());
        }
        // 3. Fetch missing chunks with bounded concurrency, verifying each.
        // Read the manifest once for per-chunk length expectations.
        let canonical = self.store.read_manifest(manifest).await?;
        let decoded = kivi_chunk::verify_manifest(manifest, &canonical)?;
        let expected: HashMap<ChunkId, u64> = decoded
            .entries
            .iter()
            .map(|entry| (entry.id, entry.len))
            .collect();
        for batch in missing.chunks(MAX_CONCURRENT_CHUNK_FETCH) {
            let mut futures_list = Vec::with_capacity(batch.len());
            for chunk in batch {
                futures_list
                    .push(self.fetch_and_stage_chunk(*chunk, &expected, sources, hint, preflight));
            }
            let results = futures::future::join_all(futures_list).await;
            for result in results {
                result?;
            }
        }
        // 4. One barrier for the whole batch, then prove durable.
        self.store.sync().await?;
        if !self.store.check_root(manifest, logical_len).await? {
            return Err(SidecarError::Missing {
                detail: format!("root {manifest} not durable after fetch"),
            });
        }
        Ok(())
    }

    async fn source_timeout(
        &self,
        source: NodeId,
        hint: Option<NodeId>,
        preflight: bool,
    ) -> Duration {
        if preflight && hint == Some(source) {
            let stats = self.transport.stats().await;
            if stats
                .get(&source)
                .is_some_and(|stats| stats.connected_bulk || stats.connected_control)
            {
                self.bulk_timeout
            } else {
                self.bulk_timeout.min(SOURCE_ATTEMPT_TIMEOUT)
            }
        } else {
            self.bulk_timeout
        }
    }

    async fn fetch_and_stage_chunk(
        &self,
        chunk: ChunkId,
        expected: &HashMap<ChunkId, u64>,
        sources: &[NodeId],
        hint: Option<NodeId>,
        preflight: bool,
    ) -> Result<(), SidecarError> {
        // Dedup re-check inside the batch (another entry may have staged).
        if self.store.is_durable_chunk(chunk).await? {
            self.store
                .metrics()
                .cache_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        self.store
            .metrics()
            .chunk_requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let expected_len = expected.get(&chunk).copied();
        let bytes = self
            .obtain_chunk(chunk, expected_len, sources, hint, preflight)
            .await?;
        // Verify: id recompute + length vs manifest entry. Source-agnostic, so
        // reconstructed bytes are held to exactly the same standard as fetched
        // ones. The redundancy coordinator already content-verified the image
        // it returned; this second verification is deliberately redundant,
        // because a reconstruction path is precisely where a subtle identity
        // mix-up would otherwise install a wrong-but-plausible value.
        let computed = kivi_codec::integrity::chunk_id(self.domain, &bytes);
        if computed != chunk {
            self.store
                .metrics()
                .verification_failures
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.store
                .metrics()
                .bulk_errors
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(SidecarError::Corrupt {
                detail: format!("chunk id mismatch: claimed {chunk}, computed {computed}"),
            });
        }
        if let Some(expected_len) = expected_len
            && bytes.len() as u64 != expected_len
        {
            self.store
                .metrics()
                .verification_failures
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(SidecarError::Corrupt {
                detail: format!("chunk {chunk} length mismatch"),
            });
        }
        self.store
            .metrics()
            .bulk_bytes_received
            .fetch_add(bytes.len() as u64, std::sync::atomic::Ordering::Relaxed);
        self.store.stage_chunk(chunk, bytes).await?;
        Ok(())
    }

    /// Awaits a resolver answer, waiting out [`ResolveError::Pending`] for a
    /// bounded number of rounds.
    ///
    /// Pending is the one verdict that time can fix: a holder staging, a peer
    /// restarting, a layout mid-publication. Anything else is a fixed fact, so
    /// it returns immediately - an unrecoverable answer must not consume the
    /// caller's budget, and a corrupt one must never earn a second attempt.
    ///
    /// The bound is small on purpose. This is a Raft append on the flush path:
    /// waiting briefly is far cheaper than failing the append (which `OpenRaft`
    /// treats as fatal), and waiting longer would be indistinguishable from
    /// blocking consensus on bulk I/O.
    ///
    /// Takes no gate state: the bound is a property of the seam, not of the
    /// node, and reading it from `self` would only suggest otherwise.
    async fn await_resolver(
        mut attempt: impl FnMut() -> ResolveFuture<Vec<u8>>,
    ) -> Result<Vec<u8>, ResolveError> {
        for round in 0..RESOLVER_PENDING_ROUNDS {
            match attempt().await {
                Err(error) if error.is_retryable() && round + 1 < RESOLVER_PENDING_ROUNDS => {
                    compio::time::sleep(PENDING_BACKOFF * u32::try_from(round + 1).unwrap_or(1))
                        .await;
                }
                other => return other,
            }
        }
        // The final round always returns; the loop exists to bound the retries.
        Err(ResolveError::Pending {
            detail: "resolver still pending after the wait bound".to_owned(),
        })
    }

    /// Combines a peer failure with the resolver's verdict into the one error
    /// the caller sees.
    ///
    /// Both verdicts are kept in the detail: "no peer had it" and "the fabric
    /// could not rebuild it" are different facts, and a report that collapses
    /// them is what made this failure undiagnosable in the first place. The
    /// *variant* comes from the resolver, because it is the last word on
    /// whether the content can be produced at all.
    fn resolution_failure(peer: &SidecarError, resolve: &ResolveError) -> SidecarError {
        let detail = format!("{peer}; resolver: {resolve}");
        if resolve.is_retryable() {
            // Not-yet is the crate's "come back later" signal, which is exactly
            // what keeps the inflight table from remembering this root as lost.
            SidecarError::Overloaded { detail }
        } else {
            SidecarError::Missing { detail }
        }
    }

    /// Resolves one manifest's canonical bytes: peers first, then the
    /// installed resolver.
    ///
    /// The peer mesh stays first because it is the cheapest complete source
    /// and the common case; the resolver is the answer to "nobody has the
    /// whole copy", which after a member is lost is a normal state rather
    /// than a failure. Both are verification-free producers - the caller
    /// verifies the content either way.
    async fn obtain_manifest(
        &self,
        manifest: ManifestId,
        sources: &[NodeId],
        hint: Option<NodeId>,
        preflight: bool,
    ) -> Result<Vec<u8>, SidecarError> {
        match self
            .fetch_manifest(manifest, sources, hint, preflight)
            .await
        {
            Ok(canonical) => Ok(canonical),
            Err(peer_error) => {
                let Some(resolver) = self.slots.resolver() else {
                    return Err(peer_error);
                };
                self.store
                    .metrics()
                    .resolutions
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let attempt = || resolver.resolve_manifest(manifest);
                match Self::await_resolver(attempt).await {
                    Ok(canonical) => {
                        self.store
                            .metrics()
                            .resolutions_recovered
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        tracing::info!(
                            %manifest,
                            "sidecar manifest recovered by the installed resolver"
                        );
                        Ok(canonical)
                    }
                    Err(resolve_error) => {
                        Err(Self::resolution_failure(&peer_error, &resolve_error))
                    }
                }
            }
        }
    }

    /// Resolves one chunk's bytes: peers first, then the installed resolver.
    ///
    /// `len` is the manifest's declared length, which the resolver needs to
    /// address the asset; `None` only when the manifest entry was absent, in
    /// which case the resolver cannot address it either and the peers remain
    /// the sole source.
    async fn obtain_chunk(
        &self,
        chunk: ChunkId,
        len: Option<u64>,
        sources: &[NodeId],
        hint: Option<NodeId>,
        preflight: bool,
    ) -> Result<Vec<u8>, SidecarError> {
        match self.fetch_chunk(chunk, sources, hint, preflight).await {
            Ok(bytes) => Ok(bytes),
            Err(peer_error) => {
                let Some(len) = len else {
                    return Err(peer_error);
                };
                let Some(resolver) = self.slots.resolver() else {
                    return Err(peer_error);
                };
                self.store
                    .metrics()
                    .resolutions
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let attempt = || resolver.resolve_chunk(chunk, len);
                match Self::await_resolver(attempt).await {
                    Ok(bytes) => {
                        self.store
                            .metrics()
                            .resolutions_recovered
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        tracing::info!(
                            %chunk,
                            "sidecar chunk recovered by the installed resolver"
                        );
                        Ok(bytes)
                    }
                    Err(resolve_error) => {
                        Err(Self::resolution_failure(&peer_error, &resolve_error))
                    }
                }
            }
        }
    }

    async fn fetch_manifest(
        &self,
        manifest: ManifestId,
        sources: &[NodeId],
        hint: Option<NodeId>,
        preflight: bool,
    ) -> Result<Vec<u8>, SidecarError> {
        let mut last_error = None;
        for source in sources {
            let request = PeerRequest::Manifest(PeerManifestRequest {
                manifest: *manifest.as_bytes(),
            });
            match self
                .transport
                .call_bulk(
                    *source,
                    self.group,
                    request,
                    self.source_timeout(*source, hint, preflight).await,
                )
                .await
            {
                Ok(PeerResponse::Manifest(response)) => {
                    if response.manifest != *manifest.as_bytes() {
                        self.store
                            .metrics()
                            .verification_failures
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        last_error = Some(SidecarError::Corrupt {
                            detail: "manifest response id mismatch".to_owned(),
                        });
                        continue;
                    }
                    self.store.metrics().bulk_bytes_received.fetch_add(
                        response.canonical.len() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    return Ok(response.canonical);
                }
                Ok(_) => {
                    last_error = Some(SidecarError::Corrupt {
                        detail: "peer answered manifest with wrong family".to_owned(),
                    });
                }
                Err(TransportError::RemoteRefused { detail }) => {
                    last_error = Some(SidecarError::Missing {
                        detail: format!("peer refused manifest: {detail}"),
                    });
                }
                Err(error) => {
                    self.store
                        .metrics()
                        .bulk_errors
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    last_error = Some(SidecarError::Missing {
                        detail: format!("manifest fetch: {error}"),
                    });
                }
            }
        }
        Err(last_error.unwrap_or(SidecarError::Missing {
            detail: format!("no source for manifest {manifest}"),
        }))
    }

    async fn fetch_chunk(
        &self,
        chunk: ChunkId,
        sources: &[NodeId],
        hint: Option<NodeId>,
        preflight: bool,
    ) -> Result<Vec<u8>, SidecarError> {
        let mut last_error = None;
        for source in sources {
            let request = PeerRequest::Chunk(PeerChunkRequest {
                chunk: *chunk.as_bytes(),
            });
            match self
                .transport
                .call_bulk(
                    *source,
                    self.group,
                    request,
                    self.source_timeout(*source, hint, preflight).await,
                )
                .await
            {
                Ok(PeerResponse::Chunk(response)) => {
                    if response.chunk != *chunk.as_bytes() {
                        self.store
                            .metrics()
                            .verification_failures
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        last_error = Some(SidecarError::Corrupt {
                            detail: "chunk response id mismatch".to_owned(),
                        });
                        continue;
                    }
                    return Ok(response.bytes);
                }
                Ok(_) => {
                    last_error = Some(SidecarError::Corrupt {
                        detail: "peer answered chunk with wrong family".to_owned(),
                    });
                }
                Err(TransportError::RemoteRefused { detail }) => {
                    last_error = Some(SidecarError::Missing {
                        detail: format!("peer refused chunk: {detail}"),
                    });
                }
                Err(error) => {
                    self.store
                        .metrics()
                        .bulk_errors
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    last_error = Some(SidecarError::Missing {
                        detail: format!("chunk fetch: {error}"),
                    });
                }
            }
        }
        Err(last_error.unwrap_or(SidecarError::Missing {
            detail: format!("no source for chunk {chunk}"),
        }))
    }
}

/// Maps a sidecar failure to the `io::Error` the Raft store must surface.
///
/// A corrupt/missing sidecar must fail the transfer, prevent the append
/// from becoming durable, and mark a transfer error - never install bad
/// bytes, ACK the append, or continue apply.
#[must_use]
pub fn sidecar_io_error(error: &SidecarError) -> std::io::Error {
    std::io::Error::other(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::future::Future;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use kivi_types::{ClusterId, NodeId, NodeIncarnation, SecurityDomainId};

    use super::{RESOLVER_PENDING_ROUNDS, SidecarGate};
    use crate::peer::{PeerIdentity, PeerRequest, PeerResponse, PeerRpcError};
    use crate::resolve::{ResolutionSlots, ResolveError};
    use crate::sidecar::SidecarStore;
    use crate::transport::{PeerHandler, PeerTransport, TlsMaterial, TransportConfig};
    use crate::types::ConsensusGroupId;

    const CLUSTER: u128 = 0x0006_0A7E;
    const GROUP_TABLET: u64 = 9;

    fn identity(node: u64) -> PeerIdentity {
        PeerIdentity {
            cluster: ClusterId::from_u128(CLUSTER),
            node: NodeId::from_u64(node),
            incarnation: NodeIncarnation::from_u64(1),
        }
    }

    /// Serves one staged value's sidecars, optionally corrupting chunk
    /// bytes to prove the fetcher rejects lies.
    struct Origin {
        manifests: HashMap<[u8; 32], Vec<u8>>,
        chunks: HashMap<[u8; 32], Vec<u8>>,
        corrupt_chunks: bool,
    }

    impl PeerHandler for Origin {
        fn handle(
            &self,
            _from: NodeId,
            _group: ConsensusGroupId,
            request: PeerRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<PeerResponse, PeerRpcError>> + Send + '_>,
        > {
            let manifests = self.manifests.clone();
            let chunks = self.chunks.clone();
            let corrupt = self.corrupt_chunks;
            Box::pin(async move {
                match request {
                    PeerRequest::Manifest(request) => manifests
                        .get(&request.manifest)
                        .map(|canonical| {
                            PeerResponse::Manifest(crate::peer::PeerManifestResponse {
                                manifest: request.manifest,
                                canonical: canonical.clone(),
                            })
                        })
                        .ok_or_else(|| PeerRpcError {
                            detail: "no such manifest".to_owned(),
                        }),
                    PeerRequest::Chunk(request) => chunks
                        .get(&request.chunk)
                        .map(|bytes| {
                            let mut bytes = bytes.clone();
                            if corrupt {
                                bytes[0] ^= 0xFF;
                            }
                            PeerResponse::Chunk(crate::peer::PeerChunkResponse {
                                chunk: request.chunk,
                                bytes,
                            })
                        })
                        .ok_or_else(|| PeerRpcError {
                            detail: "no such chunk".to_owned(),
                        }),
                    _ => Err(PeerRpcError {
                        detail: "origin serves sidecars only".to_owned(),
                    }),
                }
            })
        }
    }

    fn block_on<F, T>(future: F) -> T
    where
        F: Future<Output = T>,
    {
        compio::runtime::Runtime::new()
            .expect("compio test runtime")
            .block_on(future)
    }

    fn probe_udp() -> std::net::SocketAddr {
        std::net::UdpSocket::bind("127.0.0.1:0")
            .expect("probe binds")
            .local_addr()
            .expect("probe addr")
    }

    fn insecure_tls(dir: &std::path::Path, node: u64) -> TlsMaterial {
        TlsMaterial {
            cert: crate::tls::NodeCert::load_or_generate(dir, NodeId::from_u64(node))
                .expect("test cert generates"),
            peer_certs: HashMap::new(),
            trust: None,
            insecure_skip_verify: true,
        }
    }

    /// Fixture: an honest origin (1), an evil twin serving bit-flipped
    /// chunks (2), and a fetcher (3) with an empty sidecar store. The
    /// directory guard travels with the outputs so open files and certs
    /// stay valid for the whole test.
    struct Fixture {
        manifest: kivi_types::ManifestId,
        len: u64,
        chunks: Vec<([u8; 32], Vec<u8>)>,
        value: Vec<u8>,
        gate: SidecarGate,
        store: SidecarStore,
        _transports: Vec<PeerTransport>,
        _dir: tempfile::TempDir,
    }

    #[allow(clippy::too_many_lines)]
    async fn setup() -> Fixture {
        let domain = SecurityDomainId::from_u64(1);
        let group = ConsensusGroupId::of_tablet(kivi_types::TabletId::from_u64(GROUP_TABLET));
        let addrs = [probe_udp(), probe_udp(), probe_udp()];
        let peers: HashMap<NodeId, std::net::SocketAddr> = [1u64, 2, 3]
            .into_iter()
            .map(|id| {
                (
                    NodeId::from_u64(id),
                    addrs[usize::try_from(id).unwrap_or(1) - 1],
                )
            })
            .collect();
        let dir = tempfile::tempdir().expect("scratch");
        // Origin stages a 3-chunk value (3 MiB deterministic bytes).
        let (origin_store, _worker) =
            SidecarStore::open(&dir.path().join("origin"), domain, 64 * 1024 * 1024)
                .expect("origin store opens");
        let mut value = vec![0u8; 3 * 1024 * 1024];
        for (i, byte) in value.iter_mut().enumerate() {
            *byte = u8::try_from((i * 13 + 5) % 251).unwrap_or(0);
        }
        let staged = origin_store.stage_value(&value).await.expect("stages");
        let canonical = origin_store
            .read_manifest(staged.manifest)
            .await
            .expect("manifest readable");
        let mut chunks = Vec::new();
        for id in &staged.chunks {
            chunks.push((
                *id.as_bytes(),
                origin_store.read_chunk(*id).await.expect("chunk readable"),
            ));
        }
        let manifests: HashMap<[u8; 32], Vec<u8>> = [(*staged.manifest.as_bytes(), canonical)]
            .into_iter()
            .collect();
        let chunk_map: HashMap<[u8; 32], Vec<u8>> = chunks.clone().into_iter().collect();
        let evil_map = chunk_map.clone();
        let honest = Arc::new(Origin {
            manifests: manifests.clone(),
            chunks: chunk_map,
            corrupt_chunks: false,
        });
        let evil = Arc::new(Origin {
            manifests,
            chunks: evil_map,
            corrupt_chunks: true,
        });
        let fetcher_handler: Arc<dyn PeerHandler> = Arc::new(Origin {
            manifests: HashMap::new(),
            chunks: HashMap::new(),
            corrupt_chunks: false,
        });
        let (t1, _) = PeerTransport::open(
            TransportConfig::default(),
            identity(1),
            ClusterId::from_u128(CLUSTER),
            &peers,
            insecure_tls(&dir.path().join("t1"), 1),
            honest,
        )
        .await
        .expect("origin opens");
        let (t2, _) = PeerTransport::open(
            TransportConfig::default(),
            identity(2),
            ClusterId::from_u128(CLUSTER),
            &peers,
            insecure_tls(&dir.path().join("t2"), 2),
            evil,
        )
        .await
        .expect("evil opens");
        let (t3, _) = PeerTransport::open(
            TransportConfig::default(),
            identity(3),
            ClusterId::from_u128(CLUSTER),
            &peers,
            insecure_tls(&dir.path().join("t3"), 3),
            fetcher_handler,
        )
        .await
        .expect("fetcher opens");
        let (fetch_store, _fetch_worker) =
            SidecarStore::open(&dir.path().join("fetch"), domain, 64 * 1024 * 1024)
                .expect("fetch store opens");
        // The fetcher's gate may source from either origin. Its resolution
        // slots start empty: these tests exercise the peer path, and the
        // resolver seam has its own coverage below.
        let slots = Arc::new(crate::resolve::ResolutionSlots::new());
        let gate = SidecarGate::new(
            fetch_store.clone(),
            t3.clone(),
            vec![NodeId::from_u64(1), NodeId::from_u64(2)],
            group,
            domain,
            Duration::from_secs(15),
            Arc::clone(&slots),
        );
        Fixture {
            manifest: staged.manifest,
            len: staged.logical_len,
            chunks,
            value,
            gate,
            store: fetch_store,
            _transports: vec![t1, t2, t3],
            _dir: dir,
        }
    }

    #[test]
    fn gate_fetches_missing_sidecars_from_honest_source() {
        block_on(async {
            let fixture = setup().await;
            fixture
                .gate
                .ensure_root(fixture.manifest, fixture.len, Some(NodeId::from_u64(1)))
                .await
                .expect("honest fetch completes");
            assert_eq!(
                fixture
                    .store
                    .read_value(fixture.manifest, fixture.len)
                    .await
                    .expect("reads"),
                fixture.value
            );
        });
    }

    #[test]
    fn gate_rejects_corrupt_sidecars_and_installs_nothing() {
        block_on(async {
            let fixture = setup().await;
            let manifest = fixture.manifest;
            let len = fixture.len;
            let before = fixture
                .store
                .metrics()
                .verification_failures
                .load(std::sync::atomic::Ordering::Relaxed);
            let error = fixture
                .gate
                .ensure_root(manifest, len, Some(NodeId::from_u64(2)))
                .await
                .expect_err("corrupt fetch must fail");
            assert!(
                matches!(error, crate::sidecar::SidecarError::Corrupt { .. }),
                "corruption fails as Corrupt, got {error:?}"
            );
            // Nothing installed: the claimed chunk is absent and the root
            // is not durable, so no Raft append could count it persisted.
            for (id, _) in &fixture.chunks {
                assert!(
                    !fixture
                        .store
                        .is_durable_chunk(kivi_types::ChunkId::from_bytes(*id))
                        .await
                        .expect("checks"),
                    "corrupt chunk never becomes durable"
                );
            }
            assert!(
                !fixture
                    .store
                    .check_root(manifest, len)
                    .await
                    .expect("checks"),
                "root never becomes durable"
            );
            let after = fixture
                .store
                .metrics()
                .verification_failures
                .load(std::sync::atomic::Ordering::Relaxed);
            assert!(after > before, "verification failure is metered");
            // The error maps to `io::Error` for the Raft store, which fails
            // the append instead of acknowledging it.
            let _ = super::sidecar_io_error(&error);
        });
    }

    /// A resolver under the test's control: returns scripted bytes or a
    /// scripted failure, and counts how many times it was consulted.
    struct ScriptedResolver {
        manifest: Mutex<HashMap<kivi_types::ManifestId, Result<Vec<u8>, ResolveError>>>,
        chunks: Mutex<HashMap<kivi_types::ChunkId, Result<Vec<u8>, ResolveError>>>,
        /// Consultations per id, so a test can prove work was shared.
        calls: Mutex<HashMap<[u8; 32], u32>>,
        /// Consultation delay, used to make a coalescing test deterministic.
        delay: Duration,
    }

    impl ScriptedResolver {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                manifest: Mutex::new(HashMap::new()),
                chunks: Mutex::new(HashMap::new()),
                calls: Mutex::new(HashMap::new()),
                delay: Duration::ZERO,
            })
        }

        fn delayed(mut self: Arc<Self>, delay: Duration) -> Arc<Self> {
            Arc::get_mut(&mut self).expect("unshared").delay = delay;
            self
        }

        /// How many times one manifest was consulted.
        fn manifest_calls(&self, manifest: kivi_types::ManifestId) -> u32 {
            self.calls
                .lock()
                .expect("locked")
                .get(manifest.as_bytes())
                .copied()
                .unwrap_or(0)
        }

        fn count(&self, id: [u8; 32]) {
            *self.calls.lock().expect("locked").entry(id).or_default() += 1;
        }
    }

    impl std::fmt::Debug for ScriptedResolver {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter
                .debug_struct("ScriptedResolver")
                .field("delay", &self.delay)
                .finish_non_exhaustive()
        }
    }

    impl crate::resolve::SidecarResolver for ScriptedResolver {
        fn resolve_manifest(
            &self,
            manifest: kivi_types::ManifestId,
        ) -> crate::resolve::ResolveFuture<Vec<u8>> {
            self.count(*manifest.as_bytes());
            let answer = self
                .manifest
                .lock()
                .expect("locked")
                .get(&manifest)
                .cloned();
            let delay = self.delay;
            Box::pin(async move {
                if !delay.is_zero() {
                    compio::time::sleep(delay).await;
                }
                answer.unwrap_or(Err(ResolveError::Unavailable {
                    detail: "not scripted".to_owned(),
                }))
            })
        }

        fn resolve_chunk(
            &self,
            chunk: kivi_types::ChunkId,
            _len: u64,
        ) -> crate::resolve::ResolveFuture<Vec<u8>> {
            self.count(*chunk.as_bytes());
            let answer = self.chunks.lock().expect("locked").get(&chunk).cloned();
            let delay = self.delay;
            Box::pin(async move {
                if !delay.is_zero() {
                    compio::time::sleep(delay).await;
                }
                answer.unwrap_or(Err(ResolveError::Unavailable {
                    detail: "not scripted".to_owned(),
                }))
            })
        }
    }

    /// A gate whose peers hold nothing, so only the resolver can satisfy a
    /// request. This is the shape of a node that lost the only whole copy.
    async fn gate_without_sources() -> (
        SidecarStore,
        SidecarGate,
        Arc<ResolutionSlots>,
        tempfile::TempDir,
    ) {
        let domain = SecurityDomainId::from_u64(1);
        let group = ConsensusGroupId::of_tablet(kivi_types::TabletId::from_u64(GROUP_TABLET));
        let dir = tempfile::tempdir().expect("scratch");
        let (store, _worker) =
            SidecarStore::open(&dir.path().join("fetch"), domain, 8 * 1024 * 1024)
                .expect("store opens");
        // A transport over an empty peer map: every source attempt is
        // impossible, so a success can only have come from the resolver.
        let peers: HashMap<NodeId, std::net::SocketAddr> = HashMap::new();
        let (transport, _) = PeerTransport::open(
            TransportConfig::default(),
            identity(1),
            ClusterId::from_u128(CLUSTER),
            &peers,
            insecure_tls(&dir.path().join("t"), 1),
            Arc::new(Origin {
                manifests: HashMap::new(),
                chunks: HashMap::new(),
                corrupt_chunks: false,
            }),
        )
        .await
        .expect("transport opens");
        let slots = Arc::new(ResolutionSlots::new());
        let gate = SidecarGate::new(
            store.clone(),
            transport,
            Vec::new(),
            group,
            domain,
            Duration::from_secs(2),
            Arc::clone(&slots),
        );
        (store, gate, slots, dir)
    }

    /// One chunked value, described the way a resolver would have to produce
    /// it: the manifest to answer for, the chunks it names, and the bytes of
    /// each - plus the value itself, so a test can assert on the
    /// reconstruction without re-deriving it.
    ///
    /// A named struct rather than a tuple because the first draft returned a
    /// five-slot tuple and a test compared the reconstructed value against the
    /// canonical manifest. Indexing a tuple of same-typed fields is a mistake
    /// waiting to happen; a field name is not.
    struct Described {
        manifest: kivi_types::ManifestId,
        len: u64,
        chunks: Vec<kivi_types::ChunkId>,
        /// Chunk bytes, positionally matching `chunks`.
        pieces: Vec<Vec<u8>>,
        /// Canonical manifest bytes, for a resolver that answers with them.
        canonical: Vec<u8>,
        value: Vec<u8>,
    }

    async fn a_staged_value() -> Described {
        let dir = tempfile::tempdir().expect("scratch");
        let (store, _worker) =
            SidecarStore::open(dir.path(), SecurityDomainId::from_u64(1), 8 * 1024 * 1024)
                .expect("store opens");
        let value = (0..6_000u32)
            .map(|i| u8::try_from((i * 31 + 7) % 251).unwrap_or(0))
            .collect::<Vec<u8>>();
        let staged = store.stage_value(&value).await.expect("stages");
        let canonical = store
            .read_manifest(staged.manifest)
            .await
            .expect("manifest readable");
        let mut pieces = Vec::new();
        for id in &staged.chunks {
            pieces.push(store.read_chunk(*id).await.expect("chunk readable"));
        }
        Described {
            manifest: staged.manifest,
            len: staged.logical_len,
            chunks: staged.chunks,
            pieces,
            canonical,
            value,
        }
    }

    #[test]
    fn a_missing_whole_copy_is_resolved_from_the_fabric() {
        block_on(async {
            let described = a_staged_value().await;
            let (manifest, len) = (described.manifest, described.len);
            let (store, gate, slots, _dir) = gate_without_sources().await;
            // No peer can serve anything; only the resolver can.
            let resolver = ScriptedResolver::new();
            resolver
                .manifest
                .lock()
                .expect("locked")
                .insert(manifest, Ok(described.canonical.clone()));
            for (id, bytes) in described.chunks.iter().zip(&described.pieces) {
                resolver
                    .chunks
                    .lock()
                    .expect("locked")
                    .insert(*id, Ok(bytes.clone()));
            }
            slots.install_resolver(resolver.clone());

            gate.ensure_root(manifest, len, None)
                .await
                .expect("a lost whole copy is a reconstruction, not a failure");

            assert_eq!(
                store.read_value(manifest, len).await.expect("reads"),
                described.value,
                "the root is readable, and byte-exact, after reconstruction"
            );
            assert_eq!(
                resolver.manifest_calls(manifest),
                1,
                "one consultation per missing manifest"
            );
        });
    }

    #[test]
    fn a_reconstructed_root_installs_nothing_when_the_content_is_wrong() {
        block_on(async {
            let described = a_staged_value().await;
            let (manifest, len) = (described.manifest, described.len);
            let (store, gate, slots, _dir) = gate_without_sources().await;
            // A resolver that answers with a *valid* manifest for a different
            // value: right shape, self-consistent, wrong identity. This is what
            // a mis-addressed fragment set or a hostile peer looks like, and it
            // is harder to catch than a malformed one - the gate's own id
            // recompute is the only thing standing between it and a value
            // served under the wrong key.
            let other = (0..usize::try_from(len).expect("fits"))
                .map(|i| u8::try_from((i * 17 + 3) % 241).unwrap_or(0))
                .collect::<Vec<u8>>();
            let other_id = kivi_codec::integrity::chunk_id(SecurityDomainId::from_u64(1), &other);
            let (lying, lying_id) = kivi_chunk::build_manifest(
                SecurityDomainId::from_u64(1),
                kivi_chunk::Chunking::DEFAULT,
                kivi_chunk::ChunkCodecId::NONE,
                len,
                vec![kivi_chunk::ChunkEntry { id: other_id, len }],
            )
            .expect("builds");
            assert_ne!(lying_id, manifest, "the decoy is a different identity");
            let mut lying_canonical = Vec::new();
            lying.encode_canonical(&mut lying_canonical);
            let resolver = ScriptedResolver::new();
            resolver
                .manifest
                .lock()
                .expect("locked")
                .insert(manifest, Ok(lying_canonical));
            slots.install_resolver(resolver.clone());

            let error = gate
                .ensure_root(manifest, len, None)
                .await
                .expect_err("a lying reconstruction must fail");
            // The manifest does not hash to the id that was asked for, so it is
            // rejected as corruption - not installed, and not retried.
            assert!(
                matches!(error, crate::sidecar::SidecarError::Corrupt { .. }),
                "corruption fails as Corrupt, got {error:?}"
            );
            assert!(
                !store.is_durable_manifest(manifest).await.expect("checks"),
                "a rejected manifest is never installed, so no later read can \
                 mistake it for the real one"
            );
            assert_eq!(
                resolver.manifest_calls(manifest),
                1,
                "corruption is never retried: a second attempt is more likely to \
                 install wrong content than right"
            );
        });
    }

    #[test]
    fn an_unrecoverable_root_says_so_and_waits_for_nothing() {
        block_on(async {
            let described = a_staged_value().await;
            let (manifest, len) = (described.manifest, described.len);
            let (store, gate, slots, _dir) = gate_without_sources().await;
            let resolver = ScriptedResolver::new();
            resolver.manifest.lock().expect("locked").insert(
                manifest,
                Err(ResolveError::Unavailable {
                    detail: "not enough surviving fragments".to_owned(),
                }),
            );
            slots.install_resolver(resolver.clone());

            let error = gate
                .ensure_root(manifest, len, None)
                .await
                .expect_err("an unrecoverable root cannot be satisfied");
            // The diagnostic keeps both verdicts: the peers could not serve it
            // and the fabric could not rebuild it. Collapsing these into
            // "unavailable" is what made the old failure undiagnosable.
            let detail = error.to_string();
            assert!(
                detail.contains("not enough surviving fragments"),
                "the resolver's reason survives, got {detail}"
            );
            assert!(
                !store.is_durable_manifest(manifest).await.expect("checks"),
                "nothing is installed for a root that could not be produced"
            );
            assert_eq!(
                resolver.manifest_calls(manifest),
                1,
                "an unrecoverable answer is a fixed fact: it is not retried"
            );
        });
    }

    #[test]
    fn a_pending_root_is_waited_out_before_it_is_failed() {
        block_on(async {
            let described = a_staged_value().await;
            let (manifest, len) = (described.manifest, described.len);
            let (_store, gate, slots, _dir) = gate_without_sources().await;
            // Pending twice, then answered. A restarting member looks like
            // this, and failing on the first answer would kill the append.
            let resolver = ScriptedResolver::new();
            resolver.manifest.lock().expect("locked").insert(
                manifest,
                Err(ResolveError::Pending {
                    detail: "holder is restarting".to_owned(),
                }),
            );
            slots.install_resolver(resolver.clone());

            // With a resolver that only ever says pending, the gate must give
            // up within its bound rather than wait forever - and the error it
            // gives up with must be the *retryable* one. Reporting a
            // not-yet answer as a missing sidecar is what would turn a
            // restarting member into a permanently lost root.
            let error = gate
                .ensure_root(manifest, len, None)
                .await
                .expect_err("an endlessly pending root is bounded, not awaited");
            assert!(
                error.is_retryable(),
                "an out-of-time pending answer is a 'come back' signal, got {error:?}"
            );
            assert_eq!(
                resolver.manifest_calls(manifest),
                u32::try_from(RESOLVER_PENDING_ROUNDS).expect("rounds fit"),
                "pending is retried exactly the configured number of rounds: \
                 enough to ride out a restart, not enough to block the append"
            );
        });
    }

    #[test]
    fn concurrent_waiters_share_one_reconstruction() {
        block_on(async {
            let described = a_staged_value().await;
            let (manifest, len) = (described.manifest, described.len);
            let (store, gate, slots, _dir) = gate_without_sources().await;
            // Slow enough that every waiter is certainly in flight before the
            // first resolution lands, so the coalescing is genuinely exercised
            // rather than accidentally serialized.
            let resolver = ScriptedResolver::new()
                .delayed(Duration::from_millis(120))
                .clone();
            resolver
                .manifest
                .lock()
                .expect("locked")
                .insert(manifest, Ok(described.canonical.clone()));
            for (id, bytes) in described.chunks.iter().zip(&described.pieces) {
                resolver
                    .chunks
                    .lock()
                    .expect("locked")
                    .insert(*id, Ok(bytes.clone()));
            }
            slots.install_resolver(resolver.clone());

            let waiters: Vec<_> = (0..4)
                .map(|_| {
                    let gate = gate.clone();
                    async move { gate.ensure_root(manifest, len, None).await }
                })
                .collect();
            let joined = futures::future::join_all(waiters).await;
            for result in joined {
                result.expect("every waiter is satisfied");
            }
            assert_eq!(
                store.read_value(manifest, len).await.expect("reads"),
                described.value,
                "the root is durable once, and every waiter sees it"
            );
            assert_eq!(
                resolver.manifest_calls(manifest),
                1,
                "four waiters, one reconstruction: the inflight entry is the \
                 single point of coordination"
            );
        });
    }

    #[test]
    fn a_transient_resolution_failure_is_not_remembered_as_a_loss() {
        block_on(async {
            let described = a_staged_value().await;
            let (manifest, len) = (described.manifest, described.len);
            let (store, gate, slots, _dir) = gate_without_sources().await;
            // A resolver that is not ready yet, then is. This is a holder
            // restarting, seen from the gate.
            let answer: Mutex<Option<Result<Vec<u8>, ResolveError>>> =
                Mutex::new(Some(Err(ResolveError::Pending {
                    detail: "holder is restarting".to_owned(),
                })));
            let ready: Mutex<Option<ResolvableContent>> = Mutex::new(None);
            let resolver = Arc::new(AlwaysPendingThenReady {
                answer,
                ready,
                count: Mutex::new(0),
            });
            slots.install_resolver(resolver.clone());

            let first = gate
                .ensure_root(manifest, len, None)
                .await
                .expect_err("the first attempt has nothing to resolve from");
            // Overload, not Missing: the crate's existing "come back" signal.
            // A `Missing` here is what a remembered transient failure would
            // eventually become.
            assert!(
                first.is_retryable(),
                "a not-yet verdict must not read as a permanent loss, got {first:?}"
            );

            // The condition clears.
            *resolver.ready.lock().expect("locked") = Some(ResolvableContent {
                canonical: described.canonical.clone(),
                chunks: described
                    .chunks
                    .iter()
                    .copied()
                    .zip(described.pieces.iter().cloned())
                    .collect(),
            });

            gate.ensure_root(manifest, len, None)
                .await
                .expect("a cleared condition must not stay poisoned by the table");
            assert_eq!(
                store.read_value(manifest, len).await.expect("reads"),
                described.value,
                "the root is readable once the resolver can answer"
            );
        });
    }

    /// One root's content, as a resolver would have to hand it back: the
    /// canonical manifest bytes, and the chunk bytes keyed by the id the
    /// manifest names.
    #[derive(Clone)]
    struct ResolvableContent {
        canonical: Vec<u8>,
        chunks: Vec<(kivi_types::ChunkId, Vec<u8>)>,
    }

    /// Answers "pending" until `ready` is populated, then serves the real
    /// content. Both switches live in the resolver so a single install covers
    /// the whole sequence.
    struct AlwaysPendingThenReady {
        answer: Mutex<Option<Result<Vec<u8>, ResolveError>>>,
        ready: Mutex<Option<ResolvableContent>>,
        count: Mutex<u32>,
    }

    impl std::fmt::Debug for AlwaysPendingThenReady {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter
                .debug_struct("AlwaysPendingThenReady")
                .field("count", &self.count)
                .finish_non_exhaustive()
        }
    }

    impl crate::resolve::SidecarResolver for AlwaysPendingThenReady {
        fn resolve_manifest(
            &self,
            _manifest: kivi_types::ManifestId,
        ) -> crate::resolve::ResolveFuture<Vec<u8>> {
            *self.count.lock().expect("locked") += 1;
            if let Some(content) = self.ready.lock().expect("locked").clone() {
                return Box::pin(async move { Ok(content.canonical) });
            }
            let answer = self.answer.lock().expect("locked").clone();
            Box::pin(async move {
                answer.unwrap_or(Err(ResolveError::Pending {
                    detail: "not ready".to_owned(),
                }))
            })
        }

        fn resolve_chunk(
            &self,
            chunk: kivi_types::ChunkId,
            _len: u64,
        ) -> crate::resolve::ResolveFuture<Vec<u8>> {
            if let Some(content) = self.ready.lock().expect("locked").clone() {
                let answer = content
                    .chunks
                    .into_iter()
                    .find(|(id, _)| *id == chunk)
                    .map(|(_, bytes)| bytes);
                return Box::pin(async move {
                    answer.ok_or(ResolveError::Unavailable {
                        detail: "no scripted chunk".to_owned(),
                    })
                });
            }
            Box::pin(async move {
                Err(ResolveError::Pending {
                    detail: "not ready".to_owned(),
                })
            })
        }
    }

    #[test]
    fn no_resolver_installed_leaves_the_peer_path_exactly_as_it_was() {
        block_on(async {
            let described = a_staged_value().await;
            let (manifest, len) = (described.manifest, described.len);
            let (_store, gate, _slots, _dir) = gate_without_sources().await;
            // The pre-existing behaviour must survive: with no redundancy plane
            // there is nothing to resolve from, and a missing source is still
            // a missing source.
            let error = gate
                .ensure_root(manifest, len, None)
                .await
                .expect_err("no source at all is still a failure");
            assert!(matches!(
                error,
                crate::sidecar::SidecarError::Missing { .. }
            ));
        });
    }
}
