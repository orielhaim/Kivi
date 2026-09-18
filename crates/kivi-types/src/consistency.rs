//! First-class commit/freshness metadata for the consistency layer.
//!
//! The consistency layer reasons about exactly four facts, and nothing else:
//!
//! ```text
//! what this replica has applied ........... ReplicaFreshness::applied
//! what authority that information belongs to  ReplicaFreshness::authority
//! what commit/freshness evidence is known ... FreshnessReceipt / RosterEvidence
//! whether that evidence survives change .... authority-equality checks
//! ```
//!
//! Wall-clock age is never equated with replication freshness: a locally old
//! timestamp proves nothing about what the leader has committed. A replica may
//! claim a staleness bound only when it holds [`FreshnessReceipt`] evidence —
//! a recent authority proof (consensus barrier) — whose age on the caller's
//! monotonic clock ([`Ticks`], always passed in, never read) fits the bound,
//! whose authority still matches the current authority exactly, and whose
//! coverage boundary the replica has applied. Without such evidence the
//! replica escalates to the authority path or reports the bound unprovable;
//! it never serves out-of-bound data as success.
//!
//! Strong (`Latest`) reads travel one of three mechanisms with identical
//! external semantics and different assumptions:
//!
//! ```text
//! ConservativeLeader ... fresh Raft barrier per read. No timing assumption.
//! AlmostLocal .......... Lazy-ALR: opportunistic batch + ordered ReadSync
//!                        (no state mutation). No timing assumption.
//! RosterLease .......... Bodega-style directional roster leases. Assumes
//!                        bounded clock-rate drift (explicit deployment
//!                        contract); unavailable without it.
//! ```
//!
//! Every mechanism falls back toward the conservative baseline on any
//! uncertainty; a fallback never weakens the `Latest` contract.
//!
//! All types here are pure values: no I/O, no clocks, no randomness. The
//! execution layer (`kivi-consensus::consistency`) owns the hub that stores
//! them; this crate owns their meaning.

use core::fmt;
use core::time::Duration;

use crate::ids::{CommitPosition, NodeId, NodeIncarnation, TabletEpoch, TabletId};
use crate::position::CommitToken;
use crate::time::Ticks;
use crate::{TabletAuthority, WriteGuardGeneration};

/// Maximum entries in one tablet's strong cache. The cache is an
/// optimization: eviction only costs a fallback, never correctness.
pub const STRONG_CACHE_CAP: usize = 512;

/// Maximum value bytes cached per entry. Larger values bypass the cache and
/// serve from applied state directly, so one huge value cannot evict the
/// working set.
pub const STRONG_CACHE_VALUE_CAP: u64 = 64 * 1024;

/// Maximum wait for applied coverage on one `AtLeast` read before the read
/// fails instead of holding its caller. Waiting runs on the owner's bounded
/// sleeps; this caps the whole wait, not one sleep.
pub const AT_LEAST_WAIT_CAP: Duration = Duration::from_secs(10);

/// Snapshot of what one replica has applied, under what authority, in what
/// incarnation. The unit the whole layer reasons about: coverage checks,
/// receipt validation, and lease checks all start here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReplicaFreshness {
    /// Authority this applied state belongs to.
    pub authority: TabletAuthority,
    /// Last applied commit position (`UNASSIGNED` before the first apply).
    pub applied: CommitPosition,
    /// Incarnation of the process holding this state.
    pub incarnation: NodeIncarnation,
}

impl ReplicaFreshness {
    /// Builds a freshness snapshot from its three facts.
    #[must_use]
    pub const fn new(
        authority: TabletAuthority,
        applied: CommitPosition,
        incarnation: NodeIncarnation,
    ) -> Self {
        Self {
            authority,
            applied,
            incarnation,
        }
    }

    /// Whether this replica's applied state covers `token`: same tablet,
    /// same epoch, assigned position at or below applied.
    ///
    /// A token from another tablet, another epoch (split/merge lineage,
    /// never silently comparable), or an unassigned position never
    /// validates — the caller must reject it, never wait on it.
    ///
    /// # Errors
    ///
    /// Returns freshness rejection for foreign tablets/epochs, unassigned
    /// tokens, or positions beyond applied (the only wait-curable case).
    pub const fn covers_token(&self, token: CommitToken) -> Result<(), FreshnessReject> {
        if token.tablet().as_u64() != self.authority.tablet().as_u64() {
            return Err(FreshnessReject::WrongTablet {
                presented: token.tablet(),
                current: self.authority.tablet(),
            });
        }
        if token.epoch().as_u64() != self.authority.epoch().as_u64() {
            return Err(FreshnessReject::StaleEpoch {
                presented: token.epoch(),
                current: self.authority.epoch(),
            });
        }
        if !token.is_assigned() {
            return Err(FreshnessReject::UnassignedToken);
        }
        if token.position().as_u64() > self.applied.as_u64() {
            return Err(FreshnessReject::BeyondApplied {
                requested: token.position(),
                applied: self.applied,
            });
        }
        Ok(())
    }
}

impl fmt::Display for ReplicaFreshness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "freshness({} applied {} incarnation {})",
            self.authority, self.applied, self.incarnation
        )
    }
}

/// Why a freshness claim was rejected. [`BeyondApplied`](Self::BeyondApplied)
/// is the only variant that waiting may cure; every other variant is final
/// for this authority and must fail the read (or escalate it), never block on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum FreshnessReject {
    /// The token names a different tablet (misrouted read or cross-tablet
    /// token reuse).
    #[error("freshness for wrong tablet: presented {presented}, current is {current}")]
    WrongTablet {
        /// Tablet the token names.
        presented: TabletId,
        /// Tablet this replica serves.
        current: TabletId,
    },
    /// The token names a superseded epoch: its lineage no longer exists
    /// here (split/merge/retirement minted fresh tablet identities; an old
    /// epoch value can never accidentally validate against a new lineage).
    #[error("stale token epoch: presented {presented}, current is {current}")]
    StaleEpoch {
        /// Epoch the token names.
        presented: TabletEpoch,
        /// Current epoch.
        current: TabletEpoch,
    },
    /// The token names no log slot (`UNASSIGNED` position or invalid
    /// epoch): it vouches for nothing.
    #[error("commit token names no log slot")]
    UnassignedToken,
    /// The token is valid for this lineage but names a position this
    /// replica has not applied yet. Waiting (bounded) may cure it.
    #[error("replica behind token: requested {requested}, applied {applied}")]
    BeyondApplied {
        /// Requested commit position.
        requested: CommitPosition,
        /// Locally applied position.
        applied: CommitPosition,
    },
    /// Evidence authority no longer matches current authority (epoch or
    /// guard moved: topology change, ownership move, directory replacement).
    /// Cached freshness dies with the authority that produced it.
    #[error("freshness evidence from a superseded authority")]
    StaleAuthority,
    /// The evidence boundary exceeds local applied state (follower lag).
    #[error("evidence boundary {boundary} beyond applied {applied}")]
    EvidenceBeyondApplied {
        /// Coverage the evidence vouches for.
        boundary: CommitPosition,
        /// Locally applied position.
        applied: CommitPosition,
    },
    /// No authority evidence is held at all (cold replica, restart, or
    /// first read since invalidation).
    #[error("no freshness evidence held")]
    NoProof,
    /// Evidence exists but its age exceeds the allowed bound.
    #[error("freshness evidence too old: age {age_ms}ms exceeds bound {bound_ms}ms")]
    ProofTooOld {
        /// Evidence age in milliseconds.
        age_ms: u128,
        /// Allowed bound in milliseconds.
        bound_ms: u128,
    },
    /// A roster lease cannot authorize this read (not a member, wrong
    /// incarnation, wrong authority, or past expiry). The holder must stop
    /// serving that fast path immediately and fall back.
    #[error("roster lease invalid: {detail}")]
    InvalidLease {
        /// Which check failed (member, incarnation, authority, expiry).
        detail: LeaseInvalid,
    },
}

/// Which roster-lease check failed. Reported inside
/// [`FreshnessReject::InvalidLease`] and in metrics so operators can tell a
/// clock problem (expiry) from a topology problem (authority) at a glance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LeaseInvalid {
    /// Holder is not a member of the lease roster.
    NotMember,
    /// Holder incarnation differs from the lease's bound incarnation
    /// (restart minted a new incarnation; the old lease died with the old
    /// process).
    WrongIncarnation,
    /// Lease authority differs from current authority (leadership or
    /// topology moved on; the lease is void even if unexpired).
    WrongAuthority,
    /// `now` reached or passed `expires_at`. Expiry is strict: equality
    /// fails closed.
    Expired,
}

impl fmt::Display for LeaseInvalid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotMember => write!(f, "not a roster member"),
            Self::WrongIncarnation => write!(f, "incarnation changed"),
            Self::WrongAuthority => write!(f, "authority changed"),
            Self::Expired => write!(f, "lease expired"),
        }
    }
}

/// Authority evidence: at monotonic time `proven_at`, this replica (or the
/// lease issuer it trusts) proved that applied state covered `boundary`
/// under `authority`.
///
/// Produced by a consensus read barrier (quorum proved leadership, local
/// state applied through the boundary) or by a roster-lease grant. Consumed
/// by bounded-stale reads, the Almost-Local fast path, and the strong cache.
/// A receipt vouches for coverage, never for the absence of newer commits:
/// serving `Latest` from a receipt additionally requires the synchrony
/// assumption of the provider holding it (documented on
/// [`ReadAuthorityProvider`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FreshnessReceipt {
    /// Authority that produced the proof.
    pub authority: TabletAuthority,
    /// Coverage the proof vouches for (inclusive applied position).
    pub boundary: CommitPosition,
    /// Monotonic time the proof completed (caller clock, [`Ticks`]).
    pub proven_at: Ticks,
    /// Incarnation that proved it (restart invalidates memory-held proof).
    pub incarnation: NodeIncarnation,
}

impl FreshnessReceipt {
    /// Builds a receipt from its four facts.
    #[must_use]
    pub const fn new(
        authority: TabletAuthority,
        boundary: CommitPosition,
        proven_at: Ticks,
        incarnation: NodeIncarnation,
    ) -> Self {
        Self {
            authority,
            boundary,
            proven_at,
            incarnation,
        }
    }

    /// Whether this receipt is still usable against `current` authority and
    /// `applied` state: same authority triple (any epoch/guard movement
    /// voids it), same incarnation, and the boundary still applied.
    ///
    /// # Errors
    ///
    /// Returns freshness rejection on any authority or incarnation
    /// movement, or when the boundary exceeds applied state.
    pub const fn usable_against(
        &self,
        current: &TabletAuthority,
        applied: CommitPosition,
        incarnation: NodeIncarnation,
    ) -> Result<(), FreshnessReject> {
        if self.authority.tablet().as_u64() != current.tablet().as_u64()
            || self.authority.epoch().as_u64() != current.epoch().as_u64()
            || self.authority.guard().as_u64() != current.guard().as_u64()
        {
            return Err(FreshnessReject::StaleAuthority);
        }
        if self.incarnation.as_u64() != incarnation.as_u64() {
            return Err(FreshnessReject::StaleAuthority);
        }
        if self.boundary.as_u64() > applied.as_u64() {
            return Err(FreshnessReject::EvidenceBeyondApplied {
                boundary: self.boundary,
                applied,
            });
        }
        Ok(())
    }

    /// Age of the proof at `now` on the caller's monotonic clock.
    #[must_use]
    pub fn age(&self, now: Ticks) -> Duration {
        now.saturating_since(self.proven_at)
    }

    /// Whether this receipt satisfies a `BoundedStale(max_staleness)` claim
    /// at `now`: usable authority/coverage plus age within the bound.
    ///
    /// # Errors
    ///
    /// Returns freshness rejection when the receipt is unusable or its age
    /// exceeds the bound.
    pub fn satisfies_bound(
        &self,
        max_staleness: Duration,
        now: Ticks,
        current: &TabletAuthority,
        applied: CommitPosition,
        incarnation: NodeIncarnation,
    ) -> Result<(), FreshnessReject> {
        self.usable_against(current, applied, incarnation)?;
        let age = self.age(now);
        if age > max_staleness {
            return Err(FreshnessReject::ProofTooOld {
                age_ms: age.as_millis(),
                bound_ms: max_staleness.as_millis(),
            });
        }
        Ok(())
    }
}

impl fmt::Display for FreshnessReceipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "receipt({} boundary {} proven_at {})",
            self.authority, self.boundary, self.proven_at
        )
    }
}

/// Wire-visible proof attached to every served read. Lets the client chain
/// `AtLeast` reads (`read → receipt.token() → AtLeast(token)`) and lets
/// operators answer "what generation/evidence made this valid" without
/// reverse-engineering logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReadReceipt {
    /// Authority the read served from.
    pub authority: TabletAuthority,
    /// Applied position the read served from (inclusive).
    pub position: CommitPosition,
    /// Incarnation that served it.
    pub incarnation: NodeIncarnation,
}

impl ReadReceipt {
    /// Builds a receipt from its three facts.
    #[must_use]
    pub const fn new(
        authority: TabletAuthority,
        position: CommitPosition,
        incarnation: NodeIncarnation,
    ) -> Self {
        Self {
            authority,
            position,
            incarnation,
        }
    }

    /// The commit token this read vouches for: feeding it back as
    /// `AtLeast(token)` reproduces at least this state on any replica of
    /// the same lineage. Unassigned before the first apply (position
    /// `UNASSIGNED`): such receipts chain to nothing and `AtLeast` over
    /// them rejects as [`FreshnessReject::UnassignedToken`].
    #[must_use]
    pub const fn token(self) -> CommitToken {
        CommitToken::new(
            self.authority.tablet(),
            self.authority.epoch(),
            self.position,
        )
    }
}

impl fmt::Display for ReadReceipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "read-receipt({} position {} incarnation {})",
            self.authority, self.position, self.incarnation
        )
    }
}

/// Per-read execution context: the two clocks plus the wait budget. Both
/// clocks are supplied by the caller — production passes wall time for
/// expiry and monotonic time for ages; deterministic simulation passes
/// virtual time for both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReadContext {
    /// Logical time for expiry evaluation (Unix micros).
    pub now: crate::time::UnixMicros,
    /// Monotonic time for proof/lease ages (simulation-virtualizable).
    pub ticks: Ticks,
    /// Maximum to wait for applied coverage on `AtLeast` reads. Capped by
    /// [`AT_LEAST_WAIT_CAP`]; waiting is always bounded.
    pub wait: Duration,
}

impl ReadContext {
    /// Builds a context, capping `wait` at [`AT_LEAST_WAIT_CAP`] so no
    /// caller can request an unbounded block.
    #[must_use]
    pub const fn new(now: crate::time::UnixMicros, ticks: Ticks, wait: Duration) -> Self {
        Self { now, ticks, wait }
    }

    /// Effective wait budget after the hard cap.
    #[must_use]
    pub fn capped_wait(self) -> Duration {
        self.wait.min(AT_LEAST_WAIT_CAP)
    }
}

/// How a replica may establish read authority. The conservative path is
/// always available; the accelerators reuse explicit evidence and fall back
/// the moment that evidence is missing, uncertain, or invalidated. The
/// external contract is identical under every mode: only the mechanism —
/// and the documented assumption behind it — differs.
///
/// Fallback chain (never a weakening; every layer serves the same `Latest`):
///
/// ```text
/// RosterLease ──► AlmostLocal (Lazy-ALR) ──► ConservativeLeader barrier
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReadAuthorityProvider {
    /// Every `Latest` read proves leadership with a fresh quorum barrier.
    /// No timing assumption beyond the barrier itself; the baseline that
    /// can never be weakened.
    ConservativeLeader,
    /// True asynchronous Lazy-ALR backend for Raft/SMR (LAW work): pending
    /// reads batch opportunistically behind one lightweight ordered
    /// [`AlrFence`] sync — ordering of a write, no state mutation — and
    /// execute locally once the sync boundary is applied. A concurrent
    /// write ordered after batch formation may subsume the extra sync.
    ///
    /// No timing assumption. On any doubt (no boundary, coverage timeout,
    /// lost leadership) the read takes a fresh barrier instead.
    AlmostLocal,
    /// True local linearizable reads under an explicit bounded-clock-drift
    /// assumption (Bodega-style directional roster leases, see
    /// [`crate::roster`]): a responder serves only with a stable roster —
    /// same-roster valid grants from an intersecting majority plus local
    /// applied state covering the quorum-derived safety floor.
    ///
    /// Assumption: the deployment honors the configured
    /// [`crate::roster::LeaseParams`] drift bound. If the bound cannot be
    /// trusted the backend is unavailable and reads use Lazy-ALR or the
    /// conservative barrier. Expiry, incarnation change, authority/term
    /// movement, or missing floor coverage voids the fast path
    /// immediately. Restart loses memory-held lease state, and the restart
    /// quarantine fences re-granting until forgotten grants are unusable.
    RosterLease,
}

impl fmt::Display for ReadAuthorityProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConservativeLeader => write!(f, "conservative-leader"),
            Self::AlmostLocal => write!(f, "almost-local"),
            Self::RosterLease => write!(f, "roster-lease"),
        }
    }
}

/// Whether one read may use the [`ReadAuthorityProvider::RosterLease`]
/// fast path. Cross-tablet transactions and multi-tablet scans are not one
/// global snapshot, so until responder coverage integrates with the final
/// committed participant outcome those paths run lease-ineligible: they
/// still use Lazy-ALR or the conservative barrier (same `Latest`
/// semantics), they just never serve from a roster lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LeaseEligibility {
    /// Point reads on one tablet: the roster fast path may serve.
    Eligible,
    /// Transactions, batches, scans: skip the roster fast path.
    ConservativeOnly,
}

impl LeaseEligibility {
    /// Whether a stable roster may serve this read.
    #[must_use]
    pub const fn lease_may_serve(self) -> bool {
        match self {
            Self::Eligible => true,
            Self::ConservativeOnly => false,
        }
    }
}

impl fmt::Display for LeaseEligibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eligible => write!(f, "lease-eligible"),
            Self::ConservativeOnly => write!(f, "lease-ineligible"),
        }
    }
}

/// Identity of one Lazy-ALR synchronization: the fence that orders a batch
/// of pending reads without mutating logical state.
///
/// A fence has the ordering properties of a write (one Raft log slot,
/// committed by the same majority, applied in the same order) and the
/// mutation footprint of nothing: applying it advances the applied pointer
/// only. When a batch's fence — or a subsuming write ordered after batch
/// formation — is locally applied, every write that completed before the
/// batch formed is necessarily visible, so the whole batch executes
/// locally under exact `Latest` semantics with no timing assumption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AlrFence {
    /// Tablet whose ordered stream carries the fence.
    pub tablet: TabletId,
    /// Batch sequence within the tablet (monotonic per replica driver).
    pub batch: u64,
    /// Replica that formed the batch (diagnostics only, never authority).
    pub requester: NodeId,
}

impl AlrFence {
    /// Builds a fence identity from its three facts.
    #[must_use]
    pub const fn new(tablet: TabletId, batch: u64, requester: NodeId) -> Self {
        Self {
            tablet,
            batch,
            requester,
        }
    }
}

impl fmt::Display for AlrFence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "alr-fence(tablet {} batch {} requester {})",
            self.tablet.as_u64(),
            self.batch,
            self.requester.as_u64(),
        )
    }
}

/// Responder write-coverage report: the answering replica applied through
/// `applied` and can (when `healthy`) satisfy the logical read contract
/// from that state — inline values, chunked roots with resolvable
/// sidecars, and manifests alike. The leader gates external write success
/// on these reports (see `ResponderCoverage` in the execution layer):
/// internal Raft commit and externally completed write are separate
/// moments, and linearizability is defined over the latter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoverageReport {
    /// Reporting replica.
    pub responder: NodeId,
    /// Process lifetime that applied the state (restart voids memory).
    pub incarnation: NodeIncarnation,
    /// Applied commit position through the index mapping (inclusive).
    pub applied: CommitPosition,
    /// Whether the replica can serve the logical contract from `applied`
    /// (applied chunked roots resolve; no latched storage fault).
    pub healthy: bool,
}

impl CoverageReport {
    /// Builds a coverage report from its four facts.
    #[must_use]
    pub const fn new(
        responder: NodeId,
        incarnation: NodeIncarnation,
        applied: CommitPosition,
        healthy: bool,
    ) -> Self {
        Self {
            responder,
            incarnation,
            applied,
            healthy,
        }
    }

    /// Whether this report covers a write committed at `commit`: the
    /// responder applied at or beyond it and can serve from it. Anything
    /// else means "wait or revoke", never "assume covered".
    #[must_use]
    pub const fn covers(&self, commit: CommitPosition) -> bool {
        self.healthy && self.applied.as_u64() >= commit.as_u64()
    }
}

impl fmt::Display for CoverageReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "coverage(responder {} applied {} healthy {})",
            self.responder.as_u64(),
            self.applied.as_u64(),
            self.healthy,
        )
    }
}
/// Which path served a read, for operators, debug, and tests. Every fast
/// path names the evidence that made it valid; every fallback names why.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ServePath {
    /// Fresh quorum barrier, then local serve (conservative `Latest`).
    AuthorityBarrier,
    /// `AtLeast` token already covered locally: no leader contact.
    AtLeastCovered,
    /// `AtLeast` token covered after a bounded catch-up wait.
    AtLeastWaited,
    /// `BoundedStale` served from a receipt satisfying the bound.
    BoundedStaleProof,
    /// `BoundedStale` receipt missing/stale: escalated to a fresh barrier
    /// (which satisfies any bound) instead of serving weak data.
    BoundedStaleEscalated,
    /// `Any` served from local applied state with no freshness claim.
    AnyLocal,
    /// Served from the strong cache under a valid authority.
    StrongCache,
    /// `Latest` served after this read's Lazy-ALR batch committed its own
    /// ordered fence and the fence applied locally.
    AlrSync,
    /// `Latest` served after a concurrent write ordered past batch
    /// formation subsumed the extra fence (same boundary proof, one fewer
    /// log entry).
    AlrSubsumed,
    /// `Latest` served under one stable roster with floor coverage.
    RosterLease,
}

impl fmt::Display for ServePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AuthorityBarrier => write!(f, "authority-barrier"),
            Self::AtLeastCovered => write!(f, "at-least-covered"),
            Self::AtLeastWaited => write!(f, "at-least-waited"),
            Self::BoundedStaleProof => write!(f, "bounded-stale-proof"),
            Self::BoundedStaleEscalated => write!(f, "bounded-stale-escalated"),
            Self::AnyLocal => write!(f, "any-local"),
            Self::StrongCache => write!(f, "strong-cache"),
            Self::AlrSync => write!(f, "alr-sync"),
            Self::AlrSubsumed => write!(f, "alr-subsumed"),
            Self::RosterLease => write!(f, "roster-lease"),
        }
    }
}

/// Worker-local consistency counters. Plain integers behind the hub lock —
/// no process-global hot-path atomics. Snapshot into [`ConsistencySnapshot`]
/// for status/admin surfaces.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConsistencyMetrics {
    /// Reads served per contract: `(latest, at_least, bounded_stale, any)`.
    pub reads_by_contract: (u64, u64, u64, u64),
    /// Reads served without a fresh barrier on this read.
    pub local_serves: u64,
    /// Reads served after a fresh authority barrier.
    pub authority_serves: u64,
    /// `AtLeast` reads that waited for catch-up.
    pub at_least_waits: u64,
    /// `AtLeast` waits that timed out instead of covering.
    pub at_least_timeouts: u64,
    /// `AtLeast` tokens rejected for lineage (wrong tablet/epoch/shape).
    pub at_least_lineage_rejects: u64,
    /// `BoundedStale` reads served from satisfying receipts.
    pub bounded_stale_hits: u64,
    /// `BoundedStale` reads escalated to the authority path.
    pub bounded_stale_escalations: u64,
    /// Reads that failed because no freshness proof could be established.
    pub freshness_unprovable: u64,
    /// Strong-cache hits.
    pub strong_cache_hits: u64,
    /// Strong-cache fallbacks (miss, authority moved, entry too old).
    pub strong_cache_fallbacks: u64,
    /// Strong-cache invalidations on authority change.
    pub strong_cache_invalidations: u64,
    /// Lazy-ALR batches formed (one fence or subsumption per batch).
    pub alr_batches: u64,
    /// Reads executed inside Lazy-ALR batches.
    pub alr_reads: u64,
    /// ALR batches that committed their own fence entry.
    pub alr_syncs: u64,
    /// ALR batches whose fence was subsumed by a concurrent write.
    pub alr_subsumed: u64,
    /// ALR batches abandoned to the conservative barrier.
    pub alr_fallbacks: u64,
    /// Roster-lease fast-path hits.
    pub roster_hits: u64,
    /// Roster-lease reads with a stable roster but floor coverage behind
    /// (hold-or-fallback; never thin serves).
    pub roster_holds: u64,
    /// Roster-lease fallbacks (no stable roster, expiry, ineligible path).
    pub roster_fallbacks: u64,
    /// Roster pairings lapsed on local timers.
    pub roster_expiries: u64,
    /// Strong writes gated on explicit responder coverage.
    pub coverage_waits: u64,
    /// Coverage gates that timed out (write failed retryable, responder
    /// fenced) instead of acknowledging an uncovered write.
    pub coverage_timeouts: u64,
    /// Authority/fencing rejections on the read path.
    pub fencing_rejects: u64,
    /// Current stable-roster summary for status surfaces (`None` when no
    /// roster is stable on this replica).
    pub roster: Option<RosterSummary>,
    /// Responders this replica must currently cover before completing a
    /// strong write. Gauge (refreshed on snapshot from live engine
    /// state, not accumulated).
    pub covered_responders: u64,
}

impl ConsistencyMetrics {
    /// Records one served read: contract mix plus local/authority split.
    pub const fn record_serve(&mut self, contract: u8, via_authority: bool) {
        match contract {
            0 => self.reads_by_contract.0 += 1,
            1 => self.reads_by_contract.1 += 1,
            2 => self.reads_by_contract.2 += 1,
            _ => self.reads_by_contract.3 += 1,
        }
        if via_authority {
            self.authority_serves += 1;
        } else {
            self.local_serves += 1;
        }
    }

    /// Freezes a copy for status/admin surfaces.
    #[must_use]
    pub const fn snapshot(&self) -> ConsistencySnapshot {
        ConsistencySnapshot {
            reads_by_contract: self.reads_by_contract,
            local_serves: self.local_serves,
            authority_serves: self.authority_serves,
            at_least_waits: self.at_least_waits,
            at_least_timeouts: self.at_least_timeouts,
            at_least_lineage_rejects: self.at_least_lineage_rejects,
            bounded_stale_hits: self.bounded_stale_hits,
            bounded_stale_escalations: self.bounded_stale_escalations,
            freshness_unprovable: self.freshness_unprovable,
            strong_cache_hits: self.strong_cache_hits,
            strong_cache_fallbacks: self.strong_cache_fallbacks,
            strong_cache_invalidations: self.strong_cache_invalidations,
            alr_batches: self.alr_batches,
            alr_reads: self.alr_reads,
            alr_syncs: self.alr_syncs,
            alr_subsumed: self.alr_subsumed,
            alr_fallbacks: self.alr_fallbacks,
            roster_hits: self.roster_hits,
            roster_holds: self.roster_holds,
            roster_fallbacks: self.roster_fallbacks,
            roster_expiries: self.roster_expiries,
            coverage_waits: self.coverage_waits,
            coverage_timeouts: self.coverage_timeouts,
            fencing_rejects: self.fencing_rejects,
            roster: self.roster,
            covered_responders: self.covered_responders,
        }
    }
}

/// Copy-safe stable-roster summary for status/admin surfaces: every number
/// that makes one local serve safe, without naming the lease internals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RosterSummary {
    /// Roster generation served under.
    pub generation: u64,
    /// Consensus term the roster is bound to.
    pub term: u64,
    /// Designated responders (leader plus listed set).
    pub responders: u64,
    /// Grants backing stability (distinct grantors, same roster).
    pub grants: u64,
    /// Majority count required over current voters.
    pub required: u64,
    /// Quorum-derived safety floor (inclusive commit position).
    pub floor: CommitPosition,
    /// Local applied position at snapshot time.
    pub applied: CommitPosition,
    /// Whether the lease engine is quarantined after (re)start.
    pub quarantined: bool,
}

impl RosterSummary {
    /// One-line operator/debug explanation: why reads under this roster
    /// are safe, not just that a lease is "valid".
    #[must_use]
    pub fn explain(&self) -> String {
        use core::fmt::Write as _;
        let mut out = String::new();
        let _ = write!(
            out,
            "served locally because roster-gen={} term={} grants={}/{} floor={} applied={}",
            self.generation,
            self.term,
            self.grants,
            self.required,
            self.floor.as_u64(),
            self.applied.as_u64(),
        );
        out
    }
}

impl fmt::Display for RosterSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.explain())
    }
}

/// Frozen copy of [`ConsistencyMetrics`] for status/admin surfaces.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ConsistencySnapshot {
    /// Reads served per contract: `(latest, at_least, bounded_stale, any)`.
    pub reads_by_contract: (u64, u64, u64, u64),
    /// Reads served without a fresh barrier on this read.
    pub local_serves: u64,
    /// Reads served after a fresh authority barrier.
    pub authority_serves: u64,
    /// `AtLeast` reads that waited for catch-up.
    pub at_least_waits: u64,
    /// `AtLeast` waits that timed out instead of covering.
    pub at_least_timeouts: u64,
    /// `AtLeast` tokens rejected for lineage (wrong tablet/epoch/shape).
    pub at_least_lineage_rejects: u64,
    /// `BoundedStale` reads served from satisfying receipts.
    pub bounded_stale_hits: u64,
    /// `BoundedStale` reads escalated to the authority path.
    pub bounded_stale_escalations: u64,
    /// Reads that failed because no freshness proof could be established.
    pub freshness_unprovable: u64,
    /// Strong-cache hits.
    pub strong_cache_hits: u64,
    /// Strong-cache fallbacks (miss, authority moved, entry too old).
    pub strong_cache_fallbacks: u64,
    /// Strong-cache invalidations on authority change.
    pub strong_cache_invalidations: u64,
    /// Lazy-ALR batches formed (one fence or subsumption per batch).
    pub alr_batches: u64,
    /// Reads executed inside Lazy-ALR batches.
    pub alr_reads: u64,
    /// ALR batches that committed their own fence entry.
    pub alr_syncs: u64,
    /// ALR batches whose fence was subsumed by a concurrent write.
    pub alr_subsumed: u64,
    /// ALR batches abandoned to the conservative barrier.
    pub alr_fallbacks: u64,
    /// Roster-lease fast-path hits.
    pub roster_hits: u64,
    /// Roster-lease reads with a stable roster but floor coverage behind
    /// (hold-or-fallback; never thin serves).
    pub roster_holds: u64,
    /// Roster-lease fallbacks (no stable roster, expiry, ineligible path).
    pub roster_fallbacks: u64,
    /// Roster pairings lapsed on local timers.
    pub roster_expiries: u64,
    /// Strong writes gated on explicit responder coverage.
    pub coverage_waits: u64,
    /// Coverage gates that timed out (write failed retryable, responder
    /// fenced) instead of acknowledging an uncovered write.
    pub coverage_timeouts: u64,
    /// Authority/fencing rejections on the read path.
    pub fencing_rejects: u64,
    /// Current stable-roster summary for status surfaces (`None` when no
    /// roster is stable on this replica).
    pub roster: Option<RosterSummary>,
    /// Responders this replica must currently cover before completing a
    /// strong write (live grantor exclusions: actively granting plus
    /// revoking-but-unacknowledged). The write-completion gate blocks on
    /// exactly this set.
    pub covered_responders: u64,
}

/// Fencing composition of one tablet lineage (RFC §70, §194, §251).
///
/// Four mechanisms, one rule — a stale owner cannot authorize under the
/// current generation:
///
/// ```text
/// Raft membership ... commit fencing: a removed replica wins no quorum,
///                     so its proposals never commit.
/// Tablet identity ... lineage fencing: split/merge mint fresh tablet ids;
///                     tokens and envelopes naming a retired id mismatch.
/// (TabletEpoch,
///  WriteGuardGeneration) generation fencing: checked at apply time
///                     (TabletAuthority::check_against) and on every read
///                     plan; any movement voids cached evidence.
/// NodeIncarnation . process fencing: restart mints a new incarnation;
///                     peers and leases reject the old one.
/// ```
///
/// New lineage always means a new tablet identity starting at
/// (`INITIAL`, `INITIAL`); a stable tablet identity means one unbroken log
/// lineage, so its epoch/guard never advance beneath it. Ownership moves
/// that preserve the log (worker moves, replica migration) preserve the
/// triple exactly — envelopes in flight stay verifiable, and Raft
/// membership alone decides who may still commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuthorityTransition {
    /// Same-node worker move: execution ownership only, log lineage and
    /// Raft group unchanged. Triple preserved.
    WorkerMove,
    /// Replica migration across nodes: learner catch-up plus membership
    /// change, log lineage continuous. Triple preserved.
    ReplicaMigration,
    /// Split: parent retires (tombstone + redirect), children mint fresh
    /// tablet identities. Old tokens name the parent and mismatch.
    Split,
    /// Merge: parents retire, merged target mints a fresh identity.
    Merge,
    /// Directory replacement: routing snapshot swapped; authorities
    /// re-validated against serving state, mismatch fails closed.
    DirectoryReplace,
    /// Process restart: new incarnation, memory-held proofs and leases
    /// discarded, peers reject the old incarnation.
    IncarnationChange,
}

impl AuthorityTransition {
    /// Whether the transition preserves the authority triple of a stable
    /// tablet identity (and therefore preserves the meaning of outstanding
    /// commit tokens), or retires the lineage (tokens must reject).
    #[must_use]
    pub const fn preserves_lineage(self) -> bool {
        match self {
            Self::WorkerMove | Self::ReplicaMigration | Self::DirectoryReplace => true,
            Self::Split | Self::Merge | Self::IncarnationChange => false,
        }
    }

    /// Whether memory-held freshness evidence (receipts, leases, cache
    /// entries) survives the transition. Only pure routing swaps preserve
    /// it; everything else re-proves from authority.
    #[must_use]
    pub const fn preserves_evidence(self) -> bool {
        match self {
            Self::DirectoryReplace => true,
            Self::WorkerMove
            | Self::ReplicaMigration
            | Self::Split
            | Self::Merge
            | Self::IncarnationChange => false,
        }
    }
}

impl fmt::Display for AuthorityTransition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkerMove => write!(f, "worker-move"),
            Self::ReplicaMigration => write!(f, "replica-migration"),
            Self::Split => write!(f, "split"),
            Self::Merge => write!(f, "merge"),
            Self::DirectoryReplace => write!(f, "directory-replace"),
            Self::IncarnationChange => write!(f, "incarnation-change"),
        }
    }
}

/// Validates that the serving authority still matches the directory's
/// recorded authority for the tablet: the single cross-check between the
/// routing view and the commit view. Any drift fails closed — the replica
/// serves nothing until the two agree again.
///
/// Directory `None` (tablet absent from the snapshot) also fails: an
/// authority the directory no longer names is not servable.
///
/// # Errors
///
/// Returns stale-authority rejection on any drift or absence.
pub const fn reconcile_authority(
    serving: &TabletAuthority,
    directory: Option<&TabletAuthority>,
) -> Result<(), FreshnessReject> {
    match directory {
        None => Err(FreshnessReject::StaleAuthority),
        Some(recorded) => {
            if serving.tablet().as_u64() != recorded.tablet().as_u64()
                || serving.epoch().as_u64() != recorded.epoch().as_u64()
                || serving.guard().as_u64() != recorded.guard().as_u64()
            {
                return Err(FreshnessReject::StaleAuthority);
            }
            Ok(())
        }
    }
}

/// Derives the successor write-guard generation for an ownership move that
/// preserves the log lineage. Fails explicitly at `u64::MAX` instead of
/// wrapping: reusing a generation would resurrect a stale authority.
///
/// # Errors
///
/// Returns generation exhaustion at the top of the numbering space
/// instead of wrapping.
pub const fn next_guard_for_move(
    guard: WriteGuardGeneration,
) -> Result<WriteGuardGeneration, crate::ids::GenerationExhausted> {
    guard.next()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::CommitPosition;
    use crate::time::UnixMicros;

    fn tablet(id: u64) -> TabletId {
        TabletId::from_u64(id)
    }

    fn authority(id: u64, epoch: u64, guard: u64) -> TabletAuthority {
        TabletAuthority::new(
            tablet(id),
            TabletEpoch::from_u64(epoch),
            WriteGuardGeneration::from_u64(guard),
        )
    }

    fn freshness(id: u64, applied: u64) -> ReplicaFreshness {
        ReplicaFreshness::new(
            authority(id, 1, 1),
            CommitPosition::from_u64(applied),
            NodeIncarnation::from_u64(7),
        )
    }

    fn token(id: u64, epoch: u64, position: u64) -> CommitToken {
        CommitToken::new(
            tablet(id),
            TabletEpoch::from_u64(epoch),
            CommitPosition::from_u64(position),
        )
    }

    #[test]
    fn covered_token_serves_without_leader_contact() {
        let fresh = freshness(3, 10);
        assert_eq!(fresh.covers_token(token(3, 1, 10)), Ok(()));
        assert_eq!(fresh.covers_token(token(3, 1, 1)), Ok(()));
    }

    #[test]
    fn behind_token_reports_gap_for_bounded_wait() {
        let fresh = freshness(3, 9);
        assert_eq!(
            fresh.covers_token(token(3, 1, 10)),
            Err(FreshnessReject::BeyondApplied {
                requested: CommitPosition::from_u64(10),
                applied: CommitPosition::from_u64(9),
            })
        );
    }

    #[test]
    fn wrong_lineage_never_validates() {
        let fresh = freshness(3, 10);
        assert!(matches!(
            fresh.covers_token(token(4, 1, 5)),
            Err(FreshnessReject::WrongTablet { .. })
        ));
        // Same position, other epoch: split/merge lineage, must reject.
        assert!(matches!(
            fresh.covers_token(token(3, 2, 5)),
            Err(FreshnessReject::StaleEpoch { .. })
        ));
        // Unassigned position vouches for nothing, even when covered.
        assert_eq!(
            fresh.covers_token(token(3, 1, 0)),
            Err(FreshnessReject::UnassignedToken)
        );
        // Invalid epoch is never assigned either.
        let bad_epoch = CommitToken::new(tablet(3), TabletEpoch::INVALID, CommitPosition::FIRST);
        assert_eq!(
            fresh.covers_token(bad_epoch),
            Err(FreshnessReject::StaleEpoch {
                presented: TabletEpoch::INVALID,
                current: TabletEpoch::from_u64(1),
            })
        );
    }

    #[test]
    fn receipt_usability_tracks_authority_coverage_and_incarnation() {
        let current = authority(3, 1, 1);
        let incarnation = NodeIncarnation::from_u64(7);
        let receipt = FreshnessReceipt::new(
            current,
            CommitPosition::from_u64(9),
            Ticks::from_micros(1_000),
            incarnation,
        );
        assert_eq!(
            receipt.usable_against(&current, CommitPosition::from_u64(9), incarnation),
            Ok(())
        );
        // Follower lag: boundary beyond applied.
        assert!(matches!(
            receipt.usable_against(&current, CommitPosition::from_u64(8), incarnation),
            Err(FreshnessReject::EvidenceBeyondApplied { .. })
        ));
        // Any authority movement voids the receipt.
        let moved = authority(3, 1, 2);
        assert_eq!(
            receipt.usable_against(&moved, CommitPosition::from_u64(9), incarnation),
            Err(FreshnessReject::StaleAuthority)
        );
        // Restart voids memory-held proof.
        assert_eq!(
            receipt.usable_against(
                &current,
                CommitPosition::from_u64(9),
                NodeIncarnation::from_u64(8)
            ),
            Err(FreshnessReject::StaleAuthority)
        );
    }

    #[test]
    fn bounded_stale_bound_is_proven_or_rejected() {
        let current = authority(3, 1, 1);
        let incarnation = NodeIncarnation::from_u64(7);
        let receipt = FreshnessReceipt::new(
            current,
            CommitPosition::from_u64(9),
            Ticks::from_micros(1_000),
            incarnation,
        );
        let bound = Duration::from_millis(100);
        // 50ms old: satisfies a 100ms bound.
        assert_eq!(
            receipt.satisfies_bound(
                bound,
                Ticks::from_micros(51_000),
                &current,
                CommitPosition::from_u64(9),
                incarnation
            ),
            Ok(())
        );
        // 101ms old: outside the bound while labeled successful is
        // forbidden — the caller must escalate or fail.
        assert!(matches!(
            receipt.satisfies_bound(
                bound,
                Ticks::from_micros(101_001),
                &current,
                CommitPosition::from_u64(9),
                incarnation
            ),
            Err(FreshnessReject::ProofTooOld { .. })
        ));
        // Stale authority fails even with a fresh proof.
        let moved = authority(3, 2, 1);
        assert_eq!(
            receipt.satisfies_bound(
                Duration::from_secs(60),
                Ticks::from_micros(1_001),
                &moved,
                CommitPosition::from_u64(9),
                incarnation
            ),
            Err(FreshnessReject::StaleAuthority)
        );
    }

    #[test]
    fn receipt_age_uses_monotonic_ticks_not_wall_time() {
        let current = authority(3, 1, 1);
        let receipt = FreshnessReceipt::new(
            current,
            CommitPosition::FIRST,
            Ticks::from_micros(10_000),
            NodeIncarnation::from_u64(7),
        );
        assert_eq!(
            receipt.age(Ticks::from_micros(12_500)),
            Duration::from_micros(2_500)
        );
        // Clock skew fails closed to "no time passed", never negative age.
        assert_eq!(receipt.age(Ticks::from_micros(9_000)), Duration::ZERO);
    }

    #[test]
    fn coverage_reports_cover_only_healthy_applied_state() {
        let report = CoverageReport::new(
            NodeId::from_u64(2),
            NodeIncarnation::from_u64(7),
            CommitPosition::from_u64(41),
            true,
        );
        assert!(report.covers(CommitPosition::from_u64(41)));
        assert!(report.covers(CommitPosition::from_u64(40)));
        assert!(!report.covers(CommitPosition::from_u64(42)));
        // An unhealthy replica covers nothing, even past its applied
        // pointer: it cannot satisfy the logical contract from that
        // state (missing sidecars, latched fault).
        let unhealthy = CoverageReport::new(
            NodeId::from_u64(2),
            NodeIncarnation::from_u64(7),
            CommitPosition::from_u64(99),
            false,
        );
        assert!(!unhealthy.covers(CommitPosition::from_u64(41)));
        assert_eq!(
            report.to_string(),
            "coverage(responder 2 applied 41 healthy true)"
        );
    }

    #[test]
    fn alr_fences_name_tablet_batch_and_requester() {
        let fence = AlrFence::new(tablet(3), 17, NodeId::from_u64(2));
        assert_eq!(fence.tablet, tablet(3));
        assert_eq!(fence.batch, 17);
        assert_eq!(
            fence.to_string(),
            "alr-fence(tablet 3 batch 17 requester 2)"
        );
    }

    #[test]
    fn lease_eligibility_gates_only_the_roster_path() {
        assert!(LeaseEligibility::Eligible.lease_may_serve());
        assert!(!LeaseEligibility::ConservativeOnly.lease_may_serve());
        assert_eq!(LeaseEligibility::Eligible.to_string(), "lease-eligible");
    }

    #[test]
    fn read_receipt_chains_into_at_least_tokens() {
        let current = authority(3, 1, 1);
        let receipt = ReadReceipt::new(
            current,
            CommitPosition::from_u64(9),
            NodeIncarnation::from_u64(7),
        );
        let chained = receipt.token();
        assert_eq!(chained.tablet(), tablet(3));
        assert_eq!(chained.position(), CommitPosition::from_u64(9));
        // The issuing replica covers its own receipt.
        let fresh = ReplicaFreshness::new(
            current,
            CommitPosition::from_u64(9),
            NodeIncarnation::from_u64(7),
        );
        assert_eq!(fresh.covers_token(chained), Ok(()));
    }

    #[test]
    fn reconcile_fails_closed_on_any_drift_or_absence() {
        let serving = authority(3, 1, 1);
        assert_eq!(reconcile_authority(&serving, Some(&serving)), Ok(()));
        assert_eq!(
            reconcile_authority(&serving, None),
            Err(FreshnessReject::StaleAuthority)
        );
        assert_eq!(
            reconcile_authority(&serving, Some(&authority(3, 2, 1))),
            Err(FreshnessReject::StaleAuthority)
        );
        assert_eq!(
            reconcile_authority(&serving, Some(&authority(3, 1, 2))),
            Err(FreshnessReject::StaleAuthority)
        );
    }

    #[test]
    fn transitions_name_what_survives_them() {
        assert!(AuthorityTransition::WorkerMove.preserves_lineage());
        assert!(AuthorityTransition::ReplicaMigration.preserves_lineage());
        assert!(AuthorityTransition::DirectoryReplace.preserves_lineage());
        assert!(!AuthorityTransition::Split.preserves_lineage());
        assert!(!AuthorityTransition::Merge.preserves_lineage());
        assert!(!AuthorityTransition::IncarnationChange.preserves_lineage());
        // Only pure routing swaps preserve memory-held evidence.
        assert!(AuthorityTransition::DirectoryReplace.preserves_evidence());
        assert!(!AuthorityTransition::WorkerMove.preserves_evidence());
        assert!(!AuthorityTransition::ReplicaMigration.preserves_evidence());
    }

    #[test]
    fn context_caps_unbounded_waits() {
        let ctx = ReadContext::new(
            UnixMicros::from_micros(1_000),
            Ticks::from_micros(2_000),
            Duration::from_secs(3_600),
        );
        assert_eq!(ctx.capped_wait(), AT_LEAST_WAIT_CAP);
        let short = ReadContext::new(
            UnixMicros::from_micros(1_000),
            Ticks::from_micros(2_000),
            Duration::from_millis(50),
        );
        assert_eq!(short.capped_wait(), Duration::from_millis(50));
    }

    #[test]
    fn provider_modes_name_their_assumptions() {
        assert_eq!(
            ReadAuthorityProvider::AlmostLocal.to_string(),
            "almost-local"
        );
        assert_eq!(
            ReadAuthorityProvider::RosterLease.to_string(),
            "roster-lease"
        );
        assert_eq!(
            ReadAuthorityProvider::ConservativeLeader.to_string(),
            "conservative-leader"
        );
    }

    #[test]
    fn serve_paths_name_their_evidence() {
        assert_eq!(ServePath::AlrSync.to_string(), "alr-sync");
        assert_eq!(ServePath::AlrSubsumed.to_string(), "alr-subsumed");
        assert_eq!(ServePath::RosterLease.to_string(), "roster-lease");
    }

    #[test]
    fn roster_summaries_explain_why_a_read_is_safe() {
        let summary = RosterSummary {
            generation: 4,
            term: 9,
            responders: 3,
            grants: 2,
            required: 2,
            floor: CommitPosition::from_u64(18291),
            applied: CommitPosition::from_u64(18304),
            quarantined: false,
        };
        let text = summary.explain();
        assert!(
            text.contains("grants=2/2"),
            "explains grant count, got: {text}"
        );
        assert!(text.contains("floor=18291"), "explains floor, got: {text}");
        assert!(
            text.contains("applied=18304"),
            "explains coverage, got: {text}"
        );
    }

    #[test]
    fn metrics_record_contract_mix_and_path_split() {
        let mut metrics = ConsistencyMetrics::default();
        metrics.record_serve(0, true);
        metrics.record_serve(1, false);
        metrics.record_serve(2, false);
        metrics.record_serve(7, false);
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.reads_by_contract, (1, 1, 1, 1));
        assert_eq!(snapshot.authority_serves, 1);
        assert_eq!(snapshot.local_serves, 3);
    }

    #[test]
    fn guard_successors_fail_explicitly_at_u64_max() {
        assert_eq!(
            next_guard_for_move(WriteGuardGeneration::from_u64(u64::MAX)),
            Err(crate::ids::GenerationExhausted::WriteGuardGeneration)
        );
        assert!(
            next_guard_for_move(WriteGuardGeneration::INITIAL)
                .expect("advances")
                .supersedes(WriteGuardGeneration::INITIAL)
        );
    }
}
