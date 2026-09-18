//! First-class commit/freshness metadata for the consistency layer (RFC §66–§72,
//! Phase 7).
//!
//! The consistency layer reasons about exactly four facts, and nothing else:
//!
//! ```text
//! what this replica has applied ........... ReplicaFreshness::applied
//! what authority that information belongs to  ReplicaFreshness::authority
//! what commit/freshness evidence is known ... FreshnessReceipt / RosterLease
//! whether that evidence survives change .... authority-equality checks
//! ```
//!
//! Wall-clock age is never equated with replication freshness: a locally old
//! timestamp proves nothing about what the leader has committed. A replica may
//! claim a staleness bound only when it holds [`FreshnessReceipt`] evidence —
//! a recent authority proof (consensus barrier or valid [`RosterLease`]) —
//! whose age on the caller's monotonic clock ([`Ticks`], always passed in,
//! never read) fits the bound, whose authority still matches the current
//! authority exactly, and whose coverage boundary the replica has applied.
//! Without such evidence the replica escalates to the authority path or
//! reports the bound unprovable; it never serves out-of-bound data as success.
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

/// How long an [`FreshnessReceipt`] or [`RosterLease`] age may be trusted
/// when the caller supplies no tighter bound. Separate from any read
/// contract: this caps internal fast-path reuse, never a client promise.
pub const DEFAULT_PROOF_TTL: Duration = Duration::from_secs(2);

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

/// How a replica may establish read authority (RFC §69). The conservative
/// path is always available; the fast paths reuse explicit evidence and
/// fall back the moment that evidence is missing, stale, ambiguous, or
/// invalidated. The external contract is identical under every mode: only
/// the evidence — and the documented assumption behind reusing it — differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReadAuthorityProvider {
    /// Every `Latest` read proves leadership with a fresh quorum barrier.
    /// No timing assumption beyond the barrier itself; the baseline that
    /// can never be weakened.
    ConservativeLeader,
    /// A replica that proved leadership within `max_proof_age` (a barrier
    /// receipt on the same monotonic clock) may serve `Latest` from applied
    /// state covering that receipt's boundary without a fresh barrier.
    ///
    /// Assumption: no conflicting leadership committed newer state inside
    /// the window undetected — i.e. bounded message delay plus a timely
    /// quorum. On any doubt (no receipt, aged receipt, authority movement,
    /// coverage gap) the read takes a fresh barrier. Removing this mode
    /// changes latency only, never semantics.
    AlmostLocal {
        /// Maximum receipt age trusted without a fresh barrier.
        max_proof_age: Duration,
    },
    /// Any roster member holding a valid [`RosterLease`] may serve `Latest`
    /// from applied state covering the lease floor without contacting the
    /// leader.
    ///
    /// Assumption: the lease grant's synchrony contract held — the issuer
    /// fenced other writers for the lease duration and the holder's clock
    /// stayed within skew of the issuer's. Expiry, incarnation change, or
    /// authority movement voids the lease immediately; the holder falls
    /// back to the conservative path. Restart loses memory-held leases, so
    /// a restarted member never serves the fast path until re-granted.
    RosterLease,
}

impl fmt::Display for ReadAuthorityProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConservativeLeader => write!(f, "conservative-leader"),
            Self::AlmostLocal { max_proof_age } => {
                write!(f, "almost-local({}ms)", max_proof_age.as_millis())
            }
            Self::RosterLease => write!(f, "roster-lease"),
        }
    }
}

/// Roster-based read authority (RFC §69, Bodega mode): the lease issuer
/// (the leader, after quorum agreement) designates a subset of replicas as
/// local linearizable responders for a bounded duration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RosterLease {
    /// Authority the lease was granted under. Any movement voids it.
    pub authority: TabletAuthority,
    /// Lease generation within the authority (monotonic; renewal advances
    /// it so a delayed duplicate grant cannot resurrect an older lease).
    pub generation: u64,
    /// Designated responders, sorted ascending, deduplicated.
    pub members: Vec<NodeId>,
    /// Minimum applied coverage a holder must have to serve (the issuer's
    /// committed floor at grant time). A holder behind the floor waits
    /// (bounded) or falls back — it never serves thinner state as lease
    /// authority.
    pub floor: CommitPosition,
    /// Monotonic expiry (`Ticks` of the issuer's clock domain). Valid while
    /// `now < expires_at`; equality fails closed.
    pub expires_at: Ticks,
    /// Holder incarnation the lease was granted to. A restart mints a new
    /// incarnation and voids every lease the old process held.
    pub incarnation: NodeIncarnation,
}

impl RosterLease {
    /// Issues a lease, normalizing the roster (sorted, deduplicated).
    #[must_use]
    pub fn new(
        authority: TabletAuthority,
        generation: u64,
        members: Vec<NodeId>,
        floor: CommitPosition,
        expires_at: Ticks,
        incarnation: NodeIncarnation,
    ) -> Self {
        let mut members = members;
        members.sort();
        members.dedup();
        Self {
            authority,
            generation,
            members,
            floor,
            expires_at,
            incarnation,
        }
    }

    /// Whether `node` in `incarnation` may serve under this lease at `now`
    /// against `current` authority. Every failure names its check so the
    /// holder falls back for the right reason and metrics stay meaningful.
    ///
    /// # Errors
    ///
    /// Returns lease-invalid rejection naming the failed check.
    pub fn authorizes(
        &self,
        node: NodeId,
        incarnation: NodeIncarnation,
        current: &TabletAuthority,
        now: Ticks,
    ) -> Result<(), FreshnessReject> {
        if !self.members.contains(&node) {
            return Err(FreshnessReject::InvalidLease {
                detail: LeaseInvalid::NotMember,
            });
        }
        if self.incarnation.as_u64() != incarnation.as_u64() {
            return Err(FreshnessReject::InvalidLease {
                detail: LeaseInvalid::WrongIncarnation,
            });
        }
        if self.authority.tablet().as_u64() != current.tablet().as_u64()
            || self.authority.epoch().as_u64() != current.epoch().as_u64()
            || self.authority.guard().as_u64() != current.guard().as_u64()
        {
            return Err(FreshnessReject::InvalidLease {
                detail: LeaseInvalid::WrongAuthority,
            });
        }
        if now.as_micros() >= self.expires_at.as_micros() {
            return Err(FreshnessReject::InvalidLease {
                detail: LeaseInvalid::Expired,
            });
        }
        Ok(())
    }

    /// Whether `other` supersedes this lease (same authority, higher
    /// generation): renewal installs, delayed duplicates die.
    #[must_use]
    pub const fn superseded_by(&self, other: &Self) -> bool {
        other.generation > self.generation
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
    /// `Latest` served from a valid Almost-Local leadership receipt.
    AlmostLocalProof,
    /// `Latest` served under a valid roster lease.
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
            Self::AlmostLocalProof => write!(f, "almost-local-proof"),
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
    /// Almost-Local fast-path hits.
    pub almost_local_hits: u64,
    /// Almost-Local fallbacks to the barrier.
    pub almost_local_fallbacks: u64,
    /// Roster-lease fast-path hits.
    pub roster_hits: u64,
    /// Roster-lease fallbacks (invalid lease, coverage gap).
    pub roster_fallbacks: u64,
    /// Authority/fencing rejections on the read path.
    pub fencing_rejects: u64,
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
            almost_local_hits: self.almost_local_hits,
            almost_local_fallbacks: self.almost_local_fallbacks,
            roster_hits: self.roster_hits,
            roster_fallbacks: self.roster_fallbacks,
            fencing_rejects: self.fencing_rejects,
        }
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
    /// Almost-Local fast-path hits.
    pub almost_local_hits: u64,
    /// Almost-Local fallbacks to the barrier.
    pub almost_local_fallbacks: u64,
    /// Roster-lease fast-path hits.
    pub roster_hits: u64,
    /// Roster-lease fallbacks (invalid lease, coverage gap).
    pub roster_fallbacks: u64,
    /// Authority/fencing rejections on the read path.
    pub fencing_rejects: u64,
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
    fn roster_lease_checks_run_member_incarnation_authority_expiry() {
        let current = authority(3, 1, 1);
        let incarnation = NodeIncarnation::from_u64(7);
        let lease = RosterLease::new(
            current,
            4,
            vec![NodeId::from_u64(2), NodeId::from_u64(1)],
            CommitPosition::from_u64(9),
            Ticks::from_micros(10_000),
            incarnation,
        );
        // Roster normalized on issue.
        assert_eq!(
            lease.members,
            vec![NodeId::from_u64(1), NodeId::from_u64(2)]
        );
        let now = Ticks::from_micros(9_999);
        assert_eq!(
            lease.authorizes(NodeId::from_u64(1), incarnation, &current, now),
            Ok(())
        );
        // Non-member never serves the fast path.
        assert_eq!(
            lease.authorizes(NodeId::from_u64(9), incarnation, &current, now),
            Err(FreshnessReject::InvalidLease {
                detail: LeaseInvalid::NotMember
            })
        );
        // Restart voids the old process's lease.
        assert_eq!(
            lease.authorizes(
                NodeId::from_u64(1),
                NodeIncarnation::from_u64(8),
                &current,
                now
            ),
            Err(FreshnessReject::InvalidLease {
                detail: LeaseInvalid::WrongIncarnation
            })
        );
        // Authority movement voids even unexpired leases.
        let moved = authority(3, 1, 2);
        assert_eq!(
            lease.authorizes(NodeId::from_u64(1), incarnation, &moved, now),
            Err(FreshnessReject::InvalidLease {
                detail: LeaseInvalid::WrongAuthority
            })
        );
        // Expiry is strict: equality fails closed.
        assert_eq!(
            lease.authorizes(
                NodeId::from_u64(1),
                incarnation,
                &current,
                Ticks::from_micros(10_000)
            ),
            Err(FreshnessReject::InvalidLease {
                detail: LeaseInvalid::Expired
            })
        );
    }

    #[test]
    fn lease_renewal_supersedes_by_generation_only() {
        let current = authority(3, 1, 1);
        let incarnation = NodeIncarnation::from_u64(7);
        let old = RosterLease::new(
            current,
            4,
            vec![NodeId::from_u64(1)],
            CommitPosition::FIRST,
            Ticks::from_micros(10_000),
            incarnation,
        );
        let renewed = RosterLease::new(
            current,
            5,
            vec![NodeId::from_u64(1)],
            CommitPosition::from_u64(12),
            Ticks::from_micros(20_000),
            incarnation,
        );
        assert!(old.superseded_by(&renewed));
        assert!(!renewed.superseded_by(&old));
        assert!(!old.superseded_by(&old));
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
            ReadAuthorityProvider::AlmostLocal {
                max_proof_age: Duration::from_millis(50)
            }
            .to_string(),
            "almost-local(50ms)"
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
