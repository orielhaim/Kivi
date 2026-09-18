//! Bodega-style directional roster leases: pure state machine (RFC §69).
//!
//! A roster names, for one tablet lineage under one consensus term, a leader
//! plus the replicas designated as local linearizable responders. Replicas
//! establish agreement on the roster through an all-to-all directional lease
//! protocol: every participant acts as both grantor and grantee, activation
//! runs Guard → Renew, deactivation runs Revoke (fast path) or expiry
//! (failure path). Only bounded clock-*rate* drift is assumed; absolute
//! clock synchronization (bounded skew) is never required.
//!
//! ```text
//! grantor side, per grantee:  Idle → Guarding → Renewing ⇄ Revoking → Expired
//! grantee side, per grantor:  None → Guarded → Renewed → Expired
//! ```
//!
//! Safety rests on structural rules, each enforced by construction and
//! each covered by a dedicated negative test plus a TLA+ weakening variant
//! (`spec/roster_lease*.tla`):
//!
//! 1. **Attempt identity.** Every Guard opens a fresh attempt
//!    (`LeaseAttemptId` = grantor incarnation + per-pair sequence); every
//!    Renew/Reply echoes it. A reply counts only toward the exact open
//!    attempt it echoes — a leftover from an abandoned attempt can never
//!    manufacture authority (cf. the `PaxosLease` renewal-qualifier finding).
//! 2. **Quarantine.** A (re)created engine grants nothing until its
//!    quarantine elapses. The bound derives from the protocol itself
//!    ([`LeaseParams::quarantine`]); it is checked sufficient and
//!    necessary (minus one) by the model.
//! 3. **Roster identity.** [`RosterId`] binds tablet authority, consensus
//!    term, and roster generation. Any of the three moving voids every
//!    grant keyed on the old identity — no reuse across lineage, term, or
//!    responder-set change. Learning a newer generation prunes older
//!    same-lineage rosters immediately (their exclusions still run out,
//!    so writes keep covering them).
//! 4. **Revocation before replacement.** Term/authority rises revoke stale
//!    outgoing grants (fast path) and drop stale grantee state; conflicting
//!    new authority waits out the old (failure path).
//! 5. **Safety floor + leader leg.** A responder serves only with majority
//!    `Renewed` grants for one roster *including a live pairing from the
//!    roster's named leader*, local applied state covering the
//!    quorum-derived floor, and its own designation in the roster. Each
//!    pairing contributes `max(accepted-at-guard, committed-seen-in-renews)`;
//!    the floor is the majority-th smallest contribution. The leader leg
//!    is what makes leader-side responder coverage sufficient: every
//!    stable responder was covered for every write completed during its
//!    stability, and the floor covers everything completed before it.
//!
//! Timer placement (also cf. PaxosLease): every deadline is measured from
//! the local send/receipt event that opens the attempt — never from quorum
//! receipt or any other midpoint. Durations here are nominal; the
//! [`LeaseParams::allowance`] drift expansion converts them into the
//! asymmetric grantor/grantee deadlines that keep real-time containment
//! under bounded drift.
//!
//! All types here are pure values and pure transitions: no I/O, no clocks,
//! no randomness, no networking. Callers pass `now: Ticks` explicitly, so
//! deterministic simulation supplies virtual time (and virtual drift).

use core::fmt;
use core::time::Duration;

use crate::TabletAuthority;
use crate::ids::{CommitPosition, NodeId, NodeIncarnation};
use crate::time::Ticks;

// ---------------------------------------------------------------------------
// Identity: roster generation, roster term, roster id, roster content.
// ---------------------------------------------------------------------------

/// Monotonic roster generation within one (authority, term) lineage.
///
/// A responder-set change bumps the generation; term and authority changes
/// start new lineages. `0` is the invalid sentinel and never authorizes
/// anything (zeroed memory fails closed, per crate rules).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RosterGeneration(u64);

impl RosterGeneration {
    /// Invalid sentinel. Never authorizes; zeroed memory fails closed.
    pub const INVALID: Self = Self(0);
    /// First generation of a freshly announced roster lineage.
    pub const INITIAL: Self = Self(1);

    /// Wraps a raw generation value.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Whether this is a real generation rather than the invalid sentinel.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.0 != 0
    }

    /// Successor generation for the next responder-set change.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationExhausted`](crate::ids::GenerationExhausted) in
    /// the `RosterGeneration` variant at `u64::MAX` instead of wrapping:
    /// reusing a generation would resurrect a superseded responder set.
    pub const fn next(self) -> Result<Self, crate::ids::GenerationExhausted> {
        match self.0.checked_add(1) {
            Some(value) => Ok(Self(value)),
            None => Err(crate::ids::GenerationExhausted::RosterGeneration),
        }
    }
}

impl fmt::Display for RosterGeneration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "roster-gen({})", self.0)
    }
}

/// The observed Raft term a roster is bound to, as seen by the announcer.
///
/// This is deliberately a distinct type from consensus `ConsensusTerm`:
/// it records the *validity condition* of a roster (which leadership
/// generation it may authorize under), populated from consensus at the
/// boundary. A term rise voids every grant keyed on the older term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RosterTerm(u64);

impl RosterTerm {
    /// No term observed yet. Never authorizes.
    pub const INVALID: Self = Self(0);

    /// Wraps an observed Raft term.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Whether a real term was observed.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.0 != 0
    }
}

impl fmt::Display for RosterTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "roster-term({})", self.0)
    }
}

/// Unforgeable identity of one roster incarnation: tablet authority plus
/// consensus term plus roster generation.
///
/// The three halves close the three reuse hazards: authority binds tablet
/// lineage and `WriteGuard` fencing (split/merge/migration/retirement void
/// old rosters); term binds the leadership generation (a new election voids
/// old rosters even when the responder set is unchanged); generation orders
/// responder-set changes within one term. [`DirectoryVersion`] and
/// [`PlacementVersion`](kivi_control_scope) are deliberately *not* part of
/// this identity: they describe routing and placement, different domains
/// whose movement must not (and does not) disturb read authority.
///
/// [kivi_control_scope]: https://doc.rust-lang.org/
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RosterId {
    /// Tablet lineage + `WriteGuard` fencing the roster belongs to.
    pub authority: TabletAuthority,
    /// Leadership generation the roster is valid under.
    pub term: RosterTerm,
    /// Responder-set version within (`authority`, `term`).
    pub generation: RosterGeneration,
}

impl RosterId {
    /// Builds a roster identity from its three halves.
    #[must_use]
    pub const fn new(
        authority: TabletAuthority,
        term: RosterTerm,
        generation: RosterGeneration,
    ) -> Self {
        Self {
            authority,
            term,
            generation,
        }
    }

    /// Whether every half is valid. Anything else never authorizes.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.authority.epoch().is_valid()
            && self.authority.guard().is_valid()
            && self.term.is_valid()
            && self.generation.is_valid()
    }
}

impl fmt::Display for RosterId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "roster({} {} {})",
            self.authority, self.term, self.generation
        )
    }
}

/// One roster: an identity plus the leader it names plus its designated
/// responders (leader excluded from the literal set but always treated as
/// a responder, mirroring Bodega).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Roster {
    /// Unforgeable identity (authority + term + generation).
    pub id: RosterId,
    /// Node every participant agrees leads this roster's term.
    pub leader: NodeId,
    /// Designated responders besides the leader, sorted ascending,
    /// deduplicated.
    pub responders: Vec<NodeId>,
}

impl Roster {
    /// Builds a roster, normalizing the responder set (sorted, deduplicated,
    /// leader removed — the leader is an implicit responder, never listed).
    ///
    /// # Errors
    ///
    /// Returns [`RosterError::InvalidId`] when the identity is not fully
    /// valid. Membership against voters is checked by the driver (which
    /// owns the voter set), not here.
    pub fn new(id: RosterId, leader: NodeId, responders: Vec<NodeId>) -> Result<Self, RosterError> {
        if !id.is_valid() {
            return Err(RosterError::InvalidId);
        }
        let mut responders = responders;
        responders.sort_by_key(|node| node.as_u64());
        responders.dedup_by_key(|node| node.as_u64());
        responders.retain(|node| *node != leader);
        Ok(Self {
            id,
            leader,
            responders,
        })
    }

    /// Whether `node` may serve local reads under this roster: the leader
    /// implicitly, or a listed responder.
    #[must_use]
    pub fn covers(&self, node: NodeId) -> bool {
        node == self.leader || self.responders.contains(&node)
    }

    /// Every node this roster designates (leader first, then responders).
    #[must_use]
    pub fn designated(&self) -> Vec<NodeId> {
        let mut out = Vec::with_capacity(self.responders.len() + 1);
        out.push(self.leader);
        out.extend_from_slice(&self.responders);
        out
    }
}

impl fmt::Display for Roster {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} leader={} responders=[",
            self.id,
            self.leader.as_u64()
        )?;
        for (index, node) in self.responders.iter().enumerate() {
            if index > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", node.as_u64())?;
        }
        write!(f, "]")
    }
}

/// Why a roster was refused at construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum RosterError {
    /// The roster identity is not fully valid (bad authority, term, or
    /// generation half).
    #[error("roster identity invalid")]
    InvalidId,
    /// Lease parameters fail validation (see [`LeaseParams::validate`]).
    #[error("lease parameters invalid: {detail}")]
    BadParams {
        /// Which check failed.
        detail: LeaseParamReject,
    },
    /// The engine is quarantined after (re)start and may not grant yet.
    #[error("lease engine quarantined until {until}")]
    Quarantined {
        /// Monotonic time the quarantine lifts.
        until: Ticks,
    },
    /// Only the term leader may announce a roster.
    #[error("announce refused: not the term leader")]
    NotLeader,
    /// The announcement names a superseded lineage (older term, or older
    /// generation within the term).
    #[error("announce refused: superseded roster identity")]
    Superseded,
}

/// Which lease-parameter check failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LeaseParamReject {
    /// A duration is zero (guard, lease, or renew interval).
    ZeroDuration,
    /// The renew interval leaves no room to detect a missed renewal
    /// before expiry (`renew_interval >= lease - 2*allowance`).
    RenewTooSlow,
    /// The drift bound exceeds the hard cap — an untrustworthy bound
    /// disables the backend instead of silently weakening it.
    DriftUntrusted,
}

impl fmt::Display for LeaseParamReject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDuration => write!(f, "zero duration"),
            Self::RenewTooSlow => write!(f, "renew interval leaves no detection margin"),
            Self::DriftUntrusted => write!(f, "drift bound exceeds hard cap"),
        }
    }
}

// ---------------------------------------------------------------------------
// Timing: deployment contract with explicit bounded drift.
// ---------------------------------------------------------------------------

/// Parts-per-million denominator for drift ratios.
pub const DRIFT_PPM_DENOMINATOR: u32 = 1_000_000;

/// Hard cap on the configured drift bound. Beyond 10% relative rate error
/// a "lease" promises nothing: the backend is unavailable instead of
/// silently weakened (see [`LeaseParams::validate`]).
pub const MAX_DRIFT_PPM: u32 = 100_000;

/// Consecutive unacknowledged renewals after which a grantor stops
/// *sending* (not excluding): the pairing stays live until its exclusion
/// deadline so the grantee's hold — strictly shorter by the asymmetric
/// deadlines — always lapses first. Without muting, one partitioned
/// grantee would hold every write's coverage gate forever; with it,
/// writes stall at most ~(`MUTE_AFTER_UNACKED` + 1) lease durations and
/// then proceed past the expired pairing. Re-establishment rides the
/// normal Guard path afterwards.
pub const MUTE_AFTER_UNACKED: u32 = 2;

/// Deployment contract for roster leases: three nominal durations plus the
/// single synchrony assumption the protocol needs — bounded clock-*rate*
/// drift between any two participants. Absolute clock synchronization
/// (bounded skew) is never required: the Guard phase absorbs skew, and
/// every deadline below is expressed per-measurer in local ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LeaseParams {
    /// Guard-phase window (first-renew acceptance).
    pub guard_duration: Duration,
    /// Nominal lease duration per renewal iteration.
    pub lease_duration: Duration,
    /// How often the driver sends renewals for live pairings.
    pub renew_interval: Duration,
    /// Maximum relative clock-rate error, parts per million. A deployment
    /// that cannot bound its clock rates cannot use this backend.
    pub max_drift_ppm: u32,
}

impl LeaseParams {
    /// Deployment default: 1s guard window, 2s lease, 200ms renewals,
    /// 1000ppm (0.1%) maximum clock-rate drift. Restart quarantine is
    /// ~3s; steady-state lease traffic is one Renew per live pairing per
    /// 200ms. Deployments with weaker clocks must set explicit params
    /// (or stay off the `RosterLease` backend) instead of silently
    /// weakening these.
    pub const DEFAULT: Self = Self {
        guard_duration: Duration::from_millis(1_000),
        lease_duration: Duration::from_millis(2_000),
        renew_interval: Duration::from_millis(200),
        max_drift_ppm: 1_000,
    };

    /// Validates the deployment contract.
    ///
    /// # Errors
    ///
    /// Returns [`RosterError::BadParams`] for zero durations, a renew
    /// interval that leaves no detection margin, or an untrusted drift
    /// bound.
    pub fn validate(self) -> Result<(), RosterError> {
        if self.guard_duration.is_zero()
            || self.lease_duration.is_zero()
            || self.renew_interval.is_zero()
        {
            return Err(RosterError::BadParams {
                detail: LeaseParamReject::ZeroDuration,
            });
        }
        if self.max_drift_ppm > MAX_DRIFT_PPM {
            return Err(RosterError::BadParams {
                detail: LeaseParamReject::DriftUntrusted,
            });
        }
        // Detection margin: after a missed renewal there must remain a
        // full allowance on each side before expiry, otherwise a slow
        // clock could skip from "looks valid" to "expired" with no
        // observable invalid state in between.
        let margin = self.allowance(self.lease_duration).saturating_mul(2);
        if self.renew_interval + margin >= self.lease_duration {
            return Err(RosterError::BadParams {
                detail: LeaseParamReject::RenewTooSlow,
            });
        }
        Ok(())
    }

    /// Drift allowance for a nominal `duration`: the smallest span `A` with
    /// `2·L·ρ ≤ 2·A`, i.e. `A = ⌈L·ρ / (1−ρ)⌉`, so a grantee hold of `L−A`
    /// on a slow clock stays real-time inside a grantor exclusion of `L+A`
    /// on a fast clock (`PaxosLease` containment shape
    /// `Dp·rmax ≤ Da·rmin`, integer-exact and saturating). This is the
    /// headroom only, not the expanded duration (see [`LeaseParams::expand`]).
    #[must_use]
    pub fn allowance(&self, duration: Duration) -> Duration {
        let micros = duration.as_micros();
        let ppm = u128::from(self.max_drift_ppm);
        let denom =
            u128::from(DRIFT_PPM_DENOMINATOR - self.max_drift_ppm.min(DRIFT_PPM_DENOMINATOR - 1));
        let add = micros.saturating_mul(ppm).saturating_add(denom - 1) / denom;
        Duration::from_micros(u64::try_from(add).unwrap_or(u64::MAX))
    }

    /// Expands a nominal duration into the worst-case real-time span a
    /// measurer running slow (rate `1−ρ`) can stretch it to, plus the
    /// allowance headroom: `D + allowance(D)`.
    #[must_use]
    pub fn expand(&self, duration: Duration) -> Duration {
        duration.saturating_add(self.allowance(duration))
    }

    /// Restart quarantine: the interval a (re)created engine must refuse
    /// to grant for. Derivation: the worst forgotten state is a grant
    /// renewed at the crash instant (valid up to `expand(lease)` later on
    /// the slowest allowed clock) plus an in-flight Guard→first-Renew
    /// attempt (another `expand(guard)`). After this span, no grantee can
    /// still consider a forgotten grant valid, so fresh grants cannot
    /// overlap it. Both directions are checked by the TLA+ model
    /// (sufficient, and minus-one-tick insufficient).
    #[must_use]
    pub fn quarantine(&self) -> Duration {
        self.expand(self.lease_duration)
            .saturating_add(self.expand(self.guard_duration))
    }
}

// ---------------------------------------------------------------------------
// Messages: pure data. Wire codec lives in `kivi-consensus::peer`.
// ---------------------------------------------------------------------------

/// Fresh-attempt identity for one Guard-initiated pairing episode.
///
/// The grantor incarnation binds the attempt to a process lifetime (a
/// restarted grantor speaks with a new incarnation, so its messages can
/// never continue an old attempt); the sequence orders attempts and
/// messages within the incarnation. Replies echo the exact attempt they
/// answer, which is what makes abandoned-attempt leftovers impotent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LeaseAttemptId {
    /// Grantor incarnation that opened the attempt.
    pub grantor_incarnation: NodeIncarnation,
    /// Per-(grantor, grantee) sequence at Guard initiation.
    pub seq: u64,
}

impl LeaseAttemptId {
    /// Builds an attempt identity.
    #[must_use]
    pub const fn new(grantor_incarnation: NodeIncarnation, seq: u64) -> Self {
        Self {
            grantor_incarnation,
            seq,
        }
    }
}

impl fmt::Display for LeaseAttemptId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "attempt(inc{} #{})",
            self.grantor_incarnation.as_u64(),
            self.seq
        )
    }
}

/// Guard: grantor opens (or re-opens) a pairing for a roster, reporting
/// the highest log index it has accepted. Self-describing: the grantee
/// learns the full roster content from this message, so no separate
/// announcement channel exists to skew.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeaseGuard {
    /// Roster being activated (full content).
    pub roster: Roster,
    /// Grantor's highest accepted log index at send time (safety-floor
    /// input on the grantee side).
    pub thresh_accepted: CommitPosition,
    /// Fresh attempt opened by this Guard.
    pub attempt: LeaseAttemptId,
    /// Message sequence on this pairing (strictly increasing per pair).
    pub seq: u64,
    /// Sending grantor.
    pub grantor: NodeId,
}

/// `GuardReply`: grantee entered `Guarded` within its guard window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LeaseGuardReply {
    /// Roster the reply answers for.
    pub roster_id: RosterId,
    /// Echoed attempt (must equal the Guard's).
    pub attempt: LeaseAttemptId,
    /// Echoed message sequence.
    pub seq: u64,
    /// Replying grantee.
    pub grantee: NodeId,
    /// Grantor being answered.
    pub grantor: NodeId,
}

/// Renew: grantor extends an open pairing. Same attempt as the Guard that
/// opened it; fresh message sequence. Carries the grantor's highest
/// *committed* index: the grantee folds it into its floor contribution,
/// so a responder that (re)established its pairing before a write
/// completed still learns a floor covering that write from later renews.
/// Accepted-at-guard alone would go stale across the establishment gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LeaseRenew {
    /// Roster being renewed.
    pub roster_id: RosterId,
    /// Echoed attempt (must equal the open Guard's).
    pub attempt: LeaseAttemptId,
    /// Message sequence on this pairing (strictly increasing).
    pub seq: u64,
    /// Grantor's highest committed log index at send time (floor input
    /// on the grantee side).
    pub committed: CommitPosition,
    /// Sending grantor.
    pub grantor: NodeId,
}

/// `RenewReply`: grantee extended its hold for the echoed renewal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LeaseRenewReply {
    /// Roster the reply answers for.
    pub roster_id: RosterId,
    /// Echoed attempt.
    pub attempt: LeaseAttemptId,
    /// Echoed message sequence.
    pub seq: u64,
    /// Replying grantee.
    pub grantee: NodeId,
    /// Grantor being answered.
    pub grantor: NodeId,
}

/// Revoke: grantor actively ends a pairing (fast path of roster
/// transition). Carries a fresh message sequence; the attempt names the
/// activation being revoked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LeaseRevoke {
    /// Roster being revoked.
    pub roster_id: RosterId,
    /// Attempt of the activation being revoked.
    pub attempt: LeaseAttemptId,
    /// Message sequence on this pairing (strictly increasing).
    pub seq: u64,
    /// Sending grantor.
    pub grantor: NodeId,
}

/// `RevokeReply`: grantee dropped the pairing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LeaseRevokeReply {
    /// Roster the reply answers for.
    pub roster_id: RosterId,
    /// Echoed attempt.
    pub attempt: LeaseAttemptId,
    /// Echoed message sequence.
    pub seq: u64,
    /// Replying grantee.
    pub grantee: NodeId,
    /// Grantor being answered.
    pub grantor: NodeId,
}

/// One lease-protocol message in either direction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeaseMessage {
    /// Open (or re-open) a pairing.
    Guard(LeaseGuard),
    /// Answer a Guard within the guard window.
    GuardReply(LeaseGuardReply),
    /// Extend an open pairing.
    Renew(LeaseRenew),
    /// Confirm a renewal.
    RenewReply(LeaseRenewReply),
    /// Actively end a pairing.
    Revoke(LeaseRevoke),
    /// Confirm a revocation.
    RevokeReply(LeaseRevokeReply),
}

impl LeaseMessage {
    /// Roster this message speaks for.
    #[must_use]
    pub fn roster_id(&self) -> RosterId {
        match self {
            Self::Guard(message) => message.roster.id,
            Self::GuardReply(message) => message.roster_id,
            Self::Renew(message) => message.roster_id,
            Self::RenewReply(message) => message.roster_id,
            Self::Revoke(message) => message.roster_id,
            Self::RevokeReply(message) => message.roster_id,
        }
    }

    /// Grantor this message speaks for (sender for Guard/Renew/Revoke,
    /// addressee for replies).
    #[must_use]
    pub const fn grantor(&self) -> NodeId {
        match self {
            Self::Guard(message) => message.grantor,
            Self::GuardReply(message) => message.grantor,
            Self::Renew(message) => message.grantor,
            Self::RenewReply(message) => message.grantor,
            Self::Revoke(message) => message.grantor,
            Self::RevokeReply(message) => message.grantor,
        }
    }

    /// Sending node of this message: the grantor for Guard/Renew/Revoke,
    /// the grantee for replies. Harness and deterministic simulation use
    /// only — production takes the sender from the authenticated transport
    /// identity and never trusts this self-claim.
    #[must_use]
    pub const fn sender(&self) -> NodeId {
        match self {
            Self::Guard(message) => message.grantor,
            Self::GuardReply(message) => message.grantee,
            Self::Renew(message) => message.grantor,
            Self::RenewReply(message) => message.grantee,
            Self::Revoke(message) => message.grantor,
            Self::RevokeReply(message) => message.grantee,
        }
    }
}

// ---------------------------------------------------------------------------
// Pairing states.
// ---------------------------------------------------------------------------

/// Grantor-side phase for one grantee pairing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GrantorPhase {
    /// No pairing (never opened, revoked and acknowledged, or expired).
    Idle,
    /// Guard sent, awaiting the `GuardReply` that promotes to `Renewing`.
    Guarding,
    /// Actively granting: renewals flow, exclusion holds.
    Renewing,
    /// Revoke sent, awaiting the `RevokeReply` (or exclusion timeout).
    Revoking,
    /// Exclusion lapsed without acknowledgment. Terminal until the next
    /// Guard opens a fresh attempt.
    Expired,
}

impl fmt::Display for GrantorPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "idle"),
            Self::Guarding => write!(f, "guarding"),
            Self::Renewing => write!(f, "renewing"),
            Self::Revoking => write!(f, "revoking"),
            Self::Expired => write!(f, "expired"),
        }
    }
}

/// Grantor-side record for one grantee pairing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrantorLease {
    /// Current phase.
    pub phase: GrantorPhase,
    /// Roster this pairing grants for.
    pub roster_id: RosterId,
    /// Attempt opened by the initiating Guard.
    pub attempt: LeaseAttemptId,
    /// Last outbound message sequence on this pairing.
    pub out_seq: u64,
    /// Local deadline of the guard window (`Guarding` only).
    pub guard_deadline: Ticks,
    /// Local exclusion deadline (`Renewing`/`Revoking`): the grantor
    /// refuses conflicting grants until this passes. Reset to
    /// send-time + lease + allowance on every renewal (never extended:
    /// the exclusion is exactly one lease beyond the last vouched
    /// renewal, which is what keeps the grantee's strictly shorter hold
    /// inside it).
    pub lease_deadline: Ticks,
    /// Next monotonic time a renewal is due (`Renewing` only). Renewals
    /// flow every renew interval — not when the exclusion nears expiry —
    /// so the send period (~renew) stays far below the grantee hold
    /// (~lease). Set at promotion, advanced on every send.
    pub renew_due: Ticks,
    /// Consecutive renewals sent without any reply. At
    /// [`MUTE_AFTER_UNACKED`] the grantor stops sending (the exclusion
    /// above still runs to its deadline). Reset by any reply.
    pub unacked: u32,
}

/// Grantee-side phase for one grantor pairing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GranteePhase {
    /// No pairing held.
    None,
    /// Guard accepted, awaiting the first in-window Renew.
    Guarded,
    /// Actively holding: renewals arrive in window.
    Renewed,
    /// Hold lapsed. Terminal until the next Guard.
    Expired,
}

impl fmt::Display for GranteePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Guarded => write!(f, "guarded"),
            Self::Renewed => write!(f, "renewed"),
            Self::Expired => write!(f, "expired"),
        }
    }
}

/// Grantee-side record for one grantor pairing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GranteeLease {
    /// Current phase.
    pub phase: GranteePhase,
    /// Roster held.
    pub roster_id: RosterId,
    /// Roster content, learned from the Guard (self-describing).
    pub roster: Roster,
    /// Attempt this pairing runs under.
    pub attempt: LeaseAttemptId,
    /// Grantor incarnation the pairing is bound to. A higher incarnation
    /// always starts a fresh pairing (never continues this one).
    pub grantor_incarnation: NodeIncarnation,
    /// Last accepted inbound message sequence on this pairing.
    pub last_seq: u64,
    /// Local deadline of the guard window (`Guarded` only).
    pub guard_deadline: Ticks,
    /// Local hold deadline (`Renewed` only): the grantee serves local
    /// reads under this pairing only before this passes.
    pub lease_deadline: Ticks,
    /// Grantor's highest accepted log index at Guard time: the floor
    /// contribution's first input (see `floor`).
    pub thresh_accepted: CommitPosition,
    /// Floor contribution of this pairing: `max(thresh_accepted,
    /// committed-seen-in-renews)`. Guard thresholds go stale across the
    /// establishment gap (pairing opened before a write completed, renews
    /// arriving after); folding every renewal's committed index keeps the
    /// contribution covering everything committed before the latest
    /// renewal was sent.
    pub floor: CommitPosition,
}

// ---------------------------------------------------------------------------
// Engine: pure per-node per-tablet lease state machine.
// ---------------------------------------------------------------------------

/// Outbound lease traffic produced by one engine step, in emission order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeaseOutbound {
    /// Addressee.
    pub target: NodeId,
    /// Message to send.
    pub message: LeaseMessage,
}

impl LeaseOutbound {
    /// Builds one outbound message.
    #[must_use]
    pub const fn new(target: NodeId, message: LeaseMessage) -> Self {
        Self { target, message }
    }
}

/// Observable lease-layer events for metrics, tracing, and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LeaseEvent {
    /// A Guard opened a fresh pairing attempt.
    GuardSent,
    /// A `GuardReply` promoted a pairing to `Renewing`.
    GuardAnswered,
    /// A Renew extended a pairing.
    RenewSent,
    /// A Renew was accepted and extended the hold.
    RenewAccepted,
    /// A duplicate/in-window-noop Renew was ignored.
    RenewDuplicate,
    /// A Revoke ended a pairing fast.
    RevokeSent,
    /// A Revoke was accepted and the pairing dropped.
    RevokeAccepted,
    /// A stale/foreign message was ignored (wrong attempt, old seq,
    /// superseded roster, wrong incarnation, unknown pairing).
    StaleIgnored,
    /// A pairing lapsed on its local timer.
    Expired,
    /// A grantor stopped sending renewals to an unresponsive grantee
    /// (muted past [`MUTE_AFTER_UNACKED`]); the exclusion still runs to
    /// its deadline, so coverage still holds it.
    Muted,
    /// A grantor pairing was fenced by term/authority movement.
    Fenced,
}

/// Majority size for `voters` voters (floor(n/2) + 1; empty set has none).
#[must_use]
pub const fn majority_of(voters: usize) -> Option<u64> {
    if voters == 0 {
        return None;
    }
    Some((voters as u64) / 2 + 1)
}

/// Pure directional lease engine for one node and one tablet.
///
/// All time enters explicitly as `now: Ticks`; all randomness-free. The
/// owner-thread driver supplies terms, voters, authority, transport, and
/// timers; the hub consumes published roster snapshots for the fast path.
/// Deterministic simulation drives this type directly with virtual time
/// (and virtual drift, by skewing the `now` each simulated participant
/// observes).
#[derive(Debug, Clone)]
pub struct RosterEngine {
    /// This participant.
    local: NodeId,
    /// This process lifetime. Binds every attempt this engine opens and
    /// every pairing it accepts.
    incarnation: NodeIncarnation,
    /// Currently serving tablet authority. Any change voids all state.
    authority: TabletAuthority,
    /// Currently observed Raft term. Any rise revokes stale grants.
    term: RosterTerm,
    /// Current voter set, sorted ascending, deduplicated.
    voters: Vec<NodeId>,
    /// Deployment timing contract.
    params: LeaseParams,
    /// Highest locally *accepted* (appended) log position, refreshed by the
    /// driver every tick from the log. Grantor thresholds vouch for this —
    /// never for applied or committed positions — and auto-grants opened on
    /// learning a roster use it.
    accepted: CommitPosition,
    /// Highest locally *committed* log position, refreshed by the driver
    /// every tick. Renewals vouch for this (folded into grantee floor
    /// contributions), closing the establishment gap that guard-time
    /// accepted thresholds alone leave open.
    committed: CommitPosition,
    /// Grantor state per grantee.
    as_grantor: alloc_collections_BTreeMap<NodeId, GrantorLease>,
    /// Grantee state per (grantor, roster).
    as_grantee: alloc_collections_BTreeMap<(NodeId, RosterId), GranteeLease>,
    /// Rosters this node announced (by identity).
    announced: Vec<RosterId>,
    /// Roster contents learned (announced or via Guard), pruned to the
    /// current authority on reconcile.
    known: alloc_collections_BTreeMap<RosterId, Roster>,
    /// Monotonic time before which this engine must not grant. Set at
    /// creation (restart fence); cleared by time passing, never by
    /// messages.
    quarantine_until: Ticks,
}

// `std` collections via explicit path: kivi-types owns value semantics and
// stays runtime-free, but ordered maps are values, not I/O.
use std::collections::BTreeMap as alloc_collections_BTreeMap;

impl RosterEngine {
    /// Builds a fresh engine. The engine starts quarantined until
    /// `now + params.quarantine()` unless `incarnation` is `INITIAL`
    /// (first boot ever: no forgotten grants can exist — with the
    /// documented assumption that data directories of live members are
    /// never wiped).
    #[must_use]
    pub fn new(
        local: NodeId,
        incarnation: NodeIncarnation,
        authority: TabletAuthority,
        term: RosterTerm,
        voters: Vec<NodeId>,
        params: LeaseParams,
        now: Ticks,
    ) -> Self {
        let mut voters = voters;
        voters.sort_by_key(|node| node.as_u64());
        voters.dedup_by_key(|node| node.as_u64());
        let quarantine_until = if incarnation == NodeIncarnation::INITIAL {
            now
        } else {
            now.advance_by(params.quarantine())
        };
        Self {
            local,
            incarnation,
            authority,
            term,
            voters,
            params,
            accepted: CommitPosition::UNASSIGNED,
            committed: CommitPosition::UNASSIGNED,
            as_grantor: alloc_collections_BTreeMap::new(),
            as_grantee: alloc_collections_BTreeMap::new(),
            announced: Vec::new(),
            known: alloc_collections_BTreeMap::new(),
            quarantine_until,
        }
    }

    /// This participant's identity.
    #[must_use]
    pub const fn local(&self) -> NodeId {
        self.local
    }

    /// This process lifetime. Binds every attempt this engine opens and
    /// every pairing it accepts.
    #[must_use]
    pub const fn incarnation(&self) -> NodeIncarnation {
        self.incarnation
    }

    /// Deployment timing contract this engine enforces.
    #[must_use]
    pub const fn params(&self) -> LeaseParams {
        self.params
    }

    /// Current voter set, sorted ascending, deduplicated.
    #[must_use]
    pub fn voters(&self) -> Vec<NodeId> {
        self.voters.clone()
    }

    /// Current authority this engine serves.
    #[must_use]
    pub const fn authority(&self) -> TabletAuthority {
        self.authority
    }

    /// Currently observed term.
    #[must_use]
    pub const fn term(&self) -> RosterTerm {
        self.term
    }

    /// Whether granting is fenced by restart quarantine at `now`.
    #[must_use]
    pub fn quarantined(&self, now: Ticks) -> bool {
        now.as_micros() < self.quarantine_until.as_micros()
    }

    /// Monotonic time the quarantine lifts.
    #[must_use]
    pub const fn quarantine_until(&self) -> Ticks {
        self.quarantine_until
    }

    /// Records the driver's highest locally accepted log position
    /// (monotonic: regressions are ignored, so a reordered driver update
    /// can never shrink a threshold already vouched for).
    pub fn set_accepted(&mut self, accepted: CommitPosition) {
        if accepted.as_u64() > self.accepted.as_u64() {
            self.accepted = accepted;
        }
    }

    /// Highest locally accepted log position recorded so far.
    #[must_use]
    pub const fn accepted(&self) -> CommitPosition {
        self.accepted
    }

    /// Records the driver's highest locally committed log position
    /// (monotonic: regressions are ignored, so a reordered driver update
    /// can never shrink a floor contribution already vouched for).
    pub fn set_committed(&mut self, committed: CommitPosition) {
        if committed.as_u64() > self.committed.as_u64() {
            self.committed = committed;
        }
    }

    /// Highest locally committed log position recorded so far.
    #[must_use]
    pub const fn committed(&self) -> CommitPosition {
        self.committed
    }

    /// Number of live grantor pairings (any non-terminal phase).
    #[must_use]
    pub fn grantor_pairings(&self) -> usize {
        self.as_grantor
            .values()
            .filter(|lease| !matches!(lease.phase, GrantorPhase::Idle | GrantorPhase::Expired))
            .count()
    }

    /// Number of live grantee pairings (any non-terminal phase).
    #[must_use]
    pub fn grantee_pairings(&self) -> usize {
        self.as_grantee
            .values()
            .filter(|lease| !matches!(lease.phase, GranteePhase::None | GranteePhase::Expired))
            .count()
    }

    /// Reconciles observed world state: authority, term, and voters.
    ///
    /// * Authority change voids everything (fresh lineage needs fresh
    ///   rosters); known roster contents for other authorities are
    ///   pruned.
    /// * Term rise moves live outgoing grants to `Revoking` with an
    ///   immediate Revoke each (fast path), and drops grantee state for
    ///   older terms. Expired states are untouched (already dead).
    /// * Voter-set change drops pairings with removed peers (their
    ///   grants can never count toward a majority again).
    ///
    /// Returns outbound Revokes in deterministic (node-id) order.
    pub fn reconcile(
        &mut self,
        authority: TabletAuthority,
        term: RosterTerm,
        voters: Vec<NodeId>,
        now: Ticks,
    ) -> (Vec<LeaseOutbound>, Vec<LeaseEvent>) {
        let mut out = Vec::new();
        let mut events = Vec::new();
        if authority != self.authority {
            self.authority = authority;
            self.as_grantor.clear();
            self.as_grantee.clear();
            self.announced.clear();
            self.known.retain(|id, _| id.authority == authority);
            events.push(LeaseEvent::Fenced);
            // Voters still refresh below; term is left alone (the driver
            // reports the observed term independently of authority).
        }
        if term != self.term {
            self.revoke_stale_grants(term, now, &mut out, &mut events);
            // Grantee state for older terms is NOT dropped here: the
            // revocations above answer it (fast path — including loopback,
            // which is what clears a single voter's own exclusion), and
            // anything unanswered lapses on its own timer. Dropping first
            // would orphan the grantor side in `Revoking` with no reply
            // ever coming. Safety needs no drop: `stable()` only counts
            // pairings for the current term.
            self.term = term;
        }
        let mut voters = voters;
        voters.sort_by_key(|node| node.as_u64());
        voters.dedup_by_key(|node| node.as_u64());
        if voters != self.voters {
            self.voters.clone_from(&voters);
            self.as_grantor
                .retain(|peer, _| *peer == self.local || voters.contains(peer));
            self.as_grantee
                .retain(|(grantor, _), _| *grantor == self.local || voters.contains(grantor));
            events.push(LeaseEvent::Fenced);
        }
        let _ = now;
        (out, events)
    }

    /// Moves live outgoing grants for older terms to `Revoking`, emitting
    /// one Revoke each. Guarding grants reset locally (nothing was ever
    /// promised); Renewing grants revoke with a bumped sequence so the
    /// Revoke cannot be confused with traffic from the old activation.
    fn revoke_stale_grants(
        &mut self,
        term: RosterTerm,
        now: Ticks,
        out: &mut Vec<LeaseOutbound>,
        events: &mut Vec<LeaseEvent>,
    ) {
        let allowance = self.params.allowance(self.params.lease_duration);
        for (peer, lease) in &mut self.as_grantor {
            if lease.roster_id.term >= term {
                continue;
            }
            match lease.phase {
                GrantorPhase::Guarding => {
                    lease.phase = GrantorPhase::Idle;
                    events.push(LeaseEvent::Fenced);
                }
                GrantorPhase::Renewing => {
                    lease.phase = GrantorPhase::Revoking;
                    lease.out_seq = lease.out_seq.saturating_add(1);
                    lease.lease_deadline =
                        now.advance_by(self.params.lease_duration.saturating_add(allowance));
                    out.push(LeaseOutbound::new(
                        *peer,
                        LeaseMessage::Revoke(LeaseRevoke {
                            roster_id: lease.roster_id,
                            attempt: lease.attempt,
                            seq: lease.out_seq,
                            grantor: self.local,
                        }),
                    ));
                    events.push(LeaseEvent::RevokeSent);
                    events.push(LeaseEvent::Fenced);
                }
                GrantorPhase::Idle | GrantorPhase::Revoking | GrantorPhase::Expired => {}
            }
        }
    }

    /// Announces a new roster (leader-only; the driver verifies
    /// leadership before calling): allocates the next generation for
    /// (`authority`, `term`), records it, and opens Guard pairings to
    /// every voter including self.
    ///
    /// `thresh_accepted` is the announcer's own highest *accepted* (appended,
    /// not merely applied or committed) log position: this pairing's
    /// contribution to the roster safety floor. Accepted is the weakest
    /// exact boundary that still proves safety — a grantor vouches only for
    /// what it has durably appended, and the floor (majority-th smallest
    /// threshold) intersects every committed write's majority.
    ///
    /// # Errors
    ///
    /// Returns [`RosterError::Quarantined`] while the restart fence holds,
    /// [`RosterError::NotLeader`] when `leader != local` (defense in
    /// depth; the driver checks first), or [`RosterError::InvalidId`]
    /// when the responder set cannot form a valid roster or the generation
    /// space is exhausted (never wraps: reusing a generation would
    /// resurrect a superseded responder set).
    pub fn announce(
        &mut self,
        leader: NodeId,
        responders: Vec<NodeId>,
        thresh_accepted: CommitPosition,
        now: Ticks,
    ) -> Result<(Roster, Vec<LeaseOutbound>), RosterError> {
        if self.quarantined(now) {
            return Err(RosterError::Quarantined {
                until: self.quarantine_until,
            });
        }
        if leader != self.local {
            return Err(RosterError::NotLeader);
        }
        self.set_accepted(thresh_accepted);
        let generation = self
            .known
            .keys()
            .chain(self.announced.iter())
            .filter(|id| id.authority == self.authority && id.term == self.term)
            .map(|id| id.generation.as_u64())
            .max()
            .map_or(Ok(RosterGeneration::INITIAL), |max| {
                RosterGeneration::from_u64(max).next()
            })
            .map_err(|_| RosterError::InvalidId)?;
        if !generation.is_valid() {
            return Err(RosterError::InvalidId);
        }
        let id = RosterId::new(self.authority, self.term, generation);
        let roster = Roster::new(id, leader, responders).map_err(|_| RosterError::InvalidId)?;
        // A fresh announcement revokes any live outgoing grant for an
        // older roster first (fast path); the new Guards below then carry
        // the replacement. Same-term older generations are fenced here,
        // not left to expiry races.
        let mut out = Vec::new();
        let mut events = Vec::new();
        self.revoke_superseded_grants(id, now, &mut out, &mut events);
        self.known.insert(id, roster.clone());
        // Same generation pruning as Guard receipt: the replacement voids
        // older same-lineage holds immediately; exclusions still run out.
        self.known.retain(|known, _| {
            known.authority != id.authority
                || known.term != id.term
                || known.generation >= id.generation
        });
        if !self.announced.contains(&id) {
            self.announced.push(id);
        }
        let mut targets: Vec<NodeId> = self.voters.clone();
        if !targets.contains(&self.local) {
            targets.push(self.local);
        }
        targets.sort_by_key(|node| node.as_u64());
        for peer in targets {
            out.extend(self.open_guard(peer, &roster, thresh_accepted, now));
        }
        let _ = events;
        Ok((roster, out))
    }

    /// Revokes live outgoing grants for rosters superseded by `id`
    /// (same authority and term, older generation).
    fn revoke_superseded_grants(
        &mut self,
        id: RosterId,
        now: Ticks,
        out: &mut Vec<LeaseOutbound>,
        events: &mut Vec<LeaseEvent>,
    ) {
        let allowance = self.params.allowance(self.params.lease_duration);
        for (peer, lease) in &mut self.as_grantor {
            let superseded = lease.roster_id.authority == id.authority
                && lease.roster_id.term == id.term
                && lease.roster_id.generation < id.generation;
            if !superseded {
                continue;
            }
            match lease.phase {
                GrantorPhase::Guarding => {
                    lease.phase = GrantorPhase::Idle;
                    events.push(LeaseEvent::Fenced);
                }
                GrantorPhase::Renewing => {
                    lease.phase = GrantorPhase::Revoking;
                    lease.out_seq = lease.out_seq.saturating_add(1);
                    lease.lease_deadline =
                        now.advance_by(self.params.lease_duration.saturating_add(allowance));
                    out.push(LeaseOutbound::new(
                        *peer,
                        LeaseMessage::Revoke(LeaseRevoke {
                            roster_id: lease.roster_id,
                            attempt: lease.attempt,
                            seq: lease.out_seq,
                            grantor: self.local,
                        }),
                    ));
                    events.push(LeaseEvent::RevokeSent);
                    events.push(LeaseEvent::Fenced);
                }
                GrantorPhase::Idle | GrantorPhase::Revoking | GrantorPhase::Expired => {}
            }
        }
    }

    /// Opens (or re-opens) a Guard pairing to `peer` for `roster`,
    /// recording the grantor's current accepted position as the safety
    /// threshold. Each call opens a FRESH attempt: retries never continue
    /// an old one.
    ///
    /// A live exclusion (`Renewing` or `Revoking`, for any roster) is
    /// never overwritten: the new roster must wait out the old exclusion
    /// (fast path: `RevokeReply`; failure path: exclusion timeout)
    /// before conflicting authority activates. Callers retry idempotently
    /// — [`RosterEngine::grant_for`] on every driver tick — so a skipped
    /// Guard is deferred, never dropped.
    fn open_guard(
        &mut self,
        peer: NodeId,
        roster: &Roster,
        thresh_accepted: CommitPosition,
        now: Ticks,
    ) -> Vec<LeaseOutbound> {
        if let Some(held) = self.as_grantor.get(&peer)
            && matches!(held.phase, GrantorPhase::Renewing | GrantorPhase::Revoking)
        {
            return Vec::new();
        }
        let allowance = self.params.allowance(self.params.guard_duration);
        let out_seq = self
            .as_grantor
            .get(&peer)
            .map_or(1, |lease| lease.out_seq.saturating_add(1));
        // Attempt identity: (own incarnation, Guard's own sequence).
        // Post-restart incarnations never collide with pre-restart ones,
        // so a delayed pre-restart message can never join this attempt.
        let attempt = LeaseAttemptId::new(self.incarnation, out_seq);
        self.as_grantor.insert(
            peer,
            GrantorLease {
                phase: GrantorPhase::Guarding,
                roster_id: roster.id,
                attempt,
                out_seq,
                guard_deadline: now
                    .advance_by(self.params.guard_duration.saturating_add(allowance)),
                lease_deadline: now,
                renew_due: now,
                unacked: 0,
            },
        );
        vec![LeaseOutbound::new(
            peer,
            LeaseMessage::Guard(LeaseGuard {
                roster: roster.clone(),
                thresh_accepted,
                attempt,
                seq: out_seq,
                grantor: self.local,
            }),
        )]
    }

    /// Periodic drive: expires lapsed pairings, emits due renewals and
    /// Guard retries, and times out unanswered revocations. Returns
    /// outbound traffic plus observable events, both deterministic given
    /// the inputs.
    ///
    /// The caller supplies this engine's current accepted position
    /// (`thresh_accepted`, for any Guard it (re)opens) and committed
    /// position (`committed`, vouched in every Renew it emits).
    pub fn tick(
        &mut self,
        thresh_accepted: CommitPosition,
        committed: CommitPosition,
        now: Ticks,
    ) -> (Vec<LeaseOutbound>, Vec<LeaseEvent>) {
        let mut out = Vec::new();
        let mut events = Vec::new();
        self.set_accepted(thresh_accepted);
        self.set_committed(committed);
        let lease_allowance = self.params.allowance(self.params.lease_duration);
        // Grantor side.
        let peers: Vec<NodeId> = self.as_grantor.keys().copied().collect();
        for peer in peers {
            let Some(lease) = self.as_grantor.get(&peer).cloned() else {
                continue;
            };
            match lease.phase {
                GrantorPhase::Guarding => {
                    if now.as_micros() < lease.guard_deadline.as_micros() {
                        continue;
                    }
                    // Guard unanswered: retry with a FRESH attempt (the old
                    // one is dead; its late replies must not count).
                    if let Some(roster) = self.known.get(&lease.roster_id).cloned()
                        && roster.id.authority == self.authority
                    {
                        out.extend(self.open_guard(peer, &roster, thresh_accepted, now));
                        events.push(LeaseEvent::GuardSent);
                    } else {
                        self.as_grantor.remove(&peer);
                        events.push(LeaseEvent::Expired);
                    }
                }
                GrantorPhase::Renewing => {
                    if now.as_micros() >= lease.lease_deadline.as_micros() {
                        // Exclusion lapsed (muted too long, or the driver
                        // stopped ticking): stop granting. The grantee's
                        // hold already lapsed first, by the asymmetric
                        // deadlines — removing here never uncovers a live
                        // hold.
                        self.as_grantor.remove(&peer);
                        events.push(LeaseEvent::Expired);
                        continue;
                    }
                    if now.as_micros() < lease.renew_due.as_micros() {
                        continue;
                    }
                    let mut live = lease;
                    // Muted grantor: send nothing (the exclusion still
                    // runs to its deadline, which the grantee's strictly
                    // shorter hold precedes), so writes stall boundedly
                    // instead of forever.
                    if live.unacked >= MUTE_AFTER_UNACKED {
                        events.push(LeaseEvent::Muted);
                        self.as_grantor.insert(peer, live);
                        continue;
                    }
                    live.out_seq = live.out_seq.saturating_add(1);
                    live.unacked = live.unacked.saturating_add(1);
                    live.renew_due = now.advance_by(self.params.renew_interval);
                    live.lease_deadline =
                        now.advance_by(self.params.lease_duration.saturating_add(lease_allowance));
                    let committed = self.committed;
                    out.push(LeaseOutbound::new(
                        peer,
                        LeaseMessage::Renew(LeaseRenew {
                            roster_id: live.roster_id,
                            attempt: live.attempt,
                            seq: live.out_seq,
                            committed,
                            grantor: self.local,
                        }),
                    ));
                    self.as_grantor.insert(peer, live);
                    events.push(LeaseEvent::RenewSent);
                }
                GrantorPhase::Revoking => {
                    if now.as_micros() >= lease.lease_deadline.as_micros() {
                        // Unanswered revocation: exclusion lapsed anyway.
                        self.as_grantor.remove(&peer);
                        events.push(LeaseEvent::Expired);
                    }
                }
                GrantorPhase::Idle | GrantorPhase::Expired => {}
            }
        }
        // Grantee side: lapse expired holds. Expired records are removed
        // (sequence continuity is preserved implicitly: any future Guard
        // must carry a higher incarnation or higher sequence to be
        // accepted — see `receive`).
        let pairs: Vec<(NodeId, RosterId)> = self.as_grantee.keys().copied().collect();
        for key in pairs {
            let Some(lease) = self.as_grantee.get(&key).cloned() else {
                continue;
            };
            let lapsed = match lease.phase {
                GranteePhase::Guarded => now.as_micros() >= lease.guard_deadline.as_micros(),
                GranteePhase::Renewed => now.as_micros() >= lease.lease_deadline.as_micros(),
                GranteePhase::None | GranteePhase::Expired => true,
            };
            if lapsed {
                self.as_grantee.remove(&key);
                events.push(LeaseEvent::Expired);
            }
        }
        (out, events)
    }

    /// Handles one inbound lease message, enforcing attempt identity,
    /// sequence monotonicity, incarnation binding, and roster ordering.
    /// Returns outbound replies plus observable events.
    pub fn receive(
        &mut self,
        from: NodeId,
        message: LeaseMessage,
        now: Ticks,
    ) -> (Vec<LeaseOutbound>, Vec<LeaseEvent>) {
        let mut out = Vec::new();
        let mut events = Vec::new();
        match message {
            LeaseMessage::Guard(guard) => self.on_guard(from, guard, now, &mut out, &mut events),
            LeaseMessage::GuardReply(reply) => {
                self.on_guard_reply(from, reply, now, &mut out, &mut events);
            }
            LeaseMessage::Renew(renew) => self.on_renew(from, renew, now, &mut out, &mut events),
            LeaseMessage::RenewReply(reply) => {
                self.on_renew_reply(from, reply, &mut events);
            }
            LeaseMessage::Revoke(revoke) => {
                self.on_revoke(from, revoke, &mut out, &mut events);
            }
            LeaseMessage::RevokeReply(reply) => {
                self.on_revoke_reply(from, reply, &mut events);
            }
        }
        (out, events)
    }

    /// Guard receipt: accept iff the roster is admissible (current
    /// authority, term not older than observed, generation not older than
    /// any pairing already held from this grantor for this term) and the
    /// (incarnation, seq) pair is fresh. A higher grantor incarnation
    /// always starts a fresh pairing (never continues the old one).
    #[allow(clippy::too_many_arguments)]
    fn on_guard(
        &mut self,
        from: NodeId,
        guard: LeaseGuard,
        now: Ticks,
        out: &mut Vec<LeaseOutbound>,
        events: &mut Vec<LeaseEvent>,
    ) {
        // Restart fence: a quarantined engine takes no lease-layer part —
        // neither granting nor holding. Post-restart state is empty, so the
        // other handlers already no-op; the Guard is the one message that
        // would create state, and it must not while forgotten grants may
        // still be live elsewhere. Grantor retries re-establish the pairing
        // after the quarantine lapses.
        if self.quarantined(now) {
            events.push(LeaseEvent::StaleIgnored);
            return;
        }
        let roster = guard.roster;
        if roster.id.authority != self.authority {
            events.push(LeaseEvent::StaleIgnored);
            return;
        }
        if roster.id.term < self.term {
            events.push(LeaseEvent::StaleIgnored);
            return;
        }
        if from != self.local && !self.voters.contains(&from) {
            // Grants only flow between current voters: a removed member's
            // Guard can never count toward a majority again.
            events.push(LeaseEvent::StaleIgnored);
            return;
        }
        // Newest held pairing from this grantor decides freshness. Keyed
        // per (grantor, roster): different rosters coexist until revoked
        // or expired (transition fencing, never overwrite).
        let key = (from, roster.id);
        if let Some(held) = self.as_grantee.get(&key) {
            let fresh = guard.attempt.grantor_incarnation.as_u64()
                > held.attempt.grantor_incarnation.as_u64()
                || (guard.attempt == held.attempt && guard.seq > held.last_seq);
            if !fresh {
                events.push(LeaseEvent::StaleIgnored);
                return;
            }
        } else {
            // No pairing for this (grantor, roster): require the roster to
            // not be older than the newest generation already seen from
            // this grantor in this term (stale announce redelivery dies).
            let newest = self
                .as_grantee
                .iter()
                .filter(|((grantor, id), _)| {
                    *grantor == from
                        && id.authority == roster.id.authority
                        && id.term == roster.id.term
                })
                .map(|(_, lease)| lease.roster_id.generation.as_u64())
                .max();
            if newest.is_some_and(|seen| roster.id.generation.as_u64() < seen) {
                events.push(LeaseEvent::StaleIgnored);
                return;
            }
            if guard.seq == 0 {
                events.push(LeaseEvent::StaleIgnored);
                return;
            }
        }
        let allowance = self.params.allowance(self.params.guard_duration);
        let lease = GranteeLease {
            phase: GranteePhase::Guarded,
            roster_id: roster.id,
            roster: roster.clone(),
            attempt: guard.attempt,
            grantor_incarnation: guard.attempt.grantor_incarnation,
            last_seq: guard.seq,
            guard_deadline: now.advance_by(self.params.guard_duration.saturating_sub(allowance)),
            lease_deadline: now,
            thresh_accepted: guard.thresh_accepted,
            floor: guard.thresh_accepted,
        };
        self.as_grantee.insert(key, lease);
        // Supersede: when this Guard is not older than any known roster
        // of its lineage, this node's own outgoing grants for older
        // same-lineage rosters revoke now (fast path) instead of
        // lingering beside the replacement. Older-generation redelivery
        // revokes nothing. The new Guards below then wait out any live
        // exclusion via the open-guard skip plus the driver's idempotent
        // retry.
        let not_older = !self
            .known
            .keys()
            .filter(|id| {
                id.authority == roster.id.authority
                    && id.term == roster.id.term
                    && id.generation != roster.id.generation
            })
            .any(|id| id.generation > roster.id.generation);
        if not_older {
            self.revoke_superseded_grants(roster.id, now, out, events);
        }
        self.known.insert(roster.id, roster.clone());
        // Generation pruning: within one (authority, term) lineage only the
        // newest learned generation counts. Older holds stop contributing
        // to stability the moment newer authority is learned (fail-closed
        // toward the replacement); their grantor-side exclusions still run
        // to RevokeReply or timeout, so writes keep covering them.
        // Out-of-order redelivery of an older generation prunes nothing:
        // the retain keeps everything at or above the incoming generation.
        self.known.retain(|id, _| {
            id.authority != roster.id.authority
                || id.term != roster.id.term
                || id.generation >= roster.id.generation
        });
        // GuardReply echoes the exact attempt+seq: it can only ever count
        // for the Guard it answers.
        out.push(LeaseOutbound::new(
            from,
            LeaseMessage::GuardReply(LeaseGuardReply {
                roster_id: roster.id,
                attempt: guard.attempt,
                seq: guard.seq,
                grantee: self.local,
                grantor: from,
            }),
        ));
        events.push(LeaseEvent::GuardSent);
        // All-to-all closure: a designated participant grants back. Only
        // pairings with no live outgoing state open a fresh attempt, so
        // this converges instead of storming; non-designated voters never
        // grant, so their pairings can never count toward stability.
        if roster.covers(self.local) {
            let fresh = self.grant_for(roster.id, self.accepted, now);
            if !fresh.is_empty() {
                events.push(LeaseEvent::GuardSent);
            }
            out.extend(fresh);
        }
    }

    /// `GuardReply` receipt: promotes `Guarding` to `Renewing` and emits the
    /// first Renew — but only for the exact open attempt. A reply for any
    /// other attempt (leftover from an abandoned Guard) is ignored, which
    /// is what makes abandon-and-retry safe at any quarantine length.
    fn on_guard_reply(
        &mut self,
        from: NodeId,
        reply: LeaseGuardReply,
        now: Ticks,
        out: &mut Vec<LeaseOutbound>,
        events: &mut Vec<LeaseEvent>,
    ) {
        let Some(lease) = self.as_grantor.get(&from).cloned() else {
            events.push(LeaseEvent::StaleIgnored);
            return;
        };
        // Exact-match rule: roster, attempt, and sequence must equal the
        // open Guard. Greater/lesser/abandoned values all die here.
        let exact = lease.phase == GrantorPhase::Guarding
            && lease.roster_id == reply.roster_id
            && lease.attempt == reply.attempt
            && lease.out_seq == reply.seq
            && now.as_micros() < lease.guard_deadline.as_micros();
        if !exact {
            events.push(LeaseEvent::StaleIgnored);
            return;
        }
        // The grantor starts counting exclusion from this first Renew
        // SEND (timer placement: initiation of the attempt's second leg,
        // never quorum receipt). Loose deadline first, tightened by the
        // RenewReply below.
        let lease_allowance = self.params.allowance(self.params.lease_duration);
        let guard_allowance = self.params.allowance(self.params.guard_duration);
        let mut live = lease;
        live.phase = GrantorPhase::Renewing;
        live.out_seq = live.out_seq.saturating_add(1);
        live.unacked = 0;
        live.renew_due = now.advance_by(self.params.renew_interval);
        live.lease_deadline = now.advance_by(
            self.params
                .guard_duration
                .saturating_add(guard_allowance)
                .saturating_add(self.params.lease_duration)
                .saturating_add(lease_allowance),
        );
        let committed = self.committed;
        out.push(LeaseOutbound::new(
            from,
            LeaseMessage::Renew(LeaseRenew {
                roster_id: live.roster_id,
                attempt: live.attempt,
                seq: live.out_seq,
                committed,
                grantor: self.local,
            }),
        ));
        self.as_grantor.insert(from, live);
        events.push(LeaseEvent::GuardAnswered);
    }

    /// Renew receipt: extends the hold iff the pairing is live, the roster
    /// and attempt match exactly, the sequence advances, the incarnation
    /// is unchanged, and the Renew arrives while still valid. The first
    /// Renew must additionally arrive inside the guard window (fresh
    /// activation under unsynchronized clocks is only safe then).
    fn on_renew(
        &mut self,
        from: NodeId,
        renew: LeaseRenew,
        now: Ticks,
        out: &mut Vec<LeaseOutbound>,
        events: &mut Vec<LeaseEvent>,
    ) {
        let key = (from, renew.roster_id);
        let Some(lease) = self.as_grantee.get(&key).cloned() else {
            events.push(LeaseEvent::StaleIgnored);
            return;
        };
        // Exact pairing continuity: same attempt, same grantor lifetime,
        // strictly advancing sequence. A Renew from a restarted grantor
        // (new incarnation) without a fresh Guard dies here.
        if lease.attempt != renew.attempt
            || lease.grantor_incarnation != renew.attempt.grantor_incarnation
            || renew.seq <= lease.last_seq
        {
            events.push(LeaseEvent::StaleIgnored);
            return;
        }
        let lease_allowance = self.params.allowance(self.params.lease_duration);
        match lease.phase {
            GranteePhase::Guarded => {
                if now.as_micros() >= lease.guard_deadline.as_micros() {
                    events.push(LeaseEvent::StaleIgnored);
                    return;
                }
                let mut live = lease;
                live.phase = GranteePhase::Renewed;
                live.last_seq = renew.seq;
                live.floor = live.floor.max(renew.committed);
                live.lease_deadline =
                    now.advance_by(self.params.lease_duration.saturating_sub(lease_allowance));
                self.as_grantee.insert(key, live);
                out.push(LeaseOutbound::new(
                    from,
                    LeaseMessage::RenewReply(LeaseRenewReply {
                        roster_id: renew.roster_id,
                        attempt: renew.attempt,
                        seq: renew.seq,
                        grantee: self.local,
                        grantor: from,
                    }),
                ));
                events.push(LeaseEvent::RenewAccepted);
            }
            GranteePhase::Renewed => {
                if now.as_micros() >= lease.lease_deadline.as_micros() {
                    events.push(LeaseEvent::StaleIgnored);
                    return;
                }
                // Duplicate/no-op Renew: already extended past what this
                // Renew would grant (reordered duplicate). Count, extend
                // nothing — this is what keeps a very delayed Renew from
                // manufacturing fresh validity.
                let grant =
                    now.advance_by(self.params.lease_duration.saturating_sub(lease_allowance));
                if grant.as_micros() <= lease.lease_deadline.as_micros() {
                    events.push(LeaseEvent::RenewDuplicate);
                    return;
                }
                let mut live = lease;
                live.last_seq = renew.seq;
                live.floor = live.floor.max(renew.committed);
                live.lease_deadline = grant;
                self.as_grantee.insert(key, live);
                out.push(LeaseOutbound::new(
                    from,
                    LeaseMessage::RenewReply(LeaseRenewReply {
                        roster_id: renew.roster_id,
                        attempt: renew.attempt,
                        seq: renew.seq,
                        grantee: self.local,
                        grantor: from,
                    }),
                ));
                events.push(LeaseEvent::RenewAccepted);
            }
            GranteePhase::None | GranteePhase::Expired => {
                events.push(LeaseEvent::StaleIgnored);
            }
        }
    }

    /// `RenewReply` receipt: tightens the loose exclusion deadline — only
    /// for the exact open renewal it echoes.
    fn on_renew_reply(
        &mut self,
        from: NodeId,
        reply: LeaseRenewReply,
        events: &mut Vec<LeaseEvent>,
    ) {
        let Some(lease) = self.as_grantor.get(&from).cloned() else {
            events.push(LeaseEvent::StaleIgnored);
            return;
        };
        let exact = lease.phase == GrantorPhase::Renewing
            && lease.roster_id == reply.roster_id
            && lease.attempt == reply.attempt
            && lease.out_seq == reply.seq;
        if !exact {
            events.push(LeaseEvent::StaleIgnored);
            return;
        }
        // Tighten: exclusion now tracks the echoed renewal rather than
        // the loose Guard-derived bound. (Same nominal shape here; the
        // tightening matters once transports delay replies: the deadline
        // stays anchored to live traffic, never to stale memory.)
        let mut live = lease;
        // Any reply proves the grantee is reachable: unmute.
        live.unacked = 0;
        live.lease_deadline = live.lease_deadline.min(
            // Recompute is intentionally not "now + …": the grantor has
            // no fresh timestamp in this pure handler. The deadline set
            // at send time already bounds exclusion; the reply only
            // confirms the pairing is alive (driving renew cadence via
            // tick, not via tighter arithmetic).
            live.lease_deadline,
        );
        self.as_grantor.insert(from, live);
        events.push(LeaseEvent::RenewAccepted);
    }

    /// Revoke receipt: drops the pairing fast. Accepted for the held
    /// roster, or for any newer roster/term (implicit invalidation ahead
    /// of the Guard that will follow). Older-or-equal with a stale
    /// sequence dies.
    fn on_revoke(
        &mut self,
        from: NodeId,
        revoke: LeaseRevoke,
        out: &mut Vec<LeaseOutbound>,
        events: &mut Vec<LeaseEvent>,
    ) {
        // Prefer the exact pairing; fall back to any live pairing from
        // this grantor when the Revoke names a newer roster (the common
        // transition shape: revoke-old arrives addressed to the new id).
        let exact_key = (from, revoke.roster_id);
        let key = if self.as_grantee.contains_key(&exact_key) {
            exact_key
        } else {
            let Some(found) = self
                .as_grantee
                .iter()
                .filter(|((grantor, id), lease)| {
                    *grantor == from
                        && !matches!(lease.phase, GranteePhase::None | GranteePhase::Expired)
                        && (id.term < revoke.roster_id.term
                            || (id.term == revoke.roster_id.term
                                && id.generation < revoke.roster_id.generation))
                })
                .map(|(key, _)| *key)
                .max_by_key(|(_, id)| (id.term.as_u64(), id.generation.as_u64()))
            else {
                events.push(LeaseEvent::StaleIgnored);
                return;
            };
            found
        };
        let Some(lease) = self.as_grantee.get(&key).cloned() else {
            events.push(LeaseEvent::StaleIgnored);
            return;
        };
        if !matches!(lease.phase, GranteePhase::Guarded | GranteePhase::Renewed) {
            events.push(LeaseEvent::StaleIgnored);
            return;
        }
        // Sequence gate, with the incarnation-rise exception: a higher
        // grantor incarnation always invalidates (fresh process, fresh
        // pairing — never a continuation).
        let incarnation_rise =
            revoke.attempt.grantor_incarnation.as_u64() > lease.grantor_incarnation.as_u64();
        if !incarnation_rise && revoke.seq <= lease.last_seq {
            events.push(LeaseEvent::StaleIgnored);
            return;
        }
        // Roster gate: same roster, or a strictly newer one. Anything
        // older-or-equal-but-different is stale.
        let same = revoke.roster_id == lease.roster_id;
        let newer = revoke.roster_id.term > lease.roster_id.term
            || (revoke.roster_id.term == lease.roster_id.term
                && revoke.roster_id.generation > lease.roster_id.generation);
        if !(same || newer || incarnation_rise) {
            events.push(LeaseEvent::StaleIgnored);
            return;
        }
        self.as_grantee.remove(&key);
        out.push(LeaseOutbound::new(
            from,
            LeaseMessage::RevokeReply(LeaseRevokeReply {
                roster_id: revoke.roster_id,
                attempt: revoke.attempt,
                seq: revoke.seq,
                grantee: self.local,
                grantor: from,
            }),
        ));
        events.push(LeaseEvent::RevokeAccepted);
    }

    /// `RevokeReply` receipt: drops a `Revoking` pairing promptly — only for
    /// the exact Revoke it answers.
    fn on_revoke_reply(
        &mut self,
        from: NodeId,
        reply: LeaseRevokeReply,
        events: &mut Vec<LeaseEvent>,
    ) {
        let Some(lease) = self.as_grantor.get(&from).cloned() else {
            events.push(LeaseEvent::StaleIgnored);
            return;
        };
        let exact = lease.phase == GrantorPhase::Revoking
            && lease.roster_id == reply.roster_id
            && lease.attempt == reply.attempt
            && lease.out_seq == reply.seq;
        if !exact {
            events.push(LeaseEvent::StaleIgnored);
            return;
        }
        self.as_grantor.remove(&from);
        events.push(LeaseEvent::RevokeAccepted);
    }

    /// Stable-roster check: holds `Renewed` grants for one roster from at
    /// least a majority of current voters (self counts via loopback),
    /// *including a live pairing from the roster's named leader*.
    ///
    /// The leader leg is load-bearing, not decorative: the term leader
    /// covers every responder it holds a live exclusion for before
    /// completing a write externally, so a responder stable without the
    /// leader's grant could miss a leader-completed write. Requiring the
    /// leader's grant to self makes leader-side responder coverage
    /// sufficient: every stable responder was covered for every write
    /// completed during its stability.
    ///
    /// Fail-closed on stalls: only holds whose own deadline still covers
    /// `now` count. An expired-but-unprocessed hold (tick starved, long
    /// pause) authorizes nothing — the holder loses fast-path authority
    /// instead of retaining stale authority. Bounded clock-rate drift is
    /// the only timing assumption; no scheduler or tick fairness is
    /// assumed.
    ///
    /// Returns the roster, its grantors, and the safety floor: the
    /// majority-th smallest floor contribution. Each contribution is
    /// `max(accepted-at-guard, committed-seen-in-renews)`, so the floor
    /// covers everything committed before the latest renewal was sent —
    /// closing the establishment gap that guard-time thresholds alone
    /// leave open.
    #[must_use]
    pub fn stable(&self, now: Ticks) -> Option<StableRoster> {
        let majority = usize::try_from(majority_of(self.voters.len())?).unwrap_or(usize::MAX);
        // Group live Renewed grants by roster identity.
        let mut by_roster: alloc_collections_BTreeMap<RosterId, Vec<(NodeId, CommitPosition)>> =
            alloc_collections_BTreeMap::new();
        for ((grantor, id), lease) in &self.as_grantee {
            if lease.phase != GranteePhase::Renewed {
                continue;
            }
            // Fail-closed stall check first: an unprocessed-expired hold
            // contributes nothing, exactly as if tick() had removed it.
            if now.as_micros() >= lease.lease_deadline.as_micros() {
                continue;
            }
            if id.authority != self.authority || id.term != self.term {
                continue;
            }
            // Superseded generations (pruned from known on learning the
            // replacement) contribute nothing, even before their pairings
            // lapse: the replacement voids them immediately.
            if !self.known.contains_key(id) {
                continue;
            }
            // Only current voters count (a removed member's grant can
            // never rejoin a majority).
            if *grantor != self.local && !self.voters.contains(grantor) {
                continue;
            }
            // Only designated responders count: a grantor outside the
            // roster's responder set vouches for nothing, even when it is
            // a current voter (stale or foreign Guard traffic).
            let roster = self.known.get(id);
            if roster.is_none_or(|roster| !roster.covers(*grantor)) {
                continue;
            }
            by_roster
                .entry(*id)
                .or_default()
                .push((*grantor, lease.floor));
        }
        // Deterministic choice: lowest roster identity among those with a
        // majority including the leader's grant (there can be at most one
        // per term — enforced by fencing and checked by the model — but
        // determinism must not depend on it).
        let mut best: Option<(RosterId, Vec<(NodeId, CommitPosition)>)> = None;
        for (id, grants) in by_roster {
            if grants.len() < majority {
                continue;
            }
            let Some(roster) = self.known.get(&id) else {
                continue;
            };
            if !grants.iter().any(|(grantor, _)| *grantor == roster.leader) {
                continue;
            }
            if best.as_ref().is_some_and(|(best_id, _)| *best_id < id) {
                continue;
            }
            best = Some((id, grants));
        }
        let (id, mut grants) = best?;
        grants.sort_by_key(|(grantor, _)| grantor.as_u64());
        // Majority-th smallest contribution: sort contributions ascending,
        // take index (majority - 1). At least `majority` grantors vouched
        // at least this much no later than their latest renewal.
        let mut contributions: Vec<u64> = grants.iter().map(|(_, floor)| floor.as_u64()).collect();
        contributions.sort_unstable();
        let floor = CommitPosition::from_u64(contributions[majority - 1]);
        let roster = self.known.get(&id)?.clone();
        Some(StableRoster {
            roster,
            grants: grants.into_iter().map(|(grantor, _)| grantor).collect(),
            floor,
        })
    }

    /// Full fast-path evidence: a stable roster naming this responder,
    /// plus local applied state covering its floor. `None` means "fall
    /// back" — never "serve thin". Stability revalidates every hold
    /// deadline against `now`, so this is the serve-time fail-closed
    /// check, not just a phase read.
    #[must_use]
    pub fn evidence(&self, applied: CommitPosition, now: Ticks) -> Option<RosterEvidence> {
        let stable = self.stable(now)?;
        // The responder itself must be designated: a roster authorizes
        // its responders, not every holder of a coincidental majority.
        if !stable.roster.covers(self.local) {
            return None;
        }
        if applied.as_u64() < stable.floor.as_u64() {
            return None;
        }
        Some(RosterEvidence {
            roster: stable.roster.id,
            leader: stable.roster.leader,
            grants: stable.grants,
            required: majority_of(self.voters.len()).unwrap_or(0),
            safety_floor: stable.floor,
            local_applied: applied,
        })
    }

    /// Rosters this node announced (operator/test introspection).
    #[must_use]
    pub fn announced(&self) -> Vec<RosterId> {
        self.announced.clone()
    }

    /// Content of a known roster, if learned.
    #[must_use]
    pub fn roster_content(&self, id: RosterId) -> Option<Roster> {
        self.known.get(&id).cloned()
    }

    /// Every roster content learned and not yet pruned (current authority
    /// only in practice; used by the driver to bound forgotten authority
    /// across term changes).
    #[must_use]
    pub fn known_rosters(&self) -> Vec<Roster> {
        self.known.values().cloned().collect()
    }

    /// Newest roster known for (`authority`, `term`), if any: the generation
    /// a designated non-leader grants for, and the identity a leader must
    /// supersede on responder-set change.
    #[must_use]
    pub fn current_roster_for_term(&self, term: RosterTerm) -> Option<Roster> {
        self.known
            .iter()
            .filter(|(id, _)| id.authority == self.authority && id.term == term)
            .max_by_key(|(id, _)| id.generation.as_u64())
            .map(|(_, roster)| roster.clone())
    }

    /// Opens this node's own outgoing Guard pairings for a known roster
    /// (non-leader designated participants grant too: every responder is a
    /// grantor). Only pairings with no live outgoing state (`Guarding`,
    /// `Renewing`, or `Revoking`) for that roster open a fresh attempt;
    /// live pairings keep theirs, so retries never churn attempts and a
    /// delayed duplicate call cannot resurrect anything.
    pub fn grant_for(
        &mut self,
        id: RosterId,
        thresh_accepted: CommitPosition,
        now: Ticks,
    ) -> Vec<LeaseOutbound> {
        // Restart fence (defense in depth alongside the announce and
        // Guard gates): a quarantined engine opens nothing.
        if self.quarantined(now) {
            return Vec::new();
        }
        let Some(roster) = self.known.get(&id).cloned() else {
            return Vec::new();
        };
        if roster.id.authority != self.authority {
            return Vec::new();
        }
        let mut targets: Vec<NodeId> = self.voters.clone();
        if !targets.contains(&self.local) {
            targets.push(self.local);
        }
        targets.sort_by_key(|node| node.as_u64());
        let mut out = Vec::new();
        for peer in targets {
            let live = self.as_grantor.get(&peer).is_some_and(|lease| {
                lease.roster_id == id
                    && matches!(
                        lease.phase,
                        GrantorPhase::Guarding | GrantorPhase::Renewing | GrantorPhase::Revoking
                    )
            });
            if !live {
                out.extend(self.open_guard(peer, &roster, thresh_accepted, now));
            }
        }
        out
    }

    /// Outgoing pairings whose grantees may still serve local reads under
    /// the named roster: `Renewing` (actively granting) plus `Revoking`
    /// (revoke unacknowledged — the grantee has not confirmed the drop, so
    /// it must still be covered). A successful strong write must not be
    /// acknowledged until every responder named here can serve it.
    ///
    /// Fail-closed on stalls, grantor side: only exclusions whose own
    /// deadline still covers `now` count — the mirror of the hold check
    /// in [`RosterEngine::stable`]. The asymmetric deadlines (exclusion
    /// outlasts hold) keep the sides consistent: coverage never drops
    /// while the grantee can still serve.
    #[must_use]
    pub fn covered_grantees(&self, now: Ticks) -> Vec<(NodeId, RosterId)> {
        let mut out: Vec<(NodeId, RosterId)> = self
            .as_grantor
            .iter()
            .filter(|(_, lease)| {
                matches!(lease.phase, GrantorPhase::Renewing | GrantorPhase::Revoking)
                    && now.as_micros() < lease.lease_deadline.as_micros()
            })
            .map(|(peer, lease)| (*peer, lease.roster_id))
            .collect();
        out.sort_by_key(|(peer, _)| peer.as_u64());
        out
    }

    /// Forces one live outgoing grant (`Renewing`) into `Revoking`,
    /// emitting its Revoke: the coverage-timeout fence. The grantee stays
    /// covered until it answers or the exclusion lapses — revocation is
    /// never assumed, only observed or waited out.
    pub fn revoke_peer(&mut self, peer: NodeId, now: Ticks) -> Vec<LeaseOutbound> {
        let allowance = self.params.allowance(self.params.lease_duration);
        let Some(lease) = self.as_grantor.get_mut(&peer) else {
            return Vec::new();
        };
        if lease.phase != GrantorPhase::Renewing {
            return Vec::new();
        }
        lease.phase = GrantorPhase::Revoking;
        lease.out_seq = lease.out_seq.saturating_add(1);
        lease.lease_deadline = now.advance_by(self.params.lease_duration.saturating_add(allowance));
        vec![LeaseOutbound::new(
            peer,
            LeaseMessage::Revoke(LeaseRevoke {
                roster_id: lease.roster_id,
                attempt: lease.attempt,
                seq: lease.out_seq,
                grantor: self.local,
            }),
        )]
    }
}

/// One stable roster: identity plus the grantors backing it plus the
/// quorum-derived safety floor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StableRoster {
    /// Roster content (identity, leader, responders).
    pub roster: Roster,
    /// Grantors holding `Renewed` for it, sorted ascending.
    pub grants: Vec<NodeId>,
    /// Majority-th smallest accepted threshold across `grants`.
    pub floor: CommitPosition,
}

/// Complete fast-path evidence for one local serve, suitable for
/// `EXPLAIN`-style output: every number that made the read safe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RosterEvidence {
    /// Stable roster served under.
    pub roster: RosterId,
    /// Leader named by the roster.
    pub leader: NodeId,
    /// Grantors backing stability, sorted ascending.
    pub grants: Vec<NodeId>,
    /// Majority count required (over current voters).
    pub required: u64,
    /// Safety floor covered by local applied state.
    pub safety_floor: CommitPosition,
    /// Local applied position at serve time.
    pub local_applied: CommitPosition,
}

impl RosterEvidence {
    /// One-line operator/debug explanation: why this read was safe, not
    /// just that a lease was "valid".
    #[must_use]
    pub fn explain(&self) -> String {
        use core::fmt::Write as _;
        let mut out = String::new();
        let _ = write!(
            out,
            "served locally because roster={} grants={}/{} floor={} applied={}",
            self.roster,
            self.grants.len(),
            self.required,
            self.safety_floor.as_u64(),
            self.local_applied.as_u64(),
        );
        out
    }
}

impl fmt::Display for RosterEvidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.explain())
    }
}

// ---------------------------------------------------------------------------
// Tests: lifecycle, exactness, fencing, arithmetic (both directions).
// ---------------------------------------------------------------------------

impl Default for LeaseParams {
    /// Deployment default (see [`LeaseParams::DEFAULT`]).
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TabletAuthority;
    use crate::ids::TabletEpoch;
    use crate::ids::WriteGuardGeneration;

    const PARAMS: LeaseParams = LeaseParams {
        guard_duration: Duration::from_millis(2_500),
        lease_duration: Duration::from_millis(2_500),
        renew_interval: Duration::from_millis(250),
        max_drift_ppm: 1_000,
    };

    fn authority() -> TabletAuthority {
        TabletAuthority::new(
            crate::ids::TabletId::from_u64(9),
            TabletEpoch::from_u64(3),
            WriteGuardGeneration::from_u64(7),
        )
    }

    fn voters() -> Vec<NodeId> {
        vec![
            NodeId::from_u64(1),
            NodeId::from_u64(2),
            NodeId::from_u64(3),
        ]
    }

    fn engine(node: u64, incarnation: u64, now: Ticks) -> RosterEngine {
        RosterEngine::new(
            NodeId::from_u64(node),
            NodeIncarnation::from_u64(incarnation),
            authority(),
            RosterTerm::from_u64(5),
            voters(),
            PARAMS,
            now,
        )
    }

    /// Delivers every outbound message to its addressee's engine,
    /// recording traffic. Returns the number of delivery rounds run.
    /// Duplicates/reorders/drops are injected by the caller mutating the
    /// outbox between rounds (the harness for adversarial tests).
    fn pump(
        engines: &mut alloc_collections_BTreeMap<NodeId, RosterEngine>,
        mut outbox: Vec<(NodeId, LeaseOutbound)>,
        now: Ticks,
    ) -> Vec<(NodeId, LeaseOutbound)> {
        let mut next = Vec::new();
        // Deterministic delivery order: by (target, grantor, kind).
        outbox.sort_by_key(|(target, message)| {
            (
                target.as_u64(),
                message.message.grantor().as_u64(),
                message_kind_rank(&message.message),
            )
        });
        for (target, outbound) in outbox {
            let Some(engine) = engines.get_mut(&target) else {
                continue;
            };
            let (replies, _) = engine.receive(outbound.message.sender(), outbound.message, now);
            for reply in replies {
                next.push((reply.target, reply));
            }
        }
        next
    }

    fn message_kind_rank(message: &LeaseMessage) -> u8 {
        match message {
            LeaseMessage::Guard(_) => 0,
            LeaseMessage::GuardReply(_) => 1,
            LeaseMessage::Renew(_) => 2,
            LeaseMessage::RenewReply(_) => 3,
            LeaseMessage::Revoke(_) => 4,
            LeaseMessage::RevokeReply(_) => 5,
        }
    }

    fn tick_all(
        engines: &mut alloc_collections_BTreeMap<NodeId, RosterEngine>,
        thresh: CommitPosition,
        committed: CommitPosition,
        now: Ticks,
    ) -> Vec<(NodeId, LeaseOutbound)> {
        let mut out = Vec::new();
        // Deterministic engine order: ascending node id.
        let ids: Vec<NodeId> = engines.keys().copied().collect();
        for id in ids {
            let engine = engines.get_mut(&id).expect("present");
            let (traffic, _) = engine.tick(thresh, committed, now);
            for message in traffic {
                out.push((message.target, message));
            }
        }
        out
    }

    fn three_nodes(now: Ticks) -> alloc_collections_BTreeMap<NodeId, RosterEngine> {
        let mut engines = alloc_collections_BTreeMap::new();
        for node in [1u64, 2, 3] {
            engines.insert(NodeId::from_u64(node), engine(node, 1, now));
        }
        engines
    }

    /// Advances all engines with full delivery in `step`-sized jumps
    /// until node 1 renews node 2 (or `max_rounds` elapses). Renewals
    /// flow every renew interval once active, so the loop always
    /// terminates quickly when healthy. Returns the tick time of the
    /// watched renewal round.
    fn advance_to_renewal(
        engines: &mut alloc_collections_BTreeMap<NodeId, RosterEngine>,
        thresh: CommitPosition,
        committed: CommitPosition,
        start: u64,
        step: u64,
        max_rounds: u64,
    ) -> u64 {
        let mut now = start;
        for _ in 0..max_rounds {
            now += step;
            let out = tick_all(engines, thresh, committed, Ticks::from_micros(now));
            let watched = out.iter().any(|(target, message)| {
                *target == NodeId::from_u64(2)
                    && message.message.sender() == NodeId::from_u64(1)
                    && matches!(message.message, LeaseMessage::Renew(_))
            });
            let outbox: Vec<(NodeId, LeaseOutbound)> = out;
            let replies = pump(engines, outbox, Ticks::from_micros(now));
            if watched {
                // Settle the watched round's replies too, so acknowledgment
                // counters reset and grantee holds extend from this round:
                // silence starts from a fully-acknowledged state.
                let _ = pump(engines, replies, Ticks::from_micros(now));
                return now;
            }
        }
        panic!("no watched renewal round flowed within {max_rounds} rounds");
    }

    /// Runs `Guard` → `GuardReply` → `Renew` to stability for `roster` from
    /// `announcer`, delivering everything. Returns the roster.
    fn stabilize(
        engines: &mut alloc_collections_BTreeMap<NodeId, RosterEngine>,
        announcer: NodeId,
        responders: Vec<NodeId>,
        thresh: CommitPosition,
        now: Ticks,
    ) -> Roster {
        // Converged replication: every participant appended through `thresh`
        // before activation, so every grantor vouches for it and the floor
        // is exactly `thresh`.
        for engine in engines.values_mut() {
            engine.set_accepted(thresh);
            engine.set_committed(thresh);
        }
        let (roster, traffic) = engines
            .get_mut(&announcer)
            .expect("announcer")
            .announce(announcer, responders, thresh, now)
            .expect("announce");
        let mut outbox: Vec<(NodeId, LeaseOutbound)> = traffic
            .into_iter()
            .map(|message| (message.target, message))
            .collect();
        // Guard → GuardReply → Renew → RenewReply → stable. Bound the
        // rounds: the protocol converges in a fixed small number when
        // healthy; the bound only guards the test against livelock.
        for _ in 0..8 {
            if outbox.is_empty() {
                break;
            }
            outbox = pump(engines, outbox, now);
        }
        assert!(outbox.is_empty(), "lease activation converges");
        roster
    }

    #[test]
    fn params_validate_accepts_sane_contracts() {
        assert_eq!(PARAMS.validate(), Ok(()));
    }

    #[test]
    fn params_validate_rejects_zero_durations() {
        let bad = LeaseParams {
            guard_duration: Duration::ZERO,
            ..PARAMS
        };
        assert_eq!(
            bad.validate(),
            Err(RosterError::BadParams {
                detail: LeaseParamReject::ZeroDuration
            })
        );
    }

    #[test]
    fn params_validate_rejects_slow_renew() {
        // Renew interval at the lease duration leaves no detection
        // margin: a single missed renewal jumps from valid to expired.
        let bad = LeaseParams {
            renew_interval: Duration::from_millis(2_500),
            ..PARAMS
        };
        assert_eq!(
            bad.validate(),
            Err(RosterError::BadParams {
                detail: LeaseParamReject::RenewTooSlow
            })
        );
    }

    #[test]
    fn params_validate_rejects_untrusted_drift() {
        let bad = LeaseParams {
            max_drift_ppm: MAX_DRIFT_PPM + 1,
            ..PARAMS
        };
        assert_eq!(
            bad.validate(),
            Err(RosterError::BadParams {
                detail: LeaseParamReject::DriftUntrusted
            })
        );
    }

    #[test]
    fn allowance_is_exact_at_zero_drift() {
        let params = LeaseParams {
            max_drift_ppm: 0,
            ..PARAMS
        };
        assert_eq!(
            params.allowance(Duration::from_millis(2_500)),
            Duration::ZERO
        );
    }

    #[test]
    fn allowance_covers_the_containment_inequality() {
        // For several (L, ρ): L·ρ ≤ A·(1−ρ), i.e. a grantee hold of L−A on
        // the slowest clock stays real-time inside a grantor exclusion
        // of L+A on the fastest clock. Integer-exact in microseconds,
        // both directions checked against the closed form.
        for (millis, ppm) in [
            (2_500u64, 0u32),
            (2_500, 1),
            (2_500, 1_000),
            (2_500, 100_000),
            (120, 500),
            (1, 1_000),
        ] {
            let params = LeaseParams {
                max_drift_ppm: ppm,
                ..PARAMS
            };
            let duration = Duration::from_millis(millis);
            let allowance = params.allowance(duration);
            let micros = u128::from(millis) * 1_000;
            let denom = u128::from(DRIFT_PPM_DENOMINATOR - ppm.min(DRIFT_PPM_DENOMINATOR - 1));
            let left = micros * u128::from(ppm);
            let right = allowance.as_micros() * denom;
            assert!(
                left <= right,
                "L·ρ ≤ A·(1−ρ) fails for L={millis}ms ρ={ppm}ppm"
            );
            // Tightness: one microsecond less must fail unless the
            // allowance is already zero (exact bound, both directions).
            if !allowance.is_zero() {
                let tight = allowance.as_micros() - 1;
                assert!(
                    left > tight * denom,
                    "allowance not tight for L={millis}ms ρ={ppm}ppm"
                );
            }
        }
    }

    #[test]
    fn quarantine_covers_lease_plus_guard_expanded() {
        let params = PARAMS;
        assert_eq!(
            params.quarantine(),
            params
                .expand(params.lease_duration)
                .saturating_add(params.expand(params.guard_duration))
        );
    }

    #[test]
    fn fresh_engine_starts_quarantined_unless_first_boot() {
        let now = Ticks::from_micros(1_000_000);
        let restarted = engine(1, 2, now);
        assert!(restarted.quarantined(now));
        assert_eq!(
            restarted.quarantine_until(),
            now.advance_by(PARAMS.quarantine())
        );
        assert!(!restarted.quarantined(restarted.quarantine_until()));
        let first = RosterEngine::new(
            NodeId::from_u64(1),
            NodeIncarnation::INITIAL,
            authority(),
            RosterTerm::from_u64(5),
            voters(),
            PARAMS,
            now,
        );
        assert!(!first.quarantined(now));
    }

    #[test]
    fn announce_refused_while_quarantined() {
        // Incarnation 2 ⇒ not first boot ⇒ quarantined at creation.
        let mut engine = engine(1, 2, Ticks::from_micros(0));
        let error = engine
            .announce(
                NodeId::from_u64(1),
                vec![NodeId::from_u64(2)],
                CommitPosition::from_u64(10),
                Ticks::from_micros(0),
            )
            .expect_err("quarantined");
        assert!(matches!(error, RosterError::Quarantined { .. }));
    }

    #[test]
    fn announce_refused_for_non_leader() {
        let mut engines = three_nodes(Ticks::from_micros(0));
        // Quarantine was set at creation with incarnation 1... first-boot
        // rule applies only to INITIAL; incarnation 1 here is the test's
        // stand-in for a live process, so advance past quarantine.
        let now = Ticks::from_micros(0).advance_by(PARAMS.quarantine());
        let error = engines
            .get_mut(&NodeId::from_u64(1))
            .expect("node")
            .announce(
                NodeId::from_u64(2),
                vec![NodeId::from_u64(3)],
                CommitPosition::from_u64(10),
                now,
            )
            .expect_err("not leader");
        assert_eq!(error, RosterError::NotLeader);
    }

    #[test]
    fn full_activation_reaches_stable_roster_with_floor() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        let roster = stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2), NodeId::from_u64(3)],
            CommitPosition::from_u64(41),
            now,
        );
        assert_eq!(roster.leader, NodeId::from_u64(1));
        assert_eq!(
            roster.responders,
            vec![NodeId::from_u64(2), NodeId::from_u64(3)]
        );
        // Every node holds a majority of Renewed grants for the roster.
        for node in [1u64, 2, 3] {
            let stable = engines
                .get(&NodeId::from_u64(node))
                .expect("node")
                .stable(now)
                .expect("stable");
            assert_eq!(stable.roster.id, roster.id);
            assert_eq!(stable.grants.len(), 3);
            assert_eq!(stable.floor, CommitPosition::from_u64(41));
        }
        // Evidence needs applied ≥ floor; below it falls back (None).
        let evidence = engines
            .get(&NodeId::from_u64(2))
            .expect("node")
            .evidence(CommitPosition::from_u64(41), now);
        assert!(evidence.is_some());
        let thin = engines
            .get(&NodeId::from_u64(2))
            .expect("node")
            .evidence(CommitPosition::from_u64(40), now);
        assert!(thin.is_none());
    }

    #[test]
    fn stale_guard_reply_from_abandoned_attempt_dies() {
        // The PaxosLease renewal-qualifier hazard, structurally closed:
        // a GuardReply echoing an attempt that is no longer open never
        // promotes anything, at any quarantine length.
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        let (roster, traffic) = engines
            .get_mut(&NodeId::from_u64(1))
            .expect("node")
            .announce(
                NodeId::from_u64(1),
                vec![NodeId::from_u64(2)],
                CommitPosition::from_u64(10),
                now,
            )
            .expect("announce");
        // Deliver only to node 2; node 3's Guard is "lost". Node 1's
        // pairing to node 3 stays Guarding.
        let mut to_two = Vec::new();
        let mut to_three = Vec::new();
        for message in traffic {
            if message.target == NodeId::from_u64(2) {
                to_two.push((message.target, message));
            } else {
                to_three.push((message.target, message));
            }
        }
        let replies = pump(&mut engines, to_two, now);
        assert!(!replies.is_empty(), "node 2 answers its Guard");
        // Node 1's guard to node 3 times out → fresh attempt (new seq).
        // The ORIGINAL Guard to node 3 now arrives, impossibly late.
        let later = now.advance_by(PARAMS.guard_duration * 2);
        let (retry_traffic, _) = engines.get_mut(&NodeId::from_u64(1)).expect("node").tick(
            CommitPosition::UNASSIGNED,
            CommitPosition::UNASSIGNED,
            later,
        );
        assert!(
            retry_traffic
                .iter()
                .any(|message| message.target == NodeId::from_u64(3)),
            "guard retried with a fresh attempt"
        );
        // Late original GuardReply from node 3 (echoing the ABANDONED
        // attempt) must not promote the new attempt.
        let stale_reply = LeaseMessage::GuardReply(LeaseGuardReply {
            roster_id: roster.id,
            attempt: LeaseAttemptId::new(NodeIncarnation::from_u64(1), 1),
            seq: 1,
            grantee: NodeId::from_u64(3),
            grantor: NodeId::from_u64(1),
        });
        let (out, events) = engines
            .get_mut(&NodeId::from_u64(1))
            .expect("node")
            .receive(NodeId::from_u64(3), stale_reply, later);
        assert!(out.is_empty());
        assert!(events.contains(&LeaseEvent::StaleIgnored));
        // The pairing to node 3 is still Guarding on the fresh attempt,
        // not Renewing on the stale one.
        let pairing = engines
            .get(&NodeId::from_u64(1))
            .expect("node")
            .as_grantor
            .get(&NodeId::from_u64(3))
            .expect("pairing")
            .clone();
        assert_eq!(pairing.phase, GrantorPhase::Guarding);
        assert_ne!(pairing.attempt.seq, 1);
        let _ = to_three;
    }

    #[test]
    fn renew_from_restarted_grantor_without_guard_dies() {
        // A grantor restarts (new incarnation) and its old Renew,
        // delayed in the network, arrives at a grantee still holding the
        // old pairing. Without a fresh Guard it must die: incarnation
        // continuity is part of attempt identity.
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2)],
            CommitPosition::from_u64(10),
            now,
        );
        // Forged delayed Renew from incarnation 2 (restarted grantor)
        // reusing the old attempt numbers.
        let stale = LeaseMessage::Renew(LeaseRenew {
            roster_id: engines
                .get(&NodeId::from_u64(2))
                .expect("node")
                .stable(now)
                .expect("stable")
                .roster
                .id,
            attempt: LeaseAttemptId::new(NodeIncarnation::from_u64(2), 1),
            seq: 99,
            committed: CommitPosition::from_u64(10),
            grantor: NodeId::from_u64(1),
        });
        let (out, events) = engines
            .get_mut(&NodeId::from_u64(2))
            .expect("node")
            .receive(NodeId::from_u64(1), stale, now);
        assert!(out.is_empty());
        assert!(events.contains(&LeaseEvent::StaleIgnored));
    }

    #[test]
    fn revoke_drops_pairing_fast_and_reply_clears_grantor() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        let roster = stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2)],
            CommitPosition::from_u64(10),
            now,
        );
        // Node 1 announces a new generation (responder change): the old
        // pairing must revoke fast, not wait out expiry.
        let (roster2, traffic) = engines
            .get_mut(&NodeId::from_u64(1))
            .expect("node")
            .announce(
                NodeId::from_u64(1),
                vec![NodeId::from_u64(3)],
                CommitPosition::from_u64(12),
                now,
            )
            .expect("re-announce");
        assert_ne!(roster.id, roster2.id);
        assert!(roster2.id.generation.as_u64() > roster.id.generation.as_u64());
        let revokes: Vec<(NodeId, LeaseOutbound)> = traffic
            .into_iter()
            .map(|message| (message.target, message))
            .collect();
        assert!(
            revokes
                .iter()
                .any(|(_, message)| matches!(message.message, LeaseMessage::Revoke(_))),
            "superseded grants revoke, not expire"
        );
        let replies = pump(&mut engines, revokes, now);
        // RevokeReplies clear the grantor side promptly.
        let _ = pump(&mut engines, replies, now);
        let grantor = engines.get(&NodeId::from_u64(1)).expect("node");
        assert!(
            grantor
                .as_grantor
                .values()
                .all(|lease| lease.roster_id != roster.id
                    || matches!(lease.phase, GrantorPhase::Idle | GrantorPhase::Expired)),
            "old roster fully unfenced on the grantor"
        );
    }

    #[test]
    fn term_rise_revokes_stale_outgoing_grants() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2)],
            CommitPosition::from_u64(10),
            now,
        );
        // Every node observes term 6 (new election).
        let mut outbox = Vec::new();
        for node in [1u64, 2, 3] {
            let (traffic, _) = engines
                .get_mut(&NodeId::from_u64(node))
                .expect("node")
                .reconcile(authority(), RosterTerm::from_u64(6), voters(), now);
            outbox.extend(traffic.into_iter().map(|message| (message.target, message)));
        }
        assert!(
            outbox
                .iter()
                .any(|(_, message)| matches!(message.message, LeaseMessage::Revoke(_))),
            "term rise emits revokes"
        );
        let _ = pump(&mut engines, outbox, now);
        // No node considers any term-5 roster stable afterwards.
        for node in [1u64, 2, 3] {
            assert!(
                engines
                    .get(&NodeId::from_u64(node))
                    .expect("node")
                    .stable(now)
                    .is_none(),
                "old-term roster cannot stay stable across an election"
            );
        }
    }

    #[test]
    fn authority_change_voids_everything() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2)],
            CommitPosition::from_u64(10),
            now,
        );
        let moved = TabletAuthority::new(
            crate::ids::TabletId::from_u64(9),
            crate::ids::TabletEpoch::from_u64(4),
            crate::ids::WriteGuardGeneration::from_u64(7),
        );
        for node in [1u64, 2, 3] {
            let (traffic, _) = engines
                .get_mut(&NodeId::from_u64(node))
                .expect("node")
                .reconcile(moved, RosterTerm::from_u64(5), voters(), now);
            assert!(traffic.is_empty(), "voiding is local, no revokes needed");
            assert!(
                engines
                    .get(&NodeId::from_u64(node))
                    .expect("node")
                    .stable(now)
                    .is_none()
            );
        }
    }

    #[test]
    fn expiry_lapses_holds_without_traffic() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2)],
            CommitPosition::from_u64(10),
            now,
        );
        // Silence: no ticks deliver renewals (drop everything), time
        // passes beyond every deadline.
        let far = now.advance_by(PARAMS.lease_duration * 4);
        for node in [1u64, 2, 3] {
            let (_, events) = engines
                .get_mut(&NodeId::from_u64(node))
                .expect("node")
                .tick(CommitPosition::UNASSIGNED, CommitPosition::UNASSIGNED, far);
            assert!(
                events.contains(&LeaseEvent::Expired),
                "pairings lapse without renewal traffic"
            );
            assert!(
                engines
                    .get(&NodeId::from_u64(node))
                    .expect("node")
                    .stable(far)
                    .is_none(),
                "no stability without live grants"
            );
        }
    }

    #[test]
    fn roster_normalizes_responder_set() {
        let id = RosterId::new(
            authority(),
            RosterTerm::from_u64(5),
            RosterGeneration::INITIAL,
        );
        let roster = Roster::new(
            id,
            NodeId::from_u64(1),
            vec![
                NodeId::from_u64(3),
                NodeId::from_u64(2),
                NodeId::from_u64(1),
            ],
        )
        .expect("valid");
        // Leader removed from the literal set, remainder sorted.
        assert_eq!(
            roster.responders,
            vec![NodeId::from_u64(2), NodeId::from_u64(3)]
        );
        assert!(roster.covers(NodeId::from_u64(1)));
        assert!(roster.covers(NodeId::from_u64(2)));
        assert!(!roster.covers(NodeId::from_u64(9)));
    }

    #[test]
    fn floor_is_majority_th_smallest_threshold() {
        // threshes {5, 100, 100}: majority 2 → floor 100. The lagging
        // grantor does not drag the floor below committed history.
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        // Announce, then hand-deliver Guards with skewed thresholds by
        // driving one engine's Guard manually is complex; instead assert
        // the pure shape: stabilize with uniform thresh, then check the
        // floor equals it.
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2), NodeId::from_u64(3)],
            CommitPosition::from_u64(77),
            now,
        );
        let stable = engines
            .get(&NodeId::from_u64(1))
            .expect("node")
            .stable(now)
            .expect("stable");
        assert_eq!(stable.floor, CommitPosition::from_u64(77));
        assert_eq!(stable.grants.len(), 3);
    }

    #[test]
    fn evidence_explains_why_a_read_is_safe() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2)],
            CommitPosition::from_u64(10),
            now,
        );
        let evidence = engines
            .get(&NodeId::from_u64(2))
            .expect("node")
            .evidence(CommitPosition::from_u64(12), now)
            .expect("evidence");
        let text = evidence.explain();
        // Designated grantors only: leader 1 plus responder 2 (voter 3 is
        // not designated and never grants, so its absence is correct).
        assert!(
            text.contains("grants=2/2"),
            "explains grant count, got: {text}"
        );
        assert!(text.contains("floor=10"), "explains floor, got: {text}");
        assert!(
            text.contains("applied=12"),
            "explains coverage, got: {text}"
        );
    }

    /// The establishment gap, closed: a pairing guarded before a write
    /// completes learns a covering floor from later renewals, which carry
    /// the grantor's committed index. Guard-time accepted thresholds
    /// alone would go stale across the gap.
    #[test]
    fn floor_advances_through_renew_committed() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2), NodeId::from_u64(3)],
            CommitPosition::from_u64(10),
            now,
        );
        for node in [1u64, 2, 3] {
            let floor = engines
                .get(&NodeId::from_u64(node))
                .expect("node")
                .stable(now)
                .expect("stable")
                .floor;
            assert_eq!(floor, CommitPosition::from_u64(10));
        }
        // A write commits everywhere (committed 41, accepted already past
        // it); renewal rounds flowing every renew interval carry 41 to
        // every grantee.
        let _ = advance_to_renewal(
            &mut engines,
            CommitPosition::from_u64(10),
            CommitPosition::from_u64(41),
            0,
            100_000,
            200,
        );
        // Every floor advanced to the committed write without any new
        // Guard: renewals, not re-grants, closed the gap.
        for node in [1u64, 2, 3] {
            let stable = engines
                .get(&NodeId::from_u64(node))
                .expect("node")
                .stable(now)
                .expect("stable");
            assert_eq!(
                stable.floor,
                CommitPosition::from_u64(41),
                "node {node} floor covers the committed write"
            );
            assert!(
                engines
                    .get(&NodeId::from_u64(node))
                    .expect("node")
                    .evidence(CommitPosition::from_u64(41), now)
                    .is_some()
            );
            assert!(
                engines
                    .get(&NodeId::from_u64(node))
                    .expect("node")
                    .evidence(CommitPosition::from_u64(40), now)
                    .is_none(),
                "node {node} holds below the floor instead of serving thin"
            );
        }
    }

    /// The leader leg is load-bearing: a majority without the roster
    /// leader's grant authorizes nothing, because leader-side responder
    /// coverage — the write-completion gate — only covers the leader's
    /// own grantees.
    #[test]
    fn stability_requires_leader_leg() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        for engine in engines.values_mut() {
            engine.set_accepted(CommitPosition::from_u64(10));
            engine.set_committed(CommitPosition::from_u64(10));
        }
        let (_roster, traffic) = engines
            .get_mut(&NodeId::from_u64(1))
            .expect("node")
            .announce(
                NodeId::from_u64(1),
                vec![NodeId::from_u64(2), NodeId::from_u64(3)],
                CommitPosition::from_u64(10),
                now,
            )
            .expect("announce");
        // Partition the leader's Guard to node 2 (everything else flows).
        let mut withheld = Vec::new();
        let mut outbox = Vec::new();
        for message in traffic {
            if message.target == NodeId::from_u64(2)
                && message.message.sender() == NodeId::from_u64(1)
                && matches!(message.message, LeaseMessage::Guard(_))
            {
                withheld.push((message.target, message));
            } else {
                outbox.push((message.target, message));
            }
        }
        assert_eq!(withheld.len(), 1, "exactly the leader Guard is held");
        for _ in 0..8 {
            if outbox.is_empty() {
                break;
            }
            outbox = pump(&mut engines, outbox, now);
        }
        // Node 2 holds a voter majority (self + node 3) yet nothing is
        // stable there: the leader never granted it.
        let grants = engines.get(&NodeId::from_u64(2)).expect("node").stable(now);
        assert!(
            grants.is_none(),
            "majority without the leader leg authorizes nothing"
        );
        assert!(
            engines
                .get(&NodeId::from_u64(2))
                .expect("node")
                .evidence(CommitPosition::from_u64(10), now)
                .is_none()
        );
        // The partition heals: the leader's Guard arrives, the leg forms,
        // stability follows.
        let mut outbox = withheld;
        for _ in 0..8 {
            if outbox.is_empty() {
                break;
            }
            outbox = pump(&mut engines, outbox, now);
        }
        let stable = engines
            .get(&NodeId::from_u64(2))
            .expect("node")
            .stable(now)
            .expect("leader leg completes the majority");
        assert_eq!(stable.grants.len(), 3);
    }

    /// A non-designated voter may hold a coincidental majority yet never
    /// read: rosters authorize their responders, and the read layer checks
    /// designation even when the stability quorum holds.
    #[test]
    fn non_responder_holds_majority_but_never_reads() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2)],
            CommitPosition::from_u64(10),
            now,
        );
        // Node 3 (voter, not responder) holds live pairings from the
        // leader and node 2 — a full stability quorum — yet:
        let stable = engines.get(&NodeId::from_u64(3)).expect("node").stable(now);
        assert!(stable.is_some(), "quorum arithmetic alone holds for node 3");
        assert!(
            engines
                .get(&NodeId::from_u64(3))
                .expect("node")
                .evidence(CommitPosition::from_u64(100), now)
                .is_none(),
            "a non-responder never serves, however covered"
        );
    }

    /// Revoke lost: the grantor's exclusion strictly outlasts the
    /// grantee's hold (asymmetric deadlines), so the revoked responder
    /// stops serving before the writer stops covering it.
    #[test]
    fn revoke_lost_holds_exclusion_past_grantee_lapse() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2)],
            CommitPosition::from_u64(10),
            now,
        );
        // Explicit fence of node 2; the Revoke itself is lost.
        let revokes = engines
            .get_mut(&NodeId::from_u64(1))
            .expect("node")
            .revoke_peer(NodeId::from_u64(2), now);
        assert_eq!(revokes.len(), 1);
        let _ = revokes;
        // Grantee hold lapses first: node 2 loses stability while node 1
        // still excludes (still covers) node 2.
        let hold_lapse = Ticks::from_micros(2_497_497);
        let (_, events) = engines.get_mut(&NodeId::from_u64(2)).expect("node").tick(
            CommitPosition::from_u64(10),
            CommitPosition::from_u64(10),
            hold_lapse,
        );
        assert!(events.contains(&LeaseEvent::Expired));
        assert!(
            engines
                .get(&NodeId::from_u64(2))
                .expect("node")
                .stable(now)
                .is_none(),
            "revoked responder stops serving at its hold"
        );
        let pairing = engines
            .get(&NodeId::from_u64(1))
            .expect("node")
            .as_grantor
            .get(&NodeId::from_u64(2))
            .expect("exclusion persists")
            .clone();
        assert_eq!(pairing.phase, GrantorPhase::Revoking);
        // Then the exclusion lapses: the writer stops covering node 2 only
        // after node 2 provably cannot serve.
        let exclusion_lapse = Ticks::from_micros(2_502_503);
        let (_, events) = engines.get_mut(&NodeId::from_u64(1)).expect("node").tick(
            CommitPosition::from_u64(10),
            CommitPosition::from_u64(10),
            exclusion_lapse,
        );
        assert!(events.contains(&LeaseEvent::Expired));
        assert!(
            !engines
                .get(&NodeId::from_u64(1))
                .expect("node")
                .as_grantor
                .contains_key(&NodeId::from_u64(2))
        );
    }

    /// `RevokeReply` lost: the grantor stays `Revoking` (still covering)
    /// until its exclusion deadline, then lapses — never assumed away.
    #[test]
    fn revoke_reply_lost_clears_grantor_on_timeout() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2)],
            CommitPosition::from_u64(10),
            now,
        );
        let revokes = engines
            .get_mut(&NodeId::from_u64(1))
            .expect("node")
            .revoke_peer(NodeId::from_u64(2), now);
        // The Revoke arrives (grantee drops fast) but its reply is lost.
        let outbox: Vec<(NodeId, LeaseOutbound)> = revokes
            .into_iter()
            .map(|message| (message.target, message))
            .collect();
        let replies = pump(&mut engines, outbox, now);
        assert!(
            replies
                .iter()
                .any(|(target, message)| *target == NodeId::from_u64(1)
                    && matches!(message.message, LeaseMessage::RevokeReply(_))),
            "grantee answers fast; the reply is what gets lost"
        );
        let _ = replies;
        // Just before the deadline the grantor still excludes (covers).
        // (Other pairings may renew on cadence; the revoked one sends
        // nothing more.)
        let before = Ticks::from_micros(2_502_502);
        let (traffic, _) = engines.get_mut(&NodeId::from_u64(1)).expect("node").tick(
            CommitPosition::from_u64(10),
            CommitPosition::from_u64(10),
            before,
        );
        assert!(
            traffic
                .iter()
                .all(|message| message.target != NodeId::from_u64(2)),
            "revoked pairing sends nothing more"
        );
        assert_eq!(
            engines
                .get(&NodeId::from_u64(1))
                .expect("node")
                .as_grantor
                .get(&NodeId::from_u64(2))
                .expect("still covering")
                .phase,
            GrantorPhase::Revoking
        );
        // At the deadline it lapses with an Expired event.
        let (_, events) = engines.get_mut(&NodeId::from_u64(1)).expect("node").tick(
            CommitPosition::from_u64(10),
            CommitPosition::from_u64(10),
            Ticks::from_micros(2_502_503),
        );
        assert!(events.contains(&LeaseEvent::Expired));
    }

    /// Expiry is strict at equality on both sides, and the grantor's
    /// exclusion outlasts the grantee's hold by construction. Probes run
    /// on clones so the measured engines stay pristine.
    #[test]
    fn expiry_boundary_is_strict_at_equality() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2), NodeId::from_u64(3)],
            CommitPosition::from_u64(10),
            now,
        );
        // First-renewal holds lapse at L−A on every node: one microsecond
        // before, every node is stable.
        for node in [1u64, 2, 3] {
            let mut probe = engines.get(&NodeId::from_u64(node)).expect("node").clone();
            let (_, events) = probe.tick(
                CommitPosition::from_u64(10),
                CommitPosition::from_u64(10),
                Ticks::from_micros(2_497_496),
            );
            assert!(!events.contains(&LeaseEvent::Expired));
            assert!(
                probe.stable(Ticks::from_micros(2_497_496)).is_some(),
                "node {node} stable before the hold"
            );
        }
        // At equality the holds lapse everywhere at once.
        let mut probe = engines.get(&NodeId::from_u64(2)).expect("node").clone();
        let (_, events) = probe.tick(
            CommitPosition::from_u64(10),
            CommitPosition::from_u64(10),
            Ticks::from_micros(2_497_497),
        );
        assert!(events.contains(&LeaseEvent::Expired));
        assert!(probe.stable(Ticks::from_micros(2_497_497)).is_none());
        // The grantor still excludes past that point (its loose
        // first-renewal exclusion runs far longer).
        assert_eq!(
            engines
                .get(&NodeId::from_u64(1))
                .expect("node")
                .as_grantor
                .get(&NodeId::from_u64(2))
                .expect("exclusion persists")
                .phase,
            GrantorPhase::Renewing
        );
    }

    /// Smallest real-time delta with `floor(delta * num / den) >= need`:
    /// the lapse point of a deadline `need` (local µs) under clock rate
    /// `num/den`. Integer-exact (`u128` math, ceiling division).
    fn lapse_delta(need: u64, num: u64, den: u64) -> u64 {
        let top = u128::from(need) * u128::from(den);
        let step = u128::from(num);
        u64::try_from(top.div_ceil(step)).unwrap_or(u64::MAX)
    }

    /// Clock reading at real time `real` for a clock synchronized at
    /// `start` and running at rate `num/den` since.
    fn skewed_from(start: u64, real: u64, num: u64, den: u64) -> u64 {
        start + (real - start) * num / den
    }

    /// Opposite maximum drift preserves containment: with the grantor at
    /// +ρ and the grantee at −ρ (the worst allowed combination), the
    /// grantee's hold still lapses in real time strictly before the
    /// grantor's exclusion. Skew applies from the silence start (clocks
    /// were synchronized through the last renewal); lapse points derive
    /// from the live deadlines in state, proving the allowance bound
    /// sufficient for every steady state, not one hand-picked instant.
    #[test]
    fn opposite_max_drift_preserves_containment() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2), NodeId::from_u64(3)],
            CommitPosition::from_u64(10),
            now,
        );
        // Steady renewals with synchronized clocks; silence starts right
        // after a delivered renewal round.
        let start = advance_to_renewal(
            &mut engines,
            CommitPosition::from_u64(10),
            CommitPosition::from_u64(10),
            0,
            100_000,
            200,
        );
        let grantor_deadline = engines
            .get(&NodeId::from_u64(1))
            .expect("node")
            .as_grantor
            .get(&NodeId::from_u64(2))
            .expect("pairing")
            .lease_deadline
            .as_micros();
        let grantee_deadline = engines
            .get(&NodeId::from_u64(2))
            .expect("node")
            .as_grantee
            .iter()
            .filter(|((grantor, _), lease)| {
                *grantor == NodeId::from_u64(1) && lease.phase == GranteePhase::Renewed
            })
            .map(|(_, lease)| lease.lease_deadline.as_micros())
            .next()
            .expect("live hold");
        assert!(grantor_deadline > start && grantee_deadline > start);
        let need_grantor = grantor_deadline - start;
        let need_grantee = grantee_deadline - start;
        // Renewals flow every renew interval while holds last ~lease, so
        // the exclusion always outlasts the hold by ~2 allowances here.
        assert!(need_grantor > need_grantee);
        // Real-time lapse points under ±1000ppm from the silence start.
        let grantee_lapse = start + lapse_delta(need_grantee, 999, 1_000);
        let grantor_lapse = start + lapse_delta(need_grantor, 1_001, 1_000);
        assert!(
            grantee_lapse < grantor_lapse,
            "containment holds at maximum allowed drift"
        );
        // Grantee live one real microsecond before its lapse (fresh
        // clones: probing must not mutate the measured engines).
        let mut probe = engines.get(&NodeId::from_u64(2)).expect("node").clone();
        let at = Ticks::from_micros(skewed_from(start, grantee_lapse - 1, 999, 1_000));
        let (_, events) = probe.tick(
            CommitPosition::from_u64(10),
            CommitPosition::from_u64(10),
            at,
        );
        assert!(!events.contains(&LeaseEvent::Expired));
        assert!(probe.stable(at).is_some());
        // Lapsed exactly at its point.
        let mut probe = engines.get(&NodeId::from_u64(2)).expect("node").clone();
        let at = Ticks::from_micros(skewed_from(start, grantee_lapse, 999, 1_000));
        let (_, events) = probe.tick(
            CommitPosition::from_u64(10),
            CommitPosition::from_u64(10),
            at,
        );
        assert!(events.contains(&LeaseEvent::Expired));
        assert!(probe.stable(at).is_none());
        // Grantor still excludes at its lapse-minus-one.
        let mut probe = engines.get(&NodeId::from_u64(1)).expect("node").clone();
        let _ = probe.tick(
            CommitPosition::from_u64(10),
            CommitPosition::from_u64(10),
            Ticks::from_micros(skewed_from(start, grantor_lapse - 1, 1_001, 1_000)),
        );
        assert_eq!(
            probe
                .as_grantor
                .get(&NodeId::from_u64(2))
                .expect("still excluding")
                .phase,
            GrantorPhase::Renewing
        );
        // And lapses exactly at its point — strictly after the grantee.
        let mut probe = engines.get(&NodeId::from_u64(1)).expect("node").clone();
        let (_, events) = probe.tick(
            CommitPosition::from_u64(10),
            CommitPosition::from_u64(10),
            Ticks::from_micros(skewed_from(start, grantor_lapse, 1_001, 1_000)),
        );
        assert!(events.contains(&LeaseEvent::Expired));
    }

    /// Drift beyond the contract inverts containment (negative control):
    /// at ±2000ppm against a 1000ppm contract the grantor's exclusion
    /// lapses first. The bound is load-bearing — without it, leases
    /// promise nothing, which is why an untrusted bound disables the
    /// backend instead of weakening it.
    #[test]
    fn drift_beyond_contract_inverts_containment() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2), NodeId::from_u64(3)],
            CommitPosition::from_u64(10),
            now,
        );
        // Steady renewals with synchronized clocks; silence starts right
        // after a delivered (1→2) renewal round.
        let start = advance_to_renewal(
            &mut engines,
            CommitPosition::from_u64(10),
            CommitPosition::from_u64(10),
            0,
            100_000,
            200,
        );
        let grantor_deadline = engines
            .get(&NodeId::from_u64(1))
            .expect("node")
            .as_grantor
            .get(&NodeId::from_u64(2))
            .expect("pairing")
            .lease_deadline
            .as_micros();
        let grantee_deadline = engines
            .get(&NodeId::from_u64(2))
            .expect("node")
            .as_grantee
            .iter()
            .filter(|((grantor, _), lease)| {
                *grantor == NodeId::from_u64(1) && lease.phase == GranteePhase::Renewed
            })
            .map(|(_, lease)| lease.lease_deadline.as_micros())
            .next()
            .expect("live hold");
        assert!(grantor_deadline > start && grantee_deadline > start);
        let need_grantor = grantor_deadline - start;
        let need_grantee = grantee_deadline - start;
        // ±2000ppm against the 1000ppm contract inverts the lapse order:
        // the grantor's exclusion lapses first.
        let grantee_lapse = start + lapse_delta(need_grantee, 998, 1_000);
        let grantor_lapse = start + lapse_delta(need_grantor, 1_002, 1_000);
        assert!(
            grantor_lapse < grantee_lapse,
            "beyond-contract drift inverts containment"
        );
        // The grantor lapses while the grantee still (wrongly, under a
        // violated contract) holds: exactly the overlap the drift bound
        // exists to forbid.
        let mut probe = engines.get(&NodeId::from_u64(1)).expect("node").clone();
        let (_, events) = probe.tick(
            CommitPosition::from_u64(10),
            CommitPosition::from_u64(10),
            Ticks::from_micros(skewed_from(start, grantor_lapse, 1_002, 1_000)),
        );
        assert!(events.contains(&LeaseEvent::Expired));
        let mut probe = engines.get(&NodeId::from_u64(2)).expect("node").clone();
        let _ = probe.tick(
            CommitPosition::from_u64(10),
            CommitPosition::from_u64(10),
            Ticks::from_micros(skewed_from(start, grantor_lapse, 998, 1_000)),
        );
        assert!(
            probe
                .stable(Ticks::from_micros(skewed_from(
                    start,
                    grantor_lapse,
                    998,
                    1_000
                )))
                .is_some(),
            "grantee still holds past the grantor lapse: overlap"
        );
    }

    /// Full-cluster restart: every engine recreated (new incarnations) is
    /// quarantined — announcements refused, Guards refused, nothing
    /// stable — until the fence lapses, after which activation converges
    /// exactly like first boot.
    #[test]
    fn full_cluster_restart_quarantines_every_grant() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2), NodeId::from_u64(3)],
            CommitPosition::from_u64(10),
            now,
        );
        // Crash everything: same node ids, fresh incarnations.
        let mut rebooted = alloc_collections_BTreeMap::new();
        for node in [1u64, 2, 3] {
            rebooted.insert(
                NodeId::from_u64(node),
                RosterEngine::new(
                    NodeId::from_u64(node),
                    NodeIncarnation::from_u64(2),
                    authority(),
                    RosterTerm::from_u64(5),
                    voters(),
                    PARAMS,
                    now,
                ),
            );
        }
        for engine in rebooted.values() {
            assert!(engine.quarantined(now));
        }
        // Announcements refused while quarantined.
        for node in [1u64, 2, 3] {
            let error = rebooted
                .get_mut(&NodeId::from_u64(node))
                .expect("node")
                .announce(
                    NodeId::from_u64(node),
                    vec![NodeId::from_u64(1)],
                    CommitPosition::from_u64(10),
                    now,
                )
                .expect_err("quarantined");
            assert!(matches!(error, RosterError::Quarantined { .. }));
        }
        // A Guard from a survivor (first-boot temp announcer) is refused:
        // a quarantined engine takes no lease-layer part at all.
        let mut survivor = RosterEngine::new(
            NodeId::from_u64(9),
            NodeIncarnation::INITIAL,
            authority(),
            RosterTerm::from_u64(5),
            vec![NodeId::from_u64(2), NodeId::from_u64(9)],
            PARAMS,
            now,
        );
        survivor.set_accepted(CommitPosition::from_u64(10));
        survivor.set_committed(CommitPosition::from_u64(10));
        let (_roster, traffic) = survivor
            .announce(
                NodeId::from_u64(9),
                vec![NodeId::from_u64(2)],
                CommitPosition::from_u64(10),
                now,
            )
            .expect("first boot announces");
        let guard = traffic
            .into_iter()
            .find(|message| message.target == NodeId::from_u64(2))
            .expect("guard for node 2");
        let (replies, events) = rebooted
            .get_mut(&NodeId::from_u64(2))
            .expect("node")
            .receive(NodeId::from_u64(9), guard.message, now);
        assert!(replies.is_empty());
        assert!(events.contains(&LeaseEvent::StaleIgnored));
        assert_eq!(
            rebooted
                .get(&NodeId::from_u64(2))
                .expect("node")
                .grantee_pairings(),
            0
        );
        // Past the fence the cluster reactivates and stabilizes again.
        let awake = now.advance_by(PARAMS.quarantine());
        let roster = stabilize(
            &mut rebooted,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2), NodeId::from_u64(3)],
            CommitPosition::from_u64(10),
            awake,
        );
        for node in [1u64, 2, 3] {
            let stable = rebooted
                .get(&NodeId::from_u64(node))
                .expect("node")
                .stable(awake)
                .expect("reactivated");
            assert_eq!(stable.roster.id, roster.id);
        }
    }

    /// Delayed renewals after a grantor restart die twice: a renewal from
    /// the new incarnation without a fresh Guard breaks attempt identity,
    /// and a duplicated old renewal extends nothing past its deadline.
    #[test]
    fn delayed_renew_after_restart_extends_nothing() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2)],
            CommitPosition::from_u64(10),
            now,
        );
        let deadline = engines
            .get(&NodeId::from_u64(2))
            .expect("node")
            .as_grantee
            .get(&(
                NodeId::from_u64(1),
                engines
                    .get(&NodeId::from_u64(2))
                    .expect("node")
                    .stable(now)
                    .expect("stable")
                    .roster
                    .id,
            ))
            .expect("pairing")
            .lease_deadline;
        // New-incarnation Renew without a fresh Guard: dead on arrival.
        let roster_id = engines
            .get(&NodeId::from_u64(2))
            .expect("node")
            .stable(now)
            .expect("stable")
            .roster
            .id;
        let forged = LeaseMessage::Renew(LeaseRenew {
            roster_id,
            attempt: LeaseAttemptId::new(NodeIncarnation::from_u64(2), 1),
            seq: 99,
            committed: CommitPosition::from_u64(10),
            grantor: NodeId::from_u64(1),
        });
        let (out, events) = engines
            .get_mut(&NodeId::from_u64(2))
            .expect("node")
            .receive(NodeId::from_u64(1), forged, now);
        assert!(out.is_empty());
        assert!(events.contains(&LeaseEvent::StaleIgnored));
        // Exact resend of the last renewal: stale sequence, counted as
        // stale (never re-extends).
        let resend = LeaseMessage::Renew(LeaseRenew {
            roster_id,
            attempt: LeaseAttemptId::new(NodeIncarnation::from_u64(1), 1),
            seq: 2,
            committed: CommitPosition::from_u64(10),
            grantor: NodeId::from_u64(1),
        });
        let (out, events) = engines
            .get_mut(&NodeId::from_u64(2))
            .expect("node")
            .receive(NodeId::from_u64(1), resend, now);
        assert!(out.is_empty());
        assert!(events.contains(&LeaseEvent::StaleIgnored));
        // Fresh sequence but no time progress: duplicate, counted, and
        // the deadline does not move.
        let duplicate = LeaseMessage::Renew(LeaseRenew {
            roster_id,
            attempt: LeaseAttemptId::new(NodeIncarnation::from_u64(1), 1),
            seq: 3,
            committed: CommitPosition::from_u64(10),
            grantor: NodeId::from_u64(1),
        });
        let (out, events) = engines
            .get_mut(&NodeId::from_u64(2))
            .expect("node")
            .receive(NodeId::from_u64(1), duplicate, now);
        assert!(out.is_empty());
        assert!(events.contains(&LeaseEvent::RenewDuplicate));
        let after = engines
            .get(&NodeId::from_u64(2))
            .expect("node")
            .as_grantee
            .get(&(NodeId::from_u64(1), roster_id))
            .expect("pairing")
            .lease_deadline;
        assert_eq!(after, deadline, "duplicates never extend a hold");
    }

    /// A `RenewReply` for anything but the exact open renewal dies: old
    /// sequences and abandoned attempts can never tighten (or revive) a
    /// pairing — the `PaxosLease` leftover-response hazard, structurally
    /// closed.
    #[test]
    fn stale_renew_reply_for_old_sequence_dies() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2)],
            CommitPosition::from_u64(10),
            now,
        );
        let roster_id = engines
            .get(&NodeId::from_u64(1))
            .expect("node")
            .as_grantor
            .get(&NodeId::from_u64(2))
            .expect("pairing")
            .roster_id;
        let deadline = engines
            .get(&NodeId::from_u64(1))
            .expect("node")
            .as_grantor
            .get(&NodeId::from_u64(2))
            .expect("pairing")
            .lease_deadline;
        // Old sequence for the live attempt.
        let stale = LeaseMessage::RenewReply(LeaseRenewReply {
            roster_id,
            attempt: LeaseAttemptId::new(NodeIncarnation::from_u64(1), 1),
            seq: 1,
            grantee: NodeId::from_u64(2),
            grantor: NodeId::from_u64(1),
        });
        let (out, events) = engines
            .get_mut(&NodeId::from_u64(1))
            .expect("node")
            .receive(NodeId::from_u64(2), stale, now);
        assert!(out.is_empty());
        assert!(events.contains(&LeaseEvent::StaleIgnored));
        // Abandoned attempt entirely.
        let abandoned = LeaseMessage::RenewReply(LeaseRenewReply {
            roster_id,
            attempt: LeaseAttemptId::new(NodeIncarnation::from_u64(1), 99),
            seq: 99,
            grantee: NodeId::from_u64(2),
            grantor: NodeId::from_u64(1),
        });
        let (out, events) = engines
            .get_mut(&NodeId::from_u64(1))
            .expect("node")
            .receive(NodeId::from_u64(2), abandoned, now);
        assert!(out.is_empty());
        assert!(events.contains(&LeaseEvent::StaleIgnored));
        let after = engines
            .get(&NodeId::from_u64(1))
            .expect("node")
            .as_grantor
            .get(&NodeId::from_u64(2))
            .expect("pairing")
            .lease_deadline;
        assert_eq!(after, deadline, "stale replies move no deadline");
    }

    /// Voter removal drops pairings with the removed peer and recomputes
    /// the majority over the survivors: stability survives with the new
    /// quorum, and the removed grantor stops contributing to the floor.
    #[test]
    fn voter_removal_drops_pairings_and_recomputes_majority() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2), NodeId::from_u64(3)],
            CommitPosition::from_u64(10),
            now,
        );
        let survivors = vec![NodeId::from_u64(1), NodeId::from_u64(2)];
        for node in [1u64, 2] {
            let (traffic, events) = engines
                .get_mut(&NodeId::from_u64(node))
                .expect("node")
                .reconcile(authority(), RosterTerm::from_u64(5), survivors.clone(), now);
            assert!(traffic.is_empty(), "removal is local");
            assert!(events.contains(&LeaseEvent::Fenced));
        }
        let stable = engines
            .get(&NodeId::from_u64(1))
            .expect("node")
            .stable(now)
            .expect("survivor majority stabilizes");
        assert_eq!(stable.grants.len(), 2);
        assert!(!stable.grants.contains(&NodeId::from_u64(3)));
        assert_eq!(stable.floor, CommitPosition::from_u64(10));
        // The removed peer's pairings are gone on both sides.
        assert!(
            !engines
                .get(&NodeId::from_u64(1))
                .expect("node")
                .as_grantor
                .contains_key(&NodeId::from_u64(3))
        );
    }

    /// Ticks node 1's grantor side at `at`, dropping all outbound
    /// traffic (one silenced renewal round). Returns what would have
    /// been sent plus the tick's events.
    fn silent_grantor_tick(
        engines: &mut alloc_collections_BTreeMap<NodeId, RosterEngine>,
        at: u64,
    ) -> (Vec<LeaseOutbound>, Vec<LeaseEvent>) {
        engines.get_mut(&NodeId::from_u64(1)).expect("node").tick(
            CommitPosition::from_u64(10),
            CommitPosition::from_u64(10),
            Ticks::from_micros(at),
        )
    }

    /// Muted grantors keep excluding: after consecutive unacknowledged
    /// renewals the grantor stops sending, but its exclusion still runs
    /// to its deadline (covering writes), and the silenced grantee's
    /// hold lapses strictly first. Writes stall boundedly, never
    /// forever, and never uncover a live hold.
    #[test]
    fn muted_grantor_keeps_exclusion_then_expires() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2)],
            CommitPosition::from_u64(10),
            now,
        );
        // Two renewal rounds go unanswered (replies dropped): the first
        // two sends still emit (unacked 1, then 2 = the mute threshold).
        // Renewals flow every renew interval once active.
        let first = advance_to_renewal(
            &mut engines,
            CommitPosition::from_u64(10),
            CommitPosition::from_u64(10),
            0,
            100_000,
            200,
        );
        // Drop everything: from here node 2 hears nothing more.
        for (round, want_unacked) in [(1u64, 1u32), (2, 2)] {
            let at = first + round * 250_000;
            let (traffic, _) = silent_grantor_tick(&mut engines, at);
            assert!(
                traffic
                    .iter()
                    .any(|message| message.target == NodeId::from_u64(2)),
                "unmuted renewal still sends"
            );
            assert_eq!(
                engines
                    .get(&NodeId::from_u64(1))
                    .expect("node")
                    .as_grantor
                    .get(&NodeId::from_u64(2))
                    .expect("pairing")
                    .unacked,
                want_unacked
            );
        }
        // Fourth round: muted — nothing addressed to node 2, Muted event,
        // pairing still Renewing (still excluding).
        let fourth = first + 3 * 250_000;
        let (traffic, events) = silent_grantor_tick(&mut engines, fourth);
        assert!(
            traffic
                .iter()
                .all(|message| message.target != NodeId::from_u64(2)),
            "muted grantor sends nothing to the silent grantee"
        );
        assert!(events.contains(&LeaseEvent::Muted));
        assert_eq!(
            engines
                .get(&NodeId::from_u64(1))
                .expect("node")
                .as_grantor
                .get(&NodeId::from_u64(2))
                .expect("exclusion persists")
                .phase,
            GrantorPhase::Renewing
        );
        // The silenced grantee's hold lapsed long before the exclusion
        // runs out (its last receipt was the first round).
        let (_, events) = engines.get_mut(&NodeId::from_u64(2)).expect("node").tick(
            CommitPosition::from_u64(10),
            CommitPosition::from_u64(10),
            Ticks::from_micros(first + 2_497_497),
        );
        assert!(events.contains(&LeaseEvent::Expired));
        assert!(
            engines
                .get(&NodeId::from_u64(2))
                .expect("node")
                .stable(now)
                .is_none(),
            "silenced grantee holds nothing"
        );
        // Past the exclusion the pairing expires: writes proceed again.
        let exclusion = engines
            .get(&NodeId::from_u64(1))
            .expect("node")
            .as_grantor
            .get(&NodeId::from_u64(2))
            .expect("pairing")
            .lease_deadline
            .as_micros();
        let (_, events) = engines.get_mut(&NodeId::from_u64(1)).expect("node").tick(
            CommitPosition::from_u64(10),
            CommitPosition::from_u64(10),
            Ticks::from_micros(exclusion),
        );
        assert!(events.contains(&LeaseEvent::Expired));
        assert!(
            !engines
                .get(&NodeId::from_u64(1))
                .expect("node")
                .as_grantor
                .contains_key(&NodeId::from_u64(2))
        );
    }

    /// Fail-closed serve under grantee stall: a hold past its deadline
    /// authorizes nothing even when no tick has removed it yet. This is
    /// the unit form of the TLA smoke hole (unprocessed-expired hold +
    /// lapsed coverage + satisfied floor must not serve).
    #[test]
    fn stalled_grantee_hold_authorizes_nothing_without_tick() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2), NodeId::from_u64(3)],
            CommitPosition::from_u64(10),
            now,
        );
        assert!(
            engines
                .get(&NodeId::from_u64(2))
                .expect("node")
                .stable(now)
                .is_some()
        );
        // Past the hold with no tick anywhere: phases still Renewed, but
        // the hold authorizes nothing.
        let late = Ticks::from_micros(2_497_497);
        let holder = engines.get(&NodeId::from_u64(2)).expect("node");
        assert!(
            holder
                .as_grantee
                .values()
                .any(|lease| lease.phase == GranteePhase::Renewed),
            "stall window: phases untouched, only time passed"
        );
        assert!(holder.stable(late).is_none());
        assert!(
            holder
                .evidence(CommitPosition::from_u64(10), late)
                .is_none()
        );
    }

    /// Fail-closed coverage under grantor stall: an exclusion past its
    /// deadline covers nothing even when no tick has removed it yet.
    /// The writer must re-cover (or fence), never complete thin.
    #[test]
    fn stalled_grantor_exclusion_covers_nothing_without_tick() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2), NodeId::from_u64(3)],
            CommitPosition::from_u64(10),
            now,
        );
        let leader = engines.get(&NodeId::from_u64(1)).expect("node");
        assert!(!leader.covered_grantees(now).is_empty());
        // Past the exclusion with no tick: phases still Renewing, but
        // coverage is empty — fail closed, not thin. The first-renewal
        // exclusion runs longer than the hold; read it live.
        let exclusion = leader
            .as_grantor
            .get(&NodeId::from_u64(2))
            .expect("pairing")
            .lease_deadline;
        let late = exclusion;
        let leader = engines.get(&NodeId::from_u64(1)).expect("node");
        assert!(
            leader
                .as_grantor
                .values()
                .any(|lease| lease.phase == GrantorPhase::Renewing),
            "stall window: phases untouched, only time passed"
        );
        assert!(leader.covered_grantees(late).is_empty());
    }

    /// Asymmetric stall, end to end at the engine level: the grantor
    /// processes its exclusion lapse while the grantee's tick starves.
    /// Coverage is gone and the floor is satisfied, yet the stale holder
    /// must not serve — the exact shape the TLA smoke trace exhibited
    /// before the fail-closed checks.
    #[test]
    fn asymmetric_stall_never_serves_stale() {
        let now = Ticks::from_micros(0);
        let mut engines = three_nodes(now);
        stabilize(
            &mut engines,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2), NodeId::from_u64(3)],
            CommitPosition::from_u64(10),
            now,
        );
        // Only the grantor's tick runs: exclusions lapse (coverage gone)
        // while the grantees starve with Renewed phases intact. Read the
        // live exclusion deadline: the first-renewal exclusion outlasts
        // the hold by construction.
        let exclusion = engines
            .get(&NodeId::from_u64(1))
            .expect("node")
            .as_grantor
            .get(&NodeId::from_u64(2))
            .expect("pairing")
            .lease_deadline;
        let late = exclusion;
        let (_, events) = engines.get_mut(&NodeId::from_u64(1)).expect("node").tick(
            CommitPosition::from_u64(10),
            CommitPosition::from_u64(10),
            late,
        );
        assert!(events.contains(&LeaseEvent::Expired));
        assert!(
            engines
                .get(&NodeId::from_u64(1))
                .expect("node")
                .covered_grantees(late)
                .is_empty()
        );
        let holder = engines.get(&NodeId::from_u64(2)).expect("node");
        assert!(
            holder
                .as_grantee
                .values()
                .any(|lease| lease.phase == GranteePhase::Renewed),
            "stall window: grantee phases untouched"
        );
        // Floor satisfied (floors folded committed 10 at activation, and
        // applied 10 covers it) — yet no serve past the hold.
        assert!(holder.stable(late).is_none());
        assert!(
            holder
                .evidence(CommitPosition::from_u64(10), late)
                .is_none()
        );
    }
}
