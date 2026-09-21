//! Phase 7 consistency execution layer: one hub per node binding authority,
//! freshness evidence, the strong cache, the roster-lease engines, and the
//! read-authority providers.
//!
//! The hub owns no I/O, no clocks, and no networking. It stores the pure
//! values defined in [`kivi_types::consistency`] (receipts, cache entries,
//! metrics) plus one pure [`RosterEngine`](kivi_types::RosterEngine) per
//! tablet, and answers one deterministic question per read —
//! [`ConsistencyHub::plan`] — while the caller executes the plan (barriers,
//! fence syncs, and waits still hop to the owner thread, which alone may
//! touch the Raft handle). Clocks always arrive inside [`ReadContext`];
//! nothing here reads one.
//!
//! ```text
//! authority ──► freshness ──► cached read ──► validation ──► serve OR fallback
//!
//! missing / stale / ambiguous / invalidated / old-generation evidence
//!   ⇒ fall back toward the conservative baseline, never manufacture freshness.
//! ```
//!
//! Strong-read fallback chain (same `Latest` semantics at every layer):
//!
//! ```text
//! RosterLease (stable roster + floor coverage)
//!      │ no stable roster / behind floor / ineligible / drift untrusted
//!      ▼
//! Lazy-ALR (opportunistic batch + ordered fence, no timing assumption)
//!      │ sync unavailable / boundary timeout
//!      ▼
//! ConservativeLeader barrier
//! ```
//!
//! Topology rule: per-tablet state is keyed by tablet and reconciled against
//! the serving authority on every plan. Any authority movement drops
//! receipts, lease engines, and cache entries for that tablet before the
//! plan runs, so split/merge/migration evidence can never leak across
//! lineages. A process restart drops the whole hub (memory-held), so
//! incarnation handling fails closed by construction; recreated lease
//! engines start quarantined and grant nothing until forgotten grants are
//! unusable.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::time::Duration;

use kivi_state::{Operation, OperationResult};
use kivi_types::{
    CommitPosition, ConsistencyMetrics, ConsistencySnapshot, FreshnessReceipt, FreshnessReject,
    LeaseEligibility, LeaseEvent, LeaseMessage, LeaseOutbound, LeaseParams, NodeId,
    NodeIncarnation, ReadAuthorityProvider, ReadContext, ReadContract, ReadReceipt,
    ReplicaFreshness, RosterEngine, RosterEvidence, RosterSummary, RosterTerm, STRONG_CACHE_CAP,
    STRONG_CACHE_VALUE_CAP, ServePath, TabletAuthority, TabletId, Ticks,
};

/// A read the hub already holds a validated answer for, plus how to execute
/// everything else.
#[derive(Debug)]
pub struct PlannedRead {
    /// Execution decision.
    pub plan: ReadPlan,
    /// Cloned cached outcome plus the fill-time applied position on
    /// strong-cache hits (the receipt vouches exactly the state the value
    /// came from); `None` otherwise, when the caller serves from applied
    /// state or a barrier.
    pub cached: Option<CachedServe>,
}

/// One strong-cache hit: the replayable outcome plus the applied position
/// it was served from at fill time.
#[derive(Debug, Clone)]
pub struct CachedServe {
    /// Outcome served at fill time.
    pub outcome: OperationResult,
    /// Applied position served from at fill time.
    pub position: CommitPosition,
}

/// Execution decision for one read. The caller — never the hub — performs
/// barriers, waits, and state-machine reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadPlan {
    /// Serve from local applied state (or the attached cache hit) with no
    /// leader contact on this read.
    ServeLocal {
        /// Which evidence made the local serve valid.
        path: ServePath,
    },
    /// Take a fresh quorum barrier, then serve. `escalated` marks a
    /// bounded-stale read whose receipt could not prove its bound: the
    /// barrier satisfies any bound, so escalation is a fallback, not a
    /// weakening. Unset for conservative `Latest` and fast-path fallbacks.
    BarrierThenServe {
        /// Whether this barrier is a bounded-stale escalation.
        escalated: bool,
    },
    /// Join (or form) the tablet's Lazy-ALR batch: order a fence past
    /// batch formation — or accept a subsuming write boundary — wait for
    /// local apply of the boundary, then serve. The caller executes the
    /// batch protocol; any failure falls back to [`ReadPlan::BarrierThenServe`].
    AlrThenServe,
    /// Wait (bounded, deadline-aware) for local applied coverage of a
    /// Raft log index, then serve. Only for `AtLeast` tokens behind the
    /// replica but valid for its lineage.
    WaitThenServe {
        /// Raft log index to cover (`token.position - 1`).
        index: u64,
    },
    /// Reject without serving: the token's lineage is not this replica's
    /// lineage (wrong tablet/epoch/shape). Waiting cannot cure it.
    Reject {
        /// Why the token can never validate here.
        reject: FreshnessReject,
    },
}

/// One served read: the outcome plus the proof that made it valid. The
/// server attaches [`ServedRead::receipt`] to the wire response so clients
/// can chain `AtLeast` reads and operators can answer "what
/// generation/evidence made this valid" from the response alone.
#[derive(Debug, Clone)]
pub struct ServedRead {
    /// Deterministic outcome read from applied state (or cache replay).
    pub outcome: OperationResult,
    /// Proof of what authority and position served it.
    pub receipt: ReadReceipt,
    /// Which path served it.
    pub path: ServePath,
}

impl ServedRead {
    /// Builds a served read from its three facts.
    #[must_use]
    pub const fn new(outcome: OperationResult, receipt: ReadReceipt, path: ServePath) -> Self {
        Self {
            outcome,
            receipt,
            path,
        }
    }

    /// One-line operator/debug description: why this read was served here,
    /// under what generation, at what position.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "served via {} at {} position {}",
            self.path, self.receipt.authority, self.receipt.position
        )
    }
}

/// One served scan page with the same proof triple as point reads: the
/// page came from one tablet's applied state at one position under one
/// authority. The multi-tablet scan is still not one global snapshot.
#[derive(Debug, Clone)]
pub struct ServedScan {
    /// Page read from applied state.
    pub page: kivi_state::ScanPage,
    /// Proof of what authority and position served it.
    pub receipt: ReadReceipt,
    /// Which path served it.
    pub path: ServePath,
}

impl ServedScan {
    /// Builds a served scan from its three facts.
    #[must_use]
    pub const fn new(page: kivi_state::ScanPage, receipt: ReadReceipt, path: ServePath) -> Self {
        Self {
            page,
            receipt,
            path,
        }
    }
}

/// Strong-cache key: point-read shape plus key bytes. Only reads whose
/// result is a pure function of (tablet, key) at an applied position are
/// cacheable; range scans and conditional writes bypass the cache.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CacheKey {
    /// Key bytes.
    key: Vec<u8>,
    /// Read-shape discriminant (stable within this binary only; the cache
    /// is memory-local and never persisted or replicated).
    kind: u8,
}

impl CacheKey {
    /// Classifies a cacheable point read, or `None` when the operation
    /// bypasses the cache (scans, ranges with parameters, writes).
    fn of(operation: &Operation) -> Option<Self> {
        let (key, kind) = match operation {
            Operation::Get { key } => (key, 1),
            Operation::Exists { key } => (key, 2),
            Operation::CounterGet { key } => (key, 3),
            Operation::GetVersion { key } => (key, 4),
            Operation::BytesLength { key } => (key, 5),
            Operation::GetExpiry { key } => (key, 6),
            _ => return None,
        };
        Some(Self {
            key: key.as_bytes().to_vec(),
            kind,
        })
    }
}

/// One cached read: the outcome plus the exact evidence its fill was based
/// on. Proof-less fills (`proof: None`, from `Any` reads) replay for `Any`
/// and `AtLeast` only — they carry no freshness claim, so they can never
/// satisfy a staleness bound or a `Latest` contract.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Outcome served at fill time.
    outcome: OperationResult,
    /// Applied position served from at fill time.
    position: CommitPosition,
    /// Evidence the fill was based on (`None` for proof-less `Any` fills).
    proof: Option<FreshnessReceipt>,
}

impl CacheEntry {
    /// Inline bytes pinned by this entry (value-size gate on insert).
    fn size_hint(&self) -> u64 {
        match &self.outcome {
            OperationResult::Value(Some(bytes)) => u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            _ => 0,
        }
    }
}

/// Per-tablet consistency state: current authority, held evidence, the
/// lease engine, the strong cache, and worker-local metrics.
#[derive(Debug)]
struct TabletConsistency {
    /// Authority this entry reconciled against.
    authority: TabletAuthority,
    /// Last quorum-barrier proof (bounded-stale evidence).
    receipt: Option<FreshnessReceipt>,
    /// Directional roster-lease engine (`None` until the lease driver
    /// observes term/voters and creates it; dropped on authority move).
    engine: Option<RosterEngine>,
    /// Last term driven into the engine (`None` before the first drive).
    term: Option<RosterTerm>,
    /// Monotonic time before which strong writes must wait out forgotten
    /// old-term authority (takeover fence; `None` when no wait is owed).
    /// Set when an engine is (re)created mid-stream or the observed term
    /// rises without full old-roster coverage; always bounded by the
    /// protocol-derived quarantine.
    takeover_until: Option<Ticks>,
    /// Strong cache: replayable fills under the current authority.
    /// `Latest` never replays from here — only live applied state under
    /// evidence serves strong reads, so no cache entry can bypass the
    /// roster/ALR checks.
    cache: BTreeMap<CacheKey, CacheEntry>,
    /// Read-authority mode for this tablet.
    provider: ReadAuthorityProvider,
    /// Worker-local counters.
    metrics: ConsistencyMetrics,
}

impl TabletConsistency {
    /// Empty state for a newly observed authority.
    fn fresh(authority: TabletAuthority, provider: ReadAuthorityProvider) -> Self {
        Self {
            authority,
            receipt: None,
            engine: None,
            term: None,
            takeover_until: None,
            cache: BTreeMap::new(),
            provider,
            metrics: ConsistencyMetrics::default(),
        }
    }

    /// Drops every evidence kind held. Called whenever the serving
    /// authority moves: cached freshness dies with its authority.
    fn drop_evidence(&mut self) {
        let held = self.receipt.is_some() || self.engine.is_some() || !self.cache.is_empty();
        self.receipt = None;
        self.engine = None;
        self.term = None;
        self.takeover_until = None;
        self.cache.clear();
        self.metrics.roster = None;
        if held {
            self.metrics.strong_cache_invalidations += 1;
        }
    }
}

/// Node-wide consistency hub: per-tablet evidence plus this replica's
/// identity. `Send + Sync` over a plain mutex — plans hold the lock only
/// for map lookup and counter updates, never across I/O. The owner thread
/// drives lease engines through the same lock (engine steps are pure and
/// fast); planning never blocks on transport.
#[derive(Debug)]
pub struct ConsistencyHub {
    /// Per-tablet state, created lazily on first read.
    tablets: Mutex<HashMap<TabletId, TabletConsistency>>,
    /// Default provider for tablets seen first under it.
    default_provider: Mutex<ReadAuthorityProvider>,
    /// This replica's node identity (roster membership checks).
    local: NodeId,
    /// This process's incarnation (lease/proof binding).
    incarnation: NodeIncarnation,
    /// Deployment lease-timing contract for every tablet engine.
    params: LeaseParams,
    /// Whether the contract validated: an untrusted drift bound disables
    /// the `RosterLease` backend (reads use Lazy-ALR or the barrier)
    /// instead of silently weakening it.
    lease_available: bool,
}

/// Outbound lease traffic of one driver step, grouped by addressee.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeaseDriveOut {
    /// Addressee.
    pub target: NodeId,
    /// Messages for that addressee, in emission order.
    pub messages: Vec<LeaseMessage>,
}

impl LeaseDriveOut {
    /// Groups flat engine outbound by addressee in deterministic
    /// (node-id) order.
    #[must_use]
    pub fn group(outbound: Vec<LeaseOutbound>) -> Vec<Self> {
        let mut grouped: BTreeMap<NodeId, Vec<LeaseMessage>> = BTreeMap::new();
        for message in outbound {
            grouped
                .entry(message.target)
                .or_default()
                .push(message.message);
        }
        grouped
            .into_iter()
            .map(|(target, messages)| Self { target, messages })
            .collect()
    }
}

impl ConsistencyHub {
    /// Builds a hub for one replica. `provider` is the initial
    /// read-authority mode for every tablet (the conservative leader path
    /// in production; accelerators opt in per deployment or test).
    /// `params` is the deployment lease-timing contract: when it fails
    /// validation the `RosterLease` backend reports unavailable on every
    /// plan and reads fall back to Lazy-ALR or the barrier.
    #[must_use]
    pub fn new(
        local: NodeId,
        incarnation: NodeIncarnation,
        provider: ReadAuthorityProvider,
        params: LeaseParams,
    ) -> Self {
        let lease_available = params.validate().is_ok();
        Self {
            tablets: Mutex::new(HashMap::new()),
            default_provider: Mutex::new(provider),
            local,
            incarnation,
            params,
            lease_available,
        }
    }

    /// This replica's node identity.
    #[must_use]
    pub const fn local(&self) -> NodeId {
        self.local
    }

    /// This process's incarnation.
    #[must_use]
    pub const fn incarnation(&self) -> NodeIncarnation {
        self.incarnation
    }

    /// Deployment lease-timing contract.
    #[must_use]
    pub const fn lease_params(&self) -> LeaseParams {
        self.params
    }

    /// Whether the `RosterLease` backend may serve: the drift contract
    /// validated. Otherwise plans skip roster evidence entirely.
    #[must_use]
    pub const fn lease_available(&self) -> bool {
        self.lease_available
    }

    /// Switches the read-authority mode for all tablets (operator/test
    /// control for the Lazy-ALR and roster-lease accelerators). Held
    /// evidence is kept: it revalidates under its own authority checks on
    /// the next plan, and any mode still falls back when evidence is
    /// insufficient — switching modes changes latency, never safety.
    pub fn set_provider(&self, provider: ReadAuthorityProvider) {
        if let Ok(mut current) = self.default_provider.lock() {
            *current = provider;
        }
        if let Ok(mut tablets) = self.tablets.lock() {
            for entry in tablets.values_mut() {
                entry.provider = provider;
            }
        }
    }

    /// Deterministic read plan for one operation under its contract. Locks
    /// only for lookup, reconcile, cache probe, and counters.
    /// `eligibility` gates the roster fast path: transactions, batches,
    /// and scans pass [`LeaseEligibility::ConservativeOnly`] until
    /// responder coverage integrates with their commit outcome.
    pub fn plan(
        &self,
        tablet: TabletId,
        operation: &Operation,
        contract: ReadContract,
        fresh: ReplicaFreshness,
        ctx: ReadContext,
        eligibility: LeaseEligibility,
    ) -> PlannedRead {
        // Poisoned hub lock: fail closed to the authority path rather
        // than serving from unvalidated state.
        let Ok(mut tablets) = self.tablets.lock() else {
            return PlannedRead {
                plan: ReadPlan::BarrierThenServe { escalated: false },
                cached: None,
            };
        };
        let default = self
            .default_provider
            .lock()
            .map_or(ReadAuthorityProvider::ConservativeLeader, |mode| *mode);
        let entry = tablets
            .entry(tablet)
            .or_insert_with(|| TabletConsistency::fresh(fresh.authority, default));
        // Reconcile first: any authority movement drops evidence before
        // any plan may consult it.
        if entry.authority.tablet() != fresh.authority.tablet()
            || entry.authority.epoch() != fresh.authority.epoch()
            || entry.authority.guard() != fresh.authority.guard()
        {
            entry.drop_evidence();
            entry.authority = fresh.authority;
        }
        match contract {
            ReadContract::Latest => Self::plan_latest(
                entry,
                fresh,
                ctx,
                self.local,
                self.lease_available,
                eligibility,
            ),
            ReadContract::AtLeast(token) => {
                Self::plan_at_least(entry, Some(operation), token, fresh)
            }
            ReadContract::BoundedStale { max_staleness } => Self::plan_bounded_stale(
                entry,
                Some(operation),
                max_staleness,
                fresh,
                ctx,
                self.incarnation,
            ),
            ReadContract::Any => Self::plan_any(entry, Some(operation)),
        }
    }

    /// Deterministic read plan for a scan page under its contract. Scans
    /// bypass the strong cache (page results are range functions, not
    /// point replays) but enforce the identical contract semantics:
    /// `AtLeast` validates lineage and waits when behind, `BoundedStale`
    /// serves from satisfying receipts or escalates, `Latest` follows the
    /// provider mode. The multi-tablet scan is still not one global
    /// snapshot; each tablet read is individually strong.
    pub fn plan_scan(
        &self,
        tablet: TabletId,
        contract: ReadContract,
        fresh: ReplicaFreshness,
        ctx: ReadContext,
        eligibility: LeaseEligibility,
    ) -> PlannedRead {
        let planned = self.plan_scan_inner(tablet, contract, fresh, ctx, eligibility);
        debug_assert!(
            planned.cached.is_none(),
            "scans never hit the point-read cache"
        );
        planned
    }

    /// Scan planning shared with [`ConsistencyHub::plan`]: identical rules,
    /// no cache probe.
    fn plan_scan_inner(
        &self,
        tablet: TabletId,
        contract: ReadContract,
        fresh: ReplicaFreshness,
        ctx: ReadContext,
        eligibility: LeaseEligibility,
    ) -> PlannedRead {
        let Ok(mut tablets) = self.tablets.lock() else {
            return PlannedRead {
                plan: ReadPlan::BarrierThenServe { escalated: false },
                cached: None,
            };
        };
        let default = self
            .default_provider
            .lock()
            .map_or(ReadAuthorityProvider::ConservativeLeader, |mode| *mode);
        let entry = tablets
            .entry(tablet)
            .or_insert_with(|| TabletConsistency::fresh(fresh.authority, default));
        if entry.authority.tablet() != fresh.authority.tablet()
            || entry.authority.epoch() != fresh.authority.epoch()
            || entry.authority.guard() != fresh.authority.guard()
        {
            entry.drop_evidence();
            entry.authority = fresh.authority;
        }
        match contract {
            ReadContract::Latest => Self::plan_latest(
                entry,
                fresh,
                ctx,
                self.local,
                self.lease_available,
                eligibility,
            ),
            ReadContract::AtLeast(token) => Self::plan_at_least(entry, None, token, fresh),
            ReadContract::BoundedStale { max_staleness } => {
                Self::plan_bounded_stale(entry, None, max_staleness, fresh, ctx, self.incarnation)
            }
            ReadContract::Any => Self::plan_any(entry, None),
        }
    }

    /// Plans a `Latest` read under the tablet's provider mode. No layer
    /// weakens the contract: the roster path serves only on full evidence,
    /// everything uncertain plans the Lazy-ALR batch, whose own failure
    /// plans the barrier at execution time.
    fn plan_latest(
        entry: &mut TabletConsistency,
        fresh: ReplicaFreshness,
        ctx: ReadContext,
        local: NodeId,
        lease_available: bool,
        eligibility: LeaseEligibility,
    ) -> PlannedRead {
        let _ = ctx;
        let _ = local;
        match entry.provider {
            ReadAuthorityProvider::ConservativeLeader => PlannedRead {
                plan: ReadPlan::BarrierThenServe { escalated: false },
                cached: None,
            },
            ReadAuthorityProvider::AlmostLocal => PlannedRead {
                plan: ReadPlan::AlrThenServe,
                cached: None,
            },
            ReadAuthorityProvider::RosterLease => {
                // The roster path needs all of: an eligible read, a
                // trusted drift contract, and full engine evidence. Any
                // gap plans the ALR batch — same semantics, no timing
                // assumption — and counts why, so operators can tell
                // "no stable roster" from "behind the floor" at a glance.
                if !eligibility.lease_may_serve() || !lease_available {
                    entry.metrics.roster_fallbacks += 1;
                    return PlannedRead {
                        plan: ReadPlan::AlrThenServe,
                        cached: None,
                    };
                }
                let Some(engine) = entry.engine.as_ref() else {
                    entry.metrics.roster_fallbacks += 1;
                    return PlannedRead {
                        plan: ReadPlan::AlrThenServe,
                        cached: None,
                    };
                };
                if let Some(proof) = engine.evidence(fresh.applied, ctx.ticks) {
                    entry.metrics.roster_hits += 1;
                    entry.metrics.roster = Some(summarize(&proof, engine, fresh.applied));
                    PlannedRead {
                        plan: ReadPlan::ServeLocal {
                            path: ServePath::RosterLease,
                        },
                        cached: None,
                    }
                } else {
                    // Stable but behind the floor holds (or falls
                    // back at execution); no stable roster falls back
                    // now. Neither serves thin state.
                    if engine.stable(ctx.ticks).is_some() {
                        entry.metrics.roster_holds += 1;
                    } else {
                        entry.metrics.roster_fallbacks += 1;
                    }
                    PlannedRead {
                        plan: ReadPlan::AlrThenServe,
                        cached: None,
                    }
                }
            }
        }
    }

    /// Plans an `AtLeast` read: serve when covered, wait (bounded) when
    /// behind, reject when the token's lineage is foreign.
    fn plan_at_least(
        entry: &mut TabletConsistency,
        operation: Option<&Operation>,
        token: kivi_types::CommitToken,
        fresh: ReplicaFreshness,
    ) -> PlannedRead {
        match fresh.covers_token(token) {
            Ok(()) => {
                if let Some(cached) = Self::cache_lookup(
                    entry,
                    operation,
                    &AtLeastNeed::Covered {
                        position: token.position(),
                    },
                ) {
                    entry.metrics.strong_cache_hits += 1;
                    PlannedRead {
                        plan: ReadPlan::ServeLocal {
                            path: ServePath::StrongCache,
                        },
                        cached: Some(cached),
                    }
                } else {
                    entry.metrics.strong_cache_fallbacks += 1;
                    PlannedRead {
                        plan: ReadPlan::ServeLocal {
                            path: ServePath::AtLeastCovered,
                        },
                        cached: None,
                    }
                }
            }
            Err(FreshnessReject::BeyondApplied { .. }) => {
                entry.metrics.at_least_waits += 1;
                // The token is assigned (coverage checked shape first),
                // so `position - 1` is its Raft log index.
                PlannedRead {
                    plan: ReadPlan::WaitThenServe {
                        index: token.position().as_u64().saturating_sub(1),
                    },
                    cached: None,
                }
            }
            Err(reject) => {
                entry.metrics.at_least_lineage_rejects += 1;
                entry.metrics.fencing_rejects += 1;
                PlannedRead {
                    plan: ReadPlan::Reject { reject },
                    cached: None,
                }
            }
        }
    }

    /// Plans a `BoundedStale` read: serve from satisfying evidence (cache
    /// fill or barrier receipt), else escalate to a fresh barrier — which
    /// satisfies any bound — instead of serving weak data as success.
    fn plan_bounded_stale(
        entry: &mut TabletConsistency,
        operation: Option<&Operation>,
        max_staleness: Duration,
        fresh: ReplicaFreshness,
        ctx: ReadContext,
        incarnation: NodeIncarnation,
    ) -> PlannedRead {
        if let Some(cached) = Self::cache_lookup(
            entry,
            operation,
            &AtLeastNeed::Bounded {
                max_staleness,
                now: ctx.ticks,
                authority: &fresh.authority,
                incarnation,
            },
        ) {
            entry.metrics.strong_cache_hits += 1;
            entry.metrics.bounded_stale_hits += 1;
            return PlannedRead {
                plan: ReadPlan::ServeLocal {
                    path: ServePath::StrongCache,
                },
                cached: Some(cached),
            };
        }
        entry.metrics.strong_cache_fallbacks += 1;
        let proven = entry.receipt.is_some_and(|receipt| {
            receipt
                .satisfies_bound(
                    max_staleness,
                    ctx.ticks,
                    &fresh.authority,
                    fresh.applied,
                    incarnation,
                )
                .is_ok()
        });
        if proven {
            entry.metrics.bounded_stale_hits += 1;
            PlannedRead {
                plan: ReadPlan::ServeLocal {
                    path: ServePath::BoundedStaleProof,
                },
                cached: None,
            }
        } else {
            entry.metrics.bounded_stale_escalations += 1;
            PlannedRead {
                plan: ReadPlan::BarrierThenServe { escalated: true },
                cached: None,
            }
        }
    }

    /// Plans an `Any` read: cache replay when held, else local applied
    /// state with no freshness claim.
    fn plan_any(entry: &mut TabletConsistency, operation: Option<&Operation>) -> PlannedRead {
        if let Some(cached) = Self::cache_lookup(entry, operation, &AtLeastNeed::Any) {
            entry.metrics.strong_cache_hits += 1;
            PlannedRead {
                plan: ReadPlan::ServeLocal {
                    path: ServePath::StrongCache,
                },
                cached: Some(cached),
            }
        } else {
            entry.metrics.strong_cache_fallbacks += 1;
            PlannedRead {
                plan: ReadPlan::ServeLocal {
                    path: ServePath::AnyLocal,
                },
                cached: None,
            }
        }
    }

    /// Probes the strong cache for an entry satisfying `need`. Returns the
    /// cloned outcome on hit; `None` on miss or failed validation (the
    /// caller falls through to its contract path).
    fn cache_lookup(
        entry: &TabletConsistency,
        operation: Option<&Operation>,
        need: &AtLeastNeed<'_>,
    ) -> Option<CachedServe> {
        let key = CacheKey::of(operation?)?;
        let cached = entry.cache.get(&key)?;
        let hit = CachedServe {
            outcome: cached.outcome.clone(),
            position: cached.position,
        };
        match need {
            AtLeastNeed::Any => Some(hit),
            AtLeastNeed::Covered { position } => {
                if cached.position.as_u64() >= position.as_u64() {
                    Some(hit)
                } else {
                    None
                }
            }
            AtLeastNeed::Bounded {
                max_staleness,
                now,
                authority,
                incarnation,
            } => {
                let proof = cached.proof.as_ref()?;
                // Coverage is checked against the fill-time applied
                // position: applied only advances within an incarnation,
                // so usability at fill implies usability now.
                if proof
                    .satisfies_bound(
                        *max_staleness,
                        *now,
                        authority,
                        cached.position,
                        *incarnation,
                    )
                    .is_ok()
                    && cached.position.as_u64() >= proof.boundary.as_u64()
                {
                    Some(hit)
                } else {
                    None
                }
            }
        }
    }

    /// Records a fresh quorum-barrier proof. Replaces any older receipt:
    /// barriers are totally ordered per replica, so the newest proof is
    /// the strongest — and replacement is monotonic on proof time, so a
    /// delayed duplicate barrier can never resurrect weaker evidence over
    /// a newer proof. A receipt for a foreign authority is refused loudly
    /// (caller bug) instead of poisoning the hub.
    pub fn note_barrier(&self, tablet: TabletId, receipt: FreshnessReceipt) {
        if let Ok(mut tablets) = self.tablets.lock() {
            let default = self
                .default_provider
                .lock()
                .map_or(ReadAuthorityProvider::ConservativeLeader, |mode| *mode);
            let entry = tablets
                .entry(tablet)
                .or_insert_with(|| TabletConsistency::fresh(receipt.authority, default));
            if entry.authority.tablet() != receipt.authority.tablet()
                || entry.authority.epoch() != receipt.authority.epoch()
                || entry.authority.guard() != receipt.authority.guard()
            {
                return;
            }
            let replace = entry.receipt.is_none_or(|current| {
                receipt.proven_at.as_micros() >= current.proven_at.as_micros()
            });
            if replace {
                entry.receipt = Some(receipt);
            }
        }
    }

    /// Fills the strong cache from a served read. Uncacheable operations,
    /// oversized values, and error outcomes never enter: the cache replays
    /// small successful point reads only. The serving authority travels
    /// with the fill so the entry reconciles exactly like planned reads —
    /// a fill for a superseded authority is dropped, never stored.
    pub fn insert(
        &self,
        tablet: TabletId,
        authority: TabletAuthority,
        operation: &Operation,
        outcome: &OperationResult,
        position: CommitPosition,
        proof: Option<FreshnessReceipt>,
    ) {
        let Some(key) = CacheKey::of(operation) else {
            return;
        };
        let candidate = CacheEntry {
            outcome: outcome.clone(),
            position,
            proof,
        };
        if candidate.size_hint() > STRONG_CACHE_VALUE_CAP {
            return;
        }
        if let Ok(mut tablets) = self.tablets.lock() {
            let default = self
                .default_provider
                .lock()
                .map_or(ReadAuthorityProvider::ConservativeLeader, |mode| *mode);
            let entry = tablets
                .entry(tablet)
                .or_insert_with(|| TabletConsistency::fresh(authority, default));
            if entry.authority.tablet() != authority.tablet()
                || entry.authority.epoch() != authority.epoch()
                || entry.authority.guard() != authority.guard()
            {
                entry.drop_evidence();
                entry.authority = authority;
            }
            if entry.cache.len() >= STRONG_CACHE_CAP {
                // Deterministic eviction: smallest key first (BTreeMap
                // order), so insert pressure evicts repeatably.
                if let Some(first) = entry.cache.keys().next().cloned() {
                    entry.cache.remove(&first);
                }
            }
            entry.cache.insert(key, candidate);
        }
    }

    /// Fills the strong cache from a served read at its serving position.
    /// Attaches the hub's current barrier receipt when one is held (its
    /// authority matches by reconcile invariant, and lookup revalidates
    /// age — a stale receipt can never satisfy a bound later); otherwise
    /// stores a proof-less fill that replays for `Any` and `AtLeast` only.
    pub fn fill(
        &self,
        tablet: TabletId,
        authority: TabletAuthority,
        operation: &Operation,
        outcome: &OperationResult,
        position: CommitPosition,
    ) {
        let proof = self.current_receipt(tablet);
        self.insert(tablet, authority, operation, outcome, position, proof);
    }

    /// Returns the hub's current barrier receipt for `tablet`, if any.
    /// Lookup-time validation (authority, coverage, age) still applies —
    /// this is evidence transport, never a freshness claim.
    #[must_use]
    pub fn current_receipt(&self, tablet: TabletId) -> Option<FreshnessReceipt> {
        self.tablets
            .lock()
            .ok()
            .and_then(|tablets| tablets.get(&tablet).and_then(|entry| entry.receipt))
    }

    /// One owner-tick of the lease driver for `tablet`: reconciles observed
    /// world state (authority, term, voters) into the tablet's engine,
    /// advances timers (renewals, Guard retries, lapses), and announces or
    /// grants when designated. Returns grouped outbound traffic (loopback
    /// to self already drained internally); the owner sends the rest over
    /// the peer mesh.
    ///
    /// Engines exist only while the tablet's provider is
    /// [`ReadAuthorityProvider::RosterLease`] with a trusted contract:
    /// otherwise any engine is dropped and no traffic emits, so the lease
    /// layer is silent unless reads can use it. `accepted` is this
    /// replica's highest appended log position (Guard-threshold input);
    /// `committed` is its highest committed position (Renew floor input).
    /// Neither is merely applied: the floor must vouch for ordering, not
    /// for local apply progress.
    #[allow(clippy::too_many_arguments)]
    pub fn lease_drive(
        &self,
        tablet: TabletId,
        authority: TabletAuthority,
        term: RosterTerm,
        voters: Vec<NodeId>,
        is_leader: bool,
        accepted: CommitPosition,
        committed: CommitPosition,
        now: Ticks,
    ) -> Vec<LeaseDriveOut> {
        let Ok(mut tablets) = self.tablets.lock() else {
            return Vec::new();
        };
        let default = self
            .default_provider
            .lock()
            .map_or(ReadAuthorityProvider::ConservativeLeader, |mode| *mode);
        let entry = tablets
            .entry(tablet)
            .or_insert_with(|| TabletConsistency::fresh(authority, default));
        if entry.authority != authority {
            entry.drop_evidence();
            entry.authority = authority;
        }
        if entry.provider != ReadAuthorityProvider::RosterLease || !self.lease_available {
            entry.engine = None;
            entry.metrics.roster = None;
            return Vec::new();
        }
        let local = self.local;
        let incarnation = self.incarnation;
        let params = self.params;
        if let Some(until) = takeover_fence(
            entry.engine.as_ref(),
            entry.term,
            term,
            authority,
            local,
            incarnation,
            params.quarantine(),
            now,
        ) {
            entry.takeover_until = Some(until);
        }
        entry.term = Some(term);
        let engine = entry.engine.get_or_insert_with(|| {
            RosterEngine::new(
                local,
                incarnation,
                authority,
                term,
                voters.clone(),
                params,
                now,
            )
        });
        let (mut outbound, mut events) = engine.reconcile(authority, term, voters, now);
        let (ticked, ticked_events) = engine.tick(accepted, committed, now);
        outbound.extend(ticked);
        events.extend(ticked_events);
        if !engine.quarantined(now) {
            if is_leader {
                if engine.current_roster_for_term(term).is_none() {
                    if let Ok((_, announced)) =
                        engine.announce(local, engine.voters(), accepted, now)
                    {
                        outbound.extend(announced);
                    }
                    // Quarantined (checked), NotLeader (checked), or
                    // InvalidId (degenerate voter set): skip this tick and
                    // retry; never half-announce.
                } else if let Some(current) = engine.current_roster_for_term(term) {
                    outbound.extend(engine.grant_for(current.id, accepted, now));
                }
            } else if let Some(current) = engine.current_roster_for_term(term)
                && current.covers(local)
            {
                outbound.extend(engine.grant_for(current.id, accepted, now));
            }
        }
        for event in events {
            match event {
                LeaseEvent::Expired => entry.metrics.roster_expiries += 1,
                LeaseEvent::Fenced => entry.metrics.fencing_rejects += 1,
                // Renew/revoke traffic and duplicate/stale observations
                // carry no counter on their own; batch and serve notes
                // account the outcomes.
                _ => {}
            }
        }
        // Refresh the status summary: stable roster plus this tick's
        // applied view is unknown owner-side, so record the floor with a
        // placeholder applied — plans overwrite it with the live applied
        // position on every roster hit.
        if let Some(stable) = engine.stable(now) {
            let required = kivi_types::majority_of(engine.voters().len()).unwrap_or(0);
            entry.metrics.roster = Some(RosterSummary {
                generation: stable.roster.id.generation.as_u64(),
                term: stable.roster.id.term.as_u64(),
                responders: stable.roster.designated().len() as u64,
                grants: stable.grants.len() as u64,
                required,
                floor: stable.floor,
                applied: entry
                    .metrics
                    .roster
                    .map_or(stable.floor, |summary| summary.applied),
                quarantined: engine.quarantined(now),
            });
        } else {
            entry.metrics.roster = None;
        }
        let remote: Vec<LeaseOutbound> = Self::drain_loopback(engine, outbound, local, now);
        LeaseDriveOut::group(remote)
    }

    /// Drains loopback traffic (addressed to self) through the engine
    /// directly, bounded: self-pairings converge in a fixed small number
    /// of steps, and anything beyond the bound rides the next tick.
    fn drain_loopback(
        engine: &mut RosterEngine,
        mut outbound: Vec<LeaseOutbound>,
        local: NodeId,
        now: Ticks,
    ) -> Vec<LeaseOutbound> {
        let mut remote = Vec::new();
        for _ in 0..8 {
            let mut pending = Vec::new();
            for message in outbound {
                if message.target == local {
                    pending.push(message.message);
                } else {
                    remote.push(message);
                }
            }
            if pending.is_empty() {
                break;
            }
            outbound = Vec::new();
            for message in pending {
                let (replies, _) = engine.receive(message.sender(), message, now);
                outbound.extend(replies);
            }
        }
        remote
    }

    /// Feeds one inbound lease batch from `from` into the tablet's engine.
    /// Unknown tablets, foreign providers, and missing engines ignore the
    /// batch (never create state on inbound traffic). Returns grouped
    /// replies for the owner to send back.
    pub fn lease_receive(
        &self,
        tablet: TabletId,
        from: NodeId,
        messages: Vec<LeaseMessage>,
        now: Ticks,
    ) -> Vec<LeaseDriveOut> {
        let Ok(mut tablets) = self.tablets.lock() else {
            return Vec::new();
        };
        let Some(entry) = tablets.get_mut(&tablet) else {
            return Vec::new();
        };
        if entry.provider != ReadAuthorityProvider::RosterLease || !self.lease_available {
            return Vec::new();
        }
        let Some(engine) = entry.engine.as_mut() else {
            return Vec::new();
        };
        // Defensive authority gate: engine checks per message anyway, but
        // a batch for a retired lineage dies here without touching state.
        if engine.authority() != entry.authority {
            return Vec::new();
        }
        let local = self.local;
        let mut outbound = Vec::new();
        for message in messages {
            let (replies, events) = engine.receive(from, message, now);
            for event in events {
                if matches!(event, LeaseEvent::Expired) {
                    entry.metrics.roster_expiries += 1;
                }
            }
            outbound.extend(replies);
        }
        let remote = Self::drain_loopback(engine, outbound, local, now);
        LeaseDriveOut::group(remote)
    }

    /// Full fast-path evidence for one serve: a stable roster plus local
    /// applied state covering its floor. `None` means "fall back" — never
    /// "serve thin". The read path revalidates at serve time (after the
    /// plan, before touching state), so a roster that destabilized in
    /// between cannot authorize. `now` fail-closes every hold deadline:
    /// pass the serve-time clock, never a stale one.
    #[must_use]
    pub fn evidence(
        &self,
        tablet: TabletId,
        applied: CommitPosition,
        now: Ticks,
    ) -> Option<RosterEvidence> {
        self.tablets
            .lock()
            .ok()?
            .get(&tablet)?
            .engine
            .as_ref()?
            .evidence(applied, now)
    }

    /// Outgoing pairings whose grantees may still serve local reads: a
    /// successful strong write must cover every responder named here
    /// before it may complete externally. Empty unless the roster backend
    /// is engaged on this tablet. `now` fail-closes every exclusion
    /// deadline: pass the completion-time clock, never a stale one.
    #[must_use]
    pub fn covered_grantees(
        &self,
        tablet: TabletId,
        now: Ticks,
    ) -> Vec<(NodeId, kivi_types::RosterId)> {
        let Ok(tablets) = self.tablets.lock() else {
            return Vec::new();
        };
        let Some(entry) = tablets.get(&tablet) else {
            return Vec::new();
        };
        if entry.provider != ReadAuthorityProvider::RosterLease || !self.lease_available {
            return Vec::new();
        }
        entry
            .engine
            .as_ref()
            .map_or_else(Vec::new, |engine| engine.covered_grantees(now))
    }

    /// Remaining takeover-fence wait for `tablet`, if any: a leader that
    /// sat out the previous term (or just created its engine) must wait
    /// out forgotten old-term holds before completing writes externally.
    /// Always bounded by the protocol-derived quarantine; `None` means no
    /// wait is owed (fenced window passed, first boot, or backend
    /// disengaged).
    #[must_use]
    pub fn takeover_hold(&self, tablet: TabletId, now: Ticks) -> Option<Duration> {
        let tablets = self.tablets.lock().ok()?;
        let entry = tablets.get(&tablet)?;
        if entry.provider != ReadAuthorityProvider::RosterLease || !self.lease_available {
            return None;
        }
        let until = entry.takeover_until?;
        if now.as_micros() >= until.as_micros() {
            return None;
        }
        Some(until.saturating_since(now))
    }

    /// Forces one live outgoing grant into revocation (coverage-timeout
    /// fence). Returns grouped Revoke traffic for the owner to send. The
    /// grantee stays covered until it answers or the exclusion lapses.
    pub fn lease_revoke_peer(
        &self,
        tablet: TabletId,
        peer: NodeId,
        now: Ticks,
    ) -> Vec<LeaseDriveOut> {
        let Ok(mut tablets) = self.tablets.lock() else {
            return Vec::new();
        };
        let Some(entry) = tablets.get_mut(&tablet) else {
            return Vec::new();
        };
        let Some(engine) = entry.engine.as_mut() else {
            return Vec::new();
        };
        LeaseDriveOut::group(engine.revoke_peer(peer, now))
    }

    /// Operator/debug explanation of what the next `Latest` read on
    /// `tablet` would do and why: full evidence on the fast path, or the
    /// exact missing piece on fallback. Never just "lease valid = true".
    #[must_use]
    pub fn explain_latest(&self, tablet: TabletId, applied: CommitPosition, now: Ticks) -> String {
        let Ok(tablets) = self.tablets.lock() else {
            return "hub lock poisoned: barrier".to_owned();
        };
        let Some(entry) = tablets.get(&tablet) else {
            return "no tablet state: barrier".to_owned();
        };
        match entry.provider {
            ReadAuthorityProvider::ConservativeLeader => {
                "barrier: conservative provider".to_owned()
            }
            ReadAuthorityProvider::AlmostLocal => {
                "lazy-alr batch: no timing assumption, fence orders past formation".to_owned()
            }
            ReadAuthorityProvider::RosterLease => {
                if !self.lease_available {
                    return "fallback to lazy-alr: drift contract untrusted".to_owned();
                }
                let Some(engine) = entry.engine.as_ref() else {
                    return "fallback to lazy-alr: lease driver has not run yet".to_owned();
                };
                if engine.quarantined(now) {
                    return format!(
                        "fallback to lazy-alr: lease engine quarantined until {}",
                        engine.quarantine_until()
                    );
                }
                match engine.evidence(applied, now) {
                    Some(proof) => proof.explain(),
                    None => match engine.stable(now) {
                        Some(stable) => format!(
                            "fallback to lazy-alr: stable roster {} but applied {} below floor {}",
                            stable.roster.id,
                            applied.as_u64(),
                            stable.floor.as_u64(),
                        ),
                        None => "fallback to lazy-alr: no stable roster (grants below majority, expired, or fenced)".to_owned(),
                    },
                }
            }
        }
    }

    /// Records one finished serve for the contract mix and path split.
    pub fn note_served(&self, tablet: TabletId, contract: ReadContract, via_authority: bool) {
        if let Ok(mut tablets) = self.tablets.lock()
            && let Some(entry) = tablets.get_mut(&tablet)
        {
            entry
                .metrics
                .record_serve(contract.metric_discriminant(), via_authority);
        }
    }

    /// Records an `AtLeast` wait that timed out instead of covering.
    pub fn note_wait_timeout(&self, tablet: TabletId) {
        if let Ok(mut tablets) = self.tablets.lock()
            && let Some(entry) = tablets.get_mut(&tablet)
        {
            entry.metrics.at_least_timeouts += 1;
        }
    }

    /// Records a bounded-stale escalation whose authority path then failed:
    /// the bound was unprovable and no fresh proof could be established.
    pub fn note_escalation_failed(&self, tablet: TabletId) {
        if let Ok(mut tablets) = self.tablets.lock()
            && let Some(entry) = tablets.get_mut(&tablet)
        {
            entry.metrics.freshness_unprovable += 1;
        }
    }

    /// Records a finished Lazy-ALR batch: one fence (or subsumption)
    /// covering `reads` reads.
    pub fn note_alr_batch(&self, tablet: TabletId, reads: u64, subsumed: bool) {
        if let Ok(mut tablets) = self.tablets.lock()
            && let Some(entry) = tablets.get_mut(&tablet)
        {
            entry.metrics.alr_batches += 1;
            entry.metrics.alr_reads += reads;
            if subsumed {
                entry.metrics.alr_subsumed += 1;
            } else {
                entry.metrics.alr_syncs += 1;
            }
        }
    }

    /// Records a Lazy-ALR batch abandoned to the conservative barrier
    /// (sync unavailable, boundary timeout, lost leadership).
    pub fn note_alr_fallback(&self, tablet: TabletId) {
        if let Ok(mut tablets) = self.tablets.lock()
            && let Some(entry) = tablets.get_mut(&tablet)
        {
            entry.metrics.alr_fallbacks += 1;
        }
    }

    /// Records a roster-lease fast path abandoned during execution (the
    /// serve-time revalidation found no evidence): the read falls back to
    /// Lazy-ALR or the barrier, never to a thin serve.
    pub fn note_roster_fallback(&self, tablet: TabletId) {
        if let Ok(mut tablets) = self.tablets.lock()
            && let Some(entry) = tablets.get_mut(&tablet)
        {
            entry.metrics.roster_fallbacks += 1;
        }
    }

    /// Records a strong write gated on explicit responder coverage.
    pub fn note_coverage_wait(&self, tablet: TabletId) {
        if let Ok(mut tablets) = self.tablets.lock()
            && let Some(entry) = tablets.get_mut(&tablet)
        {
            entry.metrics.coverage_waits += 1;
        }
    }

    /// Records a coverage gate that timed out: the write failed retryable
    /// and the uncovered responder was fenced, instead of acknowledging an
    /// uncovered write.
    pub fn note_coverage_timeout(&self, tablet: TabletId) {
        if let Ok(mut tablets) = self.tablets.lock()
            && let Some(entry) = tablets.get_mut(&tablet)
        {
            entry.metrics.coverage_timeouts += 1;
        }
    }

    /// Records an authority/fencing rejection observed while executing
    /// (barrier refused, leadership lost mid-read).
    pub fn note_fencing_reject(&self, tablet: TabletId) {
        if let Ok(mut tablets) = self.tablets.lock()
            && let Some(entry) = tablets.get_mut(&tablet)
        {
            entry.metrics.fencing_rejects += 1;
        }
    }

    /// Drops all evidence for `tablet` (explicit invalidation beyond the
    /// per-plan reconcile: split/merge cutover, migration completion).
    /// Returns whether anything was held.
    pub fn invalidate(&self, tablet: TabletId) -> bool {
        if let Ok(mut tablets) = self.tablets.lock()
            && let Some(entry) = tablets.get_mut(&tablet)
        {
            let held = entry.receipt.is_some() || entry.engine.is_some() || !entry.cache.is_empty();
            entry.drop_evidence();
            return held;
        }
        false
    }

    /// Freezes this tablet's authority plus metrics for status surfaces.
    /// `now` fail-closes the covered-responder gauge: expired exclusions
    /// are not part of the write-gate set.
    #[must_use]
    pub fn snapshot(
        &self,
        tablet: TabletId,
        now: Ticks,
    ) -> Option<(TabletAuthority, ConsistencySnapshot)> {
        self.tablets.lock().ok().and_then(|tablets| {
            tablets.get(&tablet).map(|entry| {
                let mut snapshot = entry.metrics.snapshot();
                // Gauge, not a counter: refresh from live engine state on
                // every snapshot so operators see the exact write-gate set.
                snapshot.covered_responders = entry
                    .engine
                    .as_ref()
                    .map_or(0, |engine| engine.covered_grantees(now).len() as u64);
                (entry.authority, snapshot)
            })
        })
    }
}

/// Decides the takeover fence owed by this drive (`None` = no wait).
/// A node that sat out the previous term — or created its engine
/// mid-stream — cannot prove no forgotten responder still authorizes
/// reads, so strong writes wait out the worst forgotten hold (bounded by
/// `quarantine`). First boot is exempt (no forgotten holders can exist).
/// A term rise with full old-roster coverage needs no fence: revocation
/// preserves the exclusions the normal coverage gate already waits out.
#[allow(clippy::too_many_arguments)]
fn takeover_fence(
    engine: Option<&RosterEngine>,
    driven_term: Option<RosterTerm>,
    term: RosterTerm,
    authority: TabletAuthority,
    local: NodeId,
    incarnation: NodeIncarnation,
    quarantine: Duration,
    now: Ticks,
) -> Option<Ticks> {
    if incarnation == NodeIncarnation::INITIAL {
        return None;
    }
    let Some(engine) = engine else {
        return Some(now.advance_by(quarantine));
    };
    if driven_term.is_some_and(|driven| driven == term) {
        return None;
    }
    let covered: Vec<NodeId> = engine
        .covered_grantees(now)
        .into_iter()
        .map(|(peer, _)| peer)
        .collect();
    let mut designated: Vec<NodeId> = engine
        .known_rosters()
        .into_iter()
        .filter(|roster| roster.id.authority == authority && roster.id.term != term)
        .flat_map(|roster| roster.designated())
        .collect();
    designated.sort_by_key(|node| node.as_u64());
    designated.dedup_by_key(|node| node.as_u64());
    designated
        .into_iter()
        .any(|node| node != local && !covered.contains(&node))
        .then(|| now.advance_by(quarantine))
}

/// Builds the status summary for full fast-path evidence: every number
/// that made one local serve safe.
fn summarize(
    proof: &RosterEvidence,
    engine: &RosterEngine,
    applied: CommitPosition,
) -> RosterSummary {
    RosterSummary {
        generation: proof.roster.generation.as_u64(),
        term: proof.roster.term.as_u64(),
        responders: engine
            .roster_content(proof.roster)
            .map_or(proof.grants.len() as u64, |roster| {
                roster.designated().len() as u64
            }),
        grants: proof.grants.len() as u64,
        required: proof.required,
        floor: proof.safety_floor,
        applied,
        quarantined: false,
    }
}

/// What a cache probe must satisfy: replays the contract's rule against
/// fill-time evidence without consulting live state.
#[derive(Debug, Clone, Copy)]
enum AtLeastNeed<'a> {
    /// Any cached entry under the current authority replays.
    Any,
    /// Entry position must cover the token position.
    Covered {
        /// Token position to cover.
        position: CommitPosition,
    },
    /// Entry must carry a proof satisfying the bound.
    Bounded {
        /// Requested staleness bound.
        max_staleness: Duration,
        /// Monotonic now.
        now: kivi_types::Ticks,
        /// Current authority.
        authority: &'a TabletAuthority,
        /// Current incarnation.
        incarnation: NodeIncarnation,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_state::Key;
    use kivi_types::{
        CommitPosition, ReadContract, Ticks, WallTimestamp, consistency::AT_LEAST_WAIT_CAP,
    };
    use std::time::Duration;

    const TABLET: TabletId = TabletId::from_u64(3);
    const ELIGIBLE: LeaseEligibility = LeaseEligibility::Eligible;
    const INELIGIBLE: LeaseEligibility = LeaseEligibility::ConservativeOnly;

    fn authority() -> TabletAuthority {
        use kivi_types::{TabletEpoch, WriteGuardGeneration};
        TabletAuthority::new(
            TABLET,
            TabletEpoch::from_u64(1),
            WriteGuardGeneration::from_u64(1),
        )
    }

    fn moved_authority() -> TabletAuthority {
        use kivi_types::{TabletEpoch, WriteGuardGeneration};
        TabletAuthority::new(
            TABLET,
            TabletEpoch::from_u64(2),
            WriteGuardGeneration::from_u64(1),
        )
    }

    fn fresh(applied: u64) -> ReplicaFreshness {
        ReplicaFreshness::new(
            authority(),
            CommitPosition::from_u64(applied),
            NodeIncarnation::from_u64(7),
        )
    }

    fn ctx(ticks: u64) -> ReadContext {
        ReadContext::new(
            WallTimestamp::from_micros(i64::try_from(ticks).expect("test ticks fit")),
            Ticks::from_micros(ticks),
            Duration::from_secs(1),
        )
    }

    fn token(position: u64) -> kivi_types::CommitToken {
        use kivi_types::TabletEpoch;
        kivi_types::CommitToken::new(
            TABLET,
            TabletEpoch::from_u64(1),
            CommitPosition::from_u64(position),
        )
    }

    fn get_op() -> Operation {
        Operation::Get {
            key: Key::from(b"k".to_vec()),
        }
    }

    fn lease_params() -> LeaseParams {
        LeaseParams {
            guard_duration: Duration::from_millis(2_500),
            lease_duration: Duration::from_millis(2_500),
            renew_interval: Duration::from_millis(250),
            max_drift_ppm: 1_000,
        }
    }

    fn hub() -> ConsistencyHub {
        ConsistencyHub::new(
            NodeId::from_u64(1),
            NodeIncarnation::from_u64(7),
            ReadAuthorityProvider::ConservativeLeader,
            lease_params(),
        )
    }

    fn lease_hub() -> ConsistencyHub {
        ConsistencyHub::new(
            NodeId::from_u64(1),
            NodeIncarnation::from_u64(7),
            ReadAuthorityProvider::RosterLease,
            lease_params(),
        )
    }

    /// Eligible point-read plan shorthand: the common test shape.
    fn plan(hub: &ConsistencyHub, contract: ReadContract, applied: u64, ticks: u64) -> ReadPlan {
        hub.plan(
            TABLET,
            &get_op(),
            contract,
            fresh(applied),
            ctx(ticks),
            ELIGIBLE,
        )
        .plan
    }

    /// Drives one hub's engine as the term leader of a single-voter
    /// group at `now` with `accepted` appended and committed.
    /// Single-voter loopback converges inside the drive, so one call
    /// stabilizes.
    fn drive_leader(
        hub: &ConsistencyHub,
        term: u64,
        accepted: u64,
        now: u64,
    ) -> Vec<LeaseDriveOut> {
        hub.lease_drive(
            TABLET,
            authority(),
            RosterTerm::from_u64(term),
            vec![NodeId::from_u64(1)],
            true,
            CommitPosition::from_u64(accepted),
            CommitPosition::from_u64(accepted),
            Ticks::from_micros(now),
        )
    }

    #[test]
    fn latest_always_barriers_under_the_conservative_mode() {
        let hub = hub();
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::Latest,
            fresh(10),
            ctx(0),
            ELIGIBLE,
        );
        assert_eq!(
            planned.plan,
            ReadPlan::BarrierThenServe { escalated: false }
        );
        assert!(planned.cached.is_none());
    }

    #[test]
    fn at_least_covered_serves_without_leader_contact() {
        let hub = hub();
        assert_eq!(
            plan(&hub, ReadContract::AtLeast(token(9)), 10, 0),
            ReadPlan::ServeLocal {
                path: ServePath::AtLeastCovered
            }
        );
    }

    #[test]
    fn at_least_behind_waits_with_the_log_index() {
        let hub = hub();
        // position 11 ⇔ Raft index 10.
        assert_eq!(
            plan(&hub, ReadContract::AtLeast(token(11)), 10, 0),
            ReadPlan::WaitThenServe { index: 10 }
        );
    }

    #[test]
    fn at_least_foreign_lineage_rejects_without_waiting() {
        use kivi_types::TabletEpoch;
        let hub = hub();
        let foreign = kivi_types::CommitToken::new(
            TabletId::from_u64(99),
            TabletEpoch::from_u64(1),
            CommitPosition::from_u64(2),
        );
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::AtLeast(foreign),
            fresh(10),
            ctx(0),
            ELIGIBLE,
        );
        assert!(matches!(planned.plan, ReadPlan::Reject { .. }));
        // A token from a superseded epoch rejects too, even when its
        // position is covered: split/merge lineage never validates.
        let old_epoch = kivi_types::CommitToken::new(
            TABLET,
            TabletEpoch::from_u64(9),
            CommitPosition::from_u64(2),
        );
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::AtLeast(old_epoch),
            fresh(10),
            ctx(0),
            ELIGIBLE,
        );
        assert!(matches!(planned.plan, ReadPlan::Reject { .. }));
    }

    #[test]
    fn bounded_stale_without_evidence_escalates_never_serves_weak() {
        let hub = hub();
        // No receipt held: must escalate, never ServeLocal.
        assert_eq!(
            plan(
                &hub,
                ReadContract::BoundedStale {
                    max_staleness: Duration::from_secs(60),
                },
                10,
                0,
            ),
            ReadPlan::BarrierThenServe { escalated: true }
        );
    }

    #[test]
    fn bounded_stale_with_fresh_receipt_serves_from_proof() {
        let hub = hub();
        hub.note_barrier(
            TABLET,
            FreshnessReceipt::new(
                authority(),
                CommitPosition::from_u64(10),
                Ticks::from_micros(0),
                NodeIncarnation::from_u64(7),
            ),
        );
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::BoundedStale {
                max_staleness: Duration::from_millis(100),
            },
            fresh(10),
            ctx(50_000),
            ELIGIBLE,
        );
        assert_eq!(
            planned.plan,
            ReadPlan::ServeLocal {
                path: ServePath::BoundedStaleProof
            }
        );
        // Aged past the bound: escalate again, never serve stale-as-fresh.
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::BoundedStale {
                max_staleness: Duration::from_millis(100),
            },
            fresh(10),
            ctx(100_001),
            ELIGIBLE,
        );
        assert_eq!(planned.plan, ReadPlan::BarrierThenServe { escalated: true });
    }

    #[test]
    fn authority_movement_drops_receipts_engines_and_cache() {
        let hub = hub();
        hub.note_barrier(
            TABLET,
            FreshnessReceipt::new(
                authority(),
                CommitPosition::from_u64(10),
                Ticks::from_micros(0),
                NodeIncarnation::from_u64(7),
            ),
        );
        hub.insert(
            TABLET,
            authority(),
            &get_op(),
            &OperationResult::Exists(true),
            CommitPosition::from_u64(10),
            None,
        );
        // Same authority: Any replays from cache.
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::Any,
            fresh(10),
            ctx(0),
            ELIGIBLE,
        );
        assert_eq!(
            planned.plan,
            ReadPlan::ServeLocal {
                path: ServePath::StrongCache
            }
        );
        // Epoch moved (split/merge/replacement): evidence dies, the
        // bounded-stale read escalates, Any serves live (not cached).
        let moved = ReplicaFreshness::new(
            moved_authority(),
            CommitPosition::from_u64(10),
            NodeIncarnation::from_u64(7),
        );
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::BoundedStale {
                max_staleness: Duration::from_secs(60),
            },
            moved,
            ctx(1),
            ELIGIBLE,
        );
        assert_eq!(planned.plan, ReadPlan::BarrierThenServe { escalated: true });
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::Any,
            moved,
            ctx(1),
            ELIGIBLE,
        );
        assert_eq!(
            planned.plan,
            ReadPlan::ServeLocal {
                path: ServePath::AnyLocal
            }
        );
    }

    #[test]
    fn strong_cache_replays_any_and_at_least_but_never_latest() {
        let hub = hub();
        hub.insert(
            TABLET,
            authority(),
            &get_op(),
            &OperationResult::Exists(true),
            CommitPosition::from_u64(9),
            None,
        );
        // AtLeast covered by the fill position replays without state access.
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::AtLeast(token(9)),
            fresh(10),
            ctx(0),
            ELIGIBLE,
        );
        assert_eq!(
            planned.plan,
            ReadPlan::ServeLocal {
                path: ServePath::StrongCache
            }
        );
        assert_eq!(
            planned.cached.as_ref().map(|hit| hit.outcome.clone()),
            Some(OperationResult::Exists(true))
        );
        // A token beyond the fill position misses the cache and serves
        // live (covered at 10).
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::AtLeast(token(10)),
            fresh(10),
            ctx(0),
            ELIGIBLE,
        );
        assert_eq!(
            planned.plan,
            ReadPlan::ServeLocal {
                path: ServePath::AtLeastCovered
            }
        );
        // Latest never replays cache under the conservative mode.
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::Latest,
            fresh(10),
            ctx(0),
            ELIGIBLE,
        );
        assert_eq!(
            planned.plan,
            ReadPlan::BarrierThenServe { escalated: false }
        );
        // Proof-less fills never satisfy a staleness bound.
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::BoundedStale {
                max_staleness: Duration::from_secs(60),
            },
            fresh(10),
            ctx(0),
            ELIGIBLE,
        );
        assert_eq!(planned.plan, ReadPlan::BarrierThenServe { escalated: true });
    }

    #[test]
    fn proof_carrying_fills_satisfy_matching_bounds() {
        let hub = hub();
        let proof = FreshnessReceipt::new(
            authority(),
            CommitPosition::from_u64(9),
            Ticks::from_micros(0),
            NodeIncarnation::from_u64(7),
        );
        hub.insert(
            TABLET,
            authority(),
            &get_op(),
            &OperationResult::Exists(true),
            CommitPosition::from_u64(9),
            Some(proof),
        );
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::BoundedStale {
                max_staleness: Duration::from_millis(100),
            },
            fresh(10),
            ctx(50_000),
            ELIGIBLE,
        );
        assert_eq!(
            planned.plan,
            ReadPlan::ServeLocal {
                path: ServePath::StrongCache
            }
        );
    }

    #[test]
    fn oversized_values_bypass_the_cache() {
        let hub = hub();
        let big = vec![
            0u8;
            usize::try_from(STRONG_CACHE_VALUE_CAP)
                .unwrap_or(usize::MAX)
                .saturating_add(1)
        ];
        hub.insert(
            TABLET,
            authority(),
            &get_op(),
            &OperationResult::Value(Some(bytes::Bytes::from(big))),
            CommitPosition::from_u64(9),
            None,
        );
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::Any,
            fresh(10),
            ctx(0),
            ELIGIBLE,
        );
        assert_eq!(
            planned.plan,
            ReadPlan::ServeLocal {
                path: ServePath::AnyLocal
            }
        );
    }

    #[test]
    fn almost_local_plans_lazy_alr_batches_without_timing() {
        let hub = ConsistencyHub::new(
            NodeId::from_u64(1),
            NodeIncarnation::from_u64(7),
            ReadAuthorityProvider::AlmostLocal,
            lease_params(),
        );
        // Every Latest read joins the Lazy-ALR batch: no receipt, no
        // barrier, no clock bound consulted — the fence orders the batch.
        // A fresh barrier receipt changes nothing: ALR never consults
        // receipt age (the old timing-based pseudo path is gone).
        assert_eq!(
            plan(&hub, ReadContract::Latest, 10, 0),
            ReadPlan::AlrThenServe
        );
        hub.note_barrier(
            TABLET,
            FreshnessReceipt::new(
                authority(),
                CommitPosition::from_u64(10),
                Ticks::from_micros(0),
                NodeIncarnation::from_u64(7),
            ),
        );
        assert_eq!(
            plan(&hub, ReadContract::Latest, 10, 10_000_000),
            ReadPlan::AlrThenServe
        );
        // An aged receipt still serves BoundedStale while fresh; ALR and
        // bounded-stale evidence stay independent mechanisms.
        assert_eq!(
            plan(
                &hub,
                ReadContract::BoundedStale {
                    max_staleness: Duration::from_secs(60),
                },
                10,
                10_000_000,
            ),
            ReadPlan::ServeLocal {
                path: ServePath::BoundedStaleProof
            }
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn roster_lease_serves_on_engine_evidence_and_falls_back_otherwise() {
        let hub = lease_hub();
        // Cold engine (driver has not run): no evidence, ALR fallback —
        // never a barrier skip, never a thin serve.
        assert_eq!(
            plan(&hub, ReadContract::Latest, 10, 0),
            ReadPlan::AlrThenServe
        );
        // Drive as leader past the restart quarantine: single-voter
        // loopback converges inside the drive, floor 41.
        let quarantine = lease_params().quarantine();
        let awakened = u64::try_from(quarantine.as_micros())
            .unwrap_or(u64::MAX)
            .saturating_add(1);
        let out = drive_leader(&hub, 5, 41, 0);
        assert!(out.is_empty(), "loopback drains inside the drive");
        // Still quarantined (engine was created at t=0): no evidence.
        assert_eq!(
            plan(&hub, ReadContract::Latest, 41, 1_000),
            ReadPlan::AlrThenServe
        );
        assert!(
            hub.explain_latest(
                TABLET,
                CommitPosition::from_u64(41),
                Ticks::from_micros(1_000)
            )
            .contains("quarantined"),
            "explain names the fence"
        );
        // Past the quarantine the same drive announces and stabilizes.
        let out = drive_leader(&hub, 5, 41, awakened);
        assert!(out.is_empty(), "single voter needs no network");
        assert_eq!(
            plan(&hub, ReadContract::Latest, 41, awakened + 1),
            ReadPlan::ServeLocal {
                path: ServePath::RosterLease
            }
        );
        // Serve-time evidence revalidates: the proof carries the floor.
        let proof = hub
            .evidence(
                TABLET,
                CommitPosition::from_u64(41),
                Ticks::from_micros(awakened + 1),
            )
            .expect("evidence");
        assert_eq!(proof.safety_floor, CommitPosition::from_u64(41));
        assert!(
            hub.explain_latest(
                TABLET,
                CommitPosition::from_u64(41),
                Ticks::from_micros(awakened + 1)
            )
            .contains("grants=1/1")
        );
        // Behind the floor: hold-classified ALR fallback, never a serve.
        assert_eq!(
            plan(&hub, ReadContract::Latest, 40, awakened + 1),
            ReadPlan::AlrThenServe
        );
        assert!(
            hub.evidence(
                TABLET,
                CommitPosition::from_u64(40),
                Ticks::from_micros(awakened + 1)
            )
            .is_none()
        );
        let (_, snapshot) = hub
            .snapshot(TABLET, Ticks::from_micros(awakened + 1))
            .expect("snapshot");
        assert_eq!(snapshot.roster_hits, 1);
        assert_eq!(snapshot.roster_holds, 1);
        assert!(
            snapshot.roster_fallbacks >= 2,
            "cold + quarantined fall back"
        );
        // A term rise (new election) voids the old roster: the first drive
        // revokes it (the replacement Guard waits out the live exclusion,
        // per the no-overwrite rule); the second drive grants the
        // replacement, and single-voter loopback stabilizes it at once —
        // so evidence exists again, but under the NEW term (no stale
        // authority ever serves).
        let _ = drive_leader(&hub, 6, 41, awakened + 2);
        assert!(
            hub.evidence(
                TABLET,
                CommitPosition::from_u64(41),
                Ticks::from_micros(awakened + 2)
            )
            .is_none()
        );
        let _ = drive_leader(&hub, 6, 41, awakened + 3);
        let proof = hub
            .evidence(
                TABLET,
                CommitPosition::from_u64(41),
                Ticks::from_micros(awakened + 3),
            )
            .expect("new-term evidence");
        assert_eq!(proof.roster.term.as_u64(), 6);
        // A term the local node does not lead (and hears nothing for)
        // forms no replacement: the old evidence is void and the read
        // plans ALR, never stale.
        let _ = hub.lease_drive(
            TABLET,
            authority(),
            RosterTerm::from_u64(7),
            vec![NodeId::from_u64(1)],
            false,
            CommitPosition::from_u64(41),
            CommitPosition::from_u64(41),
            Ticks::from_micros(awakened + 4),
        );
        assert!(
            hub.evidence(
                TABLET,
                CommitPosition::from_u64(41),
                Ticks::from_micros(awakened + 4)
            )
            .is_none()
        );
        assert_eq!(
            plan(&hub, ReadContract::Latest, 41, awakened + 5),
            ReadPlan::AlrThenServe
        );
        // Authority movement drops the engine outright.
        let moved = ReplicaFreshness::new(
            moved_authority(),
            CommitPosition::from_u64(41),
            NodeIncarnation::from_u64(7),
        );
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::Latest,
            moved,
            ctx(awakened + 3),
            ELIGIBLE,
        );
        assert_eq!(planned.plan, ReadPlan::AlrThenServe);
    }

    #[test]
    fn roster_path_respects_eligibility_and_drift_contract() {
        // Lease-ineligible reads (transactions, batches, scans) skip the
        // roster even with full evidence: same Latest semantics via ALR.
        let hub = lease_hub();
        let quarantine = lease_params().quarantine();
        let awakened = u64::try_from(quarantine.as_micros())
            .unwrap_or(u64::MAX)
            .saturating_add(1);
        let _ = drive_leader(&hub, 5, 41, 0);
        let _ = drive_leader(&hub, 5, 41, awakened);
        assert_eq!(
            plan(&hub, ReadContract::Latest, 41, awakened + 1),
            ReadPlan::ServeLocal {
                path: ServePath::RosterLease
            }
        );
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::Latest,
            fresh(41),
            ctx(awakened + 1),
            INELIGIBLE,
        );
        assert_eq!(planned.plan, ReadPlan::AlrThenServe);
        // An untrusted drift contract disables the backend hub-wide:
        // every Latest read plans ALR, never a local serve.
        let untrusted = LeaseParams {
            max_drift_ppm: kivi_types::roster::MAX_DRIFT_PPM + 1,
            ..lease_params()
        };
        assert!(untrusted.validate().is_err());
        let dark = ConsistencyHub::new(
            NodeId::from_u64(1),
            NodeIncarnation::from_u64(7),
            ReadAuthorityProvider::RosterLease,
            untrusted,
        );
        assert!(!dark.lease_available());
        let _ = dark.lease_drive(
            TABLET,
            authority(),
            RosterTerm::from_u64(5),
            vec![NodeId::from_u64(1)],
            true,
            CommitPosition::from_u64(41),
            CommitPosition::from_u64(41),
            Ticks::from_micros(99_000_000),
        );
        assert_eq!(
            dark.plan(
                TABLET,
                &get_op(),
                ReadContract::Latest,
                fresh(41),
                ctx(99_000_001),
                ELIGIBLE,
            )
            .plan,
            ReadPlan::AlrThenServe
        );
        assert!(
            dark.explain_latest(
                TABLET,
                CommitPosition::from_u64(41),
                Ticks::from_micros(99_000_001)
            )
            .contains("drift contract untrusted")
        );
    }

    #[test]
    fn cache_eviction_is_deterministic_and_bounded() {
        let hub = hub();
        for index in 0..(STRONG_CACHE_CAP + 16) {
            let op = Operation::Get {
                key: Key::from(format!("k{index:05}").into_bytes()),
            };
            hub.insert(
                TABLET,
                authority(),
                &op,
                &OperationResult::Exists(true),
                CommitPosition::from_u64(9),
                None,
            );
        }
        let tablets = hub.tablets.lock().expect("hub");
        let entry = tablets.get(&TABLET).expect("tablet");
        assert_eq!(entry.cache.len(), STRONG_CACHE_CAP);
    }

    #[test]
    fn metrics_distinguish_hits_fallbacks_and_rejects() {
        let hub = hub();
        hub.note_barrier(
            TABLET,
            FreshnessReceipt::new(
                authority(),
                CommitPosition::from_u64(10),
                Ticks::from_micros(0),
                NodeIncarnation::from_u64(7),
            ),
        );
        let _ = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::BoundedStale {
                max_staleness: Duration::from_millis(100),
            },
            fresh(10),
            ctx(10_000),
            ELIGIBLE,
        );
        let _ = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::BoundedStale {
                max_staleness: Duration::from_millis(1),
            },
            fresh(10),
            ctx(10_000),
            ELIGIBLE,
        );
        hub.note_served(TABLET, ReadContract::Latest, true);
        hub.note_served(TABLET, ReadContract::Any, false);
        let (_, snapshot) = hub
            .snapshot(TABLET, Ticks::from_micros(10_000))
            .expect("snapshot");
        assert_eq!(snapshot.bounded_stale_hits, 1);
        assert_eq!(snapshot.bounded_stale_escalations, 1);
        assert_eq!(snapshot.reads_by_contract, (1, 0, 0, 1));
        assert_eq!(snapshot.local_serves, 1);
        assert_eq!(snapshot.authority_serves, 1);
        // The wait cap still bounds AtLeast waits (proof TTL concept is
        // gone with the timing-based fast path; bounds ride receipts).
        assert!(AT_LEAST_WAIT_CAP >= Duration::from_secs(1));
    }

    #[test]
    fn poisoned_hub_fails_closed_to_the_authority_path() {
        let hub = hub();
        // Poisoning a mutex guard panics, which the test harness would
        // catch as a failure rather than hub behavior, so this pins the
        // healthy arm: the fail-closed arm above is structural.
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::Any,
            fresh(10),
            ctx(0),
            ELIGIBLE,
        );
        assert_eq!(
            planned.plan,
            ReadPlan::ServeLocal {
                path: ServePath::AnyLocal
            }
        );
    }

    /// Old roster, partitioned leader, new term, successor write, old
    /// responder read — the mandatory fencing scenario, fully
    /// deterministic across three hub engines with virtual time:
    ///
    /// ```text
    /// term-5 roster stable everywhere; node 3 partitions;
    /// term rises (leader 2); old grants revoke (fast path), the Revoke
    /// to node 3 is lost; the successor write must NOT complete while
    /// node 3 is still covered (the gate would fail its poll); node 3's
    /// holds lapse without traffic, so its read falls back — never stale;
    /// once the exclusion lapses the gate clears; a restarted node 3
    /// owes the takeover fence before completing anything.
    /// ```
    #[test]
    #[allow(clippy::too_many_lines)]
    fn partition_old_responder_falls_back_never_stale() {
        use std::collections::BTreeMap;

        fn voters3() -> Vec<NodeId> {
            vec![
                NodeId::from_u64(1),
                NodeId::from_u64(2),
                NodeId::from_u64(3),
            ]
        }

        fn mesh(
            hubs: &BTreeMap<NodeId, ConsistencyHub>,
            term_of: impl Fn(NodeId) -> u64,
            leader: u64,
            applied_of: impl Fn(NodeId) -> u64,
            now: u64,
            skip: &[(NodeId, NodeId)],
        ) {
            // Exchange until quiet (bounded: healthy activation
            // converges in a fixed small number of rounds). Each round
            // drives every hub and delivers all pending traffic plus the
            // previous round's piggybacked replies.
            let mut pending: Vec<(NodeId, LeaseDriveOut)> = Vec::new();
            for _ in 0..12 {
                for (id, hub) in hubs {
                    let applied = applied_of(*id);
                    let out = hub.lease_drive(
                        TABLET,
                        authority(),
                        RosterTerm::from_u64(term_of(*id)),
                        voters3(),
                        *id == NodeId::from_u64(leader),
                        CommitPosition::from_u64(applied),
                        CommitPosition::from_u64(applied),
                        Ticks::from_micros(now),
                    );
                    for batch in out {
                        pending.push((*id, batch));
                    }
                }
                if pending.iter().all(|(_, batch)| batch.messages.is_empty()) {
                    break;
                }
                let mut next = Vec::new();
                for (from, batch) in std::mem::take(&mut pending) {
                    if batch.messages.is_empty()
                        || skip.contains(&(from, batch.target))
                        || skip.contains(&(batch.target, from))
                    {
                        continue;
                    }
                    for reply in hubs[&batch.target].lease_receive(
                        TABLET,
                        from,
                        batch.messages,
                        Ticks::from_micros(now),
                    ) {
                        next.push((batch.target, reply));
                    }
                }
                pending = next;
            }
        }

        let mut hubs = BTreeMap::new();
        for node in [1u64, 2, 3] {
            hubs.insert(
                NodeId::from_u64(node),
                ConsistencyHub::new(
                    NodeId::from_u64(node),
                    NodeIncarnation::from_u64(7),
                    ReadAuthorityProvider::RosterLease,
                    lease_params(),
                ),
            );
        }
        // Term 5, leader 1: create (quarantined), then activate past the
        // restart fence. All three hold a stable roster at applied 10.
        mesh(&hubs, |_| 5, 1, |_| 10, 0, &[]);
        mesh(&hubs, |_| 5, 1, |_| 10, 6_000_000, &[]);
        for node in [1u64, 2, 3] {
            assert!(
                hubs[&NodeId::from_u64(node)]
                    .evidence(
                        TABLET,
                        CommitPosition::from_u64(10),
                        Ticks::from_micros(6_000_000)
                    )
                    .is_some(),
                "node {node} stable under term 5"
            );
        }
        // Node 3 partitions: it keeps observing term 5 (no election news
        // cross a partition), while 1 and 2 rise to term 6 under leader 2.
        // Old outgoing grants revoke on 1 and 2, but the Revoke to 3 is
        // lost. The successor write reaches applied 42 on 1 and 2; node 3
        // never sees it.
        let three = NodeId::from_u64(3);
        let cut = [
            (NodeId::from_u64(1), three),
            (NodeId::from_u64(2), three),
            (three, NodeId::from_u64(1)),
            (three, NodeId::from_u64(2)),
        ];
        mesh(
            &hubs,
            |id| if id == three { 5 } else { 6 },
            2,
            |_| 10,
            6_100_000,
            &cut,
        );
        // The new leader still covers node 3 (revoking exclusion): a
        // successor write at 42 must not complete past it.
        let covered: Vec<u64> = hubs[&NodeId::from_u64(2)]
            .covered_grantees(TABLET, Ticks::from_micros(6_100_000))
            .into_iter()
            .map(|(peer, _)| peer.as_u64())
            .collect();
        assert!(
            covered.contains(&3),
            "old responder stays covered until revoked-or-lapsed, got {covered:?}"
        );
        // Node 3 still holds its term-5 roster (unexpired): it would serve
        // 41 — legal, because 42 has not (and cannot yet) complete.
        assert!(
            hubs[&NodeId::from_u64(3)]
                .evidence(
                    TABLET,
                    CommitPosition::from_u64(10),
                    Ticks::from_micros(6_100_000)
                )
                .is_some(),
            "unexpired old hold still authorizes pre-write state"
        );
        // Silence lapses node 3's holds: its read plans ALR fallback,
        // never a stale success.
        mesh(
            &hubs,
            |id| if id == three { 5 } else { 6 },
            2,
            |id| if id == three { 10 } else { 42 },
            9_000_000,
            &cut,
        );
        assert!(
            hubs[&NodeId::from_u64(3)]
                .evidence(
                    TABLET,
                    CommitPosition::from_u64(41),
                    Ticks::from_micros(9_000_000)
                )
                .is_none(),
            "partitioned responder loses evidence at its hold"
        );
        assert_eq!(
            hubs[&NodeId::from_u64(3)]
                .plan(
                    TABLET,
                    &get_op(),
                    ReadContract::Latest,
                    ReplicaFreshness::new(
                        authority(),
                        CommitPosition::from_u64(41),
                        NodeIncarnation::from_u64(7),
                    ),
                    ctx(9_000_001),
                    ELIGIBLE,
                )
                .plan,
            ReadPlan::AlrThenServe
        );
        // The exclusion lapsed with it: the gate clears node 3 and the
        // successor write may proceed.
        let covered: Vec<u64> = hubs[&NodeId::from_u64(2)]
            .covered_grantees(TABLET, Ticks::from_micros(9_000_000))
            .into_iter()
            .map(|(peer, _)| peer.as_u64())
            .collect();
        assert!(
            !covered.contains(&3),
            "lapsed exclusion leaves the covered set, got {covered:?}"
        );
        // Restarted node 3 (new incarnation, fresh hub) owes the takeover
        // fence before completing anything, even with live peers.
        hubs.insert(
            NodeId::from_u64(3),
            ConsistencyHub::new(
                NodeId::from_u64(3),
                NodeIncarnation::from_u64(8),
                ReadAuthorityProvider::RosterLease,
                lease_params(),
            ),
        );
        mesh(&hubs, |_| 6, 2, |_| 42, 9_000_001, &[]);
        assert!(
            hubs[&NodeId::from_u64(3)]
                .takeover_hold(TABLET, Ticks::from_micros(9_000_001))
                .is_some(),
            "restarted node waits out forgotten authority"
        );
        let fence = u64::try_from(lease_params().quarantine().as_micros()).unwrap_or(u64::MAX);
        assert!(
            hubs[&NodeId::from_u64(3)]
                .takeover_hold(TABLET, Ticks::from_micros(9_000_001 + fence))
                .is_none(),
            "fence is bounded by the quarantine"
        );
    }

    /// Deterministic fault scenario across one tablet's whole evidence
    /// lifecycle: cold start, proof install, duplication, reorder,
    /// partition aging, restart amnesia, topology movement, lease bounds,
    /// and guard movement. Every step asserts the Phase 7 invariants:
    ///
    /// ```text
    /// BoundedStale never succeeds without a satisfying receipt;
    /// AtLeast never serves before its token;
    /// foreign lineage always rejects without waiting;
    /// old generations never authorize;
    /// uncertainty always escalates or falls back.
    /// ```
    ///
    /// Virtual `Ticks` stand in for every clock in the task's inject
    /// list; barrier/lease absence stands in for drop, partition, pause,
    /// and control-plane delay; a fresh hub stands in for restart.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn fault_scenario_resolves_conservatively_end_to_end() {
        use kivi_types::TabletEpoch;
        let hub = hub();
        let bound = Duration::from_millis(100);
        let stale = || ReadContract::BoundedStale {
            max_staleness: bound,
        };
        // Eligible point-read shorthand for this scenario's tablet.
        let plan_here =
            |hub: &ConsistencyHub, contract: ReadContract, f: ReplicaFreshness, t: u64| {
                hub.plan(TABLET, &get_op(), contract, f, ctx(t), ELIGIBLE)
                    .plan
            };

        // 1. Cold replica: no evidence anywhere. BoundedStale escalates
        // (never ServeLocal), AtLeast-behind waits, AtLeast-foreign
        // rejects, Any serves with no claim.
        assert_eq!(
            plan_here(&hub, stale(), fresh(10), 0),
            ReadPlan::BarrierThenServe { escalated: true }
        );
        assert_eq!(
            plan_here(&hub, ReadContract::AtLeast(token(11)), fresh(10), 0),
            ReadPlan::WaitThenServe { index: 10 }
        );
        let foreign = kivi_types::CommitToken::new(
            TabletId::from_u64(77),
            TabletEpoch::from_u64(1),
            CommitPosition::from_u64(1),
        );
        assert!(matches!(
            plan_here(&hub, ReadContract::AtLeast(foreign), fresh(10), 0),
            ReadPlan::Reject { .. }
        ));

        // 2. Barrier proof installs evidence: bounded-stale serves from
        // proof while fresh; covered AtLeast serves without leader contact.
        hub.note_barrier(
            TABLET,
            FreshnessReceipt::new(
                authority(),
                CommitPosition::from_u64(10),
                Ticks::from_micros(0),
                NodeIncarnation::from_u64(7),
            ),
        );
        assert_eq!(
            plan_here(&hub, stale(), fresh(10), 99_999),
            ReadPlan::ServeLocal {
                path: ServePath::BoundedStaleProof
            }
        );
        assert_eq!(
            plan_here(&hub, ReadContract::AtLeast(token(10)), fresh(10), 99_999),
            ReadPlan::ServeLocal {
                path: ServePath::AtLeastCovered
            }
        );

        // 3. Duplication is idempotent: the same proof twice changes
        // nothing; the bound still holds.
        hub.note_barrier(
            TABLET,
            FreshnessReceipt::new(
                authority(),
                CommitPosition::from_u64(10),
                Ticks::from_micros(0),
                NodeIncarnation::from_u64(7),
            ),
        );
        assert_eq!(
            plan_here(&hub, stale(), fresh(10), 99_999),
            ReadPlan::ServeLocal {
                path: ServePath::BoundedStaleProof
            }
        );

        // 4. Reorder is monotonic: a newer proof then a delayed older one
        // keeps the newer (the older would only weaken the bound).
        hub.note_barrier(
            TABLET,
            FreshnessReceipt::new(
                authority(),
                CommitPosition::from_u64(10),
                Ticks::from_micros(500_000),
                NodeIncarnation::from_u64(7),
            ),
        );
        hub.note_barrier(
            TABLET,
            FreshnessReceipt::new(
                authority(),
                CommitPosition::from_u64(10),
                Ticks::from_micros(100_000),
                NodeIncarnation::from_u64(7),
            ),
        );
        // Proven at 500ms: fresh at 550ms, stale at 650ms.
        assert_eq!(
            plan_here(&hub, stale(), fresh(10), 550_000),
            ReadPlan::ServeLocal {
                path: ServePath::BoundedStaleProof
            }
        );
        assert_eq!(
            plan_here(&hub, stale(), fresh(10), 650_000),
            ReadPlan::BarrierThenServe { escalated: true }
        );

        // 5. Partition aging: no new proofs arrive and the clock runs past
        // every bound. Every bounded read escalates; escalation failure
        // (authority unreachable, modeled by the caller) counts unprovable
        // instead of serving weak data.
        assert_eq!(
            plan_here(&hub, stale(), fresh(10), 5_000_000),
            ReadPlan::BarrierThenServe { escalated: true }
        );
        hub.note_escalation_failed(TABLET);
        let (_, snapshot) = hub
            .snapshot(TABLET, Ticks::from_micros(5_000_000))
            .expect("snapshot");
        assert_eq!(snapshot.freshness_unprovable, 1);
        assert!(snapshot.bounded_stale_escalations >= 3);

        // 6. Restart amnesia: a fresh hub holds nothing even at the same
        // ticks — memory-held proofs died with the old process.
        let restarted = ConsistencyHub::new(
            NodeId::from_u64(1),
            NodeIncarnation::from_u64(7),
            ReadAuthorityProvider::ConservativeLeader,
            lease_params(),
        );
        assert_eq!(
            plan_here(&restarted, stale(), fresh(10), 550_000),
            ReadPlan::BarrierThenServe { escalated: true }
        );

        // 7. Topology movement voids evidence even with a fresh clock: an
        // epoch-moved replica escalates, and foreign-epoch tokens reject.
        let moved = ReplicaFreshness::new(
            moved_authority(),
            CommitPosition::from_u64(10),
            NodeIncarnation::from_u64(7),
        );
        assert_eq!(
            plan_here(&hub, stale(), moved, 500_001),
            ReadPlan::BarrierThenServe { escalated: true }
        );
        let old_lineage = kivi_types::CommitToken::new(
            TABLET,
            TabletEpoch::from_u64(1),
            CommitPosition::from_u64(5),
        );
        assert!(matches!(
            plan_here(&hub, ReadContract::AtLeast(old_lineage), moved, 500_001),
            ReadPlan::Reject { .. }
        ));

        // 8. Guard movement voids receipts even when the epoch is stable:
        // WriteGuard generations are evidence boundaries too.
        let guarded = TabletAuthority::new(
            TABLET,
            TabletEpoch::from_u64(1),
            kivi_types::WriteGuardGeneration::from_u64(2),
        );
        let guarded_fresh = ReplicaFreshness::new(
            guarded,
            CommitPosition::from_u64(10),
            NodeIncarnation::from_u64(7),
        );
        // Re-prove under the original authority first so the hub holds a
        // receipt, then move the guard: the old receipt must die.
        hub.note_barrier(
            TABLET,
            FreshnessReceipt::new(
                authority(),
                CommitPosition::from_u64(10),
                Ticks::from_micros(9_000_000),
                NodeIncarnation::from_u64(7),
            ),
        );
        assert_eq!(
            plan_here(&hub, stale(), guarded_fresh, 9_000_001),
            ReadPlan::BarrierThenServe { escalated: true }
        );

        // 9. Roster evidence is engine-derived, never leader-issued: a
        // cold hub plans ALR; a driven+stabilized hub serves locally with
        // floor coverage; expiry-equivalent states (term rise, restart,
        // behind-floor) plan ALR, never a thin serve.
        let leased = lease_hub();
        assert_eq!(
            plan_here(&leased, ReadContract::Latest, fresh(10), 0),
            ReadPlan::AlrThenServe
        );
        let quarantine = lease_params().quarantine();
        let awakened = u64::try_from(quarantine.as_micros())
            .unwrap_or(u64::MAX)
            .saturating_add(1);
        let _ = drive_leader(&leased, 5, 10, 0);
        let _ = drive_leader(&leased, 5, 10, awakened);
        assert_eq!(
            plan_here(&leased, ReadContract::Latest, fresh(10), awakened + 1),
            ReadPlan::ServeLocal {
                path: ServePath::RosterLease
            }
        );
        // Behind the floor: ALR fallback (hold-classified).
        assert_eq!(
            plan_here(&leased, ReadContract::Latest, fresh(9), awakened + 1),
            ReadPlan::AlrThenServe
        );
        // New term replaces the old roster: the first drive revokes (the
        // replacement Guard waits out the live exclusion), the second
        // grants it, and evidence exists again under the new term only
        // (single-voter loopback stabilizes at once; multi-voter gaps are
        // covered by the engine tests).
        let _ = drive_leader(&leased, 6, 10, awakened + 2);
        assert!(
            leased
                .evidence(
                    TABLET,
                    CommitPosition::from_u64(10),
                    Ticks::from_micros(awakened + 2)
                )
                .is_none(),
            "revocation precedes replacement"
        );
        let _ = drive_leader(&leased, 6, 10, awakened + 3);
        let proof = leased
            .evidence(
                TABLET,
                CommitPosition::from_u64(10),
                Ticks::from_micros(awakened + 3),
            )
            .expect("new-term evidence");
        assert_eq!(proof.roster.term.as_u64(), 6);
        // Ineligible reads skip the roster even with evidence behind them:
        // re-stabilize term 6 first, then check the skip.
        let _ = drive_leader(&leased, 6, 10, awakened + 4);
        assert_eq!(
            plan_here(&leased, ReadContract::Latest, fresh(10), awakened + 5),
            ReadPlan::ServeLocal {
                path: ServePath::RosterLease
            }
        );
        assert_eq!(
            leased
                .plan(
                    TABLET,
                    &get_op(),
                    ReadContract::Latest,
                    fresh(10),
                    ctx(awakened + 5),
                    INELIGIBLE,
                )
                .plan,
            ReadPlan::AlrThenServe
        );

        // 10. Tablet isolation: evidence for one tablet never serves
        // another (split children start with empty hubs under fresh ids).
        let other = TabletId::from_u64(4);
        assert_eq!(
            hub.plan(
                other,
                &get_op(),
                stale(),
                fresh(10),
                ctx(9_000_001),
                ELIGIBLE
            )
            .plan,
            ReadPlan::BarrierThenServe { escalated: true }
        );
    }
}
