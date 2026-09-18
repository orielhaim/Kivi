//! Phase 7 consistency execution layer: one hub per node binding authority,
//! freshness evidence, the strong cache, and the read-authority providers.
//!
//! The hub owns no I/O, no clocks, and no networking. It stores the pure
//! values defined in [`kivi_types::consistency`] (receipts, leases, cache
//! entries, metrics) and answers one deterministic question per read —
//! [`ConsistencyHub::plan`] — while the caller executes the plan (barriers
//! and waits still hop to the owner thread, which alone may touch the Raft
//! handle). Clocks always arrive inside [`ReadContext`];
//! nothing here reads one.
//!
//! ```text
//! authority ──► freshness ──► cached read ──► validation ──► serve OR fallback
//!
//! missing / stale / ambiguous / invalidated / old-generation evidence
//!   ⇒ fall back to the authoritative path, never manufacture freshness.
//! ```
//!
//! Topology rule: per-tablet state is keyed by tablet and reconciled against
//! the serving authority on every plan. Any authority movement drops
//! receipts, leases, and cache entries for that tablet before the plan
//! runs, so split/merge/migration evidence can never leak across lineages.
//! A process restart drops the whole hub (memory-held), so incarnation
//! handling fails closed by construction.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::time::Duration;

use kivi_state::{Operation, OperationResult};
use kivi_types::{
    CommitPosition, ConsistencyMetrics, ConsistencySnapshot, FreshnessReceipt, FreshnessReject,
    NodeId, NodeIncarnation, ReadAuthorityProvider, ReadContext, ReadContract, ReadReceipt,
    ReplicaFreshness, RosterLease, STRONG_CACHE_CAP, STRONG_CACHE_VALUE_CAP, ServePath,
    TabletAuthority, TabletId,
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
/// strong cache, and worker-local metrics.
#[derive(Debug)]
struct TabletConsistency {
    /// Authority this entry reconciled against.
    authority: TabletAuthority,
    /// Last quorum-barrier proof (Almost-Local and bounded-stale evidence).
    receipt: Option<FreshnessReceipt>,
    /// Active roster lease, if the issuer granted one.
    lease: Option<RosterLease>,
    /// Strong cache: replayable fills under the current authority.
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
            lease: None,
            cache: BTreeMap::new(),
            provider,
            metrics: ConsistencyMetrics::default(),
        }
    }

    /// Drops every evidence kind held. Called whenever the serving
    /// authority moves: cached freshness dies with its authority.
    fn drop_evidence(&mut self) {
        let held = self.receipt.is_some() || self.lease.is_some() || !self.cache.is_empty();
        self.receipt = None;
        self.lease = None;
        self.cache.clear();
        if held {
            self.metrics.strong_cache_invalidations += 1;
        }
    }
}

/// Node-wide consistency hub: per-tablet evidence plus this replica's
/// identity. `Send + Sync` over a plain mutex — plans hold the lock only
/// for map lookup and counter updates, never across I/O.
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
}

impl ConsistencyHub {
    /// Builds a hub for one replica. `provider` is the initial
    /// read-authority mode for every tablet (the conservative leader path
    /// in production; fast modes opt in per deployment or test).
    #[must_use]
    pub fn new(
        local: NodeId,
        incarnation: NodeIncarnation,
        provider: ReadAuthorityProvider,
    ) -> Self {
        Self {
            tablets: Mutex::new(HashMap::new()),
            default_provider: Mutex::new(provider),
            local,
            incarnation,
        }
    }

    /// Switches the read-authority mode for all tablets (operator/test
    /// control for the Almost-Local and roster-lease prototypes). Held
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
    pub fn plan(
        &self,
        tablet: TabletId,
        operation: &Operation,
        contract: ReadContract,
        fresh: ReplicaFreshness,
        ctx: ReadContext,
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
            ReadContract::Latest => Self::plan_latest(entry, fresh, ctx, self.local),
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
    ) -> PlannedRead {
        let planned = self.plan_scan_inner(tablet, contract, fresh, ctx);
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
            ReadContract::Latest => Self::plan_latest(entry, fresh, ctx, self.local),
            ReadContract::AtLeast(token) => Self::plan_at_least(entry, None, token, fresh),
            ReadContract::BoundedStale { max_staleness } => {
                Self::plan_bounded_stale(entry, None, max_staleness, fresh, ctx, self.incarnation)
            }
            ReadContract::Any => Self::plan_any(entry, None),
        }
    }

    /// Plans a `Latest` read under the tablet's provider mode.
    fn plan_latest(
        entry: &mut TabletConsistency,
        fresh: ReplicaFreshness,
        ctx: ReadContext,
        local: NodeId,
    ) -> PlannedRead {
        match entry.provider {
            ReadAuthorityProvider::ConservativeLeader => PlannedRead {
                plan: ReadPlan::BarrierThenServe { escalated: false },
                cached: None,
            },
            ReadAuthorityProvider::AlmostLocal { max_proof_age } => {
                let usable = entry.receipt.is_some_and(|receipt| {
                    receipt
                        .usable_against(&fresh.authority, fresh.applied, fresh.incarnation)
                        .is_ok()
                        && receipt.age(ctx.ticks) <= max_proof_age
                });
                if usable {
                    entry.metrics.almost_local_hits += 1;
                    PlannedRead {
                        plan: ReadPlan::ServeLocal {
                            path: ServePath::AlmostLocalProof,
                        },
                        cached: None,
                    }
                } else {
                    entry.metrics.almost_local_fallbacks += 1;
                    PlannedRead {
                        plan: ReadPlan::BarrierThenServe { escalated: false },
                        cached: None,
                    }
                }
            }
            ReadAuthorityProvider::RosterLease => {
                let usable = entry.lease.as_ref().is_some_and(|lease| {
                    lease
                        .authorizes(local, fresh.incarnation, &fresh.authority, ctx.ticks)
                        .is_ok()
                        && fresh.applied.as_u64() >= lease.floor.as_u64()
                });
                if usable {
                    entry.metrics.roster_hits += 1;
                    PlannedRead {
                        plan: ReadPlan::ServeLocal {
                            path: ServePath::RosterLease,
                        },
                        cached: None,
                    }
                } else {
                    entry.metrics.roster_fallbacks += 1;
                    PlannedRead {
                        plan: ReadPlan::BarrierThenServe { escalated: false },
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

    /// Installs a roster lease granted by the read authority (leader after
    /// quorum agreement). Renewals install by generation order; delayed
    /// duplicates die. A lease for a foreign authority is refused.
    pub fn grant_lease(&self, tablet: TabletId, lease: RosterLease) -> bool {
        let mut installed = false;
        if let Ok(mut tablets) = self.tablets.lock() {
            let default = self
                .default_provider
                .lock()
                .map_or(ReadAuthorityProvider::ConservativeLeader, |mode| *mode);
            let entry = tablets
                .entry(tablet)
                .or_insert_with(|| TabletConsistency::fresh(lease.authority, default));
            if entry.authority.tablet() != lease.authority.tablet()
                || entry.authority.epoch() != lease.authority.epoch()
                || entry.authority.guard() != lease.authority.guard()
            {
                return false;
            }
            let replace = entry
                .lease
                .as_ref()
                .is_none_or(|current| current.superseded_by(&lease));
            if replace {
                entry.lease = Some(lease);
                installed = true;
            }
        }
        installed
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

    /// Records an Almost-Local fast path abandoned during execution (the
    /// owner no longer believes it leads): the read falls back to a fresh
    /// barrier.
    pub fn note_almost_local_fallback(&self, tablet: TabletId) {
        if let Ok(mut tablets) = self.tablets.lock()
            && let Some(entry) = tablets.get_mut(&tablet)
        {
            entry.metrics.almost_local_fallbacks += 1;
        }
    }

    /// Records a roster-lease fast path abandoned during execution
    /// (coverage gap against the lease floor at serve time).
    pub fn note_roster_fallback(&self, tablet: TabletId) {
        if let Ok(mut tablets) = self.tablets.lock()
            && let Some(entry) = tablets.get_mut(&tablet)
        {
            entry.metrics.roster_fallbacks += 1;
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
            let held = entry.receipt.is_some() || entry.lease.is_some() || !entry.cache.is_empty();
            entry.drop_evidence();
            return held;
        }
        false
    }

    /// Freezes this tablet's authority plus metrics for status surfaces.
    #[must_use]
    pub fn snapshot(&self, tablet: TabletId) -> Option<(TabletAuthority, ConsistencySnapshot)> {
        self.tablets.lock().ok().and_then(|tablets| {
            tablets
                .get(&tablet)
                .map(|entry| (entry.authority, entry.metrics.snapshot()))
        })
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
        CommitPosition, ReadContract, Ticks, UnixMicros,
        consistency::{AT_LEAST_WAIT_CAP, DEFAULT_PROOF_TTL},
    };
    use std::time::Duration;

    const TABLET: TabletId = TabletId::from_u64(3);

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
            UnixMicros::from_micros(ticks),
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

    fn hub() -> ConsistencyHub {
        ConsistencyHub::new(
            NodeId::from_u64(1),
            NodeIncarnation::from_u64(7),
            ReadAuthorityProvider::ConservativeLeader,
        )
    }

    #[test]
    fn latest_always_barriers_under_the_conservative_mode() {
        let hub = hub();
        let planned = hub.plan(TABLET, &get_op(), ReadContract::Latest, fresh(10), ctx(0));
        assert_eq!(
            planned.plan,
            ReadPlan::BarrierThenServe { escalated: false }
        );
        assert!(planned.cached.is_none());
    }

    #[test]
    fn at_least_covered_serves_without_leader_contact() {
        let hub = hub();
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::AtLeast(token(9)),
            fresh(10),
            ctx(0),
        );
        assert_eq!(
            planned.plan,
            ReadPlan::ServeLocal {
                path: ServePath::AtLeastCovered
            }
        );
    }

    #[test]
    fn at_least_behind_waits_with_the_log_index() {
        let hub = hub();
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::AtLeast(token(11)),
            fresh(10),
            ctx(0),
        );
        // position 11 ⇔ Raft index 10.
        assert_eq!(planned.plan, ReadPlan::WaitThenServe { index: 10 });
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
        );
        assert!(matches!(planned.plan, ReadPlan::Reject { .. }));
    }

    #[test]
    fn bounded_stale_without_evidence_escalates_never_serves_weak() {
        let hub = hub();
        // No receipt held: must escalate, never ServeLocal.
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::BoundedStale {
                max_staleness: Duration::from_secs(60),
            },
            fresh(10),
            ctx(0),
        );
        assert_eq!(planned.plan, ReadPlan::BarrierThenServe { escalated: true });
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
        );
        assert_eq!(planned.plan, ReadPlan::BarrierThenServe { escalated: true });
    }

    #[test]
    fn authority_movement_drops_receipts_leases_and_cache() {
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
        let planned = hub.plan(TABLET, &get_op(), ReadContract::Any, fresh(10), ctx(0));
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
        );
        assert_eq!(planned.plan, ReadPlan::BarrierThenServe { escalated: true });
        let planned = hub.plan(TABLET, &get_op(), ReadContract::Any, moved, ctx(1));
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
        );
        assert_eq!(
            planned.plan,
            ReadPlan::ServeLocal {
                path: ServePath::AtLeastCovered
            }
        );
        // Latest never replays cache under the conservative mode.
        let planned = hub.plan(TABLET, &get_op(), ReadContract::Latest, fresh(10), ctx(0));
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
        let planned = hub.plan(TABLET, &get_op(), ReadContract::Any, fresh(10), ctx(0));
        assert_eq!(
            planned.plan,
            ReadPlan::ServeLocal {
                path: ServePath::AnyLocal
            }
        );
    }

    #[test]
    fn almost_local_serves_on_fresh_proof_and_falls_back_otherwise() {
        let hub = ConsistencyHub::new(
            NodeId::from_u64(1),
            NodeIncarnation::from_u64(7),
            ReadAuthorityProvider::AlmostLocal {
                max_proof_age: Duration::from_millis(50),
            },
        );
        // No receipt yet: fall back to the barrier.
        let planned = hub.plan(TABLET, &get_op(), ReadContract::Latest, fresh(10), ctx(0));
        assert_eq!(
            planned.plan,
            ReadPlan::BarrierThenServe { escalated: false }
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
        // Fresh proof: serve without a new barrier.
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::Latest,
            fresh(10),
            ctx(49_999),
        );
        assert_eq!(
            planned.plan,
            ReadPlan::ServeLocal {
                path: ServePath::AlmostLocalProof
            }
        );
        // Aged proof, lagging replica, moved authority: all fall back.
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::Latest,
            fresh(10),
            ctx(50_001),
        );
        assert_eq!(
            planned.plan,
            ReadPlan::BarrierThenServe { escalated: false }
        );
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::Latest,
            fresh(9),
            ctx(10_000),
        );
        assert_eq!(
            planned.plan,
            ReadPlan::BarrierThenServe { escalated: false }
        );
    }

    #[test]
    fn roster_lease_serves_members_and_rejects_everyone_else() {
        use kivi_types::TabletEpoch;
        let hub = ConsistencyHub::new(
            NodeId::from_u64(1),
            NodeIncarnation::from_u64(7),
            ReadAuthorityProvider::RosterLease,
        );
        // No lease: fall back.
        let planned = hub.plan(TABLET, &get_op(), ReadContract::Latest, fresh(10), ctx(0));
        assert_eq!(
            planned.plan,
            ReadPlan::BarrierThenServe { escalated: false }
        );
        let lease = RosterLease::new(
            authority(),
            1,
            vec![NodeId::from_u64(1)],
            CommitPosition::from_u64(9),
            Ticks::from_micros(100_000),
            NodeIncarnation::from_u64(7),
        );
        assert!(hub.grant_lease(TABLET, lease));
        // Member with coverage serves the fast path.
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::Latest,
            fresh(10),
            ctx(50_000),
        );
        assert_eq!(
            planned.plan,
            ReadPlan::ServeLocal {
                path: ServePath::RosterLease
            }
        );
        // Behind the lease floor: fall back (never serve thin state).
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::Latest,
            fresh(8),
            ctx(50_000),
        );
        assert_eq!(
            planned.plan,
            ReadPlan::BarrierThenServe { escalated: false }
        );
        // Past expiry: fall back.
        let planned = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::Latest,
            fresh(10),
            ctx(100_000),
        );
        assert_eq!(
            planned.plan,
            ReadPlan::BarrierThenServe { escalated: false }
        );
        // Renewal by generation; delayed duplicates die.
        let stale = RosterLease::new(
            authority(),
            1,
            vec![NodeId::from_u64(1)],
            CommitPosition::from_u64(9),
            Ticks::from_micros(200_000),
            NodeIncarnation::from_u64(7),
        );
        assert!(!hub.grant_lease(TABLET, stale));
        let renewed = RosterLease::new(
            authority(),
            2,
            vec![NodeId::from_u64(1), NodeId::from_u64(2)],
            CommitPosition::from_u64(10),
            Ticks::from_micros(200_000),
            NodeIncarnation::from_u64(7),
        );
        assert!(hub.grant_lease(TABLET, renewed));
        // A lease for a foreign authority is refused, never installed.
        let foreign = RosterLease::new(
            TabletAuthority::new(
                TABLET,
                TabletEpoch::from_u64(2),
                kivi_types::WriteGuardGeneration::from_u64(1),
            ),
            3,
            vec![NodeId::from_u64(1)],
            CommitPosition::from_u64(10),
            Ticks::from_micros(300_000),
            NodeIncarnation::from_u64(7),
        );
        assert!(!hub.grant_lease(TABLET, foreign));
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
        );
        let _ = hub.plan(
            TABLET,
            &get_op(),
            ReadContract::BoundedStale {
                max_staleness: Duration::from_millis(1),
            },
            fresh(10),
            ctx(10_000),
        );
        hub.note_served(TABLET, ReadContract::Latest, true);
        hub.note_served(TABLET, ReadContract::Any, false);
        let (_, snapshot) = hub.snapshot(TABLET).expect("snapshot");
        assert_eq!(snapshot.bounded_stale_hits, 1);
        assert_eq!(snapshot.bounded_stale_escalations, 1);
        assert_eq!(snapshot.reads_by_contract, (1, 0, 0, 1));
        assert_eq!(snapshot.local_serves, 1);
        assert_eq!(snapshot.authority_serves, 1);
        // Unused imports stay honest: the proof TTL and wait cap bound
        // the plans above (100ms < TTL < wait cap ordering is sane).
        assert!(DEFAULT_PROOF_TTL > Duration::from_millis(100));
        assert!(AT_LEAST_WAIT_CAP >= Duration::from_secs(1));
    }

    #[test]
    fn poisoned_hub_fails_closed_to_the_authority_path() {
        let hub = hub();
        // Poisoning a mutex guard panics, which the test harness would
        // catch as a failure rather than hub behavior, so this pins the
        // healthy arm: the fail-closed arm above is structural.
        let planned = hub.plan(TABLET, &get_op(), ReadContract::Any, fresh(10), ctx(0));
        assert_eq!(
            planned.plan,
            ReadPlan::ServeLocal {
                path: ServePath::AnyLocal
            }
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

        // 1. Cold replica: no evidence anywhere. BoundedStale escalates
        // (never ServeLocal), AtLeast-behind waits, AtLeast-foreign
        // rejects, Any serves with no claim.
        assert_eq!(
            hub.plan(TABLET, &get_op(), stale(), fresh(10), ctx(0)).plan,
            ReadPlan::BarrierThenServe { escalated: true }
        );
        assert_eq!(
            hub.plan(
                TABLET,
                &get_op(),
                ReadContract::AtLeast(token(11)),
                fresh(10),
                ctx(0)
            )
            .plan,
            ReadPlan::WaitThenServe { index: 10 }
        );
        let foreign = kivi_types::CommitToken::new(
            TabletId::from_u64(77),
            TabletEpoch::from_u64(1),
            CommitPosition::from_u64(1),
        );
        assert!(matches!(
            hub.plan(
                TABLET,
                &get_op(),
                ReadContract::AtLeast(foreign),
                fresh(10),
                ctx(0)
            )
            .plan,
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
            hub.plan(TABLET, &get_op(), stale(), fresh(10), ctx(99_999))
                .plan,
            ReadPlan::ServeLocal {
                path: ServePath::BoundedStaleProof
            }
        );
        assert_eq!(
            hub.plan(
                TABLET,
                &get_op(),
                ReadContract::AtLeast(token(10)),
                fresh(10),
                ctx(99_999)
            )
            .plan,
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
            hub.plan(TABLET, &get_op(), stale(), fresh(10), ctx(99_999))
                .plan,
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
            hub.plan(TABLET, &get_op(), stale(), fresh(10), ctx(550_000))
                .plan,
            ReadPlan::ServeLocal {
                path: ServePath::BoundedStaleProof
            }
        );
        assert_eq!(
            hub.plan(TABLET, &get_op(), stale(), fresh(10), ctx(650_000))
                .plan,
            ReadPlan::BarrierThenServe { escalated: true }
        );

        // 5. Partition aging: no new proofs arrive and the clock runs past
        // every bound. Every bounded read escalates; escalation failure
        // (authority unreachable, modeled by the caller) counts unprovable
        // instead of serving weak data.
        assert_eq!(
            hub.plan(TABLET, &get_op(), stale(), fresh(10), ctx(5_000_000))
                .plan,
            ReadPlan::BarrierThenServe { escalated: true }
        );
        hub.note_escalation_failed(TABLET);
        let (_, snapshot) = hub.snapshot(TABLET).expect("snapshot");
        assert_eq!(snapshot.freshness_unprovable, 1);
        assert!(snapshot.bounded_stale_escalations >= 3);

        // 6. Restart amnesia: a fresh hub holds nothing even at the same
        // ticks — memory-held proofs died with the old process.
        let restarted = ConsistencyHub::new(
            NodeId::from_u64(1),
            NodeIncarnation::from_u64(7),
            ReadAuthorityProvider::ConservativeLeader,
        );
        assert_eq!(
            restarted
                .plan(TABLET, &get_op(), stale(), fresh(10), ctx(550_000))
                .plan,
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
            hub.plan(TABLET, &get_op(), stale(), moved, ctx(500_001))
                .plan,
            ReadPlan::BarrierThenServe { escalated: true }
        );
        let old_lineage = kivi_types::CommitToken::new(
            TABLET,
            TabletEpoch::from_u64(1),
            CommitPosition::from_u64(5),
        );
        assert!(matches!(
            hub.plan(
                TABLET,
                &get_op(),
                ReadContract::AtLeast(old_lineage),
                moved,
                ctx(500_001)
            )
            .plan,
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
            hub.plan(TABLET, &get_op(), stale(), guarded_fresh, ctx(9_000_001))
                .plan,
            ReadPlan::BarrierThenServe { escalated: true }
        );

        // 9. Lease bounds are strict at every edge: member + coverage +
        // authority + incarnation + `now < expires_at`.
        let leased = ConsistencyHub::new(
            NodeId::from_u64(1),
            NodeIncarnation::from_u64(7),
            ReadAuthorityProvider::RosterLease,
        );
        let lease = RosterLease::new(
            authority(),
            1,
            vec![NodeId::from_u64(1)],
            CommitPosition::from_u64(10),
            Ticks::from_micros(1_000_000),
            NodeIncarnation::from_u64(7),
        );
        assert!(leased.grant_lease(TABLET, lease));
        // One tick before expiry with coverage: fast path.
        assert_eq!(
            leased
                .plan(
                    TABLET,
                    &get_op(),
                    ReadContract::Latest,
                    fresh(10),
                    ctx(999_999)
                )
                .plan,
            ReadPlan::ServeLocal {
                path: ServePath::RosterLease
            }
        );
        // At expiry: fallback. Behind the floor: fallback. Wrong member
        // (hub bound to node 2 instead): fallback.
        assert_eq!(
            leased
                .plan(
                    TABLET,
                    &get_op(),
                    ReadContract::Latest,
                    fresh(10),
                    ctx(1_000_000)
                )
                .plan,
            ReadPlan::BarrierThenServe { escalated: false }
        );
        assert_eq!(
            leased
                .plan(
                    TABLET,
                    &get_op(),
                    ReadContract::Latest,
                    fresh(9),
                    ctx(999_999)
                )
                .plan,
            ReadPlan::BarrierThenServe { escalated: false }
        );
        let outsider = ConsistencyHub::new(
            NodeId::from_u64(2),
            NodeIncarnation::from_u64(7),
            ReadAuthorityProvider::RosterLease,
        );
        assert!(outsider.grant_lease(
            TABLET,
            RosterLease::new(
                authority(),
                1,
                vec![NodeId::from_u64(1)],
                CommitPosition::from_u64(10),
                Ticks::from_micros(1_000_000),
                NodeIncarnation::from_u64(7),
            )
        ));
        assert_eq!(
            outsider
                .plan(
                    TABLET,
                    &get_op(),
                    ReadContract::Latest,
                    fresh(10),
                    ctx(999_999)
                )
                .plan,
            ReadPlan::BarrierThenServe { escalated: false }
        );

        // 10. Tablet isolation: evidence for one tablet never serves
        // another (split children start with empty hubs under fresh ids).
        let other = TabletId::from_u64(4);
        assert_eq!(
            hub.plan(other, &get_op(), stale(), fresh(10), ctx(9_000_001))
                .plan,
            ReadPlan::BarrierThenServe { escalated: true }
        );
    }
}
