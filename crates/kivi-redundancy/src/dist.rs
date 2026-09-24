//! Cluster-wide redundancy plane: protect, read, repair across holders.
//!
//! [`DistributedFabric`] is the cluster counterpart of the embedded
//! [`crate::RedundancyFabric`]: it plans layouts with the same
//! [`crate::plan_baseline`] and [`crate::plan_placement`] helpers, encodes
//! with the same replication/RS providers, verifies with the same content
//! hashes — but stores fragments on remote holders through a
//! [`crate::transport::FragmentTransport`] and publishes authority through a
//! [`crate::catalog::CatalogPublish`] instead of a local `CURRENT` file.
//!
//! The plane never becomes consensus: it never orders mutable state, never
//! touches Raft state, never erasure-codes Raft state, and never weakens any
//! durability gate. Every flow is `build → verify → publish → retire` with
//! generation fencing: stale completions cannot overwrite newer layouts,
//! and reads never republish.
//!
//! Reads are degraded-tolerant (any `required` pieces suffice) but strict:
//! every served fragment is content-verified against the published layout
//! and the decoded image is content-verified against the asset id before
//! exposure. Corruption is never treated as valid merely because bytes
//! arrived.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::future::join_all;
use kivi_codec::integrity::blake3_256;
use kivi_types::{NodeId, Ticks};

use crate::proto::{FragmentKey, FragmentReply, FragmentRpc, MAX_PROTECTED_KEYS, RefuseReason};
use crate::{
    AssetAssessment, AssetId, FailureScope, FragmentId, FragmentRecord, InformationAsset,
    NodeDescriptor, RedundancyError, RedundancyIntent, RedundancyLayout, RepairEngine,
    RepairReason, SchemeParams, assess_with_min_available,
    catalog::{CatalogPublish, LayoutCatalog, PublishedLayout},
    healing::{
        ControllerSnapshot, DebtCounters, MaintenanceBudgets, MaintenanceReport, RepairDebt,
        RepairDebtEntry, RepairRisk, SchemeRequirements, ScrubMode, ScrubSchedule, ScrubTarget,
    },
    independent_survivable, plan_baseline, plan_placement_with_locality, reconstruction_targets,
    transport::{FragmentTransport, PeerTarget, TransportFailure},
};

/// Cluster plane configuration: bounds for remote work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistConfig {
    /// Failure scope enforced for independence.
    pub scope: FailureScope,
    /// This node's id (`0` when the plane runs without a local identity).
    /// Reads prefer a local copy before crossing the mesh.
    pub local: NodeId,
    /// Maximum concurrent shard uploads.
    pub upload_concurrency: usize,
    /// Maximum concurrent shard fetches.
    pub fetch_concurrency: usize,
    /// Maximum memory one decode may use (data image bound).
    pub max_decode_bytes: u64,
    /// Deadline per RPC attempt.
    pub rpc_timeout: Duration,
    /// Maximum fragment reads per reconstruction.
    pub max_fragment_reads: usize,
}

impl DistConfig {
    /// Conservative defaults: 4-wide remote I/O, 64 MiB images, 30 s RPCs.
    #[must_use]
    pub const fn conservative() -> Self {
        Self {
            scope: FailureScope::Node,
            local: NodeId::from_u64(0),
            upload_concurrency: 4,
            fetch_concurrency: 4,
            max_decode_bytes: 64 * 1024 * 1024,
            rpc_timeout: Duration::from_secs(30),
            max_fragment_reads: 64,
        }
    }

    /// Validates bounds (nonzero concurrencies, nonzero timeout and caps).
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::InvalidParams`] on zero bounds.
    pub const fn validate(self) -> Result<(), RedundancyError> {
        if self.upload_concurrency == 0
            || self.fetch_concurrency == 0
            || self.max_decode_bytes == 0
            || self.max_fragment_reads == 0
        {
            return Err(RedundancyError::InvalidParams {
                detail: String::new(),
            });
        }
        if self.rpc_timeout.as_nanos() == 0 {
            return Err(RedundancyError::InvalidParams {
                detail: String::new(),
            });
        }
        Ok(())
    }
}

impl Default for DistConfig {
    fn default() -> Self {
        Self::conservative()
    }
}

/// Health assessment over installed state: the assessment, the desired
/// placement under the current view, and the probed actual holders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistributedAssessment {
    /// Health computed over actually-installed healthy fragments.
    pub assessment: AssetAssessment,
    /// Desired placement under the current view (drift reference).
    pub desired: crate::DesiredPlacement,
    /// Installed state: `(index, holder, healthy)` per published fragment.
    pub actual: Vec<(u32, NodeId, bool)>,
}

/// What one distributed repair installed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairReport {
    /// Asset repaired.
    pub asset: AssetId,
    /// Generation repaired.
    pub generation: u64,
    /// Fragments re-installed: `(index, holder)`.
    pub rebuilt: Vec<(u32, NodeId)>,
    /// A generation fence moved under the repair (work skipped, not lost:
    /// re-calling repair resumes against the new generation).
    pub skipped_stale: bool,
}

/// What one cluster-wide orphan sweep reclaimed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SweepReport {
    /// Fragments reclaimed across all reachable holders.
    pub swept: u64,
    /// Holders that did not answer (hygiene is best-effort).
    pub unreachable: Vec<NodeId>,
}

/// One conservative catalog-versus-physical-state reconciliation pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AntiEntropyReport {
    /// Physical entries inspected.
    pub inspected: u64,
    /// Entries matching the current published layout and holder.
    pub referenced: u64,
    /// Entries for a superseded accepted generation.
    pub stale_generation: u64,
    /// Entries with coordinates that no current layout references.
    pub orphaned: u64,
    /// Entries whose identity or length disagrees with layout metadata.
    pub wrong_identity: u64,
    /// Physical copies found on more than one holder.
    pub duplicate_copies: u64,
    /// Entries for which the holder could not be queried.
    pub unreachable: Vec<NodeId>,
    /// Orphan entries reclaimed by the age-protected sweep.
    pub reclaimed: u64,
    /// Orphan entries retained because authority or age was ambiguous.
    pub retained: u64,
}

/// One generation build: everything `build_publish` needs (bundled so the
/// helper stays under the argument bound).
struct BuildRequest<'a> {
    /// Asset under protection.
    asset: InformationAsset,
    /// Exact logical bytes.
    bytes: &'a [u8],
    /// Planned scheme.
    params: SchemeParams,
    /// Persisted minimum retrievable-fragment floor.
    min_available: u32,
    /// Placement locality requirement.
    locality: crate::LocalityRequirement,
    /// Allowed failure-domain labels.
    allowed: &'a [String],
    /// Cluster view for placement.
    view: &'a [NodeDescriptor],
    /// Current published record, if any.
    current: Option<&'a PublishedLayout>,
    /// Control generation publishing this build.
    control_gen: u64,
}

#[derive(Debug, Clone, Copy)]
struct ScrubHistory {
    last_metadata: Option<Instant>,
    last_full: Option<Instant>,
    last_success: Option<Instant>,
    last_metadata_tick: Option<Ticks>,
    last_full_tick: Option<Ticks>,
    last_success_tick: Option<Ticks>,
    degraded_since: Option<Instant>,
    next_attempt: Option<Instant>,
}

impl ScrubHistory {
    const fn new() -> Self {
        Self {
            last_metadata: None,
            last_full: None,
            last_success: None,
            last_metadata_tick: None,
            last_full_tick: None,
            last_success_tick: None,
            degraded_since: None,
            next_attempt: None,
        }
    }
}

#[derive(Debug, Clone)]
struct HealingTask {
    asset: AssetId,
    generation: u64,
    risk: RepairRisk,
    reason: RepairReason,
    enqueued_at: Instant,
    attempts: u32,
    relocate: bool,
}

#[derive(Debug, Clone, Copy)]
struct RepairClaim {
    owner: NodeId,
    generation: u64,
    expires_at: Instant,
}

#[derive(Debug, Default)]
struct MaintenanceState {
    schedule: ScrubSchedule,
    scan_index: usize,
    histories: HashMap<AssetId, ScrubHistory>,
    tasks: Vec<HealingTask>,
    claims: HashMap<AssetId, RepairClaim>,
    owner_wait_since: HashMap<AssetId, Instant>,
    debt_entries: HashMap<AssetId, Vec<RepairDebtEntry>>,
    health_states: HashMap<AssetId, crate::AssetHealth>,
    missing_by_asset: HashMap<AssetId, u32>,
    placement_violations: HashMap<AssetId, u32>,
    debt: RepairDebt,
    last_orphan_sweep: Option<Instant>,
    last_report: Option<MaintenanceReport>,
    last_error: Option<String>,
    in_flight_scrubs: usize,
    in_flight_repairs: usize,
}

struct MaintenanceLease<'a>(&'a AtomicBool);

impl Drop for MaintenanceLease<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Cluster-wide redundancy plane over remote holders and a layout catalog.
///
/// Shares `Arc` handles freely (`&self` methods throughout); the repair
/// queue and incarnation table hide behind locks that are never held across
/// `await`. Tablet workers only await the returned futures; CPU coding runs
/// inline here or on the [`crate::lane::RedundancyLane`] by the caller.
#[derive(Debug)]
pub struct DistributedFabric {
    /// Delivery to remote holders.
    transport: Arc<dyn FragmentTransport>,
    /// Control-plane authority reads.
    catalog: Arc<dyn LayoutCatalog>,
    /// Control-plane authority writes.
    publisher: Arc<dyn CatalogPublish>,
    /// Bounded repair queue (never held across `await`).
    repair: Mutex<RepairEngine>,
    /// Shared operators' counters.
    metrics: Arc<crate::FabricMetrics>,
    /// Remote-work bounds.
    config: DistConfig,
    /// Believed incarnations by node (`0` = undiscovered bootstrap).
    incarnations: Mutex<HashMap<NodeId, u64>>,
    /// Bounded self-healing scheduler state. It is disposable and rebuilt
    /// from catalog plus physical observations after restart.
    maintenance: Mutex<MaintenanceState>,
    /// Maintenance limits and scheduling policy.
    maintenance_budgets: MaintenanceBudgets,
    /// Grace before a non-critical temporary holder failure is replaced.
    temporary_failure_grace: Duration,
    /// Grace before orphan candidates may be reclaimed.
    orphan_grace: Duration,
    /// Monotonic origin for deterministic maintenance timestamps.
    maintenance_origin: Instant,
    /// Bounded evidence captured by degraded reads for the next tick.
    read_evidence: Mutex<VecDeque<(AssetId, u64)>>,
    /// Last observed fetch latency per source, used for deterministic ranking.
    source_latency_us: Mutex<HashMap<NodeId, u64>>,
    /// Last health verdict per asset for lock-free admin gauges.
    health_states: Mutex<HashMap<AssetId, crate::AssetHealth>>,
    /// Ensures one maintenance window owns the controller at a time.
    maintenance_active: AtomicBool,
}

impl DistributedFabric {
    /// Builds the plane over a transport, catalog, publisher, and metrics.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::InvalidParams`] on zero config bounds.
    pub fn new(
        transport: Arc<dyn FragmentTransport>,
        catalog: Arc<dyn LayoutCatalog>,
        publisher: Arc<dyn CatalogPublish>,
        metrics: Arc<crate::FabricMetrics>,
        config: DistConfig,
    ) -> Result<Self, RedundancyError> {
        config.validate()?;
        Ok(Self {
            transport,
            catalog,
            publisher,
            repair: Mutex::new(RepairEngine::with_budgets(
                crate::RepairBudgets::conservative(),
            )),
            metrics,
            config,
            incarnations: Mutex::new(HashMap::new()),
            maintenance: Mutex::new(MaintenanceState::default()),
            maintenance_budgets: MaintenanceBudgets::conservative(),
            temporary_failure_grace: Duration::from_secs(30),
            orphan_grace: Duration::from_secs(300),
            maintenance_origin: Instant::now(),
            read_evidence: Mutex::new(VecDeque::new()),
            source_latency_us: Mutex::new(HashMap::new()),
            health_states: Mutex::new(HashMap::new()),
            maintenance_active: AtomicBool::new(false),
        })
    }

    /// Returns shared metrics.
    #[must_use]
    pub fn metrics(&self) -> &Arc<crate::FabricMetrics> {
        &self.metrics
    }

    /// Returns the configuration.
    #[must_use]
    pub const fn config(&self) -> DistConfig {
        self.config
    }

    /// Builds a plane with explicit maintenance budgets.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::InvalidParams`] when transport bounds are
    /// invalid or the maintenance limits cannot make progress.
    pub fn with_maintenance_budgets(
        transport: Arc<dyn FragmentTransport>,
        catalog: Arc<dyn LayoutCatalog>,
        publisher: Arc<dyn CatalogPublish>,
        metrics: Arc<crate::FabricMetrics>,
        config: DistConfig,
        budgets: MaintenanceBudgets,
    ) -> Result<Self, RedundancyError> {
        if !budgets.validate() {
            return Err(RedundancyError::invalid_params(
                "maintenance budgets cannot make progress",
            ));
        }
        let mut fabric = Self::new(transport, catalog, publisher, metrics, config)?;
        fabric.maintenance_budgets = budgets;
        Ok(fabric)
    }

    /// Returns the active maintenance budgets.
    #[must_use]
    pub const fn maintenance_budgets(&self) -> MaintenanceBudgets {
        self.maintenance_budgets
    }

    /// Replaces maintenance budgets and returns the previous value.
    pub fn set_maintenance_budgets(&mut self, budgets: MaintenanceBudgets) -> MaintenanceBudgets {
        let previous = self.maintenance_budgets;
        self.maintenance_budgets = budgets;
        previous
    }

    /// Sets the grace period before a non-critical unavailable holder is
    /// replaced.
    pub fn set_temporary_failure_grace(&mut self, grace: Duration) {
        self.temporary_failure_grace = grace;
    }

    /// Sets the age grace used by background orphan reclamation.
    pub fn set_orphan_grace(&mut self, grace: Duration) {
        self.orphan_grace = grace;
    }

    /// Replaces the scrub schedule while retaining its current cursor when
    /// the new schedule has compatible bounds.
    pub fn set_scrub_schedule(&mut self, schedule: ScrubSchedule) {
        if let Ok(mut state) = self.maintenance.lock() {
            state.schedule = schedule;
        }
    }

    /// Returns a point-in-time self-healing controller snapshot.
    #[must_use]
    pub fn maintenance_snapshot(&self) -> ControllerSnapshot {
        let observed = self.maintenance_ticks();
        let Ok(state) = self.maintenance.lock() else {
            return ControllerSnapshot::new(
                observed,
                self.maintenance_budgets,
                ScrubSchedule::conservative(),
                RepairDebt::default(),
            );
        };
        let mut snapshot = ControllerSnapshot::new(
            observed,
            self.maintenance_budgets,
            state.schedule,
            state.debt.clone(),
        );
        snapshot.queued_work = state.tasks.len();
        snapshot.in_flight_reconstructions = state.in_flight_repairs;
        snapshot.in_flight_transfers = state.in_flight_repairs;
        snapshot.in_flight_scrubs = state.in_flight_scrubs;
        snapshot.last_report.clone_from(&state.last_report);
        snapshot.last_error.clone_from(&state.last_error);
        snapshot
    }

    /// Records a believed incarnation (control-plane epochs, reconnects).
    pub fn set_peer_incarnation(&self, node: NodeId, incarnation: u64) {
        if let Ok(mut guard) = self.incarnations.lock() {
            guard.insert(node, incarnation);
        }
    }

    /// Returns the believed incarnation (`0` = undiscovered bootstrap).
    #[must_use]
    pub fn peer_incarnation(&self, node: NodeId) -> u64 {
        let Ok(guard) = self.incarnations.lock() else {
            return 0;
        };
        guard.get(&node).copied().unwrap_or(0)
    }

    fn effective_scope(&self, view: &[NodeDescriptor]) -> FailureScope {
        if self.config.scope != FailureScope::Node {
            return self.config.scope;
        }
        if view.iter().any(|node| !node.domain.rack.is_empty()) {
            FailureScope::Rack
        } else {
            FailureScope::Node
        }
    }

    /// Protects an asset's bytes under an intent: fast path when the current
    /// layout is healthy and already satisfies the intent, else a full
    /// `build → verify → publish → retire` generation.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] on invalid intents, unsatisfiable
    /// placement, verification failures, stale control generations, or
    /// exhausted holder fallbacks.
    pub async fn protect(
        &self,
        asset: InformationAsset,
        bytes: &[u8],
        intent: &RedundancyIntent,
        view: &[NodeDescriptor],
        control_gen: u64,
    ) -> Result<PublishedLayout, RedundancyError> {
        asset.verify_bytes(bytes).map_err(|error| match error {
            RedundancyError::VerificationFailed { detail } => {
                RedundancyError::CorruptFragment { detail }
            }
            other => other,
        })?;
        intent.validate()?;
        let current = self.catalog.get(&asset.id);
        if let Some(current) = &current {
            let mut requested_domains = intent.allowed_domains.clone();
            requested_domains.sort_unstable();
            requested_domains.dedup();
            let contract_matches = current.layout.logical_len == asset.logical_len
                && current.layout.params.tolerance() >= u32::from(intent.tolerance)
                && current.layout.min_available >= intent.min_available
                && current.layout.locality == intent.locality
                && current.layout.allowed_domains == requested_domains;
            let healthy = contract_matches
                && self
                    .assess(&asset.id, view)
                    .await
                    .is_some_and(|assessment| {
                        assessment.assessment.health == crate::AssetHealth::Healthy
                    });
            if healthy {
                return Ok(current.clone());
            }
        }
        let params = plan_baseline(intent, asset.logical_len)?;
        let domains = {
            let mut keys: std::collections::HashSet<String> = std::collections::HashSet::new();
            for node in view {
                if node.health.accepts_new() {
                    keys.insert(node.domain.independence_key(self.effective_scope(view)));
                }
            }
            keys.len()
        };
        let params = crate::scheme::fit_to_domains(params, asset.logical_len, domains);
        if params.required_pieces() < intent.min_available {
            return Err(RedundancyError::invalid_params(format!(
                "placement retains {} required pieces, intent requires {}",
                params.required_pieces(),
                intent.min_available
            )));
        }
        self.build_publish(BuildRequest {
            asset,
            bytes,
            params,
            min_available: intent.min_available,
            locality: intent.locality,
            allowed: &intent.allowed_domains,
            view,
            current: current.as_ref(),
            control_gen,
        })
        .await
    }

    /// Reads an asset through its current layout: exact `required` pieces
    /// first, alternates on failure, every piece content-verified, the image
    /// content-verified before exposure. Never republishes.
    ///
    /// Returns the bytes plus whether any probe or fetch failed along the
    /// way (`degraded`).
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when no layout exists, too few pieces
    /// survive, or verification fails.
    pub async fn read(&self, asset: InformationAsset) -> Result<(Vec<u8>, bool), RedundancyError> {
        let current =
            self.catalog
                .get(&asset.id)
                .ok_or_else(|| RedundancyError::MissingFragment {
                    asset: asset.id.to_string(),
                    generation: 0,
                    index: 0,
                })?;
        let layout = &current.layout;
        if asset.logical_len != layout.logical_len {
            return Err(RedundancyError::VerificationFailed {
                detail: format!(
                    "asset {} len {} disagrees with layout {}",
                    asset.id, asset.logical_len, layout.logical_len
                ),
            });
        }
        let required = layout.params.required_pieces().max(layout.min_available) as usize;
        let (mut pieces, degraded) = self.fetch_minimum(layout, required).await?;
        pieces.sort_by_key(|(index, _)| *index);
        let decoded = decode_pieces(&pieces, layout, asset.logical_len, self.config)?;
        asset.verify_bytes(&decoded)?;
        self.metrics.decodes.fetch_add(1, Ordering::Relaxed);
        if degraded {
            self.metrics.degraded_reads.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut evidence) = self.read_evidence.lock()
                && evidence.len() < self.maintenance_budgets.max_queued_work
            {
                evidence.push_back((asset.id, current.layout.generation));
            }
        }
        Ok((decoded, degraded))
    }

    /// Transitions an asset to an explicit target scheme through a full
    /// verified generation (the old layout serves until the replacement
    /// publishes, then retires).
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] on planning, build, verify, or publish
    /// failures. The current layout is untouched on failure.
    pub async fn transition(
        &self,
        asset: InformationAsset,
        bytes: &[u8],
        target: SchemeParams,
        view: &[NodeDescriptor],
        control_gen: u64,
    ) -> Result<PublishedLayout, RedundancyError> {
        target.validate()?;
        let current = self.catalog.get(&asset.id);
        if let Some(current) = &current
            && current.layout.params == target
        {
            return Ok(current.clone());
        }
        let allowed = current
            .as_ref()
            .map_or(&[][..], |record| record.layout.allowed_domains.as_slice());
        let locality = current
            .as_ref()
            .map_or(crate::LocalityRequirement::None, |record| {
                record.layout.locality
            });
        let published = self
            .build_publish(BuildRequest {
                asset,
                bytes,
                params: target,
                min_available: current
                    .as_ref()
                    .map_or(1, |record| record.layout.min_available),
                locality,
                allowed,
                view,
                current: current.as_ref(),
                control_gen,
            })
            .await?;
        self.metrics.transitions.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .transitions_with_overlap
            .fetch_add(1, Ordering::Relaxed);
        Ok(published)
    }

    /// Planned transition deriving the target from an intent.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] on planning, build, or publish failures.
    pub async fn transition_for_intent(
        &self,
        asset: InformationAsset,
        bytes: &[u8],
        intent: &RedundancyIntent,
        view: &[NodeDescriptor],
        control_gen: u64,
    ) -> Result<PublishedLayout, RedundancyError> {
        let params = plan_baseline(intent, asset.logical_len)?;
        self.transition(asset, bytes, params, view, control_gen)
            .await
    }

    fn maintenance_ticks(&self) -> Ticks {
        Ticks::from_micros(
            u64::try_from(self.maintenance_origin.elapsed().as_micros()).unwrap_or(u64::MAX),
        )
    }

    fn owner_for_asset(asset: AssetId, generation: u64, view: &[NodeDescriptor]) -> Option<NodeId> {
        let mut active: Vec<NodeId> = view
            .iter()
            .filter(|node| node.health.accepts_new())
            .map(|node| node.id)
            .collect();
        if active.is_empty() {
            return None;
        }
        active.sort_by_key(|node| node.as_u64());
        let mut state = 0xCBF2_9CE4_8422_2325_u64;
        for byte in asset.hash {
            state ^= u64::from(byte);
            state = state.wrapping_mul(0x1000_0000_01B3);
            state ^= state >> 29;
        }
        state ^= generation;
        state = state.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let index = usize::try_from(state % active.len() as u64).unwrap_or(0);
        active.get(index).copied()
    }

    #[allow(clippy::too_many_lines)]
    fn risk_for_assessment(
        &self,
        assessed: &DistributedAssessment,
        current: &PublishedLayout,
        view: &[NodeDescriptor],
        history: ScrubHistory,
        now: Instant,
    ) -> RepairRisk {
        let by_id: HashMap<NodeId, NodeDescriptor> =
            view.iter().map(|node| (node.id, node.clone())).collect();
        let scope = self.effective_scope(view);
        let mut domains: HashMap<String, u32> = HashMap::new();
        for (index, node, healthy) in &assessed.actual {
            if !*healthy {
                continue;
            }
            let key = by_id.get(node).map_or_else(
                || format!("node:{}", node.as_u64()),
                |descriptor| descriptor.domain.independence_key(scope),
            );
            *domains.entry(key).or_insert(0) += 1;
            let _ = index;
        }
        let concentration = domains.values().copied().max().unwrap_or(0);
        let unavailable_nodes: Vec<NodeId> = view
            .iter()
            .filter(|node| {
                matches!(
                    node.health,
                    crate::NodeHealth::Unavailable | crate::NodeHealth::Removed
                )
            })
            .map(|node| node.id)
            .collect();
        let draining_nodes: Vec<NodeId> = view
            .iter()
            .filter(|node| node.health == crate::NodeHealth::Draining)
            .map(|node| node.id)
            .collect();
        let corrupt_fragments = assessed
            .assessment
            .verdicts
            .iter()
            .filter(|(_, verdict)| {
                matches!(
                    verdict,
                    crate::FragmentVerdict::ChecksumMismatch
                        | crate::FragmentVerdict::ReconstructableCorruption
                        | crate::FragmentVerdict::UnrecoverableCorruption
                )
            })
            .count();
        #[allow(clippy::cast_possible_truncation)]
        let corrupt_fragments = corrupt_fragments as u32;
        let unavailable_fragments = assessed
            .assessment
            .verdicts
            .iter()
            .filter(|(_, verdict)| {
                matches!(
                    verdict,
                    crate::FragmentVerdict::Missing | crate::FragmentVerdict::Unreadable
                )
            })
            .count();
        #[allow(clippy::cast_possible_truncation)]
        let unavailable_fragments = unavailable_fragments as u32;
        let placement_violations = assessed
            .actual
            .iter()
            .filter(|(index, node, _)| {
                assessed
                    .desired
                    .node_for(*index)
                    .is_none_or(|desired| desired != *node)
            })
            .count();
        #[allow(clippy::cast_possible_truncation)]
        let placement_violations = placement_violations as u32;
        let degradation_age = history
            .degraded_since
            .map_or(Duration::ZERO, |since| now.saturating_duration_since(since));
        let verification_age = history
            .last_success
            .map_or(Duration::ZERO, |since| now.saturating_duration_since(since));
        let reconstruction_bytes = match current.layout.params {
            SchemeParams::Replication(_) => current.layout.logical_len,
            SchemeParams::ReedSolomon(params) => {
                u64::from(params.data).saturating_mul(u64::from(params.fragment_len))
            }
        };
        let required_fragments = current
            .layout
            .params
            .required_pieces()
            .max(current.layout.min_available);
        RepairRisk {
            remaining_independent_tolerance: assessed.assessment.survivable,
            usable_fragments: assessed.assessment.usable,
            required_fragments: assessed.assessment.required,
            total_fragments: assessed.assessment.total,
            scheme_requirements: SchemeRequirements::new(
                current.layout.params.scheme(),
                current.layout.params.total_fragments(),
                required_fragments,
                current.layout.params.tolerance(),
            ),
            failure_domain_concentration: concentration,
            independent_failure_domains: domains.len().try_into().unwrap_or(u32::MAX),
            unavailable_nodes,
            draining_nodes,
            corrupt_fragments,
            unavailable_fragments,
            asset_bytes: current.layout.logical_len,
            reconstruction_bytes,
            degradation_age,
            verification_age,
            placement_violations,
        }
    }

    /// Assesses installed health: `Has`-probes every published fragment,
    /// then computes survivability over the actually-healthy holders (not
    /// the desired placement — collided or drifted installs never inflate
    /// protection).
    pub async fn assess(
        &self,
        asset: &AssetId,
        view: &[NodeDescriptor],
    ) -> Option<DistributedAssessment> {
        self.assess_with_mode(asset, view, true).await
    }

    async fn assess_metadata(
        &self,
        asset: &AssetId,
        view: &[NodeDescriptor],
    ) -> Option<DistributedAssessment> {
        self.assess_with_mode(asset, view, false).await
    }

    async fn assess_with_mode(
        &self,
        asset: &AssetId,
        view: &[NodeDescriptor],
        full_payload: bool,
    ) -> Option<DistributedAssessment> {
        let current = self.catalog.get(asset)?;
        let layout = &current.layout;
        let scope = self.effective_scope(view);
        let total = layout.params.total_fragments();
        let desired = plan_placement_with_locality(
            *asset,
            layout.generation,
            total,
            scope,
            view,
            &layout.allowed_domains,
            layout.locality,
        )
        .unwrap_or_else(|_| desired_from_layout(layout, scope));
        let mut actual = Vec::with_capacity(layout.fragments.len());
        let mut observed: Vec<(u32, Option<Option<bool>>)> =
            Vec::with_capacity(layout.fragments.len());
        for record in &layout.fragments {
            let key = FragmentKey::from_asset(*asset, layout.generation, record.id.index);
            let probe = if full_payload {
                self.has_one_checked(record.node, &key).await
            } else {
                self.metadata_one_checked(record.node, &key).await
            };
            let probe_failed = probe.is_err();
            let (present, healthy) = probe.unwrap_or((false, false));
            actual.push((record.id.index, record.node, healthy));
            observed.push((record.id.index, (!probe_failed).then_some(Some(present))));
        }
        let healthy: Vec<u32> = actual
            .iter()
            .filter(|(_, _, healthy)| *healthy)
            .map(|(index, _, _)| *index)
            .collect();
        let by_id: HashMap<NodeId, NodeDescriptor> =
            view.iter().map(|node| (node.id, node.clone())).collect();
        let installed = desired_from_layout(layout, scope);
        let survivable = independent_survivable(layout.params, scope, &installed, &healthy, &by_id);
        let verdicts = actual
            .iter()
            .map(|(index, _, healthy)| {
                // Presence without health is corruption, not service: it
                // owes a corruption-driven repair, not a routine refill.
                let observed = observed
                    .iter()
                    .find(|(observed_index, _)| observed_index == index)
                    .and_then(|(_, present)| *present);
                let verdict = if *healthy {
                    crate::FragmentVerdict::Healthy
                } else {
                    match observed {
                        Some(Some(true)) => crate::FragmentVerdict::ChecksumMismatch,
                        Some(Some(false)) => crate::FragmentVerdict::Missing,
                        _ => crate::FragmentVerdict::Unreadable,
                    }
                };
                (*index, verdict)
            })
            .collect();
        let assessment = assess_with_min_available(
            *asset,
            layout.generation,
            layout.params,
            verdicts,
            survivable,
            layout.min_available,
        );
        self.record_health(*asset, assessment.health);
        if actual
            .iter()
            .any(|(index, node, _)| desired.node_for(*index).is_none_or(|want| want != *node))
        {
            self.metrics
                .desired_vs_actual_mismatches
                .fetch_add(1, Ordering::Relaxed);
        }
        Some(DistributedAssessment {
            assessment,
            desired,
            actual,
        })
    }

    async fn probe_asset_status(
        &self,
        node: NodeId,
        asset: AssetId,
        _generation: u64,
    ) -> Result<(u64, [u8; 32]), RedundancyError> {
        let rpc = FragmentRpc::ProbeAsset {
            asset_kind: asset.kind.as_u8(),
            asset_domain: asset.domain,
            asset_hash: asset.hash,
            target_incarnation: 0,
        };
        match self.call_with_refresh(node, rpc).await? {
            FragmentReply::AssetStatus {
                generation: actual,
                layout_hash,
            } => Ok((actual, layout_hash)),
            FragmentReply::Refused { reason } => {
                Err(TransportFailure::Refused { reason }.into_error("probe-asset"))
            }
            other => Err(RedundancyError::Unreachable {
                detail: format!("asset probe answered {other:?}"),
            }),
        }
    }

    #[allow(clippy::too_many_lines)]
    /// Scrubs one published asset without changing its generation.
    ///
    /// Metadata mode verifies holder layout generations, identities, and
    /// indexed fragment health. Full mode additionally reads every payload,
    /// decodes a verified image, and records the exact evidence needed to
    /// prioritize repair.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when the catalog has no layout or a
    /// transport/control failure prevents observation.
    pub async fn scrub_asset(
        &self,
        asset: &AssetId,
        mode: ScrubMode,
    ) -> Result<crate::ScrubReport, RedundancyError> {
        let current = self
            .catalog
            .get(asset)
            .ok_or_else(|| RedundancyError::MissingFragment {
                asset: asset.to_string(),
                generation: 0,
                index: 0,
            })?;
        let layout = &current.layout;
        let encoded_layout = layout.encode();
        let expected_layout_hash = blake3_256(&encoded_layout);
        let mut report = crate::scrub_report(*asset, layout.generation, Vec::new());
        report.mode = mode;
        report.metadata_ok = true;
        report.asset_verified = mode.reads_payloads();
        let mut pieces = Vec::new();
        for record in &layout.fragments {
            let key = FragmentKey::from_asset(*asset, layout.generation, record.id.index);
            match self
                .probe_asset_status(record.node, *asset, layout.generation)
                .await
            {
                Ok((0, _)) => {
                    report.metadata_ok = false;
                    report.metadata_corrupt += 1;
                }
                Ok((observed_generation, observed_layout_hash)) => {
                    if observed_generation != layout.generation {
                        report.wrong_generation += 1;
                        report.metadata_ok = false;
                    }
                    if observed_layout_hash != expected_layout_hash {
                        report.wrong_identity += 1;
                        report.metadata_ok = false;
                    }
                }
                Err(_) => {
                    report.metadata_ok = false;
                }
            }
            let probe = if mode.reads_payloads() {
                self.has_one_checked(record.node, &key).await
            } else {
                self.metadata_one_checked(record.node, &key).await
            };
            let probe_failed = probe.is_err();
            let (present, healthy) = probe.unwrap_or((false, false));
            let mut verdict = if healthy {
                crate::FragmentVerdict::Healthy
            } else if probe_failed {
                crate::FragmentVerdict::Unreadable
            } else if present {
                crate::FragmentVerdict::ChecksumMismatch
            } else {
                crate::FragmentVerdict::Missing
            };
            if mode.reads_payloads() && healthy {
                match self.fetch_one(record).await {
                    Ok(piece) => {
                        report.bytes_checked =
                            report.bytes_checked.saturating_add(piece.1.len() as u64);
                        pieces.push(piece);
                    }
                    Err(error) => {
                        verdict = match error {
                            RedundancyError::CorruptFragment { .. } => {
                                report.truncated += 1;
                                crate::FragmentVerdict::ChecksumMismatch
                            }
                            RedundancyError::MissingFragment { .. } => {
                                crate::FragmentVerdict::Missing
                            }
                            _ => crate::FragmentVerdict::Unreadable,
                        };
                    }
                }
            }
            report.verdicts.push((record.id.index, verdict));
        }
        report.corruptions = report
            .verdicts
            .iter()
            .filter(|(_, verdict)| {
                matches!(
                    verdict,
                    crate::FragmentVerdict::ChecksumMismatch
                        | crate::FragmentVerdict::ReconstructableCorruption
                        | crate::FragmentVerdict::UnrecoverableCorruption
                )
            })
            .count();
        report.missing = report
            .verdicts
            .iter()
            .filter(|(_, verdict)| {
                matches!(
                    verdict,
                    crate::FragmentVerdict::Missing | crate::FragmentVerdict::Unreadable
                )
            })
            .count();
        report.stale = report.wrong_generation;
        if mode.reads_payloads() {
            pieces.sort_by_key(|(index, _)| *index);
            let required = layout.params.required_pieces().max(layout.min_available) as usize;
            if pieces.len() >= required {
                if let Ok(bytes) = decode_pieces(&pieces, layout, layout.logical_len, self.config) {
                    let info = InformationAsset::new(*asset, layout.logical_len);
                    report.asset_verified = info.verify_bytes(&bytes).is_ok();
                    if !report.asset_verified {
                        self.metrics
                            .reconstruction_failures
                            .fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    report.asset_verified = false;
                    self.metrics
                        .reconstruction_failures
                        .fetch_add(1, Ordering::Relaxed);
                }
            } else {
                report.asset_verified = false;
            }
        }
        self.metrics.scrub_runs.fetch_add(1, Ordering::Relaxed);
        self.metrics.scrub_assets.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .scrub_bytes
            .fetch_add(report.bytes_checked, Ordering::Relaxed);
        if mode == ScrubMode::Full {
            self.metrics.full_scrubs.fetch_add(1, Ordering::Relaxed);
        }
        self.metrics
            .missing_fragments
            .store(report.missing as u64, Ordering::Relaxed);
        if report.asset_verified {
            self.metrics
                .last_verification_age_ms
                .store(0, Ordering::Relaxed);
        }
        Ok(report)
    }

    fn debt_entries_for(
        &self,
        current: &PublishedLayout,
        assessed: &DistributedAssessment,
        risk: &RepairRisk,
        view: &[NodeDescriptor],
    ) -> Vec<RepairDebtEntry> {
        if risk.meets_scheme_requirements()
            && assessed.assessment.health == crate::AssetHealth::Healthy
            && risk.placement_violations == 0
        {
            return Vec::new();
        }
        let by_id: HashMap<NodeId, NodeDescriptor> =
            view.iter().map(|node| (node.id, node.clone())).collect();
        let mut grouped: BTreeMap<NodeId, (String, u64, u64)> = BTreeMap::new();
        for (index, record) in current.layout.fragments.iter().enumerate() {
            let index = u32::try_from(index).unwrap_or(u32::MAX);
            let verdict = assessed
                .assessment
                .verdicts
                .iter()
                .find(|(candidate, _)| *candidate == index)
                .map_or(crate::FragmentVerdict::Missing, |(_, verdict)| *verdict);
            let placement_bad = assessed
                .desired
                .node_for(record.id.index)
                .is_none_or(|desired| desired != record.node);
            if verdict.is_usable() && !placement_bad {
                continue;
            }
            let domain = by_id.get(&record.node).map_or_else(
                || format!("node:{}", record.node.as_u64()),
                |node| node.domain.independence_key(self.effective_scope(view)),
            );
            let entry = grouped.entry(record.node).or_insert_with(|| (domain, 0, 0));
            entry.1 = entry.1.saturating_add(record.stored_len);
            entry.2 = entry.2.saturating_add(1);
        }
        let critical = u64::from(risk.is_critical());
        let mut out = Vec::with_capacity(grouped.len());
        for (index, (node, (domain, bytes, fragments))) in grouped.into_iter().enumerate() {
            out.push(RepairDebtEntry::new(
                node,
                domain,
                current.layout.params.scheme(),
                DebtCounters::new(
                    u64::from(index == 0),
                    bytes,
                    fragments,
                    critical * u64::from(index == 0),
                ),
            ));
        }
        out
    }

    fn replace_debt(state: &mut MaintenanceState, asset: AssetId, entries: Vec<RepairDebtEntry>) {
        if let Some(previous) = state.debt_entries.remove(&asset) {
            for entry in previous {
                state.debt.remove(&entry);
            }
        }
        for entry in &entries {
            state.debt.add(entry);
        }
        if !entries.is_empty() {
            state.debt_entries.insert(asset, entries);
        }
    }

    fn queue_healing_task(&self, state: &mut MaintenanceState, task: HealingTask) -> bool {
        if let Some(existing) = state.tasks.iter_mut().find(|candidate| {
            candidate.asset == task.asset && candidate.generation == task.generation
        }) {
            if task.risk.priority() > existing.risk.priority() {
                existing.risk = task.risk;
            }
            existing.relocate |= task.relocate;
            return false;
        }
        if state.tasks.len() >= self.maintenance_budgets.max_queued_work {
            self.metrics
                .maintenance_throttled
                .fetch_add(1, Ordering::Relaxed);
            return false;
        }
        state.tasks.push(task);
        state.tasks.sort_by(|left, right| {
            right
                .risk
                .priority()
                .cmp(&left.risk.priority())
                .then_with(|| left.asset.cmp(&right.asset))
        });
        self.metrics
            .pending_repairs
            .store(state.tasks.len() as u64, Ordering::Relaxed);
        self.metrics.repair_bytes_pending.store(
            state
                .tasks
                .iter()
                .map(|task| task.risk.reconstruction_bytes)
                .sum(),
            Ordering::Relaxed,
        );
        self.metrics.estimated_reconstruction_bytes.store(
            state
                .tasks
                .iter()
                .map(|task| task.risk.reconstruction_bytes)
                .sum(),
            Ordering::Relaxed,
        );
        true
    }

    fn owns_maintenance(
        &self,
        state: &mut MaintenanceState,
        asset: AssetId,
        generation: u64,
        view: &[NodeDescriptor],
        now: Instant,
    ) -> bool {
        let local = self.config.local;
        let local_present = view.iter().any(|node| node.id == local);
        if !local_present {
            return true;
        }
        let owner = Self::owner_for_asset(asset, generation, view);
        if owner == Some(local) {
            state.owner_wait_since.remove(&asset);
            return true;
        }
        let owner_available = owner.is_some_and(|owner| {
            view.iter()
                .any(|node| node.id == owner && node.health.serves_reads())
        });
        if owner_available {
            state.owner_wait_since.remove(&asset);
            return false;
        }
        let waiting_since = state.owner_wait_since.entry(asset).or_insert(now);
        now.saturating_duration_since(*waiting_since) >= self.temporary_failure_grace
    }

    fn record_health(&self, asset: AssetId, health: crate::AssetHealth) {
        let Ok(mut states) = self.health_states.lock() else {
            return;
        };
        states.insert(asset, health);
        let count = |wanted| states.values().filter(|value| **value == wanted).count();
        self.metrics
            .healthy
            .store(count(crate::AssetHealth::Healthy) as u64, Ordering::Relaxed);
        self.metrics.degraded.store(
            count(crate::AssetHealth::Degraded) as u64,
            Ordering::Relaxed,
        );
        self.metrics.critical.store(
            count(crate::AssetHealth::Critical) as u64,
            Ordering::Relaxed,
        );
        self.metrics.unrecoverable.store(
            count(crate::AssetHealth::Unrecoverable) as u64,
            Ordering::Relaxed,
        );
    }

    fn refresh_health_metrics(&self, state: &MaintenanceState, catalog_len: usize) {
        let healthy = state
            .health_states
            .values()
            .filter(|health| **health == crate::AssetHealth::Healthy)
            .count();
        let degraded = state
            .health_states
            .values()
            .filter(|health| **health == crate::AssetHealth::Degraded)
            .count();
        let critical = state
            .health_states
            .values()
            .filter(|health| **health == crate::AssetHealth::Critical)
            .count();
        let unrecoverable = state
            .health_states
            .values()
            .filter(|health| **health == crate::AssetHealth::Unrecoverable)
            .count();
        self.metrics.healthy.store(
            (healthy + catalog_len.saturating_sub(state.health_states.len())) as u64,
            Ordering::Relaxed,
        );
        self.metrics
            .degraded
            .store(degraded as u64, Ordering::Relaxed);
        self.metrics
            .critical
            .store(critical as u64, Ordering::Relaxed);
        self.metrics
            .unrecoverable
            .store(unrecoverable as u64, Ordering::Relaxed);
        let now = Instant::now();
        let last_scrub_age = state
            .histories
            .values()
            .filter_map(|history| history.last_metadata.or(history.last_full))
            .map(|last| now.saturating_duration_since(last))
            .min()
            .unwrap_or(Duration::ZERO);
        let last_verification_age = state
            .histories
            .values()
            .filter_map(|history| history.last_success)
            .map(|last| now.saturating_duration_since(last))
            .min()
            .unwrap_or(Duration::ZERO);
        let degraded_time = state
            .histories
            .values()
            .filter_map(|history| history.degraded_since)
            .map(|since| now.saturating_duration_since(since))
            .max()
            .unwrap_or(Duration::ZERO);
        self.metrics.last_scrub_age_ms.store(
            u64::try_from(last_scrub_age.as_millis()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.metrics.last_verification_age_ms.store(
            u64::try_from(last_verification_age.as_millis()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.metrics.degraded_time_ms.store(
            u64::try_from(degraded_time.as_millis()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.metrics.placement_violations.store(
            state
                .placement_violations
                .values()
                .map(|violations| u64::from(*violations))
                .sum(),
            Ordering::Relaxed,
        );
        self.metrics.missing_fragments.store(
            state
                .missing_by_asset
                .values()
                .map(|count| u64::from(*count))
                .sum(),
            Ordering::Relaxed,
        );
        self.metrics.degraded_bytes.store(
            state
                .debt_entries
                .keys()
                .filter_map(|asset| self.catalog.get(asset))
                .map(|record| record.layout.logical_len)
                .sum(),
            Ordering::Relaxed,
        );
        self.metrics
            .missing_physical_bytes
            .store(state.debt.total_bytes, Ordering::Relaxed);
        self.metrics
            .failure_domain_exposure
            .store(state.debt.total_fragments, Ordering::Relaxed);
    }

    #[allow(clippy::too_many_lines)]
    /// Runs one bounded maintenance window against an already-refreshed
    /// catalog page. Callers refresh the catalog on their own schedule; this
    /// method never scans the full cluster catalog on every tick.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when the controller state is poisoned or a
    /// required catalog lookup fails. Individual asset failures are recorded
    /// and retried rather than aborting the whole controller.
    pub async fn maintenance_tick_at(
        &self,
        catalog: &[PublishedLayout],
        view: &[NodeDescriptor],
        foreground_pressure: bool,
    ) -> Result<MaintenanceReport, RedundancyError> {
        let started = self.maintenance_ticks();
        let mut report = MaintenanceReport::new(started, started, ScrubMode::Metadata);
        if self
            .maintenance_active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            report.finished_at = self.maintenance_ticks();
            return Ok(report);
        }
        let _lease = MaintenanceLease(&self.maintenance_active);
        if catalog.is_empty() {
            report.finished_at = self.maintenance_ticks();
            return Ok(report);
        }
        let evidence = {
            let mut queue = self
                .read_evidence
                .lock()
                .map_err(|_| RedundancyError::Unreadable {
                    detail: "read evidence lock poisoned".to_owned(),
                })?;
            let count = queue
                .len()
                .min(self.maintenance_budgets.max_scrub_assets.min(64));
            queue.drain(..count).collect::<Vec<_>>()
        };
        for (asset, generation) in evidence {
            let Some(current) = self.catalog.get(&asset) else {
                continue;
            };
            if current.layout.generation != generation {
                self.metrics.stale_rejected.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            let Some(assessed) = self.assess(&asset, view).await else {
                continue;
            };
            let now = Instant::now();
            let history = self
                .maintenance
                .lock()
                .map_err(|_| RedundancyError::Unreadable {
                    detail: "maintenance state lock poisoned".to_owned(),
                })?
                .histories
                .get(&asset)
                .copied()
                .unwrap_or_else(ScrubHistory::new);
            let risk = self.risk_for_assessment(&assessed, &current, view, history, now);
            if risk.meets_scheme_requirements() && risk.placement_violations == 0 {
                continue;
            }
            let mut state = self
                .maintenance
                .lock()
                .map_err(|_| RedundancyError::Unreadable {
                    detail: "maintenance state lock poisoned".to_owned(),
                })?;
            self.queue_healing_task(
                &mut state,
                HealingTask {
                    asset,
                    generation,
                    risk,
                    reason: RepairReason::FragmentMissing,
                    enqueued_at: now,
                    attempts: 0,
                    relocate: false,
                },
            );
        }
        let mut targets: Vec<ScrubTarget> = catalog
            .iter()
            .map(|record| ScrubTarget::new(record.layout.asset, record.layout.logical_len))
            .collect();
        targets.sort_by_key(|target| target.asset);
        {
            let mut state = self
                .maintenance
                .lock()
                .map_err(|_| RedundancyError::Unreadable {
                    detail: "maintenance state lock poisoned".to_owned(),
                })?;
            let live: HashSet<AssetId> = targets.iter().map(|target| target.asset).collect();
            let removed: Vec<AssetId> = state
                .health_states
                .keys()
                .copied()
                .filter(|asset| !live.contains(asset))
                .collect();
            for asset in removed {
                state.health_states.remove(&asset);
                state.missing_by_asset.remove(&asset);
                state.placement_violations.remove(&asset);
                Self::replace_debt(&mut state, asset, Vec::new());
                state.histories.remove(&asset);
            }
            state.tasks.retain(|task| live.contains(&task.asset));
        }
        let selected = {
            let mut state = self
                .maintenance
                .lock()
                .map_err(|_| RedundancyError::Unreadable {
                    detail: "maintenance state lock poisoned".to_owned(),
                })?;
            state.schedule.cursor.normalize(&targets);
            let start = state.schedule.cursor.position(&targets).unwrap_or(0);
            let inspection_limit = self
                .maintenance_budgets
                .max_scrub_assets
                .max(1)
                .min(targets.len());
            let now_tick = self.maintenance_ticks();
            let schedule = state.schedule;
            let mut selected = Vec::with_capacity(inspection_limit);
            let mut selected_bytes = 0u64;
            let mut processed = 0usize;
            for offset in 0..inspection_limit {
                let target = targets[(start + offset) % targets.len()];
                processed += 1;
                let history = *state
                    .histories
                    .entry(target.asset)
                    .or_insert_with(ScrubHistory::new);
                let has_debt = state.debt_entries.contains_key(&target.asset)
                    || state.tasks.iter().any(|task| task.asset == target.asset);
                let mode = schedule
                    .due_mode(
                        now_tick,
                        history.last_metadata_tick,
                        history.last_full_tick,
                        &schedule.cursor,
                    )
                    .or_else(|| has_debt.then_some(ScrubMode::Metadata));
                let Some(mode) = mode else {
                    continue;
                };
                if selected_bytes.saturating_add(target.bytes)
                    > self.maintenance_budgets.max_scrub_bytes
                    && !selected.is_empty()
                {
                    processed = processed.saturating_sub(1);
                    self.metrics
                        .maintenance_throttled
                        .fetch_add(1, Ordering::Relaxed);
                    break;
                }
                let generation = catalog
                    .iter()
                    .find(|record| record.layout.asset == target.asset)
                    .map_or(0, |record| record.layout.generation);
                if !self.owns_maintenance(
                    &mut state,
                    target.asset,
                    generation,
                    view,
                    Instant::now(),
                ) {
                    continue;
                }
                selected.push((target.asset, mode));
                selected_bytes = selected_bytes.saturating_add(target.bytes);
                if foreground_pressure {
                    self.metrics
                        .foreground_pressure_suppressed
                        .fetch_add(1, Ordering::Relaxed);
                    break;
                }
            }
            let next = if targets.is_empty() {
                0
            } else {
                (start + processed) % targets.len()
            };
            if targets.is_empty() || processed >= targets.len() {
                state.schedule.cursor.next_asset = None;
                state.schedule.cursor.byte_offset = 0;
                state.schedule.cursor.cycle_complete = true;
            } else {
                state.schedule.cursor.next_asset = Some(targets[next].asset);
                state.schedule.cursor.byte_offset = 0;
                state.schedule.cursor.cycle_complete = false;
            }
            state.scan_index = next;
            selected
        };
        for (asset, mode) in selected {
            if mode == ScrubMode::Full {
                report.scrub_mode = ScrubMode::Full;
            }
            if let Ok(mut state) = self.maintenance.lock() {
                state.in_flight_scrubs = state.in_flight_scrubs.saturating_add(1);
            }
            self.metrics.scrub_inflight.fetch_add(1, Ordering::Relaxed);
            let result = self.scrub_asset(&asset, mode).await;
            self.metrics.scrub_inflight.fetch_sub(1, Ordering::Relaxed);
            if let Ok(mut state) = self.maintenance.lock() {
                state.in_flight_scrubs = state.in_flight_scrubs.saturating_sub(1);
            }
            let Ok(scrub) = result else {
                self.metrics.repair_failures.fetch_add(1, Ordering::Relaxed);
                continue;
            };
            report.record_scrub(1, scrub.bytes_checked);
            for _ in 0..scrub.corruptions {
                report.record_corruption();
            }
            let Some(current) = self.catalog.get(&asset) else {
                continue;
            };
            let assessed = if mode == ScrubMode::Full {
                self.assess(&asset, view).await
            } else {
                self.assess_metadata(&asset, view).await
            };
            let Some(assessed) = assessed else {
                continue;
            };
            let now = Instant::now();
            let (history_snapshot, risk) = {
                let mut state =
                    self.maintenance
                        .lock()
                        .map_err(|_| RedundancyError::Unreadable {
                            detail: "maintenance state lock poisoned".to_owned(),
                        })?;
                let history = state
                    .histories
                    .entry(asset)
                    .or_insert_with(ScrubHistory::new);
                let now_tick = self.maintenance_ticks();
                history.last_metadata = Some(now);
                history.last_metadata_tick = Some(now_tick);
                if mode == ScrubMode::Full {
                    history.last_full = Some(now);
                    history.last_full_tick = Some(now_tick);
                }
                if mode == ScrubMode::Full && scrub.asset_verified {
                    history.last_success = Some(now);
                    history.last_success_tick = Some(now_tick);
                }
                if mode == ScrubMode::Full
                    && assessed.assessment.health == crate::AssetHealth::Healthy
                {
                    history.degraded_since = None;
                } else if history.degraded_since.is_none() {
                    history.degraded_since = Some(now);
                }
                let risk = self.risk_for_assessment(&assessed, &current, view, *history, now);
                (*history, risk)
            };
            let _ = history_snapshot;
            let temporary_only = assessed.actual.iter().any(|(_, node, healthy)| {
                !*healthy
                    && view.iter().any(|descriptor| {
                        descriptor.id == *node && descriptor.health == crate::NodeHealth::Suspect
                    })
            }) && risk.corrupt_fragments == 0
                && risk.placement_violations == 0;
            let temporary_unavailable = risk.unavailable_nodes.iter().any(|node| {
                view.iter().any(|descriptor| {
                    descriptor.id == *node && descriptor.health == crate::NodeHealth::Unavailable
                })
            });
            let grace_elapsed = history_snapshot.degraded_since.is_none_or(|since| {
                now.saturating_duration_since(since) >= self.temporary_failure_grace
            });
            let bad = (assessed.assessment.health != crate::AssetHealth::Healthy
                && (!temporary_only || risk.is_critical())
                && (!temporary_unavailable || risk.is_critical() || grace_elapsed))
                || (!risk.meets_scheme_requirements()
                    && (!temporary_unavailable || risk.is_critical() || grace_elapsed))
                || (risk.placement_violations > 0
                    && (!temporary_unavailable || risk.is_critical() || grace_elapsed))
                || scrub.metadata_corrupt > 0
                || scrub.wrong_generation > 0
                || scrub.wrong_identity > 0;
            let mut state = self
                .maintenance
                .lock()
                .map_err(|_| RedundancyError::Unreadable {
                    detail: "maintenance state lock poisoned".to_owned(),
                })?;
            if bad {
                let reason = if risk.corrupt_fragments > 0 {
                    RepairReason::CorruptionDetected
                } else if risk.unavailable_nodes.is_empty() && risk.draining_nodes.is_empty() {
                    RepairReason::FragmentMissing
                } else {
                    RepairReason::NodeUnavailable
                };
                let relocate = !risk.unavailable_nodes.is_empty()
                    || !risk.draining_nodes.is_empty()
                    || risk.placement_violations > 0;
                self.queue_healing_task(
                    &mut state,
                    HealingTask {
                        asset,
                        generation: current.layout.generation,
                        risk: risk.clone(),
                        reason,
                        enqueued_at: now,
                        attempts: 0,
                        relocate,
                    },
                );
            }
            if mode == ScrubMode::Full {
                let entries = self.debt_entries_for(&current, &assessed, &risk, view);
                Self::replace_debt(&mut state, asset, entries);
            }
            let missing = assessed
                .assessment
                .verdicts
                .iter()
                .filter(|(_, verdict)| {
                    matches!(
                        verdict,
                        crate::FragmentVerdict::Missing | crate::FragmentVerdict::Unreadable
                    )
                })
                .count();
            #[allow(clippy::cast_possible_truncation)]
            let missing = missing as u32;
            state
                .health_states
                .insert(asset, assessed.assessment.health);
            state.missing_by_asset.insert(asset, missing);
            state
                .placement_violations
                .insert(asset, risk.placement_violations);
            if mode == ScrubMode::Full
                && assessed.assessment.health == crate::AssetHealth::Healthy
                && risk.placement_violations == 0
            {
                state.tasks.retain(|task| task.asset != asset);
                if let Some(history) = state.histories.get_mut(&asset) {
                    history.next_attempt = None;
                }
            } else if mode == ScrubMode::Full
                && assessed.assessment.health != crate::AssetHealth::Unrecoverable
                && let Some(history) = state.histories.get_mut(&asset)
            {
                history.next_attempt = None;
            }
            self.refresh_health_metrics(&state, catalog.len());
        }
        let mut attempted = 0usize;
        let mut repair_bytes = 0u64;
        loop {
            if attempted >= self.maintenance_budgets.max_repair_assets
                || repair_bytes >= self.maintenance_budgets.max_repair_bytes
            {
                self.metrics
                    .maintenance_throttled
                    .fetch_add(1, Ordering::Relaxed);
                break;
            }
            let task = {
                let mut state =
                    self.maintenance
                        .lock()
                        .map_err(|_| RedundancyError::Unreadable {
                            detail: "maintenance state lock poisoned".to_owned(),
                        })?;
                let now = Instant::now();
                let index = state.tasks.iter().position(|task| {
                    state
                        .histories
                        .get(&task.asset)
                        .and_then(|history| history.next_attempt)
                        .is_none_or(|due| now >= due)
                });
                let Some(index) = index else {
                    break;
                };
                let task = state.tasks[index].clone();
                if !self.owns_maintenance(&mut state, task.asset, task.generation, view, now) {
                    state.tasks.remove(index);
                    continue;
                }
                if foreground_pressure && !task.risk.is_critical() {
                    self.metrics
                        .foreground_pressure_suppressed
                        .fetch_add(1, Ordering::Relaxed);
                    break;
                }
                if repair_bytes.saturating_add(task.risk.reconstruction_bytes)
                    > self.maintenance_budgets.max_repair_bytes
                    && attempted > 0
                {
                    break;
                }
                if let Some(claim) = state.claims.get(&task.asset)
                    && claim.owner == self.config.local
                    && claim.generation == task.generation
                    && claim.expires_at > now
                {
                    state.tasks.remove(index);
                    continue;
                }
                state.tasks.remove(index);
                state.claims.insert(
                    task.asset,
                    RepairClaim {
                        owner: self.config.local,
                        generation: task.generation,
                        expires_at: now + Duration::from_secs(60),
                    },
                );
                state.in_flight_repairs = state.in_flight_repairs.saturating_add(1);
                task
            };
            attempted += 1;
            repair_bytes = repair_bytes.saturating_add(task.risk.reconstruction_bytes);
            let started_at = Instant::now();
            self.metrics.repair_attempts.fetch_add(1, Ordering::Relaxed);
            self.metrics.repair_inflight.fetch_add(1, Ordering::Relaxed);
            let result = self.execute_healing_task(&task, view).await;
            self.metrics.repair_inflight.fetch_sub(1, Ordering::Relaxed);
            let elapsed = u64::try_from(started_at.elapsed().as_micros()).unwrap_or(u64::MAX);
            self.metrics
                .repair_duration_us_sum
                .fetch_add(elapsed, Ordering::Relaxed);
            self.metrics
                .repair_duration_us_max
                .fetch_max(elapsed, Ordering::Relaxed);
            let mut state = self
                .maintenance
                .lock()
                .map_err(|_| RedundancyError::Unreadable {
                    detail: "maintenance state lock poisoned".to_owned(),
                })?;
            state.in_flight_repairs = state.in_flight_repairs.saturating_sub(1);
            state.claims.remove(&task.asset);
            match result {
                Ok(bytes) => {
                    state.last_error = None;
                    report.record_repair(1, bytes);
                    report.record_reconstruction();
                    self.metrics
                        .recovered_bytes
                        .fetch_add(bytes, Ordering::Relaxed);
                    self.metrics
                        .repair_bytes_done
                        .fetch_add(bytes, Ordering::Relaxed);
                    let history = state
                        .histories
                        .entry(task.asset)
                        .or_insert_with(ScrubHistory::new);
                    let now = Instant::now();
                    history.last_success = Some(now);
                    history.last_success_tick = Some(self.maintenance_ticks());
                    history.degraded_since = None;
                    history.next_attempt = None;
                    state
                        .health_states
                        .insert(task.asset, crate::AssetHealth::Healthy);
                    state.missing_by_asset.insert(task.asset, 0);
                    state.placement_violations.insert(task.asset, 0);
                    Self::replace_debt(&mut state, task.asset, Vec::new());
                }
                Err(error @ RedundancyError::Unrecoverable { .. }) => {
                    state.last_error = Some(error.to_string());
                    report.record_failure();
                    self.metrics.repair_failures.fetch_add(1, Ordering::Relaxed);
                    self.metrics.repair_retries.fetch_add(1, Ordering::Relaxed);
                    let asset = task.asset;
                    let mut retry = task;
                    retry.attempts = retry.attempts.saturating_add(1);
                    retry.enqueued_at = Instant::now();
                    if let Some(history) = state.histories.get_mut(&asset) {
                        history.next_attempt = Some(Instant::now() + Duration::from_secs(5));
                    }
                    state
                        .health_states
                        .insert(asset, crate::AssetHealth::Unrecoverable);
                    let _ = self.queue_healing_task(&mut state, retry);
                }
                Err(error) => {
                    state.last_error = Some(error.to_string());
                    report.record_failure();
                    self.metrics.repair_failures.fetch_add(1, Ordering::Relaxed);
                    self.metrics.repair_retries.fetch_add(1, Ordering::Relaxed);
                    let asset = task.asset;
                    let mut retry = task;
                    retry.attempts = retry.attempts.saturating_add(1);
                    retry.enqueued_at = Instant::now();
                    let backoff = 2u64.saturating_pow(retry.attempts.min(8)).min(300);
                    if let Some(history) = state.histories.get_mut(&asset) {
                        history.next_attempt = Some(Instant::now() + Duration::from_secs(backoff));
                    }
                    let _ = self.queue_healing_task(&mut state, retry);
                }
            }
            self.refresh_health_metrics(&state, catalog.len());
            self.metrics
                .pending_repairs
                .store(state.tasks.len() as u64, Ordering::Relaxed);
            self.metrics.repair_bytes_pending.store(
                state
                    .tasks
                    .iter()
                    .map(|candidate| candidate.risk.reconstruction_bytes)
                    .sum(),
                Ordering::Relaxed,
            );
            self.metrics.estimated_reconstruction_bytes.store(
                state
                    .tasks
                    .iter()
                    .map(|candidate| candidate.risk.reconstruction_bytes)
                    .sum(),
                Ordering::Relaxed,
            );
        }
        let should_sweep = {
            let mut state = self
                .maintenance
                .lock()
                .map_err(|_| RedundancyError::Unreadable {
                    detail: "maintenance state lock poisoned".to_owned(),
                })?;
            let due = state
                .last_orphan_sweep
                .is_none_or(|last| last.elapsed() >= Duration::from_secs(300));
            if due {
                state.last_orphan_sweep = Some(Instant::now());
            }
            due
        };
        if should_sweep {
            let grace_ms = u64::try_from(self.orphan_grace.as_millis()).unwrap_or(u64::MAX);
            let protected: Vec<FragmentKey> = catalog
                .iter()
                .flat_map(|record| {
                    record.layout.fragments.iter().map(|fragment| {
                        FragmentKey::from_asset(
                            record.layout.asset,
                            record.layout.generation,
                            fragment.id.index,
                        )
                    })
                })
                .collect();
            if protected.len() > MAX_PROTECTED_KEYS {
                self.metrics
                    .maintenance_throttled
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                let mut swept = 0u64;
                for node in view {
                    let rpc = FragmentRpc::SweepOrphansGraceful {
                        target_incarnation: 0,
                        minimum_age_ms: grace_ms,
                        protected: protected.clone(),
                    };
                    if let Ok(FragmentReply::Swept { count }) =
                        self.call_with_refresh(node.id, rpc).await
                    {
                        swept = swept.saturating_add(count);
                    }
                }
                if swept > 0 {
                    self.metrics
                        .reclaimed_bytes
                        .fetch_add(swept, Ordering::Relaxed);
                }
            }
        }
        let finished = self.maintenance_ticks();
        report.finished_at = finished;
        let mut state = self
            .maintenance
            .lock()
            .map_err(|_| RedundancyError::Unreadable {
                detail: "maintenance state lock poisoned".to_owned(),
            })?;
        report.debt = state.debt.clone();
        state.last_report = Some(report.clone());
        Ok(report)
    }

    /// Runs one maintenance window using the catalog's current snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when the catalog cannot be read.
    pub async fn maintenance_tick(
        &self,
        view: &[NodeDescriptor],
        foreground_pressure: bool,
    ) -> Result<MaintenanceReport, RedundancyError> {
        let catalog = self.catalog.list();
        self.maintenance_tick_at(&catalog, view, foreground_pressure)
            .await
    }

    #[allow(clippy::too_many_lines)]
    async fn execute_healing_task(
        &self,
        task: &HealingTask,
        view: &[NodeDescriptor],
    ) -> Result<u64, RedundancyError> {
        let current =
            self.catalog
                .get(&task.asset)
                .ok_or_else(|| RedundancyError::MissingFragment {
                    asset: task.asset.to_string(),
                    generation: task.generation,
                    index: 0,
                })?;
        if current.layout.generation != task.generation {
            self.metrics.stale_rejected.fetch_add(1, Ordering::Relaxed);
            return Err(RedundancyError::StaleGeneration {
                generation: task.generation,
                current: current.layout.generation,
            });
        }
        let assessed = self.assess(&task.asset, view).await.ok_or_else(|| {
            RedundancyError::MissingFragment {
                asset: task.asset.to_string(),
                generation: task.generation,
                index: 0,
            }
        })?;
        let placement_bad = assessed.actual.iter().any(|(index, node, _)| {
            assessed
                .desired
                .node_for(*index)
                .is_none_or(|desired| desired != *node)
        });
        let holder_unavailable = current.layout.fragments.iter().any(|record| {
            view.iter()
                .any(|node| node.id == record.node && !node.health.serves_reads())
        });
        let suspect_holder = current.layout.fragments.iter().any(|record| {
            view.iter()
                .any(|node| node.id == record.node && node.health == crate::NodeHealth::Suspect)
        });
        let under_protected = assessed.assessment.need_repair.is_empty()
            && assessed.assessment.health != crate::AssetHealth::Healthy;
        let relocate = (task.relocate
            || matches!(
                task.reason,
                RepairReason::NodeUnavailable | RepairReason::DrainReplacement
            ))
            && !suspect_holder;
        if under_protected
            || relocate
            || (placement_bad && !suspect_holder)
            || holder_unavailable && !suspect_holder
        {
            let info = InformationAsset::new(task.asset, current.layout.logical_len);
            let (bytes, _) = self.read(info).await?;
            let replacement_view: Vec<NodeDescriptor> = view
                .iter()
                .filter(|node| node.health.accepts_new())
                .cloned()
                .collect();
            let control_generation = current.control_generation.saturating_add(1).max(1);
            self.build_publish(BuildRequest {
                asset: info,
                bytes: &bytes,
                params: current.layout.params,
                min_available: current.layout.min_available,
                locality: current.layout.locality,
                allowed: &current.layout.allowed_domains,
                view: &replacement_view,
                current: Some(&current),
                control_gen: control_generation,
            })
            .await?;
            self.metrics.transitions.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .transitions_with_overlap
                .fetch_add(1, Ordering::Relaxed);
            self.verify_healed_asset(&task.asset, view).await?;
            return Ok(current.layout.logical_len);
        }
        if assessed.assessment.need_repair.is_empty() {
            return Ok(0);
        }
        let source_exclusions = assessed
            .assessment
            .verdicts
            .iter()
            .filter_map(|(index, verdict)| {
                matches!(
                    verdict,
                    crate::FragmentVerdict::Missing | crate::FragmentVerdict::ChecksumMismatch
                )
                .then_some(*index)
            })
            .collect::<Vec<_>>();
        let rebuilt = self
            .execute_repair_task(
                &current.layout,
                &assessed.assessment.need_repair,
                &source_exclusions,
                view,
            )
            .await?;
        let repaired_bytes = rebuilt
            .iter()
            .filter_map(|(index, _)| {
                current
                    .layout
                    .fragments
                    .get(*index as usize)
                    .map(|record| record.stored_len)
            })
            .sum::<u64>();
        self.metrics.repair_sources.fetch_add(
            u64::from(current.layout.params.required_pieces()),
            Ordering::Relaxed,
        );
        self.metrics
            .repair_targets
            .fetch_add(rebuilt.len() as u64, Ordering::Relaxed);
        self.verify_healed_asset(&task.asset, view).await?;
        Ok(repaired_bytes)
    }

    async fn verify_healed_asset(
        &self,
        asset: &AssetId,
        view: &[NodeDescriptor],
    ) -> Result<(), RedundancyError> {
        let current = self
            .catalog
            .get(asset)
            .ok_or_else(|| RedundancyError::MissingFragment {
                asset: asset.to_string(),
                generation: 0,
                index: 0,
            })?;
        let info = InformationAsset::new(*asset, current.layout.logical_len);
        let (bytes, degraded) = self.read(info).await?;
        info.verify_bytes(&bytes)?;
        let assessed =
            self.assess(asset, view)
                .await
                .ok_or_else(|| RedundancyError::MissingFragment {
                    asset: asset.to_string(),
                    generation: current.layout.generation,
                    index: 0,
                })?;
        if degraded || assessed.assessment.health != crate::AssetHealth::Healthy {
            return Err(RedundancyError::VerificationFailed {
                detail: format!("repaired asset {asset} did not converge"),
            });
        }
        Ok(())
    }

    /// Repairs a degraded asset: queues explicit deficits through the
    /// bounded repair engine, then reconstructs each task from minimum
    /// pieces, re-encodes the missing shards, reinstalls them on the
    /// layout-named holders (generation-fenced before install and after),
    /// and `Has`-verifies. Idempotent and resumable: reinstalling identical
    /// bytes is a no-op and a fenced-out run reports `skipped_stale` for
    /// the caller to retry.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when no layout exists, reconstruction
    /// fails, or installs cannot verify.
    #[allow(clippy::too_many_lines)]
    pub async fn repair(
        &self,
        asset: &AssetId,
        view: &[NodeDescriptor],
    ) -> Result<RepairReport, RedundancyError> {
        let current = self
            .catalog
            .get(asset)
            .ok_or_else(|| RedundancyError::MissingFragment {
                asset: asset.to_string(),
                generation: 0,
                index: 0,
            })?;
        let generation = current.layout.generation;
        let assessed =
            self.assess(asset, view)
                .await
                .ok_or_else(|| RedundancyError::MissingFragment {
                    asset: asset.to_string(),
                    generation,
                    index: 0,
                })?;
        let assessment = &assessed.assessment;
        if assessment.need_repair.is_empty() {
            return Ok(RepairReport {
                asset: *asset,
                generation,
                rebuilt: Vec::new(),
                skipped_stale: false,
            });
        }
        let holders: Vec<NodeId> = current
            .layout
            .fragments
            .iter()
            .map(|record| record.node)
            .collect();
        let reason = if assessment.verdicts.iter().any(|(_, verdict)| {
            *verdict != crate::FragmentVerdict::Healthy
                && *verdict != crate::FragmentVerdict::Missing
        }) {
            RepairReason::CorruptionDetected
        } else {
            RepairReason::FragmentMissing
        };
        let targets =
            reconstruction_targets(*asset, generation, &assessment.need_repair, &holders, view);
        let source_exclusions = assessment
            .verdicts
            .iter()
            .filter_map(|(index, verdict)| {
                matches!(
                    verdict,
                    crate::FragmentVerdict::Missing | crate::FragmentVerdict::ChecksumMismatch
                )
                .then_some(*index)
            })
            .collect::<Vec<_>>();
        let target_map: HashMap<u32, NodeId> = targets.targets.into_iter().collect();
        let tasks = {
            let mut engine = self
                .repair
                .lock()
                .map_err(|_| RedundancyError::Unreadable {
                    detail: "repair lock poisoned".to_owned(),
                })?;
            engine.queue_assessment(assessment, reason, &target_map)?;
            engine.tick(false)
        };
        let mut report = RepairReport {
            asset: *asset,
            generation,
            rebuilt: Vec::new(),
            skipped_stale: false,
        };
        for task in tasks {
            if self.catalog_generation(asset) != Some(task.generation) {
                self.metrics.stale_rejected.fetch_add(1, Ordering::Relaxed);
                report.skipped_stale = true;
                continue;
            }
            match self
                .execute_repair_task(&current.layout, &task.fragments, &source_exclusions, view)
                .await
            {
                Ok(rebuilt) => report.rebuilt.extend(rebuilt),
                Err(error) => {
                    if let Ok(mut engine) = self.repair.lock() {
                        let _ = engine.requeue(task);
                    }
                    return Err(error);
                }
            }
            if self.catalog_generation(asset) != Some(generation) {
                self.metrics.stale_rejected.fetch_add(1, Ordering::Relaxed);
                report.skipped_stale = true;
                break;
            }
        }
        if !report.rebuilt.is_empty()
            && let Ok(mut state) = self.maintenance.lock()
        {
            state
                .tasks
                .retain(|task| task.asset != *asset || task.generation != generation);
            state.claims.remove(asset);
            if let Some(history) = state.histories.get_mut(asset) {
                history.next_attempt = None;
                history.degraded_since = None;
            }
            state
                .health_states
                .insert(*asset, crate::AssetHealth::Healthy);
            state.missing_by_asset.insert(*asset, 0);
            state.placement_violations.insert(*asset, 0);
            Self::replace_debt(&mut state, *asset, Vec::new());
            let pending = state.tasks.len() as u64;
            let pending_bytes: u64 = state
                .tasks
                .iter()
                .map(|task| task.risk.reconstruction_bytes)
                .sum();
            self.metrics
                .pending_repairs
                .store(pending, Ordering::Relaxed);
            self.metrics
                .repair_bytes_pending
                .store(pending_bytes, Ordering::Relaxed);
            self.metrics
                .estimated_reconstruction_bytes
                .store(pending_bytes, Ordering::Relaxed);
        }
        Ok(report)
    }

    /// Drains a node for one asset: reads through the current layout, builds
    /// a full replacement generation on the view without `node`, publishes,
    /// and drops the old fragments best-effort. Callers iterate assets (the
    /// catalog lists nothing by design — the control plane owns
    /// enumeration).
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when no layout exists, reads fail, or the
    /// replacement cannot publish.
    pub async fn drain(
        &self,
        asset: &AssetId,
        node: NodeId,
        view_without_node: &[NodeDescriptor],
        control_gen: u64,
    ) -> Result<PublishedLayout, RedundancyError> {
        let current = self
            .catalog
            .get(asset)
            .ok_or_else(|| RedundancyError::MissingFragment {
                asset: asset.to_string(),
                generation: 0,
                index: 0,
            })?;
        if !current
            .layout
            .fragments
            .iter()
            .any(|record| record.node == node)
        {
            return Ok(current);
        }
        let info = InformationAsset::new(*asset, current.layout.logical_len);
        let (bytes, _) = self.read(info).await?;
        let filtered: Vec<NodeDescriptor> = view_without_node
            .iter()
            .filter(|descriptor| descriptor.id != node)
            .cloned()
            .collect();
        self.build_publish(BuildRequest {
            asset: info,
            bytes: &bytes,
            params: current.layout.params,
            min_available: current.layout.min_available,
            locality: current.layout.locality,
            allowed: &current.layout.allowed_domains,
            view: &filtered,
            current: Some(&current),
            control_gen,
        })
        .await
    }

    /// Executes one repair task: fetch minimum, reconstruct, re-encode the
    /// missing shards, reinstall on the layout-named holders, verify.
    ///
    /// Repair never relocates: replacement bytes land on the holder the
    /// published layout names (when it serves reads), so the layout stays
    /// authoritative without a new generation. Holders that no longer serve
    /// fail the task — relocation is a new generation owned by
    /// [`Self::drain`] and [`Self::transition`], never silent adoption.
    async fn execute_repair_task(
        &self,
        layout: &RedundancyLayout,
        missing: &[u32],
        source_exclusions: &[u32],
        view: &[NodeDescriptor],
    ) -> Result<Vec<(u32, NodeId)>, RedundancyError> {
        let required = layout.params.required_pieces().max(layout.min_available) as usize;
        let (mut pieces, _) = self
            .fetch_minimum_with_view(layout, required, view, source_exclusions, true)
            .await?;
        pieces.sort_by_key(|(index, _)| *index);
        let info = InformationAsset::new(layout.asset, layout.logical_len);
        let bytes = decode_pieces(&pieces, layout, layout.logical_len, self.config)?;
        info.verify_bytes(&bytes)?;
        let shards = encode_params(&bytes, layout.params)?;
        let mut rebuilt = Vec::with_capacity(missing.len());
        for index in missing {
            // Index < total <= 40, fits `usize` on every target.
            let slot = *index as usize;
            let Some(record) = layout.fragments.get(slot) else {
                continue;
            };
            if record.id.index != *index {
                continue;
            }
            // Unknown holders are retried optimistically (stale views fail
            // naturally at the holder); positively-unserving holders fail
            // fast so drain/transition owns their relocation.
            let known_dead = view.iter().any(|descriptor| {
                descriptor.id == record.node && !descriptor.health.serves_reads()
            });
            if known_dead {
                return Err(RedundancyError::Placement {
                    detail: format!("repair holder for fragment {index} no longer serves"),
                });
            }
            let node = record.node;
            let Some(shard) = shards.get(slot) else {
                continue;
            };
            let id = FragmentId::for_bytes(
                layout.asset,
                layout.generation,
                *index,
                layout.params,
                shard,
            );
            if id.content != record.id.content {
                return Err(RedundancyError::CorruptFragment {
                    detail: format!("repair re-encode mismatch at fragment {index}"),
                });
            }
            self.put_shard(node, &id, shard).await?;
            let key = FragmentKey::from_asset(layout.asset, layout.generation, *index);
            let (_, healthy) = self.has_one_checked(node, &key).await?;
            if !healthy {
                return Err(RedundancyError::Unreachable {
                    detail: format!("repaired fragment {index} failed verification"),
                });
            }
            self.metrics
                .repair_fragments_rebuilt
                .fetch_add(1, Ordering::Relaxed);
            rebuilt.push((*index, node));
        }
        let info = InformationAsset::new(layout.asset, layout.logical_len);
        let (verified, _) = self.read(info).await?;
        info.verify_bytes(&verified)?;
        Ok(rebuilt)
    }

    /// Builds, verifies, publishes, and retires one generation (shared by
    /// protect, transition, and drain).
    #[allow(clippy::too_many_lines)]
    async fn build_publish(
        &self,
        request: BuildRequest<'_>,
    ) -> Result<PublishedLayout, RedundancyError> {
        let BuildRequest {
            asset,
            bytes,
            params,
            min_available,
            locality,
            allowed,
            view,
            current,
            control_gen,
        } = request;
        params.validate()?;
        if min_available == 0 || min_available > params.total_fragments() {
            return Err(RedundancyError::invalid_params(format!(
                "minimum available {min_available} outside 1..={}",
                params.total_fragments()
            )));
        }
        let generation = match current {
            None => 1,
            Some(record) => record
                .layout
                .generation
                .checked_add(1)
                .ok_or_else(|| RedundancyError::invalid_asset("generation space exhausted"))?,
        };
        if generation == 0 {
            return Err(RedundancyError::invalid_asset("generation space exhausted"));
        }
        let total = params.total_fragments();
        let scope = self.effective_scope(view);
        let placement = plan_placement_with_locality(
            asset.id, generation, total, scope, view, allowed, locality,
        )?;
        // Failure independence is a hard publish precondition, never a
        // best-effort hint. Exact MDS condition: after losing the `m`
        // heaviest failure domains (the worst case for the layout), at
        // least `k` fragments must remain. Replication (`k = 1`,
        // `m = copies - 1`) reduces to one copy per domain; erasure coding
        // allows parity to share a domain whenever the worst case still
        // leaves `k` fragments.
        let mut loads: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for fragment in &placement.fragments {
            *loads.entry(fragment.domain_key.clone()).or_insert(0) += 1;
        }
        let mut domain_loads: Vec<usize> = loads.values().copied().collect();
        domain_loads.sort_unstable_by(|left, right| right.cmp(left));
        #[allow(clippy::cast_possible_truncation)]
        let tolerance = params.tolerance() as usize;
        #[allow(clippy::cast_possible_truncation)]
        let required = params.required_pieces().max(min_available) as usize;
        let lost_in_worst_case: usize = domain_loads.iter().take(tolerance).sum();
        let surviving = total as usize - lost_in_worst_case;
        if surviving < required {
            return Err(RedundancyError::Placement {
                detail: format!(
                    "layout survives only {surviving} fragments after losing its {tolerance} heaviest failure domains, needs {required}"
                ),
            });
        }
        async {
            let shards = encode_params(bytes, params)?;
            let records = self
                .upload_shards(asset, generation, params, &shards, &placement, view)
                .await?;
            // Independence is re-checked against the ACTUAL installed holders
            // (fallbacks can move a fragment), never the desired plan alone.
            // Losing the `tolerance` heaviest domains must still leave enough
            // fragments to reconstruct; otherwise the build fails closed and
            // nothing is published.
            verify_installed_independence(&records, params, min_available, scope, view)?;
            self.verify_placed(asset, generation, &records).await?;
            // Re-fetch the minimum set and decode: a successful encode alone
            // never suffices — the image must verify against the asset id.
            let layout_probe = RedundancyLayout::new_with_contract(
                asset.id,
                asset.logical_len,
                generation,
                params,
                records.clone(),
                min_available,
                locality,
                allowed.to_vec(),
            )?;
            let required = params.required_pieces().max(layout_probe.min_available) as usize;
            let (mut pieces, _) = self.fetch_minimum(&layout_probe, required).await?;
            pieces.sort_by_key(|(index, _)| *index);
            let decoded = decode_pieces(&pieces, &layout_probe, asset.logical_len, self.config)?;
            asset.verify_bytes(&decoded)?;
            self.metrics.encodes.fetch_add(1, Ordering::Relaxed);
            // Associate the layout on every holder before catalog publish:
            // staged-but-unassociated fragments must stay transient (so
            // `orphans` and sweeps stay meaningful), never permanent. Holder
            // layouts without catalog authority are inert; a failed association
            // fails the build (retries are idempotent on identical bytes).
            self.push_layout(&layout_probe, control_gen).await?;
            let record = PublishedLayout {
                control_generation: control_gen,
                layout: layout_probe,
            };
            let published = match self.publisher.publish(record) {
                Ok(published) => published,
                Err(error) => {
                    if matches!(error, RedundancyError::StaleGeneration { .. }) {
                        self.metrics.stale_rejected.fetch_add(1, Ordering::Relaxed);
                    }
                    return Err(error);
                }
            };
            if let Some(old) = current
                && old.layout.generation != published.layout.generation
            {
                self.retire_layout(&old.layout).await;
            }
            Ok(published)
        }
        .await
    }

    /// Uploads every shard with deterministic fallback: on target failure
    /// the next-ranked eligible node takes the shard, and the ACTUAL holder
    /// is recorded (never the wish).
    async fn upload_shards(
        &self,
        asset: InformationAsset,
        generation: u64,
        params: SchemeParams,
        shards: &[Vec<u8>],
        placement: &crate::DesiredPlacement,
        view: &[NodeDescriptor],
    ) -> Result<Vec<FragmentRecord>, RedundancyError> {
        let mut records = Vec::with_capacity(shards.len());
        // Holders already committed in this generation: a fallback must
        // never co-locate two fragments of one layout on one failure
        // domain (that would silently reduce the advertised tolerance).
        let mut used: Vec<NodeId> = Vec::new();
        for (position, shard) in shards.iter().enumerate() {
            // Position < total <= 40, fits `u32`.
            #[allow(clippy::cast_possible_truncation)]
            let index = position as u32;
            let id = FragmentId::for_bytes(asset.id, generation, index, params, shard);
            let first = placement
                .node_for(index)
                .ok_or_else(|| RedundancyError::Placement {
                    detail: format!("placement missing fragment {index}"),
                })?;
            let mut tried: Vec<NodeId> = used.clone();
            let mut candidate = first;
            let record = loop {
                // A desired slot can name a node an earlier fragment already
                // holds (the plan's first choice may have failed and fallen
                // back). Co-location would silently reduce the advertised
                // tolerance, so move on before any bytes move.
                if used.contains(&candidate) {
                    let alternates =
                        reconstruction_targets(asset.id, generation, &[index], &tried, view);
                    let Some((_, next)) = alternates.targets.into_iter().next() else {
                        return Err(RedundancyError::Placement {
                            detail: format!("no independent holder left for fragment {index}"),
                        });
                    };
                    tried.push(candidate);
                    candidate = next;
                    continue;
                }
                match self.put_shard(candidate, &id, shard).await {
                    Ok(()) => {
                        break FragmentRecord {
                            id,
                            node: candidate,
                            stored_len: shard.len() as u64,
                        };
                    }
                    Err(last) => {
                        tried.push(candidate);
                        let alternates =
                            reconstruction_targets(asset.id, generation, &[index], &tried, view);
                        let Some((_, next)) = alternates.targets.into_iter().next() else {
                            return Err(last);
                        };
                        if tried.len() > view.len() {
                            return Err(last);
                        }
                        candidate = next;
                    }
                }
            };
            used.push(record.node);
            records.push(record);
        }
        Ok(records)
    }

    /// Stores one shard on one holder, verifying the echoed content hash.
    async fn put_shard(
        &self,
        node: NodeId,
        id: &FragmentId,
        shard: &[u8],
    ) -> Result<(), RedundancyError> {
        let rpc = FragmentRpc::Put {
            key: FragmentKey::from_asset(id.asset, id.generation, id.index),
            role: id.role.as_u8(),
            params_tag: id.params_tag,
            content: id.content,
            stored_len: shard.len() as u64,
            target_incarnation: 0,
            bytes: shard.to_vec(),
        };
        match self.call_with_refresh(node, rpc).await? {
            FragmentReply::Stored { content } if content == id.content => {
                if node == self.config.local {
                    self.metrics
                        .local_bytes_written
                        .fetch_add(shard.len() as u64, Ordering::Relaxed);
                } else {
                    self.metrics
                        .remote_bytes_sent
                        .fetch_add(shard.len() as u64, Ordering::Relaxed);
                }
                Ok(())
            }
            FragmentReply::Stored { .. } => Err(RedundancyError::CorruptFragment {
                detail: "holder echoed a wrong content hash".to_owned(),
            }),
            FragmentReply::Refused { reason } => {
                Err(TransportFailure::Refused { reason }.into_error("put"))
            }
            other => Err(RedundancyError::Unreachable {
                detail: format!("put answered {other:?}"),
            }),
        }
    }

    /// Verifies every placed fragment reports healthy before publish.
    async fn verify_placed(
        &self,
        asset: InformationAsset,
        generation: u64,
        records: &[FragmentRecord],
    ) -> Result<(), RedundancyError> {
        for record in records {
            let key = FragmentKey::from_asset(asset.id, generation, record.id.index);
            let (_, healthy) = self.has_one_checked(record.node, &key).await?;
            if !healthy {
                return Err(RedundancyError::Unreachable {
                    detail: format!(
                        "freshly placed fragment {} failed verification",
                        record.id.index
                    ),
                });
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    /// Fetches the minimum pieces for reconstruction from ranked, verified
    /// sources. Maintenance replication hedges one alternate copy; RS launches
    /// the required independent shard window concurrently and only launches
    /// alternates when a source fails verification.
    async fn fetch_minimum(
        &self,
        layout: &RedundancyLayout,
        required: usize,
    ) -> Result<(Vec<(u32, Vec<u8>)>, bool), RedundancyError> {
        self.fetch_minimum_with_view(layout, required, &[], &[], false)
            .await
    }

    #[allow(clippy::too_many_lines)]
    async fn fetch_minimum_with_view(
        &self,
        layout: &RedundancyLayout,
        required: usize,
        view: &[NodeDescriptor],
        excluded: &[u32],
        allow_hedge: bool,
    ) -> Result<(Vec<(u32, Vec<u8>)>, bool), RedundancyError> {
        let required = required.min(layout.fragments.len());
        if required == 0 {
            return Err(RedundancyError::Unrecoverable {
                detail: "layout has no reconstructable pieces".to_owned(),
            });
        }
        let latency = self
            .source_latency_us
            .lock()
            .map(|stats| stats.clone())
            .unwrap_or_default();
        let mut candidates: Vec<&FragmentRecord> = layout
            .fragments
            .iter()
            .filter(|record| !excluded.contains(&record.id.index))
            .collect();
        let fallback_candidates = candidates.clone();
        candidates.sort_by(|left, right| {
            let left_local = left.node == self.config.local;
            let right_local = right.node == self.config.local;
            let left_data = matches!(
                left.id.role,
                crate::FragmentRole::Data | crate::FragmentRole::Replica
            );
            let right_data = matches!(
                right.id.role,
                crate::FragmentRole::Data | crate::FragmentRole::Replica
            );
            right_data
                .cmp(&left_data)
                .then_with(|| right_local.cmp(&left_local))
                .then_with(|| {
                    if allow_hedge {
                        latency
                            .get(&left.node)
                            .unwrap_or(&0)
                            .cmp(latency.get(&right.node).unwrap_or(&0))
                    } else {
                        std::cmp::Ordering::Equal
                    }
                })
                .then_with(|| left.id.index.cmp(&right.id.index))
        });
        if !view.is_empty() {
            let by_id: HashMap<NodeId, NodeDescriptor> =
                view.iter().map(|node| (node.id, node.clone())).collect();
            let mut domain_used = HashSet::new();
            let mut independent = Vec::with_capacity(candidates.len());
            let mut remaining = Vec::with_capacity(candidates.len());
            for candidate in candidates {
                let key = by_id.get(&candidate.node).map_or_else(
                    || format!("node:{}", candidate.node.as_u64()),
                    |node| node.domain.independence_key(self.effective_scope(view)),
                );
                if domain_used.insert(key) {
                    independent.push(candidate);
                } else {
                    remaining.push(candidate);
                }
            }
            independent.extend(remaining);
            candidates = independent;
        }
        let initial_width = if allow_hedge && required == 1 {
            2.min(candidates.len())
        } else {
            required.min(candidates.len())
        };
        if initial_width > required {
            self.metrics
                .multi_source_fetches
                .fetch_add(1, Ordering::Relaxed);
            self.metrics
                .multi_source_hedges
                .fetch_add((initial_width - required) as u64, Ordering::Relaxed);
        }
        let ignore_hedge_errors = initial_width > required
            && candidates
                .first()
                .is_some_and(|record| record.node == self.config.local);
        let mut pieces = Vec::with_capacity(required);
        let mut degraded = false;
        let mut pending = candidates;
        let mut attempted = 0usize;
        let mut first = true;
        while !pending.is_empty() && pieces.len() < required {
            if attempted >= self.config.max_fragment_reads {
                return Err(RedundancyError::Overloaded {
                    detail: "reconstruction exceeds fragment-read bound".to_owned(),
                });
            }
            let width = if first {
                initial_width
            } else {
                self.config
                    .fetch_concurrency
                    .max(1)
                    .min(pending.len())
                    .min(required - pieces.len())
            };
            if width == 0 {
                break;
            }
            let batch: Vec<&FragmentRecord> = pending.drain(..width).collect();
            attempted += batch.len();
            for outcome in join_all(batch.iter().map(|record| self.fetch_one(record))).await {
                match outcome {
                    Ok(piece) => pieces.push(piece),
                    Err(error) => {
                        self.metrics.source_failures.fetch_add(1, Ordering::Relaxed);
                        if matches!(error, RedundancyError::CorruptFragment { .. }) {
                            self.metrics
                                .corruption_detected
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        if !ignore_hedge_errors || pieces.len() < required {
                            degraded = true;
                        }
                    }
                }
            }
            first = false;
        }
        if pieces.len() < required {
            for record in fallback_candidates {
                if pieces.len() >= required {
                    break;
                }
                if attempted >= self.config.max_fragment_reads {
                    break;
                }
                if pieces.iter().any(|(index, _)| *index == record.id.index) {
                    continue;
                }
                attempted += 1;
                match self.fetch_one(record).await {
                    Ok(piece) => pieces.push(piece),
                    Err(error) => {
                        self.metrics.source_failures.fetch_add(1, Ordering::Relaxed);
                        if matches!(error, RedundancyError::CorruptFragment { .. }) {
                            self.metrics
                                .corruption_detected
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        degraded = true;
                    }
                }
            }
        }
        if pieces.len() < required {
            self.metrics
                .reconstruction_failures
                .fetch_add(1, Ordering::Relaxed);
            return Err(RedundancyError::Unrecoverable {
                detail: format!(
                    "only {} of {required} required pieces survived",
                    pieces.len()
                ),
            });
        }
        pieces.sort_by_key(|(index, _)| *index);
        pieces.truncate(required);
        Ok((pieces, degraded))
    }

    /// Fetches and verifies one fragment against its layout record.
    async fn fetch_one(&self, record: &FragmentRecord) -> Result<(u32, Vec<u8>), RedundancyError> {
        let key = FragmentKey::from_asset(record.id.asset, record.id.generation, record.id.index);
        let start = Instant::now();
        self.metrics.fetch_count.fetch_add(1, Ordering::Relaxed);
        let rpc = FragmentRpc::Get {
            key,
            target_incarnation: 0,
        };
        let reply = match self.call_with_refresh(record.node, rpc).await {
            Ok(reply) => reply,
            Err(error) => {
                self.metrics
                    .fetch_error_count
                    .fetch_add(1, Ordering::Relaxed);
                return Err(error);
            }
        };
        let outcome = match reply {
            FragmentReply::Bytes { content, bytes } => {
                // The reply's content tag is a hint; the layout record is
                // the authority — bytes verify against it, never the hint.
                let _ = content;
                if bytes.len() as u64 != record.stored_len
                    || record.id.verify_bytes(&bytes).is_err()
                {
                    Err(RedundancyError::CorruptFragment {
                        detail: format!("fragment {} failed verification", record.id.index),
                    })
                } else {
                    Ok((record.id.index, bytes))
                }
            }
            FragmentReply::Refused { reason } => {
                Err(TransportFailure::Refused { reason }.into_error("fetch"))
            }
            other => Err(RedundancyError::Unreachable {
                detail: format!("fetch answered {other:?}"),
            }),
        };
        let elapsed = u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX);
        self.metrics
            .fetch_latency_us_sum
            .fetch_add(elapsed, Ordering::Relaxed);
        self.metrics
            .fetch_latency_us_max
            .fetch_max(elapsed, Ordering::Relaxed);
        if let Ok(mut latency) = self.source_latency_us.lock() {
            let previous = latency.get(&record.node).copied().unwrap_or(elapsed);
            latency.insert(record.node, previous.saturating_add(elapsed) / 2);
        }
        match outcome {
            Ok((index, bytes)) => {
                if record.node == self.config.local {
                    self.metrics
                        .local_bytes_read
                        .fetch_add(bytes.len() as u64, Ordering::Relaxed);
                } else {
                    self.metrics
                        .remote_bytes_received
                        .fetch_add(bytes.len() as u64, Ordering::Relaxed);
                }
                Ok((index, bytes))
            }
            Err(error) => {
                self.metrics
                    .fetch_error_count
                    .fetch_add(1, Ordering::Relaxed);
                Err(error)
            }
        }
    }

    /// Probes one holder (`Has`), mapping every failure to unhealthy.
    ///
    /// `present && !healthy` means the holder holds bytes that fail
    /// verification (length or hash): that is a corruption detection, not a
    /// missing fragment, so it is counted where it is observed. Reads may
    /// never touch a rotten copy (a healthy local copy answers instead), so
    /// probing is how rot becomes visible to repair.
    async fn has_one_checked(
        &self,
        node: NodeId,
        key: &FragmentKey,
    ) -> Result<(bool, bool), RedundancyError> {
        let rpc = FragmentRpc::Has {
            key: *key,
            target_incarnation: 0,
        };
        let timeout = self.config.rpc_timeout.min(Duration::from_secs(5));
        let first = self
            .call_with_refresh_timeout(node, rpc.clone(), timeout)
            .await;
        let reply = match first {
            Ok(reply) => reply,
            Err(_) => self.call_with_refresh_timeout(node, rpc, timeout).await?,
        };
        match reply {
            FragmentReply::Presence { present, healthy } => {
                if present && !healthy {
                    self.metrics
                        .corruption_detected
                        .fetch_add(1, Ordering::Relaxed);
                }
                Ok((present, healthy))
            }
            other => Err(RedundancyError::Unreachable {
                detail: format!("has answered {other:?}"),
            }),
        }
    }

    async fn metadata_one_checked(
        &self,
        node: NodeId,
        key: &FragmentKey,
    ) -> Result<(bool, bool), RedundancyError> {
        let rpc = FragmentRpc::Metadata {
            key: *key,
            target_incarnation: 0,
        };
        let timeout = self.config.rpc_timeout.min(Duration::from_secs(5));
        let first = self
            .call_with_refresh_timeout(node, rpc.clone(), timeout)
            .await;
        let reply = match first {
            Ok(reply) => reply,
            Err(_) => self.call_with_refresh_timeout(node, rpc, timeout).await?,
        };
        match reply {
            FragmentReply::Presence { present, healthy } => Ok((present, healthy)),
            other => Err(RedundancyError::Unreachable {
                detail: format!("metadata probe answered {other:?}"),
            }),
        }
    }

    /// Pushes the accepted layout bytes to every actual holder (layout
    /// association: binds staged fragments to authority on each node).
    /// Strict: any refusal fails the build (fencing works — a holder ahead
    /// of this generation must not be overwritten); transport errors get one
    /// retry.
    async fn push_layout(
        &self,
        layout: &RedundancyLayout,
        control_gen: u64,
    ) -> Result<(), RedundancyError> {
        let encoded = layout.encode();
        let expect = blake3_256(&encoded);
        for record in &layout.fragments {
            let rpc = FragmentRpc::PutLayout {
                asset_kind: layout.asset.kind.as_u8(),
                asset_domain: layout.asset.domain,
                asset_hash: layout.asset.hash,
                generation: layout.generation,
                control_generation: control_gen,
                layout: encoded.clone(),
                target_incarnation: 0,
            };
            let mut attempts = 0;
            loop {
                attempts += 1;
                match self.call_with_refresh(record.node, rpc.clone()).await {
                    Ok(FragmentReply::Stored { content }) if content == expect => break,
                    Ok(FragmentReply::Stored { .. }) => {
                        return Err(RedundancyError::CorruptFragment {
                            detail: "holder echoed a wrong layout hash".to_owned(),
                        });
                    }
                    Ok(FragmentReply::Refused { reason }) => {
                        return Err(TransportFailure::Refused { reason }.into_error("put-layout"));
                    }
                    Ok(other) => {
                        if attempts >= 2 {
                            return Err(RedundancyError::Unreachable {
                                detail: format!("put-layout answered {other:?}"),
                            });
                        }
                    }
                    Err(error) => {
                        if attempts >= 2 {
                            return Err(error);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Drops every fragment of a retired layout (best-effort, one retry).
    async fn retire_layout(&self, layout: &RedundancyLayout) {
        for record in &layout.fragments {
            let key = FragmentKey::from_asset(layout.asset, layout.generation, record.id.index);
            let rpc = FragmentRpc::Drop {
                key,
                target_incarnation: 0,
            };
            for _ in 0..2 {
                if self
                    .call_with_refresh(record.node, rpc.clone())
                    .await
                    .is_ok()
                {
                    break;
                }
            }
        }
    }

    /// Sweeps orphan fragments (staged bytes no accepted layout names) on
    /// every known holder.
    ///
    /// Orphans are disk hygiene only: they were never authoritative and can
    /// never answer a read, so an unreachable holder is reported rather than
    /// failing the whole sweep. The local holder is included — local RPCs
    /// short-circuit through the in-process peer handler.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::Unreachable`] only when the view names no
    /// holder at all.
    pub async fn sweep_orphans(
        &self,
        view: &[NodeDescriptor],
    ) -> Result<SweepReport, RedundancyError> {
        if view.is_empty() {
            return Err(RedundancyError::Unreachable {
                detail: "no holders known for the orphan sweep".to_owned(),
            });
        }
        let mut report = SweepReport::default();
        for node in view {
            let rpc = FragmentRpc::SweepOrphans {
                target_incarnation: 0,
            };
            match self.call_with_refresh(node.id, rpc).await {
                Ok(FragmentReply::Swept { count }) => report.swept += count,
                Ok(_) | Err(_) => report.unreachable.push(node.id),
            }
        }
        if report.swept > 0 {
            self.metrics
                .orphans_swept
                .fetch_add(report.swept, Ordering::Relaxed);
        }
        Ok(report)
    }

    /// Reconciles catalog references against holder inventories without
    /// deleting any state that could still be current or pending.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when no inventory can be reached or a
    /// holder reply is malformed. Individual unreachable holders are retained
    /// in the report and never treated as deletion permission.
    #[allow(clippy::too_many_lines)]
    pub async fn anti_entropy(
        &self,
        catalog: &[PublishedLayout],
        view: &[NodeDescriptor],
    ) -> Result<AntiEntropyReport, RedundancyError> {
        if view.is_empty() {
            return Err(RedundancyError::Unreachable {
                detail: "no holders known for anti-entropy".to_owned(),
            });
        }
        let mut expected: HashMap<FragmentKey, (NodeId, FragmentRecord)> = HashMap::new();
        for record in catalog {
            for fragment in &record.layout.fragments {
                expected.insert(
                    FragmentKey::from_asset(
                        record.layout.asset,
                        record.layout.generation,
                        fragment.id.index,
                    ),
                    (fragment.node, fragment.clone()),
                );
            }
        }
        let mut report = AntiEntropyReport::default();
        let mut seen: HashMap<FragmentKey, NodeId> = HashMap::new();
        let mut orphan_candidates = 0u64;
        let mut inventory_complete = true;
        for node in view {
            let mut offset = 0u32;
            loop {
                let rpc = FragmentRpc::InventoryPage {
                    target_incarnation: 0,
                    offset,
                    limit: 1024,
                };
                let Ok(FragmentReply::InventoryPage {
                    entries,
                    next_offset,
                }) = self.call_with_refresh(node.id, rpc).await
                else {
                    inventory_complete = false;
                    report.unreachable.push(node.id);
                    break;
                };
                for entry in entries {
                    report.inspected = report.inspected.saturating_add(1);
                    if let Some(previous) = seen.insert(entry.key, node.id)
                        && previous != node.id
                    {
                        report.duplicate_copies = report.duplicate_copies.saturating_add(1);
                    }
                    if let Some((expected_node, record)) = expected.get(&entry.key) {
                        let identity_matches = entry.content == record.id.content
                            && entry.stored_len == record.stored_len
                            && entry.role == record.id.role.as_u8()
                            && entry.params_tag == record.id.params_tag;
                        if *expected_node == node.id && identity_matches {
                            report.referenced = report.referenced.saturating_add(1);
                        } else {
                            report.wrong_identity = report.wrong_identity.saturating_add(1);
                            orphan_candidates = orphan_candidates.saturating_add(1);
                        }
                    } else if entry.layout_accepted {
                        report.stale_generation = report.stale_generation.saturating_add(1);
                    } else {
                        report.orphaned = report.orphaned.saturating_add(1);
                        orphan_candidates = orphan_candidates.saturating_add(1);
                    }
                }
                let Some(next) = next_offset else {
                    break;
                };
                if next <= offset {
                    inventory_complete = false;
                    report.unreachable.push(node.id);
                    break;
                }
                offset = next;
            }
        }
        let grace_ms = u64::try_from(self.orphan_grace.as_millis()).unwrap_or(u64::MAX);
        if inventory_complete {
            let protected: Vec<FragmentKey> = expected.keys().copied().collect();
            if protected.len() > MAX_PROTECTED_KEYS {
                self.metrics
                    .maintenance_throttled
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                for node in view {
                    let rpc = FragmentRpc::SweepOrphansGraceful {
                        target_incarnation: 0,
                        minimum_age_ms: grace_ms,
                        protected: protected.clone(),
                    };
                    match self.call_with_refresh(node.id, rpc).await {
                        Ok(FragmentReply::Swept { count }) => {
                            report.reclaimed = report.reclaimed.saturating_add(count);
                        }
                        Ok(_) | Err(_) => report.unreachable.push(node.id),
                    }
                }
            }
        }
        report.retained = orphan_candidates.saturating_sub(report.reclaimed);
        self.metrics
            .orphans_seen
            .fetch_add(report.orphaned, Ordering::Relaxed);
        self.metrics
            .orphans_retained
            .fetch_add(report.retained, Ordering::Relaxed);
        self.metrics
            .reclaimed_bytes
            .fetch_add(report.reclaimed, Ordering::Relaxed);
        Ok(report)
    }

    /// Sends one RPC, refreshing a stale incarnation and retrying once.
    /// Other refusals are returned for the caller to match on.
    async fn call_with_refresh(
        &self,
        node: NodeId,
        rpc: FragmentRpc,
    ) -> Result<FragmentReply, RedundancyError> {
        self.call_with_refresh_timeout(node, rpc, self.config.rpc_timeout)
            .await
    }

    async fn call_with_refresh_timeout(
        &self,
        node: NodeId,
        rpc: FragmentRpc,
        timeout: Duration,
    ) -> Result<FragmentReply, RedundancyError> {
        let mut incarnation = self.peer_incarnation(node);
        if incarnation == 0 && !matches!(&rpc, FragmentRpc::Hello { .. }) {
            let hello = FragmentRpc::Hello {
                target_incarnation: 0,
            };
            let hello_reply = self
                .transport
                .call(
                    PeerTarget {
                        node,
                        incarnation: 0,
                    },
                    hello,
                    timeout,
                )
                .await
                .map_err(|failure| map_failure(failure, "fragment-hello"))?;
            incarnation = match hello_reply {
                FragmentReply::Incarnation { current }
                | FragmentReply::Refused {
                    reason: RefuseReason::StaleIncarnation { current },
                } => current,
                other => {
                    return Err(RedundancyError::Unreachable {
                        detail: format!("fragment hello answered {other:?}"),
                    });
                }
            };
            if incarnation == 0 {
                return Err(RedundancyError::Unreachable {
                    detail: "fragment holder has no incarnation".to_owned(),
                });
            }
            self.note_incarnation(node, incarnation);
        }
        let mut outgoing = rpc.clone();
        set_rpc_incarnation(&mut outgoing, incarnation);
        let reply = self
            .transport
            .call(PeerTarget { node, incarnation }, outgoing, timeout)
            .await
            .map_err(|failure| map_failure(failure, "fragment-rpc"))?;
        let FragmentReply::Refused {
            reason: RefuseReason::StaleIncarnation { current },
        } = reply
        else {
            return Ok(reply);
        };
        self.note_incarnation(node, current);
        self.metrics
            .incarnation_refusals
            .fetch_add(1, Ordering::Relaxed);
        incarnation = current;
        let mut retry = rpc;
        set_rpc_incarnation(&mut retry, incarnation);
        let second = self
            .transport
            .call(PeerTarget { node, incarnation }, retry, timeout)
            .await
            .map_err(|failure| map_failure(failure, "fragment-rpc"))?;
        if matches!(
            second,
            FragmentReply::Refused {
                reason: RefuseReason::StaleIncarnation { .. },
            }
        ) {
            return Err(RedundancyError::StaleIncarnation {
                expected: current,
                current: incarnation,
            });
        }
        Ok(second)
    }

    /// Records a refreshed incarnation.
    fn note_incarnation(&self, node: NodeId, incarnation: u64) {
        if let Ok(mut guard) = self.incarnations.lock() {
            guard.insert(node, incarnation);
        }
    }

    /// Reads the catalog's current generation for fencing.
    fn catalog_generation(&self, asset: &AssetId) -> Option<u64> {
        self.catalog
            .get(asset)
            .map(|record| record.layout.generation)
    }
}

/// Derives desired placement from published records (fallback when planning
/// under a changed view fails, and the installed-state map for health math).
fn desired_from_layout(layout: &RedundancyLayout, scope: FailureScope) -> crate::DesiredPlacement {
    crate::DesiredPlacement {
        asset: layout.asset,
        generation: layout.generation,
        scope,
        fragments: layout
            .fragments
            .iter()
            .map(|record| crate::FragmentPlacement {
                index: record.id.index,
                node: record.node,
                domain_key: format!("node:{}", record.node.as_u64()),
            })
            .collect(),
    }
}

/// Sets the incarnation on every RPC variant before sending.
fn set_rpc_incarnation(rpc: &mut FragmentRpc, incarnation: u64) {
    match rpc {
        FragmentRpc::Hello {
            target_incarnation, ..
        }
        | FragmentRpc::Put {
            target_incarnation, ..
        }
        | FragmentRpc::Get {
            target_incarnation, ..
        }
        | FragmentRpc::Has {
            target_incarnation, ..
        }
        | FragmentRpc::Metadata {
            target_incarnation, ..
        }
        | FragmentRpc::Drop {
            target_incarnation, ..
        }
        | FragmentRpc::PutLayout {
            target_incarnation, ..
        }
        | FragmentRpc::AbortLayout {
            target_incarnation, ..
        }
        | FragmentRpc::GetLayout {
            target_incarnation, ..
        }
        | FragmentRpc::ProbeAsset {
            target_incarnation, ..
        }
        | FragmentRpc::SweepOrphans {
            target_incarnation, ..
        }
        | FragmentRpc::Inventory {
            target_incarnation, ..
        }
        | FragmentRpc::InventoryPage {
            target_incarnation, ..
        }
        | FragmentRpc::SweepOrphansGraceful {
            target_incarnation, ..
        } => *target_incarnation = incarnation,
    }
}

/// Maps transport failures into fabric errors at plane call sites.
fn map_failure(failure: TransportFailure, op: &str) -> RedundancyError {
    failure.into_error(op)
}

/// Verifies that the fragments actually installed still honor the scheme's
/// failure-independence contract.
///
/// Exact MDS condition: after losing the `tolerance` heaviest failure
/// domains (the worst case for the layout), at least `required` fragments
/// must remain. Replication (`k = 1`, `m = copies - 1`) therefore requires
/// one copy per domain; erasure coding lets parity share a domain only when
/// the worst case still leaves `k` fragments. Unknown nodes are treated as
/// their own domain (fail closed for co-location, open for a single
/// unknown node).
fn verify_installed_independence(
    records: &[crate::FragmentRecord],
    params: SchemeParams,
    min_available: u32,
    scope: FailureScope,
    view: &[NodeDescriptor],
) -> Result<(), RedundancyError> {
    let by_id: std::collections::HashMap<NodeId, NodeDescriptor> =
        view.iter().map(|node| (node.id, node.clone())).collect();
    let mut loads: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for record in records {
        let key = by_id.get(&record.node).map_or_else(
            || format!("node:{}", record.node.as_u64()),
            |node| node.domain.independence_key(scope),
        );
        *loads.entry(key).or_insert(0) += 1;
    }
    let mut domain_loads: Vec<usize> = loads.values().copied().collect();
    domain_loads.sort_unstable_by(|left, right| right.cmp(left));
    #[allow(clippy::cast_possible_truncation)]
    let tolerance = params.tolerance() as usize;
    #[allow(clippy::cast_possible_truncation)]
    let required = params.required_pieces().max(min_available) as usize;
    #[allow(clippy::cast_possible_truncation)]
    let total = records.len() as u32;
    let lost_in_worst_case: usize = domain_loads.iter().take(tolerance).sum();
    let surviving = total as usize - lost_in_worst_case;
    if surviving < required {
        let holders: Vec<u64> = records.iter().map(|record| record.node.as_u64()).collect();
        return Err(RedundancyError::Placement {
            detail: format!(
                "installed layout survives only {surviving} fragments after losing its {tolerance} heaviest failure domains, needs {required} (holders={holders:?} domains={domain_loads:?})"
            ),
        });
    }
    Ok(())
}

/// Encodes bytes behind the scheme abstraction (replication and RS share
/// every surrounding step).
fn encode_params(bytes: &[u8], params: SchemeParams) -> Result<Vec<Vec<u8>>, RedundancyError> {
    match params {
        SchemeParams::Replication(rule) => crate::replication::encode(bytes, rule),
        SchemeParams::ReedSolomon(rule) => crate::rs::encode(bytes, rule),
    }
}

/// Decodes verified pieces, enforcing the memory bound on coded layouts.
fn decode_pieces(
    pieces: &[(u32, Vec<u8>)],
    layout: &RedundancyLayout,
    logical_len: u64,
    config: DistConfig,
) -> Result<Vec<u8>, RedundancyError> {
    match layout.params {
        SchemeParams::Replication(_) => crate::replication::decode(pieces),
        SchemeParams::ReedSolomon(params) => {
            if (pieces.len() as u64) * u64::from(params.fragment_len) > config.max_decode_bytes {
                return Err(RedundancyError::Overloaded {
                    detail: "reconstruction exceeds memory bound".to_owned(),
                });
            }
            crate::rs::decode(pieces, params, logical_len)
        }
    }
}

#[cfg(test)]
// Test cluster setup reads long; splitting it would obscure the harness.
#[allow(clippy::too_many_lines)]
mod tests {
    use super::*;
    use crate::store::{FragmentStore, LocalFragmentStore};
    use crate::transport::LoopbackTransport;
    use std::future::Future;

    struct Cluster {
        stores: Vec<(NodeId, Arc<Mutex<LocalFragmentStore>>, tempfile::TempDir)>,
        fabric: DistributedFabric,
        view: Vec<NodeDescriptor>,
    }

    fn cluster() -> Cluster {
        cluster_with_nodes(5)
    }

    /// Cluster with `nodes` independent members (one failure domain each).
    /// Wide layouts need enough members to place every fragment
    /// independently.
    fn cluster_with_nodes(nodes: u64) -> Cluster {
        let transport = Arc::new(LoopbackTransport::new());
        let mut stores = Vec::new();
        let mut view = Vec::new();
        for id in 1..=nodes {
            let node = NodeId::from_u64(id);
            let dir = tempfile::tempdir().expect("scratch");
            let (store, _) = LocalFragmentStore::open(&dir.path().join("store"), 1).expect("opens");
            let holder = Arc::new(Mutex::new(store));
            transport.add_node(node, Arc::clone(&holder));
            stores.push((node, holder, dir));
            view.push(NodeDescriptor {
                id: node,
                domain: crate::FailureDomain::node_only(node),
                health: crate::NodeHealth::Active,
                weight: 1,
            });
        }
        let catalog = Arc::new(crate::catalog::MemCatalog::new());
        let fabric = DistributedFabric::new(
            Arc::clone(&transport) as Arc<dyn FragmentTransport>,
            Arc::clone(&catalog) as Arc<dyn LayoutCatalog>,
            Arc::clone(&catalog) as Arc<dyn CatalogPublish>,
            Arc::new(crate::FabricMetrics::new()),
            DistConfig {
                rpc_timeout: Duration::from_secs(5),
                ..DistConfig::conservative()
            },
        )
        .expect("builds");
        Cluster {
            stores,
            fabric,
            view,
        }
    }

    fn holder_mut(
        cluster: &Cluster,
        node: NodeId,
    ) -> std::sync::MutexGuard<'_, LocalFragmentStore> {
        cluster
            .stores
            .iter()
            .find(|(id, _, _)| *id == node)
            .expect("holder")
            .1
            .lock()
            .expect("unpoisoned")
    }

    fn run<T>(future: impl Future<Output = T>) -> T {
        futures::executor::block_on(future)
    }

    const DOMAIN: kivi_types::SecurityDomainId = kivi_types::SecurityDomainId::from_u64(7);

    #[test]
    fn protect_kill_read_degraded_repair_healthy() {
        let cluster = cluster();
        let bytes = vec![7u8; 10_000];
        let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);
        let published = run(cluster.fabric.protect(
            asset,
            &bytes,
            &RedundancyIntent::survive_two(),
            &cluster.view,
            5,
        ))
        .expect("protects");
        assert_eq!(published.layout.generation, 1);
        let (back, degraded) = run(cluster.fabric.read(asset)).expect("reads");
        assert_eq!(back, bytes);
        assert!(!degraded);
        // Kill one holder: the next read serves degraded from alternates.
        let victim = published.layout.fragments[0].clone();
        let key = FragmentKey::from_asset(asset.id, 1, victim.id.index);
        assert!(holder_mut(&cluster, victim.node).remove_fragment(&key, 1));
        let (back, degraded) = run(cluster.fabric.read(asset)).expect("reads degraded");
        assert_eq!(back, bytes);
        assert!(degraded);
        let assessed = run(cluster.fabric.assess(&asset.id, &cluster.view)).expect("assesses");
        assert_eq!(assessed.assessment.health, crate::AssetHealth::Degraded);
        // Repair reinstalls the missing shard; health returns.
        let report = run(cluster.fabric.repair(&asset.id, &cluster.view)).expect("repairs");
        assert_eq!(report.rebuilt.len(), 1);
        let assessed = run(cluster.fabric.assess(&asset.id, &cluster.view)).expect("assesses");
        assert_eq!(assessed.assessment.health, crate::AssetHealth::Healthy);
        let (back, degraded) = run(cluster.fabric.read(asset)).expect("reads");
        assert_eq!(back, bytes);
        assert!(!degraded);
    }

    #[test]
    fn rs_survives_within_tolerance_and_fails_closed_beyond() {
        // The baseline planner picks RS(8,2) for a 1 MiB cold asset, so the
        // cluster must be wide enough to place all ten fragments on
        // independent failure domains (a narrower cluster fails closed at
        // publish — proven by `placement_refuses_independent_domains`).
        let cluster = cluster_with_nodes(10);
        // Modulo 251 bounds the value to `0..251`, fits `u8`.
        #[allow(clippy::cast_possible_truncation)]
        let bytes: Vec<u8> = (0..1_048_576usize).map(|i| (i % 251) as u8).collect();
        let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);
        let published = run(cluster.fabric.protect(
            asset,
            &bytes,
            &RedundancyIntent::survive_two(),
            &cluster.view,
            5,
        ))
        .expect("protects");
        assert!(matches!(
            published.layout.params,
            SchemeParams::ReedSolomon(_)
        ));
        let tolerance = published.layout.params.tolerance() as usize;
        assert_eq!(tolerance, 2);
        // Within tolerance: reads stay exact.
        for record in published.layout.fragments.iter().take(tolerance) {
            let key = FragmentKey::from_asset(asset.id, 1, record.id.index);
            assert!(holder_mut(&cluster, record.node).remove_fragment(&key, 1));
        }
        let (back, degraded) = run(cluster.fabric.read(asset)).expect("reads within tolerance");
        assert_eq!(back, bytes);
        assert!(degraded);
        // Beyond tolerance: fails closed, never wrong bytes.
        let record = &published.layout.fragments[tolerance];
        let key = FragmentKey::from_asset(asset.id, 1, record.id.index);
        assert!(holder_mut(&cluster, record.node).remove_fragment(&key, 1));
        let error = run(cluster.fabric.read(asset)).expect_err("beyond tolerance fails");
        assert!(matches!(error, RedundancyError::Unrecoverable { .. }));
    }

    #[test]
    fn placement_refuses_independent_domains() {
        // A 1 MiB cold asset plans RS(8,2): ten fragments that must survive
        // two independent failures. Five domains cannot honor that contract,
        // so publication fails closed before any bytes move.
        let cluster = cluster_with_nodes(5);
        let bytes: Vec<u8> = (0..1_048_576usize)
            .map(|index| u8::try_from(index % 251).unwrap_or(0))
            .collect();
        let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);
        let error = run(cluster.fabric.protect(
            asset,
            &bytes,
            &RedundancyIntent::survive_two(),
            &cluster.view,
            1,
        ))
        .expect_err("narrow cluster cannot honor the contract");
        assert!(matches!(error, RedundancyError::Placement { .. }));
        // Nothing became authoritative: a read still fails closed.
        assert!(run(cluster.fabric.read(asset)).is_err());
    }

    #[test]
    fn local_reads_prefer_the_local_holder() {
        let mut cluster = cluster_with_nodes(3);
        let bytes = vec![23u8; 8192];
        let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);
        let published = run(cluster.fabric.protect(
            asset,
            &bytes,
            &RedundancyIntent::survive_two(),
            &cluster.view,
            1,
        ))
        .expect("protects");
        // This plane *is* the first holder.
        let local = published.layout.fragments[0].node;
        cluster.fabric.config.local = local;
        // Every remote copy disappears: only a local-preferring read can
        // still answer.
        for record in published.layout.fragments.iter().skip(1) {
            let key = FragmentKey::from_asset(asset.id, 1, record.id.index);
            assert!(holder_mut(&cluster, record.node).remove_fragment(&key, 1));
        }
        let (back, degraded) = run(cluster.fabric.read(asset)).expect("local read");
        assert_eq!(back, bytes);
        assert!(!degraded);
    }

    #[test]
    fn cluster_sweep_reclaims_remote_orphans_only() {
        let cluster = cluster_with_nodes(3);
        let bytes = vec![11u8; 4096];
        let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);
        run(cluster.fabric.protect(
            asset,
            &bytes,
            &RedundancyIntent::survive_two(),
            &cluster.view,
            1,
        ))
        .expect("protects");

        // Plant an orphan on a remote holder: staged bytes under a
        // generation no published layout names.
        let victim = cluster.view[1].id;
        let key = FragmentKey::from_asset(asset.id, 999, 0);
        let staged = {
            let holder = holder_mut(&cluster, victim);
            holder
                .stage_fragment(&key, 0, 0, blake3_256(b"orphan"), 6, b"orphan", 1)
                .expect("orphan stages");
            6u64
        };
        assert_eq!(staged, 6);
        assert_eq!(holder_mut(&cluster, victim).orphans().len(), 1);

        // A cluster-wide sweep reclaims it, and only it: the published
        // layout still reads exactly.
        let report = run(cluster.fabric.sweep_orphans(&cluster.view)).expect("sweeps");
        assert_eq!(report.swept, 1);
        assert!(report.unreachable.is_empty());
        assert!(holder_mut(&cluster, victim).orphans().is_empty());
        let (back, degraded) = run(cluster.fabric.read(asset)).expect("reads after sweep");
        assert_eq!(back, bytes);
        assert!(!degraded);
    }

    #[test]
    fn stale_publish_rejected_and_counted() {
        let cluster = cluster();
        let bytes = vec![3u8; 1000];
        let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);
        run(cluster.fabric.protect(
            asset,
            &bytes,
            &RedundancyIntent::survive_two(),
            &cluster.view,
            5,
        ))
        .expect("protects");
        // A transition on a superseded control generation fails closed.
        let error = run(cluster.fabric.transition(
            asset,
            &bytes,
            SchemeParams::Replication(crate::ReplicationParams { copies: 2 }),
            &cluster.view,
            3,
        ))
        .expect_err("stale control rejected");
        assert!(matches!(error, RedundancyError::StaleGeneration { .. }));
        assert_eq!(cluster.fabric.metrics().snapshot().stale_rejected, 1);
        // A fresh control generation publishes.
        let published = run(cluster.fabric.transition(
            asset,
            &bytes,
            SchemeParams::Replication(crate::ReplicationParams { copies: 2 }),
            &cluster.view,
            9,
        ))
        .expect("fresh control wins");
        assert_eq!(published.layout.generation, 2);
    }

    #[test]
    fn drain_moves_fragments_off_the_node() {
        let cluster = cluster();
        let bytes = vec![5u8; 8000];
        let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);
        let published = run(cluster.fabric.protect(
            asset,
            &bytes,
            &RedundancyIntent::survive_two(),
            &cluster.view,
            5,
        ))
        .expect("protects");
        let victim = published.layout.fragments[0].node;
        let filtered: Vec<NodeDescriptor> = cluster
            .view
            .iter()
            .filter(|descriptor| descriptor.id != victim)
            .cloned()
            .collect();
        let moved = run(cluster.fabric.drain(&asset.id, victim, &filtered, 6)).expect("drains");
        assert_eq!(moved.layout.generation, 2);
        assert!(
            moved
                .layout
                .fragments
                .iter()
                .all(|record| record.node != victim),
            "drained node holds nothing"
        );
        let (back, _) = run(cluster.fabric.read(asset)).expect("reads after drain");
        assert_eq!(back, bytes);
    }

    #[test]
    fn incarnation_change_refuses_then_recovers_after_refresh() {
        let cluster = cluster();
        let bytes = vec![9u8; 4000];
        let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);
        let published = run(cluster.fabric.protect(
            asset,
            &bytes,
            &RedundancyIntent::survive_two(),
            &cluster.view,
            5,
        ))
        .expect("protects");
        // Learn the epoch, then restart every holder (incarnation bump).
        for (node, holder, _) in &cluster.stores {
            cluster.fabric.set_peer_incarnation(*node, 1);
            holder.lock().expect("unpoisoned").set_incarnation(2);
        }
        // Contacted holders refuse once; the plane refreshes and retries.
        // Reads fetch only the minimum pieces, so probe every holder to
        // force contact before asserting beliefs converged.
        let (back, _) = run(cluster.fabric.read(asset)).expect("recovers after refresh");
        assert_eq!(back, bytes);
        run(cluster.fabric.assess(&asset.id, &cluster.view)).expect("assesses");
        assert!(
            cluster.fabric.metrics().snapshot().incarnation_refusals >= 1,
            "refusals are counted"
        );
        // Beliefs converged on the new epoch for every contacted holder.
        for record in &published.layout.fragments {
            assert_eq!(cluster.fabric.peer_incarnation(record.node), 2);
        }
    }

    #[test]
    fn anti_entropy_classifies_and_reclaims_aged_orphans() {
        let mut cluster = cluster_with_nodes(3);
        let bytes = vec![83u8; 4096];
        let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);
        run(cluster.fabric.protect(
            asset,
            &bytes,
            &RedundancyIntent::survive_one(),
            &cluster.view,
            1,
        ))
        .expect("protects");
        let victim = cluster.view[1].id;
        let key = FragmentKey::from_asset(asset.id, 999, 0);
        holder_mut(&cluster, victim)
            .stage_fragment(&key, 0, 0, blake3_256(b"orphan"), 6, b"orphan", 1)
            .expect("orphan stages");
        let catalog = cluster.fabric.catalog.list();
        let report = run(cluster.fabric.anti_entropy(&catalog, &cluster.view)).expect("reconciles");
        assert!(report.orphaned >= 1);
        assert_eq!(report.reclaimed, 0);
        assert!(report.retained >= 1);
        cluster.fabric.set_orphan_grace(Duration::ZERO);
        let reclaimed =
            run(cluster.fabric.anti_entropy(&catalog, &cluster.view)).expect("reconciles");
        assert!(reclaimed.reclaimed >= 1);
        assert_eq!(run(cluster.fabric.read(asset)).expect("reads").0, bytes);
    }

    #[test]
    fn autonomous_tick_repairs_missing_fragment_without_admin_call() {
        let mut cluster = cluster_with_nodes(5);
        let bytes = vec![41u8; 32_000];
        let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);
        let published = run(cluster.fabric.protect(
            asset,
            &bytes,
            &RedundancyIntent::survive_two(),
            &cluster.view,
            1,
        ))
        .expect("protects");
        cluster.fabric.config.local = DistributedFabric::owner_for_asset(
            asset.id,
            published.layout.generation,
            &cluster.view,
        )
        .expect("owner");
        let victim = published.layout.fragments[0].clone();
        let key = FragmentKey::from_asset(asset.id, published.layout.generation, victim.id.index);
        assert!(holder_mut(&cluster, victim.node).remove_fragment(&key, 1));

        let report = run(cluster.fabric.maintenance_tick(&cluster.view, false)).expect("ticks");
        assert_eq!(report.scrub_assets, 1);
        assert_eq!(report.repair_assets, 1);
        let (recovered, degraded) = run(cluster.fabric.read(asset)).expect("reads after heal");
        assert_eq!(recovered, bytes);
        assert!(!degraded);
        let health = run(cluster.fabric.assess(&asset.id, &cluster.view)).expect("assesses");
        assert_eq!(health.assessment.health, crate::AssetHealth::Healthy);
        assert!(cluster.fabric.maintenance_snapshot().debt.is_empty());
    }

    #[test]
    fn autonomous_tick_honors_temporary_failure_grace() {
        let mut cluster = cluster_with_nodes(5);
        cluster
            .fabric
            .set_temporary_failure_grace(Duration::from_secs(60));
        let bytes = vec![79u8; 18_000];
        let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);
        let published = run(cluster.fabric.protect(
            asset,
            &bytes,
            &RedundancyIntent::survive_two(),
            &cluster.view,
            1,
        ))
        .expect("protects");
        let victim = published.layout.fragments[0].node;
        let key = FragmentKey::from_asset(asset.id, published.layout.generation, 0);
        assert!(holder_mut(&cluster, victim).remove_fragment(&key, 1));
        cluster
            .view
            .iter_mut()
            .find(|node| node.id == victim)
            .expect("victim view")
            .health = crate::NodeHealth::Unavailable;
        cluster.fabric.config.local = DistributedFabric::owner_for_asset(
            asset.id,
            published.layout.generation,
            &cluster.view,
        )
        .expect("owner");
        let deferred = run(cluster.fabric.maintenance_tick(&cluster.view, false)).expect("ticks");
        assert_eq!(deferred.repair_assets, 0);
        cluster.fabric.set_temporary_failure_grace(Duration::ZERO);
        let healed = run(cluster.fabric.maintenance_tick(&cluster.view, false)).expect("ticks");
        assert_eq!(healed.repair_assets, 1);
        assert_eq!(run(cluster.fabric.read(asset)).expect("reads").0, bytes);
    }

    #[test]
    fn autonomous_tick_heals_failure_domain_collapse() {
        let mut cluster = cluster_with_nodes(5);
        let bytes = vec![71u8; 20_000];
        let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);
        let published = run(cluster.fabric.protect(
            asset,
            &bytes,
            &RedundancyIntent::survive_two(),
            &cluster.view,
            1,
        ))
        .expect("protects");
        for (index, node) in cluster.view.iter_mut().enumerate() {
            node.domain.rack = if index < 2 {
                "rack-a".to_owned()
            } else {
                format!("rack-{index}")
            };
        }
        cluster.fabric.config.local = DistributedFabric::owner_for_asset(
            asset.id,
            published.layout.generation,
            &cluster.view,
        )
        .expect("owner");
        let report = run(cluster.fabric.maintenance_tick(&cluster.view, false)).expect("ticks");
        assert_eq!(report.repair_assets, 1);
        let current = cluster.fabric.catalog.get(&asset.id).expect("catalog");
        assert!(current.layout.generation > published.layout.generation);
        let mut domains = std::collections::HashSet::new();
        for record in &current.layout.fragments {
            let node = cluster
                .view
                .iter()
                .find(|node| node.id == record.node)
                .expect("placed node");
            assert!(domains.insert(node.domain.rack.clone()));
        }
        assert_eq!(run(cluster.fabric.read(asset)).expect("reads").0, bytes);
    }

    #[test]
    fn autonomous_tick_repairs_silent_corruption() {
        let mut cluster = cluster_with_nodes(3);
        let bytes = vec![53u8; 16_000];
        let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);
        let published = run(cluster.fabric.protect(
            asset,
            &bytes,
            &RedundancyIntent::survive_two(),
            &cluster.view,
            1,
        ))
        .expect("protects");
        cluster.fabric.config.local = DistributedFabric::owner_for_asset(
            asset.id,
            published.layout.generation,
            &cluster.view,
        )
        .expect("owner");
        let victim = published.layout.fragments[0].clone();
        let path = cluster
            .stores
            .iter()
            .find(|(node, _, _)| *node == victim.node)
            .expect("holder")
            .2
            .path()
            .join("store")
            .join("assets")
            .join(format!("{}-{}", asset.id.kind.name(), asset.id.hex()))
            .join(format!("frag-{}-0000.bin", published.layout.generation));
        let mut damaged = std::fs::read(&path).expect("fragment");
        let middle = damaged.len() / 2;
        damaged[middle] ^= 0xA5;
        std::fs::write(&path, damaged).expect("corrupt");

        let report = run(cluster.fabric.maintenance_tick(&cluster.view, false)).expect("ticks");
        assert!(report.corruption_detected >= 1);
        assert_eq!(report.repair_assets, 1);
        let (recovered, degraded) = run(cluster.fabric.read(asset)).expect("reads");
        assert_eq!(recovered, bytes);
        assert!(!degraded);
    }

    #[test]
    fn autonomous_tick_fails_closed_beyond_rs_threshold() {
        let mut cluster = cluster_with_nodes(3);
        let bytes = vec![67u8; 4096];
        let asset = InformationAsset::chunk_for_bytes(&bytes, DOMAIN);
        let params = SchemeParams::ReedSolomon(crate::RsParams {
            data: 2,
            parity: 1,
            fragment_len: crate::RsParams::shard_for_len(bytes.len() as u64, 2),
        });
        let published = run(cluster
            .fabric
            .transition(asset, &bytes, params, &cluster.view, 1))
        .expect("transitions");
        cluster.fabric.config.local = DistributedFabric::owner_for_asset(
            asset.id,
            published.layout.generation,
            &cluster.view,
        )
        .expect("owner");
        for record in published.layout.fragments.iter().take(2) {
            let key =
                FragmentKey::from_asset(asset.id, published.layout.generation, record.id.index);
            assert!(holder_mut(&cluster, record.node).remove_fragment(&key, 1));
        }
        let report = run(cluster.fabric.maintenance_tick(&cluster.view, false)).expect("ticks");
        assert_eq!(report.repair_failures, 1);
        assert!(run(cluster.fabric.read(asset)).is_err());
        assert_eq!(
            run(cluster.fabric.assess(&asset.id, &cluster.view))
                .expect("assesses")
                .assessment
                .health,
            crate::AssetHealth::Unrecoverable
        );
    }
}
