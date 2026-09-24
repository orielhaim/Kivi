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

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::future::join_all;
use kivi_codec::integrity::blake3_256;
use kivi_types::NodeId;

use crate::proto::{FragmentKey, FragmentReply, FragmentRpc, RefuseReason};
use crate::{
    AssetAssessment, AssetId, FailureScope, FragmentId, FragmentRecord, InformationAsset,
    NodeDescriptor, RedundancyError, RedundancyIntent, RedundancyLayout, RepairEngine,
    RepairReason, SchemeParams, assess,
    catalog::{CatalogPublish, LayoutCatalog, PublishedLayout},
    independent_survivable, plan_baseline, plan_placement, reconstruction_targets,
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

/// One generation build: everything `build_publish` needs (bundled so the
/// helper stays under the argument bound).
struct BuildRequest<'a> {
    /// Asset under protection.
    asset: InformationAsset,
    /// Exact logical bytes.
    bytes: &'a [u8],
    /// Planned scheme.
    params: SchemeParams,
    /// Allowed failure-domain labels.
    allowed: &'a [String],
    /// Cluster view for placement.
    view: &'a [NodeDescriptor],
    /// Current published record, if any.
    current: Option<&'a PublishedLayout>,
    /// Control generation publishing this build.
    control_gen: u64,
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
        if let Some(current) = &current
            && current.layout.logical_len == asset.logical_len
            && current.layout.params.tolerance() >= u32::from(intent.tolerance)
            && self.probe_healthy(&current.layout).await
        {
            return Ok(current.clone());
        }
        let params = plan_baseline(intent, asset.logical_len)?;
        let domains = {
            let mut keys: std::collections::HashSet<String> = std::collections::HashSet::new();
            for node in view {
                if node.health.accepts_new() {
                    keys.insert(node.domain.independence_key(self.config.scope));
                }
            }
            keys.len()
        };
        let params = crate::scheme::fit_to_domains(params, asset.logical_len, domains);
        self.build_publish(BuildRequest {
            asset,
            bytes,
            params,
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
        let required = layout.params.required_pieces() as usize;
        let (mut pieces, degraded) = self.fetch_minimum(layout, required).await?;
        pieces.sort_by_key(|(index, _)| *index);
        let decoded = decode_pieces(&pieces, layout, asset.logical_len, self.config)?;
        asset.verify_bytes(&decoded)?;
        self.metrics.decodes.fetch_add(1, Ordering::Relaxed);
        if degraded {
            self.metrics.degraded_reads.fetch_add(1, Ordering::Relaxed);
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
        let published = self
            .build_publish(BuildRequest {
                asset,
                bytes,
                params: target,
                allowed: &[],
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

    /// Assesses installed health: `Has`-probes every published fragment,
    /// then computes survivability over the actually-healthy holders (not
    /// the desired placement — collided or drifted installs never inflate
    /// protection).
    pub async fn assess(
        &self,
        asset: &AssetId,
        view: &[NodeDescriptor],
    ) -> Option<DistributedAssessment> {
        let current = self.catalog.get(asset)?;
        let layout = &current.layout;
        let total = layout.params.total_fragments();
        let desired = plan_placement(
            *asset,
            layout.generation,
            total,
            self.config.scope,
            view,
            &[],
        )
        .unwrap_or_else(|_| desired_from_layout(layout, self.config.scope));
        let mut actual = Vec::with_capacity(layout.fragments.len());
        let mut observed: Vec<(u32, bool)> = Vec::with_capacity(layout.fragments.len());
        for record in &layout.fragments {
            let key = FragmentKey::from_asset(*asset, layout.generation, record.id.index);
            let (present, healthy) = self.has_one(record.node, &key).await;
            actual.push((record.id.index, record.node, healthy));
            observed.push((record.id.index, present));
        }
        let healthy: Vec<u32> = actual
            .iter()
            .filter(|(_, _, healthy)| *healthy)
            .map(|(index, _, _)| *index)
            .collect();
        let by_id: HashMap<NodeId, NodeDescriptor> =
            view.iter().map(|node| (node.id, node.clone())).collect();
        let installed = desired_from_layout(layout, self.config.scope);
        let survivable = independent_survivable(
            layout.params,
            self.config.scope,
            &installed,
            &healthy,
            &by_id,
        );
        let verdicts = actual
            .iter()
            .map(|(index, _, healthy)| {
                // Presence without health is corruption, not service: it
                // owes a corruption-driven repair, not a routine refill.
                let present = observed
                    .iter()
                    .find(|(observed_index, _)| observed_index == index)
                    .is_some_and(|(_, present)| *present);
                let verdict = if *healthy {
                    crate::FragmentVerdict::Healthy
                } else if present {
                    crate::FragmentVerdict::ChecksumMismatch
                } else {
                    crate::FragmentVerdict::Missing
                };
                (*index, verdict)
            })
            .collect();
        let assessment = assess(
            *asset,
            layout.generation,
            layout.params,
            verdicts,
            survivable,
        );
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
                .execute_repair_task(&current.layout, &task.fragments, view)
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
            allowed: &[],
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
        view: &[NodeDescriptor],
    ) -> Result<Vec<(u32, NodeId)>, RedundancyError> {
        let required = layout.params.required_pieces() as usize;
        let (mut pieces, _) = self.fetch_minimum(layout, required).await?;
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
            let (_, healthy) = self.has_one(node, &key).await;
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
        Ok(rebuilt)
    }

    /// Builds, verifies, publishes, and retires one generation (shared by
    /// protect, transition, and drain).
    async fn build_publish(
        &self,
        request: BuildRequest<'_>,
    ) -> Result<PublishedLayout, RedundancyError> {
        let BuildRequest {
            asset,
            bytes,
            params,
            allowed,
            view,
            current,
            control_gen,
        } = request;
        params.validate()?;
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
        let placement = plan_placement(
            asset.id,
            generation,
            total,
            self.config.scope,
            view,
            allowed,
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
        let required = params.required_pieces() as usize;
        let lost_in_worst_case: usize = domain_loads.iter().take(tolerance).sum();
        let surviving = total as usize - lost_in_worst_case;
        if surviving < required {
            return Err(RedundancyError::Placement {
                detail: format!(
                    "layout survives only {surviving} fragments after losing its {tolerance} heaviest failure domains, needs {required}"
                ),
            });
        }
        let shards = encode_params(bytes, params)?;
        let records = self
            .upload_shards(asset, generation, params, &shards, &placement, view)
            .await?;
        // Independence is re-checked against the ACTUAL installed holders
        // (fallbacks can move a fragment), never the desired plan alone.
        // Losing the `tolerance` heaviest domains must still leave enough
        // fragments to reconstruct; otherwise the build fails closed and
        // nothing is published.
        verify_installed_independence(&records, params, self.config.scope, view)?;
        self.verify_placed(asset, generation, &records).await?;
        // Re-fetch the minimum set and decode: a successful encode alone
        // never suffices — the image must verify against the asset id.
        let layout_probe = RedundancyLayout::new(
            asset.id,
            asset.logical_len,
            generation,
            params,
            records.clone(),
        )?;
        let required = params.required_pieces() as usize;
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
            let (_, healthy) = self.has_one(record.node, &key).await;
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

    /// Fetches the minimum pieces for reconstruction: the exact required
    /// count first (bounded windows), alternates on failure. Returns the
    /// verified pieces plus whether anything failed along the way.
    async fn fetch_minimum(
        &self,
        layout: &RedundancyLayout,
        required: usize,
    ) -> Result<(Vec<(u32, Vec<u8>)>, bool), RedundancyError> {
        let mut degraded = false;
        let mut pieces = Vec::new();
        let mut attempted = 0usize;
        // Local copies first: a verified local read never crosses the mesh
        // and is the common case for the node that serves the request.
        let mut pending: Vec<&FragmentRecord> = layout
            .fragments
            .iter()
            .filter(|record| record.node == self.config.local)
            .chain(
                layout
                    .fragments
                    .iter()
                    .filter(|record| record.node != self.config.local),
            )
            .collect();
        // First pass: the exact required count, in placement order.
        let first: Vec<&FragmentRecord> = pending.drain(..required.min(pending.len())).collect();
        for batch in first.chunks(self.config.fetch_concurrency.max(1)) {
            attempted += batch.len();
            for outcome in join_all(batch.iter().map(|record| self.fetch_one(record))).await {
                match outcome {
                    Ok(piece) => pieces.push(piece),
                    Err(error) => {
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
        // Alternates until the threshold holds or holders run out.
        while pieces.len() < required && !pending.is_empty() {
            if attempted >= self.config.max_fragment_reads {
                return Err(RedundancyError::Overloaded {
                    detail: "reconstruction exceeds fragment-read bound".to_owned(),
                });
            }
            let width = self
                .config
                .fetch_concurrency
                .max(1)
                .min(pending.len())
                .min(required - pieces.len());
            let batch: Vec<&FragmentRecord> = pending.drain(..width).collect();
            attempted += batch.len();
            for outcome in join_all(batch.iter().map(|record| self.fetch_one(record))).await {
                match outcome {
                    Ok(piece) => pieces.push(piece),
                    Err(error) => {
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
        let reply = self
            .call_with_refresh(record.node, rpc)
            .await
            .inspect_err(|_| {
                self.metrics
                    .fetch_error_count
                    .fetch_add(1, Ordering::Relaxed);
            })?;
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
    async fn has_one(&self, node: NodeId, key: &FragmentKey) -> (bool, bool) {
        let rpc = FragmentRpc::Has {
            key: *key,
            target_incarnation: 0,
        };
        match self.call_with_refresh(node, rpc).await {
            Ok(FragmentReply::Presence { present, healthy }) => {
                if present && !healthy {
                    self.metrics
                        .corruption_detected
                        .fetch_add(1, Ordering::Relaxed);
                }
                (present, healthy)
            }
            Ok(_) | Err(_) => (false, false),
        }
    }

    /// Whether every published fragment currently probes healthy.
    async fn probe_healthy(&self, layout: &RedundancyLayout) -> bool {
        for record in &layout.fragments {
            let key = FragmentKey::from_asset(layout.asset, layout.generation, record.id.index);
            let (_, healthy) = self.has_one(record.node, &key).await;
            if !healthy {
                return false;
            }
        }
        true
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

    /// Sends one RPC, refreshing a stale incarnation and retrying once.
    /// Other refusals are returned for the caller to match on.
    async fn call_with_refresh(
        &self,
        node: NodeId,
        rpc: FragmentRpc,
    ) -> Result<FragmentReply, RedundancyError> {
        let mut incarnation = self.peer_incarnation(node);
        let mut outgoing = rpc.clone();
        set_rpc_incarnation(&mut outgoing, incarnation);
        let reply = self
            .transport
            .call(
                PeerTarget { node, incarnation },
                outgoing,
                self.config.rpc_timeout,
            )
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
            .call(
                PeerTarget { node, incarnation },
                retry,
                self.config.rpc_timeout,
            )
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
        FragmentRpc::Put {
            target_incarnation, ..
        }
        | FragmentRpc::Get {
            target_incarnation, ..
        }
        | FragmentRpc::Has {
            target_incarnation, ..
        }
        | FragmentRpc::Drop {
            target_incarnation, ..
        }
        | FragmentRpc::PutLayout {
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
    let required = params.required_pieces() as usize;
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
}
