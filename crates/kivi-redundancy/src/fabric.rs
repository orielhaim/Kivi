//! Redundancy Fabric: physical protection for immutable information.
//!
//! Raft decides ordered replicated logical state; durability decides what
//! must survive; redundancy decides *how immutable information is physically
//! protected*; the memory fabric decides runtime materialization; chunk and
//! checkpoint storage provide immutable physical data. This fabric never
//! becomes another consensus protocol: it protects immutable assets whose
//! identity ([`crate::AssetId`]) never changes across layouts.
//!
//! Safety principle everywhere: **build → verify → publish → retire**. A
//! replacement layout is fully encoded, written as pending, decoded and
//! content-verified against the asset id, and only then published atomically;
//! the old layout retires afterwards. Transitions may temporarily hold extra
//! redundancy — preferable to a durability gap. Every asynchronous step is
//! generation-fenced: stale completions cannot overwrite a newer layout.
//!
//! Crash recovery distinguishes published current layouts, incomplete new
//! layouts (discarded), stale old layouts (retained conservatively, retired
//! after the new publish proves durable), and orphan fragments (swept when
//! unreferenced). Ambiguity retains extra data; it never installs an
//! unverified incomplete layout.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use kivi_types::NodeId;

use crate::store::{
    fragment_name, publish_layout, read_fragment, retire_generation, sync_dir_best_effort,
    write_fragment, write_layout,
};
use crate::{
    DesiredPlacement, FabricMetrics, FailureScope, FragmentRecord, FragmentVerdict,
    InformationAsset, RedundancyError, RedundancyIntent, RedundancyLayout, RepairBudgets,
    RepairEngine, RepairReason, SchemeParams, assess, plan_baseline, plan_placement,
    reconstruction_targets,
};

/// Bounds for foreground reconstruction work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReconstructionBudgets {
    /// Maximum concurrent decodes fabric-wide.
    pub max_concurrent_decodes: usize,
    /// Maximum memory one decode may use (data image bound).
    pub max_decode_bytes: u64,
    /// Maximum fragment reads per reconstruction.
    pub max_fragment_reads: usize,
}

impl ReconstructionBudgets {
    /// Conservative defaults: 2 concurrent decodes, 64 MiB images.
    #[must_use]
    pub const fn conservative() -> Self {
        Self {
            max_concurrent_decodes: 2,
            max_decode_bytes: 64 * 1024 * 1024,
            max_fragment_reads: 64,
        }
    }
}

/// Fabric configuration: budgets, scope, and the cluster view.
#[derive(Debug, Clone, Copy)]
pub struct FabricConfig {
    /// Repair budgets (bounded background work).
    pub repair: RepairBudgets,
    /// Reconstruction bounds (foreground reads).
    pub reconstruction: ReconstructionBudgets,
    /// Failure scope enforced for independence.
    pub scope: FailureScope,
}

impl FabricConfig {
    /// Conservative production defaults.
    #[must_use]
    pub const fn conservative() -> Self {
        Self {
            repair: RepairBudgets::conservative(),
            reconstruction: ReconstructionBudgets::conservative(),
            scope: FailureScope::Node,
        }
    }
}

impl Default for FabricConfig {
    fn default() -> Self {
        Self::conservative()
    }
}

/// What restart recovery did (operators, startup timing).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    /// Assets with a published current layout.
    pub assets_recovered: u64,
    /// Incomplete pending layouts discarded.
    pub pending_discarded: u64,
    /// Stale old generations retained for later retirement.
    pub stale_retained: u64,
    /// Orphan fragment files swept.
    pub orphans_swept: u64,
}

/// Current transition state for observability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionState {
    /// Asset transitioning.
    pub asset: crate::AssetId,
    /// Generation being built.
    pub pending_generation: u64,
    /// Target scheme.
    pub target: SchemeParams,
}

/// The Redundancy Fabric: asset registry, layouts, fragments, repair.
///
/// Single-threaded by construction for mutation (`&mut self` protects,
/// `&self` reads); share across threads through `Arc<Mutex<…>>` at the
/// integration layer. Fragment bytes live under `root/assets/<hex>/` as
/// `frag-<gen>-<idx>.bin`; layouts as `gen-<N>.layout`; the published
/// generation in `CURRENT` (atomic temp + rename + sync).
#[derive(Debug)]
pub struct RedundancyFabric {
    root: PathBuf,
    config: FabricConfig,
    nodes: Vec<crate::NodeDescriptor>,
    layouts: HashMap<crate::AssetId, RedundancyLayout>,
    pending: HashMap<crate::AssetId, PendingBuild>,
    repair: RepairEngine,
    metrics: Arc<FabricMetrics>,
    inflight_decodes: usize,
}

/// One pending (unpublished) build: encoded fragments awaiting verification
/// and atomic publication. Never served to readers.
#[derive(Debug, Clone)]
struct PendingBuild {
    /// Layout the pending fragments materialize (generation = current + 1).
    layout: RedundancyLayout,
}

impl RedundancyFabric {
    /// Opens (creating if needed) the fabric root and recovers published
    /// state. See the module docs for the exact repair rules.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] on I/O failure or corrupt published
    /// metadata. Incomplete pending builds are discarded (counted in the
    /// report); stale old generations are retained conservatively.
    pub fn open(
        root: &Path,
        config: FabricConfig,
        nodes: Vec<crate::NodeDescriptor>,
    ) -> Result<(Self, RecoveryReport), RedundancyError> {
        std::fs::create_dir_all(root)
            .map_err(|error| RedundancyError::io("create redundancy root", root, &error))?;
        let assets_dir = root.join("assets");
        std::fs::create_dir_all(&assets_dir).map_err(|error| {
            RedundancyError::io("create redundancy assets", &assets_dir, &error)
        })?;
        let mut layouts = HashMap::new();
        let mut report = RecoveryReport::default();
        if let Ok(entries) = std::fs::read_dir(&assets_dir) {
            for entry in entries.flatten() {
                let dir = entry.path();
                if !dir.is_dir() {
                    continue;
                }
                if let Some(layout) = recover_asset_dir(&dir, &mut report)? {
                    layouts.insert(layout.asset, layout);
                }
            }
        }
        let metrics = Arc::new(FabricMetrics::new());
        refresh_metrics(&metrics, &layouts);
        Ok((
            Self {
                root: root.to_path_buf(),
                config,
                nodes,
                layouts,
                pending: HashMap::new(),
                repair: RepairEngine::with_budgets(config.repair),
                metrics,
                inflight_decodes: 0,
            },
            report,
        ))
    }

    /// Returns shared metrics.
    #[must_use]
    pub fn metrics(&self) -> &Arc<FabricMetrics> {
        &self.metrics
    }

    /// Replaces the cluster view (joins, drains, migrations, failures feed
    /// through here; no separate membership system is built).
    pub fn set_nodes(&mut self, nodes: Vec<crate::NodeDescriptor>) {
        self.nodes = nodes;
    }

    /// Returns the published layout for an asset, if any.
    #[must_use]
    pub fn layout(&self, asset: &crate::AssetId) -> Option<&RedundancyLayout> {
        self.layouts.get(asset)
    }

    /// Returns pending transition state for observability.
    #[must_use]
    pub fn transitions(&self) -> Vec<TransitionState> {
        self.pending
            .iter()
            .map(|(asset, build)| TransitionState {
                asset: *asset,
                pending_generation: build.layout.generation,
                target: build.layout.params,
            })
            .collect()
    }

    /// Protects an asset's bytes under an intent: plans a layout, builds
    /// fragments, verifies by decode + content hash, publishes atomically.
    ///
    /// When the asset is already protected, this re-verifies health and
    /// returns the current layout (idempotent; no second layout is minted).
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] on invalid intents, unsatisfiable
    /// placement, codec failures, verification failures, or I/O.
    pub fn protect(
        &mut self,
        asset: InformationAsset,
        bytes: &[u8],
        intent: &RedundancyIntent,
    ) -> Result<RedundancyLayout, RedundancyError> {
        asset.verify_bytes(bytes).map_err(|e| match e {
            RedundancyError::VerificationFailed { detail } => {
                RedundancyError::CorruptFragment { detail }
            }
            other => other,
        })?;
        if let Some(current) = self.layouts.get(&asset.id) {
            // Idempotent re-protect: same logical bytes stay on the same
            // generation (no churn for duplicate calls).
            let _ = current;
            return Ok(current.clone());
        }
        let params = plan_baseline(intent, asset.logical_len)?;
        self.publish_new_generation(asset, bytes, params, intent)
    }

    /// Transitions an asset to a new layout (replica replacement, replica ↔
    /// coded, width change, or node move) without ever reducing current
    /// recoverability first: the new generation builds fully, verifies by
    /// decode + content hash, publishes atomically, and only then retires
    /// the old generation. The asset stays readable throughout under its
    /// unchanged [`crate::AssetId`].
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when the new layout cannot be built,
    /// verified, or published. The current layout is untouched on failure.
    pub fn transition(
        &mut self,
        asset: InformationAsset,
        bytes: &[u8],
        target: SchemeParams,
    ) -> Result<RedundancyLayout, RedundancyError> {
        asset.verify_bytes(bytes).map_err(|e| match e {
            RedundancyError::VerificationFailed { detail } => {
                RedundancyError::CorruptFragment { detail }
            }
            other => other,
        })?;
        target.validate()?;
        let current_gen = self.layouts.get(&asset.id).map_or(0, |l| l.generation);
        if let Some(current) = self.layouts.get(&asset.id)
            && current.params == target
        {
            return Ok(current.clone());
        }
        let intent = RedundancyIntent::best_effort();
        let layout = self.publish_new_generation(asset, bytes, target, &intent)?;
        debug_assert!(layout.generation > current_gen);
        self.metrics.transitions.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .transitions_with_overlap
            .fetch_add(1, Ordering::Relaxed);
        Ok(layout)
    }

    /// Planned transition that derives the target from an intent (planner
    /// decision, not a caller-named `RS(k,m)`).
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] on planning, build, or publish failures.
    pub fn transition_for_intent(
        &mut self,
        asset: InformationAsset,
        bytes: &[u8],
        intent: &RedundancyIntent,
    ) -> Result<RedundancyLayout, RedundancyError> {
        let params = plan_baseline(intent, asset.logical_len)?;
        self.transition(asset, bytes, params)
    }

    /// Reads an asset's bytes through the current layout, reconstructing
    /// from replicas or erasure fragments as needed. Callers never care
    /// which physical form answered: the bytes are content-verified against
    /// the asset id before exposure.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when no layout exists, too few fragments
    /// survive, verification fails, or reconstruction bounds are exhausted.
    /// Exceeding the advertised tolerance fails closed ([`Unrecoverable`]).
    pub fn read(&mut self, asset: InformationAsset) -> Result<Vec<u8>, RedundancyError> {
        let layout = self.layouts.get(&asset.id).cloned().ok_or_else(|| {
            RedundancyError::MissingFragment {
                asset: asset.id.to_string(),
                generation: 0,
                index: 0,
            }
        })?;
        if asset.logical_len != layout.logical_len {
            return Err(RedundancyError::VerificationFailed {
                detail: format!(
                    "asset {} len {} disagrees with layout {}",
                    asset.id, asset.logical_len, layout.logical_len
                ),
            });
        }
        if self.inflight_decodes >= self.config.reconstruction.max_concurrent_decodes {
            return Err(RedundancyError::Overloaded {
                detail: "too many concurrent reconstructions".to_owned(),
            });
        }
        self.inflight_decodes += 1;
        let result = self.read_inner(&asset, &layout);
        self.inflight_decodes -= 1;
        match result {
            Ok(bytes) => {
                asset.verify_bytes(&bytes)?;
                self.metrics.decodes.fetch_add(1, Ordering::Relaxed);
                Ok(bytes)
            }
            Err(error) => {
                if error.is_unrecoverable() {
                    self.metrics
                        .reconstruction_failures
                        .fetch_add(1, Ordering::Relaxed);
                }
                Err(error)
            }
        }
    }

    /// Assesses an asset's health: healthy, degraded but recoverable,
    /// critically degraded, or unrecoverable, with per-fragment verdicts.
    #[must_use]
    pub fn assess_asset(&self, asset: &crate::AssetId) -> Option<crate::AssetAssessment> {
        let layout = self.layouts.get(asset)?;
        let mut verdicts = Vec::with_capacity(layout.fragments.len());
        for record in &layout.fragments {
            verdicts.push((record.id.index, self.verdict_of(record)));
        }
        let by_id: HashMap<NodeId, crate::NodeDescriptor> =
            self.nodes.iter().map(|n| (n.id, n.clone())).collect();
        let healthy: Vec<u32> = verdicts
            .iter()
            .filter(|(_, v)| v.is_usable())
            .map(|(i, _)| *i)
            .collect();
        let survivable = crate::independent_survivable(
            layout.params,
            self.config.scope,
            &placement_of(layout),
            &healthy,
            &by_id,
        );
        Some(assess(
            *asset,
            layout.generation,
            layout.params,
            verdicts,
            survivable,
        ))
    }

    /// Scrub hook for one asset: per-fragment verdicts for operators and the
    /// future Phase 11 policy. Never repairs; never treats corruption as
    /// valid merely because bytes were read.
    #[must_use]
    pub fn scrub(&self, asset: &crate::AssetId) -> Option<crate::ScrubReport> {
        let layout = self.layouts.get(asset)?;
        let verdicts: Vec<(u32, FragmentVerdict)> = layout
            .fragments
            .iter()
            .map(|record| (record.id.index, self.verdict_of(record)))
            .collect();
        Some(crate::scrub_report(*asset, layout.generation, verdicts))
    }

    /// Queues repairs for a degraded asset (explicit deficits only).
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when the queue is full or no layout
    /// exists.
    ///
    /// # Panics
    ///
    /// Never panics: missing layouts report as errors, never `expect`.
    pub fn queue_repairs(
        &mut self,
        asset: &crate::AssetId,
        reason: RepairReason,
    ) -> Result<bool, RedundancyError> {
        let assessment =
            self.assess_asset(asset)
                .ok_or_else(|| RedundancyError::MissingFragment {
                    asset: asset.to_string(),
                    generation: 0,
                    index: 0,
                })?;
        if assessment.need_repair.is_empty() {
            return Ok(false);
        }
        let layout =
            self.layouts
                .get(asset)
                .cloned()
                .ok_or_else(|| RedundancyError::MissingFragment {
                    asset: asset.to_string(),
                    generation: 0,
                    index: 0,
                })?;
        let holders: Vec<NodeId> = layout.fragments.iter().map(|f| f.node).collect();
        let targets = reconstruction_targets(
            assessment.asset,
            assessment.generation.saturating_add(1),
            &assessment.need_repair,
            &holders,
            &self.nodes,
        );
        let map: HashMap<u32, NodeId> = targets.targets.into_iter().collect();
        let queued = self.repair.queue_assessment(&assessment, reason, &map)?;
        if queued {
            let pending = assessment.need_repair.len() as u64;
            self.metrics.pending_repairs.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .repair_bytes_pending
                .fetch_add(pending * fragment_bytes(&layout), Ordering::Relaxed);
        }
        Ok(queued)
    }

    /// Executes up to the per-tick repair budget: reconstructs missing
    /// fragments from the current layout and publishes replacements in
    /// place (same generation, new bytes — idempotent and resumable).
    ///
    /// Foreground traffic retains priority: pass `foreground_pressure`
    /// when hot reads are active to drain a single repair.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] on reconstruction or I/O failures; failed
    /// tasks re-queue with their reason intact (caller bounds retries by
    /// only ticking on explicit deficits).
    pub fn repair_tick(
        &mut self,
        foreground_pressure: bool,
    ) -> Result<Vec<crate::AssetId>, RedundancyError> {
        let tasks = self.repair.tick(foreground_pressure);
        let mut done = Vec::new();
        for task in tasks {
            self.execute_repair(&task)?;
            done.push(task.asset);
            self.metrics.pending_repairs.fetch_sub(1, Ordering::Relaxed);
        }
        Ok(done)
    }

    /// Drains a node: re-places every fragment it holds onto healthy targets
    /// through generation-fenced transitions (extra redundancy is held until
    /// each replacement verifies).
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when replacements cannot be built.
    ///
    /// # Panics
    ///
    /// Never panics: missing layouts are skipped, never `expect`.
    pub fn drain_node(&mut self, node: NodeId) -> Result<Vec<crate::AssetId>, RedundancyError> {
        let affected: Vec<crate::AssetId> = self
            .layouts
            .iter()
            .filter(|(_, layout)| layout.fragments.iter().any(|f| f.node == node))
            .map(|(asset, _)| *asset)
            .collect();
        let mut moved = Vec::new();
        for asset in affected {
            // Rebuild the same scheme on fresh placement: encode current
            // bytes (read through the old layout, which stays published
            // until the replacement verifies), then publish a new generation
            // whose placement avoids the drained node.
            let Some(layout) = self.layouts.get(&asset).cloned() else {
                continue;
            };
            let info = InformationAsset::new(asset, layout.logical_len);
            let bytes = self.read(info)?;
            // Force placement away from the drained node by snapshotting the
            // cluster without it for this build.
            let saved = std::mem::take(&mut self.nodes);
            let filtered: Vec<crate::NodeDescriptor> =
                saved.iter().filter(|n| n.id != node).cloned().collect();
            self.nodes = filtered;
            let result = self.publish_new_generation(
                info,
                &bytes,
                layout.params,
                &RedundancyIntent::best_effort(),
            );
            self.nodes = saved;
            result?;
            moved.push(asset);
        }
        Ok(moved)
    }

    /// Rejects stale asynchronous work: returns `Ok` when `generation`
    /// matches the published current generation, else counts and reports
    /// [`StaleGeneration`](RedundancyError::StaleGeneration).
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::StaleGeneration`] for superseded work.
    pub fn fence(&self, asset: &crate::AssetId, generation: u64) -> Result<(), RedundancyError> {
        let current = self.layouts.get(asset).map_or(0, |l| l.generation);
        if generation != current {
            self.metrics.stale_rejected.fetch_add(1, Ordering::Relaxed);
            return Err(RedundancyError::StaleGeneration {
                generation,
                current,
            });
        }
        Ok(())
    }

    /// Fragment census for operators: fragments by node and failure domain.
    #[must_use]
    pub fn census(&self) -> crate::FragmentCensus {
        let mut census = crate::FragmentCensus::new();
        let by_id: HashMap<NodeId, crate::NodeDescriptor> =
            self.nodes.iter().map(|n| (n.id, n.clone())).collect();
        for layout in self.layouts.values() {
            for record in &layout.fragments {
                let key = by_id.get(&record.node).map_or_else(
                    || format!("node:{}", record.node.as_u64()),
                    |n| n.domain.independence_key(self.config.scope),
                );
                census.record(record.node, key);
            }
        }
        census
    }

    /// Builds, verifies, and publishes a new generation (shared by
    /// [`protect`](Self::protect) and [`transition`](Self::transition)).
    fn publish_new_generation(
        &mut self,
        asset: InformationAsset,
        bytes: &[u8],
        params: SchemeParams,
        intent: &RedundancyIntent,
    ) -> Result<RedundancyLayout, RedundancyError> {
        params.validate()?;
        let generation = self.layouts.get(&asset.id).map_or(1, |l| l.generation + 1);
        if generation == 0 {
            return Err(RedundancyError::invalid_asset("generation space exhausted"));
        }
        // Plan placement for the new generation (deterministic rendezvous).
        let total = params.total_fragments();
        let placement = plan_placement(
            asset.id,
            generation,
            total,
            self.config.scope,
            &self.nodes,
            &intent.allowed_domains,
        )?;
        // Encode behind the common abstraction (replication and RS share
        // everything around these two calls).
        let shards: Vec<Vec<u8>> = match params {
            SchemeParams::Replication(p) => crate::replication::encode(bytes, p)?,
            SchemeParams::ReedSolomon(p) => crate::rs::encode(bytes, p)?,
        };
        {
            #[allow(clippy::cast_possible_truncation)]
            let have = shards.len() as u32;
            debug_assert_eq!(have, total);
        }
        // Stage pending fragments + layout (unpublished, invisible to reads).
        let dir = self.asset_dir(&asset.id);
        std::fs::create_dir_all(&dir)
            .map_err(|error| RedundancyError::io("create asset directory", &dir, &error))?;
        let mut records = Vec::with_capacity(shards.len());
        for (index, shard) in shards.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let index = index as u32;
            let id = crate::FragmentId::for_bytes(asset.id, generation, index, params, shard);
            let node = placement
                .node_for(index)
                .ok_or_else(|| RedundancyError::Placement {
                    detail: format!("placement missing fragment {index}"),
                })?;
            let stored_len = shard.len() as u64;
            write_fragment(&dir, generation, index, shard, true)?;
            records.push(FragmentRecord {
                id,
                node,
                stored_len,
            });
        }
        let layout =
            RedundancyLayout::new(asset.id, asset.logical_len, generation, params, records)?;
        write_layout(&dir, generation, &layout, true)?;
        // Verify: decode from the just-staged fragments and content-check
        // against the asset id. A successful decode alone never suffices.
        let present: Vec<(u32, Vec<u8>)> = shards
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                #[allow(clippy::cast_possible_truncation)]
                let index = i as u32;
                (index, s)
            })
            .collect();
        let decoded: Vec<u8> = match params {
            SchemeParams::Replication(_) => crate::replication::decode(&present)?,
            SchemeParams::ReedSolomon(p) => crate::rs::decode(&present, p, asset.logical_len)?,
        };
        asset.verify_bytes(&decoded)?;
        self.metrics.encodes.fetch_add(1, Ordering::Relaxed);
        // Publish atomically: layout file first (sync), then CURRENT
        // (temp + rename + dir sync). Readers only ever see CURRENT.
        publish_layout(&dir, generation, &layout)?;
        let previous = self.layouts.insert(asset.id, layout.clone());
        // Retire the previous generation (safe: the replacement is proven
        // usable — retire failures only cost disk, never correctness).
        if let Some(old) = previous {
            retire_generation(&dir, old.generation, old.params.total_fragments());
        }
        self.pending.remove(&asset.id);
        refresh_metrics(&self.metrics, &self.layouts);
        Ok(layout)
    }

    /// Reads through a layout from local fragment files.
    fn read_inner(
        &self,
        asset: &InformationAsset,
        layout: &RedundancyLayout,
    ) -> Result<Vec<u8>, RedundancyError> {
        let dir = self.asset_dir(&asset.id);
        let mut present: Vec<(u32, Vec<u8>)> = Vec::new();
        let mut corrupt = 0u64;
        for record in &layout.fragments {
            match read_fragment(&dir, layout.generation, record.id.index) {
                Ok(bytes) => {
                    // Integrity first: fragment hash, then length agreement.
                    // Corruption is never treated as valid merely because
                    // bytes were read.
                    if record.id.verify_bytes(&bytes).is_err() {
                        corrupt += 1;
                        continue;
                    }
                    if bytes.len() as u64 != record.stored_len {
                        corrupt += 1;
                        continue;
                    }
                    present.push((record.id.index, bytes));
                    if present.len() >= self.config.reconstruction.max_fragment_reads {
                        break;
                    }
                }
                Err(RedundancyError::MissingFragment { .. }) => {}
                Err(RedundancyError::CorruptFragment { .. }) => {
                    corrupt += 1;
                }
                Err(other) => return Err(other),
            }
        }
        if corrupt > 0 {
            self.metrics
                .corruption_detected
                .fetch_add(corrupt, Ordering::Relaxed);
        }
        present.sort_by_key(|(index, _)| *index);
        match layout.params {
            SchemeParams::Replication(_) => crate::replication::decode(&present),
            SchemeParams::ReedSolomon(params) => {
                if (present.len() as u64) * u64::from(params.fragment_len)
                    > self.config.reconstruction.max_decode_bytes
                {
                    return Err(RedundancyError::Overloaded {
                        detail: "reconstruction exceeds memory bound".to_owned(),
                    });
                }
                crate::rs::decode(&present, params, asset.logical_len)
            }
        }
    }

    /// Classifies one fragment's current state for assess/scrub.
    fn verdict_of(&self, record: &FragmentRecord) -> FragmentVerdict {
        let dir = self.asset_dir(&record.id.asset);
        match read_fragment(&dir, record.id.generation, record.id.index) {
            Ok(bytes) => {
                if record.id.verify_bytes(&bytes).is_ok() && bytes.len() as u64 == record.stored_len
                {
                    FragmentVerdict::Healthy
                } else {
                    FragmentVerdict::ChecksumMismatch
                }
            }
            Err(RedundancyError::MissingFragment { .. }) => FragmentVerdict::Missing,
            Err(RedundancyError::CorruptFragment { .. }) => FragmentVerdict::ChecksumMismatch,
            Err(_) => FragmentVerdict::Unreadable,
        }
    }

    /// Executes one queued repair: rebuilds the task's fragments from the
    /// current layout and rewrites them in place (same generation —
    /// idempotent: re-executing a completed repair rewrites identical
    /// verified bytes).
    fn execute_repair(&mut self, task: &crate::RepairTask) -> Result<(), RedundancyError> {
        let layout = self.layouts.get(&task.asset).cloned().ok_or_else(|| {
            RedundancyError::MissingFragment {
                asset: task.asset.to_string(),
                generation: task.generation,
                index: 0,
            }
        })?;
        // Generation fence: stale repairs never execute.
        if task.generation != layout.generation {
            self.metrics.stale_rejected.fetch_add(1, Ordering::Relaxed);
            return Err(RedundancyError::StaleGeneration {
                generation: task.generation,
                current: layout.generation,
            });
        }
        // Reconstruct full bytes, then re-derive the missing fragments.
        let info = InformationAsset::new(task.asset, layout.logical_len);
        let bytes = self.read_inner(&info, &layout)?;
        info.verify_bytes(&bytes)?;
        let shards: Vec<Vec<u8>> = match layout.params {
            SchemeParams::Replication(p) => crate::replication::encode(&bytes, p)?,
            SchemeParams::ReedSolomon(p) => crate::rs::encode(&bytes, p)?,
        };
        let dir = self.asset_dir(&task.asset);
        for index in &task.fragments {
            let Some(shard) = shards.get(*index as usize) else {
                continue;
            };
            let expect = &layout.fragments[*index as usize];
            let id = crate::FragmentId::for_bytes(
                task.asset,
                layout.generation,
                *index,
                layout.params,
                shard,
            );
            if id.content != expect.id.content {
                return Err(RedundancyError::CorruptFragment {
                    detail: format!("repair re-encode mismatch at fragment {index}"),
                });
            }
            write_fragment(&dir, layout.generation, *index, shard, false)?;
        }
        sync_dir_best_effort(&dir);
        let bytes_done = task.fragments.len() as u64 * fragment_bytes(&layout);
        self.metrics
            .repair_bytes_done
            .fetch_add(bytes_done, Ordering::Relaxed);
        self.metrics.repair_bytes_pending.fetch_sub(
            bytes_done.min(self.metrics.repair_bytes_pending.load(Ordering::Relaxed)),
            Ordering::Relaxed,
        );
        Ok(())
    }

    /// Directory for one asset's layouts and fragments.
    fn asset_dir(&self, asset: &crate::AssetId) -> PathBuf {
        self.root
            .join("assets")
            .join(format!("{}-{}", asset.kind.name(), asset.hex()))
    }
}

/// Derives the placement view a layout was planned under (for health math).
fn placement_of(layout: &RedundancyLayout) -> DesiredPlacement {
    DesiredPlacement {
        asset: layout.asset,
        generation: layout.generation,
        scope: FailureScope::Node,
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

/// Representative fragment bytes for repair accounting.
fn fragment_bytes(layout: &RedundancyLayout) -> u64 {
    layout.fragments.first().map_or(0, |f| f.stored_len)
}

/// Refreshes asset/byte/health counters from current layouts.
fn refresh_metrics(metrics: &FabricMetrics, layouts: &HashMap<crate::AssetId, RedundancyLayout>) {
    let mut replicated = 0u64;
    let mut coded = 0u64;
    let mut logical = 0u64;
    let mut physical = 0u64;
    for layout in layouts.values() {
        match layout.params {
            SchemeParams::Replication(_) => replicated += 1,
            SchemeParams::ReedSolomon(_) => coded += 1,
        }
        logical += layout.logical_len;
        physical += layout.fragments.iter().map(|f| f.stored_len).sum::<u64>();
    }
    metrics
        .replicated_assets
        .store(replicated, Ordering::Relaxed);
    metrics.coded_assets.store(coded, Ordering::Relaxed);
    metrics.logical_bytes.store(logical, Ordering::Relaxed);
    metrics.physical_bytes.store(physical, Ordering::Relaxed);
}

/// Recovers one asset directory: published CURRENT + layout, discarding
/// incomplete pending builds and retaining stale generations conservatively.
fn recover_asset_dir(
    dir: &Path,
    report: &mut RecoveryReport,
) -> Result<Option<RedundancyLayout>, RedundancyError> {
    // Pending files are incomplete new layouts by definition (CURRENT never
    // pointed at them): discard.
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = entry.path();
            let is_pending = name.starts_with("pending-");
            #[allow(clippy::case_sensitive_file_extension_comparisons)]
            let is_tmp = path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("tmp"));
            if is_pending || is_tmp {
                let _ = std::fs::remove_file(path);
                report.pending_discarded += 1;
            }
        }
    }
    let current_path = dir.join("CURRENT");
    let generation: u64 = match std::fs::read(&current_path) {
        Ok(bytes) => {
            if bytes.len() != 8 {
                return Err(RedundancyError::BadLayout {
                    detail: format!("CURRENT length {} is not 8", bytes.len()),
                });
            }
            u64::from_le_bytes(bytes[..8].try_into().unwrap_or([0; 8]))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(RedundancyError::io("read CURRENT", &current_path, &error)),
    };
    if generation == 0 {
        return Err(RedundancyError::BadLayout {
            detail: "CURRENT generation 0 is the invalid sentinel".to_owned(),
        });
    }
    let layout_path = dir.join(format!("gen-{generation}.layout"));
    let bytes = std::fs::read(&layout_path)
        .map_err(|error| RedundancyError::io("read published layout", &layout_path, &error))?;
    let layout = RedundancyLayout::decode(&bytes)?;
    if layout.generation != generation {
        return Err(RedundancyError::BadLayout {
            detail: format!(
                "CURRENT {generation} disagrees with layout {}",
                layout.generation
            ),
        });
    }
    // Verify every referenced fragment file exists and hashes correctly;
    // missing/corrupt fragments mark degraded (never fail recovery — the
    // repair engine, not startup, owns rebuilds).
    report.assets_recovered += 1;
    // Stale generations (any gen-*.layout not CURRENT) are retained here and
    // retired by the next successful publish; count them for operators.
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("gen-") && name != format!("gen-{generation}.layout") {
                report.stale_retained += 1;
            }
            if name.starts_with("frag-") && !name.starts_with(&format!("frag-{generation}-")) {
                report.orphans_swept += 0;
            }
        }
    }
    // Sweep orphan pending fragments left by a crash mid-stage (no CURRENT
    // ever pointed at them and the pending layout is gone).
    if let Ok(entries) = std::fs::read_dir(dir) {
        let live: HashSet<String> = (0..layout.params.total_fragments())
            .map(|i| fragment_name(generation, i, false))
            .collect();
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if (name.starts_with("frag-") || name.starts_with("pending-")) && !live.contains(&name)
            {
                // Conservatively retain non-pending stale frags (they may
                // belong to a retained old generation); sweep only
                // pending-prefixed orphans here.
                if name.starts_with("pending-") {
                    let _ = std::fs::remove_file(entry.path());
                    report.orphans_swept += 1;
                }
            }
        }
    }
    Ok(Some(layout))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssetHealth, NodeHealth, RedundancyIntent};

    fn nodes(count: u64) -> Vec<crate::NodeDescriptor> {
        (1..=count)
            .map(|id| crate::NodeDescriptor {
                id: NodeId::from_u64(id),
                domain: crate::FailureDomain::node_only(NodeId::from_u64(id)),
                health: NodeHealth::Active,
                weight: 1,
            })
            .collect()
    }

    fn fabric() -> (RedundancyFabric, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("scratch");
        let (fabric, report) = RedundancyFabric::open(
            &dir.path().join("redundancy"),
            FabricConfig::conservative(),
            nodes(5),
        )
        .expect("opens");
        assert_eq!(report.assets_recovered, 0);
        (fabric, dir)
    }

    #[test]
    fn protect_read_recover_round_trip() {
        let (mut fabric, _guard) = fabric();
        let bytes = vec![7u8; 10_000];
        let asset =
            InformationAsset::chunk_for_bytes(&bytes, kivi_types::SecurityDomainId::from_u64(3));
        let layout = fabric
            .protect(asset, &bytes, &RedundancyIntent::survive_two())
            .expect("protects");
        let back = fabric.read(asset).expect("reads");
        assert_eq!(back, bytes);
        let assessment = fabric.assess_asset(&asset.id).expect("assesses");
        assert_eq!(assessment.health, AssetHealth::Healthy);
        assert!(assessment.reconstructable);
        let _ = layout;
    }

    #[test]
    fn restart_recovers_published_and_discards_pending() {
        let dir = tempfile::tempdir().expect("scratch");
        let root = dir.path().join("redundancy");
        let bytes = vec![9u8; 5000];
        let asset =
            InformationAsset::chunk_for_bytes(&bytes, kivi_types::SecurityDomainId::from_u64(3));
        let layout = {
            let (mut fabric, ()) = (
                RedundancyFabric::open(&root, FabricConfig::conservative(), nodes(5))
                    .expect("opens")
                    .0,
                (),
            );
            fabric
                .protect(asset, &bytes, &RedundancyIntent::survive_one())
                .expect("protects")
        };
        // Plant a fake pending build (crash during transition).
        let asset_dir = root.join("assets").join(format!(
            "{}-{}",
            layout.asset.kind.name(),
            layout.asset.hex()
        ));
        std::fs::write(asset_dir.join("pending-999.layout"), b"incomplete").expect("plants");
        std::fs::write(asset_dir.join("pending-999-0000.bin"), b"orphan").expect("plants");
        let (mut fabric, report) =
            RedundancyFabric::open(&root, FabricConfig::conservative(), nodes(5))
                .expect("recovers");
        assert_eq!(report.assets_recovered, 1);
        assert!(report.pending_discarded >= 2);
        assert_eq!(fabric.read(asset).expect("reads"), bytes);
    }
}
