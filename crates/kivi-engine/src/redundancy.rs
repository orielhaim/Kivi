//! Engine redundancy integration: immutable assets behind the fabric.
//!
//! [`EngineRedundancy`] is the engine's single handle to one
//! [`kivi_redundancy::RedundancyFabric`]. It protects immutable durable
//! assets only — staged chunks, chunk manifests, checkpoint bands, dedup
//! components, and checkpoint manifests — under the baseline
//! [`kivi_redundancy::RedundancyIntent::survive_one`] intent. Raft state,
//! WAL records, and every other mutable or consensus-ordered byte never go
//! through here: the fabric never becomes another consensus protocol, and
//! this layer never weakens the Raft durability gate. Local durability is
//! always proven first (chunk-lane sync barrier, checkpoint publish); the
//! fabric holds an additional verified copy of the same immutable bytes.
//!
//! The fabric itself enforces `build → verify → publish → retire` per
//! asset, so callers never interleave their own verification: protect, then
//! trust the published layout. Every synchronous method takes `&self` (the
//! mutex is locked internally) and never panics; fallible methods surface
//! failures as [`crate::engine::EngineError::Durability`] with an `Io`
//! payload naming the redundancy operation. No new
//! [`crate::engine::EngineError`] kinds are added by this module.
//!
//! Hot-path callers (workers, connection tasks, the checkpoint worker)
//! never touch the fabric directly: [`protect_staged_value`],
//! [`EngineRedundancy::submit_spliced_value`], and
//! [`EngineRedundancy::submit_checkpoint`] submit one fire-and-forget job
//! per staged value or published checkpoint onto the bounded background
//! [`kivi_redundancy::RedundancyLane`] opened in
//! [`EngineRedundancy::open`]. Submission never blocks the caller — the
//! [`kivi_redundancy::LaneJoin`] is dropped, so the job still runs and only
//! its reply is shed — and rejected submits warn without failing the commit
//! or checkpoint they follow: local durability already proved the bytes, so
//! a fabric miss only costs redundancy, never correctness. One job encodes
//! all chunks plus the manifest (or all checkpoint artifacts) to preserve
//! protection ordering; every job runs at
//! [`kivi_redundancy::LanePriority::Background`]. Lane outcomes are
//! observable through [`EngineRedundancy::lane_stats`] (surfaced on the
//! admin plane as [`LocalEngine::redundancy_lane_stats`](crate::engine::LocalEngine::redundancy_lane_stats)
//! and [`AdminHandle::redundancy_lane_stats`](crate::engine::AdminHandle::redundancy_lane_stats)).
//!
//! Cancellation is idempotency: re-protecting the same bytes returns the
//! current layout, so protects submit with no generation-fence hook. A
//! duplicate job is wasted work, never a wrong asset.
//!
//! Foreground pressure ([`EngineRedundancy::set_lane_pressure`]) parks
//! [`kivi_redundancy::LanePriority::Background`] jobs while set. The engine
//! holds it across the checkpoint build window (the known background-heavy
//! stretch; see the checkpoint worker's pressure guard). No worker hot-path
//! pressure signal exists yet — wiring one (commit backlog, hot-read debt)
//! is future work, and the flag is documented as not wired there.
//!
//! Raft durability semantics are unchanged: the local lane sync barrier
//! still gates commits before any lane job is submitted, and the fabric
//! never gates anything — it only ever adds a verified copy after the
//! local proof.
//!
//! Shutdown drains: dropping [`EngineRedundancy`] closes the lane intake,
//! and lane workers finish staged jobs before joining, so pending protects
//! complete instead of being shed. (The checkpoint worker joins before the
//! lane drops, and its pressure guard releases on unwind, so a parked lane
//! can never wedge shutdown.)
//!
//! Admin integration point: [`EngineRedundancy::admin_snapshot`] (surfaced
//! through [`crate::engine::LocalEngine::redundancy_snapshot`] and
//! [`crate::engine::AdminHandle::redundancy_snapshot`]) carries everything a
//! future `kivi-server` `/v1/redundancy` endpoint needs. The existing admin
//! DTOs are untouched (additive only).
//!
//! Coverage: worker channel staging (plain values, conditionals, splices —
//! splices re-read the resolved value on the lane thread, never the
//! worker), connection-task direct staging (plain and conditional `Set`s via
//! lane submits), and checkpoint publication (one lane job per published
//! checkpoint) are protected. Not yet covered: reactor bridge frontier
//! completions (the parked value is not retained at completion),
//! connection-task and bridge-parked splices (no caller value survives to
//! submit; the worker splice re-read pattern could extend here), and
//! streaming-upload finalizations — all need either value retention at
//! completion or a lane re-read.

use std::path::Path;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use kivi_redundancy::{
    AssetAssessment, AssetId, FabricConfig, FabricMetrics, FabricMetricsSnapshot, FragmentCensus,
    LaneConfig, LanePriority, LaneStats, LaneValue, NodeDescriptor, RecoveryReport,
    RedundancyError, RedundancyFabric, RedundancyIntent, RedundancyLane, RedundancyLayout,
    RepairReason,
};
use kivi_types::{ChunkId, ManifestId, NodeId, SecurityDomainId};

use crate::engine::EngineError;

/// Subdirectory of the data directory holding fabric layouts and fragments.
const FABRIC_DIR_NAME: &str = "redundancy";

/// Maps a fabric failure onto the engine error surface.
///
/// All fabric failures reuse [`EngineError::Durability`] with an `Io`
/// payload naming the redundancy operation; no new error kinds are added.
fn map_redundancy(op: &'static str, error: &RedundancyError) -> EngineError {
    EngineError::Durability(kivi_durability::DurabilityError::Io {
        op,
        message: error.to_string(),
        code: None,
    })
}

/// Maps a poisoned fabric mutex onto the engine error surface.
fn lock_error(op: &'static str) -> EngineError {
    EngineError::Durability(kivi_durability::DurabilityError::Io {
        op,
        message: "redundancy fabric lock poisoned".to_owned(),
        code: None,
    })
}

/// Saturating `usize` to `u64` for lane byte declarations: oversize values
/// declare the whole budget rather than truncating it.
fn bytes_u64(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

/// Engine handle to the redundancy fabric: one shared fabric plus the
/// baseline intent every protection uses, plus the bounded background lane
/// hot paths submit best-effort protects to.
///
/// Clonable through the inner [`Arc`]s: the engine, every worker's chunk
/// access, and the checkpoint worker hold clones of the same fabric and
/// lane. The baseline intent is [`RedundancyIntent::survive_one`]: the
/// minimum durable default. Only immutable assets are ever protected through
/// this handle; Raft and other mutable state never reach it.
#[derive(Debug, Clone)]
pub struct EngineRedundancy {
    /// Shared fabric; the mutex is locked per call (never held across
    /// lane or control round-trips).
    fabric: Arc<Mutex<RedundancyFabric>>,
    /// Baseline protection intent (survive one independent failure).
    intent: RedundancyIntent,
    /// Bounded background lane for best-effort protects. The mutex only
    /// guards submission and flag reads (never job execution, which runs on
    /// lane worker threads); it also keeps this handle `Send + Sync` even
    /// though lane thread joins are not.
    lane: Arc<Mutex<RedundancyLane>>,
}

impl EngineRedundancy {
    /// Opens (creating if needed) the fabric rooted at
    /// `<data_dir>/redundancy` with [`FabricConfig::conservative`] and
    /// recovers published layouts, plus the background lane with
    /// [`LaneConfig::conservative`] for best-effort protects.
    ///
    /// `nodes` is the cluster view the placement planner uses; single-node
    /// engines pass their own [`NodeDescriptor::active`] identity. The
    /// planner still places every fragment locally when no independent peer
    /// exists (collisions are recorded and discounted by health accounting),
    /// so single-node protection succeeds, just without independence.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Durability`] when the fabric root cannot be
    /// created, published metadata is corrupt, or the lane threads cannot
    /// spawn.
    pub fn open(
        data_dir: &Path,
        nodes: Vec<NodeDescriptor>,
    ) -> Result<(Self, RecoveryReport), EngineError> {
        let root = data_dir.join(FABRIC_DIR_NAME);
        let (fabric, report) = RedundancyFabric::open(&root, FabricConfig::conservative(), nodes)
            .map_err(|error| map_redundancy("open redundancy fabric", &error))?;
        let lane = RedundancyLane::open(LaneConfig::conservative())
            .map_err(|error| map_redundancy("open redundancy lane", &error))?;
        Ok((
            Self {
                fabric: Arc::new(Mutex::new(fabric)),
                intent: RedundancyIntent::survive_one(),
                lane: Arc::new(Mutex::new(lane)),
            },
            report,
        ))
    }

    /// Returns the baseline protection intent this handle protects under.
    #[must_use]
    pub const fn intent(&self) -> &RedundancyIntent {
        &self.intent
    }

    /// Protects staged chunk bytes in the fabric under the baseline intent.
    ///
    /// The caller stages and syncs through the chunk lane first; this only
    /// adds physical protection for the same verified bytes. Idempotent:
    /// re-protecting the same bytes returns the current layout.
    ///
    /// Synchronous and mutex-locked: lane jobs and admin-initiated repair
    /// call this, never foreground request paths (those submit through
    /// [`submit_staged_value`](Self::submit_staged_value) instead).
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Durability`] on verification, planning,
    /// placement, or publication failure.
    pub fn protect_chunk(
        &self,
        id: ChunkId,
        domain: SecurityDomainId,
        bytes: &[u8],
    ) -> Result<RedundancyLayout, EngineError> {
        let mut fabric = self
            .fabric
            .lock()
            .map_err(|_| lock_error("protect chunk"))?;
        kivi_chunk::protect_staged(&mut fabric, domain, id, bytes)
            .map_err(|error| map_redundancy("protect chunk", &error))
    }

    /// Protects staged canonical manifest bytes in the fabric.
    ///
    /// Same contract as [`protect_chunk`](Self::protect_chunk), for the
    /// canonical manifest encoding the lane proved durable. Synchronous and
    /// mutex-locked like [`protect_chunk`](Self::protect_chunk).
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Durability`] on verification, planning,
    /// placement, or publication failure.
    pub fn protect_manifest(
        &self,
        id: ManifestId,
        domain: SecurityDomainId,
        canonical: &[u8],
    ) -> Result<RedundancyLayout, EngineError> {
        let mut fabric = self
            .fabric
            .lock()
            .map_err(|_| lock_error("protect chunk manifest"))?;
        kivi_chunk::protect_manifest_staged(&mut fabric, domain, id, canonical)
            .map_err(|error| map_redundancy("protect chunk manifest", &error))
    }

    /// Protects whole checkpoint band file bytes in the fabric.
    ///
    /// `hash` is the BLAKE3 of the whole file bytes (the fabric identity),
    /// not the checkpoint content hash; it is computed by the caller with
    /// [`kivi_codec::integrity::blake3_256`]. Synchronous and mutex-locked
    /// like [`protect_chunk`](Self::protect_chunk).
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Durability`] on verification, planning,
    /// placement, or publication failure.
    pub fn protect_band(
        &self,
        hash: [u8; 32],
        bytes: &[u8],
    ) -> Result<RedundancyLayout, EngineError> {
        let mut fabric = self
            .fabric
            .lock()
            .map_err(|_| lock_error("protect checkpoint band"))?;
        kivi_checkpoint::protect_band(&mut fabric, hash, bytes)
            .map_err(|error| map_redundancy("protect checkpoint band", &error))
    }

    /// Protects whole checkpoint manifest file bytes in the fabric.
    ///
    /// Same whole-file identity discipline as
    /// [`protect_band`](Self::protect_band). Synchronous and mutex-locked
    /// like [`protect_chunk`](Self::protect_chunk).
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Durability`] on verification, planning,
    /// placement, or publication failure.
    pub fn protect_checkpoint_manifest(
        &self,
        hash: [u8; 32],
        bytes: &[u8],
    ) -> Result<RedundancyLayout, EngineError> {
        let mut fabric = self
            .fabric
            .lock()
            .map_err(|_| lock_error("protect checkpoint manifest"))?;
        kivi_checkpoint::protect_manifest(&mut fabric, hash, bytes)
            .map_err(|error| map_redundancy("protect checkpoint manifest", &error))
    }

    /// Protects whole checkpoint dedup-component file bytes in the fabric.
    ///
    /// Same whole-file identity discipline as
    /// [`protect_band`](Self::protect_band). Synchronous and mutex-locked
    /// like [`protect_chunk`](Self::protect_chunk).
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Durability`] on verification, planning,
    /// placement, or publication failure.
    pub fn protect_dedup(
        &self,
        hash: [u8; 32],
        bytes: &[u8],
    ) -> Result<RedundancyLayout, EngineError> {
        let mut fabric = self
            .fabric
            .lock()
            .map_err(|_| lock_error("protect checkpoint dedup"))?;
        kivi_checkpoint::protect_dedup(&mut fabric, hash, bytes)
            .map_err(|error| map_redundancy("protect checkpoint dedup", &error))
    }

    /// Submits best-effort protection for one staged value as a single lane
    /// job: every chunk sliced from `value` by the staged lengths, then the
    /// canonical manifest, protected in order on the lane thread.
    ///
    /// Never blocks the caller and never fails it: slicing is
    /// bounds-checked (a length mismatch warns and submits nothing), a
    /// rejected submit warns, and job-internal fabric failures warn from
    /// the lane thread. The join is dropped — the job still runs, only its
    /// reply is shed. `Bytes` slices share the allocation, so the job owns
    /// its bytes for one refcount bump per piece.
    pub fn submit_staged_value(
        &self,
        domain: SecurityDomainId,
        value: &Bytes,
        staged: &crate::chunk_lane::StagedValue,
    ) {
        let Some(pieces) = slice_pieces(value, staged) else {
            return;
        };
        let manifest = staged.manifest;
        let canonical = staged.canonical.clone();
        let declared = bytes_u64(value.len()).saturating_add(bytes_u64(canonical.len()));
        let worker = self.clone();
        submit_best_effort(self, "redundancy-staged-value", declared, move || {
            protect_pieces(&worker, domain, &pieces, manifest, &canonical);
            Ok(LaneValue::Empty)
        });
    }

    /// Submits best-effort protection for a spliced chunk sequence as a
    /// single lane job: the job re-reads the resolved value through `lane`
    /// on the lane thread (the caller retains no bytes to submit), then
    /// protects chunks plus the manifest exactly like
    /// [`submit_staged_value`](Self::submit_staged_value).
    ///
    /// Never blocks the caller and never fails it: a re-read miss warns and
    /// falls back to protecting the manifest from its canonical encoding,
    /// and submit/job failures warn like
    /// [`submit_staged_value`](Self::submit_staged_value).
    pub fn submit_spliced_value(
        &self,
        lane: crate::chunk_lane::ChunkLaneHandle,
        domain: SecurityDomainId,
        staged: &crate::chunk_lane::StagedValue,
    ) {
        let proof = staged.clone();
        let declared = bytes_u64(usize::try_from(staged.logical_len).unwrap_or(usize::MAX))
            .saturating_add(bytes_u64(staged.canonical.len()));
        let worker = self.clone();
        submit_best_effort(self, "redundancy-spliced-value", declared, move || {
            match lane.read_value_blocking(proof.manifest, proof.logical_len) {
                Ok(resolved) => {
                    if let Some(pieces) = slice_pieces(&resolved, &proof) {
                        protect_pieces(&worker, domain, &pieces, proof.manifest, &proof.canonical);
                    }
                }
                Err(error) => {
                    tracing::warn!(manifest = %proof.manifest, ?error, "redundancy splice re-read failed; protecting manifest only");
                    if let Err(error) =
                        worker.protect_manifest(proof.manifest, domain, &proof.canonical)
                    {
                        tracing::warn!(manifest = %proof.manifest, ?error, "redundancy manifest protection failed; local copy stands");
                    }
                }
            }
            Ok(LaneValue::Empty)
        });
    }

    /// Submits best-effort protection for one published checkpoint as a
    /// single lane job: every new band, then dedup, then manifest,
    /// protected in order on the lane thread.
    ///
    /// Never blocks the caller and never fails it: a rejected submit warns,
    /// and job-internal fabric failures warn per artifact from the lane
    /// thread. The join is dropped like every other best-effort submit.
    pub fn submit_checkpoint(&self, artifacts: CheckpointProtection) {
        if artifacts.is_empty() {
            return;
        }
        let declared = artifacts.declared_bytes();
        let worker = self.clone();
        submit_best_effort(self, "redundancy-checkpoint", declared, move || {
            for artifact in &artifacts.bands {
                if let Err(error) = worker.protect_band(artifact.hash, &artifact.bytes) {
                    tracing::warn!(
                        ?error,
                        "redundancy band protection failed; checkpoint copy stands"
                    );
                }
            }
            if let Some(dedup) = &artifacts.dedup
                && let Err(error) = worker.protect_dedup(dedup.hash, &dedup.bytes)
            {
                tracing::warn!(
                    ?error,
                    "redundancy dedup protection failed; checkpoint copy stands"
                );
            }
            if let Some(manifest) = &artifacts.manifest
                && let Err(error) =
                    worker.protect_checkpoint_manifest(manifest.hash, &manifest.bytes)
            {
                tracing::warn!(
                    ?error,
                    "redundancy manifest protection failed; checkpoint copy stands"
                );
            }
            Ok(LaneValue::Empty)
        });
    }

    /// Sets the lane's foreground-pressure flag: while set, only
    /// [`LanePriority::Foreground`] jobs run, so best-effort protects park.
    /// A poisoned lane mutex is a silent no-op (observability never breaks
    /// protection paths).
    pub fn set_lane_pressure(&self, pressure: bool) {
        if let Ok(lane) = self.lane.lock() {
            lane.set_foreground_pressure(pressure);
        }
    }

    /// Returns the lane's foreground-pressure flag (`false` on a poisoned
    /// lane mutex, never fails).
    #[must_use]
    pub fn lane_pressure(&self) -> bool {
        self.lane
            .lock()
            .is_ok_and(|lane| lane.foreground_pressure())
    }

    /// Snapshots the background lane counters for operators (lock-free
    /// atomics; a poisoned lane mutex reports zeros rather than blocking
    /// observability).
    #[must_use]
    pub fn lane_stats(&self) -> LaneStats {
        self.lane
            .lock()
            .map_or_else(|_| LaneStats::default(), |lane| lane.stats())
    }

    /// Reads chunk bytes back through the fabric, content-verified against
    /// `id`.
    ///
    /// Synchronous and mutex-locked like
    /// [`protect_chunk`](Self::protect_chunk): reads that must prove bytes
    /// stay on the caller, never the best-effort lane.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Durability`] when no layout exists,
    /// reconstruction fails, or the bytes do not verify.
    pub fn read_chunk(
        &self,
        id: ChunkId,
        domain: SecurityDomainId,
        len: u64,
    ) -> Result<Vec<u8>, EngineError> {
        let mut fabric = self.fabric.lock().map_err(|_| lock_error("read chunk"))?;
        kivi_chunk::read_via_fabric(&mut fabric, domain, id, len)
            .map_err(|error| map_redundancy("read chunk", &error))
    }

    /// Snapshots fabric counters for the admin plane (lock-free atomics;
    /// a poisoned mutex reports zeros rather than blocking observability).
    #[must_use]
    pub fn metrics_snapshot(&self) -> FabricMetricsSnapshot {
        self.fabric.lock().map_or_else(
            |_| FabricMetrics::default().snapshot(),
            |fabric| fabric.metrics().snapshot(),
        )
    }

    /// Assesses one asset's health (`None` when the asset is unknown or the
    /// mutex is poisoned; never fails).
    #[must_use]
    pub fn assess(&self, asset: &AssetId) -> Option<AssetAssessment> {
        self.fabric
            .lock()
            .ok()
            .and_then(|fabric| fabric.assess_asset(asset))
    }

    /// Fragments by node and failure domain for operators (empty on a
    /// poisoned mutex, never fails).
    #[must_use]
    pub fn census(&self) -> FragmentCensus {
        self.fabric
            .lock()
            .map_or_else(|_| FragmentCensus::new(), |fabric| fabric.census())
    }

    /// Queues repairs for a degraded asset (explicit deficits only).
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Durability`] when the queue is full or no
    /// layout exists.
    pub fn queue_repairs(
        &self,
        asset: &AssetId,
        reason: RepairReason,
    ) -> Result<bool, EngineError> {
        let mut fabric = self
            .fabric
            .lock()
            .map_err(|_| lock_error("queue repairs"))?;
        fabric
            .queue_repairs(asset, reason)
            .map_err(|error| map_redundancy("queue repairs", &error))
    }

    /// Executes up to the per-tick repair budget.
    ///
    /// Pass `foreground_pressure` when hot reads are active so a single
    /// repair drains first. Failed tasks re-queue with their reason intact.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Durability`] on reconstruction or I/O
    /// failures.
    pub fn repair_tick(&self, foreground_pressure: bool) -> Result<Vec<AssetId>, EngineError> {
        let mut fabric = self.fabric.lock().map_err(|_| lock_error("repair tick"))?;
        fabric
            .repair_tick(foreground_pressure)
            .map_err(|error| map_redundancy("repair tick", &error))
    }

    /// Drains a node: re-places every fragment it holds onto healthy targets
    /// through generation-fenced transitions.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Durability`] when replacements cannot be
    /// built.
    pub fn drain_node(&self, node: NodeId) -> Result<Vec<AssetId>, EngineError> {
        let mut fabric = self.fabric.lock().map_err(|_| lock_error("drain node"))?;
        fabric
            .drain_node(node)
            .map_err(|error| map_redundancy("drain node", &error))
    }

    /// Builds the admin-plane snapshot: metrics, live transition count,
    /// pending repairs, and census sizes.
    #[must_use]
    pub fn admin_snapshot(&self) -> RedundancyAdminSnapshot {
        let metrics = self.metrics_snapshot();
        let (transitions, census_fragments, census_nodes) =
            self.fabric.lock().map_or((0, 0, 0), |fabric| {
                let census = fabric.census();
                (
                    fabric.transitions().len(),
                    census.by_node.values().sum(),
                    census.by_node.len(),
                )
            });
        RedundancyAdminSnapshot {
            metrics,
            transitions,
            pending_repairs: metrics.pending_repairs,
            census_fragments,
            census_nodes,
        }
    }
}

/// One immutable checkpoint artifact's whole-file identity plus its bytes.
///
/// The hash is the BLAKE3 of the whole file bytes (the fabric identity),
/// not the checkpoint content hash.
#[derive(Debug, Clone)]
pub struct ProtectedArtifact {
    /// BLAKE3 of the whole file bytes (the fabric identity).
    pub hash: [u8; 32],
    /// Whole file bytes (owned; `Bytes` clones share the allocation).
    pub bytes: Bytes,
}

/// Owned checkpoint artifacts for a single lane protection job.
///
/// The checkpoint worker reads newly-written files into this set (warning
/// on unreadable files, skipping reused bands that were protected when
/// first written) and submits it whole so one job preserves the
/// band → dedup → manifest protection order.
#[derive(Debug, Clone, Default)]
pub struct CheckpointProtection {
    /// Newly-written band files (reused bands excluded by the caller).
    pub bands: Vec<ProtectedArtifact>,
    /// Dedup-component file (`None` when the caller could not read it).
    pub dedup: Option<ProtectedArtifact>,
    /// Manifest file (`None` when the caller could not read it).
    pub manifest: Option<ProtectedArtifact>,
}

impl CheckpointProtection {
    /// Whether the set holds nothing to protect (every file unreadable, or
    /// a checkpoint that reused everything): submitting it would only waste
    /// a lane job.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bands.is_empty() && self.dedup.is_none() && self.manifest.is_none()
    }

    /// Declared lane bytes for admission accounting: the owned artifact
    /// bytes, saturating instead of overflowing.
    #[must_use]
    pub fn declared_bytes(&self) -> u64 {
        let mut total = 0u64;
        for artifact in &self.bands {
            total = total.saturating_add(bytes_u64(artifact.bytes.len()));
        }
        if let Some(dedup) = &self.dedup {
            total = total.saturating_add(bytes_u64(dedup.bytes.len()));
        }
        if let Some(manifest) = &self.manifest {
            total = total.saturating_add(bytes_u64(manifest.bytes.len()));
        }
        total
    }
}

/// Point-in-time redundancy facts for the admin plane.
///
/// Carries the fabric [`FabricMetricsSnapshot`] plus the transition count,
/// pending repairs, and census sizes a future `kivi-server`
/// `/v1/redundancy` endpoint needs. Additive: existing admin DTOs are
/// untouched.
#[derive(Debug, Clone)]
pub struct RedundancyAdminSnapshot {
    /// Fabric counters (assets, bytes, health, repair activity).
    pub metrics: FabricMetricsSnapshot,
    /// Assets with a pending (unpublished) transition build.
    pub transitions: usize,
    /// Repair tasks queued (mirrors `metrics.pending_repairs`).
    pub pending_repairs: u64,
    /// Total fragments placed across nodes.
    pub census_fragments: u64,
    /// Nodes holding at least one fragment.
    pub census_nodes: usize,
}

/// Slices staged chunk pieces out of a resolved value, bounds-checked.
///
/// `Bytes::slice` shares the allocation, so pieces cost one refcount bump
/// each. A length overflow or a slice past the value end warns and yields
/// `None` without touching the commit outcome.
fn slice_pieces(
    value: &Bytes,
    staged: &crate::chunk_lane::StagedValue,
) -> Option<Vec<(ChunkId, Bytes)>> {
    let mut pieces = Vec::with_capacity(staged.chunks.len());
    let mut offset = 0usize;
    for (id, len) in &staged.chunks {
        let len = usize::try_from(*len).unwrap_or(usize::MAX);
        let Some(end) = offset.checked_add(len) else {
            tracing::warn!(
                chunk = %id,
                "redundancy chunk protection skipped: length overflows staging offset"
            );
            return None;
        };
        if end > value.len() {
            tracing::warn!(
                chunk = %id,
                offset,
                end,
                value_len = value.len(),
                "redundancy chunk protection skipped: staged lengths disagree with staged bytes"
            );
            return None;
        }
        // Bounds just proved (`offset <= end <= value.len()`), so the
        // slice cannot panic.
        pieces.push((*id, value.slice(offset..end)));
        offset = end;
    }
    Some(pieces)
}

/// Protects already-sliced pieces plus the manifest synchronously.
///
/// Runs on the lane thread (never the caller's): per-asset failures warn
/// and continue, so one bad piece never blocks the manifest behind it.
fn protect_pieces(
    worker: &EngineRedundancy,
    domain: SecurityDomainId,
    pieces: &[(ChunkId, Bytes)],
    manifest: ManifestId,
    canonical: &[u8],
) {
    for (id, piece) in pieces {
        if let Err(error) = worker.protect_chunk(*id, domain, piece) {
            tracing::warn!(chunk = %id, ?error, "redundancy chunk protection failed; local copy stands");
        }
    }
    if let Err(error) = worker.protect_manifest(manifest, domain, canonical) {
        tracing::warn!(manifest = %manifest, ?error, "redundancy manifest protection failed; local copy stands");
    }
}

/// Submits one best-effort job to the background lane, dropping the join.
///
/// The job still runs; only its reply is shed. A poisoned lane mutex or a
/// rejected submit warns without failing the caller: local durability
/// already proved the bytes, so a miss only costs redundancy.
fn submit_best_effort(
    worker: &EngineRedundancy,
    name: &'static str,
    bytes: u64,
    run: impl FnOnce() -> Result<LaneValue, RedundancyError> + Send + 'static,
) {
    let Ok(lane) = worker.lane.lock() else {
        tracing::warn!("redundancy {name} submit skipped: lane lock poisoned; local copy stands");
        return;
    };
    match lane.submit(name, bytes, LanePriority::Background, None, run) {
        Ok(_join) => {}
        Err(error) => {
            tracing::warn!(
                ?error,
                "redundancy {name} submit rejected; local copy stands"
            );
        }
    }
}

/// Best-effort protection for one staged value: submits a single lane job
/// encoding each chunk sliced from `value` by the staged lengths plus the
/// canonical manifest, preserving protection order.
///
/// Never blocks the caller and never fails it — see
/// [`EngineRedundancy::submit_staged_value`]. `None` redundancy (ephemeral
/// mode) is a silent no-op.
pub fn protect_staged_value(
    redundancy: Option<&EngineRedundancy>,
    domain: SecurityDomainId,
    value: &Bytes,
    staged: &crate::chunk_lane::StagedValue,
) {
    let Some(worker) = redundancy else {
        return;
    };
    worker.submit_staged_value(domain, value, staged);
}
