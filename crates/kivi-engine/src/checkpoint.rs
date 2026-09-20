//! Background checkpoint worker: triggers, builds, publication, reclamation.
//!
//! One plain (non-reactor) thread per engine in durable mode. It never
//! touches tablet state directly: captures arrive through the worker
//! control channel (cloned off the hot thread at a commit boundary),
//! builds run here (CPU off the `DataWorker`), publication uses the
//! crash-safe catalog protocol, and WAL maintenance goes straight to the
//! durability-lane threads (never through the reactor).
//!
//! Trigger policy (all configurable, conservative defaults from
//! measurement): WAL bytes since the last checkpoint, applied-commit
//! advance since the last cut, elapsed time with dirty state, or an
//! explicit operator trigger. At most one build runs at a time; triggers
//! that arrive mid-build coalesce into a single follow-up (bounded
//! backpressure, never a queue).
//!
//! Reclamation invariant (spec Y): the WAL floor is the oldest cut any
//! retained checkpoint promises recovery from — the previous cut when a
//! previous checkpoint exists, else the current cut. Floors advance only
//! after the justifying checkpoints are installed; segments are deleted
//! only after the floor that covers them is durable.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, bounded};
use kivi_checkpoint::{
    ArtifactHash, CheckpointIdentity, CheckpointPolicy, CurrentRecord, LoadedManifest,
    TabletSnapshot, build_tablet, publish_tablet,
};
use kivi_durability::LaneFloor;
use kivi_types::{ClusterId, NamespaceId, NodeId, NodeIncarnation, TabletId};

use crate::worker::{ControlIngress, TabletCheckpointStatus, WorkerControl};

/// Automatic checkpoint triggers. All thresholds are conservative:
/// checkpoints are rare, bounded background work — never a fixed cadence
/// regardless of activity.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// WAL bytes (per worker, since its last checkpoint) that trigger a
    /// checkpoint of that worker's tablets.
    pub wal_bytes_threshold: u64,
    /// Applied-commit advance (per tablet, since its last cut) that
    /// triggers a checkpoint.
    pub mutations_threshold: u64,
    /// Minimum age of the last checkpoint before the elapsed-time trigger
    /// may fire (and only when dirty state exists).
    pub min_interval: Duration,
    /// Build policy (layout + codec).
    pub policy: CheckpointPolicy,
    /// Disable automatic checkpoints (manual triggers still work).
    /// Benchmarks use this to isolate commit-pipeline variables.
    pub disabled: bool,
}

impl CheckpointConfig {
    /// Conservative production default: 64 MiB of WAL per worker, or
    /// 100k applied commits per tablet, or 5 minutes with dirty state.
    pub const DEFAULT_WAL_BYTES: u64 = 64 * 1024 * 1024;
    /// Default applied-commit advance per tablet.
    pub const DEFAULT_MUTATIONS: u64 = 100_000;
    /// Default minimum checkpoint age.
    pub const DEFAULT_MIN_INTERVAL: Duration = Duration::from_secs(300);

    /// Production default triggers.
    #[must_use]
    pub const fn default_config() -> Self {
        Self {
            wal_bytes_threshold: Self::DEFAULT_WAL_BYTES,
            mutations_threshold: Self::DEFAULT_MUTATIONS,
            min_interval: Self::DEFAULT_MIN_INTERVAL,
            policy: CheckpointPolicy::DEFAULT,
            disabled: false,
        }
    }
}

/// Per-tablet checkpoint facts for the admin plane.
#[derive(Debug, Clone, Default)]
pub struct CheckpointInfo {
    /// Installed cut.
    pub cut: u64,
    /// Manifest content hash.
    pub manifest: Option<ArtifactHash>,
    /// Wall micros at publication (age reporting).
    pub created_wall_micros: u64,
    /// Bands referenced.
    pub bands: usize,
    /// Bands reused from the previous checkpoint.
    pub bands_reused: usize,
    /// Stored checkpoint bytes (new artifacts at publish).
    pub stored_bytes: u64,
    /// Last build + publish duration.
    pub duration: Duration,
    /// Whether recovery fell back to this checkpoint.
    pub fell_back: bool,
}

/// Live checkpoint state shared with the admin plane (brief mutex,
/// status only — never object state, never the data path).
#[derive(Debug, Clone, Default)]
pub struct CheckpointAdminState {
    /// Installed current checkpoint per tablet.
    pub latest: HashMap<TabletId, CheckpointInfo>,
    /// Retained previous checkpoint per tablet.
    pub previous: HashMap<TabletId, CheckpointInfo>,
    /// WAL bytes currently retained on disk (laid out, all lanes).
    pub wal_retained_bytes: u64,
    /// WAL bytes the last plan proved reclaimable (deleted or pending).
    pub wal_reclaimable_bytes: u64,
    /// Last checkpoint/reclaim error (sticky until the next success).
    pub last_error: Option<String>,
    /// Checkpoints completed.
    pub checkpoints_completed: u64,
}

/// Commands to the checkpoint worker.
#[derive(Debug)]
pub enum CheckpointCommand {
    /// Request checkpoints now (`None` = every tablet with dirty state).
    Trigger {
        /// Tablet to checkpoint, or all dirty tablets.
        tablet: Option<TabletId>,
    },
    /// Finish the in-flight build (if any) and exit.
    Shutdown,
}

/// One lane's maintenance access plus its identity for planning.
#[derive(Debug, Clone)]
pub(crate) struct LaneMaintenanceAccess {
    /// Worker that owns (or shares) the lane, for tracing.
    pub worker: kivi_types::WorkerId,
    /// Lane index.
    pub lane: u16,
    /// Direct command sender to the lane thread.
    pub maintenance: crate::commit::LaneMaintenance,
}

/// Context the checkpoint worker needs. Everything is owned, `Send`, and
/// blocking-safe (this thread never touches the reactor).
pub(crate) struct CheckpointContext {
    /// Data-directory root.
    pub data_dir: std::path::PathBuf,
    /// Namespace served.
    pub namespace: NamespaceId,
    /// Publisher identity base (incarnation = this process).
    pub cluster: ClusterId,
    /// Publisher identity base.
    pub node: NodeId,
    /// This process's incarnation (stamped into new manifests).
    pub incarnation: NodeIncarnation,
    /// Writable tablets to checkpoint.
    pub tablets: Vec<TabletId>,
    /// Owner worker per tablet (authoritative placement from engine start;
    /// used only to attribute per-worker WAL bytes to the WAL trigger).
    pub placement: HashMap<TabletId, kivi_types::WorkerId>,
    /// Control ingress bundles per worker (captures, metrics, status).
    /// Every send through a bundle pings the networked bridge, so
    /// checkpoint queries never stall behind an idle bridge.
    pub controls: Vec<(kivi_types::WorkerId, ControlIngress)>,
    /// Lane maintenance access per worker.
    pub lanes: Vec<LaneMaintenanceAccess>,
    /// Trigger configuration.
    pub config: CheckpointConfig,
    /// Installed currents (loaded at startup, updated per publish).
    pub installed: HashMap<TabletId, CurrentRecord>,
    /// Engine-global in-flight staging pins: chunk GC unions these into
    /// every plan so staged-but-uncommitted packs survive the mark.
    pub pins: Arc<crate::chunk_lane::StagingPins>,
    /// Chunk lane per worker for GC planning and reclamation.
    pub chunk_lanes: Vec<(kivi_types::WorkerId, crate::chunk_lane::ChunkLaneHandle)>,
    /// Fabric journal path per worker for journal GC.
    pub fabric_journals: Vec<(kivi_types::WorkerId, std::path::PathBuf)>,
    /// Shared admin state.
    pub admin: Arc<Mutex<CheckpointAdminState>>,
}

/// Per-tablet worker-side bookkeeping (trigger baselines).
#[derive(Debug, Clone, Default)]
struct TabletLedger {
    /// Cut of the last completed checkpoint.
    last_cut: u64,
    /// When the last checkpoint completed.
    last_completed: Option<Instant>,
    /// Manual trigger pending (coalesced).
    manual_pending: bool,
}

/// Handle to the background checkpoint worker.
pub(crate) struct CheckpointWorkerHandle {
    submit: Sender<CheckpointCommand>,
    thread: Option<JoinHandle<()>>,
}

impl CheckpointWorkerHandle {
    /// Spawns the checkpoint worker thread.
    pub(crate) fn spawn(context: CheckpointContext) -> Self {
        let (submit, receive) = bounded::<CheckpointCommand>(16);
        let thread = thread::Builder::new()
            .name("kivi-checkpoint".to_owned())
            .spawn(move || run_checkpoint(context, &receive))
            .expect("checkpoint thread spawns");
        Self {
            submit,
            thread: Some(thread),
        }
    }

    /// Requests checkpoints (operator trigger).
    pub(crate) fn trigger(&self, tablet: Option<TabletId>) {
        let _ = self.submit.try_send(CheckpointCommand::Trigger { tablet });
    }

    /// Finishes the in-flight build (if any) and joins the thread.
    pub(crate) fn shutdown(&mut self) {
        let _ = self.submit.send(CheckpointCommand::Shutdown);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Checkpoint worker body: evaluate triggers, build, publish, reclaim.
fn run_checkpoint(mut context: CheckpointContext, commands: &Receiver<CheckpointCommand>) {
    let mut ledgers: HashMap<TabletId, TabletLedger> = context
        .tablets
        .iter()
        .map(|tablet| {
            (
                *tablet,
                TabletLedger {
                    last_cut: context
                        .installed
                        .get(tablet)
                        .map_or(0, |current| current.cut),
                    last_completed: None,
                    manual_pending: false,
                },
            )
        })
        .collect();
    // WAL-byte baselines per worker (cumulative counters need a datum).
    let mut wal_baselines: HashMap<kivi_types::WorkerId, u64> = HashMap::new();
    // Retained chunked-commit journals per tablet: `(commit, manifest)`
    // entries newer than the older retained cut. Replaced wholesale on
    // every journal collection (workers return the pruned remainder), so
    // no merge logic can skew it.
    let mut chunk_journals: HashMap<TabletId, Vec<(u64, kivi_types::ManifestId)>> = HashMap::new();
    loop {
        match commands.recv_timeout(Duration::from_secs(1)) {
            Ok(CheckpointCommand::Trigger { tablet }) => match tablet {
                Some(tablet) => {
                    ledgers.entry(tablet).or_default().manual_pending = true;
                }
                None => {
                    for ledger in ledgers.values_mut() {
                        ledger.manual_pending = true;
                    }
                }
            },
            Ok(CheckpointCommand::Shutdown)
            | Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                return;
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
        }
        // Builds run synchronously and sequentially (at most one in
        // flight, trivially); triggers that arrive mid-build cannot exist
        // — the ledger flags set above coalesce everything into the next
        // evaluation, which is the documented backpressure policy.
        let due = due_tablets(&context, &ledgers, &mut wal_baselines);
        for tablet in due {
            checkpoint_tablet(&mut context, &mut ledgers, &mut chunk_journals, tablet);
        }
    }
}

/// Returns tablets whose triggers fire, in tablet order.
fn due_tablets(
    context: &CheckpointContext,
    ledgers: &HashMap<TabletId, TabletLedger>,
    wal_baselines: &mut HashMap<kivi_types::WorkerId, u64>,
) -> Vec<TabletId> {
    if context.config.disabled {
        // Automatic triggers off — but manual requests still flow below.
        return ledgers
            .iter()
            .filter(|(_, ledger)| ledger.manual_pending)
            .map(|(tablet, _)| *tablet)
            .collect();
    }
    let statuses = tablet_statuses(context);
    let metrics = commit_metrics(context);
    let mut due = Vec::new();
    for tablet in &context.tablets {
        let Some(ledger) = ledgers.get(tablet) else {
            continue;
        };
        if ledger.manual_pending {
            due.push(*tablet);
            continue;
        }
        let Some(status) = statuses.get(tablet) else {
            continue;
        };
        if status.applied <= ledger.last_cut {
            continue;
        }
        // Applied-commit advance since the last cut.
        if status.applied.saturating_sub(ledger.last_cut) >= context.config.mutations_threshold {
            due.push(*tablet);
            continue;
        }
        // Elapsed time with dirty state.
        if status.dirty > 0
            && ledger
                .last_completed
                .is_none_or(|completed| completed.elapsed() >= context.config.min_interval)
        {
            // The elapsed trigger needs a first checkpoint to age from;
            // tablets never checkpointed with dirt qualify immediately
            // only through the other triggers (avoids a checkpoint storm
            // on startup).
            if ledger.last_completed.is_some() {
                due.push(*tablet);
                continue;
            }
        }
        // WAL bytes on the owning worker since its last checkpoint.
        if worker_wal_dirty(context, *tablet, &metrics, wal_baselines) {
            due.push(*tablet);
        }
    }
    due.sort_by_key(|tablet| tablet.as_u64());
    due
}

/// Whether the tablet's owning worker wrote enough WAL since baseline.
fn worker_wal_dirty(
    context: &CheckpointContext,
    tablet: TabletId,
    metrics: &HashMap<kivi_types::WorkerId, u64>,
    wal_baselines: &mut HashMap<kivi_types::WorkerId, u64>,
) -> bool {
    let Some(worker) = context.tablet_worker(tablet) else {
        return false;
    };
    let Some(bytes) = metrics.get(&worker) else {
        return false;
    };
    let baseline = wal_baselines.entry(worker).or_insert(*bytes);
    if bytes.saturating_sub(*baseline) >= context.config.wal_bytes_threshold {
        *baseline = *bytes;
        return true;
    }
    false
}

/// Per-tablet applied-cut and dirty-band counts from the workers.
fn tablet_statuses(context: &CheckpointContext) -> HashMap<TabletId, TabletCheckpointStatus> {
    let mut out = HashMap::new();
    for (_, control) in &context.controls {
        let (respond, receive) = bounded::<Vec<TabletCheckpointStatus>>(1);
        if control
            .try_send(WorkerControl::TabletCheckpointStatus { respond })
            .is_err()
        {
            continue;
        }
        if let Ok(statuses) = receive.recv_timeout(Duration::from_secs(5)) {
            for status in statuses {
                out.insert(status.tablet, status);
            }
        }
    }
    out
}

/// Cumulative WAL bytes made durable per worker (commit metrics).
fn commit_metrics(context: &CheckpointContext) -> HashMap<kivi_types::WorkerId, u64> {
    let mut out = HashMap::new();
    for (worker, control) in &context.controls {
        let (respond, receive) = bounded::<Option<crate::commit::CommitMetricsSnapshot>>(1);
        if control
            .try_send(WorkerControl::CommitMetrics { respond })
            .is_err()
        {
            continue;
        }
        if let Ok(Some(metrics)) = receive.recv_timeout(Duration::from_secs(5)) {
            out.insert(*worker, metrics.bytes_durable);
        }
    }
    out
}

/// Captures, builds, publishes, and reclaims one tablet end to end.
fn checkpoint_tablet(
    context: &mut CheckpointContext,
    ledgers: &mut HashMap<TabletId, TabletLedger>,
    chunk_journals: &mut HashMap<TabletId, Vec<(u64, kivi_types::ManifestId)>>,
    tablet: TabletId,
) {
    let started = Instant::now();
    let outcome = checkpoint_tablet_inner(context, tablet, started);
    let ledger = ledgers.entry(tablet).or_default();
    ledger.manual_pending = false;
    match outcome {
        Ok((cut, info)) => {
            ledger.last_cut = cut;
            ledger.last_completed = Some(Instant::now());
            // Chunk GC under the new retention promise: collect journals
            // (pruned to the older retained cut) and reclaim fully-dead
            // sealed chunk packs. Best-effort like WAL reclamation —
            // failures cost disk, never correctness — so GC trouble warns
            // without failing the checkpoint that just succeeded.
            collect_chunk_journals(context, chunk_journals);
            run_chunk_gc(context, chunk_journals);
            // Fabric journal GC under the same retention promise:
            // superseded seals and aborted prepares compact away once
            // the checkpoint covers their versions.
            run_fabric_journal_gc(context);
            context
                .admin
                .lock()
                .map(|mut admin| {
                    // Demote the superseded latest to the retained
                    // previous slot (mirrors the CURRENT record chain).
                    if let Some(demoted) = admin.latest.insert(tablet, info) {
                        admin.previous.insert(tablet, demoted);
                    }
                    admin.checkpoints_completed += 1;
                    admin.last_error = None;
                })
                .ok();
            tracing::info!(
                tablet = tablet.as_u64(),
                cut,
                elapsed_ms = started.elapsed().as_millis(),
                "checkpoint complete"
            );
        }
        Err(error) => {
            context
                .admin
                .lock()
                .map(|mut admin| {
                    admin.last_error = Some(format!("{error:?}"));
                })
                .ok();
            tracing::warn!(tablet = tablet.as_u64(), ?error, "checkpoint failed");
        }
    }
}

fn checkpoint_tablet_inner(
    context: &mut CheckpointContext,
    tablet: TabletId,
    started: Instant,
) -> Result<(u64, CheckpointInfo), CheckpointFailure> {
    // 1. Capture off the hot worker (commit boundary, no waiting).
    let snapshot = capture_tablet(context, tablet)?;
    // 2. Load the previous manifest for band reuse + the chain link.
    let previous = match context.installed.get(&tablet) {
        None => None,
        Some(current) => load_previous_manifest(context, tablet, current)
            .map(|manifest| (manifest, current.manifest)),
    };
    // 3. Build (CPU off the DataWorker).
    let identity = CheckpointIdentity {
        cluster: context.cluster,
        node: context.node,
        incarnation: context.incarnation,
        namespace: context.namespace,
        tablet,
    };
    let built = build_tablet(
        &snapshot,
        previous.as_ref().map(|(manifest, hash)| (manifest, *hash)),
        &context.config.policy,
        &identity,
    )
    .map_err(CheckpointFailure::Build)?;
    // 4. Publish (crash-safe install + retention GC).
    let previous_current = context.installed.get(&tablet).cloned();
    let (installed, publish_stats) =
        publish_tablet(&context.data_dir, &built, previous_current.as_ref())
            .map_err(CheckpointFailure::Publish)?;
    // The installed map updates before reclamation: the floor promise
    // must cover the just-published cut, or its tail would never become
    // reclaimable until the NEXT checkpoint.
    context.installed.insert(tablet, installed.clone());
    // 5. Reclaim WAL under the new retention promise.
    let reclaim_stats = reclaim_wal(context)?;
    let info = CheckpointInfo {
        cut: installed.cut,
        manifest: Some(installed.manifest),
        created_wall_micros: installed.created_wall_micros,
        bands: built.bands.len(),
        bands_reused: built.stats.bands_reused,
        stored_bytes: built.stats.stored_bytes,
        duration: started.elapsed(),
        fell_back: false,
    };
    let _ = (publish_stats, reclaim_stats);
    Ok((installed.cut, info))
}

/// Collects chunked-commit journals after a publish, pruning each tablet
/// to its older retained cut (whose bands plus the WAL floor cover older
/// commits from here on). Workers return the pruned remainder, which
/// replaces the retained copy wholesale — no merge logic to skew.
fn collect_chunk_journals(
    context: &CheckpointContext,
    retained: &mut HashMap<TabletId, Vec<(u64, kivi_types::ManifestId)>>,
) {
    use std::collections::HashMap as Map;
    let bounds: Vec<(TabletId, u64)> = context
        .installed
        .keys()
        .map(|tablet| (*tablet, previous_cut(context, *tablet).unwrap_or(0)))
        .collect();
    if bounds.is_empty() {
        return;
    }
    let mut gathered: Map<TabletId, Vec<(u64, kivi_types::ManifestId)>> = Map::new();
    for (_, control) in &context.controls {
        let (respond, receive) = bounded::<crate::worker::ChunkJournals>(1);
        if control
            .try_send(WorkerControl::ChunkedJournal {
                bounds: bounds.clone(),
                respond,
            })
            .is_err()
        {
            continue;
        }
        if let Ok(journals) = receive.recv_timeout(Duration::from_secs(5)) {
            for (tablet, journal) in journals {
                gathered.insert(tablet, journal);
            }
        }
    }
    for (tablet, journal) in gathered {
        if journal.is_empty() {
            retained.remove(&tablet);
        } else {
            retained.insert(tablet, journal);
        }
    }
    // Tablets that answered nothing keep their previous remainder (a
    // missed round retains longer, never shorter).
}

/// Chunk manifests referenced by the retained checkpoints (current plus
/// previous per tablet), read from installed band files. `None` when any
/// link in the chain is unreadable: GC then skips the round entirely
/// (keep-on-doubt — disk usage, never corruption).
fn retained_band_chunk_refs(
    context: &CheckpointContext,
) -> Option<std::collections::HashSet<kivi_types::ManifestId>> {
    use kivi_checkpoint::{ArtifactKind, artifact::artifact_path, load_band, load_manifest};
    let mut out = std::collections::HashSet::new();
    for (tablet, current) in &context.installed {
        let mut hashes = vec![current.manifest];
        if let Some(previous) = current.previous {
            hashes.push(previous);
        }
        for hash in hashes {
            let path = artifact_path(&context.data_dir, *tablet, ArtifactKind::Manifest, hash);
            let bytes = std::fs::read(&path).ok()?;
            let manifest = load_manifest(hash, &bytes).ok()?;
            for band in &manifest.bands {
                let path = artifact_path(&context.data_dir, *tablet, ArtifactKind::Band, band.hash);
                let bytes = std::fs::read(&path).ok()?;
                let loaded =
                    load_band(*tablet, band.band, manifest.layout, band.hash, &bytes).ok()?;
                for record in &loaded.records {
                    if let Some(chunked) = record.object.chunk_ref() {
                        out.insert(chunked.manifest);
                    }
                }
            }
        }
    }
    Some(out)
}

/// Fabric ids referenced by the retained checkpoints (current plus
/// previous per tablet), read from installed band files. `None` when any
/// link in the chain is unreadable: GC then skips the round entirely
/// (keep-on-doubt - disk usage, never corruption).
fn retained_band_fabric_refs(
    context: &CheckpointContext,
) -> Option<std::collections::HashSet<u64>> {
    use kivi_checkpoint::{ArtifactKind, artifact::artifact_path, load_band, load_manifest};
    let mut out = std::collections::HashSet::new();
    for (tablet, current) in &context.installed {
        let mut hashes = vec![current.manifest];
        if let Some(previous) = current.previous {
            hashes.push(previous);
        }
        for hash in hashes {
            let path = artifact_path(&context.data_dir, *tablet, ArtifactKind::Manifest, hash);
            let bytes = std::fs::read(&path).ok()?;
            let manifest = load_manifest(hash, &bytes).ok()?;
            for band in &manifest.bands {
                let path = artifact_path(&context.data_dir, *tablet, ArtifactKind::Band, band.hash);
                let bytes = std::fs::read(&path).ok()?;
                let loaded =
                    load_band(*tablet, band.band, manifest.layout, band.hash, &bytes).ok()?;
                for record in &loaded.records {
                    if let Some(fabric) = record.object.fabric_ref() {
                        out.insert(fabric.id);
                    }
                }
            }
        }
    }
    Some(out)
}

/// Compacts fabric journals under the new retention promise: live ids
/// are worker-reported roots, intents, and in-flight payloads united
/// with retained-checkpoint band refs. Superseded records (older seals
/// for a live id) and orphaned records (aborted prepares) go; anything
/// else stays. Failures warn without failing the checkpoint that
/// succeeded. A worker that does not answer keeps its previous remainder
/// (a missed round retains longer, never shorter): journals compact
/// against their own worker's live set, never a partial union.
fn run_fabric_journal_gc(context: &CheckpointContext) {
    let Some(band_refs) = retained_band_fabric_refs(context) else {
        tracing::warn!("fabric journal GC skipped: retained checkpoint unreadable");
        return;
    };
    let mut live_by_worker: std::collections::HashMap<
        kivi_types::WorkerId,
        std::collections::HashSet<u64>,
    > = std::collections::HashMap::new();
    for (worker, control) in &context.controls {
        let (respond, receive) = bounded::<std::collections::HashSet<u64>>(1);
        if control
            .try_send(crate::worker::WorkerControl::FabricLiveRoots { respond })
            .is_err()
        {
            continue;
        }
        if let Ok(live) = receive.recv_timeout(Duration::from_secs(5)) {
            live_by_worker.insert(*worker, live);
        }
    }
    for (worker, journal) in &context.fabric_journals {
        let Some(live) = live_by_worker.get(worker) else {
            continue;
        };
        let keep: std::collections::HashSet<u64> = live.union(&band_refs).copied().collect();
        match crate::fabric::compact_journal(journal, &keep) {
            Ok(kept) => {
                tracing::info!(worker = %worker, kept, "fabric journal compacted");
            }
            Err(error) => {
                tracing::warn!(worker = %worker, ?error, "fabric journal compact failed; kept");
            }
        }
    }
}
fn run_chunk_gc(
    context: &CheckpointContext,
    retained: &HashMap<TabletId, Vec<(u64, kivi_types::ManifestId)>>,
) {
    let Some(mut live) = retained_band_chunk_refs(context) else {
        tracing::warn!("chunk GC skipped: retained checkpoint unreadable");
        return;
    };
    live.extend(retained.values().flatten().map(|(_, manifest)| *manifest));
    let live: Vec<kivi_types::ManifestId> = live.into_iter().collect();
    let (pinned_chunks, pinned_manifests) = context.pins.snapshot();
    let pinned_chunks: Vec<kivi_types::ChunkId> = pinned_chunks.iter().copied().collect();
    let pinned_manifests: Vec<kivi_types::ManifestId> = pinned_manifests.iter().copied().collect();
    for (worker, lane) in &context.chunk_lanes {
        let report = match lane.gc_mark_blocking(
            live.clone(),
            pinned_chunks.clone(),
            pinned_manifests.clone(),
        ) {
            Ok(report) => report,
            Err(error) => {
                tracing::warn!(worker = %worker, ?error, "chunk GC plan failed; packs kept");
                continue;
            }
        };
        if report.dead_sealed_packs.is_empty() {
            continue;
        }
        match lane.delete_packs_blocking(report.dead_sealed_packs.clone()) {
            Ok(deleted) => {
                tracing::info!(
                    worker = %worker,
                    packs = deleted.len(),
                    live_chunks = report.live_chunks,
                    live_manifests = report.live_manifests,
                    marked_bytes = report.marked_bytes,
                    "chunk GC reclaimed packs"
                );
            }
            Err(error) => {
                tracing::warn!(worker = %worker, ?error, "chunk pack deletion failed; packs kept");
            }
        }
    }
}

/// Captures one tablet's checkpoint view through the control channel.
fn capture_tablet(
    context: &CheckpointContext,
    tablet: TabletId,
) -> Result<TabletSnapshot, CheckpointFailure> {
    for (_, control) in &context.controls {
        let (respond, receive) = bounded::<Option<TabletSnapshot>>(1);
        if control
            .try_send(WorkerControl::CaptureTablet { tablet, respond })
            .is_err()
        {
            continue;
        }
        if let Ok(Some(snapshot)) = receive.recv_timeout(Duration::from_secs(30)) {
            return Ok(snapshot);
        }
    }
    Err(CheckpointFailure::Capture)
}

/// Loads the installed manifest for band reuse (None when never published
/// or when the chain is damaged — damage surfaces at publish binding, and
/// building without reuse stays correct, just less incremental).
fn load_previous_manifest(
    context: &CheckpointContext,
    tablet: TabletId,
    current: &CurrentRecord,
) -> Option<LoadedManifest> {
    let path = kivi_checkpoint::artifact::artifact_path(
        &context.data_dir,
        tablet,
        kivi_checkpoint::ArtifactKind::Manifest,
        current.manifest,
    );
    let Ok(bytes) = std::fs::read(&path) else {
        return None;
    };
    kivi_checkpoint::load_manifest(current.manifest, &bytes).ok()
}

/// Reclaims WAL under the new retention promise: seal actives, scan
/// sealed, plan against oldest-promise floors, publish the floor, delete
/// obsolete segments. Returns reclaimed bytes.
fn reclaim_wal(context: &mut CheckpointContext) -> Result<u64, CheckpointFailure> {
    use kivi_checkpoint::reclaim::plan_reclaim;
    // Floors: oldest promise per installed tablet — the verified
    // previous cut, or no promise at all. Tablets with a single
    // checkpoint (or an unverifiable previous) are omitted: their
    // records block reclamation, keeping genesis replay viable until
    // two checkpoints exist.
    let mut floors: HashMap<TabletId, u64> = HashMap::new();
    for tablet in context.installed.keys() {
        if let Some(cut) = previous_cut(context, *tablet) {
            floors.insert(*tablet, cut);
        }
    }
    // Seal every lane's active tail first so the listed sealed set is
    // stable while the (slow) build already happened... note ordering:
    // capture fixed cuts first, build ran, publish installed; sealing now
    // only affects future appends, never the promise math.
    let mut lane_workers: HashMap<u16, &LaneMaintenanceAccess> = HashMap::new();
    for access in &context.lanes {
        lane_workers.entry(access.lane).or_insert(access);
    }
    for access in lane_workers.values() {
        tracing::debug!(worker = %access.worker, lane = access.lane, "sealing active WAL segment");
        access
            .maintenance
            .seal_active_sync()
            .map_err(CheckpointFailure::Lane)?;
    }
    // List + scan sealed segments per lane.
    let wal_dir = context.data_dir.join(kivi_durability::node::WAL_DIR_NAME);
    let mut summaries: HashMap<u16, Vec<kivi_durability::SealedSegmentSummary>> = HashMap::new();
    let mut active: HashMap<u16, u64> = HashMap::new();
    let mut active_first: HashMap<u16, u64> = HashMap::new();
    let mut watermarks: HashMap<u16, u64> = HashMap::new();
    for (lane, access) in &lane_workers {
        let snapshot = access
            .maintenance
            .floor_snapshot_sync()
            .map_err(CheckpointFailure::Lane)?;
        active.insert(*lane, snapshot.active_segment);
        active_first.insert(*lane, snapshot.active_first_batch);
        watermarks.insert(*lane, snapshot.next_batch);
        let sealed = list_sealed_segments(&wal_dir, *lane, snapshot.active_segment)?;
        if sealed.is_empty() {
            summaries.insert(*lane, Vec::new());
            continue;
        }
        let scanned = access
            .maintenance
            .scan_sealed_sync(sealed)
            .map_err(CheckpointFailure::Lane)?;
        summaries.insert(*lane, scanned);
    }
    let plans = plan_reclaim(&summaries, &active, &active_first, &watermarks, &floors);
    // Publish floors first (durable promise), then delete.
    let floor_records: Vec<LaneFloor> = plans.iter().map(|plan| plan.floor).collect();
    if !floor_records.is_empty() {
        kivi_checkpoint::write_wal_floor(&context.data_dir, &floor_records)
            .map_err(CheckpointFailure::Floor)?;
    }
    let mut reclaimed_bytes = 0u64;
    for plan in &plans {
        if plan.obsolete.is_empty() {
            continue;
        }
        let access = lane_workers.get(&plan.lane).expect("planned lane present");
        // Byte accounting before deletion (best-effort stat).
        reclaimed_bytes += obsolete_bytes(&wal_dir, plan.lane, &plan.obsolete);
        let deleted = access
            .maintenance
            .delete_segments_sync(plan.obsolete.clone())
            .map_err(CheckpointFailure::Lane)?;
        if deleted.len() != plan.obsolete.len() {
            tracing::warn!(
                lane = plan.lane,
                planned = plan.obsolete.len(),
                deleted = deleted.len(),
                "WAL reclaim partially deleted; remainder kept"
            );
        }
    }
    context
        .admin
        .lock()
        .map(|mut admin| {
            admin.wal_reclaimable_bytes = 0;
            admin.wal_retained_bytes = retained_wal_bytes(&wal_dir);
        })
        .ok();
    Ok(reclaimed_bytes)
}

/// Previous cut for the floor promise (the oldest recovery point when a
/// previous checkpoint is retained).
fn previous_cut(context: &CheckpointContext, tablet: TabletId) -> Option<u64> {
    let current = context.installed.get(&tablet)?;
    let previous_hash = current.previous?;
    let bytes = std::fs::read(kivi_checkpoint::artifact::artifact_path(
        &context.data_dir,
        tablet,
        kivi_checkpoint::ArtifactKind::Manifest,
        previous_hash,
    ))
    .ok()?;
    kivi_checkpoint::load_manifest(previous_hash, &bytes)
        .ok()
        .map(|manifest| manifest.cut)
}

/// Lists sealed segments for one lane directly from its directory
/// (excluding the active tail).
fn list_sealed_segments(
    wal_dir: &std::path::Path,
    lane: u16,
    active: u64,
) -> Result<Vec<u64>, CheckpointFailure> {
    let dir = wal_dir.join(kivi_durability::wal::lane_dir_name(lane));
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(CheckpointFailure::Reclaim(
                kivi_checkpoint::CheckpointError::io("list WAL lane", &dir, &error),
            ));
        }
    };
    let mut sealed = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(seq) = name
            .strip_suffix(".wal")
            .and_then(|stem| stem.parse::<u64>().ok())
            && seq != active
        {
            sealed.push(seq);
        }
    }
    sealed.sort_unstable();
    Ok(sealed)
}

/// Sums file sizes for accounting (best-effort stat, never a promise).
fn obsolete_bytes(wal_dir: &std::path::Path, lane: u16, segments: &[u64]) -> u64 {
    let dir = wal_dir.join(kivi_durability::wal::lane_dir_name(lane));
    segments
        .iter()
        .map(|segment| {
            std::fs::metadata(dir.join(kivi_durability::wal::segment_file_name(*segment)))
                .map_or(0, |meta| meta.len())
        })
        .sum()
}

/// Sums all WAL files for admin visibility (best-effort stat).
fn retained_wal_bytes(wal_dir: &std::path::Path) -> u64 {
    let mut total = 0u64;
    let Ok(lanes) = std::fs::read_dir(wal_dir) else {
        return 0;
    };
    for lane in lanes.flatten() {
        if !lane.file_type().is_ok_and(|kind| kind.is_dir()) {
            continue;
        }
        for segment in std::fs::read_dir(lane.path())
            .into_iter()
            .flatten()
            .flatten()
        {
            total += segment.metadata().map_or(0, |meta| meta.len());
        }
    }
    total
}

/// Checkpoint failure modes (worker-internal; surfaced via admin + logs).
#[derive(Debug, thiserror::Error)]
pub(crate) enum CheckpointFailure {
    /// Tablet capture went unanswered.
    #[error("tablet capture unanswered by every worker")]
    Capture,
    /// Snapshot build failed.
    #[error("checkpoint build failed: {0}")]
    Build(kivi_checkpoint::CheckpointError),
    /// Crash-safe publication failed.
    #[error("checkpoint publication failed: {0}")]
    Publish(kivi_checkpoint::CheckpointError),
    /// WAL reclamation failed (floors unpublished or deletes failed).
    #[error("WAL reclamation failed: {0}")]
    Reclaim(kivi_checkpoint::CheckpointError),
    /// Durability-lane maintenance failed.
    #[error("durability-lane maintenance failed: {0}")]
    Lane(kivi_durability::DurabilityError),
    /// WAL-floor publication failed.
    #[error("WAL-floor publication failed: {0}")]
    Floor(kivi_checkpoint::CheckpointError),
}

impl CheckpointContext {
    /// Owning worker of one tablet from authoritative engine placement.
    fn tablet_worker(&self, tablet: TabletId) -> Option<kivi_types::WorkerId> {
        self.placement.get(&tablet).copied()
    }
}
