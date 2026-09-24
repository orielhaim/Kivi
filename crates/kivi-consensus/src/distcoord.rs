//! Distributed redundancy coordinator: cluster-wide protect/read/repair.
//!
//! Placement justification: this lives in `kivi-consensus` (not
//! `kivi-redundancy` and not `kivi-server`) because it needs the shared peer
//! mesh, the replicated control plane, and the node-local fragment store
//! together, while `kivi-redundancy` must stay transport- and
//! consensus-agnostic and `kivi-server` must stay a thin wiring layer. The
//! coordinator reuses the exact [`crate::transport::PeerTransport`] the node
//! opened (same identity, same runtime, same dials) via
//! [`crate::multi::ConsensusNode::mesh_transport`], so no second
//! mesh/runtime/identity is ever created for redundancy traffic.
//!
//! ## What this owns
//!
//! [`RedundancyCoordinator`] binds a [`crate::fragments::FragmentClient`]
//! (mesh delivery), a control-backed catalog/publisher pair (authority), a
//! [`kivi_redundancy::DistributedFabric`] (plan/encode/verify/repair), and a
//! [`kivi_redundancy::RedundancyLane`] (bounded background CPU) into one
//! operator surface. Tablet workers never block: every method below is an
//! ordinary `Send` future awaited on Tokio admin tasks; consensus itself
//! keeps running on its Compio owner threads.
//!
//! ## Raft semantics
//!
//! The coordinator never orders mutable state, never erasure-codes Raft
//! state, and never weakens any durability gate. Every publish is `build
//! (remote stage) -> verify (re-fetch minimum + decode + content-hash) ->
//! publish (control commit + wait) -> retire (best-effort drops)` with
//! generation fencing: stale completions cannot overwrite newer layouts,
//! reads never republish, and orphans (staged fragments without authority)
//! may leak but are never authoritative.
//!
//! ## Incarnation discipline
//!
//! The believed target incarnation starts at `0` (the bootstrap probe,
//! always servable) and pins on first
//! [`kivi_redundancy::proto::RefuseReason::StaleIncarnation`] carrying the
//! holder's current epoch; the fabric retries exactly once with fresh
//! belief. Operators may additionally seed beliefs via
//! [`RedundancyCoordinator::note_incarnation`] (for example from `/v1/node`
//! observations); the registry itself carries no incarnation field.
//!
//! ## Two-source recovery
//!
//! Restart recovers through two independent sources: the node-local
//! [`kivi_redundancy::LocalFragmentStore`] reopens and re-indexes its
//! fragment/layout files (see [`kivi_redundancy::store`]), while the layout
//! catalog recovers through control-log replay plus snapshot (the control
//! plane owns which generation is current). Either source alone is
//! insufficient by design: fragments without authority are orphans, and
//! authority without fragments is degraded until repair.
//!
//! ## Bounds
//!
//! Every remote RPC carries [`CoordinatorConfig::rpc_timeout`]; every
//! control commit waits at most
//! [`CoordinatorConfig::control_wait_timeout`] with
//! [`CoordinatorConfig::control_poll_interval`] polls; images are capped at
//! 8 MiB decoded per call (fragment wire caps); every served byte is
//! content-verified against its layout record and asset id before exposure.
//!
//! ## Identity reuse
//!
//! Chunk and checkpoint families reuse their canonical identities (chunk
//! `ChunkId`, manifest `ManifestId`, checkpoint `blake3_256` digests)
//! instead of inventing a parallel object system: `restore-chunk` heals the
//! sidecar cache through the same [`kivi_redundancy::AssetId`] the generic
//! bytes path uses, and checkpoint bands travel as generic bytes under
//! their whole-file identity.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use kivi_control::{ControlMutation, ControlState};
use kivi_redundancy::{
    AssetHealth, AssetId, AssetKind, CatalogPublish, DistConfig, DistributedFabric, FabricMetrics,
    FabricMetricsSnapshot, FragmentStore as _, InformationAsset, LaneConfig, LaneStats,
    LayoutCatalog, LocalFragmentStore, PublishedLayout, RedundancyError, RedundancyIntent,
    RedundancyLane, RepairReport, SchemeParams,
};
use kivi_types::{ChunkId, NodeId, SecurityDomainId};

use crate::fragments::FragmentClient;
use crate::multi::ConsensusNode;

/// Coordinator bounds: per-RPC ceilings plus control-commit waits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoordinatorConfig {
    /// Deadline per fragment RPC attempt.
    pub rpc_timeout: Duration,
    /// How long a catalog publish waits for commit visibility.
    pub control_wait_timeout: Duration,
    /// Interval between commit-visibility polls.
    pub control_poll_interval: Duration,
    /// Maximum decoded image bytes per call (fragment wire caps).
    pub max_bytes: u64,
}

impl CoordinatorConfig {
    /// Conservative defaults: 30 s RPCs, 30 s commit waits, 100 ms polls,
    /// 8 MiB images.
    #[must_use]
    pub const fn conservative() -> Self {
        Self {
            rpc_timeout: Duration::from_secs(30),
            control_wait_timeout: Duration::from_secs(30),
            control_poll_interval: Duration::from_millis(100),
            max_bytes: 8 * 1024 * 1024,
        }
    }

    /// Validates nonzero bounds.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::InvalidParams`] on zero bounds.
    pub const fn validate(self) -> Result<(), RedundancyError> {
        if self.rpc_timeout.as_nanos() == 0
            || self.control_wait_timeout.as_nanos() == 0
            || self.control_poll_interval.as_nanos() == 0
            || self.max_bytes == 0
        {
            return Err(RedundancyError::InvalidParams {
                detail: String::new(),
            });
        }
        Ok(())
    }
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self::conservative()
    }
}

/// Control-backed catalog reads: the control plane owns which generation is
/// current, so `get` replays the latest committed image through
/// [`kivi_control::redundancy::published_layout`].
#[derive(Debug)]
struct ControlCatalog {
    /// Node whose replicated control image is read.
    node: Arc<ConsensusNode>,
}

impl LayoutCatalog for ControlCatalog {
    fn get(&self, asset: &AssetId) -> Option<PublishedLayout> {
        // Sync-trait bridge: the control front is runtime-agnostic (bounded
        // owner queue plus oneshot), so driving it here with a dedicated
        // executor never needs the caller's runtime and never blocks a
        // consensus worker (progress lives on Compio threads).
        let state: Option<ControlState> = futures::executor::block_on(self.node.control_state());
        let state = state?;
        kivi_control::redundancy::published_layout(&state, asset)
    }

    fn list(&self) -> Vec<PublishedLayout> {
        let Some(state) = futures::executor::block_on(self.node.control_state()) else {
            return Vec::new();
        };
        let mut layouts: Vec<PublishedLayout> = state
            .redundancy_entries()
            .filter_map(|(key, _)| {
                let asset = key.to_asset().ok()?;
                kivi_control::redundancy::published_layout(&state, &asset)
            })
            .collect();
        layouts.sort_by_key(|record| record.layout.asset);
        layouts
    }
}

/// Control-backed catalog writes: propose the mutation, then wait (bounded)
/// until the generation is visible in the committed image.
#[derive(Debug)]
struct ControlPublisher {
    /// Node proposing control mutations.
    node: Arc<ConsensusNode>,
    /// Maximum wait for commit visibility.
    wait: Duration,
    /// Interval between visibility polls.
    poll: Duration,
}

impl ControlPublisher {
    /// Waits until `visible` reports true or the deadline passes.
    fn wait_visible(&self, visible: impl Fn() -> bool) -> bool {
        let deadline = Instant::now() + self.wait;
        while Instant::now() < deadline {
            if visible() {
                return true;
            }
            std::thread::sleep(self.poll);
        }
        visible()
    }

    /// Reads one committed image (sync-trait bridge, same discipline as
    /// [`ControlCatalog`]).
    fn snapshot(&self) -> Option<ControlState> {
        futures::executor::block_on(self.node.control_state())
    }

    /// Proposes a control mutation, retrying genuinely transient failures.
    ///
    /// The proposing path first tries the local control group and performs
    /// the standard Raft client redirect over the existing peer mesh when
    /// this node is not the leader. The publisher retries on the next tick
    /// for transient failures until the commit becomes visible or the
    /// bounded wait expires. Every retry re-uses the same fenced mutation,
    /// so a duplicate commit is an identical no-op under control's
    /// generation rules.
    fn propose_retry(&self, mutation: &ControlMutation, op: &'static str) -> Result<(), String> {
        let deadline = Instant::now() + self.wait;
        let mut last;
        loop {
            match futures::executor::block_on(
                self.node
                    .propose_control_anywhere(mutation.clone(), self.poll * 20),
            ) {
                Ok(_) => return Ok(()),
                Err(detail) => last = detail,
            }
            if Instant::now() >= deadline {
                return Err(format!("{op} not committed: {last}"));
            }
            std::thread::sleep(self.poll);
        }
    }
}

impl CatalogPublish for ControlPublisher {
    fn publish(&self, record: PublishedLayout) -> Result<PublishedLayout, RedundancyError> {
        let key = kivi_control::redundancy::layout_key(&record.layout.asset);
        let generation = record.control_generation;
        if generation == 0 {
            return Err(RedundancyError::invalid_asset(
                "control generation 0 is the invalid sentinel",
            ));
        }
        let layout_bytes = record.layout.encode();
        let mutation = ControlMutation::PublishRedundancyLayout {
            key,
            generation,
            record: kivi_control::layouts::LayoutRecord::new(generation, layout_bytes.clone()),
        };
        self.propose_retry(&mutation, "redundancy publish")
            .map_err(|detail| RedundancyError::Unreachable { detail })?;
        let asset = record.layout.asset;
        let seen = self.wait_visible(|| {
            self.snapshot()
                .and_then(|state| kivi_control::redundancy::published_layout(&state, &asset))
                .is_some_and(|current| {
                    current.control_generation == generation
                        && current.layout.encode() == layout_bytes
                })
        });
        if seen {
            return Ok(record);
        }
        // Timeout: distinguish fencing (a newer generation won) from a slow
        // commit so callers retry correctly instead of forking generations.
        let current_gen = self
            .snapshot()
            .and_then(|state| kivi_control::redundancy::published_layout(&state, &asset))
            .map_or(0, |current| current.control_generation);
        if current_gen >= generation {
            return Err(RedundancyError::StaleGeneration {
                generation,
                current: current_gen,
            });
        }
        Err(RedundancyError::Timeout {
            op: "publish-visibility".to_owned(),
        })
    }

    fn drop_asset(&self, asset: &AssetId) -> Result<(), RedundancyError> {
        let key = kivi_control::redundancy::layout_key(asset);
        let mutation = ControlMutation::DropRedundancyAsset { key };
        self.propose_retry(&mutation, "redundancy drop")
            .map_err(|detail| RedundancyError::Unreachable { detail })?;
        let seen = self.wait_visible(|| {
            self.snapshot().is_some_and(|state| {
                kivi_control::redundancy::published_layout(&state, asset).is_none()
            })
        });
        if seen {
            return Ok(());
        }
        Err(RedundancyError::Timeout {
            op: "drop-visibility".to_owned(),
        })
    }
}

/// Health view over installed state for operators.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoordinatorHealth {
    /// Asset assessed.
    pub asset: AssetId,
    /// Layout generation assessed.
    pub generation: u64,
    /// Control generation that published it.
    pub control_generation: u64,
    /// Health class over actually-installed healthy fragments.
    pub health: AssetHealth,
    /// Usable fragments now.
    pub usable: u32,
    /// Fragments required to reconstruct.
    pub required: u32,
    /// Total fragments in the layout.
    pub total: u32,
    /// Independent failures currently survivable.
    pub survivable: u32,
    /// Whether reconstruction is possible now.
    pub reconstructable: bool,
    /// Desired placement under the current view: `(index, node)`.
    pub desired: Vec<(u32, NodeId)>,
    /// Installed state: `(index, holder, healthy)` per published fragment.
    pub actual: Vec<(u32, NodeId, bool)>,
}

/// What one node drain installed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrainReport {
    /// Assets re-published away from the drained node.
    pub moved: u64,
    /// Which assets moved (sorted by kind, domain, hash).
    pub assets: Vec<AssetId>,
}

/// One catalog entry for operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogEntry {
    /// Asset identity.
    pub asset: AssetId,
    /// Control generation that published the current layout.
    pub control_generation: u64,
    /// Layout generation.
    pub generation: u64,
    /// Current scheme parameters.
    pub params: SchemeParams,
    /// Exact logical bytes.
    pub logical_len: u64,
}

/// Extended operator metrics: the fabric snapshot plus cluster-joined facts.
#[derive(Debug, Clone, PartialEq)]
pub struct CoordinatorMetrics {
    /// Live fabric counters.
    pub snapshot: FabricMetricsSnapshot,
    /// Staged fragments without authority on this node.
    pub orphans: u64,
    /// Stored fragment bytes per holder `(node, bytes)`, aggregated from
    /// published layouts (desired bytes, sorted by node).
    pub by_node_bytes: Vec<(u64, u64)>,
    /// Current generations `(asset, control generation)`, sorted.
    pub current_gens: Vec<(AssetId, u64)>,
    /// Logical bytes protected across published layouts.
    pub logical_bytes: u64,
    /// Fragment bytes published across layouts (the storage cost).
    pub physical_bytes: u64,
    /// Published layouts using replication.
    pub replicated_assets: u64,
    /// Published layouts using erasure coding.
    pub coded_assets: u64,
    /// Bounded self-healing controller state.
    pub maintenance: kivi_redundancy::ControllerSnapshot,
}

impl CoordinatorMetrics {
    /// Storage amplification (`physical / logical`), `None` when nothing is
    /// protected yet.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn amplification(&self) -> Option<f64> {
        if self.logical_bytes == 0 {
            return None;
        }
        Some(self.physical_bytes as f64 / self.logical_bytes as f64)
    }
}

/// Cluster-wide redundancy coordinator over the shared mesh and control.
///
/// See the module docs for placement, Raft, incarnation, recovery, bounds,
/// and identity-reuse contracts.
pub struct RedundancyCoordinator {
    /// Node for control snapshots/proposals, sidecar reads/stages, and the
    /// local fragment store.
    node: Arc<ConsensusNode>,
    /// Mesh delivery reusing the node's exact transport (no second mesh).
    client: FragmentClient,
    /// Cluster plane: plan/encode/verify/repair over the transport and
    /// control-backed catalog.
    fabric: DistributedFabric,
    /// Bounded background CPU (orphan sweeps; encodes stay inline bounded).
    lane: Arc<Mutex<RedundancyLane>>,
    /// Coordinator bounds.
    config: CoordinatorConfig,
    /// Last catalog page handed to the incremental maintenance scanner.
    maintenance_catalog: Mutex<Arc<Vec<PublishedLayout>>>,
    /// Last catalog refresh time.
    maintenance_catalog_refresh: Mutex<Option<Instant>>,
    /// Stops new maintenance work during orderly server shutdown.
    maintenance_stopped: AtomicBool,
    /// Last catalog-versus-physical reconciliation time.
    last_anti_entropy: Mutex<Option<Instant>>,
}

impl std::fmt::Debug for RedundancyCoordinator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RedundancyCoordinator")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl RedundancyCoordinator {
    /// Opens the coordinator over one node's mesh, control, sidecar, and
    /// fragment store.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] on invalid bounds or lane-open failure.
    pub fn open(
        node: Arc<ConsensusNode>,
        config: CoordinatorConfig,
    ) -> Result<Self, RedundancyError> {
        config.validate()?;
        let client = node.fragment_client(config.rpc_timeout);
        let catalog = Arc::new(ControlCatalog {
            node: Arc::clone(&node),
        });
        let publisher = Arc::new(ControlPublisher {
            node: Arc::clone(&node),
            wait: config.control_wait_timeout,
            poll: config.control_poll_interval,
        });
        let fabric = DistributedFabric::new(
            Arc::new(client.clone()) as Arc<dyn kivi_redundancy::FragmentTransport>,
            catalog as Arc<dyn LayoutCatalog>,
            publisher as Arc<dyn CatalogPublish>,
            Arc::new(FabricMetrics::new()),
            DistConfig {
                local: node.node(),
                rpc_timeout: config.rpc_timeout,
                ..DistConfig::conservative()
            },
        )?;
        let lane =
            RedundancyLane::open(LaneConfig::conservative()).map_err(|error| match error {
                RedundancyError::InvalidParams { .. } => RedundancyError::InvalidParams {
                    detail: "lane bounds".to_owned(),
                },
                other => other,
            })?;
        Ok(Self {
            node,
            client,
            fabric,
            lane: Arc::new(Mutex::new(lane)),
            config,
            maintenance_catalog: Mutex::new(Arc::new(Vec::new())),
            maintenance_catalog_refresh: Mutex::new(None),
            maintenance_stopped: AtomicBool::new(false),
            last_anti_entropy: Mutex::new(None),
        })
    }

    /// Returns the coordinator bounds.
    #[must_use]
    pub const fn config(&self) -> CoordinatorConfig {
        self.config
    }

    /// Returns the mesh fragment client (same shared mesh, same identity).
    #[must_use]
    pub fn fragment_client(&self) -> FragmentClient {
        self.client.clone()
    }

    /// Records a believed peer incarnation (control epochs, reconnects).
    pub fn note_incarnation(&self, node: NodeId, incarnation: u64) {
        self.fabric.set_peer_incarnation(node, incarnation);
    }

    /// Returns the believed incarnation (`0` = undiscovered bootstrap).
    #[must_use]
    pub fn peer_incarnation(&self, node: NodeId) -> u64 {
        self.fabric.peer_incarnation(node)
    }

    /// Stops accepting new maintenance work during orderly shutdown.
    pub fn stop_healing(&self) {
        self.maintenance_stopped.store(true, Ordering::Release);
    }

    /// Whether maintenance has been stopped.
    #[must_use]
    pub fn healing_stopped(&self) -> bool {
        self.maintenance_stopped.load(Ordering::Acquire)
    }

    /// Returns the distributed controller snapshot.
    #[must_use]
    pub fn maintenance_snapshot(&self) -> kivi_redundancy::ControllerSnapshot {
        self.fabric.maintenance_snapshot()
    }

    /// Runs one automatic maintenance window using a bounded catalog cache.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when no control image is available.
    pub async fn maintenance_tick(
        &self,
        foreground_pressure: bool,
    ) -> Result<kivi_redundancy::MaintenanceReport, RedundancyError> {
        if self.healing_stopped() {
            return self
                .fabric
                .maintenance_tick_at(&[], &[], foreground_pressure)
                .await;
        }
        let state =
            self.node
                .control_state()
                .await
                .ok_or_else(|| RedundancyError::Unreachable {
                    detail: "no local control image for redundancy maintenance".to_owned(),
                })?;
        let records: Vec<kivi_control::NodeRecord> = state.nodes().cloned().collect();
        let view = kivi_control::redundancy::descriptors_from_registry(&records);
        let refresh = {
            let last = self.maintenance_catalog_refresh.lock().map_err(|_| {
                RedundancyError::Unreadable {
                    detail: "maintenance catalog clock poisoned".to_owned(),
                }
            })?;
            last.is_none_or(|value| value.elapsed() >= Duration::from_secs(10))
        };
        if refresh {
            let mut entries: Vec<PublishedLayout> = state
                .redundancy_entries()
                .filter_map(|(key, _)| {
                    let asset = key.to_asset().ok()?;
                    kivi_control::redundancy::published_layout(&state, &asset)
                })
                .collect();
            entries.sort_by_key(|entry| entry.layout.asset);
            let mut cache =
                self.maintenance_catalog
                    .lock()
                    .map_err(|_| RedundancyError::Unreadable {
                        detail: "maintenance catalog lock poisoned".to_owned(),
                    })?;
            *cache = Arc::new(entries);
            let mut clock = self.maintenance_catalog_refresh.lock().map_err(|_| {
                RedundancyError::Unreadable {
                    detail: "maintenance catalog clock poisoned".to_owned(),
                }
            })?;
            *clock = Some(Instant::now());
        }
        let entries = self
            .maintenance_catalog
            .lock()
            .map_err(|_| RedundancyError::Unreadable {
                detail: "maintenance catalog lock poisoned".to_owned(),
            })?
            .clone();
        let report = self
            .fabric
            .maintenance_tick_at(entries.as_slice(), &view, foreground_pressure)
            .await?;
        let anti_entropy_owner = view
            .iter()
            .filter(|node| node.health.serves_reads())
            .map(|node| node.id)
            .min();
        let anti_entropy_due = self
            .last_anti_entropy
            .lock()
            .map_err(|_| RedundancyError::Unreadable {
                detail: "anti-entropy clock poisoned".to_owned(),
            })?
            .is_none_or(|last| last.elapsed() >= Duration::from_secs(30));
        if anti_entropy_due && anti_entropy_owner == Some(self.node.node()) {
            let _ = self.fabric.anti_entropy(entries.as_slice(), &view).await;
            if let Ok(mut last) = self.last_anti_entropy.lock() {
                *last = Some(Instant::now());
            }
        }
        Ok(report)
    }

    /// until at least `eligible_min` nodes accept new fragments.
    ///
    /// Joins are a normal startup race: the registry admits nodes as
    /// `Joining` and the reconciler activates them shortly after. Rather
    /// than failing a protect that raced activation (and would otherwise
    /// refuse to place independent copies), the view waits (bounded by
    /// [`CoordinatorConfig::control_wait_timeout`], polled at
    /// [`CoordinatorConfig::control_poll_interval`]) until enough
    /// independent holders exist for the requested protection. A cluster
    /// that genuinely cannot satisfy the request still fails closed, and
    /// the wait never blocks a consensus thread (this is an admin-task
    /// future; control reads are reactor-agnostic).
    async fn view_min(
        &self,
        eligible_min: usize,
    ) -> Option<(ControlState, Vec<kivi_redundancy::NodeDescriptor>)> {
        let deadline = Instant::now() + self.config.control_wait_timeout;
        let wanted = eligible_min.max(1);
        loop {
            if let Some(state) = self.node.control_state().await {
                let records: Vec<kivi_control::NodeRecord> = state.nodes().cloned().collect();
                let view = kivi_control::redundancy::descriptors_from_registry(&records);
                if view.iter().filter(|node| node.health.accepts_new()).count() >= wanted {
                    return Some((state, view));
                }
            }
            if Instant::now() >= deadline {
                return None;
            }
            // Kivi's own async runtime: no Tokio in this crate (the
            // consensus plane is Compio-only by architecture).
            compio::time::sleep(self.config.control_poll_interval).await;
        }
    }

    /// Placement view with no minimum (one eligible node is enough).
    async fn view(&self) -> Option<(ControlState, Vec<kivi_redundancy::NodeDescriptor>)> {
        self.view_min(1).await
    }

    /// Computes the next per-asset control generation (fencing starts at 1).
    fn next_gen(current: Option<PublishedLayout>) -> u64 {
        current.map_or(1, |record| {
            record.control_generation.saturating_add(1).max(1)
        })
    }

    /// Builds the asset identity for raw bytes under `kind`, verifying an
    /// optional expected hash when provided (mismatch fails closed).
    fn asset_for_bytes(
        kind: u8,
        domain: u64,
        bytes: &[u8],
        expect_hash: Option<[u8; 32]>,
    ) -> Result<AssetId, RedundancyError> {
        let asset_kind = AssetKind::from_u8(kind)
            .map_err(|_| RedundancyError::invalid_asset(format!("unknown asset kind {kind}")))?;
        let computed = match asset_kind {
            AssetKind::Chunk => {
                let security = SecurityDomainId::from_u64(domain);
                *kivi_codec::integrity::chunk_id(security, bytes).as_bytes()
            }
            AssetKind::ChunkManifest => {
                let security = SecurityDomainId::from_u64(domain);
                *kivi_codec::integrity::manifest_id(security, bytes).as_bytes()
            }
            AssetKind::CheckpointBand
            | AssetKind::CheckpointManifest
            | AssetKind::CheckpointDedup => {
                if domain != 0 {
                    return Err(RedundancyError::invalid_asset(
                        "checkpoint assets carry domain 0",
                    ));
                }
                kivi_codec::integrity::blake3_256(bytes)
            }
            AssetKind::GenericImmutable => kivi_codec::integrity::blake3_256(bytes),
        };
        if let Some(expect) = expect_hash
            && expect != computed
        {
            return Err(RedundancyError::VerificationFailed {
                detail: "provided hash disagrees with recomputed content hash".to_owned(),
            });
        }
        AssetId::new(asset_kind, domain_for(asset_kind, domain), computed)
            .map_err(|_| RedundancyError::invalid_asset("asset hash is the zero sentinel"))
    }

    /// Protects raw bytes under a tolerance intent (`tolerance + 1` copies
    /// for small assets, erasure coding for large cold ones via the
    /// baseline planner). Verified before publish; the old layout serves
    /// until the replacement commits.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] on oversize images, invalid intents,
    /// unsatisfiable placement, verification failures, or control-commit
    /// timeouts.
    pub async fn protect_bytes(
        &self,
        kind: u8,
        domain: u64,
        bytes: &[u8],
        tolerance: u8,
    ) -> Result<PublishedLayout, RedundancyError> {
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > self.config.max_bytes {
            return Err(RedundancyError::Overloaded {
                detail: format!("image holds {} bytes", bytes.len()),
            });
        }
        let asset_id = Self::asset_for_bytes(kind, domain, bytes, None)?;
        let asset = InformationAsset::new(asset_id, bytes.len() as u64);
        let mut intent = RedundancyIntent::best_effort();
        intent.tolerance = tolerance;
        // Independent copies need independent active nodes: wait for
        // `tolerance + 1` placement-eligible members before planning (the
        // deterministic planner refuses to publish colliding copies).
        let wanted = usize::from(tolerance).saturating_add(1);
        let Some((_state, view)) = self.view_min(wanted).await else {
            return Err(RedundancyError::Unreachable {
                detail: format!("fewer than {wanted} nodes accept new fragments"),
            });
        };
        let current = self.fabric_catalog_get(&asset_id).await;
        let control_gen = Self::next_gen(current);
        self.fabric
            .protect(asset, bytes, &intent, &view, control_gen)
            .await
    }

    /// Protects one sidecar chunk: reads the bytes from the local sidecar
    /// store first (proving the chunk path), then protects them as a chunk
    /// asset under the tolerance intent.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when the chunk is absent, over the length
    /// bound, or protection fails.
    pub async fn protect_chunk(
        &self,
        chunk: ChunkId,
        domain: SecurityDomainId,
        len: u64,
        tolerance: u8,
    ) -> Result<PublishedLayout, RedundancyError> {
        if len > self.config.max_bytes {
            return Err(RedundancyError::Overloaded {
                detail: format!("chunk declares {len} bytes"),
            });
        }
        let bytes = self
            .node
            .sidecar()
            .read_chunk(chunk)
            .await
            .map_err(|error| RedundancyError::Chunk {
                detail: format!("sidecar chunk unreadable: {error}"),
            })?;
        if bytes.len() as u64 != len {
            return Err(RedundancyError::VerificationFailed {
                detail: format!(
                    "sidecar chunk length {} disagrees with declared {len}",
                    bytes.len()
                ),
            });
        }
        self.protect_bytes(AssetKind::Chunk.as_u8(), domain.as_u64(), &bytes, tolerance)
            .await
    }

    /// Protects a canonical chunk manifest already staged in the local
    /// sidecar store. The sidecar durability gate remains authoritative; this
    /// is an additional physical protection layer and is safe to retry.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when the manifest cannot be read or
    /// protection cannot publish.
    pub async fn protect_manifest(
        &self,
        manifest: kivi_types::ManifestId,
        domain: SecurityDomainId,
        len: u64,
    ) -> Result<PublishedLayout, RedundancyError> {
        let bytes = self
            .node
            .sidecar()
            .read_manifest(manifest)
            .await
            .map_err(|error| RedundancyError::Chunk {
                detail: format!("sidecar manifest unreadable: {error}"),
            })?;
        if bytes.len() as u64 != len {
            return Err(RedundancyError::VerificationFailed {
                detail: format!(
                    "manifest length {} disagrees with declared {len}",
                    bytes.len()
                ),
            });
        }
        self.protect_bytes(AssetKind::ChunkManifest.as_u8(), domain.as_u64(), &bytes, 1)
            .await
    }

    /// Protects a canonical manifest using its staged byte length.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when the manifest is absent or cannot be
    /// published.
    pub async fn protect_manifest_auto(
        &self,
        manifest: kivi_types::ManifestId,
        domain: SecurityDomainId,
    ) -> Result<PublishedLayout, RedundancyError> {
        let bytes = self
            .node
            .sidecar()
            .read_manifest(manifest)
            .await
            .map_err(|error| RedundancyError::Chunk {
                detail: format!("sidecar manifest unreadable: {error}"),
            })?;
        self.protect_bytes(AssetKind::ChunkManifest.as_u8(), domain.as_u64(), &bytes, 1)
            .await
    }

    /// Reads one asset through its current layout: exact required pieces
    /// first, every piece content-verified, the image content-verified
    /// before exposure. Returns the bytes plus whether any probe or fetch
    /// failed along the way.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when no layout exists, too few pieces
    /// survive, or verification fails (never wrong bytes).
    pub async fn read_bytes(&self, asset: &AssetId) -> Result<(Vec<u8>, bool), RedundancyError> {
        let current = self.fabric_catalog_get(asset).await;
        let Some(current) = current else {
            return Err(RedundancyError::MissingFragment {
                asset: asset.to_string(),
                generation: 0,
                index: 0,
            });
        };
        let info = InformationAsset::new(*asset, current.layout.logical_len);
        self.fabric.read(info).await
    }

    /// Assesses installed health over actually-healthy holders (not desired
    /// placement): the operator sees exactly why an asset is
    /// Healthy/Degraded/Critical/Unrecoverable.
    pub async fn health(&self, asset: &AssetId) -> Option<CoordinatorHealth> {
        let (_state, view) = self.view().await?;
        let assessed = self.fabric.assess(asset, &view).await?;
        let current = self.fabric_catalog_get(asset).await?;
        Some(CoordinatorHealth {
            asset: *asset,
            generation: current.layout.generation,
            control_generation: current.control_generation,
            health: assessed.assessment.health,
            usable: assessed.assessment.usable,
            required: assessed.assessment.required,
            total: assessed.assessment.total,
            survivable: assessed.assessment.survivable,
            reconstructable: assessed.assessment.reconstructable,
            desired: assessed
                .desired
                .fragments
                .iter()
                .map(|placement| (placement.index, placement.node))
                .collect(),
            actual: assessed.actual,
        })
    }

    /// Repairs a degraded asset: reconstructs explicit deficits from
    /// minimum pieces and reinstalls them on the layout-named holders.
    /// Idempotent and resumable.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when no layout exists, reconstruction
    /// fails, or installs cannot verify.
    pub async fn repair(&self, asset: &AssetId) -> Result<RepairReport, RedundancyError> {
        let Some((_state, view)) = self.view().await else {
            return Err(RedundancyError::Unreachable {
                detail: "no local control image".to_owned(),
            });
        };
        self.fabric.repair(asset, &view).await
    }

    /// Drains one node across every asset that names it: reads through the
    /// current layout, builds a full replacement generation on the view
    /// without the node, publishes, and drops the old fragments
    /// best-effort. Assets already clear of the node are skipped.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when no control image exists or a
    /// replacement cannot publish (the old layout keeps serving).
    pub async fn drain(&self, node: NodeId) -> Result<DrainReport, RedundancyError> {
        let Some((state, view)) = self.view().await else {
            return Err(RedundancyError::Unreachable {
                detail: "no local control image".to_owned(),
            });
        };
        let filtered: Vec<kivi_redundancy::NodeDescriptor> = view
            .iter()
            .filter(|item| item.id != node)
            .cloned()
            .collect();
        let mut moved: u64 = 0;
        let mut assets: Vec<AssetId> = Vec::new();
        let entries: Vec<(AssetId, PublishedLayout)> = state
            .redundancy_entries()
            .filter_map(|(key, _)| {
                key.to_asset().ok().and_then(|asset| {
                    kivi_control::redundancy::published_layout(&state, &asset)
                        .map(|published| (asset, published))
                })
            })
            .collect();
        for (asset, current) in entries {
            if !current
                .layout
                .fragments
                .iter()
                .any(|record| record.node == node)
            {
                continue;
            }
            let control_gen = current.control_generation.saturating_add(1).max(1);
            match self
                .fabric
                .drain(&asset, node, &filtered, control_gen)
                .await
            {
                Ok(_) => {
                    moved = moved.saturating_add(1);
                    assets.push(asset);
                }
                Err(error) => {
                    // Placement can fail closed on small clusters (for
                    // example three distinct holders cannot be re-placed
                    // without the drained node): surface the first failure
                    // instead of silently skipping the asset.
                    return Err(error);
                }
            }
        }
        assets.sort();
        Ok(DrainReport { moved, assets })
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
        asset: &AssetId,
        target: SchemeParams,
    ) -> Result<PublishedLayout, RedundancyError> {
        target.validate()?;
        let current = self.fabric_catalog_get(asset).await;
        let Some(current) = current else {
            return Err(RedundancyError::MissingFragment {
                asset: asset.to_string(),
                generation: 0,
                index: 0,
            });
        };
        let info = InformationAsset::new(*asset, current.layout.logical_len);
        let (bytes, _) = self.fabric.read(info).await?;
        let Some((_state, view)) = self.view().await else {
            return Err(RedundancyError::Unreachable {
                detail: "no local control image".to_owned(),
            });
        };
        let control_gen = current.control_generation.saturating_add(1).max(1);
        self.fabric
            .transition(info, &bytes, target, &view, control_gen)
            .await
    }

    /// Heals the local sidecar cache from the fabric: reads through the
    /// current layout, then stages and syncs the bytes into the local
    /// sidecar store. Never proposes Raft entries and never acknowledges
    /// replication: only the cache heals, and the gate re-verifies before
    /// any future acknowledgement.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when no layout exists, reconstruction
    /// fails, or the sidecar stage/sync fails.
    pub async fn restore_chunk(
        &self,
        chunk: ChunkId,
        domain: SecurityDomainId,
        len: u64,
    ) -> Result<u64, RedundancyError> {
        let asset = InformationAsset::new(AssetId::chunk(chunk, domain), len);
        let (bytes, _) = self.fabric.read(asset).await?;
        asset
            .verify_bytes(&bytes)
            .map_err(|_| RedundancyError::VerificationFailed {
                detail: "restored chunk fails content verification".to_owned(),
            })?;
        self.node
            .sidecar()
            .stage_chunk(chunk, bytes.clone())
            .await
            .map_err(|error| RedundancyError::Chunk {
                detail: format!("sidecar stage failed: {error}"),
            })?;
        self.node
            .sidecar()
            .sync()
            .await
            .map_err(|error| RedundancyError::Chunk {
                detail: format!("sidecar sync failed: {error}"),
            })?;
        Ok(bytes.len() as u64)
    }

    /// Lists every published asset with its generations (the control plane
    /// owns enumeration; the fabric catalog lists nothing by design).
    pub async fn catalog_list(&self) -> Vec<CatalogEntry> {
        let Some(state) = self.node.control_state().await else {
            return Vec::new();
        };
        let mut out: Vec<CatalogEntry> = state
            .redundancy_entries()
            .filter_map(|(key, _)| {
                let asset = key.to_asset().ok()?;
                let published = kivi_control::redundancy::published_layout(&state, &asset)?;
                Some(CatalogEntry {
                    asset,
                    control_generation: published.control_generation,
                    generation: published.layout.generation,
                    params: published.layout.params,
                    logical_len: published.layout.logical_len,
                })
            })
            .collect();
        out.sort_by_key(|entry| (entry.asset, entry.control_generation));
        out
    }

    /// Snapshots extended operator metrics: live fabric counters plus
    /// local orphans, per-holder desired bytes, and the current
    /// asset-to-generation map.
    pub async fn metrics(&self) -> CoordinatorMetrics {
        let snapshot = self.fabric.metrics().snapshot();
        let orphans: u64 = self
            .node
            .fragments_arc()
            .map_or(0, |store| store.orphans().len() as u64);
        let Some(state) = self.node.control_state().await else {
            return CoordinatorMetrics {
                snapshot,
                orphans,
                by_node_bytes: Vec::new(),
                current_gens: Vec::new(),
                logical_bytes: 0,
                physical_bytes: 0,
                replicated_assets: 0,
                coded_assets: 0,
                maintenance: self.fabric.maintenance_snapshot(),
            };
        };
        let mut bytes_by_node: std::collections::BTreeMap<u64, u64> =
            std::collections::BTreeMap::new();
        let mut gens: Vec<(AssetId, u64)> = Vec::new();
        let mut logical_bytes = 0u64;
        let mut physical_bytes = 0u64;
        let mut replicated_assets = 0u64;
        let mut coded_assets = 0u64;
        for (key, _) in state.redundancy_entries() {
            let Ok(asset) = key.to_asset() else {
                continue;
            };
            let Some(published) = kivi_control::redundancy::published_layout(&state, &asset) else {
                continue;
            };
            gens.push((asset, published.control_generation));
            logical_bytes = logical_bytes.saturating_add(published.layout.logical_len);
            match published.layout.params {
                kivi_redundancy::SchemeParams::ReedSolomon(_) => coded_assets += 1,
                kivi_redundancy::SchemeParams::Replication(_) => replicated_assets += 1,
            }
            for record in &published.layout.fragments {
                physical_bytes = physical_bytes.saturating_add(record.stored_len);
                let entry = bytes_by_node.entry(record.node.as_u64()).or_insert(0);
                *entry = entry.saturating_add(record.stored_len);
            }
        }
        gens.sort();
        CoordinatorMetrics {
            snapshot,
            orphans,
            by_node_bytes: bytes_by_node.into_iter().collect(),
            current_gens: gens,
            logical_bytes,
            physical_bytes,
            replicated_assets,
            coded_assets,
            maintenance: self.fabric.maintenance_snapshot(),
        }
    }

    /// Snapshots background-lane counters for operators.
    #[must_use]
    pub fn lane_stats(&self) -> LaneStats {
        self.lane
            .lock()
            .map(|lane| lane.stats())
            .unwrap_or_default()
    }

    /// Sweeps orphan fragments (staged without authority) across the whole
    /// cluster.
    ///
    /// Orphans may leak but are never authoritative; sweeping only reclaims
    /// disk. Every known holder is asked, so leaked bytes on remote nodes do
    /// not accumulate. An unreachable holder is reported, not fatal: hygiene
    /// never blocks on a dead node, and each RPC is bounded by the
    /// coordinator's RPC timeout.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when the placement view is unavailable or
    /// fragments are disabled on this node.
    pub async fn sweep_orphans(&self) -> Result<kivi_redundancy::SweepReport, RedundancyError> {
        if self.node.fragments_arc().is_none() {
            return Err(RedundancyError::Unreachable {
                detail: "fragments disabled on this node".to_owned(),
            });
        }
        let Some((_state, view)) = self.view().await else {
            return Err(RedundancyError::Unreachable {
                detail: "no local control image".to_owned(),
            });
        };
        self.fabric.sweep_orphans(&view).await
    }

    /// Reads the catalog through a fresh control snapshot (single image,
    /// no wait): the fencing generation for the next publish.
    async fn fabric_catalog_get(&self, asset: &AssetId) -> Option<PublishedLayout> {
        let state = self.node.control_state().await?;
        kivi_control::redundancy::published_layout(&state, asset)
    }
}

/// Domain accompanying an asset identity: checkpoint families bind no
/// domain (their hash already binds provenance); every other family keeps
/// the caller's domain.
fn domain_for(kind: AssetKind, domain: u64) -> u64 {
    match kind {
        AssetKind::CheckpointBand | AssetKind::CheckpointManifest | AssetKind::CheckpointDedup => 0,
        _ => domain,
    }
}

/// Local fragment-store handle for coordinators that need `'static`
/// ownership (lane jobs). Kept here (not on the node front) so the only
/// new node-front surface stays the two documented accessors.
#[allow(dead_code)]
fn store_arc(node: &ConsensusNode) -> Option<Arc<LocalFragmentStore>> {
    node.fragments_arc()
}
