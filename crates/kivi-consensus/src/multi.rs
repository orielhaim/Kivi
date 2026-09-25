//! Multi-tablet consensus node: many tablet groups on fixed workers.
//!
//! The production physical Multi-Raft architecture (RFC §60):
//!
//! ```text
//! ConsensusNode front (Send + Sync)
//!   ├── worker 0 (one Compio reactor/thread)
//!   │     ├── TabletReplica(group 1) ── Raft + GroupRaftStore + machine
//!   │     └── TabletReplica(group 1+W)
//!   ├── worker 1 ── TabletReplica(group 2), …
//!   └── …
//!   ├── one shared H3 peer mesh (control + bulk per node pair, all groups)
//!   └── one shared WAL writer (cross-group fsync batching, all groups)
//! ```
//!
//! ## Ownership
//!
//! A `Raft` handle never migrates between threads: each worker pins its
//! groups' handles to its own Compio reactor, and cross-thread callers
//! communicate through bounded worker ingress channels. No locks wrap
//! `Raft` internals. Thread count scales with workers plus fixed services
//! (mesh, shared writer, sidecar blocking) — never with tablet count.
//!
//! ## Shared machinery
//!
//! Group behavior (proposals, barriers, peer RPCs, snapshots, preflight)
//! is NOT reimplemented here: every replica is an owner context over a
//! lightweight [`GroupRaftStore`], sharing one implementation with the
//! single-group node. This module owns only multi-group concerns: worker
//! assignment, the group registry, the shared mesh thread, worker
//! threads, lifecycle, and fan-out diagnostics.
//!
//! ## Dynamic placement
//!
//! The static topology seeds first-boot formation (initial members and
//! their tablet assignments, usually all-nodes-host-all-tablets);
//! afterwards the replicated control plane owns desired placement and
//! each tablet's Raft membership owns actual voters, with independent
//! Raft leadership per tablet. Replicas come and go at runtime through
//! learner replication plus joint-consensus membership changes; APIs
//! never assume equal replica sets.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use kivi_redundancy::LocalFragmentStore;
use kivi_state::Operation;
use kivi_types::{
    ClusterId, IdempotencyKey, MutationIdentity, NamespaceId, NodeId, ReadContract,
    TabletAuthority, TabletEpoch, TabletId, WallTimestamp, WriteGuardGeneration,
};

use crate::cluster::Bootstrap;
use crate::cluster::ClusterTopology;
use crate::gate::SidecarGate;
use crate::node::{
    BootstrapInputs, NodeOpenError, NodeStatus, OwnerCtx, OwnerRequest, ProposeError, ReadError,
    bootstrap_group, classify_from_store, owner_call_on, process_ticks, propose_caller_side,
    spawn_raft,
};
use crate::peer::{
    PeerControlForwardRequest, PeerControlForwardResponse, PeerRequest, PeerResponse, PeerRpcError,
};
use crate::router::PeerRouter;
use crate::shared::{
    GroupRaftStore, SharedDurabilityConfig, SharedRaftDurability, dispatch_by_group,
};
use crate::sidecar::SidecarStore;
use crate::state_machine::ReplicatedStateMachine;
use crate::tls::NodeCert;
use crate::transport::{PeerHandler, PeerTransport, TlsMaterial, TransportConfig};
use crate::types::{ConsensusError, ConsensusGroupId, ConsensusLogIndex, ControlProposeError};
use crate::worker::worker_for_tablet;

/// Depth of each worker ingress queue (same bound as the single-group
/// owner queue: burst memory only, honest backpressure when full).
const WORKER_QUEUE: usize = 256;

/// Sidecar cache bound shared by the node-wide sidecar store (matches the
/// single-group node: immutable entries, safe eviction at any time).
const SIDECAR_CACHE_BYTES: u64 = 256 * 1024 * 1024;

/// Multi-tablet node construction parameters. Everything is validated
/// before serving: topology incoherence, identity conflicts, tablet-set
/// disagreement with durable state, and corrupt history all fail `open`,
/// never mid-operation.
#[derive(Debug, Clone)]
pub struct MultiNodeConfig {
    /// Data-directory root (shared WAL lane, per-tablet state machines,
    /// sidecars live beneath it; the process lock is held for the node's
    /// life).
    pub data_dir: PathBuf,
    /// Namespace served.
    pub namespace: NamespaceId,
    /// Tablets to serve at open (unique, nonzero; empty is valid for a
    /// joining node that gains replicas through migration). The local
    /// node must be assigned every listed tablet in the static
    /// topology; tombstoned and traceless tablets are skipped at open
    /// and arrive later through the control plane.
    pub tablets: Vec<TabletId>,
    /// This process's node identity (operator-declared; verified against
    /// the data directory, adopted when fresh).
    pub local: NodeId,
    /// Static cluster definition (identity, members, per-tablet
    /// assignments).
    pub topology: ClusterTopology,
    /// WAL segment rotation target in bytes.
    pub segment_target_bytes: u64,
    /// Peer transport tuning.
    pub transport: TransportConfig,
    /// Expected peer certificates (trust anchors) by node for QUIC/TLS
    /// verification. Every other member must be present; the transport
    /// refuses to open otherwise. Empty with `insecure_peer_tls` (lab
    /// only) or when no peers are configured yet.
    pub peer_certs: HashMap<NodeId, Vec<u8>>,
    /// Test-only escape hatch for `kivi-lab` (ephemeral ports and fresh
    /// data directories per run make static pins impractical there).
    /// Never set in production.
    pub insecure_peer_tls: bool,
    /// Whether leaders pre-distribute chunked sidecars before proposing
    /// the tiny root (`true` in production; `false` forces the
    /// append-gate fallback path).
    pub preflight_enabled: bool,
    /// Deployment lease-timing contract (bounded clock-rate drift plus
    /// guard/lease/renew durations). The `RosterLease` backend is
    /// unavailable unless these validate; reads then use Lazy-ALR or the
    /// conservative barrier.
    pub lease_params: kivi_types::LeaseParams,
    /// Compio worker (reactor) count: tablet groups stripe across these
    /// deterministically ([`worker_for_tablet`]).
    pub worker_count: usize,
    /// Shared-writer batching bounds.
    pub durability: SharedDurabilityConfig,
    /// System control group participation. `None` means this node hosts
    /// no control replica (pure data host); `Some` joins the replicated
    /// control plane either by forming it (fresh directories with no
    /// seeds) or by waiting for the control leader to add it as learner
    /// (fresh directories with seeds, or any restart recovering durable
    /// control state — restarts never re-initialize).
    pub control: Option<ControlGroupConfig>,
    /// Redundancy-fragment store root. `Some` opens a node-local
    /// [`LocalFragmentStore`] there (recovering indexed state) and serves
    /// project-owned fragment RPCs from every replica; `None` disables
    /// fragments (RPCs are refused as shutting down) so existing
    /// deployments and tests keep working unchanged.
    pub fragment_store_root: Option<PathBuf>,
}

/// How this node participates in the system control group.
#[derive(Debug, Clone)]
pub struct ControlGroupConfig {
    /// Control voter set at formation (seed membership for fresh
    /// formation; ignored on restart, which recovers durable state).
    pub voters: BTreeSet<u64>,
    /// Dialable peer address per control voter.
    pub peer_addrs: std::collections::BTreeMap<u64, String>,
    /// Control endpoints that prove the cluster already exists. Empty on
    /// the founding voters (fresh directories initialize); set on a
    /// joining node (fresh directories wait for `add_learner` instead of
    /// forking history with a second initialize).
    pub seeds: Vec<std::net::SocketAddr>,
}

/// Multi-tablet consensus node: a `Send + Sync` front over fixed Compio
/// workers driving many tablet groups. Every method below is an ordinary
/// `Send` future on any runtime.
pub struct ConsensusNode {
    workers: Vec<async_channel::Sender<OwnerRequest>>,
    /// Group-to-worker routing. Mutated when dynamic replicas are
    /// created or retired; the deterministic
    /// [`worker_for_tablet`](crate::worker::worker_for_tablet) mapping
    /// stays the assignment rule, this map is its materialization.
    group_to_worker: Mutex<HashMap<ConsensusGroupId, usize>>,
    /// Local state machines per hosted tablet. Mutated on dynamic
    /// replica creation/retirement (the caller-side propose/read path
    /// needs the machine without hopping to the worker).
    machines: Mutex<HashMap<TabletId, ReplicatedStateMachine>>,
    /// Replica authorities per hosted tablet (same lifecycle as
    /// [`machines`](Self::machines)).
    authorities: Mutex<HashMap<TabletId, TabletAuthority>>,
    sidecar: SidecarStore,
    /// Node-local redundancy fragment store (`None` when disabled).
    /// Shared by every replica (content-addressed, group-independent).
    fragments: Option<Arc<LocalFragmentStore>>,
    preflight: Arc<crate::preflight::PreflightMetrics>,
    durability: SharedRaftDurability,
    /// Shared H3 mesh handle: dynamic peer admission (`add_peer_dial`)
    /// and trust (`trust_peer`) run synchronously from any thread.
    mesh: PeerTransport,
    namespace: NamespaceId,
    local: NodeId,
    cluster: ClusterId,
    incarnation: kivi_types::NodeIncarnation,
    /// Caller-side consistency hub: evidence, lease engines, strong
    /// cache, providers, metrics shared by every tablet group on this
    /// node. Shared with the workers (which drive lease engines through
    /// it). Memory-held by construction — a restart starts empty.
    hub: std::sync::Arc<crate::consistency::ConsistencyHub>,
    /// Lazy-ALR batch coordinators per hosted tablet.
    alr: Mutex<HashMap<TabletId, std::sync::Arc<crate::alr::AlrCoordinator>>>,
    /// User tablets hosted (excludes the system control group, which is
    /// tracked in the maps above under its reserved id).
    tablets: Mutex<Vec<TabletId>>,
    worker_count: usize,
    peer_addr: std::net::SocketAddr,
    data_dir: PathBuf,
    worker_threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
    mesh_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    mesh_shutdown: Mutex<Option<futures::channel::oneshot::Sender<()>>>,
    _dir_guard: kivi_durability::OpenDir,
}

impl std::fmt::Debug for ConsensusNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tablets = self
            .tablets
            .lock()
            .map_or(usize::MAX, |tablets| tablets.len());
        f.debug_struct("ConsensusNode")
            .field("node", &self.local)
            .field("cluster", &self.cluster)
            .field("tablets", &tablets)
            .field("workers", &self.worker_count)
            .field("peer", &self.peer_addr)
            .finish_non_exhaustive()
    }
}

/// Per-group wiring resolved before workers spawn: voters plus dialable
/// peer addresses from the static topology.
#[derive(Debug, Clone)]
struct GroupWiring {
    voters: BTreeSet<u64>,
    peer_addrs: std::collections::BTreeMap<u64, String>,
}

/// Group registry: the receiving side of the shared H3 mesh. Routes
/// inbound Raft RPCs directly to the owning worker by group; the network
/// layer knows only target group, RPC type, and binary body — never
/// tablet state-machine semantics.
///
/// Dynamically created replicas (migration targets) have no entry in the
/// snapshot `routes` map taken at open. The fallback computes the same
/// deterministic [`worker_for_tablet`](crate::worker::worker_for_tablet)
/// assignment the front uses, so learner replication reaches the owning
/// worker even before any map update propagates.
#[derive(Clone)]
struct GroupRegistry {
    routes: Arc<HashMap<ConsensusGroupId, async_channel::Sender<OwnerRequest>>>,
    /// All worker ingress channels in worker order (deterministic
    /// fallback for groups created after open).
    fallback: Vec<async_channel::Sender<OwnerRequest>>,
    /// Worker count for the deterministic fallback mapping.
    worker_count: usize,
    /// Worker serving group-independent bulk traffic (manifest/chunk/
    /// fragment decode with a zero tablet): owns the smallest tablet, so
    /// it always hosts at least one replica serving the shared sidecar
    /// and fragment stores.
    bulk: async_channel::Sender<OwnerRequest>,
}

impl PeerHandler for GroupRegistry {
    fn handle(
        &self,
        from: NodeId,
        group: ConsensusGroupId,
        request: PeerRequest,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<PeerResponse, PeerRpcError>> + Send + '_>,
    > {
        let routes = Arc::clone(&self.routes);
        let fallback = self.fallback.clone();
        let worker_count = self.worker_count;
        let bulk = self.bulk.clone();
        Box::pin(async move {
            let tx = if group == ConsensusGroupId::redundancy_sentinel() {
                match &request {
                    PeerRequest::Manifest(_) | PeerRequest::Chunk(_) | PeerRequest::Fragment(_) => {
                        Some(bulk)
                    }
                    _ => None,
                }
            } else if let Some(tx) = routes.get(&group).cloned() {
                Some(tx)
            } else {
                // Dynamically created group: deterministic owner.
                let tablet = group.tablet();
                (tablet.as_u64() != 0)
                    .then(|| {
                        let worker = worker_for_tablet(tablet, worker_count);
                        fallback.get(worker).cloned()
                    })
                    .flatten()
            };
            let tx = tx.ok_or_else(|| PeerRpcError {
                detail: format!("group {group} not served here"),
            })?;
            let (reply, rx) = futures::channel::oneshot::channel();
            tx.send(OwnerRequest::Serve {
                from,
                group,
                request,
                reply,
            })
            .await
            .map_err(|_| PeerRpcError {
                detail: "consensus worker shut down".to_owned(),
            })?;
            rx.await.map_err(|_| PeerRpcError {
                detail: "consensus worker shut down".to_owned(),
            })?
        })
    }
}

impl ConsensusNode {
    /// Opens the node: validates topology and tablet set, opens the data
    /// directory, recovers every group through the shared WAL in one pass,
    /// opens per-tablet state machines and the node-wide sidecar store,
    /// starts the shared mesh plus one worker reactor per configured
    /// worker, bootstraps every group (fresh formation installs membership
    /// exactly once; restarts recover and never re-bootstrap), and serves.
    ///
    /// # Errors
    ///
    /// Returns [`NodeOpenError`] on topology, identity, tablet-set, mesh,
    /// or storage failures. Restarting with conflicting durable
    /// identity/membership/tablets fails loudly; a fresh directory forms
    /// every group exactly once.
    ///
    /// Name of the tombstone file marking a retired local replica
    /// (`<data_dir>/consensus-sm/<tablet>/TOMBSTONE`, content: fencing
    /// placement generation `u64` little-endian).
    pub const TOMBSTONE_FILE: &'static str = "TOMBSTONE";

    /// Reads the tombstone generation for one tablet, if retired.
    #[must_use]
    pub fn tombstone_generation(data_dir: &std::path::Path, tablet: TabletId) -> Option<u64> {
        let bytes = std::fs::read(
            data_dir
                .join("consensus-sm")
                .join(tablet.as_u64().to_string())
                .join(Self::TOMBSTONE_FILE),
        )
        .ok()?;
        if bytes.len() < 8 {
            return None;
        }
        Some(u64::from_le_bytes(bytes[..8].try_into().unwrap_or([0; 8])))
    }

    /// Returns the tablets retired locally (tombstone present).
    fn tombstoned_tablets(data_dir: &std::path::Path) -> BTreeSet<TabletId> {
        let mut out = BTreeSet::new();
        let Ok(entries) = std::fs::read_dir(data_dir.join("consensus-sm")) else {
            return out;
        };
        for entry in entries.flatten() {
            if !entry.path().is_dir() {
                continue;
            }
            let name = entry.file_name().to_str().unwrap_or_default().to_owned();
            let Ok(id) = name.parse::<u64>() else {
                continue;
            };
            if id == 0 || id == crate::types::CONTROL_TABLET_RAW {
                continue;
            }
            let tablet = TabletId::from_u64(id);
            if Self::tombstone_generation(data_dir, tablet).is_some() {
                out.insert(tablet);
            }
        }
        out
    }

    /// Opens the node: validates topology and tablet set, opens the data
    /// directory, recovers every group through the shared WAL in one pass,
    /// opens per-tablet state machines and the node-wide sidecar store,
    /// starts the shared mesh plus one worker reactor per configured
    /// worker, bootstraps every group (fresh formation installs membership
    /// exactly once; restarts recover and never re-bootstrap), and serves.
    ///
    /// Tombstoned (retired) tablets are skipped, never resurrected; the
    /// system control replica opens alongside tablet groups when
    /// [`MultiNodeConfig::control`] participates.
    ///
    /// # Errors
    ///
    /// Returns [`NodeOpenError`] on topology, identity, tablet-set, mesh,
    /// or storage failures. Restarting with conflicting durable
    /// identity/membership/tablets fails loudly; a fresh directory forms
    /// every group exactly once.
    ///
    /// # Panics
    ///
    /// Panics when an owner-handle lock is poisoned (a lifecycle bug,
    /// never a runtime condition).
    #[allow(clippy::too_many_lines)]
    pub async fn open(config: MultiNodeConfig) -> Result<Self, NodeOpenError> {
        use NodeOpenError as Fault;
        let tablets = validated_tablets(&config)?;
        // Retired replicas stay retired across restarts: tombstoned
        // tablets are skipped (never resurrected from durable state),
        // and exempted from the tablet-set/WAL agreement checks below
        // (their records linger conservatively in the shared WAL until
        // rotation, which is retention — not resurrection).
        let tombstoned = Self::tombstoned_tablets(&config.data_dir);
        let live_tablets: Vec<TabletId> = tablets
            .iter()
            .copied()
            .filter(|tablet| !tombstoned.contains(tablet))
            .collect();
        // Data-directory identity: adopt the static topology identity on
        // first formation, advance the incarnation on every restart.
        let opened = kivi_durability::open_data_dir_with(
            &config.data_dir,
            Some(kivi_durability::NodeSeed {
                cluster: config.topology.cluster,
                node: config.local,
            }),
        )
        .map_err(|error| Fault::DataDir {
            reason: error.to_string(),
        })?;
        if opened.meta.cluster != config.topology.cluster || opened.meta.node != config.local {
            return Err(Fault::Bootstrap {
                reason: format!(
                    "directory identity (cluster {}, node {}) conflicts with topology (cluster {}, node {})",
                    opened.meta.cluster, opened.meta.node, config.topology.cluster, config.local,
                ),
            });
        }
        // Tablet-set disagreement fails loudly before any state opens —
        // except dynamic topology tablets (split children / merge targets
        // created via `ensure_group` after boot): state-machine tablets or
        // shared-WAL groups outside the static configured set recover as
        // served replicas (tombstoned groups stay excluded). A shrunk or
        // foreign static topology still fails: configured tablets missing
        // from durable state bootstrap fresh per group, and every node in
        // a static cluster shares the config, so fresh formation stays
        // coherent.
        let durable_tablets = existing_sm_tablets(&config.data_dir);
        let configured: BTreeSet<TabletId> = tablets.iter().copied().collect();
        let control_tablet = ConsensusGroupId::control().tablet();
        let mut extra: Vec<u64> = durable_tablets
            .difference(&configured)
            .filter(|tablet| **tablet != control_tablet && !tombstoned.contains(*tablet))
            .map(|tablet| tablet.as_u64())
            .collect();
        extra.sort_unstable();
        if !extra.is_empty() {
            tracing::info!(
                tablets = ?extra,
                "rejoin recovers dynamic topology tablets outside the static set"
            );
        }
        // Shared physical durability: one WAL lane recovered once; records
        // dispatch by group so N groups rebuild in a single recovery pass.
        let lane_identity = kivi_durability::LaneIdentity {
            cluster: opened.meta.cluster,
            node: opened.meta.node,
            incarnation: opened.meta.incarnation,
        };
        let (durability, recovered) = SharedRaftDurability::open(
            &config.data_dir,
            lane_identity,
            config.segment_target_bytes,
            config.durability,
        )
        .map_err(|error| Fault::LogStore {
            reason: error.to_string(),
        })?;
        // First boot ever (no durable trace anywhere — no state
        // machines, not even tombstoned or control ones, and no WAL
        // records): the only boot that may initialize data groups.
        // Every later boot is a rejoin — it recovers groups with
        // durable traces and NEVER initializes (initializing a group
        // the live cluster already runs would fork its history;
        // unknown groups arrive later through learner replication,
        // never a second initialize). Wiping a member's directory
        // defeats this (it looks like first boot): never wipe a live
        // member's directory.
        let first_boot = durable_tablets.is_empty() && recovered.is_empty();
        // Dynamic topology tablets (split children / merge targets created
        // via `ensure_group` after boot) leave durable WAL traces outside
        // the static configured set: recover them as served replicas
        // instead of refusing to open. Tombstoned groups stay excluded
        // (retired parents never resurrect); the removal oracle below
        // still fences anything the control plane dropped while down.
        let wal_extra: Vec<TabletId> = recovered
            .iter()
            .map(kivi_durability::RaftRecord::group)
            .filter(|group| {
                *group != control_tablet && !tablets.contains(group) && !tombstoned.contains(group)
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut live_tablets = live_tablets;
        for tablet in &wal_extra {
            if !live_tablets.contains(tablet) {
                tracing::info!(
                    tablet = tablet.as_u64(),
                    "rejoin recovers dynamic topology tablet outside the static set"
                );
                live_tablets.push(*tablet);
            }
        }
        live_tablets.sort_by_key(|tablet| tablet.as_u64());
        let mut groups: Vec<ConsensusGroupId> = live_tablets
            .iter()
            .map(|tablet| ConsensusGroupId::of_tablet(*tablet))
            .collect();
        // The control group shares the same recovery pass when this node
        // participates in the control plane.
        let control_group = config.control.is_some().then(ConsensusGroupId::control);
        if let Some(control) = control_group {
            groups.push(control);
        }
        let by_group = dispatch_by_group(&recovered, &groups);
        // Tablets served this boot: on first boot every configured
        // tablet initializes; on rejoin only tablets with a durable
        // trace recover (traceless tablets are skipped, never
        // initialized: the live cluster owns them, and this replica
        // joins later through learner replication).
        let served_tablets: Vec<TabletId> = live_tablets
            .iter()
            .copied()
            .filter(|tablet| {
                if first_boot
                    || durable_tablets.contains(tablet)
                    || recovered
                        .iter()
                        .any(|record| kivi_durability::RaftRecord::group(record) == *tablet)
                {
                    return true;
                }
                tracing::info!(
                    tablet = tablet.as_u64(),
                    "rejoin skips traceless tablet (joins later through learner replication)"
                );
                false
            })
            .collect();
        // Per-tablet logical views: group stores plus state machines plus
        // authorities plus per-group wiring from the static topology.
        let mut stores = HashMap::new();
        let mut machines = HashMap::new();
        let mut authorities = HashMap::new();
        let mut wirings = HashMap::new();
        // System control replica first: its snapshot-loaded image is the
        // desired-placement oracle fencing removed-while-down data
        // replicas below. Same stores, machines, and wiring as every
        // tablet group — one dedicated Raft group committing typed
        // control mutations through the shared envelope, mesh, and WAL.
        // The control tablet id is reserved and never a user tablet.
        if let Some(control_cfg) = &config.control {
            let group = ConsensusGroupId::control();
            let tablet = group.tablet();
            let authority =
                TabletAuthority::new(tablet, TabletEpoch::INITIAL, WriteGuardGeneration::INITIAL);
            let store = GroupRaftStore::from_recovered(
                config.namespace,
                group,
                &durability,
                &by_group[&group],
            )
            .map_err(|error| Fault::LogStore {
                reason: error.to_string(),
            })?;
            let machine =
                ReplicatedStateMachine::open(&config.data_dir, config.namespace, tablet, authority)
                    .map_err(|error| Fault::StateMachine {
                        reason: error.to_string(),
                    })?;
            stores.insert(tablet, store);
            machines.insert(tablet, machine);
            authorities.insert(tablet, authority);
            wirings.insert(
                tablet,
                GroupWiring {
                    voters: control_cfg.voters.clone(),
                    peer_addrs: control_cfg.peer_addrs.clone(),
                },
            );
        }
        // Snapshot-loaded control image for the removal oracle below
        // (committed-but-unapplied tails lag by milliseconds; desired
        // sets only move through plans, so staleness fails closed
        // toward recovery, never toward serving stale state — except a
        // concurrently-removed node, which the reconciler retires
        // within a pass).
        let control_image: Option<kivi_control::ControlState> = match config.control.as_ref() {
            Some(_) => match machines.get(&ConsensusGroupId::control().tablet()) {
                Some(machine) => Some(machine.tablet_control().await),
                None => None,
            },
            None => None,
        };
        let fence_generation = control_image
            .as_ref()
            .map_or(0, |state| state.placement_version().as_u64());
        let mut served_tablets = served_tablets;
        for tablet in served_tablets.clone() {
            let group = ConsensusGroupId::of_tablet(tablet);
            let authority =
                TabletAuthority::new(tablet, TabletEpoch::INITIAL, WriteGuardGeneration::INITIAL);
            let store = GroupRaftStore::from_recovered(
                config.namespace,
                group,
                &durability,
                &by_group[&group],
            )
            .map_err(|error| Fault::LogStore {
                reason: error.to_string(),
            })?;
            let machine =
                ReplicatedStateMachine::open(&config.data_dir, config.namespace, tablet, authority)
                    .map_err(|error| Fault::StateMachine {
                        reason: error.to_string(),
                    })?;
            // Removed-while-down fencing: durable voters excluding the
            // local node mean the cluster moved on without it. The
            // control desired set distinguishes a late joiner (desired
            // names it: recover, replication brings it in) from a
            // removal (desired dropped it: tombstone and skip, never
            // serve the stale copy). Empty durable state always
            // recovers (nothing stale to serve).
            if !first_boot {
                let keep = match control_image
                    .as_ref()
                    .and_then(|state| state.desired(tablet))
                {
                    Some(desired) => desired.contains(config.local),
                    None => durable_includes(&store, &machine, config.local).await,
                };
                if !keep {
                    tracing::warn!(
                        tablet = tablet.as_u64(),
                        node = config.local.as_u64(),
                        "rejoin tombstones removed replica (desired placement dropped it)"
                    );
                    write_tombstone(&config.data_dir, tablet, fence_generation);
                    served_tablets.retain(|served| *served != tablet);
                    continue;
                }
            }
            let (voters, peer_addrs) = match config.topology.membership_for(tablet) {
                Ok(wiring) => wiring,
                Err(_) => {
                    // Dynamic topology tablet (split child / merge target):
                    // the static topology predates it. Resolve voters from
                    // the replicated control desired set and dial addrs
                    // from the control registry (static flags second).
                    // Lacking both, fall back to the durable member voters
                    // with best-effort addrs (replication heals the mesh
                    // through the restarted member's outbound dials).
                    if let Some(desired) = control_image
                        .as_ref()
                        .and_then(|state| state.desired(tablet))
                    {
                        let voters: BTreeSet<u64> =
                            desired.replicas.iter().map(|node| node.as_u64()).collect();
                        let peer_addrs: BTreeMap<u64, String> = desired
                            .replicas
                            .iter()
                            .map(|node| {
                                let addr = control_image
                                    .as_ref()
                                    .and_then(|state| state.node(*node))
                                    .map(|record| record.peer.to_string())
                                    .or_else(|| {
                                        config
                                            .topology
                                            .nodes
                                            .iter()
                                            .find(|descriptor| descriptor.node == *node)
                                            .map(|descriptor| descriptor.peer.to_string())
                                    })
                                    .unwrap_or_default();
                                (node.as_u64(), addr)
                            })
                            .collect();
                        (voters, peer_addrs)
                    } else {
                        let voters = machine.member_voters().await;
                        let peer_addrs: BTreeMap<u64, String> = voters
                            .iter()
                            .map(|voter| {
                                let addr = config
                                    .topology
                                    .nodes
                                    .iter()
                                    .find(|descriptor| descriptor.node.as_u64() == *voter)
                                    .map(|descriptor| descriptor.peer.to_string())
                                    .unwrap_or_default();
                                (*voter, addr)
                            })
                            .collect();
                        (voters, peer_addrs)
                    }
                }
            };
            stores.insert(tablet, store);
            machines.insert(tablet, machine);
            authorities.insert(tablet, authority);
            wirings.insert(tablet, GroupWiring { voters, peer_addrs });
        }
        // Node-wide immutable sidecar store (content-addressed chunks plus
        // manifests; one blocking thread, shared by every replica).
        let domain = kivi_types::SecurityDomainId::from_u64(config.namespace.as_u64());
        let (sidecar, _sidecar_thread) =
            SidecarStore::open(&config.data_dir, domain, SIDECAR_CACHE_BYTES).map_err(|error| {
                Fault::Sidecar {
                    reason: error.to_string(),
                }
            })?;
        // Node-local redundancy fragment store (project-owned fragment
        // transport): opened at the caller-provided root, or disabled
        // when absent (existing deployments keep working unchanged).
        let fragments: Option<Arc<LocalFragmentStore>> = config
            .fragment_store_root
            .as_ref()
            .map(|root| {
                LocalFragmentStore::open(root, opened.meta.incarnation.as_u64())
                    .map(|(store, _recovery)| Arc::new(store))
                    .map_err(|error| Fault::Fragments {
                        reason: error.to_string(),
                    })
            })
            .transpose()?;
        // Peer TLS identity plus the static dial maps.
        let peer_tls_cert = NodeCert::load_or_generate(&config.data_dir, opened.meta.node)
            .map_err(|error| Fault::Tls {
                reason: error.to_string(),
            })?;
        let tls = TlsMaterial {
            cert: peer_tls_cert,
            peer_certs: config.peer_certs.clone(),
            trust: Some(crate::tls::empty_trust()),
            insecure_skip_verify: config.insecure_peer_tls,
        };
        let peers: HashMap<NodeId, std::net::SocketAddr> = config
            .topology
            .nodes
            .iter()
            .map(|descriptor| (descriptor.node, descriptor.peer))
            .collect();
        let peer_identity = crate::peer::PeerIdentity {
            cluster: opened.meta.cluster,
            node: opened.meta.node,
            incarnation: opened.meta.incarnation,
        };
        // Worker ingress channels plus the group registry over them. The
        // bulk fallback serves group-independent sidecars on the worker
        // owning the smallest tablet (nonempty by construction).
        let worker_count = config.worker_count;
        let mut senders = Vec::with_capacity(worker_count);
        let mut receivers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let (tx, rx) = async_channel::bounded::<OwnerRequest>(WORKER_QUEUE);
            senders.push(tx);
            receivers.push(rx);
        }
        // All served groups: recovered user tablets plus the control
        // group when participating. Tombstoned tablets stay unassigned
        // until an explicit newer creation overrides the tombstone, as
        // do traceless tablets on rejoin (they arrive through learner
        // replication, never a second initialize).
        let mut served: Vec<TabletId> = served_tablets.clone();
        if config.control.is_some() {
            served.push(ConsensusGroupId::control().tablet());
        }
        let mut group_to_worker = HashMap::new();
        for tablet in &served {
            group_to_worker.insert(
                ConsensusGroupId::of_tablet(*tablet),
                worker_for_tablet(*tablet, worker_count),
            );
        }
        // Bulk sidecars serve from the smallest served tablet's worker;
        // a fully drained node (no served groups) falls back to worker
        // 0, whose empty replica set refuses bulk loudly.
        let bulk_worker = served
            .iter()
            .min()
            .map_or(0, |smallest| worker_for_tablet(*smallest, worker_count));
        let registry = GroupRegistry {
            routes: Arc::new(
                group_to_worker
                    .iter()
                    .map(|(group, worker)| (*group, senders[*worker].clone()))
                    .collect(),
            ),
            fallback: senders.clone(),
            worker_count,
            bulk: senders[bulk_worker].clone(),
        };
        let preflight = Arc::new(crate::preflight::PreflightMetrics::default());
        // Caller-side consistency hub, shared with every worker (which
        // drive lease engines through it). Memory-held: a restart starts
        // empty, so incarnation handling fails closed by construction.
        let hub = std::sync::Arc::new(crate::consistency::ConsistencyHub::new(
            opened.meta.node,
            opened.meta.incarnation,
            kivi_types::ReadAuthorityProvider::ConservativeLeader,
            config.lease_params,
        ));
        // Shared mesh thread: one UDP endpoint plus one dial/serve pump per
        // lane serving every group. Paused until every worker bootstrapped
        // its groups (static-bootstrap race discipline, same as the
        // single-group node).
        let mut mesh_transport_config = config.transport.clone();
        mesh_transport_config.start_paused = true;
        let (mesh_ready_tx, mesh_ready_rx) = futures::channel::oneshot::channel::<
            Result<(PeerTransport, std::net::SocketAddr), NodeOpenError>,
        >();
        let (mesh_down_tx, mesh_down_rx) = futures::channel::oneshot::channel::<()>();
        let mesh_thread = std::thread::Builder::new()
            .name("kivi-consensus-mesh".to_owned())
            .spawn(move || {
                let runtime = match compio::runtime::Runtime::new() {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        let _ = mesh_ready_tx.send(Err(Fault::Owner {
                            reason: format!("compio mesh reactor failed to start: {error}"),
                        }));
                        return;
                    }
                };
                runtime.block_on(mesh_main(MeshParams {
                    transport_config: mesh_transport_config,
                    peer_identity,
                    cluster: opened.meta.cluster,
                    peers,
                    tls,
                    registry,
                    ready: mesh_ready_tx,
                    shutdown: mesh_down_rx,
                }));
            })
            .map_err(|error| Fault::Owner {
                reason: format!("consensus mesh thread failed to spawn: {error}"),
            })?;
        let (transport, peer_addr) = mesh_ready_rx.await.map_err(|_| Fault::Owner {
            reason: "consensus mesh died during startup".to_owned(),
        })??;
        // From here on, failures must tear down what started: signal the
        // mesh and every spawned worker to stop, then join them, so no
        // thread outlives a failed open.
        let teardown = |senders: &[async_channel::Sender<OwnerRequest>],
                        mesh_down: Option<futures::channel::oneshot::Sender<()>>,
                        mesh_thread: Option<std::thread::JoinHandle<()>>,
                        worker_threads: Vec<std::thread::JoinHandle<()>>| {
            if let Some(shutdown) = mesh_down {
                let _ = shutdown.send(());
            }
            for sender in senders {
                let _ = sender.try_send(OwnerRequest::Shutdown);
            }
            for thread in worker_threads {
                let _ = thread.join();
            }
            if let Some(thread) = mesh_thread {
                let _ = thread.join();
            }
        };
        // Worker threads: one Compio reactor each, owning their tablets'
        // replicas. Spawned only after the mesh is ready (gates and
        // routers need the shared transport).
        let mut worker_ready = Vec::with_capacity(worker_count);
        let mut worker_threads = Vec::with_capacity(worker_count);
        let mut mesh_down_tx = Some(mesh_down_tx);
        let mut mesh_thread = Some(mesh_thread);
        for (index, rx) in receivers.into_iter().enumerate() {
            let owned: Vec<TabletId> = served
                .iter()
                .copied()
                .filter(|tablet| worker_for_tablet(*tablet, worker_count) == index)
                .collect();
            let (ready_tx, ready_rx) =
                futures::channel::oneshot::channel::<Result<(), NodeOpenError>>();
            worker_ready.push(ready_rx);
            // Each worker receives only its own tablets' stores plus the
            // matching wiring (voters, dial addresses); empty workers host
            // nothing but still serve the ingress loop until shutdown.
            let worker_stores: HashMap<TabletId, GroupRaftStore> = owned
                .iter()
                .filter_map(|tablet| stores.get(tablet).cloned().map(|store| (*tablet, store)))
                .collect();
            let worker_wirings: HashMap<TabletId, GroupWiring> = owned
                .iter()
                .filter_map(|tablet| wirings.get(tablet).cloned().map(|wiring| (*tablet, wiring)))
                .collect();
            let params = WorkerParams {
                index,
                tablets: owned,
                stores: worker_stores,
                machines: machines.clone(),
                authorities: authorities.clone(),
                wirings: worker_wirings,
                topology: config.topology.clone(),
                transport_config: config.transport.clone(),
                transport: transport.clone(),
                sidecar: sidecar.clone(),
                fragments: fragments.clone(),
                namespace: config.namespace,
                local: opened.meta.node,
                cluster: opened.meta.cluster,
                preflight: Arc::clone(&preflight),
                preflight_enabled: config.preflight_enabled,
                hub: std::sync::Arc::clone(&hub),
                incarnation: opened.meta.incarnation,
                tick_tx: senders[index].clone(),
                data_dir: config.data_dir.clone(),
                durability: durability.clone(),
                control_seeds: config
                    .control
                    .as_ref()
                    .map(|control| control.seeds.clone())
                    .unwrap_or_default(),
                founding: first_boot,
                requests: rx,
                ready: Some(ready_tx),
            };
            let thread = std::thread::Builder::new()
                .name(format!("kivi-consensus-worker-{index}"))
                .spawn(move || {
                    let mut params = params;
                    let runtime = match compio::runtime::Runtime::new() {
                        Ok(runtime) => runtime,
                        Err(error) => {
                            if let Some(ready) = params.ready.take() {
                                let _ = ready.send(Err(Fault::Owner {
                                    reason: format!(
                                        "compio worker reactor failed to start: {error}"
                                    ),
                                }));
                            }
                            return;
                        }
                    };
                    runtime.block_on(worker_main(params));
                })
                .map_err(|error| {
                    teardown(
                        &senders,
                        mesh_down_tx.take(),
                        mesh_thread.take(),
                        std::mem::take(&mut worker_threads),
                    );
                    Fault::Owner {
                        reason: format!("consensus worker thread failed to spawn: {error}"),
                    }
                })?;
            worker_threads.push(thread);
        }
        for ready in worker_ready {
            match ready.await {
                // A worker that died or refused without reporting still
                // leaves siblings and the mesh running: tear them down so
                // no thread outlives this failed open.
                Err(_) => {
                    teardown(
                        &senders,
                        mesh_down_tx.take(),
                        mesh_thread.take(),
                        std::mem::take(&mut worker_threads),
                    );
                    return Err(Fault::Owner {
                        reason: "consensus worker died during startup".to_owned(),
                    });
                }
                Ok(Err(error)) => {
                    teardown(
                        &senders,
                        mesh_down_tx.take(),
                        mesh_thread.take(),
                        std::mem::take(&mut worker_threads),
                    );
                    return Err(error);
                }
                Ok(Ok(())) => {}
            }
        }
        // Every group bootstrapped locally: start the shared mesh.
        transport.start();
        // Startup verification per tablet: every applied chunked root must
        // resolve locally before serving reads (same fail-closed poison as
        // the single-group node; repair arrives via sidecar fetch or a
        // snapshot install). The control group holds no chunked roots.
        for tablet in &served_tablets {
            let machine = &machines[tablet];
            for (manifest, logical_len) in machine.chunked_roots().await {
                match sidecar.check_root(manifest, logical_len).await {
                    Ok(true) => {}
                    Ok(false) => {
                        machine
                            .mark_unhealthy(format!(
                                "applied root {manifest} references missing sidecars; repair required"
                            ))
                            .await;
                        break;
                    }
                    Err(error) => {
                        machine
                            .mark_unhealthy(format!("sidecar check failed: {error}"))
                            .await;
                        break;
                    }
                }
            }
        }
        Ok(Self {
            workers: senders,
            group_to_worker: Mutex::new(group_to_worker),
            machines: Mutex::new(machines),
            authorities: Mutex::new(authorities),
            sidecar,
            fragments,
            preflight,
            durability,
            mesh: transport.clone(),
            namespace: config.namespace,
            local: opened.meta.node,
            cluster: opened.meta.cluster,
            incarnation: opened.meta.incarnation,
            hub: std::sync::Arc::clone(&hub),
            alr: Mutex::new(HashMap::new()),
            tablets: Mutex::new(served_tablets),
            worker_count,
            peer_addr,
            data_dir: config.data_dir,
            worker_threads: Mutex::new(worker_threads),
            mesh_thread: Mutex::new(mesh_thread.take()),
            mesh_shutdown: Mutex::new(mesh_down_tx.take()),
            _dir_guard: opened,
        })
    }

    /// Returns the bound peer endpoint.
    #[must_use]
    pub const fn peer_addr(&self) -> std::net::SocketAddr {
        self.peer_addr
    }

    /// Returns the local node identity.
    #[must_use]
    pub const fn node(&self) -> NodeId {
        self.local
    }

    /// Returns the cluster identity.
    #[must_use]
    pub const fn cluster(&self) -> ClusterId {
        self.cluster
    }

    /// Returns this process's incarnation (advanced on every restart).
    #[must_use]
    pub const fn incarnation(&self) -> kivi_types::NodeIncarnation {
        self.incarnation
    }

    /// Returns the user tablets this node replicates, in sorted order
    /// (excludes the system control group).
    #[must_use]
    pub fn tablets(&self) -> Vec<TabletId> {
        self.tablets
            .lock()
            .map(|tablets| tablets.clone())
            .unwrap_or_default()
    }

    /// Returns the namespace served.
    #[must_use]
    pub const fn namespace(&self) -> NamespaceId {
        self.namespace
    }

    /// Whether this node hosts a control replica.
    #[must_use]
    pub fn hosts_control(&self) -> bool {
        let group = ConsensusGroupId::control();
        self.group_to_worker
            .lock()
            .is_ok_and(|map| map.contains_key(&group))
    }

    /// Reads the locally replicated control image, if this node hosts a
    /// control replica. Used by routing (leader hints), the admin plane,
    /// and the reconciler — all read-only observers, never writers.
    pub async fn control_state(&self) -> Option<kivi_control::ControlState> {
        let machine = {
            self.machines
                .lock()
                .ok()?
                .get(&ConsensusGroupId::control().tablet())
                .cloned()
        }?;
        Some(machine.tablet_control().await)
    }

    /// Returns the worker count.
    #[must_use]
    pub const fn worker_count(&self) -> usize {
        self.worker_count
    }

    /// Returns the data-directory root.
    #[must_use]
    pub fn data_dir(&self) -> &std::path::Path {
        &self.data_dir
    }

    /// Returns the immutable sidecar store (leader staging, read serving,
    /// admin diagnostics). Cloneable; the blocking worker thread is shared.
    #[must_use]
    pub fn sidecar(&self) -> &SidecarStore {
        &self.sidecar
    }

    /// Returns the node-local redundancy fragment store, if enabled
    /// ([`MultiNodeConfig::fragment_store_root`]). Coordinators borrow it
    /// for local staging; remote holders are reached through
    /// [`crate::fragments::FragmentClient`] over the peer mesh.
    #[must_use]
    pub fn fragments(&self) -> Option<&LocalFragmentStore> {
        self.fragments.as_deref()
    }

    /// Returns the node-local redundancy fragment store as a shared handle,
    /// if enabled. The coordinator moves this into background lane jobs
    /// (which require `'static` ownership); request paths keep using
    /// [`Self::fragments`].
    #[must_use]
    pub fn fragments_arc(&self) -> Option<Arc<LocalFragmentStore>> {
        self.fragments.clone()
    }

    /// Returns the shared peer mesh handle. The redundancy coordinator
    /// reuses this exact mesh (same identity, same runtime, same dials)
    /// for fragment RPCs via [`crate::fragments::FragmentClient`]: no
    /// second mesh, no second runtime, no second identity is ever created
    /// for redundancy traffic.
    #[must_use]
    pub fn mesh_transport(&self) -> crate::transport::PeerTransport {
        self.mesh.clone()
    }

    /// Builds a fragment client over the shared mesh with a per-call
    /// ceiling. One client per coordinator; the driver thread it spawns is
    /// reclaimed when the client drops.
    #[must_use]
    pub fn fragment_client(
        &self,
        timeout: std::time::Duration,
    ) -> crate::fragments::FragmentClient {
        crate::fragments::FragmentClient::new(
            self.mesh.clone(),
            crate::types::ConsensusGroupId::redundancy_sentinel(),
            timeout,
        )
    }

    /// Returns current shared-durability metrics (physical batches,
    /// fsyncs, groups per batch).
    #[must_use]
    pub fn durability_metrics(&self) -> crate::shared::SharedDurabilityMetrics {
        self.durability.metrics()
    }

    /// Returns current preflight metrics.
    #[must_use]
    pub fn preflight_metrics(&self) -> crate::preflight::PreflightMetricsSnapshot {
        self.preflight.snapshot()
    }

    /// Resolves the worker owning `tablet`'s group.
    fn worker_for(
        &self,
        tablet: TabletId,
    ) -> Result<async_channel::Sender<OwnerRequest>, ProposeError> {
        let group = ConsensusGroupId::of_tablet(tablet);
        self.group_to_worker
            .lock()
            .map_err(|_| {
                ProposeError::Consensus(ConsensusError::Unavailable {
                    reason: "consensus routing lock poisoned".to_owned(),
                })
            })?
            .get(&group)
            .map(|index| self.workers[*index].clone())
            .ok_or_else(|| {
                ProposeError::Consensus(ConsensusError::Unavailable {
                    reason: format!("tablet {} not served by this node", tablet.as_u64()),
                })
            })
    }

    /// Proposes one client mutation against `tablet`'s group: same
    /// caller-side contract as the single-group node (dedup, staging,
    /// preflight dispatch, prepare, quorum commit).
    ///
    /// # Errors
    ///
    /// Returns [`ProposeError`] for unknown tablets, routing, dedup,
    /// capability, or validation failures.
    pub async fn propose(
        &self,
        tablet: TabletId,
        op: &Operation,
        identity: Option<MutationIdentity>,
        idempotency: Option<IdempotencyKey>,
        now: WallTimestamp,
    ) -> Result<crate::node::ProposeOutcome, ProposeError> {
        let unavailable = || {
            ProposeError::Consensus(ConsensusError::Unavailable {
                reason: format!("tablet {} not served by this node", tablet.as_u64()),
            })
        };
        let machine = self
            .machines
            .lock()
            .map_err(|_| unavailable())?
            .get(&tablet)
            .cloned()
            .ok_or_else(unavailable)?;
        let authority = self
            .authorities
            .lock()
            .map_err(|_| {
                ProposeError::Consensus(ConsensusError::Unavailable {
                    reason: format!("tablet {} has no local authority", tablet.as_u64()),
                })
            })?
            .get(&tablet)
            .copied()
            .ok_or_else(|| {
                ProposeError::Consensus(ConsensusError::Unavailable {
                    reason: format!("tablet {} has no local authority", tablet.as_u64()),
                })
            })?;
        let worker = self.worker_for(tablet)?;
        propose_caller_side(
            &machine,
            &self.sidecar,
            self.namespace,
            authority,
            &worker,
            ConsensusGroupId::of_tablet(tablet),
            op,
            identity,
            idempotency,
            now,
        )
        .await
    }

    /// Serves one read against `tablet`'s group under its contract,
    /// returning the outcome with its proof triple.
    ///
    /// `eligibility` gates the roster fast path (point reads eligible;
    /// transactions, batches, and scans conservative-only).
    ///
    /// # Errors
    ///
    /// Returns [`ReadError`] for unknown tablets, routing, lineage,
    /// validation, or coverage failures.
    pub async fn read(
        &self,
        tablet: TabletId,
        op: &Operation,
        contract: ReadContract,
        ctx: kivi_types::ReadContext,
        eligibility: kivi_types::LeaseEligibility,
    ) -> Result<crate::consistency::ServedRead, ReadError> {
        let (machine, authority, worker, alr) = self.read_front(tablet)?;
        crate::node::ReadFront::bind(
            &machine,
            &worker,
            &self.hub,
            &alr,
            ConsensusGroupId::of_tablet(tablet),
            tablet,
            authority,
            self.incarnation,
        )
        .read(op, contract, ctx, eligibility)
        .await
    }

    /// Serves one bounded scan page against `tablet`'s group under its
    /// contract (each tablet read individually strong; the multi-tablet
    /// scan is not one snapshot). Scans are always lease-ineligible.
    ///
    /// # Errors
    ///
    /// Returns [`ReadError`] for unknown tablets, routing, lineage,
    /// validation, or coverage failures.
    pub async fn scan(
        &self,
        tablet: TabletId,
        spec: &kivi_state::ScanSpec,
        contract: ReadContract,
        ctx: kivi_types::ReadContext,
    ) -> Result<crate::consistency::ServedScan, ReadError> {
        let (machine, authority, worker, alr) = self.read_front(tablet)?;
        crate::node::ReadFront::bind(
            &machine,
            &worker,
            &self.hub,
            &alr,
            ConsensusGroupId::of_tablet(tablet),
            tablet,
            authority,
            self.incarnation,
        )
        .scan(
            spec,
            contract,
            ctx,
            kivi_types::LeaseEligibility::ConservativeOnly,
        )
        .await
    }

    /// Resolves the machine, authority, owner channel, and ALR
    /// coordinator for one tablet's read path (creating the coordinator
    /// lazily on first read).
    fn read_front(
        &self,
        tablet: TabletId,
    ) -> Result<
        (
            ReplicatedStateMachine,
            TabletAuthority,
            async_channel::Sender<OwnerRequest>,
            std::sync::Arc<crate::alr::AlrCoordinator>,
        ),
        ReadError,
    > {
        let unavailable = || {
            ReadError::Consensus(ConsensusError::Unavailable {
                reason: format!("tablet {} not served by this node", tablet.as_u64()),
            })
        };
        let machine = self
            .machines
            .lock()
            .map_err(|_| unavailable())?
            .get(&tablet)
            .cloned()
            .ok_or_else(unavailable)?;
        let worker = self.worker_for(tablet).map_err(|error| match error {
            ProposeError::Consensus(consensus) => ReadError::Consensus(consensus),
            _ => unavailable(),
        })?;
        let authority = self
            .authorities
            .lock()
            .map_err(|_| unavailable())?
            .get(&tablet)
            .copied()
            .ok_or_else(|| {
                ReadError::Consensus(ConsensusError::Unavailable {
                    reason: format!("tablet {} has no local authority", tablet.as_u64()),
                })
            })?;
        let alr = self
            .alr
            .lock()
            .map_err(|_| unavailable())?
            .entry(tablet)
            .or_insert_with(|| {
                std::sync::Arc::new(crate::alr::AlrCoordinator::bind(
                    machine.clone(),
                    worker.clone(),
                    std::sync::Arc::clone(&self.hub),
                    ConsensusGroupId::of_tablet(tablet),
                    tablet,
                    authority,
                    self.incarnation,
                    self.local,
                ))
            })
            .clone();
        Ok((machine, authority, worker, alr))
    }

    /// Returns this node's serving authority for `tablet`, if hosted.
    /// The server cross-checks it against the directory on every read
    /// ([`reconcile_authority`](kivi_types::reconcile_authority)): any
    /// drift between the routing view and the commit view fails closed.
    #[must_use]
    pub fn authority_for(&self, tablet: TabletId) -> Option<TabletAuthority> {
        self.authorities.lock().ok()?.get(&tablet).copied()
    }

    /// Switches the read-authority mode for every tablet on this node
    /// (operator/test control for the Lazy-ALR and roster-lease
    /// accelerators).
    pub fn set_read_provider(&self, provider: kivi_types::ReadAuthorityProvider) {
        self.hub.set_provider(provider);
    }

    /// Gathers per-group diagnostics for every local group (fans out to
    /// owning workers concurrently; a dead worker reports closed statuses
    /// rather than hanging the admin plane).
    pub async fn status_all(&self) -> Vec<NodeStatus> {
        let tablets = self.tablets();
        let routing = self.group_to_worker.lock().ok().map(|map| map.clone());
        let hub = &self.hub;
        let calls: Vec<_> = tablets
            .iter()
            .map(|tablet| {
                let group = ConsensusGroupId::of_tablet(*tablet);
                let worker = routing
                    .as_ref()
                    .and_then(|map| map.get(&group).map(|index| self.workers[*index].clone()));
                let local = self.local;
                let sidecar = self.sidecar.metrics().snapshot();
                let preflight = self.preflight.snapshot();
                async move {
                    let mut status = {
                        let Some(worker) = worker else {
                            return closed_status(group, local, sidecar, preflight);
                        };
                        owner_call_on(&worker, |reply| OwnerRequest::Status { group, reply })
                            .await
                            .unwrap_or_else(|_| closed_status(group, local, sidecar, preflight))
                    };
                    status.consistency = hub
                        .snapshot(group.tablet(), process_ticks())
                        .map_or_else(kivi_types::ConsistencySnapshot::default, |(_, snapshot)| {
                            snapshot
                        });
                    status
                }
            })
            .collect();
        futures::future::join_all(calls).await
    }

    /// Resolves the worker channel for one group, if served locally.
    fn channel_for(&self, group: ConsensusGroupId) -> Option<async_channel::Sender<OwnerRequest>> {
        self.group_to_worker
            .lock()
            .ok()?
            .get(&group)
            .map(|index| self.workers[*index].clone())
    }

    /// Gathers diagnostics for one group, overlaying live consistency
    /// counters from the caller-side hub.
    pub async fn status_for(&self, group: ConsensusGroupId) -> NodeStatus {
        let sidecar = self.sidecar.metrics().snapshot();
        let preflight = self.preflight.snapshot();
        let Some(worker) = self.channel_for(group) else {
            return closed_status(group, self.local, sidecar, preflight);
        };
        let mut status = owner_call_on(&worker, |reply| OwnerRequest::Status { group, reply })
            .await
            .unwrap_or_else(|_| closed_status(group, self.local, sidecar, preflight));
        status.consistency = self
            .hub
            .snapshot(group.tablet(), process_ticks())
            .map_or_else(kivi_types::ConsistencySnapshot::default, |(_, snapshot)| {
                snapshot
            });
        status
    }

    /// Seals a snapshot and purges one group's log through its base.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason for unknown groups or when the
    /// snapshot/purge never completes before the deadline.
    pub async fn snapshot_and_purge(
        &self,
        group: ConsensusGroupId,
    ) -> Result<ConsensusLogIndex, String> {
        let Some(worker) = self.channel_for(group) else {
            return Err(format!("group {group} not served by this node"));
        };
        let (reply, rx) = futures::channel::oneshot::channel();
        worker
            .send(OwnerRequest::SnapshotPurge { group, reply })
            .await
            .map_err(|_| "consensus worker shut down".to_owned())?;
        rx.await
            .map_err(|_| "consensus worker shut down".to_owned())?
    }

    /// Registers a learner on one group's leader (`add_learner`). The
    /// reconciler calls this with `blocking=true` so it returns once the
    /// learner is line-rate; promoting earlier can stall quorum progress.
    /// Callers must address the tablet leader: a follower answers with a
    /// not-leader reason and the reconciler retries at the leader.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason for unknown groups, worker
    /// shutdown, or membership failures.
    pub async fn add_learner(
        &self,
        tablet: TabletId,
        node: NodeId,
        addr: String,
        blocking: bool,
    ) -> Result<(), String> {
        let group = ConsensusGroupId::of_tablet(tablet);
        let Some(worker) = self.channel_for(group) else {
            return Err(format!("group {group} not served by this node"));
        };
        let (reply, rx) = futures::channel::oneshot::channel();
        worker
            .send(OwnerRequest::AddLearner {
                group,
                node,
                addr,
                blocking,
                reply,
            })
            .await
            .map_err(|_| "consensus worker shut down".to_owned())?;
        rx.await
            .map_err(|_| "consensus worker shut down".to_owned())?
    }

    /// Replaces one group's voter set through joint consensus (handled
    /// internally by `OpenRaft`). New voters must already be learners;
    /// removed voters leave or linger per `retain`. Planned migration
    /// uses `retain=false` so the retired source leaves the cluster.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason for unknown groups, worker
    /// shutdown, or membership failures.
    pub async fn change_membership(
        &self,
        tablet: TabletId,
        voters: std::collections::BTreeSet<u64>,
        retain: bool,
    ) -> Result<(), String> {
        let group = ConsensusGroupId::of_tablet(tablet);
        let Some(worker) = self.channel_for(group) else {
            return Err(format!("group {group} not served by this node"));
        };
        let (reply, rx) = futures::channel::oneshot::channel();
        worker
            .send(OwnerRequest::ChangeMembership {
                group,
                voters,
                retain,
                reply,
            })
            .await
            .map_err(|_| "consensus worker shut down".to_owned())?;
        rx.await
            .map_err(|_| "consensus worker shut down".to_owned())?
    }

    /// Asks one group's leader to hand leadership to `to`
    /// (`Trigger::transfer_leader`). Fire-and-forget: a non-leader
    /// ignores it and the reconciler re-observes and continues.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason for unknown groups or worker
    /// shutdown.
    pub async fn transfer_leader(&self, tablet: TabletId, to: NodeId) -> Result<(), String> {
        let group = ConsensusGroupId::of_tablet(tablet);
        let Some(worker) = self.channel_for(group) else {
            return Err(format!("group {group} not served by this node"));
        };
        let (reply, rx) = futures::channel::oneshot::channel();
        worker
            .send(OwnerRequest::TransferLeader { group, to, reply })
            .await
            .map_err(|_| "consensus worker shut down".to_owned())?;
        rx.await
            .map_err(|_| "consensus worker shut down".to_owned())?
    }

    /// Observes one group's actual voter/learner/leader state for the
    /// migration reconciler.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason for unknown groups or worker
    /// shutdown.
    pub async fn observe_membership(
        &self,
        tablet: TabletId,
    ) -> Result<crate::control::ObservedMembership, String> {
        let group = ConsensusGroupId::of_tablet(tablet);
        let Some(worker) = self.channel_for(group) else {
            return Err(format!("group {group} not served by this node"));
        };
        let (reply, rx) = futures::channel::oneshot::channel();
        worker
            .send(OwnerRequest::ObserveMembership { group, reply })
            .await
            .map_err(|_| "consensus worker shut down".to_owned())?;
        rx.await
            .map_err(|_| "consensus worker shut down".to_owned())?
    }

    /// Creates an empty local replica for a tablet this node has never
    /// hosted (migration target). Idempotent: an existing replica
    /// answers success. The front maps update so caller-side propose,
    /// read, and diagnostics cover the new group without a restart.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason for reserved tablets, worker
    /// shutdown, stale fenced creates, or storage failures.
    pub async fn ensure_group(
        &self,
        tablet: TabletId,
        voters: std::collections::BTreeSet<u64>,
        generation: u64,
    ) -> Result<(), String> {
        let group = ConsensusGroupId::of_tablet(tablet);
        if group.is_control() || tablet.as_u64() == 0 {
            return Err(format!("tablet {} is reserved", tablet.as_u64()));
        }
        let worker_index = worker_for_tablet(tablet, self.worker_count);
        let Some(worker) = self.workers.get(worker_index).cloned() else {
            return Err("no consensus worker exists".to_owned());
        };
        let (reply, rx) = futures::channel::oneshot::channel();
        worker
            .send(OwnerRequest::EnsureGroup {
                group,
                tablet,
                voters,
                generation,
                reply,
            })
            .await
            .map_err(|_| "consensus worker shut down".to_owned())?;
        let info = rx
            .await
            .map_err(|_| "consensus worker shut down".to_owned())??;
        if let Ok(mut machines) = self.machines.lock() {
            machines.insert(tablet, info.machine);
        }
        if let Ok(mut authorities) = self.authorities.lock() {
            authorities.insert(tablet, info.authority);
        }
        if let Ok(mut routing) = self.group_to_worker.lock() {
            routing.insert(group, worker_index);
        }
        if let Ok(mut tablets) = self.tablets.lock()
            && !tablets.contains(&tablet)
        {
            tablets.push(tablet);
            tablets.sort_by_key(|tablet| tablet.as_u64());
        }
        Ok(())
    }

    /// Retires the local replica after committed membership no longer
    /// requires it: stops its `Raft`, unregisters the group, and writes
    /// a tombstone (durable data retained conservatively). Idempotent:
    /// an already-retired source answers success.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason for reserved tablets, worker
    /// shutdown, or storage failures.
    pub async fn retire_group(&self, tablet: TabletId, generation: u64) -> Result<(), String> {
        let group = ConsensusGroupId::of_tablet(tablet);
        if group.is_control() || tablet.as_u64() == 0 {
            return Err(format!("tablet {} is reserved", tablet.as_u64()));
        }
        let worker_index = worker_for_tablet(tablet, self.worker_count);
        let Some(worker) = self.workers.get(worker_index).cloned() else {
            return Err("no consensus worker exists".to_owned());
        };
        let (reply, rx) = futures::channel::oneshot::channel();
        worker
            .send(OwnerRequest::RetireGroup {
                group,
                tablet,
                generation,
                reply,
            })
            .await
            .map_err(|_| "consensus worker shut down".to_owned())?;
        rx.await
            .map_err(|_| "consensus worker shut down".to_owned())??;
        if let Ok(mut machines) = self.machines.lock() {
            machines.remove(&tablet);
        }
        if let Ok(mut authorities) = self.authorities.lock() {
            authorities.remove(&tablet);
        }
        // A retired replica serves nothing: drop its receipts, leases,
        // and cache entries. If the tablet is re-hosted here later
        // (re-migration), it re-proves everything from authority instead
        // of resurrecting pre-retirement evidence.
        self.hub.invalidate(tablet);
        if let Ok(mut routing) = self.group_to_worker.lock() {
            routing.remove(&group);
        }
        if let Ok(mut tablets) = self.tablets.lock() {
            tablets.retain(|served| *served != tablet);
        }
        Ok(())
    }

    /// Initializes one new tablet group exactly once on this replica
    /// (split-child / merge-target founder). Other replicas join through
    /// learner replication, never a second initialize. Idempotent: an
    /// already-initialized group answers success.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason for reserved tablets, unknown
    /// groups, worker shutdown, or initialization failures.
    pub async fn initialize_group(
        &self,
        tablet: TabletId,
        voters: std::collections::BTreeSet<u64>,
        peer_addrs: std::collections::BTreeMap<u64, String>,
    ) -> Result<(), String> {
        let group = ConsensusGroupId::of_tablet(tablet);
        if group.is_control() || tablet.as_u64() == 0 {
            return Err(format!("tablet {} is reserved", tablet.as_u64()));
        }
        let worker_index = worker_for_tablet(tablet, self.worker_count);
        let Some(worker) = self.workers.get(worker_index).cloned() else {
            return Err("no consensus worker exists".to_owned());
        };
        let (reply, rx) = futures::channel::oneshot::channel();
        worker
            .send(OwnerRequest::InitializeGroup {
                group,
                tablet,
                voters,
                peer_addrs,
                reply,
            })
            .await
            .map_err(|_| "consensus worker shut down".to_owned())?;
        rx.await
            .map_err(|_| "consensus worker shut down".to_owned())?
    }

    /// Proposes one typed control-plane mutation against the system
    /// control group.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason when this node serves no control
    /// replica or the commit fails.
    pub async fn propose_control(
        &self,
        mutation: kivi_control::ControlMutation,
    ) -> Result<ConsensusLogIndex, String> {
        self.propose_control_typed_front(mutation)
            .await
            .map_err(|error| error.to_string())
    }

    /// Proposes one control mutation locally and preserves its typed
    /// leader hint or failure.
    ///
    /// # Errors
    ///
    /// Returns [`ControlProposeError`] when the local control replica is
    /// unavailable, shutting down, or not the leader.
    pub async fn propose_control_typed_front(
        &self,
        mutation: kivi_control::ControlMutation,
    ) -> Result<ConsensusLogIndex, ControlProposeError> {
        let group = ConsensusGroupId::control();
        let Some(worker) = self.channel_for(group) else {
            return Err(ControlProposeError::Unavailable {
                detail: format!("control group {group} not served by this node"),
            });
        };
        let (reply, rx) = futures::channel::oneshot::channel();
        worker
            .send(OwnerRequest::ProposeControl {
                group,
                mutation,
                reply,
            })
            .await
            .map_err(|_| ControlProposeError::Unavailable {
                detail: "consensus worker shut down".to_owned(),
            })?;
        rx.await.map_err(|_| ControlProposeError::Unavailable {
            detail: "consensus worker shut down".to_owned(),
        })?
    }

    /// Proposes a control mutation from any node, forwarding once over the
    /// existing peer mesh when this replica is not the control leader.
    ///
    /// The forwarded bytes are the canonical control mutation encoding,
    /// so the control group's Raft log still decides order and durability.
    /// At most two peer hops are attempted, with no loop and no membership
    /// change. A duplicate forwarded mutation is safe for the redundancy
    /// coordinator because publishing the same generation with the same
    /// bytes is fenced to a no-op by the control state machine.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason when the local proposal and the
    /// bounded forwarded attempts fail.
    pub async fn propose_control_anywhere(
        &self,
        mutation: kivi_control::ControlMutation,
        timeout: Duration,
    ) -> Result<ConsensusLogIndex, String> {
        let leader = match self.propose_control_typed_front(mutation.clone()).await {
            Ok(index) => return Ok(index),
            Err(ControlProposeError::NotLeader {
                leader: Some(leader),
            }) => leader,
            Err(error) => return Err(error.to_string()),
        };
        let request = PeerRequest::ControlForward(PeerControlForwardRequest {
            mutation: mutation.encode_to_vec(),
        });
        let first = self
            .mesh
            .call(
                leader,
                ConsensusGroupId::control(),
                request.clone(),
                timeout,
            )
            .await;
        match first {
            Ok(PeerResponse::ControlForward(PeerControlForwardResponse { index })) => {
                Ok(ConsensusLogIndex::new(index))
            }
            Ok(response) => Err(format!(
                "control forward to leader {} returned {response:?}",
                leader.as_u64()
            )),
            Err(error) => {
                let Some(next) = control_forward_leader(&error.to_string()) else {
                    return Err(format!("control forward failed: {error}"));
                };
                if next == leader {
                    return Err(format!("control forward failed: {error}"));
                }
                match self
                    .mesh
                    .call(next, ConsensusGroupId::control(), request, timeout)
                    .await
                {
                    Ok(PeerResponse::ControlForward(PeerControlForwardResponse { index })) => {
                        Ok(ConsensusLogIndex::new(index))
                    }
                    Ok(response) => Err(format!(
                        "control forward to leader {} returned {response:?}",
                        next.as_u64()
                    )),
                    Err(error) => Err(format!("control forward failed: {error}")),
                }
            }
        }
    }

    /// Sets the split/merge cutover fence on one local replica
    /// (idempotent). While fenced, fresh mutating proposes fail with
    /// `Fenced` (retryable); reads still serve and dedup hits still
    /// answer. The reconciler fences every parent replica before the
    /// final tail install and re-applies the fence from the persisted
    /// plan phase after any restart.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason when this node hosts no such
    /// replica.
    pub async fn set_tablet_fenced(&self, tablet: TabletId, fenced: bool) -> Result<(), String> {
        let machine = self
            .machines
            .lock()
            .map_err(|_| "machine map poisoned".to_owned())?
            .get(&tablet)
            .cloned()
            .ok_or_else(|| format!("tablet {} not served by this node", tablet.as_u64()))?;
        machine.set_fenced(fenced).await;
        Ok(())
    }

    /// Seeds one split child from the local parent replica: partitions
    /// the parent's live objects by `split_hash` (`hash < split_hash`
    /// goes left, else right) and installs the child's half plus the full
    /// parent session set (both children keep every session for
    /// exactly-once retries). Chunked roots reference the same immutable
    /// `ManifestId`s — no chunk bytes are copied or retransferred.
    ///
    /// Colocated by design (children inherit the parent replica set), so
    /// no bulk network transfer is needed: every replica seeds locally.
    /// Idempotent: re-seeding overwrites the inactive target.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason when either replica is not hosted
    /// locally.
    /// Seeds a hash split child from its local parent: keys whose partition
    /// hash falls on this side go here, including system (`\xff`) keys by
    /// the same hash rule — transaction decision records therefore land in
    /// exactly the child that post-cutover hash routing finds, so recovery
    /// always finds them. Sessions copy wholesale to both children
    /// (bounded, and required for exactly-once retries across the cutover);
    /// prepared intents are NOT copied and never migrate (the reconciler
    /// only seals parents with zero unresolved intents). Idempotent:
    /// re-seeding overwrites the inactive target.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason when any replica is not hosted
    /// locally.
    pub async fn seed_split_child(
        &self,
        parent: TabletId,
        child: TabletId,
        split_hash: u128,
        left: bool,
    ) -> Result<(usize, usize), String> {
        let (parent_machine, child_machine) = self
            .machines
            .lock()
            .map_err(|_| "machine map poisoned".to_owned())
            .and_then(|machines| {
                let parent = machines.get(&parent).cloned().ok_or_else(|| {
                    format!("parent tablet {} not served by this node", parent.as_u64())
                })?;
                let child = machines.get(&child).cloned().ok_or_else(|| {
                    format!("child tablet {} not served by this node", child.as_u64())
                })?;
                Ok((parent, child))
            })?;
        let (objects, sessions) = parent_machine.export_topology_copy().await;
        let mut half = Vec::new();
        for (key, object) in objects {
            let hash = kivi_state::PartitionHasher::V1
                .hash(self.namespace, key.as_bytes())
                .map(kivi_types::PartitionHash::as_u128);
            let is_left = hash.is_some_and(|hash| hash < split_hash);
            if is_left == left {
                half.push((key, object));
            }
        }
        let object_count = half.len();
        let session_count = sessions.len();
        child_machine.install_topology_copy((half, sessions)).await;
        Ok((object_count, session_count))
    }

    /// Seeds a merge target from two local parents: unions both object
    /// sets (ranges are disjoint, so keys cannot collide) and unions both
    /// session sets (per-session sequences are globally unique; a
    /// conflicting duplicate keeps the higher commit and never
    /// re-executes). Idempotent: re-seeding overwrites the inactive
    /// target.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason when any replica is not hosted
    /// locally.
    pub async fn seed_merge_target(
        &self,
        left: TabletId,
        right: TabletId,
        merged: TabletId,
    ) -> Result<(usize, usize), String> {
        let (left_machine, right_machine, merged_machine) = self
            .machines
            .lock()
            .map_err(|_| "machine map poisoned".to_owned())
            .and_then(|machines| {
                let left = machines.get(&left).cloned().ok_or_else(|| {
                    format!("left tablet {} not served by this node", left.as_u64())
                })?;
                let right = machines.get(&right).cloned().ok_or_else(|| {
                    format!("right tablet {} not served by this node", right.as_u64())
                })?;
                let merged = machines.get(&merged).cloned().ok_or_else(|| {
                    format!("merged tablet {} not served by this node", merged.as_u64())
                })?;
                Ok((left, right, merged))
            })?;
        let (mut objects, left_sessions) = left_machine.export_topology_copy().await;
        let (right_objects, right_sessions) = right_machine.export_topology_copy().await;
        objects.extend(right_objects);
        let sessions = union_sessions(left_sessions, right_sessions);
        let object_count = objects.len();
        let session_count = sessions.len();
        merged_machine
            .install_topology_copy((objects, sessions))
            .await;
        Ok((object_count, session_count))
    }

    /// Returns the live object count of one local replica, if hosted.
    /// Best-effort sizing signal for automatic split/merge policy (exact
    /// count, cheap). Logical bytes and load estimates refine this in a
    /// later stage; the policy architecture already carries them.
    pub async fn tablet_object_count(&self, tablet: TabletId) -> Option<usize> {
        let machine = self.machines.lock().ok()?.get(&tablet).cloned()?;
        Some(machine.object_count().await)
    }

    /// Seeds an ordered split child from its local parent: routable keys
    /// partition by the half-open `[start, end)` boundary, never a hash
    /// midpoint. Routed-by-fiat keys (system `\xff` records) copy to the
    /// right child deterministically — and post-cutover record reads
    /// route by the same ordered rule, so recovery always finds them.
    /// Non-unique index entries copy by embedded primary, so each entry
    /// stays beside the primary the unified router co-locates it with
    /// (unique entries carry no primary and copy by their own bytes).
    /// Sessions copy wholesale to the child; prepared intents are NOT
    /// copied and never migrate (a copy would fork the reservation):
    /// the reconciler only seals parents with zero unresolved intents, so
    /// there is nothing to migrate at cutover. The child inherits the
    /// parent's ordered-index flag. Idempotent: re-seeding overwrites the
    /// inactive target.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason when either replica is not hosted
    /// locally.
    pub async fn seed_split_child_ordered(
        &self,
        parent: TabletId,
        child: TabletId,
        split_key: Vec<u8>,
        left: bool,
    ) -> Result<(usize, usize), String> {
        let (parent_machine, child_machine) = self
            .machines
            .lock()
            .map_err(|_| "machine map poisoned".to_owned())
            .and_then(|machines| {
                let parent = machines.get(&parent).cloned().ok_or_else(|| {
                    format!("parent tablet {} not served by this node", parent.as_u64())
                })?;
                let child = machines.get(&child).cloned().ok_or_else(|| {
                    format!("child tablet {} not served by this node", child.as_u64())
                })?;
                Ok((parent, child))
            })?;
        let ordered = parent_machine.ordered_index_enabled().await;
        let (objects, sessions) = parent_machine.export_topology_copy().await;
        let mut half = Vec::new();
        for (key, object) in objects {
            let bytes = key.as_bytes();
            let is_left = if bytes.first().is_some_and(|byte| *byte == 0xFF) {
                // Routed-by-fiat system keys sort past every user split
                // point: right child, matching post-cutover routing.
                !left
            } else if let Some(primary) = kivi_state::parse_index_primary(bytes) {
                // Co-located index entries follow their primary across the
                // boundary; the median never anchors on an index key, so
                // its own bytes never decide placement.
                (primary.as_slice() < split_key.as_slice()) == left
            } else {
                (bytes < split_key.as_slice()) == left
            };
            if is_left {
                half.push((key, object));
            }
        }
        let object_count = half.len();
        let session_count = sessions.len();
        child_machine.install_topology_copy((half, sessions)).await;
        child_machine.set_ordered_indexing(ordered).await;
        Ok((object_count, session_count))
    }

    /// Propagates the ordered-index flag from parent to child after a hash
    /// split seed (hash children of an ordered parent cannot happen —
    /// layouts never mix — but the flag copy keeps the helper total).
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason when either replica is not hosted
    /// locally.
    pub async fn inherit_ordered_indexing(
        &self,
        parent: TabletId,
        child: TabletId,
    ) -> Result<(), String> {
        let (parent_machine, child_machine) = self
            .machines
            .lock()
            .map_err(|_| "machine map poisoned".to_owned())
            .and_then(|machines| {
                let parent = machines.get(&parent).cloned().ok_or_else(|| {
                    format!("parent tablet {} not served by this node", parent.as_u64())
                })?;
                let child = machines.get(&child).cloned().ok_or_else(|| {
                    format!("child tablet {} not served by this node", child.as_u64())
                })?;
                Ok((parent, child))
            })?;
        let ordered = parent_machine.ordered_index_enabled().await;
        child_machine.set_ordered_indexing(ordered).await;
        Ok(())
    }

    /// Enables or disables the ordered key index on one local replica
    /// (topology reconciliation drives this from the namespace layout).
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason when the replica is not hosted
    /// locally.
    pub async fn set_ordered_indexing(
        &self,
        tablet: TabletId,
        enabled: bool,
    ) -> Result<(), String> {
        let machine =
            self.machines
                .lock()
                .map_err(|_| "machine map poisoned".to_owned())
                .and_then(|machines| {
                    machines.get(&tablet).cloned().ok_or_else(|| {
                        format!("tablet {} not served by this node", tablet.as_u64())
                    })
                })?;
        machine.set_ordered_indexing(enabled).await;
        Ok(())
    }

    /// Whether one local replica maintains the ordered key index (`None`
    /// when the replica is not hosted locally).
    pub async fn ordered_index_enabled(&self, tablet: TabletId) -> Option<bool> {
        let machine = self.machines.lock().ok()?.get(&tablet).cloned()?;
        Some(machine.ordered_index_enabled().await)
    }

    /// Removes and returns every prepared intent on one local replica
    /// (topology cutover migrates intents to their new owners).
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason when the replica is not hosted
    /// locally.
    pub async fn drain_intents(
        &self,
        tablet: TabletId,
    ) -> Result<Vec<kivi_state::TxnIntent>, String> {
        let machine =
            self.machines
                .lock()
                .map_err(|_| "machine map poisoned".to_owned())
                .and_then(|machines| {
                    machines.get(&tablet).cloned().ok_or_else(|| {
                        format!("tablet {} not served by this node", tablet.as_u64())
                    })
                })?;
        Ok(machine.drain_intents().await)
    }

    /// Installs one migrated intent on one local replica (cutover path).
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason when the replica is not hosted
    /// locally.
    pub async fn restore_intent(
        &self,
        tablet: TabletId,
        intent: kivi_state::TxnIntent,
    ) -> Result<(), String> {
        let machine =
            self.machines
                .lock()
                .map_err(|_| "machine map poisoned".to_owned())
                .and_then(|machines| {
                    machines.get(&tablet).cloned().ok_or_else(|| {
                        format!("tablet {} not served by this node", tablet.as_u64())
                    })
                })?;
        machine.restore_intent(intent).await;
        Ok(())
    }

    /// Number of prepared intents on one local replica, if hosted.
    pub async fn tablet_intent_count(&self, tablet: TabletId) -> Option<usize> {
        let machine = self.machines.lock().ok()?.get(&tablet).cloned()?;
        Some(machine.pending_intent_count().await)
    }

    /// Prepared intents on one local replica, in key order, if hosted
    /// (transaction resolver consults this; never discards here).
    pub async fn tablet_intents(&self, tablet: TabletId) -> Option<Vec<kivi_state::TxnIntent>> {
        let machine = self.machines.lock().ok()?.get(&tablet).cloned()?;
        Some(machine.snapshot_intents().await)
    }

    /// Logical telemetry of one local replica, if hosted.
    pub async fn tablet_stats(
        &self,
        tablet: TabletId,
        now: WallTimestamp,
    ) -> Option<kivi_state::StoreStats> {
        let machine = self.machines.lock().ok()?.get(&tablet).cloned()?;
        Some(machine.tablet_stats(now).await)
    }

    /// Approximate median split key of one local replica, if hosted and
    /// ordered-indexed (first key past half of stored logical bytes).
    pub async fn tablet_median_key(&self, tablet: TabletId) -> Option<Vec<u8>> {
        let machine = self.machines.lock().ok()?.get(&tablet).cloned()?;
        machine.median_split_key().await
    }

    /// Adds a dialable peer link for a dynamically admitted node
    /// (replicated registry commit observed locally). Returns whether
    /// the link is new. The mesh dials it on demand; existing links
    /// are never replaced. Callable from any thread.
    #[must_use]
    pub fn add_peer_dial(&self, peer: NodeId, addr: std::net::SocketAddr) -> bool {
        self.mesh.add_peer(peer, addr)
    }

    /// Pins a dynamically admitted peer's certificate DER for TLS
    /// verification (dials and incoming connections). Static anchors
    /// keep precedence. Callable from any thread.
    pub fn trust_peer(&self, peer: NodeId, cert_der: Vec<u8>) {
        self.mesh.trust_peer(peer, cert_der);
    }

    /// Suspends one peer link (partition test hook, drain tooling). Hops
    /// through worker ingress: the suspend itself awaits compio-reactor
    /// timer state (`!Send`), so it must run on a worker reactor — never
    /// directly on a Tokio caller. The shared transport suspends once no
    /// matter which worker serves it.
    pub async fn suspend_peer(&self, peer: NodeId) {
        let (reply, rx) = futures::channel::oneshot::channel();
        if let Some(worker) = self.workers.first()
            && worker
                .send(OwnerRequest::SuspendPeer { peer, reply })
                .await
                .is_ok()
        {
            let _ = rx.await;
        }
    }

    /// Resumes a suspended peer link (same worker-ingress discipline as
    /// [`suspend_peer`](Self::suspend_peer)).
    pub async fn resume_peer(&self, peer: NodeId) {
        let (reply, rx) = futures::channel::oneshot::channel();
        if let Some(worker) = self.workers.first()
            && worker
                .send(OwnerRequest::ResumePeer { peer, reply })
                .await
                .is_ok()
        {
            let _ = rx.await;
        }
    }

    /// Stops every `Raft` instance, the peer mesh, and all workers/threads.
    /// Mesh teardown runs on the mesh thread itself: `PeerTransport`
    /// teardown awaits compio-reactor timer state (`!Send`), so it must
    /// never run on a Tokio caller.
    ///
    /// # Panics
    ///
    /// Panics when an owner-handle lock is poisoned (a lifecycle bug,
    /// never a runtime condition).
    pub async fn shutdown(&self) {
        tracing::info!(
            node = self.local.as_u64(),
            "consensus node shutdown started"
        );
        for worker in &self.workers {
            let _ = worker.send(OwnerRequest::Shutdown).await;
        }
        if let Some(shutdown) = self.mesh_shutdown.lock().expect("mesh handle holds").take() {
            let _ = shutdown.send(());
        }
        if let Some(thread) = self.mesh_thread.lock().expect("mesh handle holds").take() {
            let _ = thread.join();
        }
        for thread in self
            .worker_threads
            .lock()
            .expect("worker handles hold")
            .drain(..)
        {
            let _ = thread.join();
        }
    }
}

/// Closed per-group status for a dead or unknown worker (mirrors the
/// single-group fallback: unhealthy, never a hang).
fn closed_status(
    group: ConsensusGroupId,
    local: NodeId,
    sidecar: crate::sidecar::SidecarMetricsSnapshot,
    preflight: crate::preflight::PreflightMetricsSnapshot,
) -> NodeStatus {
    NodeStatus {
        group,
        replica: crate::types::ReplicaId::of_node(local),
        role: crate::types::ReplicaRole::Follower,
        term: crate::types::ConsensusTerm::INITIAL,
        leader: None,
        last_log: ConsensusLogIndex::NONE,
        committed: ConsensusLogIndex::NONE,
        applied: ConsensusLogIndex::NONE,
        applied_commit: kivi_types::CommitPosition::UNASSIGNED,
        snapshot: None,
        purged: None,
        healthy: false,
        detail: Some("consensus worker shut down".to_owned()),
        peers: HashMap::new(),
        sidecar,
        preflight,
        consistency: kivi_types::ConsistencySnapshot::default(),
    }
}

/// Serves one dynamic replica creation inline (rare, local-only):
/// idempotent re-create answers the existing handles; fresh tablets
/// build an uninitialized replica that joins through learner
/// replication (never a second initialize).
async fn serve_ensure(
    build: &WorkerBuildCtx,
    replicas: &mut HashMap<ConsensusGroupId, OwnerCtx<GroupRaftStore>>,
    group: ConsensusGroupId,
    tablet: TabletId,
    voters: &BTreeSet<u64>,
    generation: u64,
    reply: futures::channel::oneshot::Sender<Result<crate::node::NewReplica, String>>,
) {
    if group != ConsensusGroupId::of_tablet(tablet) {
        let _ = reply.send(Err(format!("group {group} names a foreign tablet")));
        return;
    }
    if let Some(ctx) = replicas.get(&group) {
        let _ = reply.send(Ok(crate::node::NewReplica {
            machine: ctx.machine.clone(),
            authority: ctx.authority,
        }));
        return;
    }
    match build_dynamic_replica(build, tablet, voters, generation).await {
        Ok((built_group, ctx, info)) => {
            replicas.insert(built_group, ctx);
            let _ = reply.send(Ok(info));
        }
        Err(reason) => {
            let _ = reply.send(Err(reason));
        }
    }
}

/// Serves one dynamic replica retirement inline (rare, local-only):
/// stops the `Raft`, unregisters the group, and tombstones it.
/// Idempotent: already-retired sources bump the tombstone (or answer
/// success when never hosted, which needs no fencing).
async fn serve_retire(
    build: &WorkerBuildCtx,
    replicas: &mut HashMap<ConsensusGroupId, OwnerCtx<GroupRaftStore>>,
    group: ConsensusGroupId,
    tablet: TabletId,
    generation: u64,
    reply: futures::channel::oneshot::Sender<Result<(), String>>,
) {
    if group != ConsensusGroupId::of_tablet(tablet) {
        let _ = reply.send(Err(format!("group {group} names a foreign tablet")));
        return;
    }
    if let Some(ctx) = replicas.remove(&group) {
        ctx.shutdown_raft().await;
        write_tombstone(&build.data_dir, tablet, generation);
        let _ = reply.send(Ok(()));
        return;
    }
    // Idempotent: already retired or never hosted. Bump an existing
    // tombstone so a newer retirement still fences older creates;
    // write nothing when no tombstone exists (a future create is
    // fresh, not a resurrection).
    bump_tombstone(&build.data_dir, tablet, generation);
    let _ = reply.send(Ok(()));
}

/// Serves one group initialization inline (rare, local-only): the founder
/// replica commits the initial membership for a split child or merge
/// target. Idempotent: an already-initialized group answers success
/// (`OpenRaft` reports the existing state; we treat any
/// already-initialized signal as success, never a fork).
async fn serve_initialize(
    replicas: &mut HashMap<ConsensusGroupId, OwnerCtx<GroupRaftStore>>,
    group: ConsensusGroupId,
    tablet: TabletId,
    voters: &BTreeSet<u64>,
    peer_addrs: &BTreeMap<u64, String>,
    reply: futures::channel::oneshot::Sender<Result<(), String>>,
) {
    if group != ConsensusGroupId::of_tablet(tablet) {
        let _ = reply.send(Err(format!("group {group} names a foreign tablet")));
        return;
    }
    let Some(ctx) = replicas.get(&group) else {
        let _ = reply.send(Err(format!("group {group} not served by this node")));
        return;
    };
    let members: BTreeMap<u64, openraft::BasicNode> = voters
        .iter()
        .map(|voter| {
            (
                *voter,
                openraft::BasicNode::new(peer_addrs.get(voter).cloned().unwrap_or_default()),
            )
        })
        .collect();
    match ctx.raft.initialize(members).await {
        Ok(()) => {
            let _ = reply.send(Ok(()));
        }
        Err(error) => {
            let detail = error.to_string();
            // Idempotent: a second initialize (after a retry or a
            // failover where the first already committed) is success.
            if detail.contains("already")
                || detail.contains("initialized")
                || detail.contains("exists")
                || detail.contains("NotAllowed")
            {
                let _ = reply.send(Ok(()));
            } else {
                let _ = reply.send(Err(detail));
            }
        }
    }
}

/// Writes a retirement tombstone (`generation` little-endian).
fn write_tombstone(data_dir: &std::path::Path, tablet: TabletId, generation: u64) {
    let dir = data_dir
        .join("consensus-sm")
        .join(tablet.as_u64().to_string());
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let _ = std::fs::write(
        dir.join(ConsensusNode::TOMBSTONE_FILE),
        generation.to_le_bytes(),
    );
}

/// Unions two parent session sets for a merge target: sessions keyed by
/// id, outcomes keyed by sequence (globally unique per session, so no
/// collision in practice). A conflicting duplicate keeps the higher
/// commit and never re-executes — ordinary key state cannot conflict
/// because the parent ranges are disjoint.
fn union_sessions(
    left: Vec<(
        kivi_types::SessionId,
        kivi_types::RequestSeq,
        BTreeMap<kivi_types::RequestSeq, crate::state_machine::RetainedOutcome>,
    )>,
    right: Vec<(
        kivi_types::SessionId,
        kivi_types::RequestSeq,
        BTreeMap<kivi_types::RequestSeq, crate::state_machine::RetainedOutcome>,
    )>,
) -> Vec<(
    kivi_types::SessionId,
    kivi_types::RequestSeq,
    BTreeMap<kivi_types::RequestSeq, crate::state_machine::RetainedOutcome>,
)> {
    let mut merged: BTreeMap<
        u128,
        (
            kivi_types::SessionId,
            kivi_types::RequestSeq,
            BTreeMap<kivi_types::RequestSeq, crate::state_machine::RetainedOutcome>,
        ),
    > = BTreeMap::new();
    for (session, floor, outcomes) in left.into_iter().chain(right) {
        merged
            .entry(session.as_u128())
            .and_modify(|existing| {
                if floor.as_u64() > existing.1.as_u64() {
                    existing.1 = floor;
                }
                for (seq, entry) in &outcomes {
                    match existing.2.get(seq) {
                        Some(kept) if kept.commit.as_u64() >= entry.commit.as_u64() => {}
                        _ => {
                            existing.2.insert(*seq, entry.clone());
                        }
                    }
                }
                existing
                    .2
                    .retain(|seq, _| seq.as_u64() > existing.1.as_u64());
            })
            .or_insert((session, floor, outcomes));
    }
    merged.into_values().collect()
}

/// Bumps an existing tombstone to `generation` when older; writes
/// nothing when no tombstone exists.
fn bump_tombstone(data_dir: &std::path::Path, tablet: TabletId, generation: u64) {
    let path = data_dir
        .join("consensus-sm")
        .join(tablet.as_u64().to_string())
        .join(ConsensusNode::TOMBSTONE_FILE);
    let Ok(bytes) = std::fs::read(&path) else {
        return;
    };
    let retired = u64::from_le_bytes(bytes[..8.min(bytes.len())].try_into().unwrap_or([0; 8]));
    if generation > retired {
        let _ = std::fs::write(path, generation.to_le_bytes());
    }
}

fn control_forward_leader(detail: &str) -> Option<NodeId> {
    let suffix = detail.split_once("control forward: leader ")?.1;
    let value = suffix
        .split(|character: char| !character.is_ascii_digit())
        .next()?;
    value.parse().ok().map(NodeId::from_u64)
}

/// Validates the configured tablet set: unique, nonzero, and every
/// tablet assigned to the local node in the static topology. Empty is
/// valid: a joining data node starts empty and gains replicas through
/// control-plane migration (§9, §47). Returns the sorted tablets.
fn validated_tablets(config: &MultiNodeConfig) -> Result<Vec<TabletId>, NodeOpenError> {
    use NodeOpenError as Fault;
    if config.worker_count == 0 {
        return Err(Fault::Topology {
            reason: "worker count must be nonzero".to_owned(),
        });
    }
    let mut tablets = config.tablets.clone();
    tablets.sort_by_key(|tablet| tablet.as_u64());
    tablets.dedup();
    if tablets.len() != config.tablets.len() {
        return Err(Fault::Topology {
            reason: "duplicate tablet assignment".to_owned(),
        });
    }
    for tablet in &tablets {
        if tablet.as_u64() == 0 {
            return Err(Fault::Topology {
                reason: "tablet 0 is reserved for group-independent bulk routing".to_owned(),
            });
        }
        let assignment = config
            .topology
            .assignment(*tablet)
            .ok_or_else(|| Fault::Topology {
                reason: format!("no tablet assignment for tablet {}", tablet.as_u64()),
            })?;
        if !assignment.replicas.contains(&config.local) {
            return Err(Fault::Topology {
                reason: format!(
                    "node {} replicates no tablet {}",
                    config.local.as_u64(),
                    tablet.as_u64()
                ),
            });
        }
    }
    Ok(tablets)
}

/// Reads the durable tablet set: numeric subdirectories of
/// `<data_dir>/consensus-sm` (one per opened state machine). Absent
/// directory means a fresh node (empty set).
fn existing_sm_tablets(data_dir: &std::path::Path) -> BTreeSet<TabletId> {
    let mut out = BTreeSet::new();
    let Ok(entries) = std::fs::read_dir(data_dir.join("consensus-sm")) else {
        return out;
    };
    for entry in entries.flatten() {
        if !entry.path().is_dir() {
            continue;
        }
        if let Some(name) = entry.file_name().to_str()
            && let Ok(id) = name.parse::<u64>()
            && id != 0
        {
            out.insert(TabletId::from_u64(id));
        }
    }
    out
}

/// Parameters crossing into the mesh thread at spawn. Everything is
/// owned (`Send`); the Compio reactor is built inside.
struct MeshParams {
    transport_config: TransportConfig,
    peer_identity: crate::peer::PeerIdentity,
    cluster: ClusterId,
    peers: HashMap<NodeId, std::net::SocketAddr>,
    tls: TlsMaterial,
    registry: GroupRegistry,
    ready: futures::channel::oneshot::Sender<
        Result<(PeerTransport, std::net::SocketAddr), NodeOpenError>,
    >,
    shutdown: futures::channel::oneshot::Receiver<()>,
}

/// Mesh thread main: opens the shared H3 mesh on its own Compio reactor,
/// signals readiness with the shared transport, then parks until shutdown.
async fn mesh_main(params: MeshParams) {
    use NodeOpenError as Fault;
    let MeshParams {
        transport_config,
        peer_identity,
        cluster,
        peers,
        tls,
        registry,
        ready,
        shutdown,
    } = params;
    let handler: Arc<dyn PeerHandler> = Arc::new(registry);
    let started = PeerTransport::open(
        transport_config,
        peer_identity,
        cluster,
        &peers,
        tls,
        handler,
    )
    .await;
    let (transport, peer_addr) = match started {
        Ok(started) => started,
        Err(error) => {
            let _ = ready.send(Err(Fault::Transport {
                reason: error.to_string(),
            }));
            return;
        }
    };
    if ready.send(Ok((transport.clone(), peer_addr))).is_err() {
        tracing::warn!("mesh opener gone; shutting down mesh");
        transport.shutdown().await;
        return;
    }
    tracing::info!(peer = %peer_addr, "peer mesh open");
    let _ = shutdown.await;
    tracing::info!("mesh shutdown signaled; stopping transport");
    transport.shutdown().await;
    tracing::info!("peer mesh stopped");
}

/// Parameters crossing into one worker thread at spawn. Everything is
/// owned (`Send`); the `!Send` Raft handles are built inside on the
/// worker's own Compio reactor.
struct WorkerParams {
    index: usize,
    tablets: Vec<TabletId>,
    stores: HashMap<TabletId, GroupRaftStore>,
    machines: HashMap<TabletId, ReplicatedStateMachine>,
    authorities: HashMap<TabletId, TabletAuthority>,
    wirings: HashMap<TabletId, GroupWiring>,
    topology: ClusterTopology,
    transport_config: TransportConfig,
    transport: PeerTransport,
    sidecar: SidecarStore,
    /// Node-local fragment store (shared by every replica on the node).
    fragments: Option<Arc<LocalFragmentStore>>,
    namespace: NamespaceId,
    local: NodeId,
    cluster: ClusterId,
    preflight: Arc<crate::preflight::PreflightMetrics>,
    preflight_enabled: bool,
    /// Shared consistency hub (lease engines, evidence, metrics).
    hub: std::sync::Arc<crate::consistency::ConsistencyHub>,
    /// This process's incarnation (lease/proof binding).
    incarnation: kivi_types::NodeIncarnation,
    /// Own ingress channel for the lease-tick timer task.
    tick_tx: async_channel::Sender<OwnerRequest>,
    /// Data-directory root (dynamic replica creation opens new group
    /// stores and state machines beneath it).
    data_dir: PathBuf,
    /// Shared WAL durability (dynamic replicas attach fresh logical
    /// views to the same physical lane).
    durability: SharedRaftDurability,
    /// Control seeds proving the cluster already exists. Only read for
    /// the control group on a fresh directory: empty means this node
    /// founds the control plane (initialize), set means it joins
    /// (waits for `add_learner`, never forks history).
    control_seeds: Vec<std::net::SocketAddr>,
    /// True only on a brand-new data directory: the sole boot that may
    /// initialize data groups. Rejoins recover or wait for learner
    /// replication, never initialize (that would fork live history).
    founding: bool,
    requests: async_channel::Receiver<OwnerRequest>,
    ready: Option<futures::channel::oneshot::Sender<Result<(), NodeOpenError>>>,
}

/// Owned context a worker keeps for building replicas after open
/// (migration targets). Everything is worker-local owned or shared
/// handles; the `!Send` Raft handles are built on this reactor.
#[derive(Clone)]
struct WorkerBuildCtx {
    data_dir: PathBuf,
    namespace: NamespaceId,
    local: NodeId,
    incarnation: kivi_types::NodeIncarnation,
    sidecar: SidecarStore,
    fragments: Option<Arc<LocalFragmentStore>>,
    transport: PeerTransport,
    router: PeerRouter,
    preflight: Arc<crate::preflight::PreflightMetrics>,
    preflight_enabled: bool,
    bulk_timeout: Duration,
    durability: SharedRaftDurability,
    hub: std::sync::Arc<crate::consistency::ConsistencyHub>,
}

/// Worker thread main: builds one replica per owned tablet (gate, `Raft`,
/// bootstrap), then serves the worker ingress loop until `Shutdown`.
async fn worker_main(mut params: WorkerParams) {
    let mut ready_opt = params.ready.take();
    // Raft config is validated per group inside `spawn_raft`
    // (fail-closed there); no separate pre-check here.
    let router = PeerRouter::new(params.transport.clone());
    let mut replicas: HashMap<ConsensusGroupId, OwnerCtx<GroupRaftStore>> = HashMap::new();
    for tablet in params.tablets.clone() {
        let Some(store) = params.stores.remove(&tablet) else {
            fail_ready(
                &mut ready_opt,
                format!(
                    "worker {} missing store for tablet {}",
                    params.index,
                    tablet.as_u64()
                ),
            );
            return;
        };
        match build_replica(&params, tablet, &router, store).await {
            Ok((group, ctx)) => {
                replicas.insert(group, ctx);
            }
            Err(error) => {
                fail_ready(
                    &mut ready_opt,
                    format!("worker {} replica failed: {error}", params.index),
                );
                return;
            }
        }
    }
    let ready_ok = ready_opt.take().is_some_and(|tx| tx.send(Ok(())).is_ok());
    if !ready_ok {
        for ctx in replicas.values() {
            ctx.shutdown_raft().await;
        }
        return;
    }
    tracing::info!(
        worker = params.index,
        groups = replicas.len(),
        "consensus worker serving"
    );
    let build = WorkerBuildCtx {
        data_dir: params.data_dir.clone(),
        namespace: params.namespace,
        local: params.local,
        incarnation: params.incarnation,
        sidecar: params.sidecar.clone(),
        fragments: params.fragments.clone(),
        transport: params.transport.clone(),
        router: router.clone(),
        preflight: Arc::clone(&params.preflight),
        preflight_enabled: params.preflight_enabled,
        bulk_timeout: params.transport_config.bulk_timeout,
        durability: params.durability.clone(),
        hub: std::sync::Arc::clone(&params.hub),
    };
    // Lease-driver ticker: one tick per renew quarter per owned group.
    // The loop dies with the reactor on shutdown; a full queue (shutting
    // down) ends it.
    {
        let tick_tx = params.tick_tx.clone();
        let tick_interval = crate::node::lease_tick_interval(params.hub.lease_params());
        let tick_groups: Vec<ConsensusGroupId> = replicas.keys().copied().collect();
        compio::runtime::spawn(async move {
            loop {
                compio::time::sleep(tick_interval).await;
                let mut live = true;
                for group in &tick_groups {
                    if tick_tx
                        .send(OwnerRequest::LeaseTick { group: *group })
                        .await
                        .is_err()
                    {
                        live = false;
                        break;
                    }
                }
                if !live {
                    break;
                }
            }
        })
        .detach();
    }
    worker_loop(
        replicas,
        params.local,
        params.transport,
        build,
        params.requests,
    )
    .await;
}

/// Reports worker startup failure exactly once (first failure wins; later
/// calls are silent no-ops so cleanup paths stay simple).
fn fail_ready(
    ready: &mut Option<futures::channel::oneshot::Sender<Result<(), NodeOpenError>>>,
    reason: String,
) {
    if let Some(tx) = ready.take() {
        let _ = tx.send(Err(NodeOpenError::Owner { reason }));
    }
}

/// Builds one tablet replica on the worker reactor: installs the sidecar
/// durability gate, spawns its `Raft` on the shared router, and bootstraps
/// its group (fresh formation installs membership exactly once; restarts
/// recover and verify, never re-bootstrap).
///
/// # Errors
///
/// Returns [`NodeOpenError`] when the tablet lacks wiring or the spawn or
/// bootstrap fails.
// One linear build sequence (gate, spawn, bootstrap, assemble); splitting
// it would scatter the replica construction across helpers for no gain.
#[allow(clippy::too_many_lines)]
async fn build_replica(
    params: &WorkerParams,
    tablet: TabletId,
    router: &PeerRouter,
    mut store: GroupRaftStore,
) -> Result<(ConsensusGroupId, OwnerCtx<GroupRaftStore>), NodeOpenError> {
    let group = ConsensusGroupId::of_tablet(tablet);
    let Some(machine) = params.machines.get(&tablet).cloned() else {
        return Err(NodeOpenError::Topology {
            reason: format!(
                "worker {} missing machine for tablet {}",
                params.index,
                tablet.as_u64()
            ),
        });
    };
    let Some(authority) = params.authorities.get(&tablet).copied() else {
        return Err(NodeOpenError::Topology {
            reason: format!(
                "worker {} missing authority for tablet {}",
                params.index,
                tablet.as_u64()
            ),
        });
    };
    let Some(wiring) = params.wirings.get(&tablet) else {
        return Err(NodeOpenError::Topology {
            reason: format!(
                "worker {} missing wiring for tablet {}",
                params.index,
                tablet.as_u64()
            ),
        });
    };
    // Sidecar durability gate: per-group voters minus local are candidate
    // sources (leader-first at fetch time). Installed before Raft spawns
    // so no append can report durable persistence without its sidecars.
    let mut sources: Vec<NodeId> = wiring
        .voters
        .iter()
        .map(|voter| NodeId::from_u64(*voter))
        .filter(|peer| *peer != params.local)
        .collect();
    sources.sort_by_key(|peer| peer.as_u64());
    let domain = kivi_types::SecurityDomainId::from_u64(params.namespace.as_u64());
    let gate = SidecarGate::new(
        params.sidecar.clone(),
        params.transport.clone(),
        sources,
        group,
        domain,
        params.transport_config.bulk_timeout,
    );
    store.set_gate(gate.clone());
    let raft = spawn_raft(params.local, group, router.clone(), &store, &machine).await?;
    if tablet == ConsensusGroupId::control().tablet() {
        bootstrap_control(
            &wiring.voters,
            &wiring.peer_addrs,
            &params.control_seeds,
            params.founding,
            &mut store,
            &machine,
            &raft,
        )
        .await?;
    } else {
        bootstrap_group(
            &BootstrapInputs {
                topology: &params.topology,
                tablet,
                local: params.local,
                cluster: params.cluster,
                node: params.local,
                voters: &wiring.voters,
                peer_addrs: &wiring.peer_addrs,
            },
            params.founding,
            &mut store,
            &machine,
            &raft,
        )
        .await?;
    }
    let voter_list = sorted_voters(&wiring.voters);
    Ok((
        group,
        assemble_ctx(
            raft,
            store,
            machine,
            params.sidecar.clone(),
            gate,
            params.fragments.clone(),
            params.transport.clone(),
            group,
            params.namespace,
            authority,
            tablet,
            params.local,
            params.incarnation,
            voter_list,
            params.transport_config.bulk_timeout,
            Arc::clone(&params.preflight),
            params.preflight_enabled,
            std::sync::Arc::clone(&params.hub),
        ),
    ))
}

/// Sorts voter ids into deterministic quorum order.
fn sorted_voters(voters: &BTreeSet<u64>) -> Vec<NodeId> {
    let mut list: Vec<NodeId> = voters
        .iter()
        .map(|voter| NodeId::from_u64(*voter))
        .collect();
    list.sort_by_key(|node| node.as_u64());
    list
}

/// Whether the local node may serve a recovered group: true when no
/// durable voter set survives (nothing stale to serve) or the durable
/// set still names it. A set excluding it means either a late joiner
/// (resolved through the control desired oracle by the caller) or a
/// removal (tombstoned by the caller).
async fn durable_includes(
    store: &GroupRaftStore,
    machine: &ReplicatedStateMachine,
    local: NodeId,
) -> bool {
    let mut voters = store.membership_voters().await;
    if voters.is_empty() {
        voters = machine.member_voters().await;
    }
    voters.is_empty() || voters.contains(&local.as_u64())
}

/// Installs or recovers control-group membership.
///
/// Unlike tablet groups, control membership evolves: startup flags are
/// seeds, never truth. First formation (fresh directory, no seeds)
/// installs the seed membership exactly once; joins (fresh directory
/// with seeds) skip initialization and wait for the control leader's
/// `add_learner`; restarts recover durable state and never
/// re-initialize or verify against stale flags.
///
/// # Errors
///
/// Returns [`NodeOpenError::Initialize`] when first formation fails.
async fn bootstrap_control(
    voters: &BTreeSet<u64>,
    peer_addrs: &BTreeMap<u64, String>,
    seeds: &[std::net::SocketAddr],
    founding: bool,
    store: &mut GroupRaftStore,
    machine: &ReplicatedStateMachine,
    raft: &crate::node::OwnerRaft,
) -> Result<(), NodeOpenError> {
    use NodeOpenError as Fault;
    match classify_from_store(store, machine).await {
        Bootstrap::Existing => Ok(()),
        // A fresh control store joins (never initializes) unless this
        // is a founding boot with no seeds proving otherwise.
        Bootstrap::Fresh if !founding || !seeds.is_empty() => Ok(()),
        Bootstrap::Fresh => {
            let members: BTreeMap<u64, openraft::BasicNode> = voters
                .iter()
                .map(|voter| {
                    (
                        *voter,
                        openraft::BasicNode::new(
                            peer_addrs.get(voter).cloned().unwrap_or_default(),
                        ),
                    )
                })
                .collect();
            raft.initialize(members)
                .await
                .map_err(|error| Fault::Initialize {
                    reason: error.to_string(),
                })?;
            Ok(())
        }
    }
}

/// Assembles one replica owner context: the single construction site
/// for initial and dynamic replica builds.
#[allow(clippy::too_many_arguments)]
fn assemble_ctx(
    raft: crate::node::OwnerRaft,
    store: GroupRaftStore,
    machine: ReplicatedStateMachine,
    sidecar: SidecarStore,
    gate: SidecarGate,
    fragments: Option<Arc<LocalFragmentStore>>,
    transport: PeerTransport,
    group: ConsensusGroupId,
    namespace: NamespaceId,
    authority: TabletAuthority,
    tablet: TabletId,
    local: NodeId,
    incarnation: kivi_types::NodeIncarnation,
    voters: Vec<NodeId>,
    bulk_timeout: Duration,
    preflight: Arc<crate::preflight::PreflightMetrics>,
    preflight_enabled: bool,
    hub: std::sync::Arc<crate::consistency::ConsistencyHub>,
) -> OwnerCtx<GroupRaftStore> {
    OwnerCtx {
        raft,
        store,
        machine,
        sidecar,
        gate,
        fragments,
        transport,
        group,
        namespace,
        authority,
        tablet,
        local,
        incarnation,
        voters,
        bulk_timeout,
        preflight,
        preflight_enabled,
        transfers: Rc::new(RefCell::new(HashMap::new())),
        hub,
    }
}

/// Builds one dynamic replica on a running worker (migration target):
/// fresh group store and state machine, gate, and an uninitialized
/// `Raft` that the tablet leader feeds via learner replication. Never
/// initializes membership (that would fork the group).
async fn build_dynamic_replica(
    build: &WorkerBuildCtx,
    tablet: TabletId,
    voters: &BTreeSet<u64>,
    generation: u64,
) -> Result<
    (
        ConsensusGroupId,
        OwnerCtx<GroupRaftStore>,
        crate::node::NewReplica,
    ),
    String,
> {
    let group = ConsensusGroupId::of_tablet(tablet);
    if group.is_control() || tablet.as_u64() == 0 {
        return Err(format!("tablet {} is reserved", tablet.as_u64()));
    }
    // Fencing: a stale create never resurrects a newer retirement.
    let tomb_path = build
        .data_dir
        .join("consensus-sm")
        .join(tablet.as_u64().to_string())
        .join(ConsensusNode::TOMBSTONE_FILE);
    if let Ok(bytes) = std::fs::read(&tomb_path) {
        let retired = u64::from_le_bytes(bytes[..8.min(bytes.len())].try_into().unwrap_or([0; 8]));
        if generation <= retired {
            return Err(format!(
                "stale create for tablet {} (generation {generation} <= retired {retired})",
                tablet.as_u64(),
            ));
        }
        let _ = std::fs::remove_file(&tomb_path);
    }
    let authority =
        TabletAuthority::new(tablet, TabletEpoch::INITIAL, WriteGuardGeneration::INITIAL);
    let mut store = GroupRaftStore::from_recovered(build.namespace, group, &build.durability, &[])
        .map_err(|error| format!("dynamic group store failed: {error}"))?;
    let machine = ReplicatedStateMachine::open(&build.data_dir, build.namespace, tablet, authority)
        .map_err(|error| format!("dynamic state machine failed: {error}"))?;
    let mut sources: Vec<NodeId> = voters
        .iter()
        .map(|voter| NodeId::from_u64(*voter))
        .filter(|peer| *peer != build.local)
        .collect();
    sources.sort_by_key(|peer| peer.as_u64());
    let domain = kivi_types::SecurityDomainId::from_u64(build.namespace.as_u64());
    let gate = SidecarGate::new(
        build.sidecar.clone(),
        build.transport.clone(),
        sources,
        group,
        domain,
        build.bulk_timeout,
    );
    store.set_gate(gate.clone());
    let raft = spawn_raft(build.local, group, build.router.clone(), &store, &machine)
        .await
        .map_err(|error| format!("dynamic raft spawn failed: {error}"))?;
    let voter_list = sorted_voters(voters);
    let info = crate::node::NewReplica {
        machine: machine.clone(),
        authority,
    };
    let ctx = assemble_ctx(
        raft,
        store,
        machine,
        build.sidecar.clone(),
        gate,
        build.fragments.clone(),
        build.transport.clone(),
        group,
        build.namespace,
        authority,
        tablet,
        build.local,
        build.incarnation,
        voter_list,
        build.bulk_timeout,
        Arc::clone(&build.preflight),
        build.preflight_enabled,
        std::sync::Arc::clone(&build.hub),
    );
    Ok((group, ctx, info))
}

/// Serves one worker's ingress until `Shutdown`: group-scoped requests
/// dispatch to the owning replica's [`OwnerCtx`] (each on its own task —
/// a pending large-write preflight never blocks small writes or peer
/// RPCs); node-wide suspend/resume run against the shared transport;
/// dynamic [`EnsureGroup`](OwnerRequest::EnsureGroup) /
/// [`RetireGroup`](OwnerRequest::RetireGroup) run inline (rare,
/// local-only, never network: a brief ingress pause while a replica
/// builds or stops, never a data-path stall);
/// `Shutdown` breaks the loop and stops every local `Raft`.
async fn worker_loop(
    mut replicas: HashMap<ConsensusGroupId, OwnerCtx<GroupRaftStore>>,
    local: NodeId,
    transport: PeerTransport,
    build: WorkerBuildCtx,
    requests: async_channel::Receiver<OwnerRequest>,
) {
    while let Ok(request) = requests.recv().await {
        match request {
            OwnerRequest::Shutdown => break,
            OwnerRequest::EnsureGroup {
                group,
                tablet,
                voters,
                generation,
                reply,
            } => {
                serve_ensure(
                    &build,
                    &mut replicas,
                    group,
                    tablet,
                    &voters,
                    generation,
                    reply,
                )
                .await;
            }
            OwnerRequest::RetireGroup {
                group,
                tablet,
                generation,
                reply,
            } => {
                serve_retire(&build, &mut replicas, group, tablet, generation, reply).await;
            }
            OwnerRequest::InitializeGroup {
                group,
                tablet,
                voters,
                peer_addrs,
                reply,
            } => {
                serve_initialize(&mut replicas, group, tablet, &voters, &peer_addrs, reply).await;
            }
            OwnerRequest::SuspendPeer { peer, reply } => {
                let transport = transport.clone();
                compio::runtime::spawn(async move {
                    transport.suspend_peer(peer).await;
                    let _ = reply.send(());
                })
                .detach();
            }
            OwnerRequest::ResumePeer { peer, reply } => {
                let transport = transport.clone();
                compio::runtime::spawn(async move {
                    transport.resume_peer(peer).await;
                    let _ = reply.send(());
                })
                .detach();
            }
            other => {
                // Group-independent bulk traffic decodes with a zero
                // tablet: any local replica serves it from the shared
                // sidecar/fragment stores (the registry only sends these
                // to a worker that owns at least one group).
                let dest = match &other {
                    OwnerRequest::Serve { group, request, .. }
                        if *group == ConsensusGroupId::redundancy_sentinel()
                            && matches!(
                                request,
                                PeerRequest::Manifest(_)
                                    | PeerRequest::Chunk(_)
                                    | PeerRequest::Fragment(_)
                            ) =>
                    {
                        replicas.values().next()
                    }
                    _ => other.group().and_then(|group| replicas.get(&group)),
                };
                if let Some(ctx) = dest {
                    let ctx = ctx.clone();
                    compio::runtime::spawn(async move {
                        ctx.serve_one(other).await;
                    })
                    .detach();
                } else {
                    tracing::warn!(
                        "consensus worker has no replica for group {:?}; refusing loudly",
                        other.group()
                    );
                    other.reply_unroutable(local);
                }
            }
        }
    }
    for ctx in replicas.values() {
        ctx.shutdown_raft().await;
    }
}

#[cfg(test)]
mod tests {
    use kivi_state::{Key, OperationResult};

    use super::*;

    const NS: NamespaceId = NamespaceId::from_u64(1);
    const NOW: WallTimestamp = WallTimestamp::from_micros(1_000_000);
    const CTX: kivi_types::ReadContext = kivi_types::ReadContext::new(
        NOW,
        kivi_types::Ticks::from_micros(1_000_000),
        std::time::Duration::from_secs(5),
    );

    fn block_on<F, T>(future: F) -> T
    where
        F: Future<Output = T>,
    {
        compio::runtime::Runtime::new()
            .expect("compio test runtime")
            .block_on(future)
    }

    fn probe_udp() -> std::net::SocketAddr {
        let socket = std::net::UdpSocket::bind("127.0.0.1:0").expect("probe binds");
        socket.local_addr().expect("probe addr")
    }

    /// Single-node topology over the given tablets (one voter: itself).
    fn single_topology(addr: std::net::SocketAddr, tablets: &[TabletId]) -> ClusterTopology {
        use crate::cluster::{NodeDescriptor, TabletAssignment};
        ClusterTopology {
            cluster: ClusterId::from_u128(0x0C10_57E2),
            nodes: vec![NodeDescriptor {
                node: NodeId::from_u64(1),
                peer: addr,
                native: "127.0.0.1:9001".parse().expect("addr"),
            }],
            tablets: tablets
                .iter()
                .map(|tablet| TabletAssignment {
                    tablet: *tablet,
                    replicas: vec![NodeId::from_u64(1)],
                })
                .collect(),
        }
    }

    /// Proposes until a leader emerges (single voter self-elects after its
    /// election timeout) or the deadline passes.
    async fn propose_until_leader(
        node: &ConsensusNode,
        tablet: TabletId,
        op: &Operation,
    ) -> crate::node::ProposeOutcome {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            match node.propose(tablet, op, None, None, NOW).await {
                Ok(outcome) => return outcome,
                Err(error) => {
                    let retryable = matches!(
                        error,
                        ProposeError::Consensus(
                            ConsensusError::NotLeader { .. } | ConsensusError::LeaderUnknown
                        )
                    );
                    assert!(
                        retryable,
                        "proposal fails only retryably while electing: {error:?}"
                    );
                    assert!(
                        std::time::Instant::now() < deadline,
                        "no leader emerged: {error:?}"
                    );
                    compio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            }
        }
    }

    /// Two tablets on two workers of one node share the mesh and the WAL
    /// writer while keeping independent Raft histories.
    #[test]
    fn two_tablets_two_workers_share_mesh_and_writer() {
        block_on(async {
            let dir = tempfile::tempdir().expect("scratch");
            let addr = probe_udp();
            let tablets = vec![TabletId::from_u64(1), TabletId::from_u64(2)];
            let topology = single_topology(addr, &tablets);
            let node = ConsensusNode::open(MultiNodeConfig {
                data_dir: dir.path().to_owned(),
                namespace: NS,
                tablets: tablets.clone(),
                local: NodeId::from_u64(1),
                topology,
                segment_target_bytes: 1024 * 1024,
                transport: TransportConfig::default(),
                peer_certs: HashMap::new(),
                insecure_peer_tls: true,
                preflight_enabled: true,
                lease_params: kivi_types::LeaseParams::default(),
                worker_count: 2,
                durability: SharedDurabilityConfig::default(),
                control: None,
                fragment_store_root: None,
            })
            .await
            .expect("multi opens");
            assert_eq!(node.tablets(), tablets);
            assert_eq!(node.worker_count(), 2);
            // Deterministic worker striping: tablet 1 on worker 0.
            assert_eq!(crate::worker::worker_for_tablet(tablets[0], 2), 0);
            assert_eq!(crate::worker::worker_for_tablet(tablets[1], 2), 1);
            // Independent histories: write different keys per tablet.
            for (tablet, key) in [(tablets[0], "a"), (tablets[1], "b")] {
                let outcome = propose_until_leader(
                    &node,
                    tablet,
                    &Operation::Set {
                        key: Key::from(key),
                        value: bytes::Bytes::from_static(b"v"),
                    },
                )
                .await;
                assert!(
                    matches!(outcome, crate::node::ProposeOutcome::Applied { .. }),
                    "tablet {} applies, got {outcome:?}",
                    tablet.as_u64()
                );
            }
            // Each tablet reads its own key (linearizable on the leader).
            for (tablet, key) in [(tablets[0], "a"), (tablets[1], "b")] {
                let read = node
                    .read(
                        tablet,
                        &Operation::Get {
                            key: Key::from(key),
                        },
                        ReadContract::Latest,
                        CTX,
                        kivi_types::LeaseEligibility::Eligible,
                    )
                    .await
                    .expect("tablet serves its key");
                assert_eq!(
                    read.outcome,
                    OperationResult::Value(Some(bytes::Bytes::from_static(b"v"))),
                    "tablet {} serves its key",
                    tablet.as_u64()
                );
            }
            // Both groups healthy with independent leadership; the shared
            // writer batched their barriers physically.
            let statuses = node.status_all().await;
            assert_eq!(statuses.len(), 2, "one status per group");
            assert!(statuses.iter().all(|status| status.healthy), "both healthy");
            let metrics = node.durability_metrics();
            assert!(metrics.physical_batches >= 1, "writer sealed batches");
            assert!(metrics.records >= 2, "both groups persisted: {metrics:?}");
            node.shutdown().await;
        });
    }

    /// Control plane plus dynamic lifecycle on one node: the system
    /// control group commits typed mutations, a new tablet replica is
    /// created without restart, observed, retired with a tombstone, and
    /// stays retired across reopen while control state recovers.
    ///
    /// Long because it walks the whole lifecycle in one scenario (each
    /// phase builds on the last); splitting it would re-pay open/boot
    /// costs per phase for no isolation gain.
    #[allow(clippy::too_many_lines)]
    #[test]
    fn control_plane_propose_observe_and_dynamic_lifecycle() {
        block_on(async {
            let dir = tempfile::tempdir().expect("scratch");
            let addr = probe_udp();
            let tablets = vec![TabletId::from_u64(1)];
            let topology = single_topology(addr, &tablets);
            let voters = BTreeSet::from([1u64]);
            let peer_addrs = BTreeMap::from([(1u64, addr.to_string())]);
            let config = || MultiNodeConfig {
                data_dir: dir.path().to_owned(),
                namespace: NS,
                tablets: tablets.clone(),
                local: NodeId::from_u64(1),
                topology: topology.clone(),
                segment_target_bytes: 1024 * 1024,
                transport: TransportConfig::default(),
                peer_certs: HashMap::new(),
                insecure_peer_tls: true,
                preflight_enabled: true,
                lease_params: kivi_types::LeaseParams::default(),
                worker_count: 1,
                durability: SharedDurabilityConfig::default(),
                control: Some(ControlGroupConfig {
                    voters: voters.clone(),
                    peer_addrs: peer_addrs.clone(),
                    seeds: Vec::new(),
                }),
                fragment_store_root: None,
            };
            let node = ConsensusNode::open(config())
                .await
                .expect("multi with control opens");
            assert!(node.hosts_control());
            // Single voter self-elects; control proposals commit.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            loop {
                let record = kivi_control::NodeRecord {
                    node: NodeId::from_u64(1),
                    peer: addr,
                    native: "127.0.0.1:9001".parse().expect("addr"),
                    admin: "127.0.0.1:19001".parse().expect("addr"),
                    cert_fingerprint: [3u8; 32],
                    failure_domain: String::new(),
                    weight: 1,
                    state: kivi_control::NodeState::Joining,
                };
                if node
                    .propose_control(kivi_control::ControlMutation::RegisterNode { record })
                    .await
                    .is_ok()
                {
                    break;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "control never elected"
                );
                compio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
            node.propose_control(kivi_control::ControlMutation::SetNodeState {
                node: NodeId::from_u64(1),
                state: kivi_control::NodeState::Active,
            })
            .await
            .expect("activate commits");
            let state = node.control_state().await.expect("control image");
            assert_eq!(
                state.node(NodeId::from_u64(1)).expect("registered").state,
                kivi_control::NodeState::Active
            );
            // Dynamic target creation without restart.
            let fresh = TabletId::from_u64(2);
            node.ensure_group(fresh, BTreeSet::from([1u64]), 7)
                .await
                .expect("ensure creates");
            assert!(node.tablets().contains(&fresh));
            // A fresh replica is uninitialized by design (it must join
            // through learner replication, never fork with a second
            // initialize), so it observes empty membership until the
            // tablet leader replicates to it.
            let observed = node.observe_membership(fresh).await.expect("observed");
            assert!(observed.voters.is_empty());
            // Idempotent re-create answers success.
            node.ensure_group(fresh, BTreeSet::from([1u64]), 7)
                .await
                .expect("re-ensure succeeds");
            // Retirement tombstones; a stale create refuses resurrection.
            node.retire_group(fresh, 7).await.expect("retire succeeds");
            assert!(!node.tablets().contains(&fresh));
            assert!(node.observe_membership(fresh).await.is_err());
            assert_eq!(
                ConsensusNode::tombstone_generation(dir.path(), fresh),
                Some(7)
            );
            assert!(
                node.ensure_group(fresh, BTreeSet::from([1u64]), 6)
                    .await
                    .is_err()
            );
            node.shutdown().await;
            // Release the directory lock before reopening (shutdown
            // stops threads; drop releases the guard).
            drop(node);
            // Reopen: tombstoned tablet stays skipped, control state and
            // tablet 1 recover.
            let node = ConsensusNode::open(config())
                .await
                .expect("reopen skips tombstones");
            assert_eq!(node.tablets(), tablets);
            let state = node.control_state().await.expect("control recovers");
            assert_eq!(
                state
                    .node(NodeId::from_u64(1))
                    .expect("node recovers")
                    .state,
                kivi_control::NodeState::Active
            );
            node.shutdown().await;
        });
    }

    /// A control mutation proposed on a follower is forwarded to the
    /// control leader and applied on every control replica.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn control_forward_from_follower_converges_on_all_replicas() {
        block_on(async {
            use crate::cluster::{NodeDescriptor, TabletAssignment};
            let addrs = [probe_udp(), probe_udp(), probe_udp()];
            let topology = ClusterTopology {
                cluster: ClusterId::from_u128(0x0C10_57E2),
                nodes: addrs
                    .iter()
                    .enumerate()
                    .map(|(index, peer)| NodeDescriptor {
                        node: NodeId::from_u64(index as u64 + 1),
                        peer: *peer,
                        native: format!("127.0.0.1:{}", 9000 + index)
                            .parse()
                            .expect("native address parses"),
                    })
                    .collect(),
                tablets: vec![TabletAssignment {
                    tablet: TabletId::from_u64(1),
                    replicas: vec![
                        NodeId::from_u64(1),
                        NodeId::from_u64(2),
                        NodeId::from_u64(3),
                    ],
                }],
            };
            let voters = BTreeSet::from([1u64, 2, 3]);
            let peer_addrs = BTreeMap::from([
                (1, addrs[0].to_string()),
                (2, addrs[1].to_string()),
                (3, addrs[2].to_string()),
            ]);
            let mut nodes = Vec::new();
            let mut dirs = Vec::new();
            for (index, _peer) in addrs.iter().enumerate() {
                let dir = tempfile::tempdir().expect("scratch");
                let node = ConsensusNode::open(MultiNodeConfig {
                    data_dir: dir.path().to_owned(),
                    namespace: NS,
                    tablets: vec![TabletId::from_u64(1)],
                    local: NodeId::from_u64(index as u64 + 1),
                    topology: topology.clone(),
                    segment_target_bytes: 1024 * 1024,
                    transport: TransportConfig::default(),
                    peer_certs: HashMap::new(),
                    insecure_peer_tls: true,
                    preflight_enabled: true,
                    lease_params: kivi_types::LeaseParams::default(),
                    worker_count: 1,
                    durability: SharedDurabilityConfig::default(),
                    control: Some(ControlGroupConfig {
                        voters: voters.clone(),
                        peer_addrs: peer_addrs.clone(),
                        seeds: Vec::new(),
                    }),
                    fragment_store_root: None,
                })
                .await
                .expect("control replica opens");
                nodes.push(node);

                dirs.push(dir);
            }

            let group = ConsensusGroupId::control();
            let leader_deadline = std::time::Instant::now() + Duration::from_secs(30);
            let leader_index = loop {
                let statuses = futures::future::join_all(
                    nodes
                        .iter()
                        .map(|node| node.status_for(group))
                        .collect::<Vec<_>>(),
                )
                .await;
                let leaders: Vec<usize> = statuses
                    .iter()
                    .enumerate()
                    .filter(|(_, status)| status.role == crate::types::ReplicaRole::Leader)
                    .map(|(index, _)| index)
                    .collect();
                if leaders.len() == 1 && statuses.iter().all(|status| status.leader.is_some()) {
                    break leaders[0];
                }
                assert!(
                    std::time::Instant::now() < leader_deadline,
                    "control group did not elect a leader"
                );
                compio::time::sleep(Duration::from_millis(100)).await;
            };
            let follower_index = (leader_index + 1) % nodes.len();
            let record = kivi_control::NodeRecord {
                node: NodeId::from_u64(3),
                peer: addrs[2],
                native: "127.0.0.1:9003".parse().expect("address parses"),
                admin: "127.0.0.1:19003".parse().expect("address parses"),
                cert_fingerprint: [7u8; 32],
                failure_domain: String::new(),
                weight: 1,
                state: kivi_control::NodeState::Joining,
            };
            let mutation = kivi_control::ControlMutation::RegisterNode {
                record: record.clone(),
            };
            let index = nodes[follower_index]
                .propose_control_anywhere(mutation, Duration::from_secs(10))
                .await
                .expect("follower control proposal commits");
            assert!(index.get() > 0);
            let deadline = std::time::Instant::now() + Duration::from_secs(30);
            loop {
                let states = futures::future::join_all(
                    nodes
                        .iter()
                        .map(ConsensusNode::control_state)
                        .collect::<Vec<_>>(),
                )
                .await;
                if states.iter().all(|state| {
                    state.as_ref().is_some_and(|state| {
                        state
                            .node(record.node)
                            .is_some_and(|current| current.state == record.state)
                    })
                }) {
                    break;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "control mutation did not converge on all replicas"
                );
                compio::time::sleep(Duration::from_millis(100)).await;
            }
            for node in nodes {
                node.shutdown().await;
            }
            drop(dirs);
        });
    }

    /// Unknown tablets fail loudly (never route to a wrong group).
    #[test]
    fn unknown_tablet_is_rejected() {
        block_on(async {
            let dir = tempfile::tempdir().expect("scratch");
            let addr = probe_udp();
            let tablets = vec![TabletId::from_u64(1)];
            let topology = single_topology(addr, &tablets);
            let node = ConsensusNode::open(MultiNodeConfig {
                data_dir: dir.path().to_owned(),
                namespace: NS,
                tablets,
                local: NodeId::from_u64(1),
                topology,
                segment_target_bytes: 1024 * 1024,
                transport: TransportConfig::default(),
                peer_certs: HashMap::new(),
                insecure_peer_tls: true,
                preflight_enabled: true,
                lease_params: kivi_types::LeaseParams::default(),
                worker_count: 1,
                durability: SharedDurabilityConfig::default(),
                control: None,
                fragment_store_root: None,
            })
            .await
            .expect("multi opens");
            let missing = TabletId::from_u64(99);
            assert!(
                node.propose(
                    missing,
                    &Operation::Get {
                        key: Key::from("x")
                    },
                    None,
                    None,
                    NOW
                )
                .await
                .is_err(),
                "unknown tablet proposes loudly"
            );
            assert!(
                node.read(
                    missing,
                    &Operation::Get {
                        key: Key::from("x")
                    },
                    ReadContract::Any,
                    CTX,
                    kivi_types::LeaseEligibility::Eligible,
                )
                .await
                .is_err(),
                "unknown tablet reads loudly"
            );
            // Tablet 0 is reserved for group-independent bulk routing.
            let bad = MultiNodeConfig {
                data_dir: dir.path().join("other"),
                namespace: NS,
                tablets: vec![TabletId::from_u64(0)],
                local: NodeId::from_u64(1),
                topology: single_topology(addr, &[TabletId::from_u64(0)]),
                segment_target_bytes: 1024 * 1024,
                transport: TransportConfig::default(),
                peer_certs: HashMap::new(),
                insecure_peer_tls: true,
                preflight_enabled: true,
                lease_params: kivi_types::LeaseParams::default(),
                worker_count: 1,
                durability: SharedDurabilityConfig::default(),
                control: None,
                fragment_store_root: None,
            };
            assert!(ConsensusNode::open(bad).await.is_err(), "tablet 0 refused");
            node.shutdown().await;
        });
    }
}
