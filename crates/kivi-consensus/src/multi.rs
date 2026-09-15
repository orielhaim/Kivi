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
//! ## Static placement
//!
//! Every configured tablet lists its replicas in the static topology; in
//! this stage all nodes host one replica of every tablet with independent
//! Raft leadership per tablet. APIs never assume equal replica sets: votes
//! and gates resolve per tablet from the topology.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use kivi_state::{Operation, OperationResult};
use kivi_types::{
    ClusterId, IdempotencyKey, MutationIdentity, NamespaceId, NodeId, ReadContract,
    TabletAuthority, TabletEpoch, TabletId, UnixMicros, WriteGuardGeneration,
};

use crate::cluster::ClusterTopology;
use crate::gate::SidecarGate;
use crate::node::{
    BootstrapInputs, NodeOpenError, NodeStatus, OwnerCtx, OwnerRequest, ProposeError, ReadError,
    bootstrap_group, owner_call_on, propose_caller_side, read_caller_side, spawn_raft,
};
use crate::peer::{PeerRequest, PeerResponse, PeerRpcError};
use crate::router::PeerRouter;
use crate::shared::{
    GroupRaftStore, SharedDurabilityConfig, SharedRaftDurability, dispatch_by_group,
};
use crate::sidecar::SidecarStore;
use crate::state_machine::ReplicatedStateMachine;
use crate::tls::NodeCert;
use crate::transport::{PeerHandler, PeerTransport, TlsMaterial, TransportConfig};
use crate::types::{ConsensusError, ConsensusGroupId, ConsensusLogIndex};
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
    /// Tablets replicated by this node (nonempty, unique, nonzero; the
    /// local node must replicate every one in this static stage).
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
    /// Compio worker (reactor) count: tablet groups stripe across these
    /// deterministically ([`worker_for_tablet`]).
    pub worker_count: usize,
    /// Shared-writer batching bounds.
    pub durability: SharedDurabilityConfig,
}

/// Multi-tablet consensus node: a `Send + Sync` front over fixed Compio
/// workers driving many tablet groups. Every method below is an ordinary
/// `Send` future on any runtime.
pub struct ConsensusNode {
    workers: Vec<async_channel::Sender<OwnerRequest>>,
    group_to_worker: HashMap<ConsensusGroupId, usize>,
    machines: HashMap<TabletId, ReplicatedStateMachine>,
    authorities: HashMap<TabletId, TabletAuthority>,
    sidecar: SidecarStore,
    preflight: Arc<crate::preflight::PreflightMetrics>,
    durability: SharedRaftDurability,
    namespace: NamespaceId,
    local: NodeId,
    cluster: ClusterId,
    incarnation: kivi_types::NodeIncarnation,
    tablets: Vec<TabletId>,
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
        f.debug_struct("ConsensusNode")
            .field("node", &self.local)
            .field("cluster", &self.cluster)
            .field("tablets", &self.tablets.len())
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
#[derive(Clone)]
struct GroupRegistry {
    routes: Arc<HashMap<ConsensusGroupId, async_channel::Sender<OwnerRequest>>>,
    /// Worker serving group-independent bulk sidecars (manifest/chunk
    /// decode with a zero tablet): owns the smallest tablet, so it always
    /// hosts at least one replica serving the shared sidecar store.
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
        let bulk = self.bulk.clone();
        Box::pin(async move {
            let tx = if group == ConsensusGroupId::of_tablet(TabletId::from_u64(0)) {
                match &request {
                    PeerRequest::Manifest(_) | PeerRequest::Chunk(_) => Some(bulk),
                    _ => None,
                }
            } else {
                routes.get(&group).cloned()
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
    /// # Panics
    ///
    /// Panics when an owner-handle lock is poisoned (a lifecycle bug,
    /// never a runtime condition).
    #[allow(clippy::too_many_lines)]
    pub async fn open(config: MultiNodeConfig) -> Result<Self, NodeOpenError> {
        use NodeOpenError as Fault;
        let tablets = validated_tablets(&config)?;
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
        // Tablet-set disagreement fails loudly before any state opens:
        // state-machine tablets outside the configured set, or shared-WAL
        // groups outside it, mean a shrunk or foreign topology — never a
        // silent partial cluster. Configured tablets missing from durable
        // state bootstrap fresh per group (new tablets or interrupted
        // first open); every node in a static cluster shares the config,
        // so fresh formation stays coherent.
        let durable_tablets = existing_sm_tablets(&config.data_dir);
        let configured: BTreeSet<TabletId> = tablets.iter().copied().collect();
        let mut extra: Vec<u64> = durable_tablets
            .difference(&configured)
            .map(|tablet| tablet.as_u64())
            .collect();
        extra.sort_unstable();
        if !extra.is_empty() {
            return Err(Fault::Bootstrap {
                reason: format!(
                    "data directory holds tablets {extra:?} outside the configured set"
                ),
            });
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
        let mut wal_extra: Vec<u64> = recovered
            .iter()
            .map(kivi_durability::RaftRecord::group)
            .filter(|group| !tablets.contains(group))
            .map(TabletId::as_u64)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        wal_extra.sort_unstable();
        if !wal_extra.is_empty() {
            return Err(Fault::Bootstrap {
                reason: format!(
                    "shared WAL holds groups {wal_extra:?} outside the configured tablet set"
                ),
            });
        }
        let groups: Vec<ConsensusGroupId> = tablets
            .iter()
            .map(|tablet| ConsensusGroupId::of_tablet(*tablet))
            .collect();
        let by_group = dispatch_by_group(&recovered, &groups);
        // Per-tablet logical views: group stores plus state machines plus
        // authorities plus per-group wiring from the static topology.
        let mut stores = HashMap::new();
        let mut machines = HashMap::new();
        let mut authorities = HashMap::new();
        let mut wirings = HashMap::new();
        for tablet in &tablets {
            let group = ConsensusGroupId::of_tablet(*tablet);
            let authority =
                TabletAuthority::new(*tablet, TabletEpoch::INITIAL, WriteGuardGeneration::INITIAL);
            let store = GroupRaftStore::from_recovered(
                config.namespace,
                group,
                &durability,
                &by_group[&group],
            )
            .map_err(|error| Fault::LogStore {
                reason: error.to_string(),
            })?;
            let machine = ReplicatedStateMachine::open(
                &config.data_dir,
                config.namespace,
                *tablet,
                authority,
            )
            .map_err(|error| Fault::StateMachine {
                reason: error.to_string(),
            })?;
            let (voters, peer_addrs) =
                config
                    .topology
                    .membership_for(*tablet)
                    .map_err(|error| Fault::Topology {
                        reason: error.to_string(),
                    })?;
            stores.insert(*tablet, store);
            machines.insert(*tablet, machine);
            authorities.insert(*tablet, authority);
            wirings.insert(*tablet, GroupWiring { voters, peer_addrs });
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
        // Peer TLS identity plus the static dial maps.
        let peer_tls_cert = NodeCert::load_or_generate(&config.data_dir, opened.meta.node)
            .map_err(|error| Fault::Tls {
                reason: error.to_string(),
            })?;
        let tls = TlsMaterial {
            cert: peer_tls_cert,
            peer_certs: config.peer_certs.clone(),
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
        let mut group_to_worker = HashMap::new();
        for tablet in &tablets {
            group_to_worker.insert(
                ConsensusGroupId::of_tablet(*tablet),
                worker_for_tablet(*tablet, worker_count),
            );
        }
        let smallest = tablets.iter().min().expect("tablets nonempty");
        let bulk_worker = worker_for_tablet(*smallest, worker_count);
        let registry = GroupRegistry {
            routes: Arc::new(
                group_to_worker
                    .iter()
                    .map(|(group, worker)| (*group, senders[*worker].clone()))
                    .collect(),
            ),
            bulk: senders[bulk_worker].clone(),
        };
        let preflight = Arc::new(crate::preflight::PreflightMetrics::default());
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
            let owned: Vec<TabletId> = tablets
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
                namespace: config.namespace,
                local: opened.meta.node,
                cluster: opened.meta.cluster,
                preflight: Arc::clone(&preflight),
                preflight_enabled: config.preflight_enabled,
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
        // snapshot install).
        for tablet in &tablets {
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
            group_to_worker,
            machines,
            authorities,
            sidecar,
            preflight,
            durability,
            namespace: config.namespace,
            local: opened.meta.node,
            cluster: opened.meta.cluster,
            incarnation: opened.meta.incarnation,
            tablets,
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

    /// Returns the tablets this node replicates, in sorted order.
    #[must_use]
    pub fn tablets(&self) -> &[TabletId] {
        &self.tablets
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
        now: UnixMicros,
    ) -> Result<crate::node::ProposeOutcome, ProposeError> {
        let Some(machine) = self.machines.get(&tablet) else {
            return Err(ProposeError::Consensus(ConsensusError::Unavailable {
                reason: format!("tablet {} not served by this node", tablet.as_u64()),
            }));
        };
        let Some(authority) = self.authorities.get(&tablet) else {
            return Err(ProposeError::Consensus(ConsensusError::Unavailable {
                reason: format!("tablet {} has no local authority", tablet.as_u64()),
            }));
        };
        let worker = self.worker_for(tablet)?;
        propose_caller_side(
            machine,
            &self.sidecar,
            self.namespace,
            *authority,
            &worker,
            ConsensusGroupId::of_tablet(tablet),
            op,
            identity,
            idempotency,
            now,
        )
        .await
    }

    /// Serves one read against `tablet`'s group under its contract.
    ///
    /// # Errors
    ///
    /// Returns [`ReadError`] for unknown tablets, routing, validation, or
    /// coverage failures.
    pub async fn read(
        &self,
        tablet: TabletId,
        op: &Operation,
        contract: ReadContract,
        now: UnixMicros,
    ) -> Result<OperationResult, ReadError> {
        let Some(machine) = self.machines.get(&tablet) else {
            return Err(ReadError::Consensus(ConsensusError::Unavailable {
                reason: format!("tablet {} not served by this node", tablet.as_u64()),
            }));
        };
        let worker = self.worker_for(tablet).map_err(|error| match error {
            ProposeError::Consensus(consensus) => ReadError::Consensus(consensus),
            _ => ReadError::Consensus(ConsensusError::Unavailable {
                reason: format!("tablet {} not served by this node", tablet.as_u64()),
            }),
        })?;
        read_caller_side(
            machine,
            &worker,
            ConsensusGroupId::of_tablet(tablet),
            op,
            contract,
            now,
        )
        .await
    }

    /// Gathers per-group diagnostics for every local group (fans out to
    /// owning workers concurrently; a dead worker reports closed statuses
    /// rather than hanging the admin plane).
    pub async fn status_all(&self) -> Vec<NodeStatus> {
        let calls: Vec<_> = self
            .tablets
            .iter()
            .map(|tablet| {
                let group = ConsensusGroupId::of_tablet(*tablet);
                let worker = self
                    .group_to_worker
                    .get(&group)
                    .map(|index| self.workers[*index].clone());
                let local = self.local;
                let sidecar = self.sidecar.metrics().snapshot();
                let preflight = self.preflight.snapshot();
                async move {
                    let Some(worker) = worker else {
                        return closed_status(group, local, sidecar, preflight);
                    };
                    owner_call_on(&worker, |reply| OwnerRequest::Status { group, reply })
                        .await
                        .unwrap_or_else(|_| closed_status(group, local, sidecar, preflight))
                }
            })
            .collect();
        futures::future::join_all(calls).await
    }

    /// Gathers diagnostics for one group.
    pub async fn status_for(&self, group: ConsensusGroupId) -> NodeStatus {
        let sidecar = self.sidecar.metrics().snapshot();
        let preflight = self.preflight.snapshot();
        let Some(worker) = self
            .group_to_worker
            .get(&group)
            .map(|index| self.workers[*index].clone())
        else {
            return closed_status(group, self.local, sidecar, preflight);
        };
        owner_call_on(&worker, |reply| OwnerRequest::Status { group, reply })
            .await
            .unwrap_or_else(|_| closed_status(group, self.local, sidecar, preflight))
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
        let Some(worker) = self
            .group_to_worker
            .get(&group)
            .map(|index| self.workers[*index].clone())
        else {
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
    }
}

/// Validates the configured tablet set: nonempty, unique, nonzero, and
/// every tablet assigned to the local node in the static topology.
/// Returns the sorted tablets.
fn validated_tablets(config: &MultiNodeConfig) -> Result<Vec<TabletId>, NodeOpenError> {
    use NodeOpenError as Fault;
    if config.worker_count == 0 {
        return Err(Fault::Topology {
            reason: "worker count must be nonzero".to_owned(),
        });
    }
    if config.tablets.is_empty() {
        return Err(Fault::Topology {
            reason: "cluster topology has no tablet assignment".to_owned(),
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
        transport.shutdown().await;
        return;
    }
    let _ = shutdown.await;
    transport.shutdown().await;
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
    namespace: NamespaceId,
    local: NodeId,
    cluster: ClusterId,
    preflight: Arc<crate::preflight::PreflightMetrics>,
    preflight_enabled: bool,
    requests: async_channel::Receiver<OwnerRequest>,
    ready: Option<futures::channel::oneshot::Sender<Result<(), NodeOpenError>>>,
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
    worker_loop(replicas, params.local, params.transport, params.requests).await;
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
        &mut store,
        &machine,
        &raft,
    )
    .await?;
    let mut voter_list: Vec<NodeId> = wiring
        .voters
        .iter()
        .map(|voter| NodeId::from_u64(*voter))
        .collect();
    voter_list.sort_by_key(|node| node.as_u64());
    Ok((
        group,
        OwnerCtx {
            raft,
            store,
            machine,
            sidecar: params.sidecar.clone(),
            gate,
            transport: params.transport.clone(),
            group,
            namespace: params.namespace,
            authority,
            tablet,
            local: params.local,
            voters: voter_list,
            bulk_timeout: params.transport_config.bulk_timeout,
            preflight: Arc::clone(&params.preflight),
            preflight_enabled: params.preflight_enabled,
            transfers: Rc::new(RefCell::new(HashMap::new())),
        },
    ))
}

/// Serves one worker's ingress until `Shutdown`: group-scoped requests
/// dispatch to the owning replica's [`OwnerCtx`] (each on its own task —
/// a pending large-write preflight never blocks small writes or peer
/// RPCs); node-wide suspend/resume run against the shared transport;
/// `Shutdown` breaks the loop and stops every local `Raft`.
async fn worker_loop(
    replicas: HashMap<ConsensusGroupId, OwnerCtx<GroupRaftStore>>,
    local: NodeId,
    transport: PeerTransport,
    requests: async_channel::Receiver<OwnerRequest>,
) {
    while let Ok(request) = requests.recv().await {
        match request {
            OwnerRequest::Shutdown => break,
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
                // Group-independent bulk sidecars decode with a zero
                // tablet: any local replica serves them from the shared
                // sidecar store (the registry only sends these to a worker
                // that owns at least one group).
                let dest = match &other {
                    OwnerRequest::Serve { group, request, .. }
                        if *group == ConsensusGroupId::of_tablet(TabletId::from_u64(0))
                            && matches!(
                                request,
                                PeerRequest::Manifest(_) | PeerRequest::Chunk(_)
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
    const NOW: UnixMicros = UnixMicros::from_micros(1_000_000);

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
                worker_count: 2,
                durability: SharedDurabilityConfig::default(),
            })
            .await
            .expect("multi opens");
            assert_eq!(node.tablets(), tablets.as_slice());
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
                        NOW,
                    )
                    .await
                    .expect("tablet serves its key");
                assert_eq!(
                    read,
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
                worker_count: 1,
                durability: SharedDurabilityConfig::default(),
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
                    NOW
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
                worker_count: 1,
                durability: SharedDurabilityConfig::default(),
            };
            assert!(ConsensusNode::open(bad).await.is_err(), "tablet 0 refused");
            node.shutdown().await;
        });
    }
}
